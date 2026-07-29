"""Claim extraction as a callable stage, shared by scripts/emit_claims.py (the
CLI) and POST /extract (main.py) so the two entry points can never drift --
the same reasoning as dispatch.parse_bytes for POST /parse and worker.py.

Stateless transform: document bytes in, the C3-contract claims payload out.
No DB access and no persistence here. `run_id` and `document_id` identify the
caller's run for logging/correlation only -- the claims contract has no slot
for either (the backend assigns `id`/`document_id` at persistence time, per
emit.Claim's docstring), so neither is written into the returned payload.
"""

from __future__ import annotations

import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from .docling_parser import parse_pdf_bytes
from .emit import Claim, FlagLog
from .extract import claims_from_table
from .propose import api_key_present, assertions_from_prose, claims_from_prose, prose_text
from .schemas import PageIndex
from .table_extract import extract_tables, tables_on_page
from .text_extract import blocks_on_page, extract_text_blocks

logger = logging.getLogger(__name__)


class ProseCredentialMissing(RuntimeError):
    """The prose/qualitative tiers were requested without an Anthropic
    credential in the environment. Raised before any parsing happens, so a
    caller never pays for a parse it can't finish -- the CLI maps this to an
    argparse SystemExit, the HTTP endpoint maps it to a 503."""


def _prose_claims(
    kind: str,
    pages: list[PageIndex],
    blocks,
    *,
    entity: str,
    file: str,
    flag_log: FlagLog,
    workers: int,
) -> list[Claim]:
    """Run one prose tier over every page that has prose, concurrently.

    `kind` selects the extractor: "prose" for the numeric pass, "qualitative"
    for the assertion pass. A page whose model call fails is reported on
    stderr and skipped -- a partial result that names its gaps is more useful
    than none, and the caller can re-run the named pages.
    """
    extractor = claims_from_prose if kind == "prose" else assertions_from_prose

    def run(page: PageIndex) -> tuple[int, list[Claim]]:
        return page.page, extractor(
            blocks_on_page(blocks, page.page),
            page,
            entity_hint=entity,
            file=file,
            flag_log=flag_log,
        )

    with_prose = [p for p in pages if prose_text(blocks_on_page(blocks, p.page), p).strip()]
    claims: list[Claim] = []
    failed: list[tuple[int, str]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run, p): p.page for p in with_prose}
        for future in as_completed(futures):
            try:
                _, page_claims = future.result()
                claims += page_claims
            except Exception as exc:  # noqa: BLE001 -- one bad page must not lose the run
                failed.append((futures[future], f"{type(exc).__name__}: {exc}"))
    resolved = sum(1 for c in claims if c.status != "missing")
    print(
        f"tier {kind}: {len(claims)} claims over {len(with_prose)} prose pages "
        f"({resolved} resolved)",
        file=sys.stderr,
    )
    if failed:
        print(
            f"  {len(failed)} page(s) failed and were skipped: {[p for p, _ in failed]}",
            file=sys.stderr,
        )
    return claims


def extract_claims(
    data: bytes,
    *,
    entity: str,
    run_id: str,
    document_id: str,
    source_file: str,
    prose: bool = False,
    qualitative: bool = False,
    workers: int = 8,
) -> dict:
    """Parse `data` and emit its claims as the payload that crosses the C3 seam.

    The one code path both entry points call: scripts/emit_claims.py (CLI) and
    POST /extract (main.py). Raises ProseCredentialMissing at the door, before
    any parsing, if `prose`/`qualitative` are set without a credential in the
    environment -- fail closed, not partway through a document.
    `qualitative` implies the prose tier as well as its own.
    """
    want_prose = prose or qualitative
    if want_prose and not api_key_present():
        raise ProseCredentialMissing(
            "the prose tiers call the Anthropic API and need ANTHROPIC_API_KEY "
            "(or ANTHROPIC_AUTH_TOKEN) in the environment; set it, or drop prose/"
            "qualitative to emit table claims only."
        )

    logger.info(
        "extract_claims start: run_id=%s document_id=%s entity=%s prose=%s qualitative=%s",
        run_id,
        document_id,
        entity,
        prose,
        qualitative,
    )

    result = parse_pdf_bytes(data)
    assert result.document is not None, "a successful parse always carries the DoclingDocument"
    tables = extract_tables(result.document, result.pages)
    flag_log = FlagLog(run_id=run_id)

    claims: list[Claim] = []
    for page in result.pages:
        for table in tables_on_page(tables, page.page):
            claims += claims_from_table(
                table,
                page,
                entity=entity,
                file=source_file,
                flag_log=flag_log,
            )
    print(f"tier tables: {len(claims)} claims", file=sys.stderr)

    if want_prose:
        blocks = extract_text_blocks(result.document, result.pages)
        claims += _prose_claims(
            "prose",
            result.pages,
            blocks,
            entity=entity,
            file=source_file,
            flag_log=flag_log,
            workers=workers,
        )
        if qualitative:
            claims += _prose_claims(
                "qualitative",
                result.pages,
                blocks,
                entity=entity,
                file=source_file,
                flag_log=flag_log,
                workers=workers,
            )

    payload = {
        "run_id": flag_log.run_id,
        "sha256": result.sha256,
        "source_file": source_file,
        "claims": [c.to_json() for c in claims],
        "flag_log": flag_log.to_json(),
    }
    print(
        f"\n\nemitted {len(claims)} claims "
        f"({sum(1 for c in claims if c.status != 'missing')} cited, "
        f"{sum(1 for c in claims if c.status == 'missing')} missing), "
        f"{len(flag_log.entries)} flags",
        file=sys.stderr,
    )
    return payload

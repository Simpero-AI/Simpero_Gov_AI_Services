"""Claim extraction as a callable stage, shared by scripts/emit_claims.py (the
CLI) and POST /extract (main.py) so the two entry points can never drift --
the same reasoning as dispatch.parse_bytes for POST /parse and worker.py.

Stateless transform: document bytes in, the C3-contract claims payload out.
No DB access and no persistence here. `run_id` and `correlation_id` identify
the caller's run for logging/correlation only -- the claims contract has no
slot for either (the backend assigns `id`/`document_id` at persistence time,
per emit.Claim's docstring), so neither is written into the returned payload.
`correlation_id` is deliberately not named `document_id`: that term already
means the content hash elsewhere in this codebase (emit_chunks sets
`document_id = sha256`), and a caller-supplied token is not that.
"""

from __future__ import annotations

import logging
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from .coverage import NumberMiss, document_coverage
from .docling_parser import parse_pdf_bytes
from .emit import Claim, FlagLog, PdfLocation, SkippedPage
from .extract import claims_from_table
from .propose import (
    api_key_present,
    assertions_from_prose,
    claims_from_completeness,
    claims_from_prose,
    prose_text,
)
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
) -> tuple[list[Claim], list[SkippedPage]]:
    """Run one prose tier over every page that has prose, concurrently.

    `kind` selects the extractor: "prose" for the numeric pass, "qualitative"
    for the assertion pass. A page whose model call fails is reported on
    stderr AND returned as a SkippedPage(page, tier, reason) -- a partial
    result that names its gaps is more useful than none, and both entry points
    need that name: the CLI caller can re-run it, and an HTTP caller has no
    stderr to read at all (see extract_claims' `skipped_pages`).
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
    return claims, [
        SkippedPage(page=page_no, tier=kind, reason=reason) for page_no, reason in failed
    ]


def _pdf_span(claim: Claim) -> tuple[int, int, int] | None:
    """This claim's (page, char_start, char_end), or None for a `missing` claim
    or a non-PDF location. Isolated in its own function so the char_start/
    char_end None-checks narrow cleanly for the type checker, which a single
    list-comprehension condition does not do across two different attributes.
    """
    location = claim.location
    if (
        isinstance(location, PdfLocation)
        and location.char_start is not None
        and location.char_end is not None
    ):
        return (location.page, location.char_start, location.char_end)
    return None


def _completeness_claims(
    pages: list[PageIndex],
    prior_claims: list[Claim],
    *,
    entity: str,
    file: str,
    flag_log: FlagLog,
    workers: int,
) -> tuple[list[Claim], list[SkippedPage]]:
    """Recover Gate-1 coverage misses: numbers a page printed that no claim
    gathered so far covers, but that resolve to a unique span (parser_service/
    coverage.py). Coverage is measured over every claim gathered so far, not
    just the prose tier, since a number any tier already cited is not a miss.

    Only a page with at least one addressable miss costs a model call -- a
    fully-covered page is skipped outright (see claims_from_completeness).
    Numeric only, by design; see coverage.py's module docstring.

    A page whose recovery call fails is reported on stderr AND returned as a
    SkippedPage(page, "complete", reason), the same failure model the prose
    tiers use, so the payload's `skipped_pages` names it for an HTTP caller.
    """
    claim_spans = [span for c in prior_claims if (span := _pdf_span(c)) is not None]
    coverage = document_coverage(pages, claim_spans)
    misses_by_page: dict[int, list[NumberMiss]] = defaultdict(list)
    for miss in coverage.gate1_misses:
        misses_by_page[miss.page].append(miss)
    pages_by_no = {p.page: p for p in pages}

    def run(page_no: int) -> tuple[int, list[Claim]]:
        page = pages_by_no[page_no]
        missed = [(m.text, m.context) for m in misses_by_page[page_no]]
        return page_no, claims_from_completeness(
            page, missed, entity_hint=entity, file=file, flag_log=flag_log
        )

    claims: list[Claim] = []
    failed: list[tuple[int, str]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run, page_no): page_no for page_no in misses_by_page}
        for future in as_completed(futures):
            try:
                _, page_claims = future.result()
                claims += page_claims
            except Exception as exc:  # noqa: BLE001 -- one bad page must not lose the run
                failed.append((futures[future], f"{type(exc).__name__}: {exc}"))
    recovered = sum(1 for c in claims if c.status != "missing")
    print(
        f"tier complete: coverage {coverage.coverage:.1%} "
        f"({coverage.covered}/{coverage.total}), {coverage.gate1} gate-1 miss(es) "
        f"over {len(misses_by_page)} page(s), {recovered} recovered",
        file=sys.stderr,
    )
    if failed:
        print(
            f"  {len(failed)} page(s) failed and were skipped: {[p for p, _ in failed]}",
            file=sys.stderr,
        )
    return claims, [
        SkippedPage(page=page_no, tier="complete", reason=reason) for page_no, reason in failed
    ]


def extract_claims(
    data: bytes,
    *,
    entity: str,
    run_id: str,
    correlation_id: str,
    source_file: str | None = None,
    prose: bool = False,
    complete: bool = False,
    qualitative: bool = False,
    workers: int = 8,
) -> dict:
    """Parse `data` and emit its claims as the payload that crosses the C3 seam.

    The one code path both entry points call: scripts/emit_claims.py (CLI) and
    POST /extract (main.py). Raises ProseCredentialMissing at the door, before
    any parsing, if `prose`/`complete`/`qualitative` are set without a
    credential in the environment -- fail closed, not partway through a
    document. `complete` and `qualitative` each imply the prose tier.

    `source_file` is the real filename when the caller has one (the CLI
    always does); an HTTP caller that genuinely has none should leave it
    unset rather than pass a stand-in like `correlation_id` -- the C3 contract
    documents each claim's `file` as "debug/human-readable only", and a
    correlation token dressed as a filename is a false debug signal. Every
    claim still needs a string for `file`, so an unset source_file becomes ""
    there (never a guess); the top-level `source_file` in the returned
    payload stays `None` so a caller can tell "no filename given" apart from
    "the filename was an empty string".
    """
    want_prose = prose or complete or qualitative
    if want_prose and not api_key_present():
        raise ProseCredentialMissing(
            "the prose tiers call the Anthropic API and need ANTHROPIC_API_KEY "
            "(or ANTHROPIC_AUTH_TOKEN) in the environment; set it, or drop prose/"
            "complete/qualitative to emit table claims only."
        )

    logger.info(
        "extract_claims start: run_id=%s correlation_id=%s entity=%s prose=%s "
        "complete=%s qualitative=%s",
        run_id,
        correlation_id,
        entity,
        prose,
        complete,
        qualitative,
    )

    result = parse_pdf_bytes(data)
    assert result.document is not None, "a successful parse always carries the DoclingDocument"
    tables = extract_tables(result.document, result.pages)
    flag_log = FlagLog(run_id=run_id)
    file = source_file or ""

    claims: list[Claim] = []
    skipped_pages: list[SkippedPage] = []
    for page in result.pages:
        for table in tables_on_page(tables, page.page):
            try:
                claims += claims_from_table(
                    table,
                    page,
                    entity=entity,
                    file=file,
                    flag_log=flag_log,
                )
            except Exception as exc:  # noqa: BLE001 -- one bad table must not abort the document
                # Guarded per table, not per page: a page with several tables
                # keeps every sibling table's claims when only one is
                # malformed. The page is still named in skipped_pages -- it
                # lost something real, even if not everything.
                skipped_pages.append(
                    SkippedPage(
                        page=page.page,
                        tier="tables",
                        reason=f"{type(exc).__name__}: {exc}",
                    )
                )
                print(
                    f"tier tables: page {page.page} table failed and was skipped: "
                    f"{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
    print(f"tier tables: {len(claims)} claims", file=sys.stderr)

    if want_prose:
        blocks = extract_text_blocks(result.document, result.pages)
        prose_claims, prose_failed = _prose_claims(
            "prose",
            result.pages,
            blocks,
            entity=entity,
            file=file,
            flag_log=flag_log,
            workers=workers,
        )
        claims += prose_claims
        skipped_pages.extend(prose_failed)
        if complete:
            complete_claims, complete_failed = _completeness_claims(
                result.pages,
                claims,
                entity=entity,
                file=file,
                flag_log=flag_log,
                workers=workers,
            )
            claims += complete_claims
            skipped_pages.extend(complete_failed)
        if qualitative:
            qualitative_claims, qualitative_failed = _prose_claims(
                "qualitative",
                result.pages,
                blocks,
                entity=entity,
                file=file,
                flag_log=flag_log,
                workers=workers,
            )
            claims += qualitative_claims
            skipped_pages.extend(qualitative_failed)

    payload = {
        "run_id": flag_log.run_id,
        "sha256": result.sha256,
        "source_file": source_file,
        "claims": [c.to_json() for c in claims],
        "flag_log": flag_log.to_json(),
        "skipped_pages": [
            s.to_json() for s in sorted(skipped_pages, key=lambda s: (s.page, s.tier))
        ],
    }
    print(
        f"\n\nemitted {len(claims)} claims "
        f"({sum(1 for c in claims if c.status != 'missing')} cited, "
        f"{sum(1 for c in claims if c.status == 'missing')} missing), "
        f"{len(flag_log.entries)} flags, {len(skipped_pages)} skip(s)",
        file=sys.stderr,
    )
    return payload

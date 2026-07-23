"""Demo: parse a PDF and emit its claims as JSON, the shape that crosses the C3 seam.

Runs entirely in the parse service and touches no database -- it is the left
half of the seam. The right half (scripts/ingest_claims.py in the backend repo)
reads this JSON and persists it. The two never share a runtime, exactly as the
production split intends.

    # Tables only -- deterministic, no network, no key:
    uv run python scripts/emit_claims.py <cim.pdf> --entity "PTL Group" > claims.json

    # Add the prose tiers -- one model call per prose page, needs a key:
    ANTHROPIC_API_KEY=... uv run python scripts/emit_claims.py <cim.pdf> \\
        --entity "PTL Group" --prose > claims.json

Three tiers, each a strict superset of the last:
  tables       (default)  claims_from_table       -- deterministic, no LLM
  --prose                 + claims_from_prose      -- numeric facts in prose (1 call/page)
  --qualitative           + assertions_from_prose  -- claims that carry no number (1 call/page)

--qualitative implies --prose. Both prose tiers call the Anthropic API once per
prose page and require ANTHROPIC_API_KEY (or ANTHROPIC_AUTH_TOKEN); the script
fails closed with that message rather than part way through a document. The
per-page calls are independent -- a quote only ever resolves against its own
page -- so they run concurrently, and a page whose call fails is recorded and
skipped rather than aborting the run.

Not the real extractor: entity is passed in, and attributes are the document's
own words (see parser_service/extract.py and parser_service/propose.py). This
proves the pipeline end to end on real provenance, not that the product reads a
CIM.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from parser_service.docling_parser import parse_pdf_bytes
from parser_service.emit import Claim, FlagLog
from parser_service.extract import claims_from_table
from parser_service.propose import (
    api_key_present,
    assertions_from_prose,
    claims_from_prose,
    prose_text,
)
from parser_service.schemas import PageIndex
from parser_service.table_extract import extract_tables, tables_on_page
from parser_service.text_extract import blocks_on_page, extract_text_blocks


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
    for the assertion pass. A page whose model call fails is reported on stderr
    and skipped -- a partial result that names its gaps is more useful than
    none, and the caller can re-run the named pages.
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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf_path", type=Path)
    parser.add_argument("--entity", required=True, help="The company the claims are about.")
    parser.add_argument("--run-id", default="demo-e2e")
    parser.add_argument(
        "--prose",
        action="store_true",
        help="Also read numeric facts stated in prose (one model call per prose page).",
    )
    parser.add_argument(
        "--qualitative",
        action="store_true",
        help="Also read claims that carry no number (implies --prose).",
    )
    parser.add_argument(
        "--workers", type=int, default=8, help="Concurrent prose-page calls (default 8)."
    )
    args = parser.parse_args(argv)

    want_prose = args.prose or args.qualitative
    if want_prose and not api_key_present():
        parser.error(
            "the prose tiers call the Anthropic API and need ANTHROPIC_API_KEY "
            "(or ANTHROPIC_AUTH_TOKEN) in the environment; set it, or drop --prose/"
            "--qualitative to emit table claims only."
        )

    result = parse_pdf_bytes(args.pdf_path.read_bytes())
    assert result.document is not None, "a successful parse always carries the DoclingDocument"
    tables = extract_tables(result.document, result.pages)
    flag_log = FlagLog(run_id=args.run_id)

    claims: list[Claim] = []
    for page in result.pages:
        for table in tables_on_page(tables, page.page):
            claims += claims_from_table(
                table,
                page,
                entity=args.entity,
                file=args.pdf_path.name,
                flag_log=flag_log,
            )
    print(f"tier tables: {len(claims)} claims", file=sys.stderr)

    if want_prose:
        blocks = extract_text_blocks(result.document, result.pages)
        claims += _prose_claims(
            "prose",
            result.pages,
            blocks,
            entity=args.entity,
            file=args.pdf_path.name,
            flag_log=flag_log,
            workers=args.workers,
        )
        if args.qualitative:
            claims += _prose_claims(
                "qualitative",
                result.pages,
                blocks,
                entity=args.entity,
                file=args.pdf_path.name,
                flag_log=flag_log,
                workers=args.workers,
            )

    # sha256 is the document identity the backend will map to a data_source; the
    # parser reports it, the backend assigns the id.
    payload = {
        "run_id": flag_log.run_id,
        "sha256": result.sha256,
        "source_file": args.pdf_path.name,
        "claims": [c.to_json() for c in claims],
        "flag_log": flag_log.to_json(),
    }
    json.dump(payload, sys.stdout, indent=2)
    print(
        f"\n\nemitted {len(claims)} claims "
        f"({sum(1 for c in claims if c.status != 'missing')} cited, "
        f"{sum(1 for c in claims if c.status == 'missing')} missing), "
        f"{len(flag_log.entries)} flags",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()

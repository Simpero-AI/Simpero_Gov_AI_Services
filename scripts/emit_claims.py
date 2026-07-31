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

Four tiers, each additive:
  tables       (default)  claims_from_table        -- deterministic, no LLM
  --prose                 + claims_from_prose       -- numeric facts in prose (1 call/page)
  --complete               + claims_from_completeness -- recovers a page's Gate-1
                              coverage misses (numbers printed but not claimed by
                              the tiers above); see parser_service/coverage.py.
                              Numeric only -- see coverage.py's module docstring
                              for why the qualitative tier does not get this pass.
  --qualitative            + assertions_from_prose  -- claims that carry no number (1 call/page)
  --canonicalize-attributes + SIM-344 attribute mapping -- maps every quantitative
                              claim's document-label attribute onto a closed
                              canonical vocabulary (financial-statement core, or
                              the operating_metric bucket), one batched call for
                              the whole document. Independent of the other three:
                              works on --table-only output just as well as on
                              --prose output.

--qualitative and --complete both imply --prose. --canonicalize-attributes does
not -- it runs standalone on top of --table-only output too. Every flag except
the default table tier calls the Anthropic API and requires ANTHROPIC_API_KEY
(or ANTHROPIC_AUTH_TOKEN); the script fails closed with that message rather than
part way through a document. The per-page prose calls are independent -- a quote
only ever resolves against its own page -- so they run concurrently, and a page
whose call fails is recorded and skipped rather than aborting the run.
--canonicalize-attributes is one batched call for the whole document instead, not
per-page.

This is a thin CLI over parser_service.extract_service.extract_claims -- the
same entry point POST /extract (main.py) calls, so the two can never drift.

Not the real extractor: entity is passed in, and attributes are the document's
own words (see parser_service/extract.py and parser_service/propose.py). This
proves the pipeline end to end on real provenance, not that the product reads a
CIM.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from parser_service.extract_service import ProseCredentialMissing, extract_claims


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
        "--complete",
        action="store_true",
        help=(
            "Also recover numeric Gate-1 coverage misses -- figures the page printed "
            "that no tier above claimed (implies --prose). Numeric only; see "
            "parser_service/coverage.py."
        ),
    )
    parser.add_argument(
        "--qualitative",
        action="store_true",
        help="Also read claims that carry no number (implies --prose).",
    )
    parser.add_argument(
        "--canonicalize-attributes",
        action="store_true",
        help=(
            "Also map every quantitative claim's attribute onto the closed SIM-344 "
            "canonical vocabulary (one batched model call for the whole document). "
            "Independent of --prose/--complete/--qualitative."
        ),
    )
    parser.add_argument(
        "--workers", type=int, default=8, help="Concurrent prose-page calls (default 8)."
    )
    args = parser.parse_args(argv)

    try:
        payload = extract_claims(
            args.pdf_path.read_bytes(),
            entity=args.entity,
            run_id=args.run_id,
            correlation_id=args.pdf_path.stem,
            source_file=args.pdf_path.name,
            prose=args.prose,
            complete=args.complete,
            qualitative=args.qualitative,
            canonicalize_attributes=args.canonicalize_attributes,
            workers=args.workers,
        )
    except ProseCredentialMissing as exc:
        parser.error(str(exc))
        return  # unreachable -- parser.error() always raises SystemExit

    json.dump(payload, sys.stdout, indent=2)


if __name__ == "__main__":
    main()

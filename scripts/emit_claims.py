"""Demo: parse a PDF and emit its claims as JSON, the shape that crosses the C3 seam.

Runs entirely in the parse service and touches no database -- it is the left
half of the seam. The right half (scripts/ingest_claims.py in the backend repo)
reads this JSON and persists it. The two never share a runtime, exactly as the
production split intends.

    uv run python scripts/emit_claims.py <cim.pdf> --entity "PTL Group" > claims.json

Not the real extractor: entity is passed in, and attributes are the table's own
words (see parser_service/extract.py). This proves the pipeline end to end on
real provenance, not that the product reads a CIM.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from parser_service.docling_parser import parse_pdf_bytes
from parser_service.emit import FlagLog
from parser_service.extract import claims_from_table
from parser_service.table_extract import extract_tables, tables_on_page


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf_path", type=Path)
    parser.add_argument("--entity", required=True, help="The company the claims are about.")
    parser.add_argument("--run-id", default="demo-e2e")
    args = parser.parse_args(argv)

    result = parse_pdf_bytes(args.pdf_path.read_bytes())
    assert result.document is not None, "a successful parse always carries the DoclingDocument"
    tables = extract_tables(result.document)
    flag_log = FlagLog(run_id=args.run_id)

    claims = []
    for page in result.pages:
        for table in tables_on_page(tables, page.page):
            claims += claims_from_table(
                table,
                page,
                entity=args.entity,
                file=args.pdf_path.name,
                flag_log=flag_log,
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

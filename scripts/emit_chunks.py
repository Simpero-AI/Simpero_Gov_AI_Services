"""Demo: parse a PDF and emit its retrieval chunks as JSON -- the seam the
backend's chunk ingest reads (AE-A-RETR-13 / SIM-338), the chunk-side sibling of
emit_claims. Runs entirely in the parse service and touches no database.

    uv run python scripts/emit_chunks.py <cim.pdf> > chunks.json

Deterministic and offline: chunking (SIM-238/239) is rule-based, and embeddings
are the backend's job at ingest time (it calls voyage-4-large, SIM-245), so this
emits chunk text + metadata only -- no model, no key. Each chunk carries its own
`document_id` (the source sha256; the backend maps it to a data_source id) and
`source_file`, so the seam is self-describing per chunk; the top-level `sha256`
is the document identity for the ingest to key on.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from parser_service.chunk_pipeline import chunks_for_document


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf_path", type=Path)
    args = parser.parse_args(argv)

    # Same parse -> chunk flow the deal-flow worker runs (chunk_pipeline), so the
    # demo and production can never drift on how a document is chunked.
    sha256, chunks = chunks_for_document(args.pdf_path.read_bytes(), source_file=args.pdf_path.name)

    payload = {
        "sha256": sha256,
        "source_file": args.pdf_path.name,
        "chunks": [chunk.model_dump(mode="json") for chunk in chunks],
    }
    json.dump(payload, sys.stdout, indent=2)

    by_type = Counter(chunk.element_type for chunk in chunks)
    print(
        f"\n\nemitted {len(chunks)} chunks ({dict(by_type)})",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()

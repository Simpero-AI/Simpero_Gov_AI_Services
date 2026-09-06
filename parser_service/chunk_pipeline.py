"""Parse a document and cut it into retrieval chunks -- the one place the
parse -> element-extract -> chunk_document flow lives, so the deal-flow worker
(worker.process_document) and the demo CLI (scripts/emit_chunks.py) share it and
can never drift on how a document is chunked.

The docling parse here is a read-through cache hit whenever the same bytes were
already parsed (docling_parser.py), so calling this right after extract_claims on
the same document adds only rule-based element extraction + chunking -- no second
docling pass and no model call (chunking is deterministic + offline; embeddings
are the backend's job at ingest).
"""

from __future__ import annotations

from parser_service.chunker import ChunkRecord, chunk_document
from parser_service.docling_parser import parse_pdf_bytes
from parser_service.elements import extract_chart_elements, extract_table_elements
from parser_service.table_extract import extract_tables
from parser_service.text_extract import extract_text_blocks


def chunks_for_document(data: bytes, *, source_file: str) -> tuple[str, list[ChunkRecord]]:
    """Return (document sha256, retrieval chunks) for `data`.

    `source_file` is stamped on every chunk (its provenance label). Each chunk's
    `document_id` is the parse's sha256 -- the same identity extract_claims stamps
    on its claims envelope for the same bytes, so the backend can align chunks to
    that document. A parse that yields no DoclingDocument returns (sha256, [])."""
    result = parse_pdf_bytes(data)
    if result.document is None:
        return result.sha256, []

    blocks = extract_text_blocks(result.document, result.pages)
    table_elements = extract_table_elements(extract_tables(result.document, result.pages))
    chart_elements = extract_chart_elements(result.document, result.pages)

    chunks = chunk_document(
        result.pages,
        blocks,
        table_elements,
        chart_elements,
        document_id=result.sha256,
        source_file=source_file,
    )
    return result.sha256, chunks

"""emit_chunks: the parse -> chunk -> JSON seam the backend ingest reads.

Checks the wiring and the emitted payload shape without a real parse: the heavy
parse + element extraction are stubbed and chunk_document returns canned
ChunkRecords, so this asserts every chunk round-trips into the seam with its
metadata intact (including scale_context, which the parser emits as-is -- the
backend decides how to fold it into the stored content).
"""

from __future__ import annotations

import json

from parser_service.chunker import ChunkRecord
from scripts import emit_chunks


class _Result:
    sha256 = "a" * 64
    pages: list = []
    document = object()


def _stub(monkeypatch, chunks: list[ChunkRecord]) -> None:
    monkeypatch.setattr(emit_chunks, "parse_pdf_bytes", lambda _b: _Result())
    monkeypatch.setattr(emit_chunks, "extract_text_blocks", lambda *_a: [])
    monkeypatch.setattr(emit_chunks, "extract_tables", lambda *_a: [])
    monkeypatch.setattr(emit_chunks, "extract_table_elements", lambda *_a: [])
    monkeypatch.setattr(emit_chunks, "extract_chart_elements", lambda *_a: [])
    monkeypatch.setattr(emit_chunks, "chunk_document", lambda *_a, **_k: chunks)


def test_emits_one_json_chunk_per_record_with_metadata(monkeypatch, tmp_path, capsys) -> None:
    chunks = [
        ChunkRecord(
            content="Revenue grew to $15M",
            element_type="prose",
            page=1,
            order=0,
            document_id="a" * 64,
            source_file="cim.pdf",
        ),
        ChunkRecord(
            content="| Revenue | 15 |",
            element_type="table",
            page=2,
            order=1,
            document_id="a" * 64,
            source_file="cim.pdf",
            scale_context="$ in millions",
        ),
    ]
    _stub(monkeypatch, chunks)
    pdf = tmp_path / "cim.pdf"
    pdf.write_bytes(b"%PDF-1.4 stub")

    emit_chunks.main([str(pdf)])

    payload = json.loads(capsys.readouterr().out)
    assert payload["sha256"] == "a" * 64
    assert payload["source_file"] == "cim.pdf"
    assert len(payload["chunks"]) == 2
    first = payload["chunks"][0]
    assert first["content"] == "Revenue grew to $15M"
    assert first["element_type"] == "prose"
    assert first["document_id"] == "a" * 64
    # scale_context survives the seam untouched; folding it into content is the
    # backend ingest's call (SIM-338 sub-task 3), not the parser's.
    assert payload["chunks"][1]["scale_context"] == "$ in millions"


def test_empty_document_emits_an_empty_chunk_list(monkeypatch, tmp_path, capsys) -> None:
    _stub(monkeypatch, [])
    pdf = tmp_path / "cim.pdf"
    pdf.write_bytes(b"%PDF-1.4 stub")

    emit_chunks.main([str(pdf)])

    payload = json.loads(capsys.readouterr().out)
    assert payload["chunks"] == []
    assert payload["sha256"] == "a" * 64

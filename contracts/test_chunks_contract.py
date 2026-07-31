"""Chunks-schema contract test.

The retrieval-index seam: scripts/emit_chunks.py emits chunk JSON that the
backend validates against the SAME contracts/chunks.schema.json before ingest.
Like the claims contract, this validates one CHUNK (the backend walks
payload['chunks']). A real ChunkRecord.model_dump(mode="json") is run through
the validator so a serializer drift from the schema fails here, not silently at
insert -- the one seam the claim path has a contract for and the chunk path did
not.

Run: ``uv run --with jsonschema pytest -q contracts/test_chunks_contract.py``
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from parser_service.chunker import ChunkRecord
from parser_service.schemas import BBox

SCHEMA_PATH = Path(__file__).parent / "chunks.schema.json"


@pytest.fixture(scope="module")
def validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text())
    # Fail fast if the schema itself is malformed.
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


# --- Representative valid chunks, one per shape ------------------------------
# Hand-authored, so they prove the FIXTURES match the schema; the real-emitted
# test below is what proves the emitter does.

VALID_PROSE_CHUNK = {
    "content": "Revenue was $15,295 in fiscal 2024, driven by strong demand.",
    "element_type": "prose",
    "page": 11,
    "order": 3,
    "document_id": "a" * 64,
    "source_file": "cim.pdf",
    "scale_context": "CAD in Thousands",
    "scale_multiplier": 1000.0,
    "spans": [[0, 60]],
    "bbox": None,
    "section": "Financial Summary",
    "flags": [],
}

# A chart chunk: no prose spans, cited by a region bbox instead.
VALID_CHART_CHUNK = {
    "content": "Revenue by segment",
    "element_type": "chart",
    "page": 12,
    "order": 4,
    "document_id": "a" * 64,
    "source_file": "cim.pdf",
    "scale_context": None,
    "scale_multiplier": None,
    "spans": [],
    "bbox": {"x0": 72.0, "top": 100.0, "x1": 300.0, "bottom": 240.0, "page": 12},
    "section": None,
    "flags": ["chart_data_not_extracted"],
}


@pytest.mark.parametrize("chunk", [VALID_PROSE_CHUNK, VALID_CHART_CHUNK], ids=["prose", "chart"])
def test_valid_chunks_pass(validator: Draft202012Validator, chunk: dict) -> None:
    errors = sorted(validator.iter_errors(chunk), key=str)
    assert not errors, "\n".join(e.message for e in errors)


def test_a_real_emitted_chunk_conforms(validator: Draft202012Validator) -> None:
    # emit_chunks emits ChunkRecord.model_dump(mode="json"); validate that real
    # output so a field rename/retype on ChunkRecord fails here, not at ingest.
    prose = ChunkRecord(
        content="Revenue was $15,295 in fiscal 2024.",
        element_type="prose",
        page=11,
        order=0,
        document_id="a" * 64,
        source_file="cim.pdf",
        spans=[(0, 35)],
        section="Financials",
    ).model_dump(mode="json")
    chart = ChunkRecord(
        content="Revenue by segment",
        element_type="chart",
        page=12,
        order=1,
        document_id="a" * 64,
        source_file="cim.pdf",
        bbox=BBox(x0=72.0, top=100.0, x1=300.0, bottom=240.0, page=12),
    ).model_dump(mode="json")
    # The table shape is the one that populates scale_multiplier (float), a bbox
    # citation, and flags -- the fields a prose/chart-only case never exercises.
    table = ChunkRecord(
        content="Revenue 15,295 | EBITDA 4,200",
        element_type="table",
        page=11,
        order=2,
        document_id="a" * 64,
        source_file="cim.pdf",
        scale_context="$ in thousands",
        scale_multiplier=1000.0,
        bbox=BBox(x0=50.0, top=200.0, x1=520.0, bottom=410.0, page=11),
        flags=["ragged_table_rows"],
    ).model_dump(mode="json")
    for emitted in (prose, chart, table):
        errors = sorted(validator.iter_errors(emitted), key=str)
        assert not errors, "\n".join(e.message for e in errors)


# --- Things that MUST be rejected (the contract's teeth) ---------------------


def test_unknown_element_type_rejected(validator: Draft202012Validator) -> None:
    bad = {**VALID_PROSE_CHUNK, "element_type": "paragraph"}
    assert list(validator.iter_errors(bad)), "unknown element_type should be rejected"


def test_extra_top_level_key_rejected(validator: Draft202012Validator) -> None:
    bad = {**VALID_PROSE_CHUNK, "surprise": "drift"}
    assert list(validator.iter_errors(bad)), "unexpected chunk key should be rejected"


def test_page_must_be_one_based(validator: Draft202012Validator) -> None:
    bad = {**VALID_PROSE_CHUNK, "page": 0}
    assert list(validator.iter_errors(bad)), "page < 1 should be rejected"


def test_span_must_be_a_pair(validator: Draft202012Validator) -> None:
    bad = {**VALID_PROSE_CHUNK, "spans": [[0, 10, 20]]}
    assert list(validator.iter_errors(bad)), "a span must be exactly (start, end)"


@pytest.mark.parametrize(
    "field",
    # ChunkRecord.model_dump(mode="json") emits ALL 12 fields (nulls included),
    # so every one is required -- dropping any (e.g. a serializer that stopped
    # emitting `spans` or `flags`) must fail here, not silently at ingest.
    [
        "content",
        "element_type",
        "page",
        "order",
        "document_id",
        "source_file",
        "scale_context",
        "scale_multiplier",
        "spans",
        "bbox",
        "section",
        "flags",
    ],
)
def test_required_fields(validator: Draft202012Validator, field: str) -> None:
    bad = {k: v for k, v in VALID_PROSE_CHUNK.items() if k != field}
    assert list(validator.iter_errors(bad)), f"chunk without {field!r} should be rejected"

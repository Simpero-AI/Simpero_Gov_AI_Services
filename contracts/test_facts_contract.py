"""Facts-schema contract test (CI job #7).

Validates that fact JSON — the shape the Python parse service emits across the
seam to the TS app — conforms to the shared ``facts.schema.json`` (the C3
contract). The TS side validates against the SAME schema file, so a drift on
either side of the boundary fails CI here rather than silently in production.

This is the one project-specific check: the Python->TS seam is the crux of the
architecture, and a silent shape drift there is exactly the failure mode the
product exists to prevent.

Run: ``uv run --with jsonschema pytest -q contracts/test_facts_contract.py``
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

SCHEMA_PATH = Path(__file__).parent / "facts.schema.json"


@pytest.fixture(scope="module")
def validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text())
    # Fail fast if the schema itself is malformed.
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


# --- Representative valid facts, one per location kind -----------------------
# These double as living documentation of the contract. When the emitter (DS-7)
# is built, replace/supplement these with a sample of its real output.

VALID_PDF_FACT = {
    "entity": "PTL Group",
    "attribute": "revenueTrailing5yrAvg",
    "value": {
        "raw": "$15,295",
        "normalized": 15295000,
        "unit": "CAD",
        "scale_multiplier": 1000,
        "scale_source": "page_header",
        "value_type": "currency",
    },
    "location": {
        "kind": "pdf",
        "file": "1st-App-H-PTL-Group-CIM.pdf",
        "page": 11,
        "char_start": 1502,
        "char_end": 1509,
        "bbox": [[265.4, 486.7, 300.5, 496.6]],
    },
    "confidence": "extracted",
    "flags": [],
}

VALID_XLSX_FACT = {
    "entity": "TargetCo",
    "attribute": "ebitdaFy2024",
    "value": {
        "raw": "8100000",
        "normalized": 8100000,
        "unit": "USD",
        "scale_multiplier": 1,
        "scale_source": "explicit_in_value",
        "value_type": "currency",
    },
    "location": {
        "kind": "xlsx",
        "file": "model.xlsx",
        "sheet": "Financials",
        "cell_ref": "B14",
    },
    "confidence": "formula-verified",
    "flags": [],
}

VALID_MISSING_FACT = {
    "entity": "TargetCo",
    "attribute": "churnRate",
    "value": {
        "raw": "",
        "normalized": None,
        "unit": None,
        "scale_source": "assumed_1x",
        "value_type": "text",
    },
    "location": {
        "kind": "pdf",
        "file": "deck.pdf",
        "page": 4,
        "char_start": 0,
        "char_end": 0,
    },
    "confidence": "missing",
    "flags": ["quote_unresolved"],
}


@pytest.mark.parametrize(
    "fact",
    [VALID_PDF_FACT, VALID_XLSX_FACT, VALID_MISSING_FACT],
    ids=["pdf", "xlsx", "missing"],
)
def test_valid_facts_pass(validator: Draft202012Validator, fact: dict) -> None:
    errors = sorted(validator.iter_errors(fact), key=str)
    assert not errors, "\n".join(e.message for e in errors)


# --- Things that MUST be rejected (the contract's teeth) ---------------------


def test_unknown_confidence_rejected(validator: Draft202012Validator) -> None:
    bad = {**VALID_PDF_FACT, "confidence": "probably_right"}
    assert list(validator.iter_errors(bad)), "unknown confidence should be rejected"


def test_unknown_scale_source_rejected(validator: Draft202012Validator) -> None:
    bad = json.loads(json.dumps(VALID_PDF_FACT))
    bad["value"]["scale_source"] = "guessed"
    assert list(validator.iter_errors(bad)), "unknown scale_source should be rejected"


def test_pdf_location_requires_char_span(validator: Draft202012Validator) -> None:
    # All-or-nothing provenance: a pdf fact without a char span is invalid.
    bad = json.loads(json.dumps(VALID_PDF_FACT))
    del bad["location"]["char_end"]
    assert list(validator.iter_errors(bad)), "pdf location must carry a full char span"


def test_xlsx_location_requires_cell_ref(validator: Draft202012Validator) -> None:
    bad = json.loads(json.dumps(VALID_XLSX_FACT))
    del bad["location"]["cell_ref"]
    assert list(validator.iter_errors(bad)), "xlsx location must carry sheet + cell_ref"


def test_extra_top_level_key_rejected(validator: Draft202012Validator) -> None:
    # additionalProperties:false — an unexpected key means the two sides drifted.
    bad = {**VALID_PDF_FACT, "surprise": "drift"}
    assert list(validator.iter_errors(bad)), "unexpected top-level key should be rejected"


def test_unknown_flag_rejected(validator: Draft202012Validator) -> None:
    bad = {**VALID_PDF_FACT, "flags": ["totally_made_up_flag"]}
    assert list(validator.iter_errors(bad)), "unknown flag should be rejected"

"""Claims-schema contract test (CI job #7).

Validates that claim JSON — the shape the Python parse service emits across the
seam to the backend — conforms to the shared ``claims.schema.json`` (the C3
contract). The consuming side validates against the SAME schema file, so a drift
on either side of the boundary fails CI here rather than silently in production.

This is the one project-specific check: the parse->backend seam is the crux of
the architecture, and a silent shape drift there is exactly the failure mode the
product exists to prevent.

Run: ``uv run --with jsonschema pytest -q contracts/test_claims_contract.py``
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

SCHEMA_PATH = Path(__file__).parent / "claims.schema.json"


@pytest.fixture(scope="module")
def validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text())
    # Fail fast if the schema itself is malformed.
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


# --- Representative valid claims, one per location kind ----------------------
# These double as living documentation of the contract. They are hand-authored,
# which means they prove the FIXTURES match the schema, not that the emitter
# does. When the emitter (DS-W3-7) lands, replace these with a sample of its
# real output -- otherwise this test cannot catch the drift it exists to catch.

# PDF/DOCX text extraction is emitted at `proposed`: a citation is attached, but
# nothing has checked it yet -- Verify moves it to cited|rejected. An extractor
# asserting `cited` here would be claiming a check it never ran.
VALID_PDF_CLAIM = {
    "entity": "PTL Group",
    "attribute": "revenueTrailing5yrAvg",
    "period_year": 2024,
    "period_kind": "A",
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
    "status": "proposed",
    "verification_method": None,
    "flags": [],
}

# An XLSX formula cell: HyperFormula re-executed it, so it lands `cited` by
# `formula_reexecution` -- the highest-trust path, and the point of XLSX support.
VALID_XLSX_CLAIM = {
    "entity": "TargetCo",
    "attribute": "ebitdaFy2024",
    "period_year": 2024,
    "period_kind": "A",
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
    "status": "cited",
    "verification_method": "formula_reexecution",
    "flags": [],
}

# A literal (non-formula) XLSX cell: a byte-exact read has no ambiguity worth
# checking, so it goes straight to `cited` via `direct_read`.
VALID_XLSX_LITERAL_CLAIM = {
    "entity": "TargetCo",
    "attribute": "headcount",
    "value": {
        "raw": "1200",
        "normalized": 1200,
        "unit": None,
        "scale_multiplier": 1,
        "scale_source": "explicit_in_value",
        "value_type": "count",
    },
    "location": {
        "kind": "xlsx",
        "file": "model.xlsx",
        "sheet": "Ops",
        "cell_ref": "C7",
    },
    "status": "cited",
    "verification_method": "direct_read",
    "flags": [],
}

VALID_DOCX_CLAIM = {
    "entity": "TargetCo",
    "attribute": "customerConcentrationNote",
    "value": {
        "raw": "top three customers represent 62% of revenue",
        "normalized": None,
        "unit": None,
        "scale_source": "assumed_1x",
        "value_type": "text",
    },
    "location": {
        "kind": "docx",
        "file": "management-discussion.docx",
        "paragraph": 42,
        "char_start": 118,
        "char_end": 161,
    },
    "status": "proposed",
    "verification_method": None,
    "flags": [],
}

# No citation exists in the source: emitted `missing`, never fabricated, never a
# partial citation. Surfaces to users as "Missing".
VALID_MISSING_CLAIM = {
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
    "status": "missing",
    "verification_method": None,
    "flags": ["quote_unresolved"],
}


@pytest.mark.parametrize(
    "claim",
    [
        VALID_PDF_CLAIM,
        VALID_XLSX_CLAIM,
        VALID_XLSX_LITERAL_CLAIM,
        VALID_DOCX_CLAIM,
        VALID_MISSING_CLAIM,
    ],
    ids=["pdf", "xlsx_formula", "xlsx_literal", "docx", "missing"],
)
def test_valid_claims_pass(validator: Draft202012Validator, claim: dict) -> None:
    errors = sorted(validator.iter_errors(claim), key=str)
    assert not errors, "\n".join(e.message for e in errors)


# --- Things that MUST be rejected (the contract's teeth) ---------------------


def test_unknown_status_rejected(validator: Draft202012Validator) -> None:
    bad = {**VALID_PDF_CLAIM, "status": "probably_right"}
    assert list(validator.iter_errors(bad)), "unknown status should be rejected"


def test_retired_confidence_key_rejected(validator: Draft202012Validator) -> None:
    # `confidence` was replaced by status + verification_method. An emitter still
    # sending the retired key never got the memo -- fail loudly rather than drop
    # it silently.
    bad = {**VALID_PDF_CLAIM, "confidence": "extracted"}
    assert list(validator.iter_errors(bad)), "retired `confidence` key should be rejected"


def test_unknown_verification_method_rejected(validator: Draft202012Validator) -> None:
    bad = {**VALID_XLSX_CLAIM, "verification_method": "vibes"}
    assert list(validator.iter_errors(bad)), "unknown verification_method should be rejected"


def test_cited_requires_a_verification_method(validator: Draft202012Validator) -> None:
    # Mirrors the claims-table CHECK: once a claim is cited or later we must know
    # HOW it earned that trust, not merely that it did.
    bad = {**VALID_XLSX_CLAIM, "verification_method": None}
    assert list(validator.iter_errors(bad)), "cited claim must carry a verification_method"


def test_proposed_may_omit_verification_method(validator: Draft202012Validator) -> None:
    # The inverse: nothing has been checked yet, so null is correct, not an error.
    ok = {**VALID_PDF_CLAIM, "verification_method": None}
    assert not list(validator.iter_errors(ok)), "proposed claim may have a null method"


def test_unknown_scale_source_rejected(validator: Draft202012Validator) -> None:
    bad = json.loads(json.dumps(VALID_PDF_CLAIM))
    bad["value"]["scale_source"] = "guessed"
    assert list(validator.iter_errors(bad)), "unknown scale_source should be rejected"


def test_pdf_location_requires_char_span(validator: Draft202012Validator) -> None:
    # All-or-nothing provenance: a pdf claim without a char span is invalid.
    bad = json.loads(json.dumps(VALID_PDF_CLAIM))
    del bad["location"]["char_end"]
    assert list(validator.iter_errors(bad)), "pdf location must carry a full char span"


def test_xlsx_location_requires_cell_ref(validator: Draft202012Validator) -> None:
    bad = json.loads(json.dumps(VALID_XLSX_CLAIM))
    del bad["location"]["cell_ref"]
    assert list(validator.iter_errors(bad)), "xlsx location must carry sheet + cell_ref"


def test_docx_location_requires_paragraph(validator: Draft202012Validator) -> None:
    bad = json.loads(json.dumps(VALID_DOCX_CLAIM))
    del bad["location"]["paragraph"]
    assert list(validator.iter_errors(bad)), "docx location must carry a paragraph"


def test_docx_location_requires_char_span(validator: Draft202012Validator) -> None:
    # DOCX is a linear text stream: a paragraph index alone is not a citation.
    # Same all-or-nothing rule as PDF.
    bad = json.loads(json.dumps(VALID_DOCX_CLAIM))
    del bad["location"]["char_end"]
    assert list(validator.iter_errors(bad)), "docx location must carry a full char span"


def test_file_is_optional_data_source_id_is_the_identity(validator: Draft202012Validator) -> None:
    # The claims table has no `file` column — data_source_id replaced it. `file`
    # is debug-only, so a claim without it must still validate.
    ok = json.loads(json.dumps(VALID_PDF_CLAIM))
    del ok["location"]["file"]
    ok["data_source_id"] = "3f1a5c9e-0000-4000-8000-000000000001"
    assert not list(validator.iter_errors(ok)), "file must be optional; data_source_id is identity"


def test_unknown_period_kind_rejected(validator: Draft202012Validator) -> None:
    bad = {**VALID_PDF_CLAIM, "period_kind": "Q"}
    assert list(validator.iter_errors(bad)), "period_kind must be one of A|E|P"


def test_extra_top_level_key_rejected(validator: Draft202012Validator) -> None:
    # additionalProperties:false -- an unexpected key means the two sides drifted.
    bad = {**VALID_PDF_CLAIM, "surprise": "drift"}
    assert list(validator.iter_errors(bad)), "unexpected top-level key should be rejected"


def test_unknown_flag_rejected(validator: Draft202012Validator) -> None:
    bad = {**VALID_PDF_CLAIM, "flags": ["totally_made_up_flag"]}
    assert list(validator.iter_errors(bad)), "unknown flag should be rejected"


def test_xlsx_blocking_flags_accepted(validator: Draft202012Validator) -> None:
    # The XLSX path's blocking flags must be part of the contract.
    ok = {**VALID_XLSX_CLAIM, "flags": ["formula_mismatch", "external_reference_unresolved"]}
    assert not list(validator.iter_errors(ok)), "xlsx blocking flags should validate"

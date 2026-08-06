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

from parser_service.emit import CANONICAL_ATTRIBUTES, Edge
from parser_service.scale import scale_invariant_holds

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
    "claim_ref": "11:1502-1509[#0]",
    "claim_type": "numerical",
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
    "claim_ref": "Financials!B14[#0]",
    "claim_type": "computational",
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
    "claim_ref": "Ops!C7[#0]",
    "claim_type": "numerical",
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
    "claim_ref": "42:118-161[#0]",
    "claim_type": "entity_attribute",
    "attribute": "customerConcentrationNote",
    "value": {
        "raw": "top three customers represent 62% of revenue",
        "normalized": None,
        "unit": None,
        "scale_source": "not_applicable",
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
# partial citation. Surfaces to users as "Missing". The location records the page
# that was searched -- where we LOOKED -- and carries no char span, because there
# is nothing to point at. It must not invent one.
VALID_MISSING_CLAIM = {
    "entity": "TargetCo",
    "claim_ref": "4:none[#0]",
    "claim_type": "numerical",
    "attribute": "churnRate",
    "value": {
        "raw": "",
        "normalized": None,
        "unit": None,
        "scale_source": "not_applicable",
        "value_type": "text",
    },
    "location": {
        "kind": "pdf",
        "file": "deck.pdf",
        "page": 4,
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


def test_a_value_with_a_magnitude_must_carry_its_multiplier_and_source(
    validator: Draft202012Validator,
) -> None:
    # Recording the multiplier WITHOUT its source, or the source without the
    # multiplier, proves nothing: one is a bare number nobody can audit, the
    # other asserts a header was read while withholding what it said.
    for field in ("scale_multiplier", "scale_source"):
        bad = json.loads(json.dumps(VALID_PDF_CLAIM))
        del bad["value"][field]
        assert list(validator.iter_errors(bad)), f"a numeric value must carry {field}"


def test_a_value_with_no_magnitude_needs_no_scale(validator: Draft202012Validator) -> None:
    # Scale records how raw BECAME normalized. Where nothing was produced --
    # value_type text, or an unevaluated formula stub -- there is nothing to
    # describe, and demanding a multiplier would force a fabricated one.
    ok = json.loads(json.dumps(VALID_PDF_CLAIM))
    ok["value"] = {"raw": "four", "normalized": None, "unit": None, "value_type": "text"}
    assert not list(validator.iter_errors(ok)), "text has no magnitude to scale"


def test_the_scale_arithmetic_is_checked_outside_the_schema() -> None:
    """normalized must be raw scaled by the multiplier.

    JSON Schema cannot express arithmetic across fields, so the schema alone
    cannot catch this and never will. Before scale_invariant_holds existed, a
    claim reading raw="$15,295", normalized=999, scale_source="page_header"
    validated perfectly clean -- a 15,000x error the contract had no opinion
    about, and the exact shape of every defect found in the July 2026 audit.
    """
    assert scale_invariant_holds("$15,295", 15_295_000.0, 1_000.0)
    assert not scale_invariant_holds("$15,295", 999.0, 1_000.0)
    # The specific direction that keeps recurring: the multiplier was found but
    # never applied, so the value ships a thousandth of its true magnitude.
    assert not scale_invariant_holds("$15,295", 15_295.0, 1_000.0)
    # A known 1x still has to add up.
    assert scale_invariant_holds("27.3%", 27.3, 1.0)
    assert not scale_invariant_holds("27.3%", 2_730.0, 1.0)
    # Accounting negatives keep their sign through the multiplication.
    assert scale_invariant_holds("(2.3 )", -2_300_000.0, 1_000_000.0)
    # Absent inputs are not silently satisfied.
    assert not scale_invariant_holds("$15,295", None, 1_000.0)
    assert not scale_invariant_holds("$15,295", 15_295_000.0, None)


def test_every_valid_fixture_satisfies_the_scale_arithmetic() -> None:
    # The fixtures are what everything else in this file is judged against, so
    # a fixture that violated the invariant would quietly license the defect.
    for claim in (VALID_PDF_CLAIM, VALID_XLSX_CLAIM, VALID_XLSX_LITERAL_CLAIM, VALID_DOCX_CLAIM):
        value = claim["value"]
        if value.get("normalized") is None or value["value_type"] == "text":
            continue
        assert scale_invariant_holds(
            value["raw"], value["normalized"], value.get("scale_multiplier")
        ), f"fixture {claim['attribute']!r} violates normalized == raw x multiplier"


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


def test_missing_claim_must_not_carry_a_char_span(validator: Draft202012Validator) -> None:
    # The other half of all-or-nothing, and the sharper half. A `missing` claim
    # found nothing, so char_start=0/char_end=0 is not "no span" -- it is a
    # citation to the top of the page, which a highlight UI would happily draw.
    # The schema used to require the span unconditionally, which forced exactly
    # this fabrication; it is now rejected outright.
    bad = json.loads(json.dumps(VALID_MISSING_CLAIM))
    bad["location"]["char_start"] = 0
    bad["location"]["char_end"] = 0
    assert list(validator.iter_errors(bad)), "a missing claim must not fabricate a span"


def test_missing_claim_still_records_where_it_looked(validator: Draft202012Validator) -> None:
    # Dropping the span does not license dropping the locator: "we searched
    # page 4 and found nothing" is the useful claim; "we found nothing
    # somewhere" is not.
    bad = json.loads(json.dumps(VALID_MISSING_CLAIM))
    del bad["location"]["page"]
    assert list(validator.iter_errors(bad)), "a missing claim must still cite the page searched"


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


# --- SIM-344: attribute_raw + attribute_unmapped ------------------------------


def test_attribute_raw_accepted_alongside_canonical_attribute(
    validator: Draft202012Validator,
) -> None:
    ok = {**VALID_PDF_CLAIM, "attribute": "revenue", "attribute_raw": "Revenue | 2019F"}
    assert not list(validator.iter_errors(ok)), "attribute_raw should validate alongside attribute"


def test_attribute_raw_is_optional(validator: Draft202012Validator) -> None:
    # A claim the canonicalization pass never reached (e.g. qualitative) must
    # still validate without attribute_raw.
    assert "attribute_raw" not in VALID_PDF_CLAIM
    assert not list(validator.iter_errors(VALID_PDF_CLAIM))


def test_attribute_unmapped_flag_accepted(validator: Draft202012Validator) -> None:
    ok = {**VALID_PDF_CLAIM, "flags": ["attribute_unmapped"]}
    assert not list(validator.iter_errors(ok)), "attribute_unmapped should be a valid flag"


def test_attribute_raw_present_requires_canonical_attribute(
    validator: Draft202012Validator,
) -> None:
    # VALID_PDF_CLAIM's default attribute ("revenueTrailing5yrAvg") is a stale
    # pre-SIM-344 label -- once attribute_raw is pinned (E2 ran), it must fail.
    bad = {**VALID_PDF_CLAIM, "attribute_raw": "Revenue | 5yr Avg"}
    errors = list(validator.iter_errors(bad))
    assert errors, "non-canonical attribute alongside attribute_raw should fail validation"


# --- SIM-341: the `edges` array's element shape ------------------------------
# `edges` sits beside `claims` in the emitted payload, not inside a Claim row,
# so it is validated against the schema's `$defs/edge` sub-schema directly
# rather than through the top-level Claim validator above.

VALID_SAME_FACT_EDGE = {
    "type": "same_fact",
    "from": "3:500-551[#0]",
    "to": "3:10-17[#0]",
    "basis": "page 3: table and prose tiers agree on entity, value type + normalized value",
}

VALID_CONTRADICTS_EDGE = {
    "type": "contradicts",
    "from": "5:10-21[#0]",
    "to": "5:200-233[#0]",
    "basis": "page 5: table ('$15,000,000') and prose ('$12,000,000 excluding "
    "settlement') disagree on the same attribute",
}


@pytest.fixture(scope="module")
def edge_validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text())
    edge_schema = schema["$defs"]["edge"]
    Draft202012Validator.check_schema(edge_schema)
    return Draft202012Validator(edge_schema)


@pytest.mark.parametrize(
    "edge",
    [VALID_SAME_FACT_EDGE, VALID_CONTRADICTS_EDGE],
    ids=["same_fact", "contradicts"],
)
def test_valid_edges_pass(edge_validator: Draft202012Validator, edge: dict) -> None:
    errors = sorted(edge_validator.iter_errors(edge), key=str)
    assert not errors, "\n".join(e.message for e in errors)


def test_unknown_edge_type_rejected(edge_validator: Draft202012Validator) -> None:
    bad = {**VALID_SAME_FACT_EDGE, "type": "duplicate_of"}
    assert list(edge_validator.iter_errors(bad)), "unknown edge type should be rejected"


def test_edge_extra_top_level_key_rejected(edge_validator: Draft202012Validator) -> None:
    bad = {**VALID_SAME_FACT_EDGE, "surprise": "drift"}
    assert list(edge_validator.iter_errors(bad)), "unexpected edge key should be rejected"


@pytest.mark.parametrize("field", ["type", "from", "to", "basis"])
def test_edge_requires_all_fields(edge_validator: Draft202012Validator, field: str) -> None:
    bad = {k: v for k, v in VALID_SAME_FACT_EDGE.items() if k != field}
    assert list(edge_validator.iter_errors(bad)), f"edge without {field!r} should be rejected"


def test_a_real_emitted_edge_conforms_to_the_schema(edge_validator: Draft202012Validator) -> None:
    # The literals above are hand-authored; a real Edge.to_json() can drift from
    # them and still be the thing actually shipped. Validate the serializer's own
    # output so a rename like `from_` no longer mapping to `from` fails here.
    edge = Edge(
        type="same_fact",
        from_="3:500-551[#0]",
        to="3:10-17[#0]",
        basis="page 3: table and prose tiers agree on entity, value type + normalized value",
    ).to_json()
    errors = sorted(edge_validator.iter_errors(edge), key=str)
    assert not errors, "\n".join(e.message for e in errors)


def test_canonical_attribute_def_matches_the_parser_vocabulary() -> None:
    """SIM-375: $defs/canonicalAttribute is the parser's canonical attribute
    vocabulary published into the C3 contract, so both repos and every cross-claim
    consumer (3b consistency, scoring) key on the SAME names. Keep it identical to
    emit.CANONICAL_ATTRIBUTES (CoreAttribute + operating_metric): if the parser adds
    a canonical attribute, this fails until the contract publishes it too."""
    schema = json.loads(SCHEMA_PATH.read_text())
    published = set(schema["$defs"]["canonicalAttribute"]["enum"])
    assert published == CANONICAL_ATTRIBUTES

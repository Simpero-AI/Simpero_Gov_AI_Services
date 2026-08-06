"""Claims-payload envelope contract test (SIM-349).

Validates the wrapper `extract_service.extract_claims` returns -- run_id,
sha256, source_file, claims, edges, flag_log, skipped_pages -- against
``claims_payload.schema.json``. `claims.schema.json` only ever formalized one
Claim row; `flag_log` and `skipped_pages` shipped unvalidated from the day
each was added, because no contract slot existed for either. This test closes
that gap the same way test_claims_contract.py closes it for a single claim.

`claims_payload.schema.json` references `claims.schema.json` by `$id` (the
`claims`/`edges` array items, and `flag_log[].flag_type`), so validation here
needs both schemas in one `referencing.Registry` rather than a single
standalone `Draft202012Validator`.

Run: ``uv run --with jsonschema pytest -q contracts/test_claims_payload_contract.py``
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from referencing import Registry, Resource

from parser_service import extract_service
from parser_service.emit import ClaimValue, PdfLocation

CLAIMS_SCHEMA_PATH = Path(__file__).parent / "claims.schema.json"
PAYLOAD_SCHEMA_PATH = Path(__file__).parent / "claims_payload.schema.json"


@pytest.fixture(scope="module")
def validator() -> Draft202012Validator:
    claims_schema = json.loads(CLAIMS_SCHEMA_PATH.read_text())
    payload_schema = json.loads(PAYLOAD_SCHEMA_PATH.read_text())
    # Fail fast if either schema itself is malformed.
    Draft202012Validator.check_schema(claims_schema)
    Draft202012Validator.check_schema(payload_schema)
    registry = Registry().with_resources(
        [
            (claims_schema["$id"], Resource.from_contents(claims_schema)),
            (payload_schema["$id"], Resource.from_contents(payload_schema)),
        ]
    )
    return Draft202012Validator(payload_schema, registry=registry)


# --- A representative valid payload -------------------------------------------
# Hand-authored, doubling as living documentation of the envelope shape.

VALID_CLAIM = {
    "entity": "PTL Group",
    "claim_ref": "11:1502-1509[#0]",
    "claim_type": "numerical",
    "attribute": "revenue",
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
    },
    "status": "proposed",
    "verification_method": None,
    "flags": [],
}

VALID_EDGE = {
    "type": "same_fact",
    "from": "11:1502-1509[#0]",
    "to": "11:1502-1509[#1]",
    "basis": "page 11: table and prose tiers agree on entity, value type + normalized value",
}

VALID_FLAG_LOG_ENTRY = {
    "run_id": "11111111-1111-1111-1111-111111111111",
    "stage": "claim_emission",
    "element_id": "pdf:1st-App-H-PTL-Group-CIM.pdf:p11:revenue",
    "flag_type": "scale_assumed",
}

VALID_SKIPPED_PAGE = {"page": 4, "tier": "prose", "reason": "TimeoutError: model call timed out"}

VALID_PAYLOAD = {
    "run_id": "11111111-1111-1111-1111-111111111111",
    "sha256": "a" * 64,
    "source_file": "1st-App-H-PTL-Group-CIM.pdf",
    "claims": [VALID_CLAIM],
    "edges": [VALID_EDGE],
    "flag_log": [VALID_FLAG_LOG_ENTRY],
    "skipped_pages": [VALID_SKIPPED_PAGE],
}


def test_valid_payload_passes(validator: Draft202012Validator) -> None:
    errors = sorted(validator.iter_errors(VALID_PAYLOAD), key=str)
    assert not errors, "\n".join(e.message for e in errors)


def test_source_file_null_is_accepted(validator: Draft202012Validator) -> None:
    # extract_service.extract_claims deliberately leaves source_file None when
    # the caller has no real filename -- must validate, not just be tolerated.
    ok = {**VALID_PAYLOAD, "source_file": None}
    assert not list(validator.iter_errors(ok)), "a null source_file must validate"


def test_empty_arrays_are_accepted(validator: Draft202012Validator) -> None:
    # The common case for claims/edges/flag_log/skipped_pages on a clean run.
    ok = {**VALID_PAYLOAD, "claims": [], "edges": [], "flag_log": [], "skipped_pages": []}
    assert not list(validator.iter_errors(ok)), "empty arrays are a valid, common payload shape"


# --- Things that MUST be rejected (the contract's teeth) ---------------------


def test_extra_top_level_key_rejected(validator: Draft202012Validator) -> None:
    bad = {**VALID_PAYLOAD, "surprise": "drift"}
    assert list(validator.iter_errors(bad)), "unexpected top-level key should be rejected"


@pytest.mark.parametrize(
    "key", ["run_id", "sha256", "source_file", "claims", "edges", "flag_log", "skipped_pages"]
)
def test_missing_required_key_rejected(validator: Draft202012Validator, key: str) -> None:
    bad = {k: v for k, v in VALID_PAYLOAD.items() if k != key}
    assert list(validator.iter_errors(bad)), f"payload without {key!r} should be rejected"


def test_claims_array_item_must_conform_to_the_claim_schema(
    validator: Draft202012Validator,
) -> None:
    # Proves the `claims` items actually resolve to claims.schema.json, not just
    # `type: object` -- a claim missing its required claim_ref must fail here.
    bad_claim = {k: v for k, v in VALID_CLAIM.items() if k != "claim_ref"}
    bad = {**VALID_PAYLOAD, "claims": [bad_claim]}
    assert list(validator.iter_errors(bad)), "an invalid claim inside the array should be rejected"


def test_edges_array_item_must_conform_to_the_edge_schema(
    validator: Draft202012Validator,
) -> None:
    bad_edge = {**VALID_EDGE, "type": "duplicate_of"}
    bad = {**VALID_PAYLOAD, "edges": [bad_edge]}
    assert list(validator.iter_errors(bad)), "an invalid edge inside the array should be rejected"


def test_unknown_flag_type_in_flag_log_rejected(validator: Draft202012Validator) -> None:
    bad_entry = {**VALID_FLAG_LOG_ENTRY, "flag_type": "totally_made_up_flag"}
    bad = {**VALID_PAYLOAD, "flag_log": [bad_entry]}
    assert list(validator.iter_errors(bad)), "unknown flag_type should be rejected"


def test_flag_log_entry_missing_required_field_rejected(validator: Draft202012Validator) -> None:
    bad_entry = {k: v for k, v in VALID_FLAG_LOG_ENTRY.items() if k != "element_id"}
    bad = {**VALID_PAYLOAD, "flag_log": [bad_entry]}
    assert list(validator.iter_errors(bad)), (
        "a flag_log entry without element_id should be rejected"
    )


def test_flag_log_entry_extra_key_rejected(validator: Draft202012Validator) -> None:
    bad_entry = {**VALID_FLAG_LOG_ENTRY, "surprise": "drift"}
    bad = {**VALID_PAYLOAD, "flag_log": [bad_entry]}
    assert list(validator.iter_errors(bad)), "unexpected flag_log entry key should be rejected"


def test_unknown_tier_in_skipped_pages_rejected(validator: Draft202012Validator) -> None:
    bad_entry = {**VALID_SKIPPED_PAGE, "tier": "made_up_tier"}
    bad = {**VALID_PAYLOAD, "skipped_pages": [bad_entry]}
    assert list(validator.iter_errors(bad)), "unknown skipped_pages tier should be rejected"


def test_every_known_tier_is_accepted(validator: Draft202012Validator) -> None:
    # Every tier extract_service.py actually names in a SkippedPage today.
    for tier in ("tables", "prose", "complete", "qualitative", "attribute_mapping"):
        ok_entry = {**VALID_SKIPPED_PAGE, "tier": tier}
        ok = {**VALID_PAYLOAD, "skipped_pages": [ok_entry]}
        assert not list(validator.iter_errors(ok)), f"tier {tier!r} should validate"


def test_skipped_page_extra_key_rejected(validator: Draft202012Validator) -> None:
    bad_entry = {**VALID_SKIPPED_PAGE, "surprise": "drift"}
    bad = {**VALID_PAYLOAD, "skipped_pages": [bad_entry]}
    assert list(validator.iter_errors(bad)), "unexpected skipped_pages entry key should be rejected"


# --- A real emitted payload conforms too --------------------------------------
# The fixtures above are hand-authored, which proves them consistent with the
# schema, not that extract_claims's own payload dict is. Drive the real
# function (monkeypatching only the docling/table internals, the same way
# tests/test_extract_service.py does) so a rename or shape drift in the
# `payload = {...}` construction fails here too.


class _Doc:
    pass


class _MultiPageResult:
    document = _Doc()
    pages = [type("Page", (), {"page": 1})(), type("Page", (), {"page": 2})()]
    sha256 = "0" * 64


def _fake_tables_on_page(_tables, page_no: int) -> list[str]:
    return {1: ["t1-bad", "t1-good"], 2: ["t2-good"]}[page_no]


def _fake_claims_from_table(table, page, *, entity, file, flag_log):
    if table == "t1-bad":
        raise ValueError("a malformed table")
    flag_log.log("claim_emission", f"{table}:{page.page}", "ragged_table_rows")
    return [
        extract_service.Claim(
            entity=entity,
            attribute=table,
            value=ClaimValue(
                raw="$1",
                normalized=1.0,
                unit="USD",
                value_type="currency",
                scale_multiplier=1.0,
                scale_source="explicit_in_value",
            ),
            location=PdfLocation(file=file, page=page.page, char_start=0, char_end=2),
            status="proposed",
        )
    ]


def test_a_real_emitted_payload_conforms_to_the_schema(
    validator: Draft202012Validator, monkeypatch
) -> None:
    monkeypatch.setattr(extract_service, "parse_pdf_bytes", lambda _b: _MultiPageResult())
    monkeypatch.setattr(extract_service, "extract_tables", lambda *_a, **_k: [])
    monkeypatch.setattr(extract_service, "tables_on_page", _fake_tables_on_page)
    monkeypatch.setattr(extract_service, "claims_from_table", _fake_claims_from_table)

    payload = extract_service.extract_claims(
        b"%PDF-1.4 stub",
        entity="ACME",
        run_id="run-1",
        correlation_id="doc-1",
        source_file="cim.pdf",
    )
    # A bad table (t1-bad) and one raised flag, so both flag_log and
    # skipped_pages are non-empty -- the two fields this ticket exists for.
    assert payload["skipped_pages"]
    assert payload["flag_log"]
    errors = sorted(validator.iter_errors(payload), key=str)
    assert not errors, "\n".join(e.message for e in errors)

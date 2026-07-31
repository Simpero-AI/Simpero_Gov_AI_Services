"""parser_service.extract_service.extract_claims -- the shared extraction
entry point scripts/emit_claims.py and POST /extract both call (SIM-340/E5).

The acceptance bar from the ticket: the CLI and the callable entry point must
be provably the same code path, and the callable form must return a payload
identical to what the CLI would print for the same input.
"""

from __future__ import annotations

import json

import pytest

from parser_service import extract_service
from parser_service.emit import ClaimValue, PdfLocation
from scripts import emit_claims


class _Doc:
    pass


class _Result:
    document = _Doc()
    pages: list = []
    sha256 = "0" * 64


class _Page:
    def __init__(self, page: int) -> None:
        self.page = page


def _table_claim(page: int, attribute: str = "revenue") -> extract_service.Claim:
    return extract_service.Claim(
        entity="ACME",
        attribute=attribute,
        value=ClaimValue(raw="$1", normalized=1.0, unit="$", value_type="currency"),
        location=PdfLocation(file="cim.pdf", page=page, char_start=0, char_end=2),
        status="proposed",
    )


def test_prose_without_a_key_raises_before_any_parsing(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)

    def _must_not_run(*_a, **_k):
        raise AssertionError("parse must not start when the key is absent")

    monkeypatch.setattr(extract_service, "parse_pdf_bytes", _must_not_run)

    with pytest.raises(extract_service.ProseCredentialMissing):
        extract_service.extract_claims(
            b"%PDF-1.4 stub",
            entity="ACME",
            run_id="run-1",
            correlation_id="doc-1",
            source_file="cim.pdf",
            prose=True,
        )


def test_qualitative_implies_prose_for_the_credential_check(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.setattr(
        extract_service, "parse_pdf_bytes", lambda *_a, **_k: AssertionError("unreachable")
    )

    with pytest.raises(extract_service.ProseCredentialMissing):
        extract_service.extract_claims(
            b"%PDF-1.4 stub",
            entity="ACME",
            run_id="run-1",
            correlation_id="doc-1",
            source_file="cim.pdf",
            qualitative=True,
        )


def test_table_only_tier_runs_without_touching_the_credential_check(monkeypatch) -> None:
    def _fake_key_check() -> bool:  # pragma: no cover - must not be called
        raise AssertionError("the table tier must not consult the credential")

    monkeypatch.setattr(extract_service, "api_key_present", _fake_key_check)
    monkeypatch.setattr(extract_service, "parse_pdf_bytes", lambda _b: _Result())
    monkeypatch.setattr(extract_service, "extract_tables", lambda *_a, **_k: [])

    payload = extract_service.extract_claims(
        b"%PDF-1.4 stub",
        entity="ACME",
        run_id="run-1",
        correlation_id="doc-1",
        source_file="cim.pdf",
    )
    assert payload["claims"] == []


def test_correlation_id_is_not_written_into_the_returned_payload(monkeypatch) -> None:
    # correlation_id/run_id identify the caller's run for logging/correlation --
    # the C3 claims contract has no slot for either, so only run_id (which the
    # contract's payload shape already carries) should appear. Named
    # correlation_id, not document_id, because that term already means the
    # content hash elsewhere (emit_chunks sets document_id = sha256).
    monkeypatch.setattr(extract_service, "parse_pdf_bytes", lambda _b: _Result())
    monkeypatch.setattr(extract_service, "extract_tables", lambda *_a, **_k: [])

    payload = extract_service.extract_claims(
        b"%PDF-1.4 stub",
        entity="ACME",
        run_id="run-1",
        correlation_id="a-correlation-id-that-must-not-leak",
        source_file="cim.pdf",
    )
    assert "correlation_id" not in payload
    assert "document_id" not in payload
    assert "a-correlation-id-that-must-not-leak" not in json.dumps(payload)


def test_source_file_stays_none_when_the_caller_has_none(monkeypatch) -> None:
    # A caller with no real filename must not fall back to correlation_id --
    # that value is a correlation token, not a filename, and stamping it into
    # every claim's debug `file` field would misrepresent it as one.
    monkeypatch.setattr(extract_service, "parse_pdf_bytes", lambda _b: _Result())
    monkeypatch.setattr(extract_service, "extract_tables", lambda *_a, **_k: [])

    payload = extract_service.extract_claims(
        b"%PDF-1.4 stub",
        entity="ACME",
        run_id="run-1",
        correlation_id="run-token-not-a-filename",
    )
    assert payload["source_file"] is None
    assert "run-token-not-a-filename" not in json.dumps(payload)


def test_a_bad_table_does_not_abort_the_document_and_is_reported_skipped(monkeypatch) -> None:
    # One table's failure must not take its sibling tables, other pages, or the
    # whole document down -- and the affected page is still named in
    # skipped_pages so an HTTP caller (no stderr to read) can see it.
    class _MultiPageResult:
        document = _Doc()
        pages = [_Page(1), _Page(2)]
        sha256 = "0" * 64

    def _fake_tables_on_page(_tables, page_no: int) -> list[str]:
        return {1: ["t1-bad", "t1-good"], 2: ["t2-good"]}[page_no]

    def _fake_claims_from_table(table, page, *, entity, file, flag_log):
        if table == "t1-bad":
            raise ValueError("a malformed table")
        return [_table_claim(page.page, attribute=table)]

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
    assert payload["skipped_pages"] == [
        {"page": 1, "tier": "tables", "reason": "ValueError: a malformed table"}
    ]
    attributes = {c["attribute"] for c in payload["claims"]}
    # t1-good survives its sibling t1-bad's failure; t2-good is untouched.
    assert attributes == {"t1-good", "t2-good"}


def test_skipped_pages_is_empty_when_nothing_failed(monkeypatch) -> None:
    monkeypatch.setattr(extract_service, "parse_pdf_bytes", lambda _b: _Result())
    monkeypatch.setattr(extract_service, "extract_tables", lambda *_a, **_k: [])

    payload = extract_service.extract_claims(
        b"%PDF-1.4 stub",
        entity="ACME",
        run_id="run-1",
        correlation_id="doc-1",
        source_file="cim.pdf",
    )
    assert payload["skipped_pages"] == []


def test_canonicalize_attributes_without_a_key_raises_before_any_parsing(monkeypatch) -> None:
    # SIM-344's pass calls the Anthropic API just as the prose tiers do, and it
    # is independent of them -- so it must fail closed at the door the same way
    # even when prose/complete/qualitative are all off.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)

    def _must_not_run(*_a, **_k):
        raise AssertionError("parse must not start when the key is absent")

    monkeypatch.setattr(extract_service, "parse_pdf_bytes", _must_not_run)

    with pytest.raises(extract_service.ProseCredentialMissing):
        extract_service.extract_claims(
            b"%PDF-1.4 stub",
            entity="ACME",
            run_id="run-1",
            correlation_id="doc-1",
            source_file="cim.pdf",
            canonicalize_attributes=True,
        )


def test_canonicalize_attributes_maps_table_claims_end_to_end(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    class _OnePageResult:
        document = _Doc()
        pages = [_Page(1)]
        sha256 = "0" * 64

    monkeypatch.setattr(extract_service, "parse_pdf_bytes", lambda _b: _OnePageResult())
    monkeypatch.setattr(extract_service, "extract_tables", lambda *_a, **_k: [])
    monkeypatch.setattr(extract_service, "tables_on_page", lambda *_a, **_k: ["t1"])
    monkeypatch.setattr(
        extract_service,
        "claims_from_table",
        lambda table, page, *, entity, file, flag_log: [
            _table_claim(page.page, attribute="Revenue | 2019F")
        ],
    )
    monkeypatch.setattr(
        extract_service,
        "canonicalize_attributes",
        lambda labels: {"Revenue | 2019F": ("revenue", [])},
    )

    payload = extract_service.extract_claims(
        b"%PDF-1.4 stub",
        entity="ACME",
        run_id="run-1",
        correlation_id="doc-1",
        source_file="cim.pdf",
        canonicalize_attributes=True,
    )

    assert len(payload["claims"]) == 1
    claim = payload["claims"][0]
    assert claim["attribute"] == "revenue"
    assert claim["attribute_raw"] == "Revenue | 2019F"


def test_canonicalize_attributes_defaults_to_off(monkeypatch) -> None:
    # Table-only extraction stays credential-free unless the caller opts in --
    # this is the regression this ticket must not introduce.
    def _fake_key_check() -> bool:  # pragma: no cover - must not be called
        raise AssertionError("table-only extraction must not consult the credential")

    monkeypatch.setattr(extract_service, "api_key_present", _fake_key_check)
    monkeypatch.setattr(extract_service, "parse_pdf_bytes", lambda _b: _Result())
    monkeypatch.setattr(extract_service, "extract_tables", lambda *_a, **_k: [])

    payload = extract_service.extract_claims(
        b"%PDF-1.4 stub",
        entity="ACME",
        run_id="run-1",
        correlation_id="doc-1",
        source_file="cim.pdf",
    )
    assert payload["claims"] == []


def test_cli_and_direct_call_produce_an_identical_payload_for_the_same_input(
    monkeypatch, tmp_path, capsys
) -> None:
    # The ticket's core acceptance criterion: emit_claims.py and the shared
    # entry point share one code path, so a caller of either gets the same
    # claims payload for the same document/entity/run_id.
    monkeypatch.setattr(extract_service, "parse_pdf_bytes", lambda _b: _Result())
    monkeypatch.setattr(extract_service, "extract_tables", lambda *_a, **_k: [])

    pdf = tmp_path / "cim.pdf"
    pdf.write_bytes(b"%PDF-1.4 stub")

    emit_claims.main([str(pdf), "--entity", "ACME", "--run-id", "run-shared"])
    cli_payload = json.loads(capsys.readouterr().out)

    direct_payload = extract_service.extract_claims(
        pdf.read_bytes(),
        entity="ACME",
        run_id="run-shared",
        correlation_id="cim",
        source_file="cim.pdf",
    )

    assert cli_payload == direct_payload

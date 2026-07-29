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
from scripts import emit_claims


class _Doc:
    pass


class _Result:
    document = _Doc()
    pages: list = []
    sha256 = "0" * 64


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
            document_id="doc-1",
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
            document_id="doc-1",
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
        document_id="doc-1",
        source_file="cim.pdf",
    )
    assert payload["claims"] == []


def test_document_id_is_not_written_into_the_returned_payload(monkeypatch) -> None:
    # document_id/run_id identify the caller's run for logging/correlation --
    # the C3 claims contract has no slot for either, so only run_id (which the
    # contract's payload shape already carries) should appear.
    monkeypatch.setattr(extract_service, "parse_pdf_bytes", lambda _b: _Result())
    monkeypatch.setattr(extract_service, "extract_tables", lambda *_a, **_k: [])

    payload = extract_service.extract_claims(
        b"%PDF-1.4 stub",
        entity="ACME",
        run_id="run-1",
        document_id="a-document-id-that-must-not-leak",
        source_file="cim.pdf",
    )
    assert "document_id" not in payload
    assert "a-document-id-that-must-not-leak" not in json.dumps(payload)


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
        document_id="cim",
        source_file="cim.pdf",
    )

    assert cli_payload == direct_payload

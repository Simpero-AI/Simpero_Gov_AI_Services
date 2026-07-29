"""The emit_claims CLI: the committed entry point for the parse-to-claims seam.

scripts/emit_claims.py is now a thin argparse wrapper over
parser_service.extract_service.extract_claims (SIM-340/E5) -- that shared
function is where parsing actually happens, so these tests patch it there,
not on the CLI module. The behaviour under test is still the CLI's own: the
prose tiers must refuse to start without a credential rather than failing
part way through a document, and the table tier needs neither key nor network.
"""

from __future__ import annotations

import json

import pytest

from parser_service import extract_service
from scripts import emit_claims


def test_prose_without_a_key_fails_closed_before_any_work(monkeypatch, tmp_path) -> None:
    # A missing credential must stop the run at the door, not after a parse and
    # not part way through a document. It surfaces as a SystemExit (the CLI's
    # argparse.error translation of ProseCredentialMissing), and it must fire
    # before parse_pdf_bytes is ever called.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)

    called = False

    def _must_not_run(*_a, **_k):
        nonlocal called
        called = True
        raise AssertionError("parse must not start when the key is absent")

    monkeypatch.setattr(extract_service, "parse_pdf_bytes", _must_not_run)
    pdf = tmp_path / "cim.pdf"
    pdf.write_bytes(b"%PDF-1.4 stub")

    with pytest.raises(SystemExit):
        emit_claims.main([str(pdf), "--entity", "ACME", "--prose"])
    assert called is False


def test_qualitative_implies_prose_and_also_needs_a_key(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.setattr(
        extract_service, "parse_pdf_bytes", lambda *_a, **_k: AssertionError("unreachable")
    )
    pdf = tmp_path / "cim.pdf"
    pdf.write_bytes(b"%PDF-1.4 stub")

    with pytest.raises(SystemExit):
        emit_claims.main([str(pdf), "--entity", "ACME", "--qualitative"])


def test_tables_only_needs_no_key(monkeypatch, tmp_path) -> None:
    # The default tier calls no model and must run with the environment stripped.
    # A present key must not be consulted for a table-only run, so the guard is
    # never reached -- proven by leaving parse stubbed and asserting it ran.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)

    ran = False

    def _fake_key_check() -> bool:  # pragma: no cover - must not be called
        raise AssertionError("the table tier must not consult the credential")

    monkeypatch.setattr(extract_service, "api_key_present", _fake_key_check)

    class _Doc:
        pass

    class _Result:
        document = _Doc()
        pages: list = []
        sha256 = "0" * 64

    def _fake_parse(_bytes):
        nonlocal ran
        ran = True
        return _Result()

    monkeypatch.setattr(extract_service, "parse_pdf_bytes", _fake_parse)
    monkeypatch.setattr(extract_service, "extract_tables", lambda *_a, **_k: [])
    pdf = tmp_path / "cim.pdf"
    pdf.write_bytes(b"%PDF-1.4 stub")

    emit_claims.main([str(pdf), "--entity", "ACME"])
    assert ran is True


def test_cli_stdout_is_pure_json_with_no_stderr_chatter_mixed_in(
    monkeypatch, tmp_path, capsys
) -> None:
    # json.dump goes to stdout; every diagnostic line is on stderr (see
    # extract_service.extract_claims) -- a downstream consumer piping stdout
    # into a JSON parser must never see a stray print land in the payload.
    class _Doc:
        pass

    class _Result:
        document = _Doc()
        pages: list = []
        sha256 = "0" * 64

    monkeypatch.setattr(extract_service, "parse_pdf_bytes", lambda _b: _Result())
    monkeypatch.setattr(extract_service, "extract_tables", lambda *_a, **_k: [])
    pdf = tmp_path / "cim.pdf"
    pdf.write_bytes(b"%PDF-1.4 stub")

    emit_claims.main([str(pdf), "--entity", "ACME", "--run-id", "run-xyz"])
    captured = capsys.readouterr()

    payload = json.loads(captured.out)
    assert payload["run_id"] == "run-xyz"
    assert payload["source_file"] == "cim.pdf"
    assert payload["claims"] == []
    assert "tier tables" in captured.err

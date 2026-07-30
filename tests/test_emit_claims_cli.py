"""The emit_claims CLI: the committed entry point for the parse-to-claims seam.

The behaviour under test is the tier gating, not the model -- above all that the
prose tiers refuse to start without a credential rather than failing part way
through a document, and that the table tier needs neither key nor network.
"""

from __future__ import annotations

import json

import pytest

from scripts import emit_claims


def test_prose_without_a_key_fails_closed_before_any_work(monkeypatch, tmp_path) -> None:
    # A missing credential must stop the run at the door, not after a parse and
    # not part way through a document. It is a SystemExit (argparse error), and
    # it must fire before parse_pdf_bytes is ever called.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)

    called = False

    def _must_not_run(*_a, **_k):
        nonlocal called
        called = True
        raise AssertionError("parse must not start when the key is absent")

    monkeypatch.setattr(emit_claims, "parse_pdf_bytes", _must_not_run)
    pdf = tmp_path / "cim.pdf"
    pdf.write_bytes(b"%PDF-1.4 stub")

    with pytest.raises(SystemExit):
        emit_claims.main([str(pdf), "--entity", "ACME", "--prose"])
    assert called is False


def test_qualitative_implies_prose_and_also_needs_a_key(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.setattr(
        emit_claims, "parse_pdf_bytes", lambda *_a, **_k: AssertionError("unreachable")
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

    monkeypatch.setattr(emit_claims, "api_key_present", _fake_key_check)

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

    monkeypatch.setattr(emit_claims, "parse_pdf_bytes", _fake_parse)
    monkeypatch.setattr(emit_claims, "extract_tables", lambda *_a, **_k: [])
    pdf = tmp_path / "cim.pdf"
    pdf.write_bytes(b"%PDF-1.4 stub")

    emit_claims.main([str(pdf), "--entity", "ACME"])
    assert ran is True


def test_a_failed_prose_page_is_recorded_and_the_run_still_succeeds(
    monkeypatch, tmp_path, capsys
) -> None:
    # A page whose model call raises must not lose the run: the good page's
    # claim still comes out, and the bad page is named in the payload's
    # `skipped_pages` -- not just printed to stderr and forgotten.
    monkeypatch.setattr(emit_claims, "api_key_present", lambda: True)

    class _Doc:
        pass

    class _Page:
        def __init__(self, page: int) -> None:
            self.page = page

    class _Result:
        document = _Doc()
        pages = [_Page(1), _Page(2)]
        sha256 = "0" * 64

    class _FakeClaim:
        status = "proposed"

        def to_json(self) -> dict:
            return {"entity": "ACME", "status": "proposed"}

    def _fake_claims_from_prose(_blocks, page, **_kwargs):
        if page.page == 2:
            raise ValueError("model call failed")
        return [_FakeClaim()]

    monkeypatch.setattr(emit_claims, "parse_pdf_bytes", lambda _bytes: _Result())
    monkeypatch.setattr(emit_claims, "extract_tables", lambda *_a, **_k: [])
    monkeypatch.setattr(emit_claims, "extract_text_blocks", lambda *_a, **_k: [])
    monkeypatch.setattr(emit_claims, "blocks_on_page", lambda *_a, **_k: [])
    monkeypatch.setattr(emit_claims, "prose_text", lambda *_a, **_k: "some prose")
    monkeypatch.setattr(emit_claims, "claims_from_prose", _fake_claims_from_prose)

    pdf = tmp_path / "cim.pdf"
    pdf.write_bytes(b"%PDF-1.4 stub")

    emit_claims.main([str(pdf), "--entity", "ACME", "--prose", "--workers", "1"])

    payload = json.loads(capsys.readouterr().out.split("\n\nemitted", 1)[0])
    assert len(payload["claims"]) == 1
    assert payload["skipped_pages"] == [
        {"page": 2, "tier": "prose", "reason": "ValueError: model call failed"}
    ]

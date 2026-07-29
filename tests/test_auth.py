"""POST /parse's X-Parser-Key enforcement (SIM: parser-droplet-deployment §1.1/§1.2).

Fail-closed, not "enforce only when the secret happens to be set": unset key ->
503, absent/wrong header -> 401, correct header -> 200. GET /health is untouched
by this and stays open (exercised implicitly by every other test module, which
never sends a key to it).
"""

from io import BytesIO

import pytest
from conftest import TEST_PARSER_API_KEY
from fastapi.testclient import TestClient
from reportlab.pdfgen import canvas

from parser_service import config
from parser_service.main import app


def _minimal_pdf_bytes() -> bytes:
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer)
    pdf.drawString(72, 720, "auth test")
    pdf.showPage()
    pdf.save()
    return buffer.getvalue()


def test_parse_without_header_is_rejected() -> None:
    # No client-level default header here -- this is the "someone hits the
    # endpoint with nothing at all" case.
    client = TestClient(app)
    response = client.post("/parse", content=_minimal_pdf_bytes())
    assert response.status_code == 401


def test_parse_with_wrong_header_is_rejected() -> None:
    client = TestClient(app, headers={"X-Parser-Key": "not-the-right-key"})
    response = client.post("/parse", content=_minimal_pdf_bytes())
    assert response.status_code == 401


def test_parse_with_correct_header_succeeds() -> None:
    client = TestClient(app, headers={"X-Parser-Key": TEST_PARSER_API_KEY})
    response = client.post("/parse", content=_minimal_pdf_bytes())
    assert response.status_code == 200


def test_parse_with_unset_api_key_is_misconfigured(monkeypatch: pytest.MonkeyPatch) -> None:
    # The autouse conftest fixture always sets PARSER_API_KEY; this is the one
    # test that needs it genuinely unset, so it undoes that fixture locally.
    monkeypatch.delenv("PARSER_API_KEY", raising=False)
    config.get_settings.cache_clear()
    try:
        client = TestClient(app, headers={"X-Parser-Key": TEST_PARSER_API_KEY})
        response = client.post("/parse", content=_minimal_pdf_bytes())
        assert response.status_code == 503
    finally:
        config.get_settings.cache_clear()

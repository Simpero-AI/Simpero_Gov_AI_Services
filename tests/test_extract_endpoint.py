"""POST /extract -- the HTTP form of scripts/emit_claims.py (SIM-340/E5).

Shares parser_service.extract_service.extract_claims with the CLI (see
tests/test_extract_service.py for the CLI/endpoint payload-equivalence proof)
and the same X-Parser-Key fail-closed dependency as POST /parse (see
tests/test_auth.py) -- these tests cover what's specific to this route: its
own required headers, and how it maps extract_claims' failure modes to HTTP.
"""

from __future__ import annotations

from io import BytesIO

from conftest import TEST_PARSER_API_KEY
from fastapi.testclient import TestClient
from reportlab.pdfgen import canvas

from parser_service import extract_service
from parser_service.main import app


def _minimal_pdf_bytes() -> bytes:
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer)
    pdf.drawString(72, 720, "Revenue for ACME Corp was $15,000 in 2024.")
    pdf.showPage()
    pdf.save()
    return buffer.getvalue()


def _headers(**overrides: str) -> dict[str, str]:
    headers = {
        "X-Parser-Key": TEST_PARSER_API_KEY,
        "X-Run-Id": "run-1",
        "X-Document-Id": "doc-1",
        "X-Entity": "ACME Corp",
    }
    headers.update(overrides)
    return headers


def test_extract_without_parser_key_is_rejected() -> None:
    # Same fail-closed dependency as POST /parse -- proves it's actually wired
    # onto this route, not just imported.
    client = TestClient(app)
    headers = _headers()
    del headers["X-Parser-Key"]
    response = client.post("/extract", content=_minimal_pdf_bytes(), headers=headers)
    assert response.status_code == 401


def test_extract_without_required_headers_is_a_validation_error() -> None:
    client = TestClient(app)
    response = client.post(
        "/extract",
        content=_minimal_pdf_bytes(),
        headers={"X-Parser-Key": TEST_PARSER_API_KEY},
    )
    assert response.status_code == 422


def test_extract_table_tier_succeeds_and_returns_the_c3_payload_shape() -> None:
    client = TestClient(app)
    response = client.post("/extract", content=_minimal_pdf_bytes(), headers=_headers())

    assert response.status_code == 200
    body = response.json()
    assert body["run_id"] == "run-1"
    assert body["source_file"] == "doc-1"
    assert isinstance(body["claims"], list)
    assert isinstance(body["flag_log"], list)
    # document_id is caller-side correlation only -- not part of the C3 contract.
    assert "document_id" not in body


def test_extract_honors_an_explicit_source_file_header() -> None:
    client = TestClient(app)
    response = client.post(
        "/extract",
        content=_minimal_pdf_bytes(),
        headers=_headers(**{"X-Source-File": "cim.pdf"}),
    )
    assert response.status_code == 200
    assert response.json()["source_file"] == "cim.pdf"


def test_extract_prose_without_a_credential_is_a_503(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)

    def _must_not_run(*_a, **_k):
        raise AssertionError("parse must not start when the key is absent")

    monkeypatch.setattr(extract_service, "parse_pdf_bytes", _must_not_run)

    client = TestClient(app)
    response = client.post(
        "/extract",
        content=_minimal_pdf_bytes(),
        headers=_headers(**{"X-Prose": "true"}),
    )
    assert response.status_code == 503


def test_extract_qualitative_header_implies_prose_for_the_credential_check(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.setattr(
        extract_service, "parse_pdf_bytes", lambda *_a, **_k: AssertionError("unreachable")
    )

    client = TestClient(app)
    response = client.post(
        "/extract",
        content=_minimal_pdf_bytes(),
        headers=_headers(**{"X-Qualitative": "true"}),
    )
    assert response.status_code == 503


def test_extract_zero_byte_body_maps_parse_error_to_its_own_status_code() -> None:
    # Reuses docling_parser's zero_byte_pdf rejection via ParseError, same case
    # test_pdf_parser.py exercises for POST /parse.
    client = TestClient(app)
    response = client.post("/extract", content=b"", headers=_headers())
    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "zero_byte_pdf"

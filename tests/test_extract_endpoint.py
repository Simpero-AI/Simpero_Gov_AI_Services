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

from parser_service import extract_service, main
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
        "X-Correlation-Id": "doc-1",
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
    # No X-Source-File was sent, and the correlation id must not be guessed in
    # as a stand-in filename -- it is a correlation token, not a filename.
    assert body["source_file"] is None
    assert isinstance(body["claims"], list)
    assert isinstance(body["flag_log"], list)
    assert isinstance(body["skipped_pages"], list)
    # correlation_id is caller-side correlation only -- not part of the C3 contract.
    assert "correlation_id" not in body
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


def test_undo_header_mojibake_recovers_a_utf8_entity_sent_as_latin1() -> None:
    # ASGI headers are latin-1 by spec; a client sending a UTF-8 encoded
    # company name round-trips through Starlette as mojibake. Re-encoding as
    # latin-1 and decoding as UTF-8 recovers the original string.
    original = "Café Corp"
    mojibake = original.encode("utf-8").decode("latin-1")
    assert main._undo_header_mojibake(mojibake) == original


def test_undo_header_mojibake_is_a_noop_for_ascii() -> None:
    assert main._undo_header_mojibake("ACME Corp") == "ACME Corp"


def test_undo_header_mojibake_leaves_a_non_utf8_value_unchanged() -> None:
    # A lone latin-1 character (0xE9) is not a valid standalone UTF-8 byte --
    # left exactly as given rather than raising.
    value = "é"
    assert main._undo_header_mojibake(value) == value


def test_extract_accepts_a_non_ascii_entity_header() -> None:
    # X-Entity carries a real company name, which is not guaranteed ASCII. httpx
    # itself refuses a non-ASCII str header value, so the raw UTF-8 bytes a real
    # client would put on the wire are passed directly -- exactly what Starlette
    # decodes as latin-1 mojibake on the way in, and _undo_header_mojibake exists
    # to reverse.
    client = TestClient(app)
    headers: dict[str, str | bytes] = dict(_headers())
    headers["X-Entity"] = "Café Corp".encode()
    response = client.post("/extract", content=_minimal_pdf_bytes(), headers=headers)
    assert response.status_code == 200

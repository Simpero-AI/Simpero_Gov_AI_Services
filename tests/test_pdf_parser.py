from hashlib import sha256
from io import BytesIO

from fastapi.testclient import TestClient
from pypdf import PdfWriter
from reportlab.pdfgen import canvas

from services.parser.parser_service.main import app
from services.parser.parser_service.pdf_parser import parse_pdf_bytes


client = TestClient(app)


def make_text_pdf(lines: list[str]) -> bytes:
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer)
    y = 760
    for line in lines:
        pdf.drawString(72, y, line)
        y -= 24
    pdf.save()
    return buffer.getvalue()


def make_blank_pdf(page_count: int = 1) -> bytes:
    writer = PdfWriter()
    for _ in range(page_count):
        writer.add_blank_page(width=612, height=792)
    buffer = BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def test_parse_pdf_returns_page_paragraph_and_char_offsets() -> None:
    pdf_bytes = make_text_pdf(
        [
            "Revenue increased to $10 million.",
            "EBITDA margin was 25%.",
        ]
    )

    result = parse_pdf_bytes(pdf_bytes)

    assert result.sha256 == sha256(pdf_bytes).hexdigest()
    assert len(result.chunks) == 2
    assert result.chunks[0].page == 1
    assert result.chunks[0].paragraph == 1
    assert result.chunks[0].text == "Revenue increased to $10 million."
    assert result.chunks[0].char_start == 0
    assert result.chunks[0].char_end == len(result.chunks[0].text)
    assert result.chunks[1].char_start > result.chunks[0].char_end


def test_parse_endpoint_accepts_pdf_bytes() -> None:
    pdf_bytes = make_text_pdf(["The company serves enterprise customers."])

    response = client.post(
        "/parse",
        content=pdf_bytes,
        headers={"Content-Type": "application/pdf"},
    )

    assert response.status_code == 200
    assert response.headers["X-Content-SHA256"] == sha256(pdf_bytes).hexdigest()
    assert response.json()[0]["text"] == "The company serves enterprise customers."


def test_zero_byte_pdf_is_rejected() -> None:
    response = client.post(
        "/parse",
        content=b"",
        headers={"Content-Type": "application/pdf"},
    )

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "zero_byte_pdf"


def test_corrupt_pdf_is_rejected() -> None:
    response = client.post(
        "/parse",
        content=b"not a pdf",
        headers={"Content-Type": "application/pdf"},
    )

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "corrupt_pdf"


def test_duplicate_pdf_is_rejected_by_known_sha_header() -> None:
    pdf_bytes = make_text_pdf(["Duplicate source document."])
    digest = sha256(pdf_bytes).hexdigest()

    response = client.post(
        "/parse",
        content=pdf_bytes,
        headers={
            "Content-Type": "application/pdf",
            "X-Known-SHA256": digest,
        },
    )

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "duplicate_pdf"


def test_page_limit_guard_rejects_over_110_pages() -> None:
    response = client.post(
        "/parse",
        content=make_blank_pdf(page_count=111),
        headers={"Content-Type": "application/pdf"},
    )

    assert response.status_code == 413
    assert response.json()["detail"]["code"] == "pdf_too_large"


def test_blank_pdf_without_text_is_rejected() -> None:
    response = client.post(
        "/parse",
        content=make_blank_pdf(),
        headers={"Content-Type": "application/pdf"},
    )

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "no_extractable_text"

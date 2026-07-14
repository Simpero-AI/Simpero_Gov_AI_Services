import shutil
from hashlib import sha256
from io import BytesIO
from pathlib import Path

from fastapi.testclient import TestClient
from pypdf import PdfWriter
from reportlab.pdfgen import canvas

from services.parser.parser_service.docling_parser import parse_pdf_bytes
from services.parser.parser_service.main import app

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


def copy_test_pdf_if_needed() -> Path | None:
    dest_dir = Path(__file__).parent.parent / "test_data"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / "1st-App-H-PTL-Group-CIM.pdf"

    if not dest_path.exists():
        src_path = Path("p:/simpero_GOV_AI/scripts/examples/1st-app-h-ptl/1st-App-H-PTL-Group-CIM.pdf")
        if src_path.exists():
            shutil.copy(src_path, dest_path)

    return dest_path if dest_path.exists() else None


def test_parse_pdf_returns_page_index_and_char_coordinates() -> None:
    pdf_bytes = make_text_pdf(
        [
            "Revenue increased to $10 million.",
            "EBITDA margin was 25%.",
        ]
    )

    result = parse_pdf_bytes(pdf_bytes)

    assert result.sha256 == sha256(pdf_bytes).hexdigest()
    assert len(result.pages) == 1
    page = result.pages[0]

    assert page.page == 1
    assert "Revenue increased to $10 million." in page.text
    assert "EBITDA margin was 25%." in page.text

    # Assert hard invariants
    assert len(page.text) == len(page.char_map)
    for i, char in enumerate(page.text):
        assert char == page.char_map[i].char
        # Bbox coordinates should be valid floats
        assert isinstance(page.char_map[i].x0, float)
        assert isinstance(page.char_map[i].top, float)
        assert isinstance(page.char_map[i].x1, float)
        assert isinstance(page.char_map[i].bottom, float)


def test_parse_endpoint_accepts_pdf_bytes() -> None:
    pdf_bytes = make_text_pdf(["The company serves enterprise customers."])

    response = client.post(
        "/parse",
        content=pdf_bytes,
        headers={"Content-Type": "application/pdf"},
    )

    assert response.status_code == 200
    assert response.headers["X-Content-SHA256"] == sha256(pdf_bytes).hexdigest()

    pages = response.json()
    assert len(pages) == 1
    assert "The company serves enterprise customers." in pages[0]["text"]
    assert len(pages[0]["text"]) == len(pages[0]["char_map"])


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
    # A completely blank PDF generated without any text cells
    response = client.post(
        "/parse",
        content=make_blank_pdf(page_count=1),
        headers={"Content-Type": "application/pdf"},
    )

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "no_extractable_text"


def test_cim_pdf_acceptance_and_normalization() -> None:
    pdf_path = copy_test_pdf_if_needed()
    if not pdf_path or not pdf_path.exists():
        # Skip if CIM PDF not found in local workspace or sibling repo
        return

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    result = parse_pdf_bytes(pdf_bytes)
    digest = sha256(pdf_bytes).hexdigest()

    # 1st-App-H-PTL-Group-CIM.pdf is 19 pages
    assert len(result.pages) == 19

    # Assert cache file is created
    cache_file = Path("services/parser/cache") / f"{digest}.json"
    assert cache_file.exists()

    # Hard invariant test across all pages
    for page in result.pages:
        assert len(page.text) == len(page.char_map)
        for i, char in enumerate(page.text):
            assert char == page.char_map[i].char

    # PDF Page index 11 (0-indexed page 10) is the income statement page.
    # In 1st-App-H-PTL-Group-CIM.pdf, page index 11 has the value 3,817 which originally renders as "3 ,817"
    page_11 = result.pages[10]

    # Verify the space has been collapsed and normalized
    assert "3,817" in page_11.text
    assert "3 ,817" not in page_11.text

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import re

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from .schemas import ParsedParagraph

MAX_PAGES = 110
MIN_TEXT_CHARS_FOR_TEXT_PDF = 20


class ParseError(Exception):
    def __init__(self, code: str, message: str, status_code: int = 422) -> None:
        self.code = code
        self.message = message
        self.status_code = status_code
        super().__init__(message)


@dataclass(frozen=True)
class ParseResult:
    sha256: str
    chunks: list[ParsedParagraph]


def parse_known_hashes(value: str | None) -> set[str]:
    if not value:
        return set()
    return {item.strip().lower() for item in value.split(",") if item.strip()}


def parse_pdf_bytes(pdf_bytes: bytes, known_sha256s: set[str] | None = None) -> ParseResult:
    if len(pdf_bytes) == 0:
        raise ParseError("zero_byte_pdf", "Uploaded PDF is empty.", 400)

    digest = sha256(pdf_bytes).hexdigest()
    if digest in (known_sha256s or set()):
        raise ParseError(
            "duplicate_pdf",
            "Uploaded PDF matches an existing data source SHA-256 hash.",
            409,
        )

    try:
        reader = PdfReader(BytesIO(pdf_bytes))
    except (PdfReadError, ValueError, OSError) as exc:
        raise ParseError("corrupt_pdf", "Uploaded file is not a readable PDF.", 400) from exc

    if reader.is_encrypted:
        raise ParseError("encrypted_pdf", "Encrypted PDFs are not supported.", 422)

    page_count = len(reader.pages)
    if page_count > MAX_PAGES:
        raise ParseError(
            "pdf_too_large",
            f"PDF has {page_count} pages; maximum allowed is {MAX_PAGES}.",
            413,
        )

    page_texts: list[str] = []
    image_pages = 0
    for page in reader.pages:
        text = page.extract_text() or ""
        page_texts.append(text)
        if _page_has_images(page):
            image_pages += 1

    total_text = "".join(page_texts).strip()
    if len(total_text) < MIN_TEXT_CHARS_FOR_TEXT_PDF:
        if image_pages > 0:
            raise ParseError(
                "scanned_image_pdf",
                "PDF appears to be image-only or scanned; OCR is required before parsing.",
                422,
            )
        raise ParseError("no_extractable_text", "PDF contains no extractable text.", 422)

    chunks = _chunk_pages(page_texts)
    if not chunks:
        raise ParseError("no_extractable_text", "PDF contains no extractable paragraphs.", 422)

    return ParseResult(sha256=digest, chunks=chunks)


def _chunk_pages(page_texts: list[str]) -> list[ParsedParagraph]:
    chunks: list[ParsedParagraph] = []
    cursor = 0

    for page_index, page_text in enumerate(page_texts, start=1):
        normalized_text = _normalize_text(page_text)
        paragraphs = _split_paragraphs(normalized_text)
        search_from = 0

        for paragraph_index, paragraph in enumerate(paragraphs, start=1):
            local_start = normalized_text.find(paragraph, search_from)
            if local_start < 0:
                local_start = search_from
            local_end = local_start + len(paragraph)
            search_from = local_end

            chunks.append(
                ParsedParagraph(
                    page=page_index,
                    paragraph=paragraph_index,
                    text=paragraph,
                    char_start=cursor + local_start,
                    char_end=cursor + local_end,
                )
            )

        cursor += len(normalized_text) + 1

    return chunks


def _normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


def _split_paragraphs(text: str) -> list[str]:
    if not text:
        return []

    blank_line_groups = [part.strip() for part in re.split(r"\n{2,}", text) if part.strip()]
    if len(blank_line_groups) > 1:
        return [re.sub(r"\s+", " ", part) for part in blank_line_groups]

    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if not lines:
        return []

    paragraphs: list[str] = []
    current: list[str] = []
    for line in lines:
        current.append(line)
        if line.endswith((".", "?", "!", ":")):
            paragraphs.append(" ".join(current))
            current = []

    if current:
        paragraphs.append(" ".join(current))

    return paragraphs


def _page_has_images(page: object) -> bool:
    resources = getattr(page, "get", lambda _key, _default=None: None)("/Resources", {})
    if not resources:
        return False

    xobjects = resources.get("/XObject") if hasattr(resources, "get") else None
    if not xobjects:
        return False

    try:
        xobjects = xobjects.get_object()
    except AttributeError:
        pass

    for value in xobjects.values():
        try:
            obj = value.get_object()
        except AttributeError:
            obj = value
        if hasattr(obj, "get") and obj.get("/Subtype") == "/Image":
            return True

    return False

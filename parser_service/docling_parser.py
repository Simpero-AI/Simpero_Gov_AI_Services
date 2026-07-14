import contextlib
import re
import tempfile
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc.page import TextCellUnit
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from .schemas import CharBox, PageIndex

MAX_PAGES = 110


class ParseError(Exception):
    def __init__(self, code: str, message: str, status_code: int = 422) -> None:
        self.code = code
        self.message = message
        self.status_code = status_code
        super().__init__(message)


@dataclass(frozen=True)
class DoclingParseResult:
    sha256: str
    pages: list[PageIndex]


def parse_known_hashes(value: str | None) -> set[str]:
    if not value:
        return set()
    return {item.strip().lower() for item in value.split(",") if item.strip()}


def normalize_numeric_tokens(text: str, char_map: list[CharBox]) -> tuple[str, list[CharBox]]:
    new_text_list = []
    new_char_map = []
    i = 0
    n = len(text)
    pattern = re.compile(r"^(\d)\s+([.,])\s*(\d)")

    while i < n:
        match = pattern.match(text[i:])
        if match:
            sep = match.group(2)
            g0 = match.group(0)

            idx_d1 = i
            idx_sep = i + g0.find(sep)
            idx_d2 = i + len(g0) - 1

            new_text_list.append(text[idx_d1])
            new_char_map.append(char_map[idx_d1])

            new_text_list.append(text[idx_sep])
            new_char_map.append(char_map[idx_sep])

            new_text_list.append(text[idx_d2])
            new_char_map.append(char_map[idx_d2])

            i += len(g0)
        else:
            new_text_list.append(text[i])
            new_char_map.append(char_map[i])
            i += 1

    return "".join(new_text_list), new_char_map


def parse_pdf_bytes(pdf_bytes: bytes, known_sha256s: set[str] | None = None) -> DoclingParseResult:
    if len(pdf_bytes) == 0:
        raise ParseError("zero_byte_pdf", "Uploaded PDF is empty.", 400)

    digest = sha256(pdf_bytes).hexdigest()
    if digest in (known_sha256s or set()):
        raise ParseError(
            "duplicate_pdf",
            "Uploaded PDF matches an existing data source SHA-256 hash.",
            409,
        )

    # Pre-flight check via pypdf to fast-fail on corrupt, encrypted, or too large PDFs
    try:
        reader = PdfReader(BytesIO(pdf_bytes))
        if reader.is_encrypted:
            raise ParseError("encrypted_pdf", "Encrypted PDFs are not supported.", 422)
        page_count = len(reader.pages)
        if page_count > MAX_PAGES:
            raise ParseError(
                "pdf_too_large",
                f"PDF has {page_count} pages; maximum allowed is {MAX_PAGES}.",
                413,
            )
    except (PdfReadError, ValueError, OSError) as exc:
        if isinstance(exc, ParseError):
            raise exc
        raise ParseError("corrupt_pdf", "Uploaded file is not a readable PDF.", 400) from exc

    # Write to a temporary file to convert with Docling converter
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
        tmp_file.write(pdf_bytes)
        tmp_path = Path(tmp_file.name)

    try:
        pipeline_options = PdfPipelineOptions()
        pipeline_options.generate_parsed_pages = True
        pipeline_options.do_table_structure = True
        # This PDF lane is text-layer only (DS-1 acceptance corpus has no scanned pages).
        # OCR is unnecessary here and was fabricating cells on blank pages (breaking the
        # no_extractable_text guard); native text extraction gives real coordinates anyway.
        pipeline_options.do_ocr = False

        converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )

        result = converter.convert(tmp_path)
        if not result or not result.pages:
            raise ParseError("no_extractable_text", "PDF contains no extractable text.", 422)

        # Docling silently drops a page from result.pages if its preprocess stage
        # fails (e.g. a native layout-model crash on that page) rather than raising.
        # Returning a truncated PageIndex list as 200 OK would be silent data loss
        # for a citation pipeline — fail loudly instead.
        if len(result.pages) != page_count:
            raise ParseError(
                "incomplete_parse",
                f"Docling parsed {len(result.pages)} of {page_count} pages; "
                "one or more pages failed during layout preprocessing.",
                422,
            )

        doc = result.document

        # Save DoclingDocument to cache directory for downstream tickets
        cache_dir = Path("services/parser/cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{digest}.json"
        doc.save_as_json(cache_file)

        page_indices: list[PageIndex] = []
        for page in result.pages:
            page_no = page.page_no
            parsed_page = page.parsed_page

            if not parsed_page:
                page_indices.append(PageIndex(page=page_no, text="", char_map=[]))
                continue

            # Iterate over word cells in reading order
            words = list(parsed_page.iterate_cells(unit_type=TextCellUnit.WORD))

            if not words:
                page_indices.append(PageIndex(page=page_no, text="", char_map=[]))
                continue

            # Resolve each word's BoundingRectangle (corner points) to a BoundingBox
            # (.l/.t/.r/.b) once up front — BoundingRectangle itself has no .l/.t/.r/.b.
            word_boxes = [(word, word.rect.to_bounding_box()) for word in words]

            # Group words into visual lines
            lines: list[list[tuple]] = []
            current_line: list[tuple] = []
            for word, bbox in word_boxes:
                if not current_line:
                    current_line.append((word, bbox))
                else:
                    _, prev_bbox = current_line[-1]
                    # BoundingBox here is TOPLEFT-origin (top < bottom numerically),
                    # so height/overlap are bottom-minus-top, not top-minus-bottom.
                    overlap_y = min(prev_bbox.b, bbox.b) - max(prev_bbox.t, bbox.t)
                    height = min(prev_bbox.b - prev_bbox.t, bbox.b - bbox.t)

                    if height > 0 and (overlap_y > 0.5 * height) and (bbox.l >= prev_bbox.l - 5.0):
                        current_line.append((word, bbox))
                    else:
                        lines.append(current_line)
                        current_line = [(word, bbox)]
            if current_line:
                lines.append(current_line)

            # Construct flat page text and character map
            page_text_parts = []
            page_char_map = []

            for line_idx, line in enumerate(lines):
                line_text_parts = []
                line_char_map = []

                for word_idx, (word, bbox) in enumerate(line):
                    word_text = word.text

                    # Interpolate character coordinates within the word bounding box
                    char_boxes = []
                    w = bbox.r - bbox.l
                    char_w = w / len(word_text) if len(word_text) > 0 else 0
                    for j, char in enumerate(word_text):
                        char_boxes.append(
                            CharBox(
                                char=char,
                                x0=float(bbox.l + j * char_w),
                                top=float(bbox.t),
                                x1=float(bbox.l + (j + 1) * char_w),
                                bottom=float(bbox.b),
                                page=page_no,
                            )
                        )

                    # Space between words on the same line
                    if word_idx > 0:
                        _, prev_bbox = line[word_idx - 1]
                        space_box = CharBox(
                            char=" ",
                            x0=float(prev_bbox.r),
                            top=float(prev_bbox.t),
                            x1=float(bbox.l),
                            bottom=float(prev_bbox.b),
                            page=page_no,
                        )
                        line_text_parts.append(" ")
                        line_char_map.append(space_box)

                    line_text_parts.append(word_text)
                    line_char_map.extend(char_boxes)

                # Newline between lines
                if line_idx > 0:
                    _, prev_line_last_bbox = lines[line_idx - 1][-1]
                    newline_box = CharBox(
                        char="\n",
                        x0=float(prev_line_last_bbox.r),
                        top=float(prev_line_last_bbox.t),
                        x1=float(prev_line_last_bbox.r),
                        bottom=float(prev_line_last_bbox.b),
                        page=page_no,
                    )
                    page_text_parts.append("\n")
                    page_char_map.append(newline_box)

                page_text_parts.extend(line_text_parts)
                page_char_map.extend(line_char_map)

            page_text = "".join(page_text_parts)

            # Run numeric token normalization to collapse spaces inside numbers
            normalized_text, normalized_char_map = normalize_numeric_tokens(
                page_text, page_char_map
            )

            # Verify and enforce lengths/characters match exactly. Raised as ParseError
            # rather than `assert` so the check can't be stripped under `python -O`.
            if len(normalized_text) != len(normalized_char_map):
                raise ParseError(
                    "char_map_invariant_violation",
                    f"Page {page_no}: text length ({len(normalized_text)}) != "
                    f"char_map length ({len(normalized_char_map)}).",
                    500,
                )
            for k, ch in enumerate(normalized_text):
                if ch != normalized_char_map[k].char:
                    raise ParseError(
                        "char_map_invariant_violation",
                        f"Page {page_no}: char mismatch at index {k}.",
                        500,
                    )

            page_indices.append(
                PageIndex(page=page_no, text=normalized_text, char_map=normalized_char_map)
            )

        # `result.pages` is non-empty even for a blank page (Docling still emits a
        # page object with no cells), so the earlier `not result.pages` check never
        # catches a text-free PDF. Check aggregate extracted text instead.
        if not any(page.text for page in page_indices):
            raise ParseError("no_extractable_text", "PDF contains no extractable text.", 422)

        return DoclingParseResult(sha256=digest, pages=page_indices)

    finally:
        with contextlib.suppress(OSError):
            tmp_path.unlink()

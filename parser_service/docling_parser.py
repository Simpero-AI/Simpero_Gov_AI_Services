import contextlib
import logging
import re
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path

from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.base_models import Page as DoclingPage
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc.document import DoclingDocument
from docling_core.types.doc.page import TextCellUnit
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from .config import get_settings
from .document_cache import get_document_cache
from .errors import ParseError
from .normalize import kept_indices
from .schemas import CharBox, PageIndex

logger = logging.getLogger(__name__)

_PAGE_NUMBER_RE = re.compile(r"^\d{1,4}$")
BOILERPLATE_ZONE_LINES = 5
MIN_BOILERPLATE_REPEAT_PAGES = 3

# Bumped when the cached parse-result shape changes, so a stale entry written by
# an older format is treated as a miss and re-parsed rather than mis-read.
_PARSE_CACHE_FORMAT = "parse-result-v1"


@dataclass(frozen=True)
class DoclingParseResult:
    sha256: str
    pages: list[PageIndex]
    # In-memory raw DoclingDocument, kept for in-request table/element extraction
    # (DS-W3-2/DS-W3-6). Not serialized by the /parse endpoint, which returns
    # only pages; the Spaces cache is a separate cross-service optimization.
    document: DoclingDocument | None = None


def parse_known_hashes(value: str | None) -> set[str]:
    if not value:
        return set()
    return {item.strip().lower() for item in value.split(",") if item.strip()}


def normalize_numeric_tokens(text: str, char_map: list[CharBox]) -> tuple[str, list[CharBox]]:
    """Collapse split numeric tokens ("3 ,817" -> "3,817") in the flat page
    index, dropping each collapsed space's char_map entry so text and char_map
    stay the same length and every surviving character still points at its own
    source glyph.

    The rule itself lives in normalize.py — shared verbatim with table cell
    values (DS-W3-2) so the two can never drift apart.
    """
    keep = kept_indices(text)
    return "".join(text[i] for i in keep), [char_map[i] for i in keep]


def char_cells_by_page(pdf_bytes: bytes) -> dict[int, list]:
    """Per-glyph cells for every page, read from docling-parse directly.

    Docling's pipeline drops these (see _real_char_cells), so they are read from
    the parser underneath it -- the same library, already a dependency, no model
    involved. Per-character geometry on a digital-born PDF is a deterministic
    lookup, not a capability that has to be inferred.

    Fails soft: a page that cannot be read contributes nothing and that page
    keeps word precision, which is the behaviour this replaces rather than a new
    failure mode.
    """
    try:
        from docling_parse.pdf_parser import DoclingPdfParser
    except ImportError:  # pragma: no cover - docling-parse ships with docling
        return {}

    cells: dict[int, list] = {}
    try:
        parser = DoclingPdfParser()
        document = parser.load(BytesIO(pdf_bytes))
        for page_no in range(1, document.number_of_pages() + 1):
            try:
                page = document.get_page(page_no)
            except Exception:
                continue
            # Whitespace cells are dropped because the two sequences count it
            # differently: char cells carry the spaces between words (a page
            # opens [' ', ' ', 'B', 'e', 'a', 'r', ',']) while the word cells
            # matched against them do not ('Bear,'). Left in, the very first
            # word mismatches and _char_boxes_for_word disables the char path
            # for the rest of the page -- which is exactly why every box on
            # this corpus was word precision. A word's own text never contains
            # a space, so nothing citable is lost.
            #
            # Boxes are converted to TOP-LEFT here, at the one place the page
            # height is in hand. A cell arrives bottom-left while word cells and
            # char_map are top-left, so an unconverted box lands the right
            # distance from the wrong edge -- a footer at y=37 reading as y=37
            # from the top. Merely un-inverting it (min/max) is not enough and
            # looks like it worked: the box stops being upside down and stays in
            # the wrong coordinate system.
            height = page.dimension.height if page.dimension else None
            if height is None:
                continue
            glyphs = []
            for cell in page.char_cells:
                if not cell.text.strip():
                    continue
                box = cell.rect.to_bounding_box().to_top_left_origin(height)
                glyphs.append((cell.text, float(box.l), float(box.t), float(box.r), float(box.b)))
            if glyphs:
                cells[page_no] = glyphs
    except Exception:
        logger.warning("docling-parse char cells unavailable; falling back to word precision")
        return {}
    return cells


def _char_boxes_for_word(
    word_text: str, word_bbox, page_no: int, char_cells: list | None, cursor: int
) -> tuple[list[CharBox], int, bool]:
    """Build per-character boxes for one word.

    If `char_cells` holds real per-glyph cells and the next len(word_text) of
    them match this word's text exactly, each character gets its own real
    bounding box (precision="char"). Otherwise every character in this word
    gets the word's own full bounding box (precision="word") — an honest
    statement of word-level precision, never a fabricated per-glyph estimate.

    Returns (char_boxes, new_cursor, still_usable). Once a mismatch is found,
    `still_usable` is False for the remainder of the page — a partial
    misalignment means the assumed reading-order correspondence between
    char_cells and word text has broken down and can no longer be trusted.
    """
    if char_cells is not None and cursor + len(word_text) <= len(char_cells):
        candidate = char_cells[cursor : cursor + len(word_text)]
        if all(cell[0] == ch for cell, ch in zip(candidate, word_text, strict=True)):
            # Already TOP-LEFT: char_cells_by_page converts at the source, where
            # the page height is available.
            boxes = [
                CharBox(
                    char=ch,
                    x0=x0,
                    top=top,
                    x1=x1,
                    bottom=bottom,
                    page=page_no,
                    precision="char",
                )
                for ch, (_, x0, top, x1, bottom) in zip(word_text, candidate, strict=True)
            ]
            return boxes, cursor + len(word_text), True

    boxes = [
        CharBox(
            char=ch,
            x0=float(word_bbox.l),
            top=float(word_bbox.t),
            x1=float(word_bbox.r),
            bottom=float(word_bbox.b),
            page=page_no,
            precision="word",
        )
        for ch in word_text
    ]
    return boxes, cursor, False


def _build_page_index(
    page: DoclingPage, page_no: int, supplied_char_cells: list | None = None
) -> PageIndex:
    parsed_page = page.parsed_page
    if not parsed_page:
        return PageIndex(page=page_no, text="", char_map=[])

    # Iterate over word cells in reading order
    words = list(parsed_page.iterate_cells(unit_type=TextCellUnit.WORD))
    if not words:
        return PageIndex(page=page_no, text="", char_map=[])

    # Per-glyph geometry, or None -- in which case every box on this page is
    # word precision, stated as such rather than estimated.
    char_cells = supplied_char_cells
    char_cursor = 0

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

            char_boxes, char_cursor, still_usable = _char_boxes_for_word(
                word_text, bbox, page_no, char_cells, char_cursor
            )
            if not still_usable:
                char_cells = None

            # Space between words on the same line. The gap between two real
            # word boxes, not a per-glyph estimate — always word-level.
            if word_idx > 0:
                _, prev_bbox = line[word_idx - 1]
                space_box = CharBox(
                    char=" ",
                    x0=float(prev_bbox.r),
                    top=float(prev_bbox.t),
                    x1=float(bbox.l),
                    bottom=float(prev_bbox.b),
                    page=page_no,
                    precision="word",
                )
                line_text_parts.append(" ")
                line_char_map.append(space_box)

            line_text_parts.append(word_text)
            line_char_map.extend(char_boxes)

        # Newline between lines — a zero-width marker at the prior line's end,
        # not a glyph; always word-level.
        if line_idx > 0:
            _, prev_line_last_bbox = lines[line_idx - 1][-1]
            newline_box = CharBox(
                char="\n",
                x0=float(prev_line_last_bbox.r),
                top=float(prev_line_last_bbox.t),
                x1=float(prev_line_last_bbox.r),
                bottom=float(prev_line_last_bbox.b),
                page=page_no,
                precision="word",
            )
            page_text_parts.append("\n")
            page_char_map.append(newline_box)

        page_text_parts.extend(line_text_parts)
        page_char_map.extend(line_char_map)

    page_text = "".join(page_text_parts)

    # Run numeric token normalization to collapse spaces inside numbers
    normalized_text, normalized_char_map = normalize_numeric_tokens(page_text, page_char_map)

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

    return PageIndex(page=page_no, text=normalized_text, char_map=normalized_char_map)


def _line_spans(text: str) -> list[tuple[int, int, str]]:
    """Return (start, end, line_text) for each '\\n'-split line, end exclusive."""
    spans = []
    start = 0
    for line in text.split("\n"):
        end = start + len(line)
        spans.append((start, end, line))
        start = end + 1
    return spans


def _normalize_line(line: str) -> str:
    return " ".join(line.split())


def tag_boilerplate(page_indices: list[PageIndex]) -> None:
    """Tag repeating headers/footers and page-number furniture as is_boilerplate.

    Docling's page_header/page_footer layout labels are not reliably assigned on
    real CIM documents (verified empirically against the fixed test corpus), so
    this falls back to the ticket's documented approach: lines near the top/bottom
    of each page whose normalized text repeats verbatim across >= 3 pages (banners,
    running headers/footers), plus short digit-only lines in that same zone when
    the *pattern* of a page-number-shaped line there recurs across >= 3 pages.
    Mutates char_map entries in place; does not strip any text.
    """
    text_zone_lines: list[tuple[int, int, int, str]] = []
    digit_zone_lines: list[tuple[int, int, int]] = []

    for page_idx, page in enumerate(page_indices):
        spans = _line_spans(page.text)
        zone = spans[:BOILERPLATE_ZONE_LINES] + spans[-BOILERPLATE_ZONE_LINES:]
        for start, end, line in zone:
            norm = _normalize_line(line)
            if not norm:
                continue
            if _PAGE_NUMBER_RE.match(norm):
                digit_zone_lines.append((page_idx, start, end))
            else:
                text_zone_lines.append((page_idx, start, end, norm))

    # Repeating banner/header/footer text: same normalized line in >= N distinct pages.
    pages_by_text: dict[str, set[int]] = defaultdict(set)
    for page_idx, _, _, norm in text_zone_lines:
        pages_by_text[norm].add(page_idx)
    repeating_texts = {
        text for text, pages in pages_by_text.items() if len(pages) >= MIN_BOILERPLATE_REPEAT_PAGES
    }

    for page_idx, start, end, norm in text_zone_lines:
        if norm in repeating_texts:
            for char_box in page_indices[page_idx].char_map[start:end]:
                char_box.is_boilerplate = True

    # Page-number furniture: short digit-only lines in the header/footer zone.
    # The digit itself varies per page, so we key on the *structural pattern*
    # (a page-number-shaped line in this zone) recurring across >= N pages,
    # not on exact text repetition.
    if len({page_idx for page_idx, _, _ in digit_zone_lines}) >= MIN_BOILERPLATE_REPEAT_PAGES:
        for page_idx, start, end in digit_zone_lines:
            for char_box in page_indices[page_idx].char_map[start:end]:
                char_box.is_boilerplate = True


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

    # Read-through: a cached finished parse for these exact bytes lets us skip
    # docling entirely -- extraction and chunking each parse the same PDF today.
    # The cache holds the finished PageIndex list AND the DoclingDocument, because
    # pages are built from the ConversionResult (not itself cacheable), so the raw
    # document alone cannot rebuild them. Best-effort: any miss or malformed entry
    # falls through to a full parse.
    document_cache = get_document_cache()
    if document_cache.enabled:
        cached = document_cache.get_json(f"{digest}.json")
        if isinstance(cached, dict) and cached.get("format") == _PARSE_CACHE_FORMAT:
            try:
                pages = [PageIndex.model_validate(p) for p in cached["pages"]]
                cached_document = cached.get("document")
                document = (
                    DoclingDocument.model_validate(cached_document)
                    if cached_document is not None
                    else None
                )
                logger.info(
                    "parse cache hit: sha256=%s pages=%d (docling skipped)", digest[:16], len(pages)
                )
                return DoclingParseResult(sha256=digest, pages=pages, document=document)
            except Exception as exc:  # noqa: BLE001 -- a bad cache entry must never block a parse
                logger.warning(
                    "parse cache rehydrate failed for sha256=%s (%s); re-parsing",
                    digest[:16],
                    exc,
                )

    settings = get_settings()

    # Pre-flight check via pypdf to fast-fail on corrupt, encrypted, or too large PDFs
    try:
        reader = PdfReader(BytesIO(pdf_bytes))
        if reader.is_encrypted:
            raise ParseError("encrypted_pdf", "Encrypted PDFs are not supported.", 422)
        page_count = len(reader.pages)
        if page_count > settings.max_pages:
            raise ParseError(
                "pdf_too_large",
                f"PDF has {page_count} pages; maximum allowed is {settings.max_pages}.",
                413,
            )
    except (PdfReadError, ValueError, OSError) as exc:
        if isinstance(exc, ParseError):
            raise exc
        raise ParseError("corrupt_pdf", "Uploaded file is not a readable PDF.", 400) from exc

    logger.info("parse start: sha256=%s pages=%d bytes=%d", digest[:16], page_count, len(pdf_bytes))

    # Write to a temporary file to convert with Docling converter
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
        tmp_file.write(pdf_bytes)
        tmp_path = Path(tmp_file.name)

    try:
        pipeline_options = PdfPipelineOptions()
        pipeline_options.generate_parsed_pages = True
        pipeline_options.do_table_structure = True
        # This PDF lane is text-layer only (DS-W3-1 acceptance corpus has no scanned pages).
        # OCR is unnecessary here and was fabricating cells on blank pages (breaking the
        # no_extractable_text guard); native text extraction gives real coordinates anyway.
        pipeline_options.do_ocr = False

        converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )

        result = converter.convert(tmp_path)

        # Docling returns partial_success when a page fails internally but its
        # (empty) page object is still present — convert() only raises on total
        # failure. Such a page would pass the count check below and ship as a
        # silently-blank PageIndex, so fail closed on any non-success status.
        if result.status != ConversionStatus.SUCCESS:
            logger.warning(
                "incomplete parse: sha256=%s docling status=%s", digest[:16], result.status.value
            )
            raise ParseError(
                "incomplete_parse",
                f"Docling conversion status was {result.status.value}; "
                "one or more pages failed during processing.",
                422,
            )

        if len(result.pages) != page_count:
            logger.warning(
                "incomplete parse: sha256=%s docling returned %d of %d pages",
                digest[:16],
                len(result.pages),
                page_count,
            )
            raise ParseError(
                "incomplete_parse",
                f"Docling parsed {len(result.pages)} of {page_count} pages; "
                "one or more pages failed during layout preprocessing.",
                422,
            )

        # Per-glyph geometry the pipeline drops; see char_cells_by_page.
        char_cells = char_cells_by_page(pdf_bytes)
        page_indices = [
            _build_page_index(page, page.page_no, char_cells.get(page.page_no))
            for page in result.pages
        ]
        tag_boilerplate(page_indices)

        # A blank page still yields an (empty) PageIndex rather than being absent,
        # so check aggregate extracted text rather than page count for this guard.
        if not any(page.text for page in page_indices):
            logger.warning(
                "no extractable text: sha256=%s (likely scanned or image-only)", digest[:16]
            )
            raise ParseError("no_extractable_text", "PDF contains no extractable text.", 422)

        # Cache the finished parse -- the PageIndex list AND the DoclingDocument --
        # so a repeat parse of these bytes skips docling. Written only after every
        # fail-closed guard above passes, so the cache never holds an invalid parse.
        # Backed by object storage, encrypted at rest; best-effort, never fatal.
        if document_cache.enabled:
            document_cache.put_json(
                f"{digest}.json",
                {
                    "format": _PARSE_CACHE_FORMAT,
                    "pages": [page.model_dump() for page in page_indices],
                    "document": result.document.export_to_dict(),
                },
            )
            logger.debug("cached parse result: sha256=%s pages=%d", digest[:16], len(page_indices))

        logger.info(
            "parse complete: sha256=%s pages=%d empty_pages=%d",
            digest[:16],
            len(page_indices),
            sum(1 for p in page_indices if not p.text),
        )
        return DoclingParseResult(sha256=digest, pages=page_indices, document=result.document)

    finally:
        with contextlib.suppress(OSError):
            tmp_path.unlink()

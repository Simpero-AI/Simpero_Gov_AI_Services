"""DS-W3 DOCX path -- native Docling read + paragraph index.

Docling reads DOCX natively (InputFormat.DOCX), so this lane needs no new
library. What it does NOT give is provenance: every DOCX text item comes back
with no prov at all -- no page, no charspan, no bbox. That is not a gap in
Docling, it is the truth about the format. A Word file is a flow document with
no page layout, so there is no geometry to report; pagination only exists once
something renders it.

So the paragraph index is built here from Docling's ordered body items, exactly
the way DS-W3-1 builds the PDF positioned index from Docling's output rather
than trusting prov. The trust bar is identical to the PDF lane; only the shape
differs:

    PDF   -> (page, char_start, char_end, bbox)
    DOCX  -> (paragraph, char_start, char_end)          -- no geometry, by nature

Both resolve through the same exact-span core (resolver.find_exact_span) and are
normalized by the same numeric-token rule (normalize.normalize_numeric_text), so
a quote resolves identically whichever lane read the document, and a claim is
citable on exactly the same terms: found exactly once, or not citable.
"""

import logging
import tempfile
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from docling.datamodel.base_models import ConversionStatus
from docling.document_converter import DocumentConverter
from docling_core.types.doc.document import DoclingDocument

from .config import get_settings
from .errors import ParseError
from .normalize import normalize_numeric_text
from .ooxml import reject_decompression_bomb
from .schemas import ParagraphIndex

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DocxParseResult:
    sha256: str
    paragraphs: list[ParagraphIndex]
    # In-memory raw DoclingDocument, kept for in-request table/element work the
    # same way the PDF lane does. Not serialized by the endpoint.
    document: DoclingDocument | None = None


def parse_docx_bytes(docx_bytes: bytes, known_sha256s: set[str] | None = None) -> DocxParseResult:
    """Parse DOCX bytes into a paragraph index, fail-closed.

    Mirrors parse_pdf_bytes' guards so the lanes reject the same things for the
    same reasons: empty upload, duplicate content hash, decompression bomb,
    unreadable file, an incomplete Docling conversion, and a document with no
    extractable text (a DOCX that yields nothing is a failure to surface, never
    an empty success).
    """
    if len(docx_bytes) == 0:
        raise ParseError("zero_byte_docx", "Uploaded DOCX is empty.", 400)

    digest = sha256(docx_bytes).hexdigest()
    if digest in (known_sha256s or set()):
        raise ParseError(
            "duplicate_docx",
            "Uploaded DOCX matches an existing data source SHA-256 hash.",
            409,
        )

    settings = get_settings()
    reject_decompression_bomb(docx_bytes, settings.max_ooxml_uncompressed_bytes, kind="DOCX")

    logger.info("docx parse start: sha256=%s bytes=%d", digest[:16], len(docx_bytes))

    # Docling converts from a path; write the untrusted bytes to a temp file.
    with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp_file:
        tmp_file.write(docx_bytes)
        tmp_path = Path(tmp_file.name)

    try:
        # No pipeline options: the DOCX backend does no layout/OCR work — there
        # is no page image to analyse. Any failure reading these untrusted bytes
        # is a fail-closed 400, never an unhandled 500.
        try:
            result = DocumentConverter().convert(tmp_path)
        except ParseError:
            raise
        except Exception as exc:  # noqa: BLE001 -- untrusted-input boundary, fail closed
            logger.warning("docx parse failed: sha256=%s error=%s", digest[:16], exc)
            raise ParseError("corrupt_docx", "Uploaded file is not a readable DOCX.", 400) from exc

        # Same rule as the PDF lane: a non-success status means part of the
        # document silently did not convert, and a partial parse must never ship
        # as if it were whole.
        if result.status != ConversionStatus.SUCCESS:
            logger.warning(
                "incomplete parse: sha256=%s docling status=%s", digest[:16], result.status.value
            )
            raise ParseError(
                "incomplete_parse",
                f"Docling returned {result.status.value} for this DOCX.",
                422,
            )

        paragraphs = _build_paragraph_index(result.document, settings.max_paragraphs)
    finally:
        tmp_path.unlink(missing_ok=True)

    if not paragraphs:
        logger.warning("no extractable text: sha256=%s", digest[:16])
        raise ParseError(
            "no_extractable_text",
            "DOCX produced no extractable text.",
            422,
        )

    logger.info("docx parse complete: sha256=%s paragraphs=%d", digest[:16], len(paragraphs))
    return DocxParseResult(sha256=digest, paragraphs=paragraphs, document=result.document)


def _build_paragraph_index(doc: DoclingDocument, max_paragraphs: int) -> list[ParagraphIndex]:
    """Docling's ordered body text items -> the paragraph index.

    `paragraph` is the item's 0-based position in body order, which is the only
    stable locator a flow document has. Empty items are skipped rather than
    indexed: a blank paragraph is not citable, and keeping it would shift every
    subsequent locator for no benefit. Text is normalized with the same rule as
    the page index so the resolver's contract is identical across lanes.
    """
    paragraphs: list[ParagraphIndex] = []
    for item in doc.texts:
        text = normalize_numeric_text(item.text or "")
        if not text.strip():
            continue
        if len(paragraphs) >= max_paragraphs:
            raise ParseError(
                "docx_too_large",
                f"DOCX has more than {max_paragraphs} paragraphs.",
                413,
            )
        paragraphs.append(
            ParagraphIndex(
                paragraph=len(paragraphs),
                text=text,
                label=getattr(item, "label", None) and str(item.label),
            )
        )
    return paragraphs

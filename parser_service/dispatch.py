"""Format detection and byte-to-ParseResponse dispatch, shared by the HTTP
route (main.py) and the SAQ worker (worker.py). Moved out of main.py so the
worker can reuse the exact same branch structure without importing FastAPI.
"""

from io import BytesIO
from typing import Literal
from zipfile import BadZipFile, ZipFile

from pydantic import BaseModel

from .docling_parser import parse_pdf_bytes
from .docx_parser import parse_docx_bytes
from .errors import ParseError
from .schemas import PageIndex, ParagraphIndex, XlsxSheetRecord
from .xlsx_parser import parse_xlsx_bytes

SourceFormat = Literal["pdf", "xlsx", "docx"]


class ParseResponse(BaseModel):
    """One tagged result for any supported document.

    The three formats share a pipeline and a trust bar, not a shape: a PDF has
    pages with geometry, an XLSX has addressed cells, a DOCX has paragraphs and
    no geometry at all. Rather than flatten that into a lowest-common-denominator
    blob, the response says which lane read the document and carries that lane's
    index. Exactly one of pages/sheets/paragraphs is populated, per `kind`.
    """

    kind: SourceFormat
    sha256: str
    pages: list[PageIndex] | None = None
    sheets: list[XlsxSheetRecord] | None = None
    paragraphs: list[ParagraphIndex] | None = None


def detect_format(data: bytes) -> SourceFormat:
    """Identify the document from its own bytes.

    Never trusts a filename or a client-supplied content-type: this is untrusted
    input, and dispatching the wrong lane on a lie is how a parser gets fed
    something it never guards against. XLSX and DOCX are both OOXML zips, so they
    are told apart by the part each format must contain.
    """
    if data.startswith(b"%PDF-"):
        return "pdf"

    if data.startswith(b"PK\x03\x04"):
        try:
            with ZipFile(BytesIO(data)) as archive:
                names = set(archive.namelist())
        except BadZipFile as exc:
            raise ParseError(
                "unsupported_format", "Uploaded file is not a readable document.", 400
            ) from exc
        if "xl/workbook.xml" in names:
            return "xlsx"
        if "word/document.xml" in names:
            return "docx"

    raise ParseError("unsupported_format", "Uploaded file is not a PDF, XLSX or DOCX.", 415)


def parse_bytes(data: bytes, known_hashes: set[str]) -> ParseResponse:
    """Sniff the format and run the matching lane, tagging the result by kind.

    Shared by POST /parse (main.py) and the SAQ worker (worker.py) so both
    entry points dispatch identically. ParseError propagates uncaught -- each
    caller maps it to its own failure shape (HTTPException for the route,
    a rejected-job dict for the worker).
    """
    # An empty body has no format to sniff. Route it to the PDF lane so the
    # caller still gets the specific zero_byte_pdf rejection rather than a
    # vaguer "unsupported format" for input that stated no format.
    kind: SourceFormat = "pdf" if not data else detect_format(data)

    if kind == "pdf":
        pdf = parse_pdf_bytes(data, known_hashes)
        return ParseResponse(kind="pdf", sha256=pdf.sha256, pages=pdf.pages)
    elif kind == "xlsx":
        xlsx = parse_xlsx_bytes(data)
        return ParseResponse(kind="xlsx", sha256=xlsx.sha256, sheets=xlsx.sheets)
    else:
        docx = parse_docx_bytes(data, known_hashes)
        return ParseResponse(kind="docx", sha256=docx.sha256, paragraphs=docx.paragraphs)

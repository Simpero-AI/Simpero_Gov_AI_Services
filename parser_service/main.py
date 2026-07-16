import logging
from io import BytesIO
from typing import Literal
from zipfile import BadZipFile, ZipFile

from fastapi import FastAPI, Header, HTTPException, Request, Response
from pydantic import BaseModel

from .docling_parser import parse_known_hashes, parse_pdf_bytes
from .docx_parser import parse_docx_bytes
from .errors import ParseError
from .schemas import PageIndex, ParagraphIndex, XlsxSheetRecord
from .xlsx_parser import parse_xlsx_bytes

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Simpero Parser Service",
    version="0.1.0",
    description="Alpha parse service: PDF, XLSX and DOCX to an exactly-citable index.",
)

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


@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok", "service": "parser"}


@app.post("/parse", response_model=ParseResponse)
async def parse(
    request: Request,
    response: Response,
    x_known_sha256: str | None = Header(default=None),
    x_known_sha256s: str | None = Header(default=None),
) -> ParseResponse:
    """Parse any supported document into its exactly-citable index."""
    known_hashes = parse_known_hashes(x_known_sha256) | parse_known_hashes(x_known_sha256s)
    data = await request.body()
    logger.info("parse request: %d bytes", len(data))

    try:
        # An empty body has no format to sniff. Route it to the PDF lane so the
        # caller still gets the specific zero_byte_pdf rejection rather than a
        # vaguer "unsupported format" for a request that stated no format.
        kind: SourceFormat = "pdf" if not data else detect_format(data)

        if kind == "pdf":
            pdf = parse_pdf_bytes(data, known_hashes)
            result = ParseResponse(kind="pdf", sha256=pdf.sha256, pages=pdf.pages)
            count = f"pages={len(pdf.pages)}"
        elif kind == "xlsx":
            xlsx = parse_xlsx_bytes(data)
            result = ParseResponse(kind="xlsx", sha256=xlsx.sha256, sheets=xlsx.sheets)
            count = f"sheets={len(xlsx.sheets)}"
        else:
            docx = parse_docx_bytes(data, known_hashes)
            result = ParseResponse(kind="docx", sha256=docx.sha256, paragraphs=docx.paragraphs)
            count = f"paragraphs={len(docx.paragraphs)}"
    except ParseError as exc:
        logger.warning(
            "parse rejected: code=%s status=%d bytes=%d: %s",
            exc.code,
            exc.status_code,
            len(data),
            exc.message,
        )
        raise HTTPException(
            status_code=exc.status_code,
            detail={"code": exc.code, "message": exc.message},
        ) from exc

    logger.info("parse ok: kind=%s sha256=%s %s", kind, result.sha256[:16], count)
    response.headers["X-Content-SHA256"] = result.sha256
    return result

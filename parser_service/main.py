import logging

from fastapi import FastAPI, Header, HTTPException, Request, Response

from .docling_parser import ParseError, parse_known_hashes, parse_pdf_bytes
from .schemas import PageIndex

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Simpero Parser Service",
    version="0.1.0",
    description="Alpha PDF parser with character-span coordinates.",
)


@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok", "service": "parser"}


@app.post("/parse", response_model=list[PageIndex])
async def parse_pdf(
    request: Request,
    response: Response,
    x_known_sha256: str | None = Header(default=None),
    x_known_sha256s: str | None = Header(default=None),
) -> list[PageIndex]:
    known_hashes = parse_known_hashes(x_known_sha256) | parse_known_hashes(x_known_sha256s)
    pdf_bytes = await request.body()
    logger.info("parse request: %d bytes", len(pdf_bytes))

    try:
        result = parse_pdf_bytes(pdf_bytes, known_hashes)
    except ParseError as exc:
        logger.warning(
            "parse rejected: code=%s status=%d bytes=%d: %s",
            exc.code,
            exc.status_code,
            len(pdf_bytes),
            exc.message,
        )
        raise HTTPException(
            status_code=exc.status_code,
            detail={"code": exc.code, "message": exc.message},
        ) from exc

    logger.info("parse ok: sha256=%s pages=%d", result.sha256[:16], len(result.pages))
    response.headers["X-Content-SHA256"] = result.sha256
    return result.pages

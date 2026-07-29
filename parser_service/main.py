import hmac
import logging

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response

from .config import ParserSettings, get_settings
from .dispatch import ParseResponse, parse_bytes
from .docling_parser import parse_known_hashes
from .errors import ParseError

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Simpero Parser Service",
    version="0.1.0",
    description="Alpha parse service: PDF, XLSX and DOCX to an exactly-citable index.",
)


def verify_parser_key(
    x_parser_key: str | None = Header(default=None),
    settings: ParserSettings = Depends(get_settings),
) -> None:
    """Fail-closed shared-secret check for POST /parse.

    An unset key means "refuse to serve" (503), never "auth disabled" -- a
    droplet deployed with the key missing from its env must not silently
    become a public parse farm. hmac.compare_digest is used instead of `==`
    for a constant-time comparison.
    """
    if not settings.api_key:
        raise HTTPException(status_code=503, detail="parser service misconfigured")
    if not x_parser_key or not hmac.compare_digest(x_parser_key, settings.api_key):
        raise HTTPException(status_code=401, detail="missing or invalid X-Parser-Key")


@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok", "service": "parser"}


@app.post("/parse", response_model=ParseResponse, dependencies=[Depends(verify_parser_key)])
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
        result = parse_bytes(data, known_hashes)
        if result.kind == "pdf":
            count = f"pages={len(result.pages or [])}"
        elif result.kind == "xlsx":
            count = f"sheets={len(result.sheets or [])}"
        else:
            count = f"paragraphs={len(result.paragraphs or [])}"
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

    logger.info("parse ok: kind=%s sha256=%s %s", result.kind, result.sha256[:16], count)
    response.headers["X-Content-SHA256"] = result.sha256
    return result

import hmac
import logging

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response

from .config import ParserSettings, get_settings
from .dispatch import ParseResponse, parse_bytes
from .docling_parser import parse_known_hashes
from .errors import ParseError
from .extract_service import ProseCredentialMissing, extract_claims

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


def _undo_header_mojibake(value: str) -> str:
    """Recover a non-ASCII header value Starlette decoded as latin-1.

    ASGI headers are latin-1 by spec, so a client that sent a UTF-8 encoded
    company name (a real requirement -- entity names are not ASCII-only) comes
    through as mojibake: each original UTF-8 byte becomes one latin-1
    character. Re-encoding as latin-1 recovers those original bytes, and
    decoding them as UTF-8 recovers the intended string. A no-op for plain
    ASCII. A value that is not valid UTF-8 either is left exactly as Starlette
    gave it -- an imperfect but readable string beats a 500 on every request
    with an accented company name.
    """
    try:
        return value.encode("latin-1").decode("utf-8")
    except UnicodeError:
        return value


@app.post("/extract", dependencies=[Depends(verify_parser_key)])
async def extract(
    request: Request,
    x_run_id: str = Header(...),
    x_correlation_id: str = Header(...),
    x_entity: str = Header(...),
    x_source_file: str | None = Header(default=None),
    x_prose: bool = Header(default=False),
    x_qualitative: bool = Header(default=False),
    x_canonicalize_attributes: bool = Header(default=False),
    x_audit: bool = Header(default=False),
) -> dict:
    """Parse a PDF and emit its claims -- the callable form of scripts/emit_claims.py.

    Calls the same extract_claims entry point the CLI does, so the two can
    never drift (SIM-340/E5). Stateless: the claims payload is returned in the
    response body, never persisted here -- persistence is the backend's job.
    """
    data = await request.body()
    logger.info(
        "extract request: run_id=%s correlation_id=%s %d bytes",
        x_run_id,
        x_correlation_id,
        len(data),
    )

    try:
        payload = extract_claims(
            data,
            entity=_undo_header_mojibake(x_entity),
            run_id=x_run_id,
            correlation_id=x_correlation_id,
            source_file=x_source_file,
            prose=x_prose,
            qualitative=x_qualitative,
            canonicalize_attributes=x_canonicalize_attributes,
            audit=x_audit,
        )
    except ProseCredentialMissing as exc:
        logger.warning("extract rejected: run_id=%s missing prose credential", x_run_id)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ParseError as exc:
        logger.warning(
            "extract rejected: run_id=%s code=%s status=%d bytes=%d: %s",
            x_run_id,
            exc.code,
            exc.status_code,
            len(data),
            exc.message,
        )
        raise HTTPException(
            status_code=exc.status_code,
            detail={"code": exc.code, "message": exc.message},
        ) from exc

    logger.info(
        "extract ok: run_id=%s correlation_id=%s claims=%d",
        x_run_id,
        x_correlation_id,
        len(payload["claims"]),
    )
    return payload

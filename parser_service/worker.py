"""SAQ worker consuming parse jobs from the shared "parse" Valkey queue that
Simpero_AI_Gov_Alpha's app/jobs/parse_client.py enqueues onto (SIM: see that
module's docstring for the full cross-repo contract).

Run via `saq parser_service.worker.settings`. Reuses dispatch.parse_bytes --
the same pipeline POST /parse uses -- so the HTTP and queue entry points can
never drift. Results are never returned through the queue itself (SAQ job
results have their own size/ttl limits and Valkey isn't meant to hold parsed
document bodies); instead they're written to Spaces and the job returns a
bucket+key pointer.
"""

import asyncio
import logging

from botocore.exceptions import ClientError
from saq import Queue
from saq.types import Context, SettingsDict

from .config import get_settings
from .dispatch import parse_bytes
from .document_cache import build_spaces_client
from .errors import ParseError

logger = logging.getLogger(__name__)

parser_settings = get_settings()

if not parser_settings.valkey_url:
    # Fail-closed, same reasoning as api_key in config.py: a worker process
    # that starts without a queue URL should crash immediately, not sit there
    # doing nothing while jobs pile up unconsumed.
    raise RuntimeError("PARSER_VALKEY_URL is unset -- worker refuses to start")

queue = Queue.from_url(parser_settings.valkey_url, name=parser_settings.queue_name)


async def parse_document(
    ctx: Context, *, spaces_key: str, known_sha256s: list[str] | None = None
) -> dict:
    """Parse a document already uploaded to Spaces at `spaces_key`, writing
    the result back to Spaces as a bucket+key pointer.

    Function name and kwargs must match Alpha's enqueue_parse_job exactly --
    SAQ registers/dispatches tasks by this name.
    """
    client = build_spaces_client(parser_settings)
    if client is None:
        # Misconfigured worker (started with a valkey_url but no Spaces
        # credentials) must fail loudly, not silently no-op every job.
        raise RuntimeError("Spaces is not configured -- worker cannot fetch source documents")

    try:
        obj = client.get_object(Bucket=parser_settings.spaces_bucket, Key=spaces_key)
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code in {"NoSuchKey", "404"}:
            return {
                "status": "rejected",
                "code": "source_not_found",
                "message": f"{spaces_key}: {exc}",
            }
        raise  # any other Spaces error is unexpected -- let SAQ retry/fail the job

    data = obj["Body"].read()
    # known_sha256s comparisons elsewhere in this codebase (parse_known_hashes
    # in docling_parser.py) are lowercase; match that here.
    known_hashes = {h.lower() for h in (known_sha256s or [])}

    try:
        result = await asyncio.to_thread(parse_bytes, data, known_hashes)
    except ParseError as exc:
        logger.warning("parse job rejected: key=%s code=%s: %s", spaces_key, exc.code, exc.message)
        return {"status": "rejected", "code": exc.code, "message": exc.message}

    if result.kind == "pdf":
        count = len(result.pages or [])
    elif result.kind == "xlsx":
        count = len(result.sheets or [])
    else:
        count = len(result.paragraphs or [])

    results_key = f"{parser_settings.results_key_prefix.rstrip('/')}/{result.sha256}.json"
    # Deliberately not best-effort like document_cache's writes: a job that
    # reports success with an unwritten/unreadable result is silent data
    # loss, so a failure here must propagate and fail the job.
    client.put_object(
        Bucket=parser_settings.spaces_bucket,
        Key=results_key,
        Body=result.model_dump_json().encode(),
        ContentType="application/json",
        ServerSideEncryption="AES256",
    )

    return {
        "status": "parsed",
        "kind": result.kind,
        "sha256": result.sha256,
        "bucket": parser_settings.spaces_bucket,
        "key": results_key,
        "count": count,
    }


async def _normalize_job_policy(ctx: Context) -> None:
    """SAQ before_process hook.

    Alpha's enqueue_parse_job doesn't set timeout/retries/ttl at enqueue time,
    so SAQ's defaults (10s timeout!) would kill every real parse almost
    immediately. update() is required, not optional: SAQ's sweeper reads the
    job back from Valkey, so an in-memory-only mutation doesn't survive a
    sweep check.
    """
    # ctx["job"] directly trips reportTypedDictNotRequiredAccess -- Context
    # marks "job" optional even though before_process is always called with
    # one; .get + assert narrows it for pyright without a blanket ignore.
    job = ctx.get("job")
    assert job is not None, "before_process is always invoked with a job in ctx"
    job.timeout = 1800
    job.retries = 2
    job.ttl = 86400
    await job.update()


settings: SettingsDict = {
    "queue": queue,
    "functions": [parse_document],
    "before_process": _normalize_job_policy,
    "concurrency": 1,
}

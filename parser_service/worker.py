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
import json
import logging
import os
from uuid import uuid4

from botocore.exceptions import ClientError
from saq import Queue
from saq.types import Context, SettingsDict

from .config import get_settings
from .dispatch import parse_bytes
from .document_cache import build_spaces_client
from .errors import ParseError
from .extract_service import ProseCredentialMissing, extract_claims

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


async def process_document(
    ctx: Context,
    *,
    spaces_key: str,
    entity: str,
    run_id: str | None = None,
    correlation_id: str | None = None,
    known_sha256s: list[str] | None = None,
    audit: bool = True,
    sector_options: list[str] | None = None,
    geo_options: list[str] | None = None,
) -> dict:
    """Combined parse + claim-extraction + binding-audit job for the deal flow.

    Fetches `spaces_key` from Spaces the same way `parse_document` does, then
    calls `extract_claims` -- the same entry point `POST /extract` and
    `scripts/emit_claims.py` call, so the queue and CLI paths can never drift.
    `extract_claims`'s internal parse is a read-through cache hit when the
    document was parsed before (docling_parser.py), so this is not a second
    docling pass. The claims envelope is written to Spaces and only a
    `{bucket, key}` pointer returned -- never the payload inline, same reason as
    `parse_document` (SAQ/Valkey are not for large bodies).

    Runs the FULL extraction (prose tier + attribute canonicalization), not
    table-only: the deal flow's downstream edge, reconciliation, and consistency
    stages need the second tier, canonical attributes, and claim_type, none of
    which a table-only run produces (see the extract_claims call below).

    Function name and kwargs must match Alpha's enqueue_process_document_job
    exactly -- SAQ dispatches by this name, with no validation across the repos.

    Requires an Anthropic credential in THIS process's environment whenever
    `audit` is set (Alpha always sets it): `extract_claims` fails closed with
    `ProseCredentialMissing` before any parsing otherwise -- see the deploy note.
    """
    # extract_claims requires run_id/correlation_id (keyword-only, no defaults).
    # Alpha does not send them in the enqueue payload today, so default to fresh
    # ids. Preferred: Alpha sends its analysis_run id as run_id so ingested
    # claims correlate to the run, and that value is used instead; until then a
    # placeholder is fine -- Alpha re-associates claims to the run at ingest.
    run_id = run_id or str(uuid4())
    correlation_id = correlation_id or str(uuid4())

    client = build_spaces_client(parser_settings)
    if client is None:
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

    # known_sha256s is deliberately NOT forwarded: it is parse_pdf_bytes's
    # duplicate-REJECTION list, not an extract_claims parameter (extract_claims
    # has no such argument). Alpha always sends None and deal-level dedup already
    # happened at presign, so there is nothing to forward. It is accepted in the
    # signature only to match the enqueue payload's keys.
    try:
        # extract_claims is blocking (docling + a thread pool of Anthropic
        # calls), same as parse_bytes -- offload it so the event loop is free.
        payload = await asyncio.to_thread(
            extract_claims,
            data,
            entity=entity,
            run_id=run_id,
            correlation_id=correlation_id,
            source_file=spaces_key,
            audit=audit,
            # The deal flow needs the full claim graph, not just table claims.
            # `prose` adds the second extraction tier: the E1 reducer only emits
            # same_fact/contradicts edges when a fact is seen by >=2 tiers
            # (extract_service.py:376,415), so a table-only run yields zero edges
            # by construction -- and claim_type="computational" is set only on
            # prose-tier claims (emit.py:950), which the backend consistency pass
            # (SIM-372) routes on. `canonicalize_attributes` maps attributes onto
            # the canonical vocabulary that both the consistency rules and the
            # reconciliation grouping (SIM-371) key off. Both call the Anthropic
            # API -- already a precondition here since `audit` is on -- and both
            # fail soft (prose_failed / the canonicalize try-except in
            # extract_service.py), so they cannot regress the table claims that
            # already ingest today.
            prose=True,
            canonicalize_attributes=True,
            # Path B (mandate-fit screening): classify the target's sector + HQ
            # from the parsed prose so the backend can map them onto the org's
            # mandate and screen gs_07/gs_08. Hard-forced on for the deal flow like
            # prose/canonicalize above -- the deal flow always needs it; it reuses
            # the same Anthropic credential and fails soft (deal_profile is None on
            # any classification error), so it cannot regress the claims that
            # already ingest today. The org's approved options (when Alpha sends
            # them) let the classifier judge mandate fit against the same expanded
            # lists gs_07/gs_08 check; None -> raw sector/HQ only, no fit.
            deal_profile=True,
            sector_options=sector_options,
            geo_options=geo_options,
        )
    except ProseCredentialMissing:
        # A deployment/config problem, not a bad document: fail the SAQ job
        # outright rather than returning a soft "rejected". Every call fails the
        # same way until the worker env has a credential, and that should surface
        # as a job failure, not as per-document rejections producing no claims.
        logger.error("process_document: no Anthropic credential in worker env (audit=%s)", audit)
        raise
    except ParseError as exc:
        logger.warning(
            "process_document rejected: key=%s code=%s: %s", spaces_key, exc.code, exc.message
        )
        return {"status": "rejected", "code": exc.code, "message": exc.message}

    results_key = f"{parser_settings.results_key_prefix.rstrip('/')}/{payload['sha256']}.json"
    client.put_object(
        Bucket=parser_settings.spaces_bucket,
        Key=results_key,
        Body=json.dumps(payload).encode(),
        ContentType="application/json",
        ServerSideEncryption="AES256",
    )

    return {
        "status": "parsed",
        "bucket": parser_settings.spaces_bucket,
        "key": results_key,
        "sha256": payload["sha256"],
        "count": len(payload["claims"]),
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
    if getattr(job, "function", None) == "process_document":
        # parse + extraction + binding audit: docling plus per-claim Anthropic
        # calls -- a different latency/failure profile than parse alone, so it
        # gets its own numbers rather than inheriting parse_document's. Starting
        # points, not measured -- revisit with real latency data. retries=1 (not
        # 2): extract_claims hits the paid, non-deterministic Anthropic API, so
        # bound whole-job re-runs.
        job.timeout = 3600
        job.retries = 1
    else:
        # parse_document: docling only, CPU-bound.
        job.timeout = 1800
        job.retries = 2
    job.ttl = 86400
    await job.update()


# Neither docling nor the underlying torch/transformers stack release memory
# back to the OS once allocated (torch's CPU caching allocator keeps freed
# tensors pooled for reuse rather than returning them) -- confirmed on the
# staging droplet 2026-08-12: memory climbed job over job (721Mi -> 867Mi
# after one successful parse, -> 1.6Gi partway into a second, failed one)
# and never dropped back down between jobs, even after the failed one.
# There is no in-process way to force that allocator to shrink, so the
# worker recycles its own process after every completed job instead --
# `restart: unless-stopped` in docker-compose.prod.yml brings it back up
# fresh, paying a model-reload cost per job in exchange for never carrying
# accumulated memory into the next one. Revisit the threshold (currently 1,
# i.e. every job) once real usage data says less-frequent recycling is safe.
_RECYCLE_AFTER_N_JOBS = 1
_jobs_completed_this_process = 0


async def _recycle_worker(ctx: Context) -> None:
    """SAQ after_process hook -- runs in a `finally` block, so it fires for
    every job regardless of success, failure, or cancellation (see
    saq.worker.Worker.process). Deliberately os._exit, not sys.exit: this
    must be an immediate, unconditional process termination, not something
    an enclosing `except Exception` (SAQ wraps this hook's own call in one)
    could swallow or log as "Failed to run after process hook" instead of
    actually exiting.
    """
    global _jobs_completed_this_process
    _jobs_completed_this_process += 1
    if _jobs_completed_this_process >= _RECYCLE_AFTER_N_JOBS:
        logger.info(
            "Recycling worker process after %d job(s) to release accumulated memory",
            _jobs_completed_this_process,
        )
        os._exit(0)


settings: SettingsDict = {
    "queue": queue,
    "functions": [parse_document, process_document],
    "before_process": _normalize_job_policy,
    "after_process": _recycle_worker,
    "concurrency": 1,
}

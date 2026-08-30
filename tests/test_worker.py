"""parser_service.worker -- the SAQ consumer side of the shared "parse" queue.

No live Valkey/Redis connection anywhere in this file: Queue.from_url only
parses the URL, it doesn't connect eagerly, so the dummy PARSER_VALKEY_URL
set below is never actually dialed.
"""

import io
import os
from typing import cast

from parser_service import config

# worker.py reads settings (and raises at import time if valkey_url is unset)
# at module import, so the env var must be set and the settings cache cleared
# before `import parser_service.worker` below -- the autouse conftest fixture
# runs per-test, too late for a module-level import during collection.
os.environ.setdefault("PARSER_VALKEY_URL", "redis://localhost:6379/0")
config.get_settings.cache_clear()

import pytest  # noqa: E402
from reportlab.pdfgen import canvas  # noqa: E402
from saq.types import Context  # noqa: E402

from parser_service import worker  # noqa: E402

# Context requires a "worker" key that's meaningless outside a running SAQ
# worker; the functions under test never touch it, so an empty/partial dict
# cast to Context (rather than a real saq.worker.Worker) keeps these tests
# free of any live SAQ machinery.
_CTX = cast(Context, {})


def _minimal_pdf_bytes() -> bytes:
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer)
    pdf.drawString(72, 720, "worker test")
    pdf.showPage()
    pdf.save()
    return buffer.getvalue()


class _StubS3:
    """Records get_object/put_object calls; ignores Bucket (may be None in
    tests since Spaces isn't provisioned) and indexes objects by Key only."""

    def __init__(self, objects: dict[str, bytes]) -> None:
        self.objects = objects
        self.get_calls: list[str] = []
        self.put_calls: list[dict] = []
        self.get_error: Exception | None = None

    def get_object(self, Bucket: str | None = None, Key: str = ""):  # noqa: N803 -- matches boto3's signature
        self.get_calls.append(Key)
        if self.get_error is not None:
            raise self.get_error
        return {"Body": io.BytesIO(self.objects[Key])}

    def put_object(self, **kwargs) -> None:
        self.put_calls.append(kwargs)


def test_queue_name_and_task_registration_are_pinned() -> None:
    # Mirrors Alpha's tests/test_parse_client.py contract pin, opposite side:
    # a mismatch here means jobs are enqueued and never picked up, with no
    # error on either side.
    assert worker.queue.name == "parse"
    assert worker.settings["functions"] == [worker.parse_document, worker.process_document]
    assert worker.parse_document.__name__ == "parse_document"
    assert worker.process_document.__name__ == "process_document"


def test_after_process_hook_is_registered() -> None:
    # Without this, the worker never recycles and accumulated torch/docling
    # memory (confirmed 2026-08-12: never released between jobs) just keeps
    # growing until something eventually fails under memory pressure.
    assert worker.settings.get("after_process") == worker._recycle_worker


async def test_recycle_worker_exits_once_the_threshold_is_reached(monkeypatch) -> None:
    monkeypatch.setattr(worker, "_jobs_completed_this_process", 0)
    monkeypatch.setattr(worker, "_RECYCLE_AFTER_N_JOBS", 1)
    exit_calls: list[int] = []
    monkeypatch.setattr(worker.os, "_exit", lambda code: exit_calls.append(code))

    await worker._recycle_worker(_CTX)

    assert exit_calls == [0]


async def test_recycle_worker_waits_for_the_configured_count(monkeypatch) -> None:
    monkeypatch.setattr(worker, "_jobs_completed_this_process", 0)
    monkeypatch.setattr(worker, "_RECYCLE_AFTER_N_JOBS", 2)
    exit_calls: list[int] = []
    monkeypatch.setattr(worker.os, "_exit", lambda code: exit_calls.append(code))

    await worker._recycle_worker(_CTX)  # job 1 of 2 -- must not exit yet
    assert exit_calls == []

    await worker._recycle_worker(_CTX)  # job 2 of 2 -- now it should
    assert exit_calls == [0]


async def test_parse_document_happy_path_writes_pointer_not_raw_data(monkeypatch) -> None:
    pdf_bytes = _minimal_pdf_bytes()
    stub = _StubS3({"some/key.pdf": pdf_bytes})
    monkeypatch.setattr(worker, "build_spaces_client", lambda settings: stub)

    result = await worker.parse_document(ctx=_CTX, spaces_key="some/key.pdf")

    assert result["status"] == "parsed"
    assert result["key"].endswith(".json")
    assert stub.get_calls == ["some/key.pdf"]
    assert len(stub.put_calls) == 1
    assert stub.put_calls[0]["Key"] == result["key"]
    assert stub.put_calls[0]["ServerSideEncryption"] == "AES256"
    # Pointer shape only -- no raw parsed content back through the queue.
    assert "pages" not in result
    assert "sheets" not in result
    assert "paragraphs" not in result


async def test_parse_document_rejection_does_not_write_a_result(monkeypatch) -> None:
    # Empty bytes hit dispatch.parse_bytes' zero_byte_pdf rejection -- the
    # same case test_pdf_parser.py's test_zero_byte_pdf_is_rejected exercises
    # via the HTTP route.
    stub = _StubS3({"empty/key.pdf": b""})
    monkeypatch.setattr(worker, "build_spaces_client", lambda settings: stub)

    result = await worker.parse_document(ctx=_CTX, spaces_key="empty/key.pdf")

    assert result["status"] == "rejected"
    assert result["code"] == "zero_byte_pdf"
    assert stub.put_calls == []


async def test_parse_document_propagates_unexpected_spaces_errors(monkeypatch) -> None:
    stub = _StubS3({})
    stub.get_error = RuntimeError("spaces unavailable")
    monkeypatch.setattr(worker, "build_spaces_client", lambda settings: stub)

    with pytest.raises(RuntimeError, match="spaces unavailable"):
        await worker.parse_document(ctx=_CTX, spaces_key="some/key.pdf")


class _ExtractRecorder:
    """Stands in for extract_service.extract_claims: records the kwargs it was
    called with, and either returns a fixed payload or raises."""

    def __init__(self, payload: dict | None = None, raises: Exception | None = None) -> None:
        self.calls: list[dict] = []
        self._payload = payload
        self._raises = raises

    def __call__(self, data: bytes, **kwargs) -> dict:
        self.calls.append(kwargs)
        if self._raises is not None:
            raise self._raises
        assert self._payload is not None
        return self._payload


async def test_process_document_writes_pointer_and_supplies_run_id(monkeypatch) -> None:
    stub = _StubS3({"deal/doc.pdf": b"document-bytes"})
    monkeypatch.setattr(worker, "build_spaces_client", lambda settings: stub)
    rec = _ExtractRecorder(
        payload={
            "run_id": "ignored",
            "sha256": "abc123",
            "source_file": "deal/doc.pdf",
            "claims": [{}, {}, {}],
            "edges": [],
            "flag_log": [],
            "skipped_pages": [],
        }
    )
    monkeypatch.setattr(worker, "extract_claims", rec)

    result = await worker.process_document(ctx=_CTX, spaces_key="deal/doc.pdf", entity="Target Co")

    # Pointer written to Spaces, payload never inline (same contract as parse_document).
    assert result["status"] == "parsed"
    assert result["sha256"] == "abc123"
    assert result["count"] == 3
    assert result["key"].endswith("abc123.json")
    assert "claims" not in result
    assert len(stub.put_calls) == 1
    assert stub.put_calls[0]["Key"] == result["key"]
    assert stub.put_calls[0]["ServerSideEncryption"] == "AES256"

    call = rec.calls[0]
    assert call["entity"] == "Target Co"
    assert call["audit"] is True
    # The deal flow runs the FULL extraction so the downstream edge,
    # reconciliation, and consistency stages have a second tier, canonical
    # attributes, and claim_type to work with -- a table-only run emits zero
    # edges by construction (extract_service.py:376,415).
    assert call["prose"] is True
    # The qualitative assertion tier is on in the deal flow (market_definition,
    # competitive_position, ... -> the Market tab and other qualitative surfaces).
    assert call["qualitative"] is True
    assert call["canonicalize_attributes"] is True
    # extract_claims requires run_id/correlation_id; process_document supplies
    # both (fallback here, since _CTX carries no job to take a key from).
    assert call["run_id"] and call["correlation_id"]
    # known_sha256s must NOT be forwarded -- extract_claims has no such param.
    assert "known_sha256s" not in call


async def test_process_document_rejection_does_not_write_a_result(monkeypatch) -> None:
    stub = _StubS3({"deal/doc.pdf": b"document-bytes"})
    monkeypatch.setattr(worker, "build_spaces_client", lambda settings: stub)
    rec = _ExtractRecorder(raises=worker.ParseError("no_extractable_text", "needs OCR"))
    monkeypatch.setattr(worker, "extract_claims", rec)

    result = await worker.process_document(ctx=_CTX, spaces_key="deal/doc.pdf", entity="Target Co")

    assert result["status"] == "rejected"
    assert result["code"] == "no_extractable_text"
    assert stub.put_calls == []


async def test_process_document_missing_credential_raises_not_rejects(monkeypatch) -> None:
    # A missing Anthropic credential is a deploy/config failure, not a bad
    # document: it must raise (fail the job), never return a soft "rejected".
    stub = _StubS3({"deal/doc.pdf": b"document-bytes"})
    monkeypatch.setattr(worker, "build_spaces_client", lambda settings: stub)
    rec = _ExtractRecorder(raises=worker.ProseCredentialMissing("no Anthropic credential"))
    monkeypatch.setattr(worker, "extract_claims", rec)

    with pytest.raises(worker.ProseCredentialMissing):
        await worker.process_document(ctx=_CTX, spaces_key="deal/doc.pdf", entity="Target Co")

    assert stub.put_calls == []


async def test_normalize_job_policy_gives_process_document_its_own_numbers() -> None:
    class _StubJob:
        function = "process_document"

        def __init__(self) -> None:
            self.timeout = 10
            self.retries = 5
            self.ttl = 600
            self.updated = False

        async def update(self) -> None:
            self.updated = True

    job = _StubJob()
    await worker._normalize_job_policy(ctx=cast(Context, {"job": job}))

    assert job.timeout == 7200  # 2h: qualitative adds a second per-prose-page pass
    assert job.retries == 1  # bounded: extract_claims hits the paid Anthropic API
    assert job.ttl == 86400
    assert job.updated is True


async def test_normalize_job_policy_sets_timeout_retries_ttl_and_persists() -> None:
    class _StubJob:
        def __init__(self) -> None:
            self.timeout = 10
            self.retries = 1
            self.ttl = 600
            self.updated = False

        async def update(self) -> None:
            self.updated = True

    job = _StubJob()
    # _StubJob duck-types saq.job.Job's timeout/retries/ttl/update() surface
    # without depending on the real class; cast for the same reason as _CTX.
    await worker._normalize_job_policy(ctx=cast(Context, {"job": job}))

    assert job.timeout == 1800
    assert job.retries == 2
    assert job.ttl == 86400
    assert job.updated is True

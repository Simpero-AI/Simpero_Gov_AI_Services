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
    assert worker.settings["functions"] == [worker.parse_document]
    assert worker.parse_document.__name__ == "parse_document"


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

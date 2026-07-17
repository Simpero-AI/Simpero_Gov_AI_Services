import io

from parser_service.config import ParserSettings
from parser_service.document_cache import (
    NullDocumentCache,
    SpacesDocumentCache,
    build_document_cache,
)


class _FakeS3:
    def __init__(self) -> None:
        self.store: dict[tuple[str, str], bytes] = {}
        self.last_put: dict = {}
        self.fail = False

    def put_object(self, **kwargs) -> None:
        if self.fail:
            raise RuntimeError("spaces unavailable")
        self.last_put = kwargs
        self.store[(kwargs["Bucket"], kwargs["Key"])] = kwargs["Body"]

    def get_object(self, Bucket: str, Key: str) -> dict:
        if self.fail:
            raise RuntimeError("spaces unavailable")
        return {"Body": io.BytesIO(self.store[(Bucket, Key)])}


def test_cache_disabled_when_spaces_not_configured() -> None:
    cache = build_document_cache(ParserSettings())
    assert isinstance(cache, NullDocumentCache)
    assert cache.enabled is False
    assert cache.get_json("k") is None
    cache.put_json("k", {"a": 1})  # no-op, must not raise


def test_spaces_cache_round_trip_and_prefix() -> None:
    s3 = _FakeS3()
    cache = SpacesDocumentCache("bucket", "parser/document-cache", s3)
    cache.put_json("abc.json", {"pages": 3})
    assert ("bucket", "parser/document-cache/abc.json") in s3.store
    assert cache.get_json("abc.json") == {"pages": 3}


def test_spaces_cache_writes_server_side_encrypted() -> None:
    s3 = _FakeS3()
    SpacesDocumentCache("b", "", s3).put_json("k", {})
    assert s3.last_put["ServerSideEncryption"] == "AES256"
    assert s3.last_put["Key"] == "k"  # empty prefix -> no leading slash


def test_spaces_cache_is_fail_open() -> None:
    s3 = _FakeS3()
    s3.fail = True
    cache = SpacesDocumentCache("b", "p", s3)
    cache.put_json("k", {"a": 1})  # storage error must not propagate
    assert cache.get_json("k") is None


def test_spaces_configured_requires_all_credentials() -> None:
    partial = ParserSettings(spaces_bucket="b", spaces_endpoint_url="https://x")
    assert partial.spaces_configured is False
    full = ParserSettings(
        spaces_bucket="b",
        spaces_endpoint_url="https://x",
        spaces_access_key_id="k",
        spaces_secret_access_key="s",
    )
    assert full.spaces_configured is True


def test_build_selects_spaces_when_configured(monkeypatch) -> None:
    import boto3

    monkeypatch.setattr(boto3, "client", lambda *a, **k: _FakeS3())
    settings = ParserSettings(
        spaces_bucket="b",
        spaces_region="nyc3",
        spaces_endpoint_url="https://nyc3.digitaloceanspaces.com",
        spaces_access_key_id="k",
        spaces_secret_access_key="s",
    )
    assert isinstance(build_document_cache(settings), SpacesDocumentCache)

"""Cache for the raw DoclingDocument that DS-W3-2/DS-W3-6 consume.

Backed by object storage (DigitalOcean Spaces, S3-compatible): parsed CIM
content is encrypted at rest at the infrastructure level and the cache is shared
across parse workers. Until Spaces is provisioned the cache is disabled — the
parser re-derives structure on demand rather than writing confidential content
to unencrypted local disk.

The cache is best-effort: object-storage errors are logged and swallowed, so a
failed read or write never fails a parse.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from typing import Any, Protocol

from .config import ParserSettings, get_settings

logger = logging.getLogger(__name__)


class DocumentCache(Protocol):
    enabled: bool

    def get_json(self, key: str) -> dict[str, Any] | None: ...

    def put_json(self, key: str, document: dict[str, Any]) -> None: ...


class NullDocumentCache:
    """No-op cache used until object storage is configured."""

    enabled = False

    def get_json(self, key: str) -> dict[str, Any] | None:
        return None

    def put_json(self, key: str, document: dict[str, Any]) -> None:
        return None


class SpacesDocumentCache:
    """S3-compatible object-storage cache (DigitalOcean Spaces).

    Objects are written with server-side encryption; Spaces also encrypts at rest
    at the bucket level. Keys are content-addressed under a configurable prefix.
    All storage errors are non-fatal.
    """

    enabled = True

    def __init__(self, bucket: str, key_prefix: str, client: Any) -> None:
        self._bucket = bucket
        self._prefix = key_prefix.strip("/")
        self._client = client

    def _object_key(self, key: str) -> str:
        return f"{self._prefix}/{key}" if self._prefix else key

    def get_json(self, key: str) -> dict[str, Any] | None:
        try:
            obj = self._client.get_object(Bucket=self._bucket, Key=self._object_key(key))
            return json.loads(obj["Body"].read())
        except Exception as exc:  # noqa: BLE001 -- cache reads are best-effort
            logger.warning("document cache read failed (%s): %s", key, exc)
            return None

    def put_json(self, key: str, document: dict[str, Any]) -> None:
        try:
            self._client.put_object(
                Bucket=self._bucket,
                Key=self._object_key(key),
                Body=json.dumps(document).encode(),
                ContentType="application/json",
                ServerSideEncryption="AES256",
            )
        except Exception as exc:  # noqa: BLE001 -- cache writes are best-effort
            logger.warning("document cache write failed (%s): %s", key, exc)


def build_document_cache(settings: ParserSettings) -> DocumentCache:
    bucket = settings.spaces_bucket
    endpoint = settings.spaces_endpoint_url
    access_key = settings.spaces_access_key_id
    secret_key = settings.spaces_secret_access_key
    if not (bucket and endpoint and access_key and secret_key):
        return NullDocumentCache()
    import boto3

    client = boto3.client(
        "s3",
        region_name=settings.spaces_region,
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )
    return SpacesDocumentCache(bucket, settings.spaces_key_prefix, client)


@lru_cache
def get_document_cache() -> DocumentCache:
    return build_document_cache(get_settings())

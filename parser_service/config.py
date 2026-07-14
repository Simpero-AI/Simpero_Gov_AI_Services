from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

# This service is deployed independently of the main app (its own Dockerfile,
# its own process/port) so it gets its own small settings module rather than
# importing app.config — the two are not meant to share a runtime.


class ParserSettings(BaseSettings):
    # Upper bound on pages accepted per document. Docling runs ML layout/table
    # models per page, so this is primarily a cost/latency guard against
    # accidentally-huge uploads, not a hard technical ceiling.
    max_pages: int = 110

    # Object storage (DigitalOcean Spaces, S3-compatible) for the raw
    # DoclingDocument cache that DS-W3-2/DS-W3-6 consume. Spaces encrypts at rest
    # at the bucket level, so confidential content is never persisted unencrypted
    # at the application level, and the cache is durable and shared across parse
    # workers. Unset until Spaces is provisioned; while unset the cache is
    # disabled (see document_cache.build_document_cache) and nothing is written
    # to local disk. Credentials come from the environment (PARSER_SPACES_*),
    # never committed.
    spaces_bucket: str | None = None
    spaces_region: str | None = None
    spaces_endpoint_url: str | None = None
    spaces_access_key_id: str | None = None
    spaces_secret_access_key: str | None = None
    # Content-addressed under this prefix. Org-scoping (a per-tenant prefix) is a
    # follow-up: parse_pdf_bytes has no org context yet, and sha256 keys are
    # shared across tenants until it does.
    spaces_key_prefix: str = "parser/document-cache"

    @property
    def spaces_configured(self) -> bool:
        return bool(
            self.spaces_bucket
            and self.spaces_endpoint_url
            and self.spaces_access_key_id
            and self.spaces_secret_access_key
        )

    model_config = SettingsConfigDict(
        env_prefix="PARSER_",
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> ParserSettings:
    # lru_cache: settings are immutable at runtime; re-reading env on every
    # call is wasteful.
    return ParserSettings()

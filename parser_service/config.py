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

    # Upper bound on sheets accepted per XLSX workbook (DS-W3-5). openpyxl has
    # no ML cost like Docling's per-page models, but this is still a guard
    # against an accidentally-huge upload (some models carry dozens of
    # scenario/sensitivity sheets) turning into an unbounded parse.
    max_sheets: int = 50

    # Upper bound on paragraphs accepted per DOCX. The flow-document analog of
    # max_pages: a guard against an accidentally-huge upload, not a hard ceiling.
    max_paragraphs: int = 20_000

    # Upper bound on an OOXML file's total UNCOMPRESSED size, shared by the XLSX
    # and DOCX lanes (both are a zip of XML, both get expanded whole into memory,
    # so a decompression bomb — a few KB on disk expanding to gigabytes — would
    # OOM the worker either way). Checked from the zip directory's declared sizes
    # before any parse. 500 MB clears any real model or CIM with wide headroom.
    max_ooxml_uncompressed_bytes: int = 500_000_000

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

    # Shared-secret checked against the X-Parser-Key header on POST /parse.
    # Unset means "refuse to serve" (fail-closed), not "auth disabled" -- see
    # main.py's parse_auth dependency.
    api_key: str | None = None

    # DigitalOcean Managed Valkey URL for the SAQ parse-job queue (worker.py).
    # Unset means the worker refuses to start (fail-closed) -- see worker.py's
    # module-level check, same reasoning as api_key above.
    valkey_url: str | None = None

    # Cross-repo contract: Simpero_AI_Gov_Alpha's app/jobs/parse_client.py
    # enqueues onto a queue named exactly "parse" (its own PARSE_QUEUE_NAME
    # constant, guarded by a unit test on that side). Do not rename without
    # coordinating both repos.
    queue_name: str = "parse"

    # Where worker.py writes parse results in Spaces, as bucket+key pointers.
    # Per-environment override expected (e.g. "parser/parse-results/staging"),
    # same pattern as spaces_key_prefix above.
    results_key_prefix: str = "parser/parse-results"

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

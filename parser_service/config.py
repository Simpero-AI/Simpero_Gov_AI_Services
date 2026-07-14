from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# This service is deployed independently of the main app (its own Dockerfile,
# its own process/port) so it gets its own small settings module rather than
# importing app.config — the two are not meant to share a runtime.


class ParserSettings(BaseSettings):
    # Upper bound on pages accepted per document. Docling runs ML layout/table
    # models per page, so this is primarily a cost/latency guard against
    # accidentally-huge uploads, not a hard technical ceiling.
    max_pages: int = 110

    # Root directory for the cached raw DoclingDocument JSON (kept for
    # DS-W3-2/DS-W3-6, which need the full structure, not just the flat
    # PageIndex). Must be an absolute path — the service's working directory
    # is not guaranteed by uvicorn or by container orchestration, so a
    # relative path here would silently write wherever the process happened
    # to be launched from.
    #
    # Defaults under /tmp, matching this service's existing HF_HOME/TORCH_HOME
    # convention (Dockerfile) — guaranteed writable without special permissions
    # in any container or CI runner, unlike e.g. /var/lib, which typically
    # requires root. /tmp is not persistent across container restarts; a real
    # deployment that needs the cache to survive restarts should override
    # PARSER_CACHE_DIR to a mounted, persistent volume.
    #
    # SECURITY NOTE: parsed CIM/financial documents are written to this cache
    # unencrypted. This service does not implement application-level
    # encryption-at-rest, because doing so safely requires a defined key-
    # management story (rotation, access control) that doesn't exist yet —
    # bolting on ad-hoc encryption without one would be a false sense of
    # security, not a real fix. Until that exists, this cache must live on
    # storage that is encrypted at rest at the infrastructure/volume level,
    # and access to the host/container must be restricted accordingly. If
    # that guarantee can't be met in a given deployment, disable the cache
    # entirely via `enable_cache=False` rather than accept unencrypted
    # confidential documents on disk.
    cache_dir: Path = Path("/tmp/simpero-parser/cache")

    # Escape hatch for environments that can't guarantee the cache_dir's
    # storage is encrypted at rest (see SECURITY NOTE above). DS-W3-2/DS-W3-6
    # depend on this cache, so disabling it is a real trade-off, not free.
    enable_cache: bool = True

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

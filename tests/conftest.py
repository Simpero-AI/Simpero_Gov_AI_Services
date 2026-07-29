from collections.abc import Iterator

import pytest

from parser_service import config

TEST_PARSER_API_KEY = "test-parser-key"
# Dummy, never actually connected to (Queue.from_url only parses the URL).
# Set globally so any test importing parser_service.worker sees a configured
# worker rather than its import-time fail-closed RuntimeError.
TEST_PARSER_VALKEY_URL = "redis://localhost:6379/0"


@pytest.fixture(autouse=True)
def _parser_api_key(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    # Every test gets a valid PARSER_API_KEY by default so the ~15 existing
    # client.post("/parse", ...) call sites (which rely on TestClient's default
    # headers) keep working unmodified. test_auth.py overrides this per-test to
    # exercise the unset/mismatched/absent cases. monkeypatch.setenv reverts
    # automatically; the explicit cache_clear() calls are still needed because
    # get_settings is an lru_cache that would otherwise leak across tests.
    monkeypatch.setenv("PARSER_API_KEY", TEST_PARSER_API_KEY)
    monkeypatch.setenv("PARSER_VALKEY_URL", TEST_PARSER_VALKEY_URL)
    config.get_settings.cache_clear()
    yield
    config.get_settings.cache_clear()

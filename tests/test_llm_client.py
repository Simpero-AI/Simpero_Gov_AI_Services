"""The Anthropic client factory. Every model call in the parser is built here so
they share one retry/timeout posture: the SDK's own backoff absorbs a fan-out
burst of 429/5xx/timeout instead of a page's call exhausting the default 2
retries and being dropped (the fewer-claims-but-faster regression)."""

from __future__ import annotations

import anthropic

import parser_service.llm_client as llm_client


def test_make_client_widens_the_retry_and_timeout_budget(monkeypatch) -> None:
    captured: dict = {}

    class _FakeAnthropic:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(anthropic, "Anthropic", _FakeAnthropic)

    llm_client.make_client()

    # Comfortably above the SDK default of 2, and a real timeout -- the two knobs
    # that let the SDK backoff soak up rate-limit pressure under the fan-out.
    assert captured["max_retries"] == llm_client._MAX_RETRIES
    assert captured["max_retries"] > 2
    assert captured["timeout"] == llm_client._TIMEOUT_S
    assert captured["timeout"] > 0

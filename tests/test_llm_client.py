"""The Anthropic client factory. Every model call in the parser is built here so
they share one retry/timeout posture: the SDK's own backoff absorbs a fan-out
burst of 429/5xx/timeout instead of a page's call exhausting the default 2
retries and being dropped (the fewer-claims-but-faster regression)."""

from __future__ import annotations

import importlib

import anthropic

import parser_service.llm_client as llm_client


def test_make_client_widens_the_retry_and_read_budget_but_keeps_connect_fast(monkeypatch) -> None:
    captured: dict = {}

    class _FakeAnthropic:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(anthropic, "Anthropic", _FakeAnthropic)

    llm_client.make_client()

    # Comfortably above the SDK default of 2 -- the knob that lets the SDK backoff
    # soak up rate-limit pressure under the fan-out.
    assert captured["max_retries"] == llm_client._MAX_RETRIES
    assert captured["max_retries"] > 2
    # Read widened, connect kept fast so a dead endpoint fails quick, not in ~10 min.
    assert captured["timeout"].read == llm_client._TIMEOUT_S
    assert captured["timeout"].connect == 5.0


def test_retry_and_timeout_budgets_are_env_tunable(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_MAX_RETRIES", "3")
    monkeypatch.setenv("ANTHROPIC_TIMEOUT_S", "42")
    reloaded = importlib.reload(llm_client)
    try:
        assert reloaded._MAX_RETRIES == 3
        assert reloaded._TIMEOUT_S == 42.0
    finally:
        # Restore module-level defaults so later tests see the unpatched values.
        monkeypatch.delenv("ANTHROPIC_MAX_RETRIES", raising=False)
        monkeypatch.delenv("ANTHROPIC_TIMEOUT_S", raising=False)
        importlib.reload(llm_client)

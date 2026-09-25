"""The Anthropic client factory. Every model call in the parser is built here so
they share one retry/timeout posture: the SDK's own backoff absorbs a fan-out
burst of 429/5xx/timeout instead of a page's call exhausting the default 2
retries and being dropped (the fewer-claims-but-faster regression)."""

from __future__ import annotations

import importlib

import anthropic
import pytest
from pydantic import BaseModel, ValidationError

import parser_service.llm_client as llm_client
from parser_service.llm_client import (
    _GRAMMAR_RETRIES,
    is_credit_exhausted,
    is_grammar_timeout,
    is_usage_limited,
    parse_with_retry,
)


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
    # Only `read` gets the long budget (the model's generation time). connect,
    # write, and pool stay short so a dead endpoint, a stalled upload, or a wait for
    # a pooled connection fails fast (and the SDK retries), not in ~10 min.
    assert captured["timeout"].read == llm_client._TIMEOUT_S
    assert captured["timeout"].connect == 5.0
    assert captured["timeout"].write == 30.0
    assert captured["timeout"].pool == 10.0


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


class _RetryModel(BaseModel):
    required_field: int


class _Grammar400(Exception):
    """Stand-in for the SDK's BadRequestError: a 400 (status_code=400) whose
    message names a grammar-compilation timeout. is_grammar_timeout gates on both
    the 400 status and the "grammar" substring, so the stub carries both."""

    status_code = 400


class _BadRequest400(Exception):
    """Stand-in for the SDK's BadRequestError on a bare invalid_request_error: a
    400 that is not billing, not usage-cap, and not grammar (its message never
    says "grammar"). This is the "Invalid request data" transient."""

    status_code = 400


def test_is_grammar_timeout_gates_on_both_the_400_status_and_the_message() -> None:
    # A transient grammar-compilation 400 -> ours.
    assert is_grammar_timeout(_Grammar400("Grammar compilation timed out"))
    # "grammar" without a 400 status is NOT a retryable grammar timeout -- a
    # non-400 error that merely mentions grammar must not be retried.
    assert not is_grammar_timeout(Exception("something went wrong with the grammar"))
    # A 400 that is not about grammar (a real bad request) is not ours either.
    assert not is_grammar_timeout(_Grammar400("invalid request: unknown field"))
    assert not is_grammar_timeout(Exception("rate limit exceeded"))


def test_parse_with_retry_narrows_a_transient_grammar_timeout(monkeypatch) -> None:
    # A 400 the SDK will not retry (400s are non-retryable) is retried in place.
    monkeypatch.setattr("parser_service.llm_client.time.sleep", lambda _s: None)
    calls = {"n": 0}

    def call():
        calls["n"] += 1
        if calls["n"] == 1:
            raise _Grammar400("Grammar compilation timed out")
        return "ok"

    assert parse_with_retry(call, page_no=1, what="numeric proposal") == "ok"
    assert calls["n"] == 2


def test_parse_with_retry_gives_up_after_the_grammar_budget(monkeypatch) -> None:
    monkeypatch.setattr("parser_service.llm_client.time.sleep", lambda _s: None)
    calls = {"n": 0}

    def call():
        calls["n"] += 1
        raise _Grammar400("Grammar compilation timed out")

    with pytest.raises(_Grammar400):
        parse_with_retry(call, page_no=1, what="numeric proposal")
    # one initial attempt plus _GRAMMAR_RETRIES retries, then the caller sees it.
    assert calls["n"] == _GRAMMAR_RETRIES + 1


def test_parse_with_retry_narrows_a_transient_bad_request_400(monkeypatch) -> None:
    # A bare "Invalid request data" 400 (not billing/usage/grammar) has been seen
    # recovering on an identical re-run, so it is retried once.
    monkeypatch.setattr("parser_service.llm_client.time.sleep", lambda _s: None)
    calls = {"n": 0}

    def call():
        calls["n"] += 1
        if calls["n"] == 1:
            raise _BadRequest400("Invalid request data")
        return "ok"

    assert parse_with_retry(call, page_no=1, what="numeric proposal") == "ok"
    assert calls["n"] == 2


def test_parse_with_retry_gives_up_on_a_persistent_bad_request_after_the_budget(
    monkeypatch,
) -> None:
    # A genuinely permanent 400 exhausts the retry budget and then surfaces --
    # bounded to the same fan-out cost the grammar path accepts.
    monkeypatch.setattr("parser_service.llm_client.time.sleep", lambda _s: None)
    calls = {"n": 0}

    def call():
        calls["n"] += 1
        raise _BadRequest400("Invalid request data")

    with pytest.raises(_BadRequest400):
        parse_with_retry(call, page_no=1, what="numeric proposal")
    assert calls["n"] == _GRAMMAR_RETRIES + 1


class _LowBalance400(Exception):
    """Stand-in for the SDK's BadRequestError on a depleted balance: a 400 whose
    message is Anthropic's 'credit balance is too low' invalid_request error."""

    status_code = 400


class _Payment402(Exception):
    status_code = 402


class _UsageLimit400(Exception):
    """Stand-in for the SDK's BadRequestError on a hit usage/spend cap: a 400
    whose message is Anthropic's 'reached your specified API usage limits'
    invalid_request error (distinct from a depleted balance)."""

    status_code = 400


def test_is_credit_exhausted_matches_low_balance_or_402_but_not_transients() -> None:
    assert is_credit_exhausted(
        _LowBalance400("Your credit balance is too low to access the Anthropic API.")
    )
    assert is_credit_exhausted(_Payment402("payment required"))
    # NARROW: a transient rate limit or a grammar-compilation 400 is not exhaustion
    # and must not be misread as one (that would turn a retryable/transient into a
    # hard, run-failing error).
    assert not is_credit_exhausted(Exception("rate limit exceeded"))
    assert not is_credit_exhausted(_Grammar400("Grammar compilation timed out"))


def test_parse_with_retry_fails_loud_on_credit_exhaustion(monkeypatch) -> None:
    # A depleted balance must raise AnthropicCreditExhausted (loud), never be
    # narrowed/retried or passed through as a raw error a caller would swallow
    # into a skipped page (the silent "no financials" bug).
    monkeypatch.setattr("parser_service.llm_client.time.sleep", lambda _s: None)
    calls = {"n": 0}

    def call():
        calls["n"] += 1
        raise _LowBalance400("Your credit balance is too low to access the Anthropic API.")

    # Reference the class via the module: an earlier test reloads llm_client, so
    # parse_with_retry raises whatever AnthropicCreditExhausted currently lives in
    # the module dict -- match against that same object, not the import binding.
    with pytest.raises(llm_client.AnthropicCreditExhausted):
        parse_with_retry(call, page_no=1, what="numeric proposal")
    assert calls["n"] == 1  # not retried -- exhaustion is non-transient


def test_is_usage_limited_matches_the_spend_cap_message_but_not_transients() -> None:
    assert is_usage_limited(
        _UsageLimit400(
            "You have reached your specified API usage limits. You will regain "
            "access on 2026-10-01 at 00:00 UTC."
        )
    )
    # NARROW: neither a depleted balance nor a transient rate limit is a usage cap
    # (each has its own handling), and must not be misread as one.
    assert not is_usage_limited(
        _LowBalance400("Your credit balance is too low to access the Anthropic API.")
    )
    assert not is_usage_limited(Exception("rate limit exceeded"))
    assert not is_usage_limited(_Grammar400("Grammar compilation timed out"))


def test_parse_with_retry_fails_loud_on_a_usage_cap(monkeypatch) -> None:
    # A hit usage/spend cap blocks every call until an admin raises it, so it must
    # raise AnthropicCreditExhausted (loud) like a depleted balance -- never be
    # narrowed/retried or swallowed into a skipped page (the silent-empty bug this
    # closes: a capped account previously "succeeded" with only docling claims).
    monkeypatch.setattr("parser_service.llm_client.time.sleep", lambda _s: None)
    calls = {"n": 0}

    def call():
        calls["n"] += 1
        raise _UsageLimit400(
            "You have reached your specified API usage limits. You will regain "
            "access on 2026-10-01 at 00:00 UTC."
        )

    with pytest.raises(llm_client.AnthropicCreditExhausted):
        parse_with_retry(call, page_no=1, what="numeric proposal")
    assert calls["n"] == 1  # not retried -- a usage cap is non-transient


def test_parse_with_retry_reraises_a_non_transient_error(monkeypatch) -> None:
    # 429/5xx/timeout are the SDK's job (via make_client's max_retries); by the
    # time one reaches here its budget is spent, so it must propagate, not loop.
    monkeypatch.setattr("parser_service.llm_client.time.sleep", lambda _s: None)
    calls = {"n": 0}

    def call():
        calls["n"] += 1
        raise RuntimeError("rate limit exceeded")

    with pytest.raises(RuntimeError):
        parse_with_retry(call, page_no=1, what="numeric proposal")
    assert calls["n"] == 1


def test_parse_with_retry_still_narrows_a_malformed_body_once() -> None:
    calls = {"n": 0}

    def call():
        calls["n"] += 1
        if calls["n"] == 1:
            _RetryModel.model_validate({})  # raises ValidationError (missing field)
        return "ok"

    assert parse_with_retry(call, page_no=1, what="numeric proposal") == "ok"
    assert calls["n"] == 2


def test_parse_with_retry_reraises_a_body_malformed_twice() -> None:
    calls = {"n": 0}

    def call():
        calls["n"] += 1
        _RetryModel.model_validate({})

    with pytest.raises(ValidationError):
        parse_with_retry(call, page_no=1, what="numeric proposal")
    assert calls["n"] == 2  # exactly one retry, then the caller sees it


def test_parse_with_retry_budgets_grammar_and_validation_independently(monkeypatch) -> None:
    # The two transients interleave: a grammar timeout, then a malformed body, then
    # another grammar timeout, then success. Each has its own budget (1 validation
    # retry, _GRAMMAR_RETRIES grammar retries), so the combined path still resolves
    # rather than exhausting one budget on the other's failures.
    monkeypatch.setattr("parser_service.llm_client.time.sleep", lambda _s: None)
    scripted = [
        _Grammar400("Grammar compilation timed out"),
        None,  # placeholder -- a ValidationError is raised for this attempt below
        _Grammar400("Grammar compilation timed out"),
    ]
    calls = {"n": 0}

    def call():
        i = calls["n"]
        calls["n"] += 1
        if i == 1:
            _RetryModel.model_validate({})  # ValidationError on the 2nd attempt
        if i < len(scripted) and isinstance(scripted[i], Exception):
            raise scripted[i]
        return "ok"

    assert parse_with_retry(call, page_no=1, what="numeric proposal") == "ok"
    # grammar(0) -> validation(1) -> grammar(2) -> ok(3): 4 attempts, both budgets spent.
    assert calls["n"] == 4

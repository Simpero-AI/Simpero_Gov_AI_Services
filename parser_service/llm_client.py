"""One place to construct the Anthropic client, so every model call in the
parser inherits the same retry/timeout posture.

Why this exists: extraction fans out EXTRACT_WORKERS model calls at once (16 by
default since #55) to parallelize the prose tiers and the per-claim binding
audit. That burst routinely pushes past the account's rate limit, so individual
page calls hit 429/529/timeout. The SDK retries exactly those with exponential
backoff -- but its default of 2 retries is too little headroom for a 16-wide
burst: the retries exhaust, the error propagates, and extract_service._prose_tiers
catches it and drops that whole page's claims. The result is a silent partial
extraction -- fewer claims AND a faster run, because failed calls short-circuit
the full generation. Raising the retry ceiling lets the SDK's own backoff absorb
the transient pressure (self-throttling to whatever the rate limit allows)
instead of losing pages.

Grammar-compilation 400s are a separate transient the SDK will NOT retry (400s
are non-retryable); parse_with_retry (below) narrows those in place. Both the
prose proposal path and the binding-audit path call it, so neither drops a page
to a transient the other already handles.
"""

import logging
import os
import time

from pydantic import ValidationError

logger = logging.getLogger(__name__)

# The SDK already retries 429/500/overloaded/timeout/connection with exponential
# backoff; these only widen the budget so a fan-out burst is absorbed rather than
# dropped. Env-tunable so ops can match the account's rate-limit headroom without
# a redeploy, same posture as EXTRACT_WORKERS.
_MAX_RETRIES = int(os.getenv("ANTHROPIC_MAX_RETRIES", "8") or "8")
_TIMEOUT_S = float(os.getenv("ANTHROPIC_TIMEOUT_S", "600") or "600")

# A transient grammar-compilation timeout is retried in place, with linear
# backoff, before the caller sees a failure. Mirrors the sibling agent layer.
_GRAMMAR_RETRIES = 2
_GRAMMAR_BACKOFF_S = 2.0


def make_client():
    """A configured anthropic.Anthropic with the widened retry/timeout budget the
    fan-out needs. The `import anthropic` stays local for tidiness, not to save
    import cost -- propose.py, imported by every caller that reaches here, now
    imports anthropic at module scope, so it is already loaded by this point."""
    import anthropic
    import httpx

    # httpx.Timeout, not a bare float: a bare float sets EVERY phase (connect/read/
    # write/pool) to _TIMEOUT_S, so a hard-down endpoint, a stalled request-body
    # upload, or a wait for a free pooled connection would each hang the full read
    # budget (~10 min) per attempt before the retry/backoff could react. Only
    # `read` needs that budget -- it is the model's generation time; connect,
    # write, and pool stay short so those failures react fast and the client's
    # retries (max_retries above) kick in. `read` is left unset so it inherits the
    # _TIMEOUT_S positional default.
    return anthropic.Anthropic(
        max_retries=_MAX_RETRIES,
        timeout=httpx.Timeout(_TIMEOUT_S, connect=5.0, write=30.0, pool=10.0),
    )


def is_grammar_timeout(exc: Exception) -> bool:
    """Whether `exc` is a *transient* server-side grammar-compilation timeout.

    The structured-output path (`messages.parse`) compiles the JSON schema into a
    grammar server-side; under load the API occasionally returns a 400 naming a
    grammar-compilation timeout for a request that succeeds when re-run. A 400 is
    non-retryable, so the client's own max_retries backoff never fires on it --
    this is the one 4xx worth narrowing.

    Gated on the 400 status, not just the word "grammar": a PERMANENT
    grammar-compilation 400 (a genuinely invalid schema, e.g. after an
    output_format change) also carries "grammar", and matching on the substring
    alone would retry that deploy bug 3x per page across a 16-wide fan-out --
    turning a fail-fast into a slow burn. The status check keeps the narrow
    transient intent; a non-400 that merely mentions grammar is not ours."""
    if "grammar" not in str(exc).lower():
        return False
    # A BadRequestError always carries status_code == 400 (the SDK sets it in
    # APIStatusError.__init__), and getattr also picks up a class-attribute
    # status_code on a hand-rolled/mocked error, so this one check covers both --
    # a separate isinstance(BadRequestError) branch would be unreachable.
    return getattr(exc, "status_code", None) == 400


class AnthropicCreditExhausted(RuntimeError):
    """The Anthropic account cannot make calls for a billing/quota reason -- a
    present-but-unusable key. Two causes are handled identically (see
    is_credit_exhausted / is_usage_limited): a depleted CREDIT balance, and a
    configured USAGE/SPEND CAP the account has hit.

    Distinct from a MISSING key (ProseCredentialMissing, raised before any call)
    and from a transient 429/5xx (which the SDK retries): both are NON-retryable
    account-level errors that would otherwise be caught per page and recorded as a
    SkippedPage, silently yielding a partial/empty extraction with a clean 200 --
    a deal that shows "no financials" when the real cause is billing. Raising it
    lets extract_claims fail LOUD so the run is marked failed/degraded instead.
    Ops fix: top up credit, or raise/reset the usage limit in the Console (or wait
    for it to reset), then re-run."""


def is_credit_exhausted(exc: Exception) -> bool:
    """Whether `exc` is an Anthropic insufficient-credit error.

    Anthropic signals a depleted balance as a non-retryable 400 invalid_request
    error whose message is "Your credit balance is too low to access the
    Anthropic API ..." (some deployments use a 402 status). Matched on that
    stable phrase OR a 402 -- deliberately NARROW so a transient 429/5xx (retried
    by the SDK) or a grammar-compilation 400 is never misread as exhaustion and
    turned into a hard failure."""
    if getattr(exc, "status_code", None) == 402:
        return True
    return "credit balance is too low" in str(exc).lower()


def is_usage_limited(exc: Exception) -> bool:
    """Whether `exc` is an Anthropic account usage/spend-cap error.

    Distinct from a depleted balance (is_credit_exhausted): the account has credit
    but has hit a configured spend/usage limit, which Anthropic signals as a
    non-retryable 400 invalid_request error reading "You have reached your
    specified API usage limits. You will regain access on <date> ...". Like an
    empty balance it blocks EVERY call until an admin raises/resets the limit (or
    it resets on the boundary), so it must fail LOUD, not be swallowed per page
    into a silently-empty extraction. Matched on the stable distinctive phrase --
    deliberately NARROW, so a transient 429 rate limit ("rate limit exceeded") or
    another 400 is never misread as an account cap."""
    return "specified api usage limit" in str(exc).lower()


def parse_with_retry(call, *, page_no: int, what: str):
    """Run a structured-output call, narrowing two transient failures.

    A malformed/truncated body (ValidationError) is retried once: measured across
    full-document runs, a page occasionally comes back with an empty or truncated
    body whose parse fails but succeeds on an identical re-run, and left unhandled
    it destroys that page's whole extraction. One retry, not a loop -- malformed
    twice is a real failure the caller should see.

    A transient grammar-compilation timeout (see is_grammar_timeout) is retried
    up to _GRAMMAR_RETRIES times with linear backoff. 429/5xx/timeout are NOT
    handled here -- the client's own max_retries backoff covers those (see
    make_client), which is the fix for the fan-out page-loss.

    Every raise stays reachable: this narrows known transients, it never pretends
    a call succeeded.
    """
    validation_retried = False
    grammar_attempts = 0
    while True:
        try:
            return call()
        except ValidationError as exc:
            if validation_retried:
                raise
            validation_retried = True
            logger.warning(
                "page %s: %s returned an unparseable body (%s); retrying once",
                page_no,
                what,
                type(exc).__name__,
            )
        except Exception as exc:  # noqa: BLE001 -- re-raised unless it is a known transient
            # A billing/quota block (depleted balance OR a hit usage/spend cap) is
            # non-retryable and must fail LOUD, not be narrowed as a transient or
            # (upstream) swallowed into a skipped page. Both are 400s, so they are
            # checked before the grammar-timeout narrowing -- which also matches on
            # a 400 -- so neither can be mistaken for a grammar timeout.
            if is_credit_exhausted(exc):
                raise AnthropicCreditExhausted(
                    "Anthropic credit balance is exhausted; top up the account "
                    "(or raise the spend cap) and re-run."
                ) from exc
            if is_usage_limited(exc):
                raise AnthropicCreditExhausted(
                    "Anthropic usage/spend limit reached for this account; raise "
                    "or reset the limit in the Console (or wait for it to reset) "
                    "and re-run."
                ) from exc
            if not is_grammar_timeout(exc) or grammar_attempts >= _GRAMMAR_RETRIES:
                raise
            grammar_attempts += 1
            logger.warning(
                "page %s: %s hit a transient grammar-compilation timeout; retry %d/%d",
                page_no,
                what,
                grammar_attempts,
                _GRAMMAR_RETRIES,
            )
            time.sleep(_GRAMMAR_BACKOFF_S * grammar_attempts)

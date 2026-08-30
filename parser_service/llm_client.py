"""One place to construct the Anthropic client, so every model call in the
parser inherits the same retry/timeout posture.

Why this exists: extraction fans out EXTRACT_WORKERS model calls at once (16 by
default since #55) to parallelize the prose tiers and the per-claim binding
audit. That burst routinely pushes past the account's rate limit, so individual
page calls hit 429/529/timeout. The SDK retries exactly those with exponential
backoff -- but its default of 2 retries is too little headroom for a 16-wide
burst: the retries exhaust, the error propagates, and extract_service._prose_claims
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
    """A configured anthropic.Anthropic. `anthropic` is imported lazily (kept out
    of module import cost) exactly as the call sites this replaces did."""
    import anthropic
    import httpx

    # httpx.Timeout, not a bare float: a float sets every phase (connect/read/
    # write/pool) to _TIMEOUT_S, collapsing the SDK's 5s connect timeout to the
    # full read budget -- a hard-down endpoint would then hang ~10 min per attempt
    # before the retry/backoff could react. Widen only the read; keep connect fast.
    return anthropic.Anthropic(
        max_retries=_MAX_RETRIES,
        timeout=httpx.Timeout(_TIMEOUT_S, connect=5.0),
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
    if getattr(exc, "status_code", None) == 400:
        return True
    import anthropic

    return isinstance(exc, anthropic.BadRequestError)


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

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
are non-retryable); those are narrowed in propose._parse_with_retry, not here.
"""

import os

# The SDK already retries 429/500/overloaded/timeout/connection with exponential
# backoff; these only widen the budget so a fan-out burst is absorbed rather than
# dropped. Env-tunable so ops can match the account's rate-limit headroom without
# a redeploy, same posture as EXTRACT_WORKERS.
_MAX_RETRIES = int(os.getenv("ANTHROPIC_MAX_RETRIES", "8") or "8")
_TIMEOUT_S = float(os.getenv("ANTHROPIC_TIMEOUT_S", "600") or "600")


def make_client():
    """A configured anthropic.Anthropic. `anthropic` is imported lazily (kept out
    of module import cost) exactly as the call sites this replaces did."""
    import anthropic

    return anthropic.Anthropic(max_retries=_MAX_RETRIES, timeout=_TIMEOUT_S)

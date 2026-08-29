"""Binding audit -- an independent reader that checks a cited claim, and only flags.

Every other pass in this service PROPOSES; this one DISPOSES of nothing. It reads
a claim that already resolved to an exact span and asks a single question the
resolver structurally cannot: does that span, read on its own, actually justify
binding this (entity, attribute) to this value? The 80-claim audit that motivated
this put material binding errors at ~6% -- claims carrying a real span and a
scale-valid magnitude, so nothing downstream catches them.

WHAT THIS IS ALLOWED TO DO
==========================
Produce a verdict. Nothing else. It never emits a claim, never moves a span,
never supplies a magnitude (scale.py still owns that), and never writes a status
-- promoting a claim to `verified` would assert a cross-source check nobody ran,
and suppressing one is a `rejected` the parser deliberately cannot reach. The
verdict is advisory: a reviewer, or a later gated pass, decides what to do with it.

WHY A MODEL AND NOT A RULE
==========================
Two of the failure modes are semantic and a substring rule cannot see them: a
figure heading a LIST bound to its first noun ("116 universities, colleges,
councils..." -> "universities"), and a services rate-card line read as a good's
price. The other two are shape ("greater than X" stored as exact X) and sanity
(a regional casino with $52bn of revenue, from a scale header bleeding onto a
value it does not govern). The check is adversarial: assume the binding is wrong
until the span proves it, and quote the span text that settles it.

WHY IT IS STILL SAFE
====================
It reads the model's output with a DIFFERENT prompt and DIFFERENT framing, so it
is genuinely a second actor, not the proposer grading itself. And it must ground
its verdict in a quote from the span it was given -- a verdict it cannot support
from the span is not one it is allowed to return.
"""

from __future__ import annotations

import logging
import os
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from .llm_client import make_client

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-opus-4-8"

# The named ways a binding goes wrong, measured on real CIMs. "supported" is the
# pass. Anything else is a flag, never a mutation.
BindingVerdict = Literal[
    "supported",
    "basket_collapse",  # value counts/sums a LIST, bound to one member of it
    "entity_not_in_span",  # the entity names something the span does not
    "attribute_mismatch",  # the span asserts a different thing than the attribute
    "fee_vs_price",  # a services rate-card line read as a good's price
    "bound_as_point",  # a bound ("greater than X") stored as an exact value
    "scale_absurd",  # the magnitude is implausible for this entity/attribute
]


class AuditVerdict(BaseModel):
    """One binding audit result. Advisory. Never mutates the claim it judges."""

    verdict: BindingVerdict = Field(description="supported, or the named failure mode")
    evidence: str = Field(
        description="The span text that settles it, quoted verbatim. A verdict that "
        "cannot be grounded in the span is not allowed."
    )


_SYSTEM = """\
You audit ONE claim already extracted from a confidential information memorandum and cited \
to an exact span of the source. Your job is not to extract anything. It is to decide, reading \
the span on its own, whether it justifies the claim -- and to FLAG it if it does not.

Assume the binding is WRONG until the span proves it right. A claim a careful reader would \
query is a claim you flag.

You are given the span (verbatim source text), a little surrounding context, and the claim: an \
entity, an attribute, a raw value, and the scaled number it was stored as. Decide which holds:

- supported: reading only the span, this entity does have this attribute at this value. The \
number is plausible for what it measures.
- basket_collapse: the value counts or sums a LIST of things and the claim binds it to just one \
of them. "116 universities, colleges and councils" is 116 across the whole list, not 116 \
universities.
- entity_not_in_span: the entity names a party, place or thing the span does not actually name. \
The subject company may be implied; a DIFFERENT named entity may not be imported from elsewhere.
- attribute_mismatch: the span asserts a different thing than the attribute claims. A per-share \
figure labelled total; an ironing FEE labelled a shirt's price is the special case below.
- fee_vs_price: the value is a SERVICE charge (a rate card, "@ per item") read as the price of a \
physical good.
- bound_as_point: the span states a BOUND ("greater than", "up to", "at least", "and over") and \
the claim stored it as an exact value.
- scale_absurd: the scaled number is implausible for this entity and attribute by orders of \
magnitude -- e.g. a single regional property with tens of billions in annual revenue, or a \
headcount in the millions. This is how a mis-applied "(in thousands)" header shows up.

Quote, verbatim from the span, the words that settle your verdict. If you cannot ground it in \
the span, the verdict is `supported` -- you do not invent a fault you cannot point to.
"""


def _parse_once(client, *, model, system, user, page_no, what):
    """One structured call, retried once on a malformed body (same as propose.py)."""

    def call():
        return client.messages.parse(
            model=model,
            max_tokens=4000,
            thinking={"type": "adaptive"},
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
            output_format=AuditVerdict,
        )

    try:
        return call()
    except ValidationError:
        logger.warning("audit of %s (%s): unparseable body; retrying once", page_no, what)
        return call()


def audit_claim(
    *,
    entity: str,
    attribute: str,
    value_raw: str,
    value_normalized: float | None,
    unit: str | None,
    quote: str,
    context: str,
    page: int,
    model: str = DEFAULT_MODEL,
    client=None,
) -> AuditVerdict:
    """Audit one cited claim's binding. Returns a verdict; mutates nothing.

    `quote` is the resolved span verbatim; `context` is a small window around it.
    The proposed binding is shown so the auditor can refute it -- but the verdict
    must be grounded in the span, not in whether the binding sounds plausible.
    """
    if client is None:
        client = make_client()

    shown_value = (
        value_raw
        if value_normalized is None
        else f"{value_raw} (stored as {value_normalized:g}{' ' + unit if unit else ''})"
    )
    user = (
        f'SPAN (verbatim source):\n"""\n{quote}\n"""\n\n'
        f'CONTEXT around the span:\n"""\n{context}\n"""\n\n'
        f"CLAIM under audit:\n"
        f"  entity:    {entity}\n"
        f"  attribute: {attribute}\n"
        f"  value:     {shown_value}\n"
    )
    response = _parse_once(
        client, model=model, system=_SYSTEM, user=user, page_no=page, what=attribute
    )
    parsed = response.parsed_output
    if parsed is None:
        # A model that returned nothing has not found a fault it can ground; the
        # audit fails safe to `supported` rather than inventing a flag.
        return AuditVerdict(verdict="supported", evidence="")
    return parsed


def api_key_present() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))

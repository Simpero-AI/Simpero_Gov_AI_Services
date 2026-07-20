"""Tier-2 claim proposals over prose -- the model locates, the resolver decides.

Tier 1 reads tables, where structure hands you the claim: row label x column header is
the attribute, the cell is the value. That is why claims_from_table is small, and it is
why it stops at the table's edge. Measured against reference sets over two real CIMs,
44% and 59% of their facts are prose, chart labels and footnotes -- and 99-100% of those
are already sitting in the page text, positioned and citable. It has never been a seeing
problem; nothing was reading.

A sentence carries no row label. "the total size of the local market ... was estimated to
be considerably greater than £42 million in 2003" has an entity, an attribute, a value, a
period and a bound, and none of them are recoverable from layout. That is the whole reason
this module exists and the whole reason it uses a model.

WHAT MAKES A MODEL SAFE HERE
============================
Nothing in this module is trusted. It proposes; emit.py disposes.

A proposal names a QUOTE, and that quote must resolve to exactly one span in the page text
or the claim is emitted `missing` with no span at all. The resolver is unchanged and
unaware of where the quote came from -- the same rule that governs a table cell. So the
blast radius of a hallucination is a wrong `attribute` STRING. It is not a wrong number: the
value is parsed from the resolved source text by scale.py, never from anything the model
computed. It is not a wrong citation: a span that does not exist cannot be resolved, and a
span that appears twice is refused as ambiguous rather than guessed at.

Restating instead of quoting is therefore not a style preference. A restated number cannot
be located, so it fails closed and the claim is lost. The system prompt says so, and the
resolver enforces it whether or not the model complied.

WHAT THIS STILL DOES NOT ESTABLISH
==================================
That the quote is real and unique does NOT establish that the (entity, attribute) binding
is justified by it. A model can quote "£42 million" perfectly, resolve it to one span, and
label it the wrong metric -- and this module will emit that. Verifying the binding is a
separate pass against a separate context, and it is deliberately not done here: a proposer
grading its own proposal is not a check.
"""

from __future__ import annotations

import logging
import os

from pydantic import BaseModel, Field

from .emit import Claim, FlagLog, emit_pdf_claim
from .scale import ValueType, has_parseable_magnitude, holds_one_number
from .schemas import PageIndex, TextBlockRecord

logger = logging.getLogger(__name__)

# Sonnet-always is the locked floor for extraction; Opus is the current default and the
# quality-sensitive choice for a pass whose output enters the claims spine.
DEFAULT_MODEL = "claude-opus-4-8"

# Docling labels whose blocks carry assertions. Advisory, per text_extract's warning:
# page_header was measured labelling a reproduced press clipping, so furniture is excluded
# by is_boilerplate (measured, repetition-based) rather than by trusting the label.
PROSE_LABELS = frozenset({"text", "paragraph", "list_item", "footnote", "caption"})

_SYSTEM = """\
You extract claims from a page of a confidential information memorandum (CIM) so they can \
be stored with exact provenance.

A claim is one fact: an entity, an attribute, and a value that the page states.

THE QUOTE IS THE WHOLE CONTRACT.

`quote` must be copied VERBATIM from the page text you are given -- character for \
character, including the currency mark, the comma grouping, the decimal point, any \
parentheses around a negative, and any surrounding word you need to make it unique. Do not \
normalise it. Do not reformat it. Do not compute it. Do not summarise it.

A restated number cannot be located in the source, so a claim carrying one is DISCARDED. \
The quote you write is resolved against the page by exact match: it must appear there \
EXACTLY ONCE. If a figure appears several times on the page, extend the quote with \
adjacent words until it is unique, keeping every character contiguous and verbatim.

`value_text` is the value TOKEN on its own -- "£42 million", "18,454", "54.0%", "(6)" -- \
copied character for character from inside `quote`. The quote is the citation and may be a \
whole clause; value_text is the number itself. Getting this wrong misreads the magnitude: \
in "In 2003, 18,454 students attended", the value is 18,454 and 2003 is a date.

Rules:
- Extract only what the page ASSERTS. Never infer, never compute a total, never carry a \
figure over from another page.
- `entity` is who the fact is about, in the document's own words. A market, a competitor, \
a named university and the subject company are all different entities. Do not default \
everything to the subject company.
- `attribute` names the metric in the document's own words, not a canonical vocabulary.
- `value_type`: currency for money, percent for a rate, count for a countable quantity, \
date for a period, ratio for a multiple, text where there is no magnitude.
- Prefer fewer, well-grounded claims over many speculative ones. A page with no factual \
assertions yields none, and that is a correct answer.
"""


class ProposedClaim(BaseModel):
    """One model-proposed claim, before any verification."""

    quote: str = Field(description="Verbatim from the page, appearing exactly once.")
    value_text: str = Field(
        description="The value token itself, exactly as it appears inside `quote`."
    )
    entity: str = Field(description="Who the fact is about, in the document's words.")
    attribute: str = Field(description="What is being measured, in the document's words.")
    value_type: ValueType = Field(description="currency|percent|count|date|ratio|text")


class PageProposals(BaseModel):
    claims: list[ProposedClaim]


def prose_text(blocks: list[TextBlockRecord], page: PageIndex) -> str:
    """The page's prose, as the model should see it.

    Blocks rather than the flat page text: the flat index interleaves table cell contents
    with prose in reading order, and feeding a model a table flattened into a sentence
    invites it to propose claims the table path already emits, with worse attributes.

    Running headers and footers are excluded by label rather than by is_boilerplate,
    because that flag lives on char_map ranges and these are whole blocks. The label is
    safe in this direction: PROSE_LABELS is an allowlist, so a page_footer mislabelled as
    prose would have to be mislabelled INTO the set, and the observed failure went the
    other way -- real content labelled page_header.
    """
    kept: list[str] = []
    for block in blocks:
        if block.label not in PROSE_LABELS:
            continue
        text = block.text_normalized.strip()
        if text:
            kept.append(text)
    return "\n\n".join(kept)


def propose_for_page(
    blocks: list[TextBlockRecord],
    page: PageIndex,
    *,
    entity_hint: str,
    file: str,
    model: str = DEFAULT_MODEL,
    client=None,
) -> list[ProposedClaim]:
    """Ask the model for the claims one page's prose asserts.

    One call per page: a page is the unit the quote must resolve against, so it is also
    the unit that bounds a proposal's blast radius. Returns [] for a page with no prose.

    `entity_hint` is context, not an instruction -- the document's subject company, so the
    model can distinguish it from a competitor or a market it also mentions. It is
    deliberately not a default: stamping every claim with one entity is the defect this
    replaces on the table path.
    """
    text = prose_text(blocks, page)
    if not text.strip():
        return []

    if client is None:
        import anthropic

        client = anthropic.Anthropic()

    response = client.messages.parse(
        model=model,
        max_tokens=16000,
        thinking={"type": "adaptive"},
        # The system prompt is byte-identical across every page of every document, so it
        # is the whole cacheable prefix. The page text follows it and varies per call.
        system=[{"type": "text", "text": _SYSTEM, "cache_control": {"type": "ephemeral"}}],
        messages=[
            {
                "role": "user",
                "content": (
                    f"Document: {file}\nPage: {page.page}\n"
                    f"Subject company (context, not a default): {entity_hint}\n\n"
                    f"Page text:\n{text}"
                ),
            }
        ],
        output_format=PageProposals,
    )
    parsed = response.parsed_output
    return list(parsed.claims) if parsed else []


def claims_from_prose(
    blocks: list[TextBlockRecord],
    page: PageIndex,
    *,
    entity_hint: str,
    file: str,
    flag_log: FlagLog,
    model: str = DEFAULT_MODEL,
    client=None,
) -> list[Claim]:
    """Propose claims for one page's prose and emit each through the citation boundary.

    Every proposal goes through emit_pdf_claim unchanged, so a quote that does not resolve
    -- or resolves twice -- comes back `missing` with no span rather than being dropped.
    That matters here more than on the table path: a dropped proposal is invisible, while
    a `missing` one is a measurable record of the model having claimed something the page
    could not support.
    """
    proposals = propose_for_page(
        blocks, page, entity_hint=entity_hint, file=file, model=model, client=client
    )
    claims: list[Claim] = []
    for proposal in proposals:
        # value_text is checked against the quote rather than trusted. A token the
        # model wrote but did not copy is exactly the restatement the quote rule
        # exists to catch, and letting it through would put a magnitude in the
        # store that came from the model instead of the page. Falling back to the
        # quote is safe: that is what every table caller does.
        # `.strip()` before the containment test: "" and " " are substrings of
        # almost any quote, so an unset value_text passed this vacuously and
        # went on to be emitted as the claim's raw -- a claim whose value reads
        # as the empty string.
        named_value = proposal.value_text.strip()
        value_text = proposal.value_text if named_value and named_value in proposal.quote else None
        if proposal.value_text and value_text is None:
            logger.warning(
                "page %s: value_text %r is not inside the quote; parsing the quote instead",
                page.page,
                proposal.value_text,
            )

        # Nothing the model says is trusted, including its claim that the value
        # is a number. "five years", "one manager", "three" arrive typed count
        # or date and carry no magnitude at all. determine_scale raises on those
        # by contract -- correctly, since there is no honest ScaleResult for a
        # value that has none -- and an escaping raise took down not the claim
        # but the whole PAGE: 12 of 78 prose pages and 154 reference claims on
        # the first full-document run.
        #
        # Typing it "text" keeps the claim, its verbatim quote, its resolved
        # span and its bbox, and records no magnitude. A magnitude-less claim is
        # visibly magnitude-less in the store and still counts in recall, which
        # is this module's stance that a proposal the page cannot support
        # becomes a measurable record rather than a silent drop. Catching the
        # raise instead would swallow the claim AND the scale-invariant
        # AssertionError next to it, which must stay loud.
        value_type = proposal.value_type
        scaled_text = value_text if value_text is not None else proposal.quote
        if value_type != "text":
            if not has_parseable_magnitude(scaled_text):
                reason = "has no numeric content"
            elif not holds_one_number(scaled_text):
                # A sentence holding several numbers, with no value token naming
                # which one is meant. Parsing it anyway takes the leftmost, and
                # the leftmost is routinely a year.
                reason = "holds more than one number and none was named as the value"
            else:
                reason = ""
            if reason:
                logger.warning(
                    "page %s: %r/%r typed %s %s in %r; emitting as text",
                    page.page,
                    proposal.entity,
                    proposal.attribute,
                    value_type,
                    reason,
                    scaled_text,
                )
                value_type = "text"

        claims.append(
            emit_pdf_claim(
                proposal.entity,
                proposal.attribute,
                proposal.quote,
                page,
                value_type=value_type,
                origin="prose",
                file=file,
                flag_log=flag_log,
                value_text=value_text,
            )
        )
    resolved = sum(1 for claim in claims if claim.status != "missing")
    logger.info(
        "page %s: %d proposal(s), %d resolved to an exact span",
        page.page,
        len(proposals),
        resolved,
    )
    return claims


def api_key_present() -> bool:
    """Whether a credential is available, without reading its value."""
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))

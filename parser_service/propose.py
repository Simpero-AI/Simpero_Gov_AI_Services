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
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from .emit import (
    CORE_ATTRIBUTES,
    Claim,
    ClaimType,
    FlagLog,
    emit_pdf_claim,
    gate_canonical_attribute,
)
from .resolver import contains_flexible, find_exact_span
from .scale import ValueType, has_parseable_magnitude, holds_one_number
from .schemas import PageIndex, TextBlockRecord

logger = logging.getLogger(__name__)

_STAGE_ASSERTION = "prose_assertion"

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
- `claim_type`: the KIND of assertion -- numerical (a directly stated number), \
computational (a total/margin/ratio computed from other numbers), temporal (the value IS \
a date or period), or comparative (a stated change or comparison). Use unknown only if \
none fit.
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
    claim_type: ClaimType = Field(
        default="unknown",
        description=(
            "The KIND of assertion. numerical: a directly stated number. computational: a "
            "figure that is a total/subtotal/margin/ratio computed from other numbers. "
            "temporal: the value IS a date or period. comparative: a stated change or "
            "comparison (grew 20%, up from $Y). Use unknown only if none of these fit."
        ),
    )


class PageProposals(BaseModel):
    claims: list[ProposedClaim]


# --------------------------------------------------------------------------- #
# The qualitative arm.
#
# A separate call with a separate prompt, deliberately: `_SYSTEM` above is the
# byte-identical cacheable prefix the measured numeric recall was obtained
# under (91% of 79 human-selected stated facts on a real CIM, 95% on its
# property matrix). Merging the two jobs into one prompt would put that number
# back in play to save an API call, and it would have to be re-earned by
# measurement rather than preserved by construction.
#
# The gap this closes was measured, not imagined: a human marked 34 passages in
# a real CIM as claims the pipeline missed, and almost all were assertions with
# no magnitude -- outsourcing decisions, common directorships, target market,
# who the competitors are. The value-centric contract above cannot express them.
# --------------------------------------------------------------------------- #

# Read off those 34 marks rather than invented. There is deliberately no "other"
# member: a sentence fitting nothing has nowhere to go, which is the point.
# Known gaps -- IP, litigation, employment, environmental, insurance -- are
# absent because the evidence did not contain them. Add a class from an observed
# refusal on a real document, never from imagination.
AssertionClass = Literal[
    "related_party",
    "operating_model",
    "market_definition",
    "competitive_position",
    "commercial_terms",
    "risk_or_dependency",
    "plan_or_commitment",
]

# A generous runaway backstop, not a precision limit: a page states as many
# distinct qualitative assertions as it states, and a dense risk or financials
# page legitimately clears the old cap of three. The bound survives only so a
# degenerate response cannot stand dozens of restatements up as claims. It stays
# in code, never as a grammar `max_length`, because a cap in the output grammar
# turns an over-long response into a ValidationError that escapes the call and
# destroys the whole page -- the exact blast radius already fixed twice here.
MAX_ASSERTIONS_PER_PAGE = 15


_ASSERTION_SYSTEM = """\
You extract QUALITATIVE claims from a page of a confidential information memorandum \
(CIM) -- assertions about the business that carry no number -- so they can be stored with \
exact provenance.

A separate pass already extracts every stated figure on this page. Do not repeat one here. \
If the point of a sentence is its number, it belongs to that pass, not to you.

THE QUOTE IS THE WHOLE CONTRACT.

`quote` must be copied VERBATIM from the page text you are given -- character for \
character, including punctuation and capitalisation. Do not normalise it, reformat it, or \
summarise it. It is resolved against the page by exact match and must appear there EXACTLY \
ONCE. If the sentence appears more than once, extend the quote with adjacent words until it \
is unique, keeping every character contiguous and verbatim. A restated sentence cannot be \
located, so a claim carrying one is DISCARDED.

THE ADMISSION TEST. Propose a claim only when ALL FOUR hold. Apply them in order and stop \
at the first failure.

1. SUBJECT. The quote names who or what the claim is about -- a named party, a named or \
described competitor, a market, a site, a facility, or the subject company. Copy that noun \
phrase into `subject_text`, verbatim from inside `quote`.
   THE ONE EXCEPTION is the subject company named to you above. A CIM writes whole \
paragraphs about itself without naming itself, and a sentence like "Directed primarily at \
young people, typically student and young professionals" is a real claim about the subject \
company with no subject of its own. When the subject is elided that way, set `subject_text` \
to the clause carrying the assertion and `entity` to the subject company.
   A pronoun whose referent is some OTHER party in a different sentence is not a subject: \
extend the quote to include the referent, or drop the claim.

2. PREDICATE. The sentence asserts something about that subject which a diligence request \
could confirm or contradict from a document or a site visit -- a contract, a lease, a \
filing, a company register, an org chart, a supplier list, a price list, a site plan, a \
photograph, market data. Copy the clause that carries the assertion into `predicate_text`, \
verbatim from inside `quote`. If the only way to check it is to ask someone's opinion, STOP.

3. NEGATION. Ask: could a rival document state the OPPOSITE of this as a plain fact?
   "dry cleaning facilities will not be available on-site" -> the opposite, "will be \
available on-site", is a plain fact. ADMIT.
   "an innovative and exciting format" -> the opposite, "an uninnovative format", is \
nobody's stated fact. STOP.
   This is the test that separates a claim from marketing copy, and it does NOT turn on \
whether the sentence sounds dry. A sentence written in promotional language still passes if \
a factual counter-assertion survives: "the main competitive advantages lie in the breadth \
of the revenue model" passes, because "the advantage lies in price, not breadth" is a \
statable fact and the breadth of a revenue model is checkable by counting its lines. "a \
market that is clearly mature, but also saturated in terms of advertising and promotion" \
passes, because "the market is not mature" is a plain factual counter-assertion.
   Strip the evaluative words and test what is left. If nothing is left, STOP.
   A superlative with no stated basis ("the leading operator"), a gradable quality \
adjective ("state of the art", "convenient", "exciting", "relaxed"), and a future outcome \
asserted with no mechanism ("will be highly profitable") all fail here.

4. SELF-CONTAINED. `quote` together with `entity` states the whole claim. Nothing outside \
the quote is needed to know what is being asserted.

`entity` is who the claim is about, in the document's own words, and it must be READABLE IN \
THE QUOTE -- it is `subject_text`, or a name appearing inside it. The single exception is \
the subject company, given to you as context. Never name an entity you took from elsewhere \
on the page: a subject the quote cannot support is a binding nothing can check.

`attribute` names WHAT IS BEING ASSERTED, as a short noun phrase, with the evaluative words \
removed: "on-site dry cleaning availability", "directors' other directorships", "target \
customer segment", "basis of competitive advantage", "competitor collection point \
locations". A few words, never a sentence, never a restatement of the quote, and never an \
adjective standing alone.

`assertion_class` is the closed list below. A claim fitting none of them is not one you \
should propose, and saying so is the correct answer rather than a failure.
  related_party        -- directors, shareholders, affiliates, common control, connected \
companies, incorporation, a transaction with a connected person.
  operating_model      -- what will or will not be provided, how, by whom, from where; \
outsourcing, staffing, sites, opening arrangements, a service offered or excluded.
  market_definition    -- who the customer is stated to be; which segment, demographic, \
geography or channel; a stated qualitative property of the market itself such as maturity, \
saturation, fragmentation or seasonality.
  competitive_position -- who the competitors are said to be, and the advantage, \
disadvantage or barrier asserted relative to them.
  commercial_terms     -- leases, licences, contracts, tenure, exclusivity, renewal basis, \
restrictive covenants, pricing policy stated without a figure.
  risk_or_dependency   -- a stated dependence, constraint, contingency, regulatory \
requirement, key person, single supplier, or condition the plan rests on.
  plan_or_commitment   -- a forward step the business states it will take, with a \
mechanism: entering a segment, expansion, rollout, funding or exit intention. Not an \
outcome it merely hopes for.

Extract every DISTINCT checkable assertion the page makes. Do not hold yourself to a \
handful, and do not drop a claim because the page already stated another of the same \
assertion_class -- a risk page can carry several distinct risk_or_dependency and each \
is its own claim. What you must not do is restate one assertion as several: one claim \
per distinct point, not one per sentence that repeats it. Order them most substantive \
first.

You are not summarising the page. Most CIM pages yield none or one, and a page whose prose \
asserts nothing checkable yields NONE -- that is a correct answer, not a failure. Prefer \
the sentence a buyer's lawyer would put on a diligence list.

Page furniture is never a claim: a URL, a file path, a print date stamp, a running header \
or a page number asserts nothing about the business.
"""


class ProposedAssertion(BaseModel):
    """One model-proposed qualitative claim, before any verification.

    Field order is evidence first, label last, and that is deliberate: naming
    the class first invites the model to pick a category and then hunt for a
    sentence to fit it, which manufactures claims. Quote, then the subject and
    predicate copied out of it, then a label for what was copied.

    `subject_text` and `predicate_text` are GUARD INPUTS, not stored fields.
    They turn "the subject is in the span" and "the assertion is in the span"
    from prompt exhortations into deterministic containment tests -- which
    matters because prompt language has already been measured to fail here once
    ("Do not default everything to the subject company" did not stop entities
    being imported from elsewhere on the page).
    """

    quote: str = Field(description="Verbatim from the page, appearing exactly once.")
    subject_text: str = Field(description="The subject noun phrase, verbatim from inside `quote`.")
    predicate_text: str = Field(description="The asserting clause, verbatim from inside `quote`.")
    entity: str = Field(description="Who the claim is about; must be readable in the quote.")
    attribute: str = Field(description="What is asserted, as a noun phrase, evaluative words cut.")
    assertion_class: AssertionClass
    claim_type: ClaimType = Field(
        default="unknown",
        description=(
            "The KIND of assertion. entity_attribute: a non-numeric attribute of an entity "
            "(what it is, does, or owns). comparative: a stated comparison or ranking between "
            "entities. regulatory: a compliance, legal, or licensing requirement. Use unknown "
            "only if none of these fit."
        ),
    )


class PageAssertions(BaseModel):
    assertions: list[ProposedAssertion] = Field(default_factory=list)


def _parse_with_retry(call, *, page_no: int, what: str):
    """Run a structured-output call, retrying once on a malformed response.

    Measured across four full-document runs: a page occasionally comes back with
    an empty or truncated body, and the ValidationError raised while parsing it
    escapes and destroys that page's entire extraction. It is transient -- every
    observed instance succeeded when the same page was re-run (ACEP p20 failed
    once and passed on the next run with identical inputs).

    One retry, not a loop: a response malformed twice is a real failure and the
    caller should see it rather than have it retried into a timeout. The raise
    is deliberately still reachable -- this narrows a transient, it does not
    pretend the page succeeded.
    """
    try:
        return call()
    except ValidationError as exc:
        logger.warning(
            "page %s: %s returned an unparseable body (%s); retrying once",
            page_no,
            what,
            type(exc).__name__,
        )
        return call()


def _is_boilerplate_block(block: TextBlockRecord, page: PageIndex) -> bool:
    """True when a block resolves to a span that is ENTIRELY running furniture.

    A page header/footer that repeats across the document is tagged is_boilerplate
    on its char_map range by docling_parser.tag_boilerplate. A block whose whole
    resolved span carries that flag is that furniture, even when Docling mislabelled
    it "text" or "footnote" instead of page_header/page_footer -- the case the label
    allowlist alone misses.

    `all`, not `any` (mirroring inspect.is_boilerplate_token): a block is furniture
    only when every character of it is flagged, so a real sentence that merely abuts
    a footer is kept. A block that does not resolve to a unique span is kept -- fail
    open, because an unresolved block is not evidence of furniture. This does drop a
    repeated table FOOTNOTE that repetition flagged as furniture, footnote facts and
    all; that recall cost is accepted to keep headers and footers out of the model's
    input entirely.
    """
    span = find_exact_span(block.text_normalized, page.text, where=f"prose block p{page.page}")
    if span is None:
        return False
    start, end = span
    chars = page.char_map[start:end]
    return bool(chars) and all(char.is_boilerplate for char in chars)


def prose_text(blocks: list[TextBlockRecord], page: PageIndex) -> str:
    """The page's prose, as the model should see it.

    Blocks rather than the flat page text: the flat index interleaves table cell contents
    with prose in reading order, and feeding a model a table flattened into a sentence
    invites it to propose claims the table path already emits, with worse attributes.

    Running headers and footers are excluded two ways: by label (page_header/page_footer
    are not in PROSE_LABELS) and, for the furniture Docling mislabels as prose, by
    is_boilerplate -- a block whose whole span is flagged repeating furniture is dropped
    even when its label says "text". See _is_boilerplate_block for the recall trade this
    makes on repeated footnotes.
    """
    kept: list[str] = []
    for block in blocks:
        if block.label not in PROSE_LABELS:
            continue
        text = block.text_normalized.strip()
        if not text:
            continue
        if _is_boilerplate_block(block, page):
            continue
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

    response = _parse_with_retry(
        lambda: client.messages.parse(
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
        ),
        page_no=page.page,
        what="numeric proposal",
    )
    parsed = response.parsed_output
    return list(parsed.claims) if parsed else []


def _emit_numeric_proposal(
    proposal: ProposedClaim, page: PageIndex, *, file: str, flag_log: FlagLog
) -> Claim:
    """Emit one numeric prose proposal through the citation boundary.

    Shared by the first pass (claims_from_prose) and the completeness re-pass
    (claims_from_completeness) so both apply the identical value_text-in-quote
    check and magnitude-guard downgrade -- a recovered claim is held to exactly
    the same contract as an original one, never a looser one.
    """
    # value_text is checked against the quote rather than trusted. A token the
    # model wrote but did not copy is exactly the restatement the quote rule
    # exists to catch. `.strip()` first: "" and " " are substrings of almost any
    # quote, so an unset value_text would pass the containment test vacuously.
    named_value = proposal.value_text.strip()
    value_text = proposal.value_text if named_value and named_value in proposal.quote else None
    if proposal.value_text and value_text is None:
        logger.warning(
            "page %s: value_text %r is not inside the quote; parsing the quote instead",
            page.page,
            proposal.value_text,
        )

    # Nothing the model says is trusted, including that the value is a number.
    # A count/date carrying no magnitude would make determine_scale raise and take
    # the whole page down; typing it "text" keeps the claim, its quote and its span
    # while recording no magnitude.
    value_type = proposal.value_type
    scaled_text = value_text if value_text is not None else proposal.quote
    downgrade_flags: list[str] = []
    if value_type != "text":
        if not has_parseable_magnitude(scaled_text):
            reason = "has no numeric content"
        elif not holds_one_number(scaled_text):
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
            downgrade_flags.append("magnitude_unparseable")

    return emit_pdf_claim(
        proposal.entity,
        proposal.attribute,
        proposal.quote,
        page,
        value_type=value_type,
        origin="prose",
        file=file,
        flag_log=flag_log,
        value_text=value_text,
        claim_kind="quantitative",
        claim_type=proposal.claim_type,
        extra_flags=downgrade_flags,
    )


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
    claims = [
        _emit_numeric_proposal(proposal, page, file=file, flag_log=flag_log)
        for proposal in proposals
    ]
    resolved = sum(1 for claim in claims if claim.status != "missing")
    logger.info(
        "page %s: %d proposal(s), %d resolved to an exact span",
        page.page,
        len(proposals),
        resolved,
    )
    return claims


def propose_completion_for_page(
    page: PageIndex,
    missed: list[tuple[str, str]],
    *,
    entity_hint: str,
    file: str,
    model: str = DEFAULT_MODEL,
    client=None,
) -> list[ProposedClaim]:
    """The completeness re-pass over ONE page: given (number, context) pairs the
    first pass printed but did not claim, recover the ones that state a real fact.

    Reuses `_SYSTEM` -- the numeric prose contract, byte-identical so it shares the
    cached prefix -- and steers it with the missed-numbers list in the user
    message. The framing lets the model DECLINE a non-fact (a page number, a
    document id, a footnote marker, an unlabelled chart value); recovering junk is
    worse than leaving it, and the resolver + the binding verifier are the backstop
    for the ones it does propose.
    """
    if not missed:
        return []
    if client is None:
        import anthropic

        client = anthropic.Anthropic()

    listing = "\n".join(f'- {number}   printed in: "...{context}..."' for number, context in missed)
    user = (
        f"Document: {file}\nPage: {page.page}\n"
        f"Subject company (context, not a default): {entity_hint}\n\n"
        f"A first extraction pass already ran on this page. The numbers below were "
        f"printed on it but were NOT captured as claims. Recover only the ones that "
        f"state a real fact the page asserts. IGNORE a number that is page furniture "
        f"-- a page number, a document tracking id, a table-of-contents entry, a "
        f"footnote marker -- or a plotted chart value you cannot tie to a named entity "
        f"and attribute from the surrounding text. Recovering a non-fact is worse than "
        f"leaving it.\n\n"
        f"Numbers to consider:\n{listing}\n\n"
        f"Full page text (copy every quote VERBATIM from here; it must appear once):\n"
        f"{page.text}"
    )
    response = _parse_with_retry(
        lambda: client.messages.parse(
            model=model,
            max_tokens=16000,
            thinking={"type": "adaptive"},
            system=[{"type": "text", "text": _SYSTEM, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
            output_format=PageProposals,
        ),
        page_no=page.page,
        what="completeness",
    )
    parsed = response.parsed_output
    return list(parsed.claims) if parsed else []


def claims_from_completeness(
    page: PageIndex,
    missed: list[tuple[str, str]],
    *,
    entity_hint: str,
    file: str,
    flag_log: FlagLog,
    model: str = DEFAULT_MODEL,
    client=None,
) -> list[Claim]:
    """Recover claims for a page's coverage misses, emitted through the same
    numeric boundary as the first pass. A recovered quote that does not resolve
    (or resolves twice) still comes back `missing`, so the re-pass can only ADD
    genuinely-cited claims -- it cannot lower precision by construction."""
    proposals = propose_completion_for_page(
        page, missed, entity_hint=entity_hint, file=file, model=model, client=client
    )
    claims = [
        _emit_numeric_proposal(proposal, page, file=file, flag_log=flag_log)
        for proposal in proposals
    ]
    resolved = sum(1 for claim in claims if claim.status != "missing")
    logger.info(
        "page %s completeness: %d proposal(s), %d resolved to an exact span",
        page.page,
        len(proposals),
        resolved,
    )
    return claims


# --------------------------------------------------------------------------- #
# SIM-344 / E2: attribute vocabulary mapping.
#
# Proposer-with-code-gate, same shape as everywhere else in this module: the
# model proposes, and nothing it says is trusted past a deterministic check --
# here, emit.gate_canonical_attribute's literal membership test in
# emit.CORE_ATTRIBUTES. The blast radius of a bad proposal is a claim landing
# in the OPERATING_METRIC bucket with attribute_unmapped set, never a wrong
# canonical name silently accepted.
#
# One call for the whole document's DISTINCT raw labels, not one per claim or
# per page: a CIM repeats "Revenue | 2019F"-shaped labels across every table
# and prose mention, and mapping is keyed on the label text alone, so there is
# nothing page-scoped about it worth paying for twice.
# --------------------------------------------------------------------------- #

# A small, cheap model for a closed-vocabulary classification task, deliberately
# not DEFAULT_MODEL (Opus) -- that budget is reserved for extraction quality the
# claims spine's provenance depends on. Mapping a label onto ~25 fixed strings
# is exactly the kind of task where the code gate, not model strength, is what
# earns trust.
ATTRIBUTE_MODEL = "claude-haiku-4-5-20251001"

_ATTRIBUTE_SYSTEM = f"""\
You map financial-statement labels from a confidential information memorandum \
(CIM) onto a closed canonical vocabulary.

For each label, decide ONE of three things:

1. It names one of these {len(CORE_ATTRIBUTES)} core financial-statement line \
items -- answer with EXACTLY that string, nothing else:
{", ".join(sorted(CORE_ATTRIBUTES))}

2. It is a sector or operating metric, NOT a core financial-statement line item \
(a room count, an occupancy rate, same-store sales growth, a headcount, a square \
footage, a customer count, and similar). Answer exactly: operating_metric

3. It looks like it SHOULD be a core financial-statement line item -- plausibly a \
revenue, cost, profit, balance-sheet or cash-flow figure -- but you cannot \
confidently place it in exactly one of the fixed names above (an ambiguous or \
nonstandard label, a subtotal you cannot classify, an adjusted figure where you \
are unsure which convention applies). Answer exactly: core_unmapped

Never invent a canonical name outside the fixed list above. Never guess when \
unsure -- answer core_unmapped instead; a wrong guess is worse than an honest \
"unmapped".

gross_margin, net_margin and ebitda_margin are three separate names, not one \
generic "margin" -- pick the one the label actually names (gross profit margin, \
net income margin, EBITDA margin); if the label just says "margin" with no \
qualifier telling you which, that is the ambiguous case in (3), core_unmapped.

A label may carry a period, section banner or column header appended to it \
("Revenue | 2019F", "TURNOVER | Coffee Shop | YEAR 1") -- classify what the label \
NAMES, ignoring the period/section/column qualifiers riding along with it.

Each label is numbered. Answer with that same number, not the label text.
"""


class AttributeMapping(BaseModel):
    """One label's proposed classification, before the code gate runs.

    Identifies the label by its 1-based position in the input list, not by
    echoing the label text back -- a model reply need only reproduce a
    quote verbatim when the resolver depends on finding that exact text in
    a document (propose.py's module docstring); this is a caller-supplied
    list with no such text-matching step, and keying the result on the
    model's echo of it made a whitespace/punctuation-level rewrite (a
    trimmed space, a normalized dash) silently miss its lookup and fall
    back to core_unmapped -- fail-closed, but a real recall loss. A
    position number cannot drift the way echoed text can.
    """

    index: int = Field(description="The label's number from the list, exactly as given.")
    canonical: str = Field(
        description=(
            "One of the fixed core attribute names, or the literal string "
            "'operating_metric', or the literal string 'core_unmapped'."
        )
    )


class AttributeMappings(BaseModel):
    mappings: list[AttributeMapping] = Field(default_factory=list)


# Distinct labels per model call. One call for every label overflows the
# response on a real CIM (800+ labels): the JSON truncates past max_tokens,
# the parse fails twice, and _canonicalize_quantitative_claims skips the whole
# tier -- every claim keeps its raw label and 3b consistency has no canonical
# operands (SIM-382). This bounds each response well under max_tokens.
_ATTRIBUTE_BATCH = 80


def propose_attribute_mappings(
    raw_labels: list[str],
    *,
    model: str = ATTRIBUTE_MODEL,
    client=None,
) -> dict[str, str]:
    """Ask the model to classify each of `raw_labels` against the closed
    vocabulary, keyed by the raw label. Nothing here is trusted yet --
    gate_canonical_attribute (emit.py) is the code gate that turns a proposal
    into an actual canonical attribute; see canonicalize_attributes below.

    Returns {} for an empty input without a network call: a page-less or
    attribute-less run has nothing to map.
    """
    if not raw_labels:
        return {}
    if client is None:
        import anthropic

        client = anthropic.Anthropic()

    mapped: dict[str, str] = {}
    for offset in range(0, len(raw_labels), _ATTRIBUTE_BATCH):
        chunk = raw_labels[offset : offset + _ATTRIBUTE_BATCH]
        listing = "\n".join(f"{i}. {label}" for i, label in enumerate(chunk, start=1))
        try:
            response = _parse_with_retry(
                lambda listing=listing: client.messages.parse(
                    model=model,
                    max_tokens=8000,
                    system=[
                        {
                            "type": "text",
                            "text": _ATTRIBUTE_SYSTEM,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                    messages=[{"role": "user", "content": f"Labels:\n{listing}"}],
                    output_format=AttributeMappings,
                ),
                page_no=0,
                what="attribute mapping",
            )
        except ValidationError:
            # A batch malformed twice: skip it. Its labels are absent from the
            # result, so canonicalize_attributes defaults them to core_unmapped
            # rather than one bad batch sinking the whole document's mapping.
            logger.warning(
                "attribute mapping: a batch of %d labels failed twice; skipping it",
                len(chunk),
            )
            continue
        parsed = response.parsed_output
        if parsed is None:
            continue
        # An out-of-range index (a hallucinated number, an off-by-one) is dropped
        # rather than trusted -- canonicalize_attributes' default-to-core_unmapped
        # for any label missing already covers it, the same fail-closed path.
        for mapping in parsed.mappings:
            if 1 <= mapping.index <= len(chunk):
                mapped[chunk[mapping.index - 1]] = mapping.canonical
    return mapped


def canonicalize_attributes(
    raw_labels: list[str],
    *,
    model: str = ATTRIBUTE_MODEL,
    client=None,
) -> dict[str, tuple[str, list[str]]]:
    """raw label -> (canonical attribute, extra flags) for every DISTINCT label
    in `raw_labels`, proposer + code gate both applied.

    A label the model's response omitted entirely is treated exactly like an
    out-of-enum answer -- gated to OPERATING_METRIC with attribute_unmapped --
    rather than silently left unmapped with no record of the gap.
    """
    unique = sorted(set(raw_labels))
    proposed = propose_attribute_mappings(unique, model=model, client=client)
    return {raw: gate_canonical_attribute(proposed.get(raw, "core_unmapped")) for raw in unique}


def api_key_present() -> bool:
    """Whether a credential is available, without reading its value."""
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def propose_assertions_for_page(
    blocks: list[TextBlockRecord],
    page: PageIndex,
    *,
    entity_hint: str,
    file: str,
    model: str = DEFAULT_MODEL,
    client=None,
) -> list[ProposedAssertion]:
    """Ask the model for the qualitative claims one page's prose asserts.

    Its own call and its own byte-identical system prompt, so this prefix caches
    exactly as the numeric one does and neither job dilutes the other's context.
    """
    text = prose_text(blocks, page)
    if not text.strip():
        return []

    if client is None:
        import anthropic

        client = anthropic.Anthropic()

    response = _parse_with_retry(
        lambda: client.messages.parse(
            model=model,
            max_tokens=16000,
            thinking={"type": "adaptive"},
            system=[
                {"type": "text", "text": _ASSERTION_SYSTEM, "cache_control": {"type": "ephemeral"}}
            ],
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
            output_format=PageAssertions,
        ),
        page_no=page.page,
        what="qualitative proposal",
    )
    parsed = response.parsed_output
    return list(parsed.assertions) if parsed else []


def _within_budget(proposals: list[ProposedAssertion]) -> list[ProposedAssertion]:
    """Truncate to MAX_ASSERTIONS_PER_PAGE, keeping model order.

    Enforced here rather than in the output grammar on purpose: a grammar cap
    turns an over-long response into a ValidationError that escapes the call and
    takes the whole page with it, which is the blast radius this module has
    already had to close twice. Model order is kept -- the prompt asks for most
    substantive first, so a page that overruns the backstop chooses what it loses.

    There is deliberately no per-assertion_class bound. Distinct assertions of
    one class are distinct claims, and collapsing them to one was silently
    dropping real recall on dense risk and competition pages. Suppressing a
    restatement of a single assertion is the model's job (the prompt says so),
    not the budget's; the only bound that remains is the runaway backstop.
    """
    return proposals[:MAX_ASSERTIONS_PER_PAGE]


def assertions_from_prose(
    blocks: list[TextBlockRecord],
    page: PageIndex,
    *,
    entity_hint: str,
    file: str,
    flag_log: FlagLog,
    model: str = DEFAULT_MODEL,
    client=None,
) -> list[Claim]:
    """Qualitative claims for one page, emitted through the same citation boundary.

    The guards BLOCK here rather than flag, which is the opposite of the numeric
    path's choice and deliberate: a numeric claim whose entity is unsupported
    still carries a real magnitude read from the page, so dropping it would cost
    measured recall for no gain. A qualitative claim whose asserting clause is
    not in its span has nothing left that is true -- its value IS the text.
    """
    proposals = propose_assertions_for_page(
        blocks, page, entity_hint=entity_hint, file=file, model=model, client=client
    )
    claims: list[Claim] = []
    # Guards first, budget second. Reversed, a proposal that the containment
    # guards are about to reject still consumes its class's slot and displaces a
    # valid one behind it -- recall lost order-dependently, differently on every
    # run. The budget must choose among claims that can actually be emitted.
    supported: list[ProposedAssertion] = []
    for proposal in proposals:
        # Containment via the resolver's own grammar, not a hand-rolled `in`.
        # Two different answers to "is this inside that" is how the digit-test
        # precondition came to be implemented three incompatible ways.
        unsupported = None
        if not contains_flexible(proposal.quote, proposal.predicate_text):
            unsupported = f"predicate {proposal.predicate_text!r}"
        elif not contains_flexible(proposal.quote, proposal.subject_text):
            unsupported = f"subject {proposal.subject_text!r}"
        elif not (
            contains_flexible(proposal.quote, proposal.entity) or proposal.entity == entity_hint
        ):
            unsupported = f"entity {proposal.entity!r}"
        if unsupported is not None:
            logger.warning(
                "page %s: %s is not readable in the quote; dropping assertion",
                page.page,
                unsupported,
            )
            # binding_unsupported, NOT quote_unresolved. The quote resolves fine;
            # what fails is the model's binding to it. Logging a precision signal
            # under the resolver's recall flag makes both numbers unreadable, and
            # this log is the only precision instrument the project has.
            flag_log.log(
                _STAGE_ASSERTION,
                f"pdf:{file}:p{page.page}:{proposal.attribute}",
                "binding_unsupported",
                detail=proposal.quote,
            )
            continue
        supported.append(proposal)

    for proposal in _within_budget(supported):
        claims.append(
            emit_pdf_claim(
                proposal.entity,
                proposal.attribute,
                proposal.quote,
                page,
                value_type="text",
                origin="prose",
                file=file,
                flag_log=flag_log,
                claim_kind="qualitative",
                assertion_class=proposal.assertion_class,
                claim_type=proposal.claim_type,
                stage=_STAGE_ASSERTION,
            )
        )
    logger.info("page %s: %d qualitative claim(s)", page.page, len(claims))
    return claims

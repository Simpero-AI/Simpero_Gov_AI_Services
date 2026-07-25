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

from .emit import Claim, FlagLog, emit_pdf_claim
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

# Enforced in code after parsing, never as a grammar `max_length`. A cap in the
# output grammar turns an over-long response into a ValidationError that escapes
# the call and destroys the whole page -- the exact blast radius already fixed
# twice on this path.
MAX_ASSERTIONS_PER_PAGE = 3


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

AT MOST THREE claims from this page, and AT MOST ONE per assertion_class. Order them most \
substantive first -- the ones past the third are discarded, so a page returning six has \
chosen which three it loses by the order it wrote them in.

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
        downgrade_flags: list[str] = []
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
                # A positive mark, so a guard downgrade is never mistaken for a
                # qualitative claim. Both end up value_type "text"; only this
                # flag says which one lost a magnitude it was supposed to have.
                downgrade_flags.append("magnitude_unparseable")

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
                claim_kind="quantitative",
                extra_flags=downgrade_flags,
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
    """At most one per assertion_class, then at most MAX_ASSERTIONS_PER_PAGE.

    Enforced here rather than in the output grammar on purpose: a grammar cap
    turns an over-long response into a ValidationError that escapes the call and
    takes the whole page with it, which is the blast radius this module has
    already had to close twice. Model order is kept -- the prompt asks for most
    substantive first, so a page that overruns chooses what it loses.

    One-per-class is the sharper of the two bounds: it stops a competition page
    returning three restatements of the same competitive_position.
    """
    seen: set[str] = set()
    kept: list[ProposedAssertion] = []
    for p in proposals:
        if p.assertion_class in seen:
            continue
        seen.add(p.assertion_class)
        kept.append(p)
        if len(kept) == MAX_ASSERTIONS_PER_PAGE:
            break
    return kept


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
                stage=_STAGE_ASSERTION,
            )
        )
    logger.info("page %s: %d qualitative claim(s)", page.page, len(claims))
    return claims

"""Tier-2 prose proposals.

Every test here runs against a stub client. What is being tested is not the model -- it is
that nothing the model says is trusted: a quote that does not resolve, or resolves twice,
must fail closed exactly as a table cell would, and the emitted value must come from the
source text rather than from anything the proposal asserted.
"""

from __future__ import annotations

from types import SimpleNamespace

import anthropic
import httpx
import pytest
from pydantic import ValidationError

from parser_service.emit import CORE_ATTRIBUTES, OPERATING_METRIC, FlagLog, PdfLocation
from parser_service.propose import (
    MAX_ASSERTIONS_PER_PAGE,
    AttributeMapping,
    AttributeMappings,
    PageAssertions,
    PageProposals,
    ProposedAssertion,
    ProposedClaim,
    _normalize_attribute_label,
    assertions_from_prose,
    canonicalize_attributes,
    claims_from_completeness,
    claims_from_prose,
    propose_attribute_mappings,
    propose_for_page,
    prose_text,
)
from parser_service.schemas import CharBox, PageIndex, TextBlockRecord


def _page(text: str, page_no: int = 1) -> PageIndex:
    char_map: list[CharBox] = []
    x = 0.0
    for character in text:
        char_map.append(
            CharBox(
                char=character,
                x0=x,
                top=100.0,
                x1=x + 5.0,
                bottom=110.0,
                page=page_no,
                precision="char",
            )
        )
        x += 5.0
    return PageIndex(page=page_no, text=text, char_map=char_map)


def _block(text: str, label: str = "text", order: int = 0) -> TextBlockRecord:
    return TextBlockRecord(
        page=1,
        order=order,
        label=label,
        text=text,
        text_normalized=text,
        x0=0.0,
        top=0.0,
        x1=10.0,
        bottom=10.0,
        bbox_source="docling_native",
    )


class _StubClient:
    """Stands in for anthropic.Anthropic, returning fixed proposals."""

    def __init__(self, claims: list[ProposedClaim]) -> None:
        self._claims = claims
        self.calls: list[dict] = []
        self.messages = SimpleNamespace(parse=self._parse)

    def _parse(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(parsed_output=PageProposals(claims=self._claims))


# --------------------------------------------------------------------------- #
# The citation boundary. Nothing the model says is trusted.
# --------------------------------------------------------------------------- #


def test_a_verbatim_quote_resolves_and_carries_a_real_span() -> None:
    page = _page("The market was estimated at £42 million in 2003.")
    client = _StubClient(
        [
            ProposedClaim(
                quote="£42 million",
                value_text="£42 million",
                entity="Bristol student market",
                attribute="alcoholicDrinksMarketSize",
                value_type="currency",
            )
        ]
    )
    claims = claims_from_prose(
        [_block(page.text)],
        page,
        entity_hint="BarWash",
        file="bw.pdf",
        flag_log=FlagLog(),
        client=client,
    )

    assert len(claims) == 1
    claim = claims[0]
    assert claim.status == "proposed"
    location = claim.location
    assert isinstance(location, PdfLocation)
    assert location.char_start is not None and location.char_end is not None
    assert page.text[location.char_start : location.char_end] == "£42 million"
    # The magnitude came from the source token via scale.py, not from the proposal.
    assert claim.value.normalized == 42_000_000.0


def test_the_model_assigned_claim_type_flows_to_the_emitted_claim() -> None:
    # SIM-364: the prose tier's claim_type is the model's to set (via ProposedClaim);
    # it must reach the emitted claim unchanged -- the hybrid's model half.
    page = _page("Total revenue was £42 million in 2003.")
    client = _StubClient(
        [
            ProposedClaim(
                quote="£42 million",
                value_text="£42 million",
                entity="BarWash",
                attribute="totalRevenue",
                value_type="currency",
                claim_type="computational",
            )
        ]
    )
    claims = claims_from_prose(
        [_block(page.text)],
        page,
        entity_hint="BarWash",
        file="bw.pdf",
        flag_log=FlagLog(),
        client=client,
    )
    assert len(claims) == 1
    assert claims[0].claim_type == "computational"


def test_an_unclassified_prose_proposal_falls_back_to_unknown() -> None:
    # ProposedClaim.claim_type defaults to unknown, so a model that omits it yields a
    # visibly-untyped claim rather than a wrongly-guessed one.
    page = _page("The market was estimated at £42 million in 2003.")
    client = _StubClient(
        [
            ProposedClaim(
                quote="£42 million",
                value_text="£42 million",
                entity="Bristol student market",
                attribute="marketSize",
                value_type="currency",
            )
        ]
    )
    claims = claims_from_prose(
        [_block(page.text)],
        page,
        entity_hint="BarWash",
        file="bw.pdf",
        flag_log=FlagLog(),
        client=client,
    )
    assert claims[0].claim_type == "unknown"


def test_a_restated_quote_fails_closed() -> None:
    # The page says "£42 million"; the model "helpfully" normalised it. A restated number
    # cannot be located, so the claim must be `missing` rather than cited to nothing.
    page = _page("The market was estimated at £42 million in 2003.")
    client = _StubClient(
        [
            ProposedClaim(
                quote="42000000",
                value_text="42000000",
                entity="Bristol student market",
                attribute="alcoholicDrinksMarketSize",
                value_type="currency",
            )
        ]
    )
    flag_log = FlagLog()
    claims = claims_from_prose(
        [_block(page.text)],
        page,
        entity_hint="BarWash",
        file="bw.pdf",
        flag_log=flag_log,
        client=client,
    )

    assert claims[0].status == "missing"
    location = claims[0].location
    assert isinstance(location, PdfLocation)
    assert location.char_start is None and location.char_end is None
    assert [entry.flag_type for entry in flag_log.entries] == ["quote_unresolved"]


def test_an_ambiguous_quote_is_refused_rather_than_guessed() -> None:
    # The same figure twice on one page: which occurrence the claim refers to is
    # unknowable, so the resolver fails closed exactly as it does for a table cell.
    page = _page("Revenue was £42 million. Costs were £42 million.")
    client = _StubClient(
        [
            ProposedClaim(
                quote="£42 million",
                value_text="£42 million",
                entity="BarWash",
                attribute="revenue",
                value_type="currency",
            )
        ]
    )
    claims = claims_from_prose(
        [_block(page.text)],
        page,
        entity_hint="BarWash",
        file="bw.pdf",
        flag_log=FlagLog(),
        client=client,
    )
    assert claims[0].status == "missing"


def test_a_hallucinated_quote_cannot_invent_a_citation() -> None:
    page = _page("The company operates four venues in Bristol.")
    client = _StubClient(
        [
            ProposedClaim(
                quote="£99 million in EBITDA",
                value_text="£99 million",
                entity="BarWash",
                attribute="ebitda",
                value_type="currency",
            )
        ]
    )
    claims = claims_from_prose(
        [_block(page.text)],
        page,
        entity_hint="BarWash",
        file="bw.pdf",
        flag_log=FlagLog(),
        client=client,
    )
    assert claims[0].status == "missing"
    location = claims[0].location
    assert isinstance(location, PdfLocation)
    assert location.char_start is None


def test_the_proposal_never_supplies_the_number() -> None:
    # Even when the quote resolves, the magnitude is parsed from the resolved source text.
    # The proposal has no field for a value, so there is nothing for a model to get wrong.
    assert set(ProposedClaim.model_fields) == {
        "quote",
        "value_text",
        "entity",
        "attribute",
        "value_type",
        # A category (like value_type), not a magnitude -- still no number for the model
        # to supply. SIM-364.
        "claim_type",
    }
    # value_text names WHICH token in the quote is the value; it is not a number the
    # model supplies. The magnitude is still parsed from the source text by scale.py.
    assert ProposedClaim.model_fields["value_text"].annotation is str


# --------------------------------------------------------------------------- #
# What the model is shown.
# --------------------------------------------------------------------------- #


def test_only_prose_blocks_are_sent() -> None:
    blocks = [
        _block("Real prose about the market.", "text", 0),
        _block("BarWash Limited Confidential", "page_footer", 1),
        _block("Section 3", "section_header", 2),
        _block("A bullet point", "list_item", 3),
    ]
    text = prose_text(blocks, _page("irrelevant"))

    assert "Real prose about the market." in text
    assert "A bullet point" in text
    assert "Confidential" not in text, "running furniture must not be sent"
    assert "Section 3" not in text, "a heading asserts nothing"


def test_prose_text_drops_a_block_flagged_entirely_boilerplate() -> None:
    # A footer Docling mislabels "text" (so the allowlist passes it) but
    # tag_boilerplate flagged as repeating furniture: its whole span is
    # is_boilerplate, so it must not reach the model.
    real = "The company operates four gaming properties in Nevada."
    footer = "Version 2.0 January 2005"
    page = _page(real + "\n\n" + footer)
    start = page.text.index(footer)
    for char_box in page.char_map[start : start + len(footer)]:
        char_box.is_boilerplate = True

    out = prose_text([_block(real, "text", 0), _block(footer, "text", 1)], page)
    assert real in out
    assert footer not in out, "a fully-boilerplate block must not reach the model"


def test_prose_text_keeps_a_block_only_partly_boilerplate() -> None:
    # Exclusion is all-chars, not any-char: a real sentence that merely abuts a
    # footer (one flagged char) is kept, so furniture removal never costs real prose.
    text = "Real revenue grew strongly across every segment."
    page = _page(text)
    page.char_map[0].is_boilerplate = True

    assert prose_text([_block(text, "text", 0)], page) == text


def test_a_page_with_no_prose_costs_nothing() -> None:
    client = _StubClient([])
    result = propose_for_page(
        [_block("BarWash Limited", "page_footer")],
        _page("BarWash Limited"),
        entity_hint="BarWash",
        file="bw.pdf",
        client=client,
    )
    assert result == []
    assert client.calls == [], "a page with no prose must not reach the model"


def test_the_system_prompt_is_cached_and_the_page_is_not() -> None:
    # The system prompt is byte-identical across every page of every document, so it is
    # the cacheable prefix; the page text varies per call and must follow it.
    page = _page("Turnover in the Bristol venue reached 1,309 units.")
    client = _StubClient([])
    propose_for_page([_block(page.text)], page, entity_hint="BarWash", file="bw.pdf", client=client)

    call = client.calls[0]
    assert call["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert "Bristol venue reached 1,309" in call["messages"][0]["content"]
    assert "Bristol venue reached 1,309" not in call["system"][0]["text"], (
        "page text in the system prompt would break the cached prefix on every call"
    )


def test_the_entity_hint_is_offered_as_context_not_as_a_default() -> None:
    # Stamping one entity on every claim is the defect this path must not inherit from
    # the table path, where --entity is a single flag for the whole document.
    page = _page("The market was £42 million.")
    client = _StubClient([])
    propose_for_page([_block(page.text)], page, entity_hint="BarWash", file="bw.pdf", client=client)

    prompt = client.calls[0]["messages"][0]["content"]
    assert "not a default" in prompt
    assert "BarWash" in prompt


@pytest.mark.parametrize("field", ["quote", "entity", "attribute", "value_type"])
def test_a_proposal_missing_a_required_field_is_rejected_before_emission(field: str) -> None:
    # Every field is load-bearing: no quote means no citation, no value_type means no
    # scale decision. A partial proposal must not reach the emitter at all.
    payload: dict[str, str] = {
        "quote": "£42 million",
        "value_text": "£42 million",
        "entity": "BarWash",
        "attribute": "revenue",
        "value_type": "currency",
    }
    del payload[field]
    with pytest.raises(ValidationError):
        ProposedClaim(**payload)  # pyright: ignore[reportArgumentType]


def test_a_value_token_not_present_in_the_quote_is_not_trusted() -> None:
    """value_text names WHICH token in the quote is the value. A token the model
    wrote but did not copy is the same restatement the quote rule exists to catch,
    so it is checked against the quote rather than believed.
    """
    page = _page("In 2003, 18,454 students attended Bristol University.")
    client = _StubClient(
        [
            ProposedClaim(
                quote="In 2003, 18,454 students attended",
                value_text="18454",  # normalised — never appears on the page
                entity="Bristol University",
                attribute="studentsAttending",
                value_type="count",
            )
        ]
    )
    claims = claims_from_prose(
        [_block(page.text)],
        page,
        entity_hint="BarWash",
        file="bw.pdf",
        flag_log=FlagLog(),
        client=client,
    )
    # The uncopied token is refused, and the quote it falls back to holds two
    # numbers -- so which one is the value is unknown, and parsing it anyway
    # takes the leftmost and calls the year a student count. The claim keeps its
    # citation and records no magnitude instead. This assertion previously
    # stopped at `status == "proposed"`, which the 2003 reading also satisfied.
    assert claims[0].status == "proposed"
    assert claims[0].value.raw == "In 2003, 18,454 students attended"
    assert claims[0].value.value_type == "text"
    assert claims[0].value.normalized is None


def test_the_value_token_settles_the_magnitude_not_the_leftmost_number() -> None:
    """A prose quote is a sentence CONTAINING the value, so parsing its leftmost
    number takes the year: "In 2003, 18,454 students attended" yields 2003. Both
    this and a parenthetical read as an accounting negative were observed on the
    first real page this ran against.
    """
    page = _page("In 2003, 18,454 students attended Bristol University.")
    client = _StubClient(
        [
            ProposedClaim(
                quote="In 2003, 18,454 students attended",
                value_text="18,454",
                entity="Bristol University",
                attribute="studentsAttending",
                value_type="count",
            )
        ]
    )
    claims = claims_from_prose(
        [_block(page.text)],
        page,
        entity_hint="BarWash",
        file="bw.pdf",
        flag_log=FlagLog(),
        client=client,
    )
    assert claims[0].value.normalized == 18_454.0, "the year must not become the value"
    # The citation still covers the whole quote, not just the token.
    location = claims[0].location
    assert isinstance(location, PdfLocation)
    assert page.text[location.char_start : location.char_end] == "In 2003, 18,454 students attended"


# --------------------------------------------------------------------------- #
# One bad proposal must not cost the page.
# --------------------------------------------------------------------------- #


def test_a_spelled_out_quantity_does_not_take_down_its_page() -> None:
    """The measured defect: a proposal typed count whose value is "five years"
    has no magnitude, determine_scale raises on it by contract, and the raise
    escaped this loop and destroyed every other claim on the page. Twelve of 78
    prose pages were lost this way on the first full-document run.
    """
    page = _page("The lease runs five years. Turnover reached £42 million.")
    client = _StubClient(
        [
            ProposedClaim(
                quote="five years",
                value_text="five years",
                entity="BarWash",
                attribute="leaseTerm",
                value_type="count",
            ),
            ProposedClaim(
                quote="£42 million",
                value_text="£42 million",
                entity="BarWash",
                attribute="turnover",
                value_type="currency",
            ),
        ]
    )
    claims = claims_from_prose(
        [_block(page.text)],
        page,
        entity_hint="BarWash",
        file="bw.pdf",
        flag_log=FlagLog(),
        client=client,
    )

    assert len(claims) == 2, "the sibling claim on the same page must survive"
    assert claims[1].value.normalized == 42_000_000.0


@pytest.mark.parametrize(
    "value_text", ["five years", "eight minutes", "one manager", "three", "six years"]
)
def test_a_magnitude_less_value_is_typed_text_not_dropped(value_text: str) -> None:
    # Every one of these was observed in a real run. The claim is kept: it has a
    # verbatim quote and a resolved span, so it is a citable assertion that
    # simply carries no number. Dropping it would make the loss invisible, which
    # is the failure mode this module's docstring rejects by name.
    page = _page(f"The agreement covers {value_text} from completion.")
    client = _StubClient(
        [
            ProposedClaim(
                quote=f"covers {value_text} from completion",
                value_text=value_text,
                entity="BarWash",
                attribute="term",
                value_type="count",
            )
        ]
    )
    claims = claims_from_prose(
        [_block(page.text)],
        page,
        entity_hint="BarWash",
        file="bw.pdf",
        flag_log=FlagLog(),
        client=client,
    )

    assert len(claims) == 1
    claim = claims[0]
    assert claim.status == "proposed"
    assert claim.value.value_type == "text"
    assert claim.value.normalized is None
    assert claim.value.raw == value_text
    # The citation is untouched: a claim with no magnitude still points at the page.
    location = claim.location
    assert isinstance(location, PdfLocation)
    assert page.text[location.char_start : location.char_end] == (
        f"covers {value_text} from completion"
    )


def test_the_downgrade_tests_the_same_text_the_scaler_would_have_parsed() -> None:
    # emit_pdf_claim parses value_text when given and the quote otherwise, so the
    # guard must ask about the same string. Here value_text was not copied
    # verbatim, so it is discarded and the QUOTE is what gets parsed -- and the
    # quote does contain a number, so this must stay a real currency claim.
    page = _page("Turnover reached £42 million last year.")
    client = _StubClient(
        [
            ProposedClaim(
                quote="£42 million",
                value_text="forty-two million",
                entity="BarWash",
                attribute="turnover",
                value_type="currency",
            )
        ]
    )
    claims = claims_from_prose(
        [_block(page.text)],
        page,
        entity_hint="BarWash",
        file="bw.pdf",
        flag_log=FlagLog(),
        client=client,
    )

    assert claims[0].value.value_type == "currency"
    assert claims[0].value.normalized == 42_000_000.0


def test_a_value_type_of_text_is_left_alone() -> None:
    # A proposal that already says it has no magnitude needs no downgrade and
    # must not be re-examined for one.
    page = _page("The venue trades as Bar Wash Bristol.")
    client = _StubClient(
        [
            ProposedClaim(
                quote="Bar Wash Bristol",
                value_text="Bar Wash Bristol",
                entity="BarWash",
                attribute="tradingName",
                value_type="text",
            )
        ]
    )
    claims = claims_from_prose(
        [_block(page.text)],
        page,
        entity_hint="BarWash",
        file="bw.pdf",
        flag_log=FlagLog(),
        client=client,
    )

    assert claims[0].value.value_type == "text"
    assert claims[0].value.raw == "Bar Wash Bristol"


def test_a_prose_claim_does_not_inherit_a_table_banner() -> None:
    """The Bar Wash pages 38-39 defect, end to end. The page carries a "£'000"
    banner over a table; the sentence beside it quotes average customer spend.
    Every one of the nine magnitude errors in the first full-document run was
    this shape, and none of them flagged.
    """
    page = _page("Trading summary £'000. Average spend on alcohol and food was £14.25 per head.")
    client = _StubClient(
        [
            ProposedClaim(
                quote="£14.25 per head",
                value_text="£14.25",
                entity="BarWash",
                attribute="average spend on alcohol and food",
                value_type="currency",
            )
        ]
    )
    claims = claims_from_prose(
        [_block(page.text)],
        page,
        entity_hint="BarWash",
        file="bw.pdf",
        flag_log=FlagLog(),
        client=client,
    )

    assert claims[0].value.normalized == 14.25, "this shipped as 14250.0"
    assert claims[0].value.scale_source == "assumed_1x"
    assert "scale_assumed" in claims[0].flags


def test_an_unset_value_text_does_not_become_the_claims_raw() -> None:
    # "" is a substring of every quote, so an unset value_text passed the
    # verbatim check vacuously and was carried through as the value -- emitting
    # a claim whose raw reads as the empty string.
    page = _page("The market was estimated at £42 million in that year.")
    client = _StubClient(
        [
            ProposedClaim(
                quote="estimated at £42 million",
                value_text="",
                entity="market",
                attribute="size",
                value_type="currency",
            )
        ]
    )
    claims = claims_from_prose(
        [_block(page.text)],
        page,
        entity_hint="BarWash",
        file="bw.pdf",
        flag_log=FlagLog(),
        client=client,
    )

    assert claims[0].value.raw != ""
    assert claims[0].value.normalized == 42_000_000.0


def test_a_sentence_holding_two_numbers_does_not_pick_one_and_kill_the_page() -> None:
    """The second page-killer, found by review rather than by a run. When no
    value token is named, the quote is parsed -- and a quote carrying both a
    year and an inline multiplier reads 42 million through the suffix grammar
    and 2003 through the fallback. scale_invariant_holds catches the mismatch by
    raising, which is right, but the raise escaped and took the page with it,
    exactly as the digitless ValueError did.
    """
    quote = "In 2003, the total market was greater than £42 million"
    page = _page(f"{quote} overall. Turnover of £7,917 was recorded that year.")
    client = _StubClient(
        [
            ProposedClaim(
                quote=quote,
                value_text="42000000",  # restated, so not trusted
                entity="market",
                attribute="size",
                value_type="currency",
            ),
            ProposedClaim(
                quote="Turnover of £7,917",
                value_text="£7,917",
                entity="BarWash",
                attribute="turnover",
                value_type="currency",
            ),
        ]
    )
    claims = claims_from_prose(
        [_block(page.text)],
        page,
        entity_hint="BarWash",
        file="bw.pdf",
        flag_log=FlagLog(),
        client=client,
    )

    assert len(claims) == 2, "the sibling claim on the same page must survive"
    assert claims[0].value.value_type == "text"
    assert claims[0].value.normalized is None
    assert claims[1].value.normalized == 7_917.0


# --------------------------------------------------------------------------- #
# The qualitative arm.
#
# A human marked 34 passages in a real CIM as claims the pipeline missed; almost
# all were assertions carrying no number, which the value-centric contract
# cannot express. These pin the parts that must not drift.
# --------------------------------------------------------------------------- #


class _StubAssertionClient:
    def __init__(self, assertions: list[ProposedAssertion]) -> None:
        self._assertions = assertions
        self.calls: list[dict] = []
        self.messages = SimpleNamespace(parse=self._parse)

    def _parse(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(parsed_output=PageAssertions(assertions=self._assertions))


def _assertion(**kw) -> ProposedAssertion:
    base = {
        "quote": "dry cleaning facilities will not be available on-site",
        "subject_text": "dry cleaning facilities",
        "predicate_text": "will not be available on-site",
        "entity": "dry cleaning facilities",
        "attribute": "on-site dry cleaning availability",
        "assertion_class": "operating_model",
    }
    base.update(kw)
    return ProposedAssertion(**base)  # pyright: ignore[reportArgumentType]


def test_a_qualitative_assertion_is_emitted_with_a_real_citation() -> None:
    page = _page("For planning issues, dry cleaning facilities will not be available on-site.")
    claims = assertions_from_prose(
        [_block(page.text)],
        page,
        entity_hint="BarWash",
        file="bw.pdf",
        flag_log=FlagLog(),
        client=_StubAssertionClient([_assertion()]),
    )

    assert len(claims) == 1
    claim = claims[0]
    assert claim.status == "proposed"
    assert claim.claim_kind == "qualitative"
    assert claim.assertion_class == "operating_model"
    assert claim.value.value_type == "text"
    assert claim.value.normalized is None, "a qualitative claim has no magnitude by design"
    location = claim.location
    assert isinstance(location, PdfLocation)
    assert page.text[location.char_start : location.char_end] == (
        "dry cleaning facilities will not be available on-site"
    )


def test_an_assertion_whose_predicate_is_not_in_the_span_is_dropped() -> None:
    # The guards BLOCK here rather than flag, unlike the numeric path: a numeric
    # claim with a shaky entity still carries a magnitude read from the page,
    # but a qualitative claim's value IS its text -- if the asserting clause is
    # not in the span there is nothing left that is true.
    page = _page("For planning issues, dry cleaning facilities will not be available on-site.")
    flag_log = FlagLog()
    claims = assertions_from_prose(
        [_block(page.text)],
        page,
        entity_hint="BarWash",
        file="bw.pdf",
        flag_log=flag_log,
        client=_StubAssertionClient([_assertion(predicate_text="will be outsourced to a partner")]),
    )

    assert claims == []
    # binding_unsupported, not quote_unresolved: the quote resolves perfectly,
    # it is the model's binding to it that fails. Logging a precision signal
    # under the resolver's recall flag makes both numbers unreadable.
    assert [e.flag_type for e in flag_log.entries] == ["binding_unsupported"]


def test_an_entity_imported_from_elsewhere_on_the_page_is_dropped() -> None:
    # The measured binding defect: "187 ensuite" bound to "Chantry Court", a
    # name the span never contains. On this path it is refused outright.
    page = _page("The property offers 187 ensuite rooms to students in Bristol.")
    claims = assertions_from_prose(
        [_block(page.text)],
        page,
        entity_hint="BarWash",
        file="bw.pdf",
        flag_log=FlagLog(),
        client=_StubAssertionClient(
            [
                _assertion(
                    quote="The property offers 187 ensuite rooms",
                    subject_text="The property",
                    predicate_text="offers 187 ensuite rooms",
                    entity="Chantry Court",
                    attribute="room count",
                )
            ]
        ),
    )
    assert claims == []


def test_the_subject_company_may_be_the_entity_without_being_named() -> None:
    # A CIM writes paragraphs about itself without naming itself. The entity
    # hint is the single permitted exception to entity-must-be-in-the-span.
    page = _page("Directed primarily at young people, typically students and young professionals.")
    claims = assertions_from_prose(
        [_block(page.text)],
        page,
        entity_hint="BarWash",
        file="bw.pdf",
        flag_log=FlagLog(),
        client=_StubAssertionClient(
            [
                _assertion(
                    quote="Directed primarily at young people",
                    subject_text="Directed primarily at young people",
                    predicate_text="Directed primarily at young people",
                    entity="BarWash",
                    attribute="target customer segment",
                    assertion_class="market_definition",
                )
            ]
        ),
    )
    assert len(claims) == 1
    assert claims[0].entity == "BarWash"


# Distinct, prefix-free quote tokens for the budget tests: none is a substring
# of another, so each resolves to exactly one span and the count under test is
# never confounded by a fail-closed missing.
_NATO = [
    "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel",
    "india", "juliet", "kilo", "lima", "mike", "november", "oscar", "papa",
    "quebec", "romeo", "sierra", "tango", "uniform", "victor", "whiskey",
    "xray", "yankee", "zulu",
]  # fmt: skip


def test_an_overlong_page_truncates_in_code_not_in_the_grammar() -> None:
    # A cap in the output grammar makes an over-long response a ValidationError
    # that escapes the call and destroys the page -- the blast radius this module
    # has already had to close three times. A response past the backstop must
    # truncate to the backstop, in model order, and never raise.
    words = _NATO[: MAX_ASSERTIONS_PER_PAGE + 3]
    text = ". ".join(words) + "."
    page = _page(text)
    proposals = [_assertion(quote=w, subject_text=w, predicate_text=w, entity=w) for w in words]
    claims = assertions_from_prose(
        [_block(text)],
        page,
        entity_hint="BarWash",
        file="bw.pdf",
        flag_log=FlagLog(),
        client=_StubAssertionClient(proposals),
    )
    assert len(claims) == MAX_ASSERTIONS_PER_PAGE, "truncated to the backstop, not raised"
    assert [c.entity for c in claims] == words[:MAX_ASSERTIONS_PER_PAGE], "model order kept"


def test_distinct_assertions_of_one_class_are_all_kept() -> None:
    # The per-assertion_class bound used to collapse these to one, silently
    # costing recall on dense risk and competition pages. Distinct assertions are
    # distinct claims even when they share a class.
    quotes = [
        "alpha holds a licence",
        "bravo depends on a single supplier",
        "charlie renews annually",
    ]
    text = ". ".join(quotes) + "."
    page = _page(text)
    proposals = [
        _assertion(
            quote=q,
            subject_text=q,
            predicate_text=q,
            entity=q,
            assertion_class="risk_or_dependency",
        )
        for q in quotes
    ]
    claims = assertions_from_prose(
        [_block(text)],
        page,
        entity_hint="BarWash",
        file="bw.pdf",
        flag_log=FlagLog(),
        client=_StubAssertionClient(proposals),
    )
    assert len(claims) == 3, "same-class assertions are no longer collapsed to one"
    assert {c.assertion_class for c in claims} == {"risk_or_dependency"}


def test_a_malformed_response_is_retried_rather_than_costing_the_page() -> None:
    """Observed across four full-document runs: a page occasionally returns an
    empty body, the ValidationError escapes, and that page's whole extraction is
    lost. Every observed instance passed when re-run with identical inputs.
    """
    page = _page("For planning issues, dry cleaning facilities will not be available on-site.")

    class _FlakyClient(_StubAssertionClient):
        def __init__(self, assertions):
            super().__init__(assertions)
            self.attempts = 0

        def _parse(self, **kwargs):
            self.attempts += 1
            if self.attempts == 1:
                raise ValidationError.from_exception_data("PageAssertions", [])
            return super()._parse(**kwargs)

    client = _FlakyClient([_assertion()])
    claims = assertions_from_prose(
        [_block(page.text)],
        page,
        entity_hint="BarWash",
        file="bw.pdf",
        flag_log=FlagLog(),
        client=client,
    )
    assert client.attempts == 2, "retried exactly once"
    assert len(claims) == 1, "the page survived"


def test_a_guard_downgrade_is_distinguishable_from_a_qualitative_claim() -> None:
    # Both end up value_type "text". Only the flag says which one lost a
    # magnitude it was supposed to have -- without it the two populations are
    # indistinguishable in the store and neither can be measured.
    page = _page("The agreement covers five years from completion.")
    claims = claims_from_prose(
        [_block(page.text)],
        page,
        entity_hint="BarWash",
        file="bw.pdf",
        flag_log=FlagLog(),
        client=_StubClient(
            [
                ProposedClaim(
                    quote="covers five years from completion",
                    value_text="five years",
                    entity="BarWash",
                    attribute="term",
                    value_type="count",
                )
            ]
        ),
    )
    downgraded = claims[0]
    assert downgraded.value.value_type == "text"
    assert downgraded.claim_kind == "quantitative", "a downgrade is not a qualitative claim"
    assert "magnitude_unparseable" in downgraded.flags


def test_a_rejected_proposal_does_not_consume_a_backstop_slot() -> None:
    """Guards run before the backstop, not after. Reversed, a proposal the
    containment guards are about to reject would still occupy one of the
    MAX_ASSERTIONS_PER_PAGE slots and drop a valid claim past the bound -- recall
    lost order-dependently, differently on every run. The budget must choose
    among claims that can actually be emitted.
    """
    valid = _NATO[:MAX_ASSERTIONS_PER_PAGE]
    text = ". ".join(valid) + "."
    page = _page(text)
    proposals = [
        # rejected, and first in line -- predicate is not readable in the quote
        _assertion(
            quote=valid[0],
            subject_text=valid[0],
            predicate_text="outsourced to an industry player",
            entity=valid[0],
        ),
        *(_assertion(quote=w, subject_text=w, predicate_text=w, entity=w) for w in valid),
    ]
    claims = assertions_from_prose(
        [_block(text)],
        page,
        entity_hint="BarWash",
        file="bw.pdf",
        flag_log=FlagLog(),
        client=_StubAssertionClient(proposals),
    )
    assert len(claims) == MAX_ASSERTIONS_PER_PAGE, (
        "every valid claim survives; the rejected proposal took no slot"
    )


def test_a_missing_qualitative_claim_is_still_marked_qualitative() -> None:
    # The module's docstring calls the missing population "a measurable record
    # of the model having claimed something the page could not support". It is
    # only measurable if the missing row says which arm produced it.
    page = _page("The venue operates a licensed bar on the ground floor.")
    claims = assertions_from_prose(
        [_block(page.text)],
        page,
        entity_hint="BarWash",
        file="bw.pdf",
        flag_log=FlagLog(),
        client=_StubAssertionClient(
            [
                _assertion(
                    quote="a licensed bar on the ground floor",
                    subject_text="a licensed bar",
                    predicate_text="on the ground floor",
                    entity="a licensed bar",
                )
            ]
        ),
    )
    # Resolves fine here; the point is the field is threaded at all.
    assert claims[0].claim_kind == "qualitative"
    assert claims[0].assertion_class == "operating_model"


def test_a_downgrade_flag_survives_an_unresolved_quote() -> None:
    # flags was seeded AFTER the missing returns, so magnitude_unparseable was
    # silently discarded on exactly the claims whose provenance failed.
    page = _page("The agreement covers five years from completion.")
    claims = claims_from_prose(
        [_block(page.text)],
        page,
        entity_hint="BarWash",
        file="bw.pdf",
        flag_log=FlagLog(),
        client=_StubClient(
            [
                ProposedClaim(
                    quote="a term the page never states",
                    value_text="five years",
                    entity="BarWash",
                    attribute="term",
                    value_type="count",
                )
            ]
        ),
    )
    assert claims[0].status == "missing"
    assert "magnitude_unparseable" in claims[0].flags
    assert "quote_unresolved" in claims[0].flags


def test_completeness_recovers_a_missed_number_through_the_same_boundary() -> None:
    # A number the first pass left uncovered is put in front of the model, and a
    # recovered proposal is emitted through the exact same citation boundary.
    page = _page("Historical EBITDA was $242.5 for the year.")
    client = _StubClient(
        [
            ProposedClaim(
                quote="$242.5",
                value_text="$242.5",
                entity="CUS",
                attribute="historical EBITDA",
                value_type="currency",
            )
        ]
    )
    claims = claims_from_completeness(
        page,
        [("$242.5", "Historical EBITDA was $242.5 for the year.")],
        entity_hint="CUS",
        file="d.pdf",
        flag_log=FlagLog(),
        client=client,
    )
    assert len(claims) == 1 and claims[0].status == "proposed"
    loc = claims[0].location
    assert isinstance(loc, PdfLocation)
    assert page.text[loc.char_start : loc.char_end] == "$242.5"
    # the missed number was actually put in front of the model
    assert "$242.5" in client.calls[0]["messages"][0]["content"]


def test_completeness_makes_no_model_call_when_there_are_no_misses() -> None:
    # A page whose numbers were all covered costs nothing: no misses, no call.
    client = _StubClient([])
    claims = claims_from_completeness(
        _page("All covered here."),
        [],
        entity_hint="X",
        file="d.pdf",
        flag_log=FlagLog(),
        client=client,
    )
    assert claims == []


# --------------------------------------------------------------------------- #
# SIM-344: propose_attribute_mappings / canonicalize_attributes.
#
# Every test here runs against a stub client, same discipline as the rest of
# this file: what is under test is not the model, it is that nothing it says
# is trusted past emit.gate_canonical_attribute's literal enum check.
# --------------------------------------------------------------------------- #


class _StubAttributeClient:
    """Stands in for anthropic.Anthropic, returning fixed attribute mappings."""

    def __init__(self, mappings: list[AttributeMapping]) -> None:
        self._mappings = mappings
        self.calls: list[dict] = []
        self.messages = SimpleNamespace(parse=self._parse)

    def _parse(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(parsed_output=AttributeMappings(mappings=self._mappings))


def test_propose_attribute_mappings_makes_no_call_for_an_empty_batch() -> None:
    def _must_not_run(**_k):
        raise AssertionError("an empty batch has nothing to map")

    client = SimpleNamespace(messages=SimpleNamespace(parse=_must_not_run))
    assert propose_attribute_mappings([], client=client) == {}


def test_propose_attribute_mappings_returns_the_models_answers_keyed_by_position() -> None:
    client = _StubAttributeClient(
        [
            AttributeMapping(index=1, canonical="revenue"),
            AttributeMapping(index=2, canonical="operating_metric"),
        ]
    )
    result = propose_attribute_mappings(["Revenue | 2019F", "Total Rooms"], client=client)
    assert result == {"Revenue | 2019F": "revenue", "Total Rooms": "operating_metric"}
    # Every core attribute is listed in the prompt so the model has the fixed
    # vocabulary in front of it, not just examples.
    prompt = client.calls[0]["system"][0]["text"]
    assert all(attr in prompt for attr in CORE_ATTRIBUTES)


def test_propose_attribute_mappings_is_immune_to_a_non_verbatim_echo() -> None:
    # Keying by position, not by the model re-typing the label, is the fix for
    # the recall loss the review flagged: a model that answers with a
    # whitespace/punctuation-level rewrite of the label (trimmed space,
    # normalized dash) must still land on the right label -- an index cannot
    # drift the way echoed text can.
    client = _StubAttributeClient([AttributeMapping(index=1, canonical="revenue")])
    result = propose_attribute_mappings(["Revenue | 2019F "], client=client)
    assert result == {"Revenue | 2019F ": "revenue"}


def test_propose_attribute_mappings_drops_an_out_of_range_index() -> None:
    # A hallucinated or off-by-one index must not raise or silently corrupt
    # another label's answer -- it is dropped, and canonicalize_attributes'
    # missing-answer fallback (attribute_unmapped) covers the gap the same way
    # an omitted label already does.
    client = _StubAttributeClient(
        [AttributeMapping(index=0, canonical="revenue"), AttributeMapping(index=5, canonical="x")]
    )
    result = propose_attribute_mappings(["Revenue | 2019F"], client=client)
    assert result == {}


def test_a_batch_api_error_is_skipped_not_raised() -> None:
    # A genuine transient the SDK retries exhausted (here a connection error; 429s
    # and 5xx behave the same) is caught: the batch's labels are simply absent, no
    # exception escapes the run.
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")

    def _raise_api_error(**_k):
        raise anthropic.APIConnectionError(request=request)

    client = SimpleNamespace(messages=SimpleNamespace(parse=_raise_api_error))
    assert propose_attribute_mappings(["Revenue | 2019F"], client=client) == {}


def test_a_config_error_propagates_rather_than_being_swallowed_per_batch() -> None:
    # A revoked/expired credential (401) or a genuinely invalid request (a
    # non-grammar 4xx) is a deploy/config problem that fails every batch
    # identically -- it must propagate (surfacing as one document-level
    # attribute_mapping SkippedPage), not be silently defaulted to the catch-all
    # like a transient. A 429, by contrast, stays a transient and is skipped.
    response = httpx.Response(
        401, request=httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    )

    def _raise_auth(**_k):
        raise anthropic.AuthenticationError("invalid x-api-key", response=response, body=None)

    client = SimpleNamespace(messages=SimpleNamespace(parse=_raise_auth))
    with pytest.raises(anthropic.AuthenticationError):
        propose_attribute_mappings(["Revenue | 2019F"], client=client)


def test_a_programming_bug_propagates_rather_than_being_swallowed() -> None:
    # A code defect (TypeError) is NOT a transient -- the narrowed except must let
    # it propagate loudly, not silently default the batch to operating_metric.
    def _raise_type_error(**_k):
        raise TypeError("bad kwarg")

    client = SimpleNamespace(messages=SimpleNamespace(parse=_raise_type_error))
    with pytest.raises(TypeError):
        propose_attribute_mappings(["Revenue | 2019F"], client=client)


def test_canonicalize_attributes_gates_a_literal_core_answer_through_clean() -> None:
    client = _StubAttributeClient([AttributeMapping(index=1, canonical="revenue")])
    result = canonicalize_attributes(["Revenue | 2019F"], client=client)
    assert result == {"Revenue | 2019F": ("revenue", [])}


def test_canonicalize_attributes_gates_a_hallucinated_answer_to_unmapped() -> None:
    # The model answered something plausible-sounding but not in the fixed
    # list -- the code gate must not trust it just because it looks like an
    # attribute name.
    client = _StubAttributeClient([AttributeMapping(index=1, canonical="adjusted_ebitda")])
    result = canonicalize_attributes(["Adjusted EBITDA (non-GAAP)"], client=client)
    assert result == {"Adjusted EBITDA (non-GAAP)": (OPERATING_METRIC, ["attribute_unmapped"])}


def test_canonicalize_attributes_treats_a_missing_answer_as_unmapped_not_dropped() -> None:
    # The model's response omitted this label entirely -- must not be silently
    # skipped; it still gets a canonical value and the flag that records the gap.
    client = _StubAttributeClient([])
    result = canonicalize_attributes(["Some Label"], client=client)
    assert result == {"Some Label": (OPERATING_METRIC, ["attribute_unmapped"])}


def test_canonicalize_attributes_dedupes_before_calling_the_model() -> None:
    client = _StubAttributeClient([AttributeMapping(index=1, canonical="revenue")])
    canonicalize_attributes(
        ["Revenue | 2019F", "Revenue | 2019F", "Revenue | 2019F"], client=client
    )
    assert len(client.calls) == 1
    assert client.calls[0]["messages"][0]["content"].count("Revenue | 2019F") == 1


def test_normalize_attribute_label_strips_period_basis_and_scale_only() -> None:
    n = _normalize_attribute_label
    # period / scale variants of one metric fold to one key
    assert n("EBITDA 2001") == n("EBITDA") == n("EBITDA ($ in millions)") == "ebitda"
    assert n("% Margin (2001)") == n("2001 % Margin") == "% margin"
    assert n("projected net revenue 2006E PF") == n("projected net revenue 2007E")
    # semantic content is untouched, so distinct metrics keep distinct keys
    assert n("Gross margin") != n("Net margin")
    # the structural "|" is NOT split: an entity's two metrics stay separate
    assert n("Aquarius (1) | Slots") != n("Aquarius (1) | Hotel Rooms")


def test_canonicalize_maps_a_metric_once_across_period_and_scale_variants() -> None:
    # EBITDA / EBITDA 2001 / EBITDA ($ in millions) name one metric across a
    # period and a scale tag -- classified ONCE, the verdict fanned back to every
    # variant, so a metric cannot land on two different canonicals per period.
    client = _StubAttributeClient([AttributeMapping(index=1, canonical="ebitda")])
    labels = ["EBITDA", "EBITDA 2001", "EBITDA ($ in millions)"]
    result = canonicalize_attributes(labels, client=client)
    assert result == {lbl: ("ebitda", []) for lbl in labels}
    assert len(client.calls) == 1  # one metric, one classification
    assert client.calls[0]["messages"][0]["content"].count("EBITDA") == 1


def test_canonicalize_does_not_merge_distinct_metrics_sharing_an_entity_prefix() -> None:
    # The "|" here separates entity from metric ("Aquarius (1) | Slots"), not
    # metric from period, so normalization must NOT split it and fuse "Slots"
    # with "Casino Square Footage" -- each is classified in its own right.
    client = _StubAttributeClient(
        [
            AttributeMapping(index=1, canonical="operating_metric"),
            AttributeMapping(index=2, canonical="operating_metric"),
        ]
    )
    labels = ["Aquarius (1) | Casino Square Footage", "Aquarius (1) | Slots"]
    result = canonicalize_attributes(labels, client=client)
    assert set(result) == set(labels)
    sent = client.calls[0]["messages"][0]["content"]
    assert "Casino Square Footage" in sent and "Slots" in sent

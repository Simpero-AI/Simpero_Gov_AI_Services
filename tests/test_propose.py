"""Tier-2 prose proposals.

Every test here runs against a stub client. What is being tested is not the model -- it is
that nothing the model says is trusted: a quote that does not resolve, or resolves twice,
must fail closed exactly as a table cell would, and the emitted value must come from the
source text rather than from anything the proposal asserted.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from parser_service.emit import FlagLog, PdfLocation
from parser_service.propose import (
    PageProposals,
    ProposedClaim,
    claims_from_prose,
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
    # Falls back to parsing the quote rather than accepting an uncopied token.
    assert claims[0].status == "proposed"
    assert claims[0].value.raw == "In 2003, 18,454 students attended"


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

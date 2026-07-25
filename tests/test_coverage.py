"""Numeric coverage instrument: does every meaningful number become a claim, and
when it doesn't, is the miss classified by the right gate."""

from __future__ import annotations

from parser_service.coverage import document_coverage, page_coverage
from parser_service.schemas import CharBox, PageIndex


def _page(text: str, page_no: int = 1, boiler: tuple[int, int] | None = None) -> PageIndex:
    char_map = [
        CharBox(
            char=c,
            x0=0.0,
            top=0.0,
            x1=1.0,
            bottom=1.0,
            page=page_no,
            is_boilerplate=(boiler is not None and boiler[0] <= i < boiler[1]),
            precision="word",
        )
        for i, c in enumerate(text)
    ]
    return PageIndex(page=page_no, text=text, char_map=char_map)


def test_a_number_a_claim_span_covers_is_covered() -> None:
    text = "Revenue was $15,295 for the year."
    page = _page(text)
    s = text.index("$15,295")
    cov = page_coverage(page, [(s - 4, s + len("$15,295") + 4)])
    assert cov.total == 1
    assert cov.covered == 1
    assert cov.misses == []


def test_an_uncovered_but_resolvable_number_is_a_gate1_miss() -> None:
    # 27.3% is meaningful, appears once, and no claim covers it: the model simply
    # did not propose it -- addressable by asking for more.
    page = _page("Operating margin was 27.3% overall.")
    cov = page_coverage(page, [])
    assert cov.total == 1 and cov.covered == 0
    assert [m.gate for m in cov.misses] == [1]
    assert cov.misses[0].text == "27.3%"


def test_an_uncovered_ambiguous_number_is_a_gate2_miss() -> None:
    # "$28.1" appears twice, so it cannot resolve to a unique span -- a
    # provenance-layer loss, not a proposal loss.
    page = _page("EBITDA of $28.1 here, and $28.1 again there.")
    cov = page_coverage(page, [])
    assert cov.total == 2
    assert all(m.gate == 2 for m in cov.misses)


def test_a_bare_small_integer_is_not_a_meaningful_number() -> None:
    # A page number carries no $, %, suffix, comma, or decimal, so it is never a
    # coverage target -- otherwise every page would report false misses.
    cov = page_coverage(_page("Please see page 5 for the detail."), [])
    assert cov.total == 0


def test_a_number_in_furniture_is_not_counted() -> None:
    text = "Confidential -- $9,999,999 -- draft footer"
    page = _page(text, boiler=(0, len(text)))
    cov = page_coverage(page, [])
    assert cov.total == 0


def test_document_coverage_aggregates_and_scopes_spans_by_page() -> None:
    p1 = _page("Total revenue was $10,000 this year.", page_no=1)
    p2 = _page("Costs reached $5,000 in the period.", page_no=2)
    s1 = p1.text.index("$10,000")
    # A claim on page 1 covers p1's number; page 2's number is left uncovered.
    cov = document_coverage([p1, p2], [(1, s1, s1 + len("$10,000"))])
    assert cov.total == 2
    assert cov.covered == 1
    assert cov.gate1 == 1 and cov.gate2 == 0
    assert cov.coverage == 0.5
    assert cov.gate1_misses[0].page == 2

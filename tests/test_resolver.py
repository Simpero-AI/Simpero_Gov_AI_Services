"""DS-W3-3 exact-span resolver tests.

Two layers, like the rest of the parser suite:
- Fast, CI-portable tests that build synthetic PageIndex objects and fully
  exercise the fail-closed contract (exact match, ambiguity, normalization
  dependency, multi-line bboxes) with no fixtures.
- local_corpus-marked acceptance tests that resolve the ticket's verified
  corpus strings ($15,295, 27.3%, 3,817) against the real PTL PDF-page 11.
"""

import os
from pathlib import Path

import pytest

from parser_service.resolver import resolve, resolve_in_cell, union_bbox
from parser_service.schemas import CharBox, PageIndex, TableCellRecord

# parse_pdf_bytes (and the heavy docling import behind it) is pulled in lazily by
# the local_corpus fixture only — the fast tests below need nothing from docling.


def make_page(text: str, page_no: int = 1) -> PageIndex:
    """Build a PageIndex whose char_map has plausible TOPLEFT-origin geometry:
    each visual line steps down in `top`, characters step right in `x`, and a
    newline is a zero-width marker at the end of its line."""
    char_map: list[CharBox] = []
    x = 0.0
    top = 0.0
    line_height = 10.0
    for ch in text:
        if ch == "\n":
            char_map.append(
                CharBox(
                    char="\n",
                    x0=x,
                    top=top,
                    x1=x,
                    bottom=top + line_height,
                    page=page_no,
                    precision="word",
                )
            )
            x = 0.0
            top += 20.0
        else:
            char_map.append(
                CharBox(
                    char=ch,
                    x0=x,
                    top=top,
                    x1=x + 5.0,
                    bottom=top + line_height,
                    page=page_no,
                    precision="word",
                )
            )
            x += 5.0
    return PageIndex(page=page_no, text=text, char_map=char_map)


def test_resolve_unique_quote_returns_exact_span() -> None:
    page = make_page("Revenue $15,295 total", page_no=3)

    span = resolve("$15,295", page)

    assert span is not None
    assert span.page == 3
    # char_end is exclusive: the slice reproduces the quote exactly.
    assert page.text[span.char_start : span.char_end] == "$15,295"
    assert span.char_start == page.text.index("$15,295")
    assert span.char_end - span.char_start == len("$15,295")
    # Single-line quote -> one line box, and it equals the union bbox.
    assert len(span.line_bboxes) == 1
    assert span.line_bboxes[0] == span.bbox
    assert span.bbox.x1 > span.bbox.x0
    assert span.bbox.bottom > span.bbox.top


def test_resolve_not_present_returns_none() -> None:
    page = make_page("Revenue $15,295 total")
    assert resolve("not-on-this-page", page) is None


def test_resolve_ambiguous_returns_none_and_logs(caplog: pytest.LogCaptureFixture) -> None:
    page = make_page("$15,295 here and $15,295 again", page_no=7)

    with caplog.at_level("WARNING"):
        result = resolve("$15,295", page)

    assert result is None
    assert any("Ambiguous quote" in r.message for r in caplog.records)
    assert any("7" in r.getMessage() for r in caplog.records)


def test_resolve_empty_quote_returns_none() -> None:
    page = make_page("anything at all")
    assert resolve("", page) is None


def test_resolve_handles_leading_dollar_and_trailing_percent() -> None:
    page = make_page("Gross margin 27.3% on revenue $15,295 today")

    dollar = resolve("$15,295", page)
    percent = resolve("27.3%", page)

    assert dollar is not None
    assert page.text[dollar.char_start : dollar.char_end] == "$15,295"
    assert percent is not None
    assert page.text[percent.char_start : percent.char_end] == "27.3%"


def test_resolve_requires_normalized_text() -> None:
    # DS-W3-1 dependency: "3,817" resolves against normalized page text, but
    # returns None against the raw "3 ,817" — a correct fail-closed result, and
    # the reason DS-W3-1 must run before the resolver.
    normalized = make_page("Gross Margin 3,817 thousand")
    raw = make_page("Gross Margin 3 ,817 thousand")

    span = resolve("3,817", normalized)
    assert span is not None
    assert normalized.text[span.char_start : span.char_end] == "3,817"

    assert resolve("3,817", raw) is None


def test_resolve_is_exact_not_fuzzy() -> None:
    # A near-miss that is not a verbatim substring must never resolve.
    page = make_page("Revenue $15,295 total")
    assert resolve("15295", page) is None  # comma dropped
    assert resolve("$15295", page) is None  # comma dropped
    assert resolve("$15,295.00", page) is None  # extra trailing characters


def test_resolve_substring_that_is_present_resolves_but_prefix_ambiguity_does_not() -> None:
    # "15,295" is a genuine substring of "$15,295" and appears once -> resolves.
    page = make_page("Revenue $15,295 total")
    inner = resolve("15,295", page)
    assert inner is not None
    assert page.text[inner.char_start : inner.char_end] == "15,295"

    # "Gross Margin" is a prefix of "Gross Margin %": on a page carrying both it
    # appears twice, so it is ambiguous and fails closed. This is the real PTL
    # p.11 shape (a Gross Margin row and a Gross Margin % row).
    page2 = make_page("Gross Margin\nGross Margin %")
    assert resolve("Gross Margin", page2) is None


def test_resolve_overlapping_occurrences_are_ambiguous() -> None:
    page = make_page("aaa")
    assert resolve("aa", page) is None


def test_resolve_multiline_quote_returns_one_bbox_per_line() -> None:
    page = make_page("Revenue\n$15,295")

    span = resolve("Revenue\n$15,295", page)

    assert span is not None
    # Two visual lines -> two line boxes, each on a different vertical band.
    assert len(span.line_bboxes) == 2
    top_line, bottom_line = span.line_bboxes
    assert bottom_line.top > top_line.top
    # The coarse union envelope spans both lines.
    assert span.bbox.top == top_line.top
    assert span.bbox.bottom == bottom_line.bottom
    # ...and is taller than either individual line box (it covers the gap).
    assert (span.bbox.bottom - span.bbox.top) > (top_line.bottom - top_line.top)


def test_resolve_matches_a_quote_that_wraps_a_line() -> None:
    # The extractor emits the quote with a space, but the value wraps in the
    # source (a real '\n' in page.text). Whitespace-flexible matching resolves it
    # instead of silently dropping the fact.
    page = make_page("Total debt\nto net worth")

    span = resolve("Total debt to net worth", page)

    assert span is not None
    assert page.text[span.char_start : span.char_end] == "Total debt\nto net worth"
    assert len(span.line_bboxes) == 2
    assert span.line_bboxes[1].top > span.line_bboxes[0].top


def test_resolve_is_flexible_across_any_whitespace_run() -> None:
    # Multiple spaces (or mixed whitespace) in the source still resolve; only
    # whitespace is flexible, so this is not fuzzy matching.
    page = make_page("Gross   Margin")
    span = resolve("Gross Margin", page)
    assert span is not None
    assert page.text[span.char_start : span.char_end] == "Gross   Margin"

    # ...but a non-whitespace difference still fails closed.
    assert resolve("Gross-Margin", page) is None


def test_union_bbox_takes_min_and_max_over_boxes() -> None:
    chars = [
        CharBox(char="a", x0=10.0, top=5.0, x1=15.0, bottom=12.0, page=2, precision="word"),
        CharBox(char="b", x0=15.0, top=6.0, x1=22.0, bottom=14.0, page=2, precision="word"),
        CharBox(char="c", x0=3.0, top=4.0, x1=9.0, bottom=11.0, page=2, precision="word"),
    ]

    box = union_bbox(chars)

    assert box.x0 == 3.0
    assert box.top == 4.0
    assert box.x1 == 22.0
    assert box.bottom == 14.0
    assert box.page == 2


def test_union_bbox_rejects_empty() -> None:
    with pytest.raises(ValueError):
        union_bbox([])


def test_resolve_span_bbox_page_matches_page() -> None:
    page = make_page("only $42 here", page_no=9)
    span = resolve("$42", page)
    assert span is not None
    assert span.bbox.page == 9
    assert all(b.page == 9 for b in span.line_bboxes)


# --------------------------------------------------------------------------- #
# Real-corpus acceptance (the ticket's DS-W3-3 acceptance tests).
# --------------------------------------------------------------------------- #


def _ptl_pdf_path() -> Path | None:
    # Opt-in via PARSER_LOCAL_CORPUS_DIR (the confidential CIM is never committed),
    # matching the rest of the parser suite. Unset -> the local_corpus tests skip.
    root = os.environ.get("PARSER_LOCAL_CORPUS_DIR")
    if not root:
        return None
    path = Path(root) / "1st-app-h-ptl/1st-App-H-PTL-Group-CIM.pdf"
    return path if path.exists() else None


@pytest.fixture(scope="module")
def ptl_page_11() -> PageIndex:
    pdf_path = _ptl_pdf_path()
    if not pdf_path or not pdf_path.exists():
        pytest.skip(
            "Real PTL CIM not available on this machine (confidential document, "
            "never committed to this repo)."
        )
    from parser_service.docling_parser import parse_pdf_bytes

    result = parse_pdf_bytes(pdf_path.read_bytes())
    # PDF page index 11 is the income statement (0-indexed 10).
    return result.pages[10]


@pytest.mark.local_corpus
def test_ptl_page_11_resolves_verified_values(ptl_page_11: PageIndex) -> None:
    page = ptl_page_11
    for value in ["$15,295", "27.3%", "3,817"]:
        assert page.text.count(value) == 1, f"expected {value!r} to be unique on PDF-page 11"
        span = resolve(value, page)
        assert span is not None, f"{value!r} should resolve"
        assert span.page == 11
        assert page.text[span.char_start : span.char_end] == value
        assert span.bbox.x1 > span.bbox.x0
        assert span.bbox.bottom > span.bbox.top
        assert span.line_bboxes


@pytest.mark.local_corpus
def test_ptl_page_11_missing_quote_returns_none(ptl_page_11: PageIndex) -> None:
    assert resolve("not-on-this-page-zzzptl", ptl_page_11) is None


@pytest.mark.local_corpus
def test_ptl_page_11_ambiguous_value_fails_closed(ptl_page_11: PageIndex) -> None:
    # 21.2% appears on the Gross Margin % row for multiple forecast years, so it
    # is genuinely ambiguous on the page and must fail closed.
    page = ptl_page_11
    if page.text.count("21.2%") > 1:
        assert resolve("21.2%", page) is None
    else:
        pytest.skip("21.2% is not duplicated in this parse; ambiguity covered by fast tests")


# --------------------------------------------------------------------------- #
# resolve_in_cell — citing a value the page repeats.
# --------------------------------------------------------------------------- #


def _cell_over(page: PageIndex, start: int, end: int, *, row: int = 0, col: int = 0):
    """A cell whose box encloses exactly page.char_map[start:end]."""
    boxes = page.char_map[start:end]
    return TableCellRecord(
        row=row,
        col=col,
        row_span=1,
        col_span=1,
        text=page.text[start:end],
        text_normalized=page.text[start:end],
        column_header=False,
        row_header=False,
        page=page.page,
        x0=min(b.x0 for b in boxes),
        top=min(b.top for b in boxes),
        x1=max(b.x1 for b in boxes),
        bottom=max(b.bottom for b in boxes),
        bbox_source="docling_native",
    )


def test_resolve_in_cell_cites_a_value_the_page_repeats() -> None:
    # The gap DS-W3-8's harness surfaced on PTL p.11: "21.2%" three times in one
    # row. Page-wide the question has three answers, so resolve() must fail
    # closed and all three real figures are dropped. Scoped to the cell it has
    # exactly one -- the caller always knew which cell it read.
    page = make_page("Gross Margin % 27.3% 21.2% 21.2% 21.2%")
    assert page.text.count("21.2%") == 3
    assert resolve("21.2%", page) is None, "page-wide must stay ambiguous"

    second = page.text.index("21.2%", page.text.index("21.2%") + 1)
    span = resolve_in_cell("21.2%", page, _cell_over(page, second, second + 5))

    assert span is not None
    assert (span.char_start, span.char_end) == (second, second + 5)
    assert span.page == page.page
    assert span.bbox is not None


def test_resolve_in_cell_span_is_page_relative_not_cell_relative() -> None:
    # A citation must address the page the reader sees. A cell-local offset
    # would look plausible and point at the wrong text.
    page = make_page("Revenue 15,295 and Margin 21.2%")
    start = page.text.index("21.2%")
    span = resolve_in_cell("21.2%", page, _cell_over(page, start, start + 5))

    assert span is not None
    assert span.char_start == start
    assert page.text[span.char_start : span.char_end] == "21.2%"


def test_resolve_in_cell_fails_closed_when_the_cell_has_no_box() -> None:
    # No region means no way to scope the question, and a claim is not citable
    # on a cell whose own bounds are unknown.
    page = make_page("Margin 21.2%")
    cell = _cell_over(page, 7, 12)
    cell = cell.model_copy(update={"x0": None, "top": None, "x1": None, "bottom": None})

    assert resolve_in_cell("21.2%", page, cell) is None


def test_resolve_in_cell_fails_closed_when_the_quote_is_outside_the_cell() -> None:
    # The value is on the page, but not in THIS cell. Citing it anyway would
    # attribute a neighbour's number to this cell -- the wrong-span failure the
    # resolver exists to prevent.
    page = make_page("A 27.3% B 21.2%")
    other = page.text.index("27.3%")

    assert resolve_in_cell("21.2%", page, _cell_over(page, other, other + 5)) is None


def test_resolve_in_cell_ignores_a_repeat_outside_its_own_box() -> None:
    # Two identical values; the cell encloses one. Scoping must pick that one
    # and must not see the other as an ambiguity.
    page = make_page("21.2% then later 21.2%")
    first = page.text.index("21.2%")
    span = resolve_in_cell("21.2%", page, _cell_over(page, first, first + 5))

    assert span is not None
    assert (span.char_start, span.char_end) == (first, first + 5)

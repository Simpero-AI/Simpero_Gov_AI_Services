"""DS-W3-4 scale capture tests.

Two layers, like the rest of the parser suite:
- Fast, CI-portable tests exercising normalize_financial_token and
  determine_scale's resolution order (inline -> column_header -> page_header
  -> assumed_1x) against synthetic PageIndex/TableRecord fixtures.
- local_corpus-marked acceptance tests resolving the ticket's PTL PDF-page 11
  case ($15,295 under a "CAD (in Thousands)" header) against the real corpus.
"""

import shutil
from pathlib import Path

import pytest

from parser_service.scale import (
    ScaleResult,
    determine_scale,
    normalize_financial_token,
    scale_phrase_in_text,
)
from parser_service.schemas import CharBox, PageIndex, TableCellRecord, TableRecord


def _page(text: str, page_no: int = 1) -> PageIndex:
    char_map = [
        CharBox(char=ch, x0=0.0, top=0.0, x1=1.0, bottom=1.0, page=page_no, precision="word")
        for ch in text
    ]
    return PageIndex(page=page_no, text=text, char_map=char_map)


def _cell(row: int, col: int, text: str, text_normalized: str | None = None) -> TableCellRecord:
    return TableCellRecord(
        row=row,
        col=col,
        row_span=1,
        col_span=1,
        text=text,
        text_normalized=text_normalized if text_normalized is not None else text,
        column_header=(row == 0),
        row_header=(col == 0 and row > 0),
        page=1,
        x0=0.0,
        top=0.0,
        x1=1.0,
        bottom=1.0,
    )


def _table(cells: list[TableCellRecord], num_rows: int, num_cols: int) -> TableRecord:
    return TableRecord(
        page=1,
        num_rows=num_rows,
        num_cols=num_cols,
        cells=cells,
        cell_provenance_ok=True,
        header_row=0,
        column_headers_reliable=True,
    )


# --------------------------------------------------------------------------- #
# normalize_financial_token -- the ported inline (explicit_in_value) case.
# --------------------------------------------------------------------------- #


def test_inline_million_suffix() -> None:
    assert normalize_financial_token("$4.8M") == (4.8, 1_000_000.0, None)


def test_inline_thousand_suffix_no_currency_symbol() -> None:
    assert normalize_financial_token("500K") == (500.0, 1_000.0, None)


def test_inline_billion_suffix() -> None:
    assert normalize_financial_token("$2.1B") == (2.1, 1_000_000_000.0, None)


def test_inline_doubled_letter_mm_is_still_millions() -> None:
    # "MM" is the common oil & gas / accounting shorthand for millions -- the
    # ported grammar's continuation group accepts a repeated M/B letter as a
    # spelling variant, not a distinct multiplier.
    assert normalize_financial_token("$10MM") == (10.0, 1_000_000.0, None)


def test_inline_bn_mn_abbreviations_are_recognized() -> None:
    # "Bn"/"Mn" are common CIM notation the MVP's port dropped. The
    # continuation group now completes the abbreviation, and the "n" keeps the
    # required trailing word boundary intact. This is a recall gain with no
    # precision cost -- no non-magnitude token ends a number with "bn"/"mn".
    assert normalize_financial_token("$3Bn") == (3.0, 1_000_000_000.0, None)
    assert normalize_financial_token("$5Mn") == (5.0, 1_000_000.0, None)
    assert normalize_financial_token("3bn") == (3.0, 1_000_000_000.0, None)


def test_inline_spelled_out_word() -> None:
    assert normalize_financial_token("4.8 million") == (4.8, 1_000_000.0, None)


def test_inline_negative_sign_and_accounting_parens() -> None:
    assert normalize_financial_token("-$4.8M") == (-4.8, 1_000_000.0, None)
    assert normalize_financial_token("($4.8M)") == (-4.8, 1_000_000.0, None)


def test_inline_percent_is_self_scaling() -> None:
    assert normalize_financial_token("27.3%") == (27.3, 1.0, "%")


def test_inline_negative_percent() -> None:
    assert normalize_financial_token("-45%") == (-45.0, 1.0, "%")
    assert normalize_financial_token("(45%)") == (-45.0, 1.0, "%")


def test_inline_comma_formatted_number_with_suffix() -> None:
    assert normalize_financial_token("$1,200K") == (1200.0, 1_000.0, None)


def test_bare_number_has_no_inline_scale() -> None:
    # No suffix, no percent sign -> the multiplier is unknown, not 1. The
    # caller must resolve it from context; this must NOT be treated as an
    # implicit explicit_in_value multiplier of 1.
    assert normalize_financial_token("$15,295") is None
    assert normalize_financial_token("3,817") is None


def test_series_b_does_not_fire_the_billions_rule() -> None:
    # Adversarial case from the ticket: "B" in "Series B" is not adjacent to
    # any digit, so it can never be read as a billions suffix. Only "$10M"
    # is a candidate, and it resolves as millions.
    assert normalize_financial_token("Series B raised $10M") == (10.0, 1_000_000.0, None)
    # And with no dollar figure present at all, nothing fires.
    assert normalize_financial_token("Series B") is None


def test_unrecognized_unit_suffix_does_not_fire() -> None:
    # "kg" -- the K-pattern's continuation group only accepts "thousand", and
    # the trailing "g" breaks the required word boundary after "k". A weight
    # unit glued to a number must not be read as a scale suffix.
    assert normalize_financial_token("5kg") is None


def test_filing_reference_does_not_fire() -> None:
    # "10-K" -- the hyphen breaks suffix adjacency, so this is not read as
    # "10 thousand".
    assert normalize_financial_token("Form 10-K") is None


# --------------------------------------------------------------------------- #
# scale_phrase_in_text -- the public phrase lookup shared with DS-W3-5/6.
# --------------------------------------------------------------------------- #


def test_scale_phrase_in_text_returns_rightmost_phrase() -> None:
    # Nearest-wins, like the page-header rule: a later phrase supersedes an
    # earlier one, so the rightmost match is returned.
    assert scale_phrase_in_text("(in thousands) ... (in millions)") == (
        1_000_000.0,
        None,
        "(in millions)",
    )


def test_scale_phrase_in_text_captures_currency() -> None:
    assert scale_phrase_in_text("CAD (in Thousands)") == (1_000.0, "CAD", "CAD (in Thousands)")


def test_scale_phrase_in_text_returns_none_when_absent() -> None:
    assert scale_phrase_in_text("Revenue and gross margin, no scale phrase") is None


# --------------------------------------------------------------------------- #
# Non-dollar thousands markers.
#
# Only the dollar spellings were recognised, so a sterling document lost its
# scale outright: measured over a 102-page UK CIM, 89% of cited claims fell back
# to assumed_1x and every currency figure normalized a thousandth of its value.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("text", "currency"),
    [
        # How a UK statement writes it: in the column header cell, unparenthesised.
        ("£'000", "GBP"),
        ("£000", "GBP"),
        ("YEAR 1 £'000", "GBP"),
        ("PRO-FORMA 5 YEAR BALANCE SHEET £'000's", "GBP"),
        # Parenthesised, as it appears in prose and table titles.
        ("(£'000s)", "GBP"),
        ("(£000's)", "GBP"),
        ("(£ in thousands)", "GBP"),
        ("€'000", "EUR"),
        ("US$000", "USD"),
        # The dollar and bare spellings that already worked must keep working.
        ("($000s)", None),
        ("(000s)", None),
        ("(000)", None),
        ("'000", None),
    ],
)
def test_a_thousands_marker_is_read_whatever_its_currency(text: str, currency: str | None) -> None:
    found = scale_phrase_in_text(text)
    assert found is not None, f"{text!r} declares thousands"
    multiplier, unit, _context = found
    assert multiplier == 1_000.0
    # "£" names GBP unambiguously, so unlike a bare "$" it settles the unit too.
    assert unit == currency


@pytest.mark.parametrize(
    "text",
    [
        # A currency mark alone is not a scale declaration. This is the header of
        # the same document's appendix tables, which are denominated in FULL
        # pounds -- reading it as thousands would multiply every one of them.
        "£",
        "Jan-06 £",
        # An ordinary number contains "000" and must never declare a scale.
        "£1,000",
        "£42,012",
        "12,000",
        "1,000",
        "Wet Sales 42,012",
        "Total 1,059,922",
        "$1,000,000",
        "Room 1000",
    ],
)
def test_a_bare_currency_mark_or_a_plain_number_declares_no_scale(text: str) -> None:
    assert scale_phrase_in_text(text) is None


def test_a_parenthesised_marker_is_reported_once_at_its_widest() -> None:
    # "(£'000s)" contains "£'000", so both patterns match the same declaration.
    # It must be reported once, as the full parenthesised span, or the
    # nearest-wins rules could pick the inner one and truncate scale_context.
    assert scale_phrase_in_text("(£'000s)") == (1_000.0, "GBP", "(£'000s)")


# --------------------------------------------------------------------------- #
# determine_scale -- full resolution order.
# --------------------------------------------------------------------------- #


def test_determine_scale_explicit_in_value_short_circuits_context() -> None:
    page = _page("no scale phrase anywhere on this page $4.8M")
    result = determine_scale(
        "$4.8M", page, char_start=page.text.index("$4.8M"), value_type="currency"
    )

    assert result.scale_source == "explicit_in_value"
    assert result.scale_multiplier == 1_000_000.0
    assert result.normalized == 4_800_000.0
    assert result.flags == []


def test_determine_scale_percent_ignores_a_preceding_page_scale_header() -> None:
    # Regression for the correctness hazard this module exists to avoid: a
    # percentage on a "(in Thousands)" page must not be multiplied by 1000.
    text = "CAD (in Thousands)\nGross Margin % 27.3%"
    page = _page(text)
    result = determine_scale("27.3%", page, char_start=text.index("27.3%"), value_type="percent")

    assert result.scale_source == "explicit_in_value"
    assert result.scale_multiplier == 1.0
    assert result.normalized == 27.3
    assert result.unit == "%"


def test_determine_scale_page_header_found_before_value() -> None:
    text = "CAD (in Thousands)\nRevenue $15,295 total"
    page = _page(text)
    result = determine_scale(
        "$15,295", page, char_start=text.index("$15,295"), value_type="currency"
    )

    assert result.scale_source == "page_header"
    assert result.scale_multiplier == 1_000.0
    assert result.unit == "CAD"
    assert result.scale_context == "CAD (in Thousands)"
    assert result.normalized == 15_295_000.0
    assert result.flags == []


def test_determine_scale_page_header_strips_trailing_whitespace_in_context() -> None:
    # The verified corpus form: the scale phrase's own line renders with a
    # trailing space before the newline. scale_context must not carry it.
    text = "CAD (in Thousands) \nRevenue $15,295 total"
    page = _page(text)
    result = determine_scale(
        "$15,295", page, char_start=text.index("$15,295"), value_type="currency"
    )

    assert result.scale_context == "CAD (in Thousands)"


def test_determine_scale_paren_000_page_header() -> None:
    text = "Amounts in (000s)\nTotal $5,000"
    page = _page(text)
    result = determine_scale("$5,000", page, char_start=text.index("$5,000"), value_type="currency")

    assert result.scale_source == "page_header"
    assert result.scale_multiplier == 1_000.0
    assert result.unit is None
    assert result.scale_context == "(000s)"


def test_determine_scale_page_header_only_considers_text_before_the_value() -> None:
    # A scale phrase that appears AFTER the value must not apply to it.
    text = "Revenue $15,295 total, reported (in Millions) below"
    page = _page(text)
    result = determine_scale(
        "$15,295", page, char_start=text.index("$15,295"), value_type="currency"
    )

    assert result.scale_source == "assumed_1x"


def test_determine_scale_page_header_nearest_phrase_wins() -> None:
    # A later scale declaration on the same page supersedes an earlier one.
    text = "(in Thousands) ... (in Millions) ... Revenue $15,295"
    page = _page(text)
    result = determine_scale(
        "$15,295", page, char_start=text.index("$15,295"), value_type="currency"
    )

    assert result.scale_multiplier == 1_000_000.0
    assert result.scale_context == "(in Millions)"


def test_determine_scale_lowercase_word_before_phrase_is_not_read_as_currency() -> None:
    text = "the (in thousands) figure is $500"
    page = _page(text)
    result = determine_scale("$500", page, char_start=text.index("$500"), value_type="currency")

    assert result.scale_source == "page_header"
    assert result.unit is None
    assert result.scale_context == "(in thousands)"


def test_determine_scale_column_header_walk_finds_scale_above_value() -> None:
    header = _cell(0, 1, "(in Thousands)")
    value = _cell(1, 1, "$15,295")
    table = _table([header, value], num_rows=2, num_cols=2)
    page = _page("no page-level scale phrase here $15,295")

    result = determine_scale(
        "$15,295",
        page,
        char_start=page.text.index("$15,295"),
        value_type="currency",
        table=table,
        cell=value,
    )

    assert result.scale_source == "column_header"
    assert result.scale_multiplier == 1_000.0
    assert result.scale_context == "(in Thousands)"
    assert result.normalized == 15_295_000.0


def test_determine_scale_column_header_takes_precedence_over_page_header() -> None:
    header = _cell(0, 1, "(in Millions)")
    value = _cell(1, 1, "$15,295")
    table = _table([header, value], num_rows=2, num_cols=2)
    text = "CAD (in Thousands)\n$15,295"
    page = _page(text)

    result = determine_scale(
        "$15,295",
        page,
        char_start=text.index("$15,295"),
        value_type="currency",
        table=table,
        cell=value,
    )

    assert result.scale_source == "column_header"
    assert result.scale_multiplier == 1_000_000.0


def test_determine_scale_column_header_ignores_a_different_columns_header() -> None:
    # The PTL shape: col 0 carries "Income Statement CAD (in Thousands)" (the
    # row-label header), but the value's own column (col 1) header is just
    # "Trailing 5 Year Average" -- no scale phrase there, so the column walk
    # must not find one, and resolution falls through to the page level.
    row_label_header = _cell(0, 0, "Income Statement CAD (in Thousands)")
    value_col_header = _cell(0, 1, "Trailing 5 Year Average")
    value = _cell(1, 1, "$15,295")
    table = _table([row_label_header, value_col_header, value], num_rows=2, num_cols=2)
    text = "Income Statement CAD (in Thousands)\nRevenue $15,295"
    page = _page(text)

    result = determine_scale(
        "$15,295",
        page,
        char_start=text.index("$15,295", 10),
        value_type="currency",
        table=table,
        cell=value,
    )

    assert result.scale_source == "page_header"
    assert result.unit == "CAD"


def test_determine_scale_assumed_1x_is_flagged_never_silent() -> None:
    text = "Revenue $15,295 total, no scale phrase anywhere on this page"
    page = _page(text)
    result = determine_scale(
        "$15,295", page, char_start=text.index("$15,295"), value_type="currency"
    )

    assert result.scale_source == "assumed_1x"
    assert result.scale_multiplier == 1.0
    assert result.normalized == 15_295.0
    assert result.unit is None
    assert result.scale_context is None
    assert result.flags == ["scale_assumed"]


def test_determine_scale_raises_on_non_numeric_raw() -> None:
    page = _page("no digits here")
    with pytest.raises(ValueError):
        determine_scale("not-a-number", page, char_start=0, value_type="currency")


def test_scale_result_is_the_contract_shaped_model() -> None:
    result = determine_scale("$4.8M", _page("$4.8M"), char_start=0, value_type="currency")
    assert isinstance(result, ScaleResult)
    assert result.raw == "$4.8M"


# --------------------------------------------------------------------------- #
# value_type gate -- only currency is scaled by a header.
# --------------------------------------------------------------------------- #


def test_determine_scale_count_is_not_rescaled_by_page_header() -> None:
    # The headline correctness gate: a headcount on a "(in Thousands)" income
    # statement is 1,200 people, not 1,200,000. Only currency is header-scaled.
    text = "Summary financials (in Thousands)\nHeadcount 1200"
    page = _page(text)
    result = determine_scale("1200", page, char_start=text.index("1200"), value_type="count")

    assert result.scale_source == "explicit_in_value"
    assert result.scale_multiplier == 1.0
    assert result.normalized == 1200.0
    assert result.unit is None
    assert result.flags == []


def test_determine_scale_ratio_is_not_rescaled_by_page_header() -> None:
    text = "Leverage (in Thousands)\nDebt / EBITDA 1.2"
    page = _page(text)
    result = determine_scale("1.2", page, char_start=text.index("1.2"), value_type="ratio")

    assert result.scale_source == "explicit_in_value"
    assert result.scale_multiplier == 1.0
    assert result.normalized == 1.2
    assert result.unit == "ratio"


def test_determine_scale_percent_without_percent_sign_is_not_rescaled() -> None:
    # A percentage whose "%" lives in the column header, not the cell: the cell
    # value is a bare "27.3". value_type=percent still forbids header scaling
    # and restores the "%" unit.
    text = "Margins (in Thousands)\nGross Margin 27.3"
    page = _page(text)
    result = determine_scale("27.3", page, char_start=text.index("27.3"), value_type="percent")

    assert result.scale_source == "explicit_in_value"
    assert result.scale_multiplier == 1.0
    assert result.normalized == 27.3
    assert result.unit == "%"


def test_determine_scale_date_is_not_rescaled() -> None:
    text = "Fiscal years (in Thousands): 2024"
    page = _page(text)
    result = determine_scale("2024", page, char_start=text.index("2024"), value_type="date")

    assert result.scale_source == "explicit_in_value"
    assert result.scale_multiplier == 1.0
    assert result.normalized == 2024.0
    assert result.unit is None


def test_determine_scale_text_value_type_is_rejected() -> None:
    # value_type=text has no numeric magnitude; the contract makes its
    # normalized value null, so scale capture does not apply.
    page = _page("North America")
    with pytest.raises(ValueError):
        determine_scale("North America", page, char_start=0, value_type="text")


def test_determine_scale_count_ignores_a_column_scale_header() -> None:
    # Even a column-header scale phrase does not scale a count: the header
    # declares the magnitude of the money in that column, not a count that
    # happens to share it.
    header = _cell(0, 1, "(in Thousands)")
    value = _cell(1, 1, "1200")
    table = _table([header, value], num_rows=2, num_cols=2)
    page = _page("no page-level scale phrase here 1200")

    result = determine_scale(
        "1200",
        page,
        char_start=page.text.index("1200"),
        value_type="count",
        table=table,
        cell=value,
    )

    assert result.scale_source == "explicit_in_value"
    assert result.scale_multiplier == 1.0
    assert result.normalized == 1200.0


def test_determine_scale_bracketed_negative_currency_is_negative() -> None:
    # Accounting-negative notation without a suffix: the bare-number fallback
    # must honor the "(...)" so a loss is not read as a gain. Under a
    # "(in Thousands)" header the scaled magnitude stays negative.
    text = "CAD (in Thousands)\nNet income ($15,295)"
    page = _page(text)
    result = determine_scale(
        "($15,295)", page, char_start=text.index("($15,295)"), value_type="currency"
    )

    assert result.scale_source == "page_header"
    assert result.scale_multiplier == 1_000.0
    assert result.normalized == -15_295_000.0


def test_determine_scale_column_header_honors_a_merged_banner_span() -> None:
    # A "CAD (in Thousands)" banner merged across the value columns is stored
    # at its start column (col 1) with col_span=3. A value in col 3 must still
    # find it -- the exact-column walk would miss it.
    banner = _cell(0, 1, "CAD (in Thousands)").model_copy(update={"col_span": 3})
    value = _cell(1, 3, "$15,295")
    table = _table([banner, value], num_rows=2, num_cols=4)
    page = _page("no page-level scale phrase here $15,295")

    result = determine_scale(
        "$15,295",
        page,
        char_start=page.text.index("$15,295"),
        value_type="currency",
        table=table,
        cell=value,
    )

    assert result.scale_source == "column_header"
    assert result.scale_multiplier == 1_000.0
    assert result.unit == "CAD"
    assert result.scale_context == "CAD (in Thousands)"


# --------------------------------------------------------------------------- #
# Real-corpus acceptance (the ticket's DS-W3-4 acceptance tests).
# --------------------------------------------------------------------------- #


def _ptl_pdf_path() -> Path | None:
    dest_dir = Path(__file__).parent / "test_data"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / "1st-App-H-PTL-Group-CIM.pdf"
    if not dest_path.exists():
        src = Path("p:/simpero_GOV_AI/scripts/examples/1st-app-h-ptl/1st-App-H-PTL-Group-CIM.pdf")
        if src.exists():
            shutil.copy(src, dest_path)
    return dest_path if dest_path.exists() else None


@pytest.fixture(scope="module")
def ptl_page_11_and_tables() -> tuple[PageIndex, list[TableRecord]]:
    pdf_path = _ptl_pdf_path()
    if not pdf_path or not pdf_path.exists():
        pytest.skip(
            "Real PTL CIM not available on this machine (confidential document, "
            "never committed to this repo)."
        )
    from parser_service.docling_parser import parse_pdf_bytes
    from parser_service.table_extract import extract_tables, tables_on_page

    result = parse_pdf_bytes(pdf_path.read_bytes())
    assert result.document is not None
    tables = extract_tables(result.document, result.pages)
    # PDF page index 11 is the income statement (0-indexed 10).
    return result.pages[10], tables_on_page(tables, 11)


def _cell_by_label(table: TableRecord, label: str, col: int) -> TableCellRecord:
    row = next(c.row for c in table.cells if c.col == 0 and c.text.strip() == label)
    return next(c for c in table.cells if c.row == row and c.col == col)


@pytest.mark.local_corpus
def test_ptl_page_11_revenue_scales_from_page_header(
    ptl_page_11_and_tables: tuple[PageIndex, list[TableRecord]],
) -> None:
    page, tables = ptl_page_11_and_tables
    assert len(tables) == 1
    table = tables[0]

    revenue_cell = _cell_by_label(table, "Revenue", col=1)
    assert revenue_cell.text_normalized == "$15,295"

    from parser_service.resolver import resolve

    span = resolve("$15,295", page)
    assert span is not None

    result = determine_scale(
        "$15,295",
        page,
        char_start=span.char_start,
        value_type="currency",
        table=table,
        cell=revenue_cell,
    )

    assert result.scale_source == "page_header"
    assert result.scale_context == "CAD (in Thousands)"
    assert result.unit == "CAD"
    assert result.scale_multiplier == 1_000.0
    assert result.normalized == 15_295_000.0


@pytest.mark.local_corpus
def test_ptl_page_11_gross_margin_percent_is_not_rescaled(
    ptl_page_11_and_tables: tuple[PageIndex, list[TableRecord]],
) -> None:
    page, _tables = ptl_page_11_and_tables
    from parser_service.resolver import resolve

    span = resolve("27.3%", page)
    assert span is not None

    result = determine_scale("27.3%", page, char_start=span.char_start, value_type="percent")

    assert result.scale_source == "explicit_in_value"
    assert result.normalized == 27.3
    assert result.unit == "%"


# --------------------------------------------------------------------------- #
# normalize_financial_token — digitless tokens are not numbers.
# --------------------------------------------------------------------------- #


def test_time_of_day_is_not_a_million() -> None:
    # PTL PDF-page 15 carries a bid deadline: "2:00 p.m. NDT June 14, 2018".
    # The number group used to be "[\d,.]+", which matches a bare ".", so the
    # "." in "p.m." parsed as the number and the "m" as the millions suffix --
    # float(".") then raised ValueError and took the whole run down. A date is
    # not a financial token; it must simply not match.
    assert normalize_financial_token("2:00 p.m. NDT June 14, 2018") is None


@pytest.mark.parametrize(
    "text",
    [
        "p.m.",
        "a.m.",
        ".",
        ".M",
        "..K",
        "e.g. B",
        "N.B. see notes",
    ],
)
def test_digitless_tokens_never_parse_as_numbers(text: str) -> None:
    # A suffix letter next to punctuation is prose, not a magnitude. Anything
    # without a digit must return None rather than reaching float().
    assert normalize_financial_token(text) is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("$4.8M", (4.8, 1_000_000.0, None)),
        ("$.5M", (0.5, 1_000_000.0, None)),  # leading decimal is still a number
        ("500K", (500.0, 1_000.0, None)),
        ("$3Bn", (3.0, 1_000_000_000.0, None)),
        ("1.5MM", (1.5, 1_000_000.0, None)),
        ("4.8 million", (4.8, 1_000_000.0, None)),
        ("($4.8M)", (-4.8, 1_000_000.0, None)),
        ("27.3%", (27.3, 1.0, "%")),
    ],
)
def test_real_scale_tokens_still_parse(text: str, expected: tuple) -> None:
    # The guard above must not cost recall on the notations CIMs actually use.
    assert normalize_financial_token(text) == expected


# --------------------------------------------------------------------------- #
# Scale phrases with the currency INSIDE the parens -- "($ in millions)".
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("text", "multiplier", "currency"),
    [
        # The spelling that was missed. A real Bear Stearns CIM prints this
        # throughout; without it every currency figure normalized 1,000,000x too
        # small, flagged assumed_1x.
        ("($ in millions)", 1_000_000.0, None),
        ("($ in thousands)", 1_000.0, None),
        # An unambiguous symbol or code inside the parens does name a currency.
        ("(US$ in millions)", 1_000_000.0, "USD"),
        ("(C$ in thousands)", 1_000.0, "CAD"),
        ("(£ in millions)", 1_000_000.0, "GBP"),
        ("(€ in millions)", 1_000_000.0, "EUR"),
        ("(USD in thousands)", 1_000.0, "USD"),
        # Still-supported spellings must not regress.
        ("CAD (in Thousands)", 1_000.0, "CAD"),
        ("(in millions)", 1_000_000.0, None),
        ("(000s)", 1_000.0, None),
        ("($000s)", 1_000.0, None),
        # A scale phrase that wraps across a line break -- real documents do
        # this, and \s+ must span the newline.
        ("(in\nmillions)", 1_000_000.0, None),
    ],
)
def test_scale_phrase_spellings(text: str, multiplier: float, currency: str | None) -> None:
    found = scale_phrase_in_text(text)
    assert found is not None, f"{text!r} should be recognized as a scale phrase"
    assert found[0] == multiplier
    assert found[1] == currency


def test_bare_dollar_names_a_magnitude_but_not_a_currency() -> None:
    # "$" could be USD, CAD, AUD... The multiplier is trustworthy, the currency
    # is not. Returning None here is what makes the caller raise ambiguous_unit
    # instead of inventing a confident, wrong unit.
    found = scale_phrase_in_text("($ in millions)")
    assert found is not None
    multiplier, currency, _ = found
    assert multiplier == 1_000_000.0
    assert currency is None


def test_lowercase_word_is_still_not_read_as_a_currency_code() -> None:
    # The [A-Z]{3} slots stay case-sensitive: a lowercase three-letter word next
    # to a scale phrase must not be captured as a currency.
    found = scale_phrase_in_text("the (in thousands) note")
    assert found is not None
    multiplier, currency, _ = found
    assert multiplier == 1_000.0
    assert currency is None

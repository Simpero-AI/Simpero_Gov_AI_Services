"""DS-W3 table extractor tests.

Fast, CI-portable tests against synthetic PageIndex/TableRecord fixtures (same
convention as test_resolver.py / test_scale.py / test_emit.py), plus a
local_corpus-marked run over the real PTL CIM.

What is deliberately NOT tested here: whether the heuristic picks GOOD
attributes. It picks the table's own words, and "good" is the real extractor's
problem. What is tested is that it never fabricates a citation, never invents a
name for a value the table cannot label, and never silently drops a cell.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from parser_service.emit import FlagLog, PdfLocation
from parser_service.extract import (
    attribute_for,
    claims_from_table,
    infer_value_type_for,
    is_confident_currency,
    resolve_period,
    section_banners,
)
from parser_service.schemas import CharBox, PageIndex, TableCellRecord, TableRecord


def make_page(text: str, page_no: int = 1) -> PageIndex:
    char_map: list[CharBox] = []
    x = 0.0
    top = 0.0
    for ch in text:
        if ch == "\n":
            char_map.append(
                CharBox(
                    char="\n",
                    x0=x,
                    top=top,
                    x1=x,
                    bottom=top + 10.0,
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
                    bottom=top + 10.0,
                    page=page_no,
                    precision="word",
                )
            )
            x += 5.0
    return PageIndex(page=page_no, text=text, char_map=char_map)


def _cell(row: int, col: int, text: str) -> TableCellRecord:
    return TableCellRecord(
        row=row,
        col=col,
        row_span=1,
        col_span=1,
        text=text,
        text_normalized=text,
        column_header=(row == 0),
        row_header=(col == 0),
        page=1,
        x0=0.0,
        top=0.0,
        x1=1.0,
        bottom=1.0,
        bbox_source="docling_native",
    )


def _table(
    cells: list[TableCellRecord],
    *,
    header_row: int | None = 0,
    headers_reliable: bool = True,
) -> TableRecord:
    return TableRecord(
        page=1,
        num_rows=max((c.row for c in cells), default=0) + 1,
        num_cols=max((c.col for c in cells), default=0) + 1,
        cells=cells,
        cell_provenance_ok=True,
        header_row=header_row,
        column_headers_reliable=headers_reliable,
    )


# --------------------------------------------------------------------------- #
# value_type: the substring collisions this module exists to prevent.
#
# Every case here is a real line item. Matching the vocabularies as substrings
# rather than whole words stripped the scale header off genuine money and
# reported it a millionth of its value, next to correctly-scaled siblings.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("raw", "attribute"),
    [
        # "count" hides inside "ac-count-s". These four rows shipped as 4.4,
        # 5.2, 5.9 and 6.7 on a "($ in millions)" balance sheet.
        ("$5.2", "Accounts payable | 2004"),
        ("(2.3 )", "Accounts payable and accrued expenses | 2005"),
        ("$10.1", "Accounts receivable, net | 2003"),
        # "date" hides inside "consoli-date-d" and "liqui-date-d".
        ("1,234.5", "Consolidated net revenues | 2005"),
        ("12.0", "Liquidated damages | 2004"),
        # "unit" hides inside "comm-unit-y" and "opport-unit-y".
        ("3.4", "Community reinvestment expense | 2005"),
        # "machine" hides inside "Machine-ry".
        ("88.1", "Machinery and equipment | 2004"),
        # A metric noun outranks a count noun in the same label.
        ("201.0", "Member's Equity | 2004"),
        ("28.2", "Member contribution | 2004"),
        ("2.5", "Customer list, net | 2005"),
        ("47.2", "Property and equipment, net | 2003"),
        # "acquired" is a genuine word here, so anchoring alone does not save
        # it -- the value is not date-shaped, which is what settles it.
        ("(109.4 )", "Acq. of Flamingo Laughlin, net of cash acquired | 2004"),
        ("0.3", "Cash acquired from subsidiary contributed by parent | 2003"),
    ],
)
def test_money_is_not_mistyped_by_a_word_that_merely_contains_a_count_or_date_word(
    raw: str, attribute: str
) -> None:
    assert infer_value_type_for(raw, attribute) == "currency"


@pytest.mark.parametrize(
    ("raw", "attribute", "expected"),
    [
        # Trailing "net" is the accounting qualifier -- net of depreciation or
        # allowance -- and the value is money either way.
        ("2.5", "Customer list, net | 2005", "currency"),
        ("47.2", "Property and equipment, net | 2003", "currency"),
        # Leading "net" qualifies the noun that follows, and that noun is often
        # countable. Typing these currency would let a "($ in millions)" header
        # multiply a room count by a million.
        ("1,200", "Net rooms added", "count"),
        ("4,500", "Net new subscribers | 2024", "count"),
        ("312", "Net units shipped", "count"),
        # A real metric noun still wins from either position.
        ("88.1", "Net revenues | 2005", "currency"),
        ("9.1", "Net cash provided by operating activities | 2004", "currency"),
    ],
)
def test_net_names_an_amount_only_when_it_trails(raw: str, attribute: str, expected: str) -> None:
    assert infer_value_type_for(raw, attribute) == expected


@pytest.mark.parametrize(
    ("raw", "attribute", "expected"),
    [
        # A period CAPTION says when a figure was measured. It arrives as a
        # column header appended to every attribute in its table, so reading its
        # "Year"/"Months" as a count strips the scale header off every money row
        # underneath -- a subtotal shipping 1,000,000x smaller than the line it
        # totals, unflagged.
        ("2,400", "Total | Year Ended December 31,", "currency"),
        ("143.1", "Casino | Year Ended December 31, 2006", "currency"),
        ("73.5", "Total | Twelve Months Ended December 31, 2006", "currency"),
        ("88.2", "Food and beverage | Fiscal Year 2005", "currency"),
        # A SPAN of time is still counted.
        ("7", "ACEP Tenure (In Years)", "count"),
        ("25", "General Manager Tenure Years", "count"),
        ("12", "Lease term in months", "count"),
        # A count noun with no metric noun must not de-scale a real expense line.
        ("1,234", "Employee benefits | 2005", "currency"),
    ],
)
def test_a_period_caption_is_not_a_count_of_periods(
    raw: str, attribute: str, expected: str
) -> None:
    assert infer_value_type_for(raw, attribute) == expected


def test_vocabularies_do_not_contain_words_they_document_as_excluded() -> None:
    """A mechanical rewrite of these lists once scraped quoted words out of the
    surrounding COMMENTS and into the vocabulary they were documenting as
    excluded -- the singular "year" landed in _COUNT_NOUNS from a comment saying
    it was deliberately left out. Nothing about that failure is visible in a
    diff or in a corpus measurement, so it is asserted directly.
    """
    from parser_service.extract import (
        _COUNT_NOUNS,
        _DURATION_NOUNS,
        _METRIC_NOUNS,
        _PERIOD_CAPTION_WORDS,
    )

    # "year" singular spells a period caption, never a count.
    assert "year" not in _COUNT_NOUNS
    assert "year" not in _DURATION_NOUNS
    # The tiers are consulted in order, so an overlap would make the later one
    # unreachable for that word.
    assert not _METRIC_NOUNS & _COUNT_NOUNS
    assert not _METRIC_NOUNS & _DURATION_NOUNS
    assert not _COUNT_NOUNS & _DURATION_NOUNS
    assert not _DURATION_NOUNS & _PERIOD_CAPTION_WORDS


@pytest.mark.parametrize(
    ("raw", "attribute", "expected"),
    [
        # A value that declares its own unit needs no label at all.
        ("27.3%", "Gross Margin", "percent"),
        ("150 bps", "Spread over LIBOR", "percent"),
        ("7.5x", "TEV / EBITDA", "ratio"),
        ("$15,295", "Revenue | 2019F", "currency"),
        ("1,300 square feet", "Stratosphere", "count"),
        ("4 acres", "Site", "count"),
        # No "$", but on a financial page a bare grouped number is money --
        # PTL p.11's Gross Margin row prints exactly this. Typing it `count`
        # would refuse the header scale and show it 1000x too small.
        ("4,171", "Gross Margin", "currency"),
        ("$77.3", "Cash and cash equivalents | 2003", "currency"),
        # Genuine counts stay counts however the table is denominated.
        ("1,309", "Stratosphere | Slot Machines", "count"),
        ("2,444", "Stratosphere | Hotel Rooms (1)", "count"),
        ("2,444", "Hotel Rooms", "count"),
        ("49", "Table Games", "count"),
        ("80,000", "Gaming Square Footage", "count"),
        ("22,154", "Conventions Held", "count"),
        ("412", "Licensed Beds", "count"),
        ("1,284", "Number of units", "count"),
        # A duration is counted, not priced.
        ("7", "ACEP Tenure (In Years)", "count"),
        # Dates need BOTH a date label and a date-shaped value.
        ("1998", "Date Acquired", "date"),
        ("1998", "Stratosphere | Date Acquired", "date"),
        ("March '07", "Completion Date of Recent Renovation", "date"),
        # A bare year with an unhelpful label is still a year, not $2bn.
        ("2006", "Aquarius", "date"),
        # No magnitude to scale: emit.py gives these a null normalized rather
        # than inventing a number from whatever digits parse first.
        ("four", "propertyCount", "text"),
        ("-", "Member contribution | 2003", "text"),
        ("N/A", "Occupancy | 2005", "text"),
        ("In 2001 and 2002, the Company expanded", "slotFloorExpansion", "text"),
    ],
)
def test_value_type_is_judged_from_the_value_and_its_attribute(
    raw: str, attribute: str, expected: str
) -> None:
    assert infer_value_type_for(raw, attribute) == expected


# --------------------------------------------------------------------------- #
# resolve_period -- SIM-345
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("column_header", "expected_year", "expected_kind"),
    [
        # The ticket's own acceptance figures.
        ("2019F", 2019, "P"),
        ("FY19E", 2019, "E"),
        ("2020A", 2020, "A"),
        # "P" spells the same forward-looking kind as "F".
        ("2020P", 2020, "P"),
        # "FY" with a full four-digit year, with and without a separating space.
        ("FY2019E", 2019, "E"),
        ("FY 2019E", 2019, "E"),
        # A footnote marker trailing the suffix.
        ("2008E (1)", 2008, "E"),
        # An unqualified year names the year with no kind -- the header does not
        # say which, so nothing is guessed.
        ("2020", 2020, None),
    ],
)
def test_resolve_period_reads_the_column_header_suffix(
    column_header: str, expected_year: int | None, expected_kind: str | None
) -> None:
    assert resolve_period(column_header) == (expected_year, expected_kind)


@pytest.mark.parametrize(
    "column_header",
    [
        # LTM/TTM name a trailing period this contract has no slot for yet
        # (deferred; see resolve_period's docstring) -- unresolved, not guessed.
        "LTM",
        "TTM",
        "LTM 2020",
        # A bare two-digit number with no "FY" marker is too easy to collide
        # with an unrelated count or footnote number to trust as a year.
        "19",
        "19E",
        # Not period-shaped at all.
        "Hotel Rooms",
        "",
    ],
)
def test_resolve_period_does_not_guess_what_it_cannot_read(column_header: str) -> None:
    assert resolve_period(column_header) == (None, None)


# --------------------------------------------------------------------------- #
# attribute_for
# --------------------------------------------------------------------------- #


def test_attribute_combines_row_label_and_column_header() -> None:
    cells = [_cell(0, 0, ""), _cell(0, 1, "2019F"), _cell(1, 0, "Revenue"), _cell(1, 1, "$15,295")]
    table = _table(cells)
    assert attribute_for(table, cells[3]) == "Revenue | 2019F"


def test_attribute_is_row_label_alone_when_columns_are_unlabeled() -> None:
    # header_row=None is DS-W3-2 saying this table has no header row to key
    # off. The row label alone is weaker but honest; inventing a column name
    # would not be.
    cells = [_cell(0, 0, "Revenue"), _cell(0, 1, "$15,295")]
    table = _table(cells, header_row=None)
    assert attribute_for(table, cells[1]) == "Revenue"


def test_attribute_uses_the_structural_header_even_when_docling_flags_disagree() -> None:
    # column_headers_reliable reports whether DOCLING's per-cell column_header
    # markers agree with the structural inference. It is a diagnostic about
    # Docling, not a verdict on header_row -- and on every financial statement in
    # a CIM those markers disagree while the structural header is perfectly good.
    #
    # This flag used to gate the attribute, so the column was dropped and all
    # four years of a line item collapsed onto one name: four rows reading
    # "Cash and cash equivalents" with four different numbers. Whether a header
    # can be trusted is _infer_header_row's call, made structurally.
    cells = [
        _cell(0, 0, ""),
        _cell(0, 1, "2003"),
        _cell(1, 0, "Cash and cash equivalents"),
        _cell(1, 1, "$77.3"),
    ]
    table = _table(cells, headers_reliable=False)
    assert attribute_for(table, cells[3]) == "Cash and cash equivalents | 2003"


# --------------------------------------------------------------------------- #
# In-table section banners.
# --------------------------------------------------------------------------- #


def _pl_table() -> TableRecord:
    """A P&L shaped like the real one: banner rows, and subtotals with no label.

    r0            YEAR 1   YEAR 2
    r1  TURNOVER                        <- banner
    r2    Coffee Shop     41       92
    r3                   577    1,309   <- section subtotal, no label
    r4  COST OF SALES                   <- banner
    r5    Coffee Shop     31       71
    """
    return _table(
        [
            _cell(0, 0, ""),
            _cell(0, 1, "YEAR 1"),
            _cell(0, 2, "YEAR 2"),
            _cell(1, 0, "TURNOVER"),
            _cell(2, 0, "Coffee Shop"),
            _cell(2, 1, "41"),
            _cell(2, 2, "92"),
            _cell(3, 0, ""),
            _cell(3, 1, "577"),
            _cell(3, 2, "1,309"),
            _cell(4, 0, "COST OF SALES"),
            _cell(5, 0, "Coffee Shop"),
            _cell(5, 1, "31"),
            _cell(5, 2, "71"),
        ]
    )


def test_a_label_row_with_no_figures_governs_the_rows_below_it() -> None:
    banners = section_banners(_pl_table())
    assert banners[2] == "TURNOVER"
    assert banners[3] == "TURNOVER"
    assert banners[5] == "COST OF SALES"
    # The banner rows are not governed by themselves.
    assert 1 not in banners
    assert 4 not in banners


def test_a_row_with_any_figure_is_data_however_it_is_titled() -> None:
    # "GROSS PROFIT 312" reads like a heading and is not one.
    table = _table(
        [
            _cell(0, 0, ""),
            _cell(0, 1, "YEAR 1"),
            _cell(1, 0, "GROSS PROFIT"),
            _cell(1, 1, "312"),
            _cell(2, 0, "Margin"),
            _cell(2, 1, "54.0%"),
        ]
    )
    assert section_banners(table) == {}


def test_the_banner_separates_two_rows_that_would_otherwise_be_one_claim() -> None:
    # "Coffee Shop | YEAR 1" appears twice on a real P&L with different numbers:
    # once as revenue, once as the cost of earning it. Without the banner the
    # two claims are the same string.
    table = _pl_table()
    banners = section_banners(table)
    revenue = next(c for c in table.cells if c.row == 2 and c.col == 1)
    cost = next(c for c in table.cells if c.row == 5 and c.col == 1)

    assert attribute_for(table, revenue, banners.get(2)) == "TURNOVER | Coffee Shop | YEAR 1"
    assert attribute_for(table, cost, banners.get(5)) == "COST OF SALES | Coffee Shop | YEAR 1"


def test_the_banner_names_a_subtotal_row_the_table_left_unlabelled() -> None:
    # A section subtotal is printed as figures with an empty label cell, so it
    # had no attribute and was dropped -- 431 cells on one real CIM, including
    # every turnover and cost-of-sales total.
    table = _pl_table()
    subtotal = next(c for c in table.cells if c.row == 3 and c.col == 1)
    assert attribute_for(table, subtotal, "TURNOVER") == "TURNOVER | YEAR 1"


def test_an_unlabelled_row_with_no_banner_still_has_no_name() -> None:
    # Fail closed: a value nothing in the table can name has no attribute, and
    # inventing one would be worse than the recall gap.
    table = _pl_table()
    subtotal = next(c for c in table.cells if c.row == 3 and c.col == 1)
    assert attribute_for(table, subtotal, None) is None


def test_the_banner_makes_a_countable_noun_read_as_the_money_it_is() -> None:
    # "Coffee Shop" is a revenue line on a hospitality P&L, but "shop" is a
    # count noun, so alone it typed as a count, refused the scale header and
    # shipped 41 where the truth was 41,000 -- with no flag, because a
    # non-currency type self-scales at a KNOWN 1.0.
    assert infer_value_type_for("41", "Coffee Shop | YEAR 1") == "count"
    assert infer_value_type_for("41", "TURNOVER | Coffee Shop | YEAR 1") == "currency"
    assert infer_value_type_for("31", "COST OF SALES | Coffee Shop | YEAR 1") == "currency"


def test_claims_from_table_reads_rows_under_their_banner() -> None:
    page = make_page("TURNOVER Coffee Shop 41 92 577 1,309 COST OF SALES Coffee Shop 31 71")
    claims = claims_from_table(
        _pl_table(), page, entity="BarWash", file="bw.pdf", flag_log=FlagLog()
    )
    attributes = [c.attribute for c in claims]

    assert "TURNOVER | Coffee Shop | YEAR 1" in attributes
    assert "COST OF SALES | Coffee Shop | YEAR 1" in attributes
    # The previously-nameless subtotal is now claimed.
    assert "TURNOVER | YEAR 1" in attributes


def test_attribute_is_none_when_the_row_has_no_label() -> None:
    # A value whose own table cannot say what it is has no attribute to claim.
    cells = [_cell(0, 0, ""), _cell(0, 1, "2019F"), _cell(1, 0, "   "), _cell(1, 1, "$15,295")]
    table = _table(cells)
    assert attribute_for(table, cells[3]) is None


# --------------------------------------------------------------------------- #
# claims_from_table
# --------------------------------------------------------------------------- #


def test_emits_one_cited_claim_per_numeric_cell() -> None:
    page = make_page("CAD (in Thousands)\n2019F Revenue $15,295 Margin 27.3%")
    cells = [
        _cell(0, 0, ""),
        _cell(0, 1, "2019F"),
        _cell(1, 0, "Revenue"),
        _cell(1, 1, "$15,295"),
        _cell(2, 0, "Margin"),
        _cell(2, 1, "27.3%"),
    ]
    claims = claims_from_table(
        _table(cells), page, entity="PTL Group", file="ptl.pdf", flag_log=FlagLog()
    )

    assert [c.attribute for c in claims] == ["Revenue | 2019F", "Margin | 2019F"]
    assert all(c.entity == "PTL Group" for c in claims)
    assert all(c.status == "proposed" for c in claims)
    # The currency took the page header; the percent did NOT — it self-scales.
    assert claims[0].value.normalized == 15_295_000.0
    assert claims[0].value.scale_source == "page_header"
    assert claims[1].value.normalized == 27.3
    for claim in claims:
        location = claim.location
        assert isinstance(location, PdfLocation)
        assert location.char_start is not None and location.char_end is not None
        # SIM-345: both cells sit under the same "2019F" column header.
        assert claim.period_year == 2019
        assert claim.period_kind == "P"


def test_skips_labels_headers_and_prose() -> None:
    # The header row and label column NAME things; a prose cell in a financial
    # table is context. None of them is a claim.
    page = make_page("2019F Revenue $15,295 Notes see appendix")
    cells = [
        _cell(0, 0, ""),
        _cell(0, 1, "2019F"),
        _cell(1, 0, "Revenue"),
        _cell(1, 1, "$15,295"),
        _cell(2, 0, "Notes"),
        _cell(2, 1, "see appendix"),
    ]
    claims = claims_from_table(_table(cells), page, entity="E", file="f.pdf", flag_log=FlagLog())
    assert [c.attribute for c in claims] == ["Revenue | 2019F"]


def test_a_superscript_marked_cell_is_not_mistaken_for_a_figure() -> None:
    # The gate that decides "does this cell hold a number" used
    # `any(ch.isdigit() for ch in text)`, which is True for superscripts and
    # circled digits while scale.py's "\d" is not. A unit like "m2" written with
    # a superscript two, or a circled footnote mark, therefore passed the gate,
    # was typed currency by the default, and reached determine_scale -- which
    # raises on it, uncaught, taking the run with it. Both gates now ask the
    # parser that will read the value.
    page = make_page("2019F Floor area m\u00b2 Revenue $15,295")
    cells = [
        _cell(0, 0, ""),
        _cell(0, 1, "2019F"),
        _cell(1, 0, "Floor area"),
        _cell(1, 1, "m\u00b2"),
        _cell(2, 0, "Revenue"),
        _cell(2, 1, "$15,295"),
    ]
    claims = claims_from_table(_table(cells), page, entity="E", file="f.pdf", flag_log=FlagLog())

    assert [c.attribute for c in claims] == ["Revenue | 2019F"], (
        "the superscript cell names a unit, not a magnitude"
    )


def test_an_uncitable_cell_is_missing_not_dropped() -> None:
    # The value is not on the page, so it cannot be cited. It must come back
    # `missing` rather than vanish: a dropped cell is invisible, a missing claim
    # is a recall gap you can see.
    page = make_page("nothing numeric on this page at all")
    cells = [_cell(0, 0, ""), _cell(0, 1, "2019F"), _cell(1, 0, "Revenue"), _cell(1, 1, "$99,999")]
    flag_log = FlagLog()
    claims = claims_from_table(_table(cells), page, entity="E", file="f.pdf", flag_log=flag_log)

    assert len(claims) == 1
    assert claims[0].status == "missing"
    location = claims[0].location
    assert isinstance(location, PdfLocation)
    assert location.char_start is None and location.char_end is None
    assert [e.flag_type for e in flag_log.entries] == ["quote_unresolved"]
    # SIM-345: the period was read off the column header, not the resolved
    # span, so it survives even though the value itself could not be cited.
    assert claims[0].period_year == 2019
    assert claims[0].period_kind == "P"


# --------------------------------------------------------------------------- #
# Real corpus.
# --------------------------------------------------------------------------- #


def _ptl_pdf_path() -> Path | None:
    root = os.environ.get("PARSER_LOCAL_CORPUS_DIR")
    if not root:
        return None
    path = Path(root) / "1st-App-H-PTL-Group-CIM.pdf"
    return path if path.exists() else None


@pytest.mark.local_corpus
def test_ptl_income_statement_yields_valid_claims() -> None:
    pdf_path = _ptl_pdf_path()
    if pdf_path is None:
        pytest.skip("PARSER_LOCAL_CORPUS_DIR not set or PTL CIM not present")

    import json

    from jsonschema import Draft202012Validator

    from parser_service.docling_parser import parse_pdf_bytes
    from parser_service.table_extract import extract_tables, tables_on_page

    schema_path = Path(__file__).parent.parent / "contracts" / "claims.schema.json"
    validator = Draft202012Validator(json.loads(schema_path.read_text()))

    result = parse_pdf_bytes(pdf_path.read_bytes())
    assert result.document is not None, "a successful parse always carries the DoclingDocument"
    page = next(p for p in result.pages if p.page == 11)
    table = tables_on_page(extract_tables(result.document), 11)[0]
    claims = claims_from_table(table, page, entity="PTL Group", file="ptl.pdf", flag_log=FlagLog())

    assert claims, "p.11 is the income statement; it must yield claims"
    for claim in claims:
        errors = list(validator.iter_errors(claim.to_json()))
        assert not errors, "\n".join(e.message for e in errors)

    # The ticket's own acceptance figure, through the full extract path.
    revenue = next(c for c in claims if c.attribute.startswith("Revenue |"))
    assert revenue.value.normalized == 15_295_000.0
    assert revenue.value.scale_source == "page_header"


def test_is_confident_currency_positive_signals_and_default() -> None:
    # An inline currency mark makes it confident regardless of the label.
    assert is_confident_currency("$13.6", "Property Summary | Stratosphere")
    # A metric-noun label makes it confident even with no mark on the value.
    assert is_confident_currency("15,295", "TURNOVER | Coffee Shop | YEAR 1")
    # A bare count with neither a mark nor a metric noun is NOT confident: this is
    # the SIM-323 case, where the value typed currency only by the fallthrough
    # default and must not be allowed to bind a page banner.
    assert not is_confident_currency("1,309", "Property Summary | Stratosphere")
    assert not is_confident_currency("80,000", "Property Summary | Aquarius")


def test_a_two_row_header_stacks_into_the_column_label_and_types_the_count() -> None:
    # "Hotel" and "Rooms" wrap across two header rows; folded, the column label is
    # "Hotel Rooms", whose count noun types the column count rather than currency.
    cells = [
        _cell(0, 0, "Property"),
        _cell(0, 1, "Slots"),
        _cell(0, 2, "Hotel"),
        _cell(1, 1, "(1)"),
        _cell(1, 2, "Rooms"),
        _cell(2, 0, "Stratosphere"),
        _cell(2, 1, "1,309"),
        _cell(2, 2, "2,444"),
    ]
    table = TableRecord(
        page=1,
        num_rows=3,
        num_cols=3,
        cells=cells,
        cell_provenance_ok=True,
        header_row=0,
        header_continuation=[1],
        column_headers_reliable=False,
    )
    room = next(c for c in cells if c.row == 2 and c.col == 2)
    attr = attribute_for(table, room)
    assert attr is not None and attr.endswith("Hotel Rooms")
    assert infer_value_type_for("2,444", attr) == "count"

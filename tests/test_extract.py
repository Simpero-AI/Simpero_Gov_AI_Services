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
    infer_value_type,
    infer_value_type_for,
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
# infer_value_type
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("27.3%", "percent"),
        ("$15,295", "currency"),
        # No "$", but on a financial page a bare grouped number is money —
        # PTL p.11's Gross Margin row prints exactly this. Typing it `count`
        # would refuse the header scale and show it 1000x too small.
        ("4,171", "currency"),
    ],
)
def test_infer_value_type(raw: str, expected: str) -> None:
    assert infer_value_type(raw) == expected


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
# value_type from the row label -- what stops a scale header hitting a count.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("raw", "label", "expected"),
    [
        # Counts stay counts however the table is denominated. Before this, a
        # "($ in millions)" header turned each of these into billions.
        ("1,309", "Slot Machines", "count"),
        ("2,444", "Hotel Rooms", "count"),
        ("49", "Table Games", "count"),
        ("80,000", "Gaming Square Footage", "count"),
        ("7", "ACEP Tenure (In Years)", "count"),
        # Dates are periods, not amounts.
        ("1998", "Date Acquired", "date"),
        ("March '07", "Completion Date of Recent Renovation", "date"),
        # A bare year with an unhelpful label is still a year, not $2bn.
        ("2006", "Aquarius", "date"),
        # Money is still money -- including the bare form with no "$", which is
        # the PTL case this module must not regress.
        ("$15,295", "Revenue", "currency"),
        ("4,171", "Gross Margin", "currency"),
        ("$77.3", "Cash and cash equivalents", "currency"),
        # Percent wins regardless of label.
        ("27.3%", "Gross Margin %", "percent"),
        # The metric can live in the COLUMN instead of the row -- a property
        # summary is laid out that way, and reading only the row label
        # ("Stratosphere") would type every count as currency.
        ("1,309", "Stratosphere | Slot Machines", "count"),
        ("2,444", "Stratosphere | Hotel Rooms (1)", "count"),
        ("1998", "Stratosphere | Date Acquired", "date"),
    ],
)
def test_value_type_is_judged_from_the_attribute(raw: str, label: str, expected: str) -> None:
    assert infer_value_type_for(raw, label) == expected


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

"""Fast, CI-portable unit tests for DS-W3-2 table extraction.

Complements test_table_fidelity.py (which needs the real, confidential CIM
corpus and a ~6-minute parse). These tests use lightweight synthetic
stand-ins for Docling's TableItem so they run in milliseconds on any machine
with no fixtures — and, critically, exercise paths the real corpus can't:
the cell_provenance_ok=False branch (this corpus has zero missing bboxes) and
the extractor's error handling.
"""

import logging
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest
from pydantic import ValidationError

from parser_service.schemas import (
    CharBox,
    PageIndex,
    TableCellRecord,
    TableRecord,
)
from parser_service.table_extract import (
    _build_table_record,
    _infer_header_row,
    extract_tables,
    reconstruct_bbox,
    tables_on_page,
)

if TYPE_CHECKING:
    from docling_core.types.doc.document import (
        DoclingDocument,  # pyright: ignore[reportPrivateImportUsage]
        TableItem,  # pyright: ignore[reportPrivateImportUsage]
    )


def _bbox(left: float, top: float, right: float, bottom: float) -> SimpleNamespace:
    return SimpleNamespace(l=left, t=top, r=right, b=bottom)


def _cell(
    row: int,
    col: int,
    text: str,
    bbox: SimpleNamespace | None,
    *,
    row_span: int = 1,
    col_span: int = 1,
) -> SimpleNamespace:
    return SimpleNamespace(
        start_row_offset_idx=row,
        start_col_offset_idx=col,
        row_span=row_span,
        col_span=col_span,
        text=text,
        column_header=(row == 0),
        row_header=(col == 0 and row > 0),
        bbox=bbox,
    )


def _table(page_no: int, num_rows: int, num_cols: int, cells: list) -> "TableItem":
    # Duck-typed stand-in: _build_table_record only reads .prov/.data fields.
    # Cast so the type checker accepts it as the TableItem the function expects.
    return cast(
        "TableItem",
        SimpleNamespace(
            prov=[SimpleNamespace(page_no=page_no)],
            data=SimpleNamespace(num_rows=num_rows, num_cols=num_cols, table_cells=cells),
        ),
    )


def test_build_table_record_all_cells_have_bbox_sets_provenance_ok() -> None:
    table = _table(
        3,
        2,
        2,
        [
            _cell(0, 0, "H1", _bbox(0, 0, 10, 5)),
            _cell(0, 1, "H2", _bbox(10, 0, 20, 5)),
            _cell(1, 0, "A", _bbox(0, 5, 10, 10)),
            _cell(1, 1, "B", _bbox(10, 5, 20, 10)),
        ],
    )
    record = _build_table_record(table, None)

    assert record.cell_provenance_ok is True
    assert record.page == 3
    assert record.num_rows == 2
    assert record.num_cols == 2
    assert all(c.x0 is not None for c in record.cells)


def test_build_table_record_missing_bbox_flags_provenance_and_nulls_coords() -> None:
    # The DS-W3-2 guarantee: a cell with no source coordinate must not be
    # silently shipped as citable. This is the branch the real corpus never
    # hits (it has full coverage), so it is only covered here.
    table = _table(
        7,
        2,
        2,
        [
            _cell(0, 0, "H1", _bbox(0, 0, 10, 5)),
            _cell(0, 1, "H2", _bbox(10, 0, 20, 5)),
            _cell(1, 0, "no-coord", None),
            _cell(1, 1, "B", _bbox(10, 5, 20, 10)),
        ],
    )
    record = _build_table_record(table, None)

    assert record.cell_provenance_ok is False
    missing = [(c.row, c.col) for c in record.cells if c.x0 is None]
    assert missing == [(1, 0)]
    bad_cell = next(c for c in record.cells if c.row == 1 and c.col == 0)
    assert bad_cell.x0 is None
    assert bad_cell.top is None
    assert bad_cell.x1 is None
    assert bad_cell.bottom is None


def test_build_table_record_propagates_page_span_and_header_flags() -> None:
    table = _table(
        4,
        2,
        2,
        [
            _cell(0, 0, "Merged", _bbox(0, 0, 20, 5), col_span=2),
            _cell(1, 0, "RowLabel", _bbox(0, 5, 10, 10)),
            _cell(1, 1, "Val", _bbox(10, 5, 20, 10)),
        ],
    )
    record = _build_table_record(table, None)

    assert {c.page for c in record.cells} == {4}
    merged = next(c for c in record.cells if c.text == "Merged")
    assert merged.col_span == 2
    assert merged.column_header is True
    row_label = next(c for c in record.cells if c.text == "RowLabel")
    assert row_label.row_header is True
    # bbox corners map to the schema's x0/top/x1/bottom (l/t/r/b).
    assert (merged.x0, merged.top, merged.x1, merged.bottom) == (0.0, 0.0, 20.0, 5.0)


def test_tables_on_page_filters_by_page_number() -> None:
    def rec(page: int) -> TableRecord:
        return TableRecord(
            page=page,
            num_rows=0,
            num_cols=0,
            cells=[],
            cell_provenance_ok=True,
            header_row=None,
            column_headers_reliable=False,
        )

    tables = [rec(11), rec(17), rec(11), rec(20)]
    assert [t.page for t in tables_on_page(tables, 11)] == [11, 11]
    assert [t.page for t in tables_on_page(tables, 17)] == [17]
    assert tables_on_page(tables, 99) == []


def test_extract_tables_builds_one_record_per_docling_table() -> None:
    doc = cast(
        "DoclingDocument",
        SimpleNamespace(
            tables=[
                _table(11, 1, 1, [_cell(0, 0, "Revenue", _bbox(1, 1, 2, 2))]),
                _table(17, 1, 1, [_cell(0, 0, "Ratio", _bbox(3, 3, 4, 4))]),
            ]
        ),
    )
    records = extract_tables(doc)
    assert [r.page for r in records] == [11, 17]
    assert all(isinstance(r, TableRecord) for r in records)


def test_extract_tables_returns_empty_for_a_document_with_no_tables() -> None:
    doc = cast("DoclingDocument", SimpleNamespace(tables=[]))
    assert extract_tables(doc) == []


def _valid_cell_kwargs() -> dict:
    return dict(
        row=0,
        col=0,
        row_span=1,
        col_span=1,
        text="x",
        text_normalized="x",
        column_header=False,
        row_header=False,
        page=1,
        x0=0.0,
        top=0.0,
        x1=1.0,
        bottom=1.0,
    )


def test_table_cell_record_rejects_negative_indices() -> None:
    valid = _valid_cell_kwargs()
    with pytest.raises(ValidationError):
        TableCellRecord(**{**valid, "row": -1})
    with pytest.raises(ValidationError):
        TableCellRecord(**{**valid, "col": -1})
    with pytest.raises(ValidationError):
        TableCellRecord(**{**valid, "page": 0})


def test_table_cell_record_rejects_zero_span() -> None:
    valid = _valid_cell_kwargs()
    with pytest.raises(ValidationError):
        TableCellRecord(**{**valid, "row_span": 0})
    with pytest.raises(ValidationError):
        TableCellRecord(**{**valid, "col_span": 0})


def test_table_cell_record_allows_null_coordinates() -> None:
    # A provenance-missing cell is representable (coords None) — the record
    # itself is still valid; it is cell_provenance_ok on the parent table that
    # signals the problem.
    cell = TableCellRecord(
        **{**_valid_cell_kwargs(), "x0": None, "top": None, "x1": None, "bottom": None}
    )
    assert cell.x0 is None


def test_cell_text_is_normalized_but_raw_text_is_preserved() -> None:
    # QA finding F2: Docling's raw cell text keeps split numeric tokens; the
    # extractor must expose a normalized value for fact parsing WITHOUT
    # destroying the verbatim source text (still needed for provenance).
    table = _table(
        11,
        2,
        2,
        [
            _cell(0, 0, "Income Statement", _bbox(0, 0, 10, 5)),
            _cell(0, 1, "2019F", _bbox(10, 0, 20, 5)),
            _cell(1, 0, "Gross Margin", _bbox(0, 5, 10, 10)),
            _cell(1, 1, "3 ,817", _bbox(10, 5, 20, 10)),
        ],
    )
    record = _build_table_record(table, None)

    split = next(c for c in record.cells if c.row == 1 and c.col == 1)
    assert split.text == "3 ,817", "verbatim source text must be preserved"
    assert split.text_normalized == "3,817", "fact parsing reads the normalized value"

    # Cells with nothing to normalize carry the value through unchanged.
    label = next(c for c in record.cells if c.text == "Gross Margin")
    assert label.text_normalized == "Gross Margin"


def test_cell_normalization_uses_the_same_rule_as_the_page_index() -> None:
    # The tightened rule (F1) must apply to table cells too — a spaced list in a
    # cell must not be corrupted into a number.
    table = _table(
        1,
        1,
        2,
        [
            _cell(0, 0, "Notes 3 , 4", _bbox(0, 0, 10, 5)),
            _cell(0, 1, "$2 ,094", _bbox(10, 0, 20, 5)),
        ],
    )
    record = _build_table_record(table, None)

    notes = next(c for c in record.cells if c.col == 0)
    currency = next(c for c in record.cells if c.col == 1)
    assert notes.text_normalized == "Notes 3 , 4", "spaced list must not collapse"
    assert currency.text_normalized == "$2,094", "currency-prefixed split token must collapse"


def test_header_row_inferred_from_structure_when_labels_present() -> None:
    # QA finding F3: headers come from structure. Row 0 carries non-numeric
    # value-column labels => it is the header row, and Docling's flags agree.
    table = _table(
        11,
        2,
        3,
        [
            _cell(0, 0, "Income Statement", _bbox(0, 0, 10, 5)),
            _cell(0, 1, "2018F", _bbox(10, 0, 20, 5)),
            _cell(0, 2, "2019F", _bbox(20, 0, 30, 5)),
            _cell(1, 0, "Revenue", _bbox(0, 5, 10, 10)),
            _cell(1, 1, "15295", _bbox(10, 5, 20, 10)),
            _cell(1, 2, "17146", _bbox(20, 5, 30, 10)),
        ],
    )
    record = _build_table_record(table, None)

    assert record.header_row == 0
    assert record.column_headers_reliable is True


def test_header_row_is_none_when_columns_are_unlabeled() -> None:
    # The Pitchbook p.17 shape: a section label in col 0, no real header, and
    # Docling wrongly flagging the first DATA row's numeric cells as headers.
    # Structure must win: header_row=None and the flag declared untrustworthy,
    # so downstream never invents column names out of data values.
    data_cells = [
        _cell(1, 0, "Accounts receivable turnover", _bbox(0, 5, 10, 10)),
        _cell(1, 1, "14.35", _bbox(10, 5, 20, 10)),
        _cell(1, 2, "83.04%", _bbox(20, 5, 30, 10)),
    ]
    # Reproduce Docling's mislabel: mark the data row's cells column_header=True.
    for cell in data_cells:
        cell.column_header = True

    table = _table(
        17,
        2,
        3,
        [_cell(0, 0, "Turnover Ratios:", _bbox(0, 0, 10, 5)), *data_cells],
    )
    record = _build_table_record(table, None)

    assert record.header_row is None, "a table with no labeled columns has no header row"
    assert record.column_headers_reliable is False, (
        "Docling flagged a data row as header — flags must be declared unreliable"
    )


def test_header_inference_rejects_numeric_first_row() -> None:
    # Even if row 0 has value-column cells, numeric ones mean it is data, not a
    # header (guards against a table whose header Docling dropped entirely).
    table = _table(
        5,
        1,
        3,
        [
            _cell(0, 0, "Revenue", _bbox(0, 0, 10, 5)),
            _cell(0, 1, "15,295", _bbox(10, 0, 20, 5)),
            _cell(0, 2, "17,146", _bbox(20, 0, 30, 5)),
        ],
    )
    record = _build_table_record(table, None)

    assert record.header_row is None
    assert record.column_headers_reliable is False


# --- Reconstruction fallback: recover a cell bbox from the positioned index ---


def _page_index(text: str, x0: float = 100.0, y: float = 200.0, w: float = 6.0) -> PageIndex:
    """A positioned index laying each character out left to right at one y."""
    boxes = [
        CharBox(
            char=ch,
            x0=x0 + i * w,
            top=y,
            x1=x0 + (i + 1) * w,
            bottom=y + 10.0,
            page=1,
            precision="word",
        )
        for i, ch in enumerate(text)
    ]
    return PageIndex(page=1, text=text, char_map=boxes)


def testreconstruct_bbox_unique_match_unions_char_boxes() -> None:
    page = _page_index("Revenue 3,817 total")
    bbox = reconstruct_bbox("3,817", page)
    assert bbox is not None
    x0, top, x1, bottom = bbox
    start = page.text.index("3,817")
    end = start + len("3,817") - 1
    assert x0 == page.char_map[start].x0
    assert x1 == page.char_map[end].x1
    assert bottom > top


def testreconstruct_bbox_fails_closed_on_absent_or_ambiguous() -> None:
    assert reconstruct_bbox("9,999", _page_index("Revenue 3,817 total")) is None
    assert reconstruct_bbox("3,817", _page_index("3,817 versus 3,817 again")) is None
    assert reconstruct_bbox("", _page_index("anything")) is None


def test_missing_bbox_is_reconstructed_from_page_index() -> None:
    # Docling gave this cell no bbox, but its (split-token) text resolves uniquely
    # in the positioned index, so the fallback recovers a citable coordinate.
    page = _page_index("Revenue 3,817 total")
    table = _table(
        1, 1, 2, [_cell(0, 0, "Revenue", _bbox(0, 0, 50, 10)), _cell(0, 1, "3 ,817", None)]
    )
    record = _build_table_record(table, page)

    recovered = next(c for c in record.cells if c.text == "3 ,817")
    assert recovered.bbox_source == "reconstructed"
    assert recovered.x0 is not None and recovered.x1 is not None and recovered.x1 > recovered.x0
    assert record.cell_provenance_ok is True
    assert record.reconstructed_cells == 1


def test_missing_bbox_unresolvable_stays_not_provenance_ok() -> None:
    page = _page_index("Revenue 3,817 total")
    table = _table(1, 1, 1, [_cell(0, 0, "value not on the page", None)])
    record = _build_table_record(table, page)

    assert record.cells[0].bbox_source is None
    assert record.cells[0].x0 is None
    assert record.cell_provenance_ok is False
    assert record.reconstructed_cells == 0


def test_reconstructed_cells_are_logged(caplog: pytest.LogCaptureFixture) -> None:
    # Recovering a cell's geometry from the flat index is an audit event.
    page = _page_index("Revenue 3,817 total")
    table = _table(
        1, 1, 2, [_cell(0, 0, "Revenue", _bbox(0, 0, 50, 10)), _cell(0, 1, "3 ,817", None)]
    )
    with caplog.at_level(logging.INFO):
        _build_table_record(table, page)
    assert any("reconstruction fallback" in r.getMessage() for r in caplog.records)


def test_uncitable_table_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    page = _page_index("Revenue 3,817 total")
    table = _table(1, 1, 1, [_cell(0, 0, "value not on the page", None)])
    with caplog.at_level(logging.WARNING):
        _build_table_record(table, page)
    assert any("not citable" in r.getMessage() for r in caplog.records)


def test_native_bbox_takes_precedence_over_reconstruction() -> None:
    page = _page_index("Revenue 3,817 total")
    table = _table(1, 1, 1, [_cell(0, 0, "3,817", _bbox(1, 2, 3, 4))])
    record = _build_table_record(table, page)

    assert record.cells[0].bbox_source == "docling_native"
    assert record.cells[0].x0 == 1
    assert record.reconstructed_cells == 0


def test_extract_tables_threads_page_indices_for_fallback() -> None:
    page = _page_index("Revenue 3,817 total")
    table = _table(1, 1, 1, [_cell(0, 0, "3,817", None)])
    doc = cast("DoclingDocument", SimpleNamespace(tables=[table]))

    without = extract_tables(doc)
    assert without[0].cell_provenance_ok is False  # no index -> native only

    with_index = extract_tables(doc, [page])
    assert with_index[0].cell_provenance_ok is True
    assert with_index[0].reconstructed_cells == 1


def _rec(row: int, col: int, text: str) -> TableCellRecord:
    """A TableCellRecord, which is what _infer_header_row reads (the module's
    other _cell helper builds Docling-shaped stand-ins for _build_table_record)."""
    return TableCellRecord(
        row=row,
        col=col,
        row_span=1,
        col_span=1,
        text=text,
        text_normalized=text,
        column_header=False,
        row_header=(col == 0),
        page=1,
        x0=0.0,
        top=0.0,
        x1=1.0,
        bottom=1.0,
        bbox_source="docling_native",
    )


def test_header_row_found_below_a_title_and_caption() -> None:
    """A financial statement stacks rows before its columns:

        row 0  Consolidated Balance Sheet        ($ in millions)   <- title
        row 1  As of December 31,                                  <- caption
        row 2           2003     2004                              <- the header
        row 3  Cash and cash equivalents  $77.3  $75.2             <- data

    Only row 2 spans the value columns, so only row 2 is the header. Taking row 0
    named every value after the table's title and collapsed both years of a line
    item onto one attribute.
    """
    cells = [
        _rec(0, 0, "Consolidated Balance Sheet"),
        _rec(0, 2, "($ in millions)"),
        _rec(1, 1, "As of December 31,"),
        _rec(2, 1, "2003"),
        _rec(2, 2, "2004"),
        _rec(3, 0, "Cash and cash equivalents"),
        _rec(3, 1, "$77.3"),
        _rec(3, 2, "$75.2"),
        _rec(4, 0, "Accounts receivable"),
        _rec(4, 1, "$10.1"),
        _rec(4, 2, "$11.4"),
    ]
    assert _infer_header_row(cells) == 2


def test_header_row_still_found_at_row_zero_for_a_plain_table() -> None:
    # The common shape must not regress: labels on row 0, data beneath.
    cells = [
        _rec(0, 0, "Property"),
        _rec(0, 1, "Slots"),
        _rec(0, 2, "Hotel Rooms"),
        _rec(1, 0, "Stratosphere"),
        _rec(1, 1, "1,309"),
        _rec(1, 2, "2,444"),
        _rec(2, 0, "Aquarius"),
        _rec(2, 1, "1,021"),
        _rec(2, 2, "1,907"),
    ]
    assert _infer_header_row(cells) == 0


def test_a_table_of_bare_figures_still_has_no_header() -> None:
    # The Pitchbook case: the first row spanning the value columns IS data, so
    # the table genuinely has no header and must say so rather than name columns
    # after data values.
    cells = [
        _rec(0, 0, "Segment A"),
        _rec(0, 1, "14.35"),
        _rec(0, 2, "83.04"),
        _rec(1, 0, "Segment B"),
        _rec(1, 1, "12.10"),
        _rec(1, 2, "77.90"),
    ]
    assert _infer_header_row(cells) is None

"""DS-W3-6 element processing tests.

Fast, CI-portable unit tests against synthetic TableRecord fixtures and
duck-typed Docling stand-ins (same convention as test_table_extract.py) --
no real corpus or Docling pipeline run needed.
"""

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

from services.parser.parser_service.elements import extract_chart_elements, extract_table_elements
from services.parser.parser_service.schemas import CharBox, PageIndex, TableCellRecord, TableRecord

if TYPE_CHECKING:
    from docling_core.types.doc.document import (
        DoclingDocument,  # pyright: ignore[reportPrivateImportUsage]
    )

# --------------------------------------------------------------------------- #
# Fixture builders.
# --------------------------------------------------------------------------- #


def _cell(
    row: int,
    col: int,
    text: str,
    *,
    row_span: int = 1,
    col_span: int = 1,
    x0: float | None = 0.0,
    top: float | None = 0.0,
    x1: float | None = 1.0,
    bottom: float | None = 1.0,
    text_normalized: str | None = None,
) -> TableCellRecord:
    return TableCellRecord(
        row=row,
        col=col,
        row_span=row_span,
        col_span=col_span,
        text=text,
        text_normalized=text_normalized if text_normalized is not None else text,
        column_header=(row == 0),
        row_header=False,
        page=1,
        x0=x0,
        top=top,
        x1=x1,
        bottom=bottom,
        bbox_source="docling_native" if x0 is not None else None,
    )


def _table(
    cells: list[TableCellRecord],
    num_rows: int,
    num_cols: int,
    *,
    page: int = 1,
    header_row: int | None = 0,
    cell_provenance_ok: bool = True,
) -> TableRecord:
    return TableRecord(
        page=page,
        num_rows=num_rows,
        num_cols=num_cols,
        cells=cells,
        cell_provenance_ok=cell_provenance_ok,
        header_row=header_row,
        column_headers_reliable=True,
    )


def _positioned_page(
    text: str, x0: float = 0.0, y: float = 0.0, w: float = 5.0, page_no: int = 1
) -> PageIndex:
    """A positioned index laying each character out left to right at one y."""
    char_map = [
        CharBox(
            char=ch,
            x0=x0 + i * w,
            top=y,
            x1=x0 + (i + 1) * w,
            bottom=y + 10.0,
            page=page_no,
            precision="word",
        )
        for i, ch in enumerate(text)
    ]
    return PageIndex(page=page_no, text=text, char_map=char_map)


def _picture_bbox(left: float, top: float, right: float, bottom: float) -> SimpleNamespace:
    return SimpleNamespace(l=left, t=top, r=right, b=bottom)


def _picture(
    page_no: int, bbox: tuple[float, float, float, float], caption: str = ""
) -> SimpleNamespace:
    prov = [SimpleNamespace(page_no=page_no, bbox=_picture_bbox(*bbox))]
    return SimpleNamespace(prov=prov, caption_text=lambda doc: caption)


def _doc(pictures: list[SimpleNamespace]) -> "DoclingDocument":
    # Duck-typed stand-in: extract_chart_elements only reads .pictures. Cast
    # so the type checker accepts it as the DoclingDocument the function
    # expects (same convention as test_table_extract.py's `_table`/`doc`).
    return cast("DoclingDocument", SimpleNamespace(pictures=pictures))


# --------------------------------------------------------------------------- #
# extract_table_elements -- ragged-row structural check.
# --------------------------------------------------------------------------- #


def test_well_formed_table_not_ragged_every_cell_has_value_and_bbox() -> None:
    cells = [
        _cell(0, 0, "Revenue"),
        _cell(0, 1, "2024"),
        _cell(1, 0, "Total"),
        _cell(1, 1, "15295"),
    ]
    table = _table(cells, num_rows=2, num_cols=2)

    elements = extract_table_elements([table])

    assert len(elements) == 1
    el = elements[0]
    assert el.kind == "table"
    assert el.ragged_table_rows is False
    assert el.flags == []
    assert el.cell_provenance_ok is True
    assert len(el.cells) == 4
    for cell in el.cells:
        assert cell.text
        assert cell.x0 is not None


def test_column_shifted_row_triggers_ragged_flag() -> None:
    # Row 1 is missing a cell -- a column shift dropped one entry, the one
    # failure mode the ticket names explicitly.
    cells = [
        _cell(0, 0, "A"),
        _cell(0, 1, "B"),
        _cell(0, 2, "C"),
        _cell(1, 0, "X"),
        _cell(1, 1, "Y"),  # col 2 missing
    ]
    table = _table(cells, num_rows=2, num_cols=3)

    el = extract_table_elements([table])[0]

    assert el.ragged_table_rows is True
    assert "ragged_table_rows" in el.flags


def test_missing_entire_row_triggers_ragged_flag() -> None:
    cells = [_cell(0, 0, "A"), _cell(0, 1, "B")]  # row 1 entirely absent
    table = _table(cells, num_rows=2, num_cols=2)

    el = extract_table_elements([table])[0]

    assert el.ragged_table_rows is True


def test_extra_cell_in_a_row_also_triggers_ragged_flag() -> None:
    cells = [
        _cell(0, 0, "A"),
        _cell(0, 1, "B"),
        _cell(1, 0, "X"),
        _cell(1, 1, "Y"),
        _cell(1, 2, "Z"),  # row 1 has one more cell than num_cols allows
    ]
    table = _table(cells, num_rows=2, num_cols=2)

    el = extract_table_elements([table])[0]

    assert el.ragged_table_rows is True


def test_merged_header_cell_is_not_ragged() -> None:
    # Regression: a legitimately merged header (col_span=2, one entry
    # covering two columns) used to be miscounted as a missing cell and
    # flagged ragged -- a false positive on a completely normal table shape
    # already exercised elsewhere in this codebase
    # (test_table_extract.py::test_build_table_record_propagates_page_span_and_header_flags).
    cells = [
        _cell(0, 0, "Merged Header", col_span=2),
        _cell(1, 0, "RowLabel"),
        _cell(1, 1, "Val"),
    ]
    table = _table(cells, num_rows=2, num_cols=2)

    el = extract_table_elements([table])[0]

    assert el.ragged_table_rows is False
    assert el.flags == []


def test_row_spanning_cell_is_not_ragged() -> None:
    # Same false-positive class, vertical this time: a cell spanning two
    # rows (row_span=2) means the second row it covers has no entry of its
    # own for that column -- legitimate, not a gap.
    cells = [
        _cell(0, 0, "Spans2Rows", row_span=2),
        _cell(0, 1, "A"),
        _cell(1, 1, "B"),
    ]
    table = _table(cells, num_rows=2, num_cols=2)

    el = extract_table_elements([table])[0]

    assert el.ragged_table_rows is False


# --------------------------------------------------------------------------- #
# extract_table_elements -- table-region scale capture (feeds DS-4).
# --------------------------------------------------------------------------- #


def test_table_scale_context_captured_from_header_row() -> None:
    cells = [
        _cell(0, 0, "Income Statement"),
        _cell(0, 1, "CAD (in Thousands)"),
        _cell(1, 0, "Revenue"),
        _cell(1, 1, "15295"),
    ]
    table = _table(cells, num_rows=2, num_cols=2)

    el = extract_table_elements([table])[0]

    assert el.scale_multiplier == 1000.0
    assert el.scale_unit == "CAD"
    assert el.scale_context == "CAD (in Thousands)"


def test_table_scale_context_none_when_no_phrase_present() -> None:
    cells = [
        _cell(0, 0, "Revenue"),
        _cell(0, 1, "2024"),
        _cell(1, 0, "Total"),
        _cell(1, 1, "15295"),
    ]
    table = _table(cells, num_rows=2, num_cols=2)

    el = extract_table_elements([table])[0]

    assert el.scale_multiplier is None
    assert el.scale_unit is None
    assert el.scale_context is None


def test_table_scale_context_none_when_header_row_unknown() -> None:
    # Unlabeled-column table (DS-2's Pitchbook p.17 shape): no header_row to
    # scan, so no scale is captured at the table-region level even if a
    # coincidental scale-looking phrase sits in row 0.
    cells = [_cell(0, 0, "A"), _cell(0, 1, "in Thousands")]
    table = _table(cells, num_rows=1, num_cols=2, header_row=None)

    el = extract_table_elements([table])[0]

    assert el.scale_multiplier is None


# --------------------------------------------------------------------------- #
# extract_table_elements -- bbox union and multi-table handling.
# --------------------------------------------------------------------------- #


def test_bbox_is_union_of_cell_bboxes() -> None:
    cells = [
        _cell(0, 0, "A", x0=0.0, top=0.0, x1=10.0, bottom=5.0),
        _cell(0, 1, "B", x0=10.0, top=0.0, x1=20.0, bottom=5.0),
        _cell(1, 0, "C", x0=0.0, top=5.0, x1=10.0, bottom=10.0),
        _cell(1, 1, "D", x0=10.0, top=5.0, x1=20.0, bottom=10.0),
    ]
    table = _table(cells, num_rows=2, num_cols=2)

    el = extract_table_elements([table])[0]

    assert el.bbox is not None
    assert (el.bbox.x0, el.bbox.top, el.bbox.x1, el.bbox.bottom) == (0.0, 0.0, 20.0, 10.0)
    assert el.bbox.page == 1


def test_bbox_none_when_no_cell_has_coordinates() -> None:
    cells = [_cell(0, 0, "x", x0=None, top=None, x1=None, bottom=None)]
    table = _table(cells, num_rows=1, num_cols=1, cell_provenance_ok=False)

    el = extract_table_elements([table])[0]

    assert el.bbox is None
    assert el.cell_provenance_ok is False


def test_multiple_tables_produce_one_element_each() -> None:
    t1 = _table([_cell(0, 0, "A")], num_rows=1, num_cols=1, page=3)
    t2 = _table([_cell(0, 0, "B")], num_rows=1, num_cols=1, page=7)

    elements = extract_table_elements([t1, t2])

    assert [e.page for e in elements] == [3, 7]


def test_extract_table_elements_empty_input() -> None:
    assert extract_table_elements([]) == []


# --------------------------------------------------------------------------- #
# extract_chart_elements -- charts flagged honestly, never guessed.
# --------------------------------------------------------------------------- #


def test_chart_element_flagged_and_captures_caption() -> None:
    doc = _doc(
        [
            _picture(
                page_no=1, bbox=(100.0, 100.0, 200.0, 200.0), caption="Figure 1: Revenue by quarter"
            )
        ]
    )

    elements = extract_chart_elements(doc)

    assert len(elements) == 1
    el = elements[0]
    assert el.kind == "chart"
    assert el.page == 1
    assert el.flags == ["chart_data_not_extracted"]
    assert el.caption_text == "Figure 1: Revenue by quarter"
    assert (el.bbox.x0, el.bbox.top, el.bbox.x1, el.bbox.bottom) == (100.0, 100.0, 200.0, 200.0)


def test_chart_element_captures_nearby_page_text() -> None:
    # Text laid out just outside the chart's own bbox, within the padded
    # search region -- catches axis labels / legends Docling didn't formally
    # caption-link.
    page = _positioned_page("Q1 Q2 Q3 Q4", x0=95.0, y=205.0)
    doc = _doc([_picture(page_no=1, bbox=(100.0, 100.0, 200.0, 200.0))])

    elements = extract_chart_elements(doc, [page])

    assert elements[0].surrounding_text == "Q1 Q2 Q3 Q4"


def test_chart_element_no_page_index_yields_empty_surrounding_text() -> None:
    doc = _doc([_picture(page_no=1, bbox=(0.0, 0.0, 10.0, 10.0))])

    elements = extract_chart_elements(doc)

    assert elements[0].surrounding_text == ""


def test_picture_without_provenance_is_skipped() -> None:
    picture_no_prov = SimpleNamespace(prov=[], caption_text=lambda doc: "orphan")
    doc = _doc([picture_no_prov])

    assert extract_chart_elements(doc) == []


def test_multiple_charts_across_pages() -> None:
    doc = _doc(
        [
            _picture(page_no=2, bbox=(0.0, 0.0, 10.0, 10.0), caption="chart A"),
            _picture(page_no=5, bbox=(0.0, 0.0, 10.0, 10.0), caption="chart B"),
        ]
    )

    elements = extract_chart_elements(doc)

    assert [e.page for e in elements] == [2, 5]
    assert [e.caption_text for e in elements] == ["chart A", "chart B"]


def test_extract_chart_elements_empty_document() -> None:
    assert extract_chart_elements(_doc([])) == []

"""DS-W3-6 element processing -- tables and charts as StoredElement records
with provenance.

Once DS-2 (table_extract.py) decides how Docling delivers a table (native
cells or the reconstruction fallback), this module turns each table region
into a structured, citable element and adds a free structural self-check the
underlying cell grid can't do for itself: row-length uniformity. A
column-shifted row (a wrapped cell knocking a row out of alignment) can look
like a perfectly well-formed grid and still be wrong -- two extraction
methods agreeing doesn't catch it, since both can share the same mistake, but
counting cells per row does.

Charts are handled honestly: capture the region, its caption, and nearby page
text, and flag chart_data_not_extracted. Plotted values are pixels, not
machine-read numbers -- this module never guesses one.
"""

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from .scale import scale_phrase_in_text
from .schemas import BBox, PageIndex, TableCellRecord, TableRecord

if TYPE_CHECKING:
    from docling_core.types.doc.document import (
        DoclingDocument,  # pyright: ignore[reportPrivateImportUsage]
    )

# How far (in source points) around a picture's own bbox to look for caption /
# axis-label / legend text that Docling didn't formally link as a caption.
_CHART_CONTEXT_PADDING = 50.0


class TableElement(BaseModel):
    kind: Literal["table"] = "table"
    page: int = Field(ge=1)
    # None only when no cell in the table has a resolved coordinate -- neither
    # the Docling-native path nor the reconstruction fallback placed any cell;
    # see TableRecord.cell_provenance_ok.
    bbox: BBox | None
    cells: list[TableCellRecord]
    cell_provenance_ok: bool
    # True when this table's rows don't all cover the same number of columns
    # -- the column-shift signal a naive grid gets confidently wrong. A
    # per-row cell count that doesn't match num_cols means Docling (or the
    # reconstruction fallback) misread the row's column boundaries.
    ragged_table_rows: bool
    # Scale phrase found in the table's own header row, if any (DS-W3-4's
    # phrase grammar, reused via scale.py's public scale_phrase_in_text) --
    # captured once here so DS-7's per-fact scale resolution doesn't have to
    # re-scan this table's header for every cell.
    scale_multiplier: float | None = None
    scale_unit: str | None = None
    scale_context: str | None = None
    flags: list[str] = Field(default_factory=list)


class ChartElement(BaseModel):
    kind: Literal["chart"] = "chart"
    page: int = Field(ge=1)
    bbox: BBox
    # Docling's own caption mechanism (FloatingItem.captions), resolved to text.
    caption_text: str
    # Page text near the chart's bbox that Docling did not formally link as a
    # caption -- catches axis labels / legends sitting just outside the
    # image region. Text only, never an interpretation of the plotted image.
    surrounding_text: str
    # Always ["chart_data_not_extracted"] -- plotted values are pixels, not
    # machine-read; this module never fabricates a number from a chart image.
    flags: list[str] = Field(default_factory=list)


StoredElement = TableElement | ChartElement


def _is_ragged(table: TableRecord) -> bool:
    """True if the cells don't tile the table's row/col grid exactly once
    per position -- the row-length uniformity / column-shift signal a naive
    grid gets confidently wrong.

    Counts covered *grid positions* (each cell fills row..row+row_span and
    col..col+col_span), not raw cell entries. A raw entry count breaks on any
    legitimately merged cell: a header cell with col_span=2 is ONE entry that
    covers two columns, so counting entries alone reads a normal merged
    table as missing a cell -- exactly the false positive this ticket's
    ragged check must not produce on well-formed tables (this shape is
    already exercised elsewhere in this codebase, see
    test_table_extract.py::test_build_table_record_propagates_page_span_and_header_flags).
    A genuinely ragged/column-shifted table still leaves a real gap (fewer
    covered positions than num_rows * num_cols) or an overlap (a position
    covered twice, which also produces fewer *unique* positions than
    expected), so both failure modes are still caught.
    """
    covered: set[tuple[int, int]] = set()
    for cell in table.cells:
        for row in range(cell.row, cell.row + cell.row_span):
            for col in range(cell.col, cell.col + cell.col_span):
                covered.add((row, col))
    return len(covered) != table.num_rows * table.num_cols


def _union_cell_bboxes(table: TableRecord) -> BBox | None:
    located = [c for c in table.cells if c.x0 is not None]
    if not located:
        return None
    return BBox(
        x0=min(c.x0 for c in located),  # type: ignore[type-var]
        top=min(c.top for c in located),  # type: ignore[type-var]
        x1=max(c.x1 for c in located),  # type: ignore[type-var]
        bottom=max(c.bottom for c in located),  # type: ignore[type-var]
        page=table.page,
    )


def _table_scale_context(table: TableRecord) -> tuple[float, str | None, str] | None:
    """The nearest scale phrase in the table's own header row, if any."""
    if table.header_row is None:
        return None
    for cell in table.cells:
        if cell.row != table.header_row:
            continue
        found = scale_phrase_in_text(cell.text_normalized)
        if found is not None:
            return found
    return None


def extract_table_elements(tables: list[TableRecord]) -> list[TableElement]:
    """One TableElement per DS-2 TableRecord, with the ragged-row check and
    table-level scale capture applied."""
    elements: list[TableElement] = []
    for table in tables:
        ragged = _is_ragged(table)
        scale = _table_scale_context(table)
        elements.append(
            TableElement(
                page=table.page,
                bbox=_union_cell_bboxes(table),
                cells=table.cells,
                cell_provenance_ok=table.cell_provenance_ok,
                ragged_table_rows=ragged,
                scale_multiplier=scale[0] if scale else None,
                scale_unit=scale[1] if scale else None,
                scale_context=scale[2] if scale else None,
                flags=["ragged_table_rows"] if ragged else [],
            )
        )
    return elements


def _nearby_text(page: PageIndex, bbox: BBox, padding: float = _CHART_CONTEXT_PADDING) -> str:
    lo_x, hi_x = bbox.x0 - padding, bbox.x1 + padding
    lo_y, hi_y = bbox.top - padding, bbox.bottom + padding
    indices = [
        i
        for i, c in enumerate(page.char_map)
        if c.page == bbox.page and lo_x <= c.x0 <= hi_x and lo_y <= c.top <= hi_y
    ]
    return "".join(page.text[i] for i in indices)


def extract_chart_elements(
    doc: "DoclingDocument", page_indices: list[PageIndex] | None = None
) -> list[ChartElement]:
    """One ChartElement per picture/chart Docling found. Consumes the
    in-memory DoclingDocument (doc.pictures), same convention as DS-2's
    extract_tables(doc.tables). When page_indices is supplied, nearby page
    text is captured alongside Docling's own caption; without them, only the
    caption is available."""
    by_page = {pi.page: pi for pi in (page_indices or [])}
    elements: list[ChartElement] = []
    for picture in doc.pictures:
        if not picture.prov:
            continue
        prov = picture.prov[0]
        bbox = BBox(
            x0=prov.bbox.l,
            top=prov.bbox.t,
            x1=prov.bbox.r,
            bottom=prov.bbox.b,
            page=prov.page_no,
        )
        page_index = by_page.get(prov.page_no)
        surrounding = _nearby_text(page_index, bbox) if page_index else ""
        elements.append(
            ChartElement(
                page=prov.page_no,
                bbox=bbox,
                caption_text=picture.caption_text(doc),
                surrounding_text=surrounding,
                flags=["chart_data_not_extracted"],
            )
        )
    return elements

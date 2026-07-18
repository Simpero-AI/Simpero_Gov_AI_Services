import logging
import re
from collections import Counter
from typing import Literal

from docling_core.types.doc.document import (
    DoclingDocument,
    TableItem,  # pyright: ignore[reportPrivateImportUsage]
)

from .normalize import normalize_numeric_text
from .resolver import resolve
from .schemas import PageIndex, TableCellRecord, TableRecord

logger = logging.getLogger(__name__)


def _bbox_is_valid(bbox) -> bool:
    """A cell coordinate is citable only if present and geometrically real —
    positive width and non-zero height. Origin-agnostic (x always increases
    left to right; height is checked as non-zero, not signed). Catches dropped
    or inverted rects that a bare None check would pass as valid provenance."""
    return bbox is not None and bbox.r > bbox.l and bbox.b != bbox.t


def _looks_numeric(text: str) -> bool:
    """True if a cell value reads as a figure rather than a label.

    Deliberately crude — any ASCII letter disqualifies it. That is what
    separates a real header ("2018F 1", "Historical Cost", "Trailing 5 Year
    Average") from a data value ("14.35", "83.04%", "$2,846,381", "-13.19").
    """
    stripped = text.strip()
    if not stripped:
        return False
    return not any(ch.isalpha() for ch in stripped) and any(ch.isdigit() for ch in stripped)


# A negative lookahead for a further digit, NOT \b: a word boundary fails on
# "2006E" because "E" is a word character, and an estimate column is exactly the
# spelling a CIM uses. This still refuses to match inside a longer number.
_YEAR_RE = re.compile(r"^(?:19|20)\d{2}(?!\d)")


def _looks_like_year(text: str) -> bool:
    """A period column label: "2003", "2006E", "2008E (1)", "FY2005".

    Year headers are the one kind of column label that reads as a figure, so the
    all-text test alone would reject them and leave a financial statement with no
    header at all.
    """
    stripped = text.strip().lstrip("FY").lstrip()
    return bool(_YEAR_RE.match(stripped))


def _value_columns(cells: list[TableCellRecord]) -> set[int]:
    """The columns that actually carry figures, judged across the whole table.

    A column counts when at least two rows put a figure in it, so a stray number
    in a title or a footnote does not create a phantom data column.
    """
    counts: Counter[int] = Counter()
    for row in {c.row for c in cells}:
        figures = [c for c in cells if c.row == row and c.col >= 1 and _looks_numeric(c.text)]
        if len(figures) >= 2:
            for cell in figures:
                counts[cell.col] += 1
    return {col for col, n in counts.items() if n >= 2}


def _infer_header_row(cells: list[TableCellRecord]) -> int | None:
    """Locate the header row from table structure, ignoring Docling's flag.

    Docling's `column_header` is not trustworthy on tables with unlabeled
    columns: on Pitchbook PDF-page 17 it flags the first *data* row's numeric
    cells rather than any real header (there isn't one — the table has section
    labels above bare ratio columns). Relying on the flag there would invent
    column names out of data values.

    Structural rule: the header row is the TOPMOST row that spans the table's
    value columns. A column label has to sit above the column it labels, so this
    naturally skips the rows that do not span them — a title banner
    ("Consolidated Balance Sheet"), a scale banner ("($ in millions)"), a period
    caption ("As of December 31,") — and lands on the row that does.

    Looking only at row 0, as this did originally, breaks on every financial
    statement in a CIM, because they stack three rows before the data:

        row 0  Consolidated Balance Sheet          ($ in millions)   <- title
        row 1  As of December 31,                                    <- caption
        row 2           2003     2004     2005     2006E             <- the header
        row 5  Cash and cash equivalents  $77.3  $75.2 ...           <- data

    Row 0 was returned as the header there, so every value in the table was
    named after the title instead of its year, and all four years of a line item
    collapsed onto one indistinguishable attribute.

    The first row spanning the value columns is the header rather than data
    because a header precedes its data by definition. It only counts as one if
    its labels are either all text ("Slots", "Hotel Rooms") or all periods
    ("2003", "2006E") -- a spanning row of mixed figures is the first DATA row,
    which is the Pitchbook case, and there the table genuinely has no header.
    """
    value_cols = _value_columns(cells)
    if not value_cols:
        # No column carries figures more than once, so there is nothing to align
        # a header against. Fall back to the original row-0 rule.
        labels = [c for c in cells if c.row == 0 and c.col >= 1 and c.text.strip()]
        if not labels or any(_looks_numeric(c.text) for c in labels):
            return None
        return 0

    for row in sorted({c.row for c in cells}):
        occupied = {c.col for c in cells if c.row == row and c.text.strip()}
        if not value_cols.issubset(occupied):
            continue
        labels = [c for c in cells if c.row == row and c.col in value_cols and c.text.strip()]
        if all(not _looks_numeric(c.text) for c in labels):
            return row
        if all(_looks_like_year(c.text) for c in labels):
            return row
        # The first row spanning the value columns is already data: no header.
        return None
    return None


def _reconstruct_bbox(
    text_normalized: str, page_index: PageIndex
) -> tuple[float, float, float, float] | None:
    """Recover a cell's coordinates from the page's positioned index when Docling
    exposed no valid cell bbox: resolve the cell text against the page via the
    DS-W3-3 exact-span resolver (fail closed on absent/ambiguous — a reconstructed
    coordinate is never a guess). The page index is normalized, so match on
    text_normalized."""
    span = resolve(text_normalized, page_index)
    if span is None:
        return None
    bbox = span.bbox
    return (bbox.x0, bbox.top, bbox.x1, bbox.bottom)


def _build_table_record(table: TableItem, page_index: PageIndex | None) -> TableRecord:
    page_no = table.prov[0].page_no

    cells: list[TableCellRecord] = []
    provenance_ok = True
    reconstructed = 0
    for cell in table.data.table_cells:
        # Split numeric tokens survive in Docling's raw cell text; the flat page
        # index normalizes them but the table structure does not. Normalize here,
        # with the same rule, so downstream claim extraction never has to re-derive
        # it (DS-W3-2 finding F2), and so the reconstruction match uses the same
        # form as the page index.
        text_normalized = normalize_numeric_text(cell.text)
        bbox = cell.bbox
        source: Literal["docling_native", "reconstructed"] | None
        if bbox is not None and _bbox_is_valid(bbox):
            x0, top, x1, bottom = bbox.l, bbox.t, bbox.r, bbox.b
            source = "docling_native"
        else:
            rebuilt = _reconstruct_bbox(text_normalized, page_index) if page_index else None
            if rebuilt is not None:
                x0, top, x1, bottom = rebuilt
                source = "reconstructed"
                reconstructed += 1
            else:
                x0 = top = x1 = bottom = None
                source = None
                provenance_ok = False
        cells.append(
            TableCellRecord(
                row=cell.start_row_offset_idx,
                col=cell.start_col_offset_idx,
                row_span=cell.row_span,
                col_span=cell.col_span,
                text=cell.text,
                text_normalized=text_normalized,
                column_header=cell.column_header,
                row_header=cell.row_header,
                page=page_no,
                x0=x0,
                top=top,
                x1=x1,
                bottom=bottom,
                bbox_source=source,
            )
        )

    header_row = _infer_header_row(cells)
    # Docling's flags are only trustworthy when they mark exactly the row that
    # structure says is the header. Anything else (no header, or flags on a data
    # row) means consumers must ignore column_header for this table.
    flagged_rows = {c.row for c in cells if c.column_header}
    column_headers_reliable = header_row is not None and flagged_rows == {header_row}

    # Provenance is audit-critical: surface where a table's geometry was recovered
    # from the flat index rather than read from Docling, and where it stayed uncitable.
    if reconstructed:
        logger.info(
            "table page %s: recovered %d of %d cell bbox(es) via reconstruction fallback",
            page_no,
            reconstructed,
            len(cells),
        )
    if not provenance_ok:
        unresolved = sum(1 for c in cells if c.bbox_source is None)
        logger.warning(
            "table page %s: not citable — %d of %d cell(s) have no resolvable coordinate",
            page_no,
            unresolved,
            len(cells),
        )

    return TableRecord(
        page=page_no,
        num_rows=table.data.num_rows,
        num_cols=table.data.num_cols,
        cells=cells,
        cell_provenance_ok=provenance_ok,
        reconstructed_cells=reconstructed,
        header_row=header_row,
        column_headers_reliable=column_headers_reliable,
    )


def extract_tables(
    doc: DoclingDocument, page_indices: list[PageIndex] | None = None
) -> list[TableRecord]:
    """Structured, per-cell table records from a parsed DoclingDocument.

    Consumes the in-memory document (DoclingParseResult.document) — no cache
    round-trip — so table extraction runs in the same request as the parse and
    does not depend on the (optional) Spaces document cache.

    When page_indices (DoclingParseResult.pages) are supplied, a cell that Docling
    gives no valid bbox is recovered from that page's positioned index by
    exact-span match, so a table stays citable even where Docling drops cell
    geometry. Without them, only Docling-native cell bboxes are used.
    """
    by_page = {pi.page: pi for pi in (page_indices or [])}
    return [_build_table_record(t, by_page.get(t.prov[0].page_no)) for t in doc.tables]


def tables_on_page(tables: list[TableRecord], page_no: int) -> list[TableRecord]:
    return [t for t in tables if t.page == page_no]

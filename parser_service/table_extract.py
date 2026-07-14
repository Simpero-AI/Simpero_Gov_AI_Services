from docling_core.types.doc.document import (
    DoclingDocument,
    TableItem,  # pyright: ignore[reportPrivateImportUsage]
)

from .config import get_settings
from .normalize import normalize_numeric_text
from .schemas import TableCellRecord, TableRecord


class TableExtractError(Exception):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(message)


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


def _infer_header_row(cells: list[TableCellRecord]) -> int | None:
    """Locate the header row from table structure, ignoring Docling's flag.

    Docling's `column_header` is not trustworthy on tables with unlabeled
    columns: on Pitchbook PDF-page 17 it flags the first *data* row's numeric
    cells rather than any real header (there isn't one — the table has section
    labels above bare ratio columns). Relying on the flag there would invent
    column names out of data values.

    Structural rule: row 0 is the header row only if it actually carries
    non-empty value-column labels (col >= 1) and none of them read as figures.
    Otherwise this table has no header row and we say so (None) instead of
    guessing.
    """
    labels = [c for c in cells if c.row == 0 and c.col >= 1 and c.text.strip()]
    if not labels:
        return None
    if any(_looks_numeric(c.text) for c in labels):
        return None
    return 0


def _build_table_record(table: TableItem) -> TableRecord:
    page_no = table.prov[0].page_no

    cells: list[TableCellRecord] = []
    provenance_ok = True
    for cell in table.data.table_cells:
        bbox = cell.bbox
        if bbox is None:
            provenance_ok = False
        cells.append(
            TableCellRecord(
                row=cell.start_row_offset_idx,
                col=cell.start_col_offset_idx,
                row_span=cell.row_span,
                col_span=cell.col_span,
                text=cell.text,
                # Split numeric tokens survive in Docling's raw cell text; the
                # flat page index normalizes them but the table structure does
                # not. Normalize here, with the same rule, so downstream fact
                # extraction never has to re-derive it (DS-W3-2 finding F2).
                text_normalized=normalize_numeric_text(cell.text),
                column_header=cell.column_header,
                row_header=cell.row_header,
                page=page_no,
                x0=bbox.l if bbox else None,
                top=bbox.t if bbox else None,
                x1=bbox.r if bbox else None,
                bottom=bbox.b if bbox else None,
            )
        )

    header_row = _infer_header_row(cells)
    # Docling's flags are only trustworthy when they mark exactly the row that
    # structure says is the header. Anything else (no header, or flags on a data
    # row) means consumers must ignore column_header for this table.
    flagged_rows = {c.row for c in cells if c.column_header}
    column_headers_reliable = header_row is not None and flagged_rows == {header_row}

    return TableRecord(
        page=page_no,
        num_rows=table.data.num_rows,
        num_cols=table.data.num_cols,
        cells=cells,
        cell_provenance_ok=provenance_ok,
        header_row=header_row,
        column_headers_reliable=column_headers_reliable,
    )


def extract_tables_for_digest(digest: str) -> list[TableRecord]:
    """Extract structured, per-cell table records for a previously parsed PDF.

    Reads the chunked DoclingDocument JSON cache that
    docling_parser.parse_pdf_bytes writes to settings.cache_dir (one file per
    page_range chunk) rather than re-running Docling — DS-W3-1 persists the
    raw document specifically so DS-W3-2/DS-W3-6 can consume table/element
    structure without a second parse.
    """
    settings = get_settings()
    if not settings.enable_cache:
        raise TableExtractError(
            "cache_disabled",
            "PARSER_ENABLE_CACHE is False in this deployment, so no raw "
            "DoclingDocument was persisted for DS-W3-2/DS-W3-6 to read. Table "
            "extraction requires the cache — see config.py's SECURITY NOTE.",
        )

    chunk_files = sorted(settings.cache_dir.glob(f"{digest}_p*.json"))
    if not chunk_files:
        raise TableExtractError(
            "no_cached_document",
            f"No cached DoclingDocument chunks found for digest {digest}. "
            "Call parse_pdf_bytes first.",
        )

    tables: list[TableRecord] = []
    for chunk_file in chunk_files:
        doc = DoclingDocument.load_from_json(chunk_file)
        tables.extend(_build_table_record(table) for table in doc.tables)

    return tables


def tables_on_page(tables: list[TableRecord], page_no: int) -> list[TableRecord]:
    return [t for t in tables if t.page == page_no]

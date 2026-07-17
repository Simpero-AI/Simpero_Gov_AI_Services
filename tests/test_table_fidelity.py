"""DS-2 (DS-W3-2) table fidelity gate: verifies Docling reads the fixed
corpus's tables correctly AND exposes a per-cell source bbox for every cell.

Like DS-W3-1's real-corpus tests, this depends on confidential CIM documents
that live only in a sibling repo checkout on some local machines and are
never committed here — no CI-portable equivalent exists because the gate's
ground truth (hand-checked row/column values) is specific to these real
documents. Marked local_corpus and always explicitly skipped, never silently,
when the fixture isn't present.
"""

import os
from pathlib import Path

import pytest

from parser_service.docling_parser import parse_pdf_bytes
from parser_service.schemas import TableRecord
from parser_service.table_extract import extract_tables, tables_on_page

pytestmark = pytest.mark.local_corpus


def _local_corpus_pdf(relative_path: str) -> Path | None:
    root = os.environ.get("PARSER_LOCAL_CORPUS_DIR")
    if not root:
        return None
    path = Path(root) / relative_path
    return path if path.exists() else None


def _row_by_label(table: TableRecord, label: str, *, normalized: bool = False) -> dict[int, str]:
    target_row = None
    for cell in table.cells:
        if cell.col == 0 and cell.text.strip() == label:
            target_row = cell.row
            break
    assert target_row is not None, f"row label '{label}' not found in table"
    return {
        cell.col: (cell.text_normalized if normalized else cell.text)
        for cell in table.cells
        if cell.row == target_row
    }


@pytest.fixture(scope="module")
def ptl_tables() -> list[TableRecord]:
    pdf_path = _local_corpus_pdf("1st-app-h-ptl/1st-App-H-PTL-Group-CIM.pdf")
    if not pdf_path:
        pytest.skip("Real PTL CIM not available (set PARSER_LOCAL_CORPUS_DIR).")
    result = parse_pdf_bytes(pdf_path.read_bytes())
    assert result.document is not None
    return extract_tables(result.document, result.pages)


@pytest.fixture(scope="module")
def pitchbook_tables() -> list[TableRecord]:
    pdf_path = _local_corpus_pdf("sell-side-pitchbook/Sell-Side-Pitchbook-CIM-Sample.pdf")
    if not pdf_path:
        pytest.skip("Real Pitchbook CIM not available (set PARSER_LOCAL_CORPUS_DIR).")
    result = parse_pdf_bytes(pdf_path.read_bytes())
    assert result.document is not None
    return extract_tables(result.document, result.pages)


def test_ptl_page_11_income_statement_structure_and_values(
    ptl_tables: list[TableRecord],
) -> None:
    tables = tables_on_page(ptl_tables, 11)
    assert len(tables) == 1
    table = tables[0]

    header = _row_by_label(table, "Income Statement CAD (in Thousands)")
    assert header[1] == "Trailing 5 Year Average"

    revenue = _row_by_label(table, "Revenue")
    assert revenue[1] == "$15,295"

    gross_margin_pct = _row_by_label(table, "Gross Margin %")
    assert gross_margin_pct[1] == "27.3%"

    # Docling's RAW cell text keeps the split token ("3 ,817") — preserved
    # verbatim for provenance. QA finding F2 added `text_normalized`, which
    # applies the same rule as the flat page index, so downstream fact
    # extraction reads a clean "3,817" without re-deriving the normalization.
    gross_margin = _row_by_label(table, "Gross Margin")
    assert gross_margin[3] == "3 ,817"

    gross_margin_norm = _row_by_label(table, "Gross Margin", normalized=True)
    assert gross_margin_norm[3] == "3,817"

    # Every split token on this page normalizes, currency prefixes included.
    normalized_by_raw = {
        c.text: c.text_normalized for c in table.cells if c.text != c.text_normalized
    }
    assert normalized_by_raw == {
        "3 ,817": "3,817",
        "4 ,007": "4,007",
        "1 ,723": "1,723",
        "1 ,774": "1,774",
        "$2 ,094": "$2,094",
        "$2 ,233": "$2,233",
    }


def test_ptl_page_11_header_row_inferred_from_structure(ptl_tables: list[TableRecord]) -> None:
    # QA finding F3: the header row is derived from structure, not from
    # Docling's advisory column_header flag. On this well-formed table the two
    # agree, so the flags are declared reliable.
    table = tables_on_page(ptl_tables, 11)[0]

    assert table.header_row == 0
    assert table.column_headers_reliable is True
    header_labels = [c.text for c in sorted(table.cells, key=lambda c: c.col) if c.row == 0]
    assert header_labels[0] == "Income Statement CAD (in Thousands)"
    assert "Trailing 5 Year Average" in header_labels


def test_ptl_page_11_every_cell_has_resolvable_bbox(ptl_tables: list[TableRecord]) -> None:
    tables = tables_on_page(ptl_tables, 11)
    assert len(tables) == 1
    table = tables[0]

    assert table.cell_provenance_ok is True
    assert len(table.cells) == table.num_rows * table.num_cols
    for cell in table.cells:
        assert cell.x0 is not None
        assert cell.top is not None
        assert cell.x1 is not None
        assert cell.bottom is not None
        assert cell.x1 > cell.x0
        assert cell.bottom > cell.top


def test_pitchbook_page_20_clean_baseline_table(pitchbook_tables: list[TableRecord]) -> None:
    tables = tables_on_page(pitchbook_tables, 20)
    assert len(tables) == 1
    table = tables[0]

    assert table.cell_provenance_ok is True

    # Explicit-header baseline: structure and Docling's flags agree here.
    assert table.header_row == 0
    assert table.column_headers_reliable is True

    header = _row_by_label(table, "Assets Available for Acquisition")
    assert header[1] == "Historical Cost"
    assert header[2] == "Adjusted Value"

    receivables = _row_by_label(table, "Accounts Receivable")
    assert receivables[1] == "$2,846,381"
    assert receivables[2] == "$2,846,381"

    # 6 data rows x 2 value columns = 12 currency values, per the ticket.
    currency_cells = [cell for cell in table.cells if cell.row > 0 and cell.col > 0]
    assert len(currency_cells) == 12


def test_pitchbook_page_17_unlabeled_columns_structure(
    pitchbook_tables: list[TableRecord],
) -> None:
    tables = tables_on_page(pitchbook_tables, 17)
    assert len(tables) == 1
    table = tables[0]

    assert table.cell_provenance_ok is True
    assert table.num_cols == 6
    assert table.num_rows == 39

    # This table has no real header row in the source PDF — just section labels
    # ("Turnover Ratios:", "Debt Ratios:", ...) above unlabeled ratio columns.
    # Docling MISLABELS the first *data* row's numeric cells as column headers.
    # Pin that quirk, so it is visible if Docling ever changes:
    first_data_row_values = [cell for cell in table.cells if cell.row == 1 and cell.col > 0]
    assert first_data_row_values, "expected value cells on the first data row"
    assert all(cell.column_header for cell in first_data_row_values), (
        "known Docling quirk on unlabeled-column tables: the first data "
        "row's numeric cells come back column_header=True"
    )

    # QA finding F3: structure must override that bogus flag. The extractor
    # reports NO header row and declares the flags unreliable, so downstream
    # cannot mistake data values ("14.35", "83.04%") for column names.
    assert table.header_row is None, "unlabeled-column table has no header row"
    assert table.column_headers_reliable is False, (
        "Docling flagged a data row as header — must be reported as unreliable"
    )

    gross_margin_pct = _row_by_label(table, "Gross Margin %")
    assert gross_margin_pct[1] == "26.17%"


def test_gate_tables_have_full_cell_provenance(
    ptl_tables: list[TableRecord], pitchbook_tables: list[TableRecord]
) -> None:
    """DS-2 gate decision: Docling-native cells path. Every target table's
    cells expose a resolvable source bbox, so the coordinate-reconstruction
    fallback (mapping cells back to DS-1's char_map by bbox overlap) is not
    needed for this corpus."""
    ptl_p11 = tables_on_page(ptl_tables, 11)
    pb_p17 = tables_on_page(pitchbook_tables, 17)
    pb_p20 = tables_on_page(pitchbook_tables, 20)

    assert len(ptl_p11) == 1
    assert len(pb_p17) == 1
    assert len(pb_p20) == 1

    for table in (*ptl_p11, *pb_p17, *pb_p20):
        assert table.cell_provenance_ok is True
        assert all(cell.x0 is not None for cell in table.cells)

"""CI-portable table fidelity gate.

The real-corpus gate (test_table_fidelity.py) needs confidential CIMs and skips
in CI. This exercises the same path — parse -> extract_tables on the in-memory
DoclingDocument -> per-cell structure, values, normalization, and coordinates —
against a synthetic bordered table generated at test time, so the DS-W3-2 gate
(including per-cell coordinate validation) actually runs on every commit.
"""

import io

import pytest
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle

from services.parser.parser_service.docling_parser import parse_pdf_bytes
from services.parser.parser_service.schemas import TableRecord
from services.parser.parser_service.table_extract import extract_tables


def _income_statement_pdf() -> bytes:
    """A bordered income statement Docling's table model detects, reproducing the
    corpus's hard cases (split-token '3 ,817', a scale-style header, currency and
    percentage values) with no confidential content."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    data = [
        ["Income Statement", "FY2021", "FY2022"],
        ["Revenue", "3 ,817", "4,007"],
        ["EBITDA", "1,204", "1,690"],
        ["EBITDA margin", "31.5%", "37.5%"],
    ]
    table = Table(data)
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ]
        )
    )
    doc.build([table])
    return buffer.getvalue()


def _cell(table: TableRecord, row: int, col: int):
    return next(c for c in table.cells if c.row == row and c.col == col)


def _coord(value: float | None) -> float:
    assert value is not None, "cell coordinate must be present"
    return value


@pytest.fixture(scope="module")
def synthetic_table() -> TableRecord:
    result = parse_pdf_bytes(_income_statement_pdf())
    assert result.document is not None, "parse must expose the in-memory DoclingDocument"
    tables = extract_tables(result.document)
    assert len(tables) == 1, "Docling should detect exactly one table"
    return tables[0]


def test_structure_and_values(synthetic_table: TableRecord) -> None:
    t = synthetic_table
    assert (t.num_rows, t.num_cols) == (4, 3)
    assert len(t.cells) == 12
    assert _cell(t, 0, 1).text == "FY2021"
    assert _cell(t, 1, 0).text == "Revenue"
    assert _cell(t, 1, 2).text == "4,007"
    assert _cell(t, 3, 1).text == "31.5%"


def test_split_token_normalized_in_cell(synthetic_table: TableRecord) -> None:
    revenue_fy21 = _cell(synthetic_table, 1, 1)
    assert revenue_fy21.text == "3 ,817"  # raw preserved for provenance
    assert revenue_fy21.text_normalized == "3,817"  # F2 normalization in the cell


def test_every_cell_has_valid_provenance(synthetic_table: TableRecord) -> None:
    t = synthetic_table
    assert t.cell_provenance_ok is True
    for c in t.cells:
        assert _coord(c.x1) > _coord(c.x0)
        assert _coord(c.bottom) != _coord(c.top)


def test_cell_coordinates_reflect_grid_layout(synthetic_table: TableRecord) -> None:
    # Per-cell coordinates place columns left-to-right and rows top-to-bottom —
    # the "per-cell coordinate validation" the gate is named for, run in CI.
    t = synthetic_table
    x_by_col = [_coord(_cell(t, 0, col).x0) for col in range(t.num_cols)]
    assert x_by_col == sorted(x_by_col) and len(set(x_by_col)) == t.num_cols
    y_by_row = [_coord(_cell(t, row, 0).top) for row in range(t.num_rows)]
    assert y_by_row == sorted(y_by_row) and len(set(y_by_row)) == t.num_rows

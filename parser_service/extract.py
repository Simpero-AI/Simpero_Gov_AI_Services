"""Claim candidates from a parsed table -- deliberately a heuristic, not a model.

emit.py answers "cite this value exactly". Something has to answer the question
before it: WHICH values, and what is each one about? emit_pdf_table_cell_claim
takes entity, attribute and value_type as inputs; this module is the smallest
honest thing that supplies them for a financial table.

WHAT THIS IS NOT
================
This is not the extractor. It reads one well-formed financial table and keys
each cell off its row label and column header. That is enough to put real,
fully-cited claims in the spine from a real CIM -- it is NOT enough to read a
CIM. It cannot:

  - identify the entity (the caller passes it in; this module has no idea whose
    revenue it is reading)
  - name an attribute in the product's vocabulary ("Revenue | 2019F" is the
    table's own words, not `revenueLatestUsd`)
  - read prose, or any table whose meaning is not carried by its row labels
  - know a period from a fiscal-year convention (period_year/period_kind are
    left for a later pass; the year, where present, is in the column header
    the attribute string quotes)

The real extractor is an open design question (deterministic vs LLM). Treat this
as the floor: the thing that proves the pipeline end-to-end and gives that
design something concrete to beat. It is small on purpose, so that replacing it
costs nothing.

WHY IT STILL FAIL-CLOSES
========================
Being a heuristic is not licence to be sloppy about provenance. Every claim it
proposes goes through emit.py, so the citation rules hold unchanged: an
unresolvable value is `missing` with no span, scale is captured with its source,
and nothing is fabricated. A bad GUESS here costs a wrong `attribute` string --
never a wrong citation.
"""

from __future__ import annotations

from .emit import Claim, FlagLog, emit_pdf_table_cell_claim
from .scale import ValueType
from .schemas import PageIndex, TableCellRecord, TableRecord

_LABEL_COL = 0


def _has_digit(text: str) -> bool:
    return any(ch.isdigit() for ch in text)


def _cell_at(table: TableRecord, row: int, col: int) -> TableCellRecord | None:
    return next((c for c in table.cells if c.row == row and c.col == col), None)


def infer_value_type(raw: str) -> ValueType:
    """Read a value_type off the cell's own text.

    A trailing "%" is a percent; everything else in a financial table is read as
    currency, including bare tokens with no "$". That is correct for a CIM
    income statement and is load-bearing -- PTL PDF-page 11 prints

        CAD (in Thousands)
        Revenue       $15,295  $17,146 ...
        Gross Margin    4,171    3,631 ...

    where only the Revenue row carries "$" and Gross Margin is money too. Typing
    the bare tokens as `count` would make DS-W3-4 refuse the header scale and
    report an income-statement figure at a thousandth of its value.

    The cost is that a genuine headcount in a scaled table reads 1000x high. On a
    financial page that trade is right, and it is why this module is scoped to
    financial tables. It is also why the real extractor must not inherit this
    function: it should know the attribute's type, not guess it from a string.
    """
    return "percent" if "%" in raw else "currency"


def attribute_for(table: TableRecord, cell: TableCellRecord) -> str | None:
    """The table's own name for this cell: "<row label> | <column header>".

    Deliberately the source document's words rather than a product attribute
    (`revenueLatestUsd`). Mapping a CIM's phrasing onto a fixed vocabulary is the
    extractor's real job and is not guessed here -- inventing a canonical name
    from a string match would be a silent, confident error of exactly the kind
    this codebase refuses elsewhere.

    Returns None when the row has no label, because a value whose own table
    cannot say what it is has no attribute to claim.
    """
    label_cell = _cell_at(table, cell.row, _LABEL_COL)
    if label_cell is None or not label_cell.text_normalized.strip():
        return None
    label = label_cell.text_normalized.strip()

    # header_row is None for a table with unlabeled columns. The row label alone
    # is then the whole attribute -- weaker, but honest. DS-W3-2 also marks
    # column_headers_reliable=False for headers it does not trust; a header we
    # were told not to trust must not be baked into an attribute name.
    if table.header_row is None or not table.column_headers_reliable:
        return label

    header_cell = _cell_at(table, table.header_row, cell.col)
    if header_cell is None or not header_cell.text_normalized.strip():
        return label
    return f"{label} | {header_cell.text_normalized.strip()}"


def claims_from_table(
    table: TableRecord,
    page: PageIndex,
    *,
    entity: str,
    file: str,
    flag_log: FlagLog,
    section: str | None = None,
) -> list[Claim]:
    """Propose one claim per numeric data cell in `table`.

    Skips the header row and the label column (they name things, they are not
    values), and any cell with no digit in it -- a prose cell in a financial
    table is context, not a claim. `entity` is the caller's to supply: a table
    does not know whose numbers it holds.

    Every surviving cell goes through emit_pdf_table_cell_claim, so a cell whose
    value cannot be cited comes back `missing` rather than being dropped
    silently. That matters: a dropped cell is invisible, a `missing` claim is a
    recall gap you can see.
    """
    claims: list[Claim] = []
    for cell in sorted(table.cells, key=lambda c: (c.row, c.col)):
        if cell.col == _LABEL_COL or cell.row == table.header_row:
            continue
        raw = cell.text_normalized.strip()
        if not raw or not _has_digit(raw):
            continue

        attribute = attribute_for(table, cell)
        if attribute is None:
            continue

        claims.append(
            emit_pdf_table_cell_claim(
                entity,
                attribute,
                table,
                cell,
                page,
                value_type=infer_value_type(raw),
                file=file,
                flag_log=flag_log,
                section=section,
            )
        )
    return claims

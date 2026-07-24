"""AE-A-RETR-2 chunk-metadata acceptance criteria: document_id, scale_context,
and element_type stamped at chunk-creation time, on hand-built parser output."""

from __future__ import annotations

from parser_service.chunker import ChunkRecord, chunk_document, chunk_table
from parser_service.elements import ChartElement, TableElement
from parser_service.schemas import BBox, CharBox, PageIndex, TableCellRecord, TextBlockRecord


def _char(ch: str, page: int = 1) -> CharBox:
    return CharBox(char=ch, x0=0, top=0, x1=1, bottom=1, page=page, is_boilerplate=False, precision="word")


def _page(text: str, page: int = 1) -> PageIndex:
    return PageIndex(page=page, text=text, char_map=[_char(c, page) for c in text])


def _block(order: int, label: str, text: str, page: int = 1) -> TextBlockRecord:
    return TextBlockRecord(
        page=page, order=order, label=label, text=text, text_normalized=text,
        x0=0, top=float(order), x1=10, bottom=float(order) + 1, bbox_source="docling_native",
    )


def _cell(row: int, col: int, text: str, *, header: bool = False, page: int = 1) -> TableCellRecord:
    return TableCellRecord(
        row=row, col=col, row_span=1, col_span=1, text=text, text_normalized=text,
        column_header=header, row_header=False, page=page,
        x0=float(col), top=float(row), x1=float(col) + 1, bottom=float(row) + 1,
        bbox_source="docling_native",
    )


def _table(cells: list[TableCellRecord], *, scale_context: str | None = None, mult: float | None = None) -> TableElement:
    return TableElement(
        page=1, bbox=None, cells=cells, cell_provenance_ok=True, ragged_table_rows=False,
        scale_multiplier=mult, scale_unit=None, scale_context=scale_context, flags=[],
    )


def test_document_id_stamped_on_every_chunk() -> None:
    page = _page("MARKET\n\nThe market is large and fragmented.")
    blocks = [_block(0, "section_header", "MARKET"), _block(1, "text", "The market is large and fragmented.")]
    chunks = chunk_document([page], blocks, [], [], document_id="sha-xyz", source_file="d.pdf")
    assert chunks
    assert all(c.document_id == "sha-xyz" for c in chunks)


def test_element_type_labels_prose_table_and_chart() -> None:
    page = _page("Some prose about the firm here.")
    blocks = [_block(0, "text", "Some prose about the firm here.")]
    table = _table([
        _cell(0, 0, "Metric", header=True), _cell(0, 1, "2005", header=True),
        _cell(1, 0, "Revenue"), _cell(1, 1, "15,295"),
    ])
    chart = ChartElement(
        page=1, bbox=BBox(x0=0, top=0, x1=10, bottom=10, page=1),
        caption_text="Rev chart", surrounding_text="", flags=["chart_data_not_extracted"],
    )
    chunks = chunk_document([page], blocks, [table], [chart], document_id="s", source_file="d.pdf")
    assert {c.element_type for c in chunks} == {"prose", "table", "chart"}
    assert [c.element_type for c in chunks if "Revenue" in c.content] == ["table"]


def test_scale_context_carried_onto_table_chunk() -> None:
    cells = [
        _cell(0, 0, "($ in thousands)", header=True), _cell(0, 1, "2005", header=True),
        _cell(1, 0, "Revenue"), _cell(1, 1, "15,295"),
    ]
    page = _page("dummy")
    chunks = chunk_table(
        _table(cells, scale_context="($ in thousands)", mult=1000.0),
        page, 0, document_id="s", source_file="d.pdf",
    )
    assert chunks[0].scale_context == "($ in thousands)"
    assert chunks[0].scale_multiplier == 1000.0
    assert chunks[0].element_type == "table"


def test_scale_context_survives_when_header_is_in_a_different_chunk() -> None:
    # The silent-1000x guard: the scale banner sits on the page but NOT inside the
    # table's own serialized content, yet the table chunk still carries it.
    page = _page("Financial summary ($ in millions)\n\nother text")
    cells = [_cell(0, 0, "Revenue", header=True), _cell(1, 0, "28.1")]
    chunks = chunk_table(_table(cells, scale_context=None), page, 0, document_id="s", source_file="d.pdf")
    assert chunks[0].scale_context == "($ in millions)"
    assert "($ in millions)" not in chunks[0].content


def test_scale_context_on_every_fragment_of_a_split_table() -> None:
    header = [_cell(0, 0, "Property", header=True), _cell(0, 1, "Rooms", header=True)]
    body = []
    long_name = "Very long property name that forces the table over the size cap " * 3
    for r in range(1, 30):
        body.append(_cell(r, 0, f"{long_name} {r}"))
        body.append(_cell(r, 1, str(1000 + r)))
    page = _page("dummy")
    frags = chunk_table(
        _table(header + body, scale_context="(rooms)"), page, 0, document_id="s", source_file="d.pdf"
    )
    assert len(frags) > 1
    assert all(f.scale_context == "(rooms)" for f in frags)


def test_chunk_record_names_document_and_element_type() -> None:
    c = ChunkRecord(
        content="x", element_type="prose", page=1, order=0, document_id="s", source_file="d.pdf"
    )
    assert c.document_id == "s" and c.element_type == "prose"

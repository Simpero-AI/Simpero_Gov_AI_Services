"""AE-A-RETR-1 boundary-rule acceptance criteria, exercised on hand-built parser
output so the rules are pinned without a Docling parse in the loop."""

from __future__ import annotations

import re

from parser_service.chunker import (
    OVERLAP_TOKENS,
    TARGET_TOKENS,
    chunk_document,
    chunk_table,
)
from parser_service.elements import ChartElement, TableElement
from parser_service.schemas import BBox, CharBox, PageIndex, TableCellRecord, TextBlockRecord


def _char(ch: str, page: int = 1, boiler: bool = False) -> CharBox:
    return CharBox(
        char=ch, x0=0, top=0, x1=1, bottom=1, page=page, is_boilerplate=boiler, precision="word"
    )


def _page(text: str, *, boilerplate_substr: str | None = None, page: int = 1) -> PageIndex:
    """A page whose char_map is 1:1 with text; chars inside `boilerplate_substr`
    are flagged is_boilerplate."""
    boiler_range: range | None = None
    if boilerplate_substr and boilerplate_substr in text:
        start = text.index(boilerplate_substr)
        boiler_range = range(start, start + len(boilerplate_substr))
    char_map = [
        _char(c, page=page, boiler=(boiler_range is not None and i in boiler_range))
        for i, c in enumerate(text)
    ]
    return PageIndex(page=page, text=text, char_map=char_map)


def _block(order: int, label: str, text: str, page: int = 1) -> TextBlockRecord:
    return TextBlockRecord(
        page=page,
        order=order,
        label=label,
        text=text,
        text_normalized=text,
        x0=0,
        top=float(order),
        x1=10,
        bottom=float(order) + 1,
        bbox_source="docling_native",
    )


def _cell(row: int, col: int, text: str, *, header: bool = False, page: int = 1) -> TableCellRecord:
    return TableCellRecord(
        row=row,
        col=col,
        row_span=1,
        col_span=1,
        text=text,
        text_normalized=text,
        column_header=header,
        row_header=False,
        page=page,
        x0=float(col),
        top=float(row),
        x1=float(col) + 1,
        bottom=float(row) + 1,
        bbox_source="docling_native",
    )


def _table(cells: list[TableCellRecord]) -> TableElement:
    return TableElement(
        page=1,
        bbox=None,
        cells=cells,
        cell_provenance_ok=True,
        ragged_table_rows=False,
        scale_multiplier=None,
        scale_unit=None,
        scale_context=None,
        flags=[],
    )


def _collapse(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def test_section_heading_carried_onto_chunks() -> None:
    text = "MARKET\n\nThe market is large and fragmented."
    page = _page(text)
    blocks = [
        _block(0, "section_header", "MARKET"),
        _block(1, "text", "The market is large and fragmented."),
    ]
    chunks = chunk_document([page], blocks, [], [], document_id="doc-sha", source_file="d.pdf")
    assert chunks and chunks[0].section == "MARKET"


def test_boilerplate_excluded_by_label_and_by_flag() -> None:
    text = "Real sentence one.\n\nCONFIDENTIAL DRAFT\n\nReal sentence two."
    page = _page(text, boilerplate_substr="CONFIDENTIAL DRAFT")
    blocks = [
        _block(0, "text", "Real sentence one."),
        _block(1, "page_footer", "CONFIDENTIAL DRAFT"),  # excluded by label
        _block(2, "text", "Real sentence two."),
    ]
    chunks = chunk_document([page], blocks, [], [], document_id="doc-sha", source_file="d.pdf")
    assert not any("CONFIDENTIAL" in c.content for c in chunks)


def test_boilerplate_excluded_when_label_is_prose_but_chars_flagged() -> None:
    # Label says "text" (Docling mislabels furniture), but the characters are
    # flagged is_boilerplate -- the second, independent signal must still exclude it.
    text = "Genuine content here.\n\nPage 5 of 40 confidential footer."
    page = _page(text, boilerplate_substr="Page 5 of 40 confidential footer.")
    blocks = [
        _block(0, "text", "Genuine content here."),
        _block(1, "text", "Page 5 of 40 confidential footer."),
    ]
    chunks = chunk_document([page], blocks, [], [], document_id="doc-sha", source_file="d.pdf")
    assert not any("confidential footer" in c.content for c in chunks)


def test_prose_span_round_trips_to_page_text() -> None:
    text = "The first fact.\n\nThe second fact."
    page = _page(text)
    blocks = [_block(0, "text", "The first fact."), _block(1, "text", "The second fact.")]
    chunks = chunk_document([page], blocks, [], [], document_id="doc-sha", source_file="d.pdf")
    assert chunks
    for c in chunks:
        assert c.spans
        for s, e in c.spans:
            assert _collapse(page.text[s:e]) in _collapse(c.content)


def test_table_kept_whole_as_structured_content_not_prose() -> None:
    cells = [
        _cell(0, 0, "Metric", header=True),
        _cell(0, 1, "2005", header=True),
        _cell(1, 0, "Revenue"),
        _cell(1, 1, "15,295"),
    ]
    page = _page("Metric | 2005\nRevenue | 15,295")
    chunks = chunk_table(_table(cells), page, 0, document_id="doc-sha", source_file="d.pdf")
    assert len(chunks) == 1
    assert "Revenue | 15,295" in chunks[0].content
    # cited by bbox (cell-precise), not a char span
    assert chunks[0].spans == [] and chunks[0].bbox is not None


def test_large_table_splits_by_row_group_repeating_header_never_mid_row() -> None:
    header = [_cell(0, 0, "Property", header=True), _cell(0, 1, "Rooms", header=True)]
    body = []
    long_name = "Arizona Charlie's Boulder Station Resort and Casino Property " * 3
    for r in range(1, 30):
        body.append(_cell(r, 0, f"{long_name} {r}"))
        body.append(_cell(r, 1, str(1000 + r)))
    cells = header + body
    page = _page("dummy")
    frags = chunk_table(_table(cells), page, 0, document_id="doc-sha", source_file="d.pdf")
    assert len(frags) > 1, "table should have split"
    for frag in frags:
        assert "Property | Rooms" in frag.content, "every fragment repeats the header"
        for line in frag.content.splitlines():
            assert line.count(" | ") == 1, "no line is a partial row"


def test_prose_overlap_repeats_boundary_content() -> None:
    sentences = [f"Sentence number {i} carries a distinct fact about the firm." for i in range(60)]
    text = "\n\n".join(sentences)
    page = _page(text)
    blocks = [_block(i, "text", s) for i, s in enumerate(sentences)]
    chunks = chunk_document([page], blocks, [], [], document_id="doc-sha", source_file="d.pdf")
    assert len(chunks) >= 2, "prose should have split into multiple chunks"
    first_sentences = set(chunks[0].content.split("\n\n"))
    second_sentences = set(chunks[1].content.split("\n\n"))
    assert first_sentences & second_sentences, "no overlap between adjacent prose chunks"


def test_chart_becomes_its_own_chunk_cited_by_bbox() -> None:
    chart = ChartElement(
        page=3,
        bbox=BBox(x0=0, top=0, x1=100, bottom=100, page=3),
        caption_text="Revenue by segment",
        surrounding_text="FY2005 FY2006",
        flags=["chart_data_not_extracted"],
    )
    page = _page("dummy", page=3)
    chunks = chunk_document([page], [], [], [chart], document_id="doc-sha", source_file="d.pdf")
    assert len(chunks) == 1
    assert chunks[0].spans == [] and chunks[0].bbox is not None
    assert "Revenue by segment" in chunks[0].content


def test_provisional_size_cap_is_isolated_and_overlap_is_a_quarter() -> None:
    # Guards the "size cap is not locked" contract: it must stay a plain module
    # constant with a documented ratio, so the SIM-65 relock touches one place.
    assert isinstance(TARGET_TOKENS, int)
    assert OVERLAP_TOKENS == TARGET_TOKENS // 4

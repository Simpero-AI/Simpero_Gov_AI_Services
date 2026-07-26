"""AE-A-RETR-1 chunking -- cut a parsed document into immutable, citable,
searchable pieces, one document element at a time.

WHY ELEMENT-BASED, NOT TOKEN-WINDOWED
=====================================
A retrieval index cut by a fixed 512-token window across the flat page text
scored 68% page-retrieval accuracy on FinanceBench; cutting on the document's
own elements scored 84% (IBM/Unstructured, arXiv:2402.05131). The parser
already emits that element structure -- prose blocks, whole tables, chart
regions -- so the input is free. This module reads those elements and decides
the ONE thing element extraction did not: where the boundaries of a searchable
piece fall.

THE BOUNDARY RULES (locked; AE-A-RETR-1)
========================================
1. Element-based and PAGE-BOUNDED. A chunk never spans two pages, because the
   citation surface (PageIndex.text + char_map) is per-page: a span that
   crossed a page break could not resolve back to an exact source location.
2. A table is one chunk, serialized keeping its grid -- never flattened to
   prose, never summarized. Over the size cap, it splits by ROW GROUPS with the
   header row repeated in each fragment (duplicated, never divided), so no
   fragment loses the column labels its numbers are meaningless without.
3. Prose is packed from consecutive text blocks up to the size cap, with
   overlap between adjacent chunks so a fact straddling a boundary survives in
   both. A section heading starts a new chunk and rides in it.
4. A chart region is its own chunk (caption + nearby text), cited by page+bbox
   -- charts are the one element with no contiguous character span, because
   their text is gathered by geometry, not read in reading order.
5. Boilerplate -- running headers/footers, repeated disclaimers -- never enters
   a chunk. Boilerplate in every chunk makes every chunk look similar to every
   other, which is the one thing a retrieval index must never do.

THE SIZE CAP IS NOT ONE OF THE RULES
====================================
TARGET_TOKENS below is a documented STARTING POINT, not a locked decision. The
number is settled empirically against the golden set (SIM-65 / GS-4), not here;
changing it later costs a re-chunk + re-embed, so it is isolated as one swappable
constant rather than threaded through the boundary logic that IS locked.

Chunk-level metadata (document_id, scale_context, element_type) is a separate
step, AE-A-RETR-2, layered onto these records at creation.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from .elements import ChartElement, TableElement
from .resolver import find_exact_span
from .schemas import BBox, PageIndex, TableCellRecord, TextBlockRecord

# Provisional starting point (~512 tokens, 25% overlap), NOT a locked decision --
# see the module docstring. The boundary rules do not depend on this number.
TARGET_TOKENS = 512
OVERLAP_TOKENS = 128

# Token count is estimated, not tokenized: the real tokenizer arrives with the
# embedding model (SIM-245), which is itself not yet locked. A character proxy
# keeps chunking independent of that choice; it is deliberately a single
# function so the eventual tokenizer swaps in one place.
_CHARS_PER_TOKEN = 4

# Blocks that carry prose worth indexing. Mirrors propose.PROSE_LABELS -- the
# same allowlist the numeric/qualitative extractors read from, so a block that
# is a claim source and a block that is a chunk are the same set.
PROSE_LABELS = frozenset({"text", "paragraph", "list_item", "footnote", "caption"})
# A heading governs the blocks beneath it; it starts a chunk and names a section.
SECTION_LABEL = "section_header"
# Docling's running-furniture labels. Excluded by label AND, below, by the
# repetition-based char_map.is_boilerplate flag -- because the label alone is
# not trustworthy (a reproduced press clipping was labelled page_header on a
# real CIM), so a block is furniture if EITHER signal says so.
BOILERPLATE_LABELS = frozenset({"page_header", "page_footer"})


class ChunkRecord(BaseModel):
    """One immutable, citable, searchable piece of a parsed document.

    `spans` are (char_start, char_end) pairs into PageIndex.text, one per source
    block, so page.text[char_start:char_end] round-trips the block's own text
    even for a multi-block prose chunk (the flat page index interleaves table
    cells between prose blocks, so a single covering range could swallow table
    text -- segments keep each piece exact). A table and a chart carry no span:
    a serialized grid is not a contiguous page.text substring, and a chart's
    text is gathered by geometry, so `bbox` is the citation for both.
    """

    content: str
    page: int = Field(ge=1)
    order: int = Field(ge=0, description="Document-order sort key, stable across a run.")
    source_file: str

    spans: list[tuple[int, int]] = Field(default_factory=list)
    bbox: BBox | None = None
    section: str | None = None
    flags: list[str] = Field(default_factory=list)


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // _CHARS_PER_TOKEN)


def _resolve_span(text_normalized: str, page: PageIndex) -> tuple[int, int] | None:
    """A block's exact (char_start, char_end) in page.text, or None when it does
    not resolve uniquely -- the same fail-closed rule the claim path cites under."""
    return find_exact_span(text_normalized, page.text, where=f"chunk on page {page.page}")


def _is_boilerplate(block: TextBlockRecord, span: tuple[int, int] | None, page: PageIndex) -> bool:
    """True when a block is running furniture and must not enter a chunk.

    Two independent signals, because neither is sufficient alone: Docling's label
    mislabels real content as page_header, and the repetition flag lives on
    characters, not blocks. A block is furniture if its label says so, OR if ANY
    of the characters of its resolved span are boilerplate-flagged.

    The char test is deliberately ANY, not a majority: tag_boilerplate marks the
    repeated runs it is confident about (header/footer-zone lines repeating across
    >= 3 pages), and a block carrying even one flagged run is either furniture or a
    body block fused to a footer -- both put a repeated string into the index,
    which is exactly what rule 5 forbids. A body sentence that merely shares a word
    with the footer is not flagged, because the flag is set on repeated LINES, not
    on shared vocabulary.
    """
    if block.label in BOILERPLATE_LABELS:
        return True
    if span is None:
        return False
    start, end = span
    return any(c.is_boilerplate for c in page.char_map[start:end])


# --------------------------------------------------------------------------- #
# Prose
# --------------------------------------------------------------------------- #


def _blocks_bbox(blocks: list[TextBlockRecord], page_no: int) -> BBox:
    return BBox(
        x0=min(b.x0 for b in blocks),  # type: ignore[type-var]
        top=min(b.top for b in blocks),  # type: ignore[type-var]
        x1=max(b.x1 for b in blocks),  # type: ignore[type-var]
        bottom=max(b.bottom for b in blocks),  # type: ignore[type-var]
        page=page_no,
    )


def _flush_prose(
    group: list[tuple[TextBlockRecord, tuple[int, int] | None]],
    page: PageIndex,
    *,
    source_file: str,
    section: str | None,
) -> ChunkRecord | None:
    """Turn a group of accumulated prose blocks into one ChunkRecord."""
    if not group:
        return None
    content = "\n\n".join(block.text_normalized for block, _ in group)
    spans = [span for _, span in group if span is not None]
    located = [block for block, _ in group if block.x0 is not None]
    bbox = _blocks_bbox(located, page.page) if located else None
    flags: list[str] = []
    if any(span is None for _, span in group):
        # A block that did not resolve to a unique span is still indexed (its
        # content is real), but the miss is recorded rather than hidden: an
        # unspanned chunk is a citation gap, and citation gaps must stay visible.
        flags.append("span_unresolved")
    return ChunkRecord(
        content=content,
        page=page.page,
        order=group[0][0].order,
        source_file=source_file,
        spans=spans,
        bbox=bbox,
        section=section,
        flags=flags,
    )


def chunk_prose_page(
    blocks: list[TextBlockRecord],
    page: PageIndex,
    *,
    source_file: str,
    section: str | None,
) -> tuple[list[ChunkRecord], str | None]:
    """Chunk one page's prose blocks; return (chunks, section carried out).

    `section` flows in from the previous page and out to the next, because a
    section spans pages while a chunk cannot: a page after a heading inherits
    the heading's section name without re-stating it.
    """
    ordered = sorted(blocks, key=lambda b: b.order)
    chunks: list[ChunkRecord] = []
    group: list[tuple[TextBlockRecord, tuple[int, int] | None]] = []
    group_tokens = 0

    def flush() -> None:
        nonlocal group, group_tokens
        chunk = _flush_prose(group, page, source_file=source_file, section=section)
        if chunk is not None:
            chunks.append(chunk)
        # Overlap: carry the trailing blocks that sum to ~OVERLAP_TOKENS into the
        # next chunk, so a fact on the boundary survives in both.
        carried: list[tuple[TextBlockRecord, tuple[int, int] | None]] = []
        carried_tokens = 0
        for item in reversed(group):
            t = _estimate_tokens(item[0].text_normalized)
            if carried_tokens + t > OVERLAP_TOKENS and carried:
                break
            carried.insert(0, item)
            carried_tokens += t
        group = carried
        group_tokens = carried_tokens

    for block in ordered:
        if block.label == SECTION_LABEL:
            span = _resolve_span(block.text_normalized, page)
            # A heading that is itself running furniture (a repeated banner
            # Docling mislabelled section_header) is not a section boundary and
            # not content -- skip it before it can name a section or enter a chunk.
            if _is_boilerplate(block, span, page):
                continue
            # A heading is a hard boundary: flush what precedes it (no overlap
            # carried across a section change), then start the new section's
            # first chunk with the heading itself.
            if group:
                chunk = _flush_prose(group, page, source_file=source_file, section=section)
                if chunk is not None:
                    chunks.append(chunk)
                group, group_tokens = [], 0
            section = block.text_normalized.strip() or section
            group.append((block, span))
            group_tokens += _estimate_tokens(block.text_normalized)
            continue
        if block.label not in PROSE_LABELS:
            continue
        span = _resolve_span(block.text_normalized, page)
        if _is_boilerplate(block, span, page):
            continue
        tokens = _estimate_tokens(block.text_normalized)
        if group_tokens + tokens > TARGET_TOKENS and group_tokens > 0:
            flush()
        group.append((block, span))
        group_tokens += tokens

    chunk = _flush_prose(group, page, source_file=source_file, section=section)
    if chunk is not None:
        chunks.append(chunk)
    return chunks, section


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #


def _row_indices(cells: list[TableCellRecord]) -> list[int]:
    return sorted({c.row for c in cells})


def _serialize_rows(cells: list[TableCellRecord], rows: list[int]) -> str:
    lines: list[str] = []
    for r in rows:
        row_cells = sorted((c for c in cells if c.row == r), key=lambda c: c.col)
        lines.append(" | ".join(c.text_normalized for c in row_cells))
    return "\n".join(lines)


def _cells_bbox(cells: list[TableCellRecord], page_no: int) -> BBox | None:
    located = [c for c in cells if c.x0 is not None]
    if not located:
        return None
    return BBox(
        x0=min(c.x0 for c in located),  # type: ignore[type-var]
        top=min(c.top for c in located),  # type: ignore[type-var]
        x1=max(c.x1 for c in located),  # type: ignore[type-var]
        bottom=max(c.bottom for c in located),  # type: ignore[type-var]
        page=page_no,
    )


def _header_rows(table: TableElement) -> list[int]:
    """The rows repeated in every fragment. Docling's per-cell column_header flag
    is advisory (it marks the first data row on unlabeled tables), so the header
    is taken as every row at or above the topmost cell flagged column_header;
    when nothing is flagged, there is no header to repeat."""
    flagged = [c.row for c in table.cells if c.column_header]
    if not flagged:
        return []
    top = min(_row_indices(table.cells))
    return [r for r in _row_indices(table.cells) if top <= r <= max(flagged)]


def chunk_table(
    table: TableElement,
    page: PageIndex,
    order: int,
    *,
    source_file: str,
) -> list[ChunkRecord]:
    """One table -> one chunk, or several ROW-GROUP fragments when it exceeds the
    size cap. Each fragment repeats the header rows, so no fragment's numbers are
    orphaned from their column labels."""
    all_rows = _row_indices(table.cells)
    header_rows = _header_rows(table)
    body_rows = [r for r in all_rows if r not in header_rows]

    def make(rows: list[int], suffix_flags: list[str]) -> ChunkRecord:
        cells = [c for c in table.cells if c.row in rows]
        return ChunkRecord(
            content=_serialize_rows(table.cells, rows),
            page=page.page,
            order=order,
            source_file=source_file,
            # A serialized grid is not a contiguous page.text substring, so a
            # table cites by its cell-precise bbox, not a char span -- the same
            # citation the claim path already trusts for table cells.
            spans=[],
            bbox=_cells_bbox(cells, page.page),
            flags=list(table.flags) + suffix_flags,
        )

    whole_rows = header_rows + body_rows
    if _estimate_tokens(_serialize_rows(table.cells, whole_rows)) <= TARGET_TOKENS or not body_rows:
        return [make(whole_rows, [])]

    # Over the cap: emit header + as many body rows as fit, repeating the header.
    fragments: list[ChunkRecord] = []
    header_tokens = _estimate_tokens(_serialize_rows(table.cells, header_rows))
    current: list[int] = []
    current_tokens = header_tokens
    for r in body_rows:
        row_tokens = _estimate_tokens(_serialize_rows(table.cells, [r]))
        if current and current_tokens + row_tokens > TARGET_TOKENS:
            fragments.append(make(header_rows + current, ["table_split"]))
            current, current_tokens = [], header_tokens
        current.append(r)
        current_tokens += row_tokens
    if current:
        fragments.append(make(header_rows + current, ["table_split"] if fragments else []))
    return fragments


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #


def chunk_chart(chart: ChartElement, order: int, *, source_file: str) -> ChunkRecord | None:
    """A chart region as one chunk: caption + the text gathered around it. Cited
    by page+bbox -- a chart has no contiguous character span. Skipped when it
    carries no text at all (an image with nothing to index)."""
    parts = [t for t in (chart.caption_text.strip(), chart.surrounding_text.strip()) if t]
    if not parts:
        return None
    return ChunkRecord(
        content="\n\n".join(parts),
        page=chart.page,
        order=order,
        source_file=source_file,
        spans=[],
        bbox=chart.bbox,
        flags=list(chart.flags),
    )


# --------------------------------------------------------------------------- #
# Document
# --------------------------------------------------------------------------- #


def chunk_document(
    pages: list[PageIndex],
    blocks: list[TextBlockRecord],
    table_elements: list[TableElement],
    chart_elements: list[ChartElement],
    *,
    source_file: str,
) -> list[ChunkRecord]:
    """Cut a whole parsed PDF into chunks, in document order.

    Order is assigned page by page so a consumer can restore reading order from a
    flat chunk list. Section context flows across pages; every other boundary is
    page-local by construction.
    """
    blocks_by_page: dict[int, list[TextBlockRecord]] = {}
    for block in blocks:
        blocks_by_page.setdefault(block.page, []).append(block)
    tables_by_page: dict[int, list[TableElement]] = {}
    for table in table_elements:
        tables_by_page.setdefault(table.page, []).append(table)
    charts_by_page: dict[int, list[ChartElement]] = {}
    for chart in chart_elements:
        charts_by_page.setdefault(chart.page, []).append(chart)

    chunks: list[ChunkRecord] = []
    section: str | None = None
    order = 0
    for page in sorted(pages, key=lambda p: p.page):
        prose, section = chunk_prose_page(
            blocks_by_page.get(page.page, []), page, source_file=source_file, section=section
        )
        for chunk in prose:
            chunk.order = order
            order += 1
            chunks.append(chunk)
        for table in tables_by_page.get(page.page, []):
            for chunk in chunk_table(table, page, order, source_file=source_file):
                chunk.order = order
                order += 1
                chunks.append(chunk)
        for chart in charts_by_page.get(page.page, []):
            chunk = chunk_chart(chart, order, source_file=source_file)
            if chunk is not None:
                chunk.order = order
                order += 1
                chunks.append(chunk)
    return chunks

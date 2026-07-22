"""DS-W3-3 exact-span resolver — the citation trust boundary.

A claim exists only if its quote resolves to an exact, unambiguous span in the
source page. Matching is deterministic and whitespace-flexible: every
non-whitespace character must match literally, while a run of whitespace in the
quote matches any run of whitespace in the page text. That single relaxation is
principled, not fuzzy — it lets a quote emitted with a space resolve against
source text that wraps with a newline (the flat page index carries a real '\\n'
between visual lines), which an exact-substring match would silently drop.
Found exactly once -> a real span + bbox; not found, or found more than once ->
None. No "closest sentence", no fallback slice. Approximating a citation is
worse than dropping the claim, because a wrong span looks correct and points at
unrelated text — the failure mode this ticket exists to remove.
"""

import logging
import re

from .schemas import (
    BBox,
    CharBox,
    PageIndex,
    ParagraphIndex,
    ParagraphSpan,
    Span,
    TableCellRecord,
)

logger = logging.getLogger(__name__)

_WHITESPACE = re.compile(r"\s+")


def _flexible_pattern(quote: str) -> str:
    """Regex matching the quote with whitespace runs treated flexibly.

    Non-whitespace tokens are matched literally (`re.escape`); each gap between
    them matches any run of whitespace. The quote's own leading/trailing
    whitespace is insignificant. Callers guarantee a non-whitespace-only quote,
    so the token list is never empty.
    """
    tokens = [t for t in _WHITESPACE.split(quote) if t]
    return r"\s+".join(re.escape(t) for t in tokens)


def union_bbox(chars: list[CharBox]) -> BBox:
    """Merge char boxes into one enclosing box. TOPLEFT origin (top < bottom).

    Every char in a PageIndex belongs to the same page, so the union takes its
    page from the first box. Requires a non-empty list — resolve() only calls it
    with the matched characters of a quote it already found.
    """
    if not chars:
        raise ValueError("union_bbox requires at least one char box")
    pages = {c.page for c in chars}
    if len(pages) != 1:
        raise ValueError(f"union_bbox requires chars on one page, got {sorted(pages)}")
    return BBox(
        x0=min(c.x0 for c in chars),
        top=min(c.top for c in chars),
        x1=max(c.x1 for c in chars),
        bottom=max(c.bottom for c in chars),
        page=chars[0].page,
    )


def _line_bboxes(chars: list[CharBox]) -> list[BBox]:
    """One union box per visual line, splitting the matched chars on newlines.

    The flat page index carries a real '\\n' CharBox between visual lines, so a
    quote that wraps contains newline entries here. Producing a box per run
    between newlines gives the correct thing to highlight, instead of one
    rectangle that would span the gap between lines. Newline-only input yields an
    empty list (there is no line content to draw).
    """
    boxes: list[BBox] = []
    run: list[CharBox] = []
    for c in chars:
        if c.char == "\n":
            if run:
                boxes.append(union_bbox(run))
                run = []
        else:
            run.append(c)
    if run:
        boxes.append(union_bbox(run))
    return boxes


def find_exact_span(quote: str, text: str, *, where: str) -> tuple[int, int] | None:
    """The citation trust boundary itself, shared by every source format.

    Returns (char_start, char_end) for a quote that occurs exactly once in
    `text`, else None. Fail-closed: an empty (or whitespace-only) quote, a quote
    not present, or a quote present more than once all return None. Only
    whitespace is flexible; every other character must match verbatim, and the
    quote must already be normalized the same way `text` is — e.g. "3,817"
    resolves only after DS-W3-1 has collapsed the raw "3 ,817".

    Deliberately geometry-free and format-agnostic: a PDF page and a DOCX
    paragraph are both just linear text, and they must never disagree about what
    counts as a resolved citation. Two implementations of this rule would be two
    trust boundaries that can drift — so there is exactly one. `where` is used
    only to make the ambiguity log locatable.
    """
    if not quote.strip():
        return None

    pattern = _flexible_pattern(quote)
    # Lookahead so overlapping occurrences are still counted (e.g. "aa" in "aaa"
    # appears twice); group(1) captures the real span at each start position.
    matches = [(m.start(1), m.end(1)) for m in re.finditer(rf"(?=({pattern}))", text)]
    if not matches:
        return None

    if len(matches) > 1:
        # Ambiguous: the quote resolves at more than one place, so which instance
        # the claim refers to is unknowable. Fail closed and log it — these are
        # recall gaps (a real value we refused to cite) that must stay visible.
        logger.warning(
            "Ambiguous quote not resolved in %s: %r appears more than once",
            where,
            quote,
        )
        return None

    return matches[0]


def contains_flexible(haystack: str, needle: str) -> bool:
    """Whether `needle` appears in `haystack` under this module's matching rule.

    The same relaxation find_exact_span uses -- a run of whitespace matches any
    run of whitespace, every other character literally -- so a caller asking
    "is this clause inside that quote" gets the answer the resolver would give,
    rather than a second, subtly different grammar. Hand-rolled containment
    tests are how `has_parseable_magnitude`'s precondition came to be answered
    three different ways.

    A blank needle is False, never vacuously True. "" is a substring of
    everything, and that exact vacuity already shipped one defect here: an
    unset value_text passed the containment test and became a claim's raw.
    """
    if not needle.strip():
        return False
    return re.search(_flexible_pattern(needle), haystack) is not None


def resolve(quote: str, page: PageIndex) -> Span | None:
    """Resolve a verbatim quote to its exact span on a PDF page, or None.

    The PDF path: the shared exact-span rule plus the geometry the positioned
    index carries, so the resulting Span always has a real bbox.
    """
    found = find_exact_span(quote, page.text, where=f"page {page.page}")
    if found is None:
        return None

    start, end = found
    chars = page.char_map[start:end]
    return Span(
        char_start=start,
        char_end=end,
        page=page.page,
        bbox=union_bbox(chars),
        line_bboxes=_line_bboxes(chars),
    )


# How far past a cell's border a glyph's centre may sit and still belong to it.
#
# Docling's cell rect can be NARROWER than the text inside it. Measured on a real
# CIM: the cell holding "(6)" spans x 307.19..315.95 while its closing ")" spans
# 315.66..318.17 -- the glyph is 88% outside its own cell, and its centre (316.92)
# misses by a point.
#
# This never showed while every CharBox carried its whole WORD's box: all three
# glyphs shared one centre, which sat comfortably inside. Real per-glyph geometry
# made the test bite, dropping the last character and turning 130 resolvable cells
# into `missing` on one document -- the exact failure this function's docstring
# predicted a stricter test would cause.
#
# Sized against the gap between cells, not the glyph: adjacent value cells on that
# page start ~26pt apart, so a few points cannot reach a neighbour's text. A pad
# large enough to steal from the next cell would be a precision bug; this is two
# orders of magnitude below that.
#
# HORIZONTAL ONLY, and that is not an oversight. Rows sit ~8pt apart, so the same
# pad applied vertically reaches into the row above and below: tried at 3pt in
# both axes it pulled neighbouring rows' digits into the cell, made quotes
# ambiguous, and -- because ambiguity fails closed -- took `missing` on one
# document from 879 to 1,474. The clipping this corrects is a horizontal
# phenomenon (a glyph running past the right border); vertically the cell rect
# already contains its text.
_CELL_EDGE_TOLERANCE_PT = 3.0


def _chars_in_cell(page: PageIndex, cell: TableCellRecord) -> list[int]:
    """Indices of the page chars whose centres lie inside `cell`'s box, within a
    small edge tolerance.

    Centre-point containment rather than full overlap: a glyph box can spill a
    fraction of a point past a tight cell border without belonging to the
    neighbour, and a stricter test would drop the first or last character of a
    value and turn a resolvable cell into a miss. The tolerance extends that
    same reasoning to a border tight enough to clip the glyph outright -- see
    _CELL_EDGE_TOLERANCE_PT.
    """
    if cell.x0 is None or cell.top is None or cell.x1 is None or cell.bottom is None:
        return []
    pad = _CELL_EDGE_TOLERANCE_PT
    return [
        i
        for i, char in enumerate(page.char_map)
        if char.page == cell.page
        and cell.x0 - pad <= (char.x0 + char.x1) / 2 <= cell.x1 + pad
        and cell.top <= (char.top + char.bottom) / 2 <= cell.bottom
    ]


def resolve_in_cell(quote: str, page: PageIndex, cell: TableCellRecord) -> Span | None:
    """Resolve a quote inside one table cell's own region, or None.

    Same trust rule as `resolve` -- the shared exact-span core, fail-closed --
    but searching only the characters that sit inside the cell's box instead of
    the whole page.

    This exists because page-wide search cannot cite a repeated value, and
    financial tables repeat values constantly. PTL PDF-page 11:

        Gross Margin %   27.3%   21.2%   21.2%   21.2%

    Searching the page for "21.2%" finds three matches, so `resolve` fails
    closed and all three real figures are dropped -- a recall gap DS-W3-8's
    harness renders as three red boxes. But the ambiguity is an artefact of the
    question, not the document: the caller knows exactly which cell it is
    reading, and each cell's box contains exactly one "21.2%". Scoping the
    search to the cell asks a question that has one answer.

    The returned span is still page-relative, because a citation must address
    the page the reader sees, not a cell-local offset.

    Requires the cell's characters to be contiguous in reading order, and fails
    closed otherwise. A non-contiguous run means the cell's box encloses text
    that the page index interleaves with another cell's, so no single
    (char_start, char_end) could describe it, and a span that silently spanned
    the gap would cite the neighbour's text too.
    """
    indices = _chars_in_cell(page, cell)
    if not indices:
        return None

    if indices != list(range(indices[0], indices[-1] + 1)):
        logger.warning(
            "Cell (%d,%d) on page %d encloses a non-contiguous char run; not citable",
            cell.row,
            cell.col,
            page.page,
        )
        return None

    offset = indices[0]
    cell_text = page.text[offset : indices[-1] + 1]
    found = find_exact_span(
        quote, cell_text, where=f"page {page.page} cell ({cell.row},{cell.col})"
    )
    if found is None:
        return None

    start, end = (offset + found[0], offset + found[1])
    chars = page.char_map[start:end]
    return Span(
        char_start=start,
        char_end=end,
        page=page.page,
        bbox=union_bbox(chars),
        line_bboxes=_line_bboxes(chars),
    )


def resolve_in_paragraph(quote: str, paragraph: ParagraphIndex) -> ParagraphSpan | None:
    """Resolve a verbatim quote to its exact span in a DOCX paragraph, or None.

    Identical trust rule to the PDF path — same `find_exact_span`, same
    fail-closed semantics — with no geometry, because a Word file has no page
    layout to report (Docling returns no prov for DOCX at all). The citation is
    (paragraph, char_start, char_end), exactly what the claims contract's docx
    location carries.
    """
    found = find_exact_span(quote, paragraph.text, where=f"paragraph {paragraph.paragraph}")
    if found is None:
        return None

    start, end = found
    return ParagraphSpan(char_start=start, char_end=end, paragraph=paragraph.paragraph)

"""DS-W3-3 exact-span resolver — the citation trust boundary.

A fact exists only if its quote resolves to an exact, unambiguous span in the
source page. This is deterministic exact-substring matching: found exactly once
-> a real span + bbox; not found, or found more than once -> None. There is no
fuzzy matching, no "closest sentence", no fallback slice. Approximating a
citation is worse than dropping the fact, because a wrong span looks correct and
points at unrelated text — the exact failure mode this ticket exists to remove.
"""

import logging

from .schemas import BBox, CharBox, PageIndex, Span

logger = logging.getLogger(__name__)


def union_bbox(chars: list[CharBox]) -> BBox:
    """Merge char boxes into one enclosing box. TOPLEFT origin (top < bottom).

    Every char in a PageIndex belongs to the same page, so the union takes its
    page from the first box. Requires a non-empty list — resolve() only calls it
    with the matched characters of a quote it already found.
    """
    if not chars:
        raise ValueError("union_bbox requires at least one char box")
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


def resolve(quote: str, page: PageIndex) -> Span | None:
    """Resolve a verbatim quote to its exact span on a page, or None.

    Fail-closed: an empty quote, a quote not present, or a quote present more
    than once all return None. The quote must be a verbatim substring of
    page.text (the extractor must emit verbatim quotes, never restatements) and
    must already be normalized the same way page.text is — e.g. "3,817" resolves
    only after DS-W3-1 has collapsed the raw "3 ,817"; resolving it against
    un-normalized text is a correct None, not a bug.
    """
    if not quote:
        return None

    first = page.text.find(quote)
    if first == -1:
        return None

    if page.text.find(quote, first + 1) != -1:
        # Ambiguous: the same string appears more than once, so which instance
        # the fact refers to is unknowable. Fail closed and log it — these are
        # recall gaps (a real value we refused to cite) that must stay visible.
        logger.warning(
            "Ambiguous quote not resolved on page %s: %r appears more than once",
            page.page,
            quote,
        )
        return None

    chars = page.char_map[first : first + len(quote)]
    return Span(
        char_start=first,
        char_end=first + len(quote),
        page=page.page,
        bbox=union_bbox(chars),
        line_bboxes=_line_bboxes(chars),
    )

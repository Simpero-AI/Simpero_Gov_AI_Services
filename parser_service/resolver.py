"""DS-W3-3 exact-span resolver — the citation trust boundary.

A fact exists only if its quote resolves to an exact, unambiguous span in the
source page. Matching is deterministic and whitespace-flexible: every
non-whitespace character must match literally, while a run of whitespace in the
quote matches any run of whitespace in the page text. That single relaxation is
principled, not fuzzy — it lets a quote emitted with a space resolve against
source text that wraps with a newline (the flat page index carries a real '\\n'
between visual lines), which an exact-substring match would silently drop.
Found exactly once -> a real span + bbox; not found, or found more than once ->
None. No "closest sentence", no fallback slice. Approximating a citation is
worse than dropping the fact, because a wrong span looks correct and points at
unrelated text — the failure mode this ticket exists to remove.
"""

import logging
import re

from .schemas import BBox, CharBox, PageIndex, Span

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


def resolve(quote: str, page: PageIndex) -> Span | None:
    """Resolve a verbatim quote to its exact span on a page, or None.

    Fail-closed: an empty (or whitespace-only) quote, a quote not present, or a
    quote present more than once all return None. Non-whitespace content must
    match verbatim (the extractor emits verbatim quotes, never restatements) and
    must already be normalized the same way page.text is — e.g. "3,817" resolves
    only after DS-W3-1 has collapsed the raw "3 ,817". Only whitespace is matched
    flexibly, so a quote that wraps a line still resolves.
    """
    if not quote.strip():
        return None

    pattern = _flexible_pattern(quote)
    # Lookahead so overlapping occurrences are still counted (e.g. "aa" in "aaa"
    # appears twice); group(1) captures the real span at each start position.
    matches = [(m.start(1), m.end(1)) for m in re.finditer(rf"(?=({pattern}))", page.text)]
    if not matches:
        return None

    if len(matches) > 1:
        # Ambiguous: the quote resolves at more than one place, so which instance
        # the fact refers to is unknowable. Fail closed and log it — these are
        # recall gaps (a real value we refused to cite) that must stay visible.
        logger.warning(
            "Ambiguous quote not resolved on page %s: %r appears more than once",
            page.page,
            quote,
        )
        return None

    start, end = matches[0]
    chars = page.char_map[start:end]
    return Span(
        char_start=start,
        char_end=end,
        page=page.page,
        bbox=union_bbox(chars),
        line_bboxes=_line_bboxes(chars),
    )

"""Numeric coverage -- a self-measuring recall instrument.

Every financially-meaningful number a page prints should end up as a claim, or be
accounted for. This sweeps the page for those numbers (inspect.detect_numeric_tokens
-- the same detector the coverage overlay uses, already filtered to figures that
carry a "$", "%", scale suffix, decimal, or thousands comma, so page numbers and
footnote markers are out) and checks each against the spans of the claims the
pipeline actually emitted:

  covered     -- a claim's span covers the number: the pipeline extracted it.
  gate1 miss  -- uncovered, but the number resolves to a UNIQUE span, so a claim
                 could have cited it -- the model simply did not propose it. This
                 is the recall gap worth closing (a completeness re-pass, or an
                 ensemble that unions several extraction passes).
  gate2 miss  -- uncovered and the number does not resolve uniquely (it appears
                 more than once on the page, or was normalised away): a
                 provenance-layer loss, not a proposal loss.

The point of splitting the two is that they take different fixes, and only the
gate-1 population is addressable by asking the model for more. Deterministic, and
needs NO ground truth: the page and the claims it produced are enough to say what
was left on the floor -- which is what makes it usable before any golden set.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Literal

from pydantic import BaseModel, Field

from .inspect import detect_numeric_tokens, is_boilerplate_token
from .resolver import find_exact_span
from .schemas import PageIndex

# How much surrounding text to keep with a miss, so a completeness re-pass (or a
# human) can see what the number is about without re-opening the page.
_CONTEXT_PAD = 60


class NumberMiss(BaseModel):
    """A financially-meaningful number on the page that no claim covers."""

    page: int
    text: str
    start: int
    end: int
    context: str
    # 1: resolvable but unclaimed (the model did not propose it) -- addressable.
    # 2: unresolvable/ambiguous -- a provenance loss, not a proposal loss.
    gate: Literal[1, 2]


class PageCoverage(BaseModel):
    page: int
    total: int
    covered: int
    misses: list[NumberMiss] = Field(default_factory=list)


class DocumentCoverage(BaseModel):
    total: int
    covered: int
    gate1: int
    gate2: int
    pages: list[PageCoverage] = Field(default_factory=list)

    @property
    def coverage(self) -> float:
        """Fraction of meaningful numbers that became a claim. 1.0 when the page
        prints no such number -- an empty page is fully covered, not undefined."""
        return self.covered / self.total if self.total else 1.0

    @property
    def gate1_misses(self) -> list[NumberMiss]:
        return [m for p in self.pages for m in p.misses if m.gate == 1]


def _overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start < b_end and b_start < a_end


def _window(text: str, start: int, end: int) -> str:
    return text[max(0, start - _CONTEXT_PAD) : end + _CONTEXT_PAD].strip()


def page_coverage(page: PageIndex, claim_spans: list[tuple[int, int]]) -> PageCoverage:
    """Coverage for one page. `claim_spans` are the (char_start, char_end) ranges
    of the claims emitted FROM THIS PAGE -- a number is covered when a claim's span
    overlaps it (the number sits inside the quote the claim cites)."""
    tokens = [t for t in detect_numeric_tokens(page.text) if not is_boilerplate_token(t, page)]
    covered = 0
    misses: list[NumberMiss] = []
    for token in tokens:
        if any(_overlaps(token.start, token.end, cs, ce) for cs, ce in claim_spans):
            covered += 1
            continue
        span = find_exact_span(token.text, page.text, where=f"coverage p{page.page}")
        resolvable = span is not None
        misses.append(
            NumberMiss(
                page=page.page,
                text=token.text,
                start=token.start,
                end=token.end,
                context=_window(page.text, token.start, token.end),
                gate=1 if resolvable else 2,
            )
        )
    return PageCoverage(page=page.page, total=len(tokens), covered=covered, misses=misses)


def document_coverage(
    pages: list[PageIndex], claim_spans: list[tuple[int, int, int]]
) -> DocumentCoverage:
    """Coverage for a whole document. `claim_spans` are (page, char_start, char_end)
    for every emitted, cited claim -- the caller pulls these off the claims (a
    `missing` claim has no span and contributes nothing, correctly, since it cited
    nothing)."""
    spans_by_page: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for page_no, start, end in claim_spans:
        spans_by_page[page_no].append((start, end))

    page_reports = [page_coverage(page, spans_by_page.get(page.page, [])) for page in pages]
    total = sum(p.total for p in page_reports)
    covered = sum(p.covered for p in page_reports)
    gate1 = sum(1 for p in page_reports for m in p.misses if m.gate == 1)
    gate2 = sum(1 for p in page_reports for m in p.misses if m.gate == 2)
    return DocumentCoverage(
        total=total, covered=covered, gate1=gate1, gate2=gate2, pages=page_reports
    )

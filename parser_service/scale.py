"""DS-W3-4 scale capture -- determines a numeric fact's scale multiplier and
records HOW it was determined (scale_source), per the C3 facts contract
(contracts/facts.schema.json).

A financial document states scale in a header, not on the value: PTL PDF-page
11 prints Revenue as "$15,295" under a "CAD (in Thousands)" header several
rows up -- the true value is $15,295,000. An assumed-1x value and a
genuine-1x value normalize to the identical number, so a silently-assumed
scale is indistinguishable from a known one unless scale_source is recorded
alongside it. This module never returns a bare multiplier without saying
where it came from, and never guesses a currency it did not actually read.

Resolution order (first match wins):
  1. Inline in the value itself ("$4.8M", "500K", "27.3%") -> explicit_in_value.
  2. A scale phrase in the value's own table column, walking upward
     (DS-W3-2 table structure) -> column_header.
  3. A scale phrase anywhere earlier on the page -> page_header.
  4. Nothing found -> multiplier 1, scale_source=assumed_1x, flagged
     ("scale_assumed") -- never silent, never indistinguishable from a
     verified 1x.
"""

import re
from typing import Literal

from pydantic import BaseModel, Field

from .schemas import PageIndex, TableCellRecord, TableRecord

ScaleSource = Literal["explicit_in_value", "column_header", "page_header", "assumed_1x"]


class ScaleResult(BaseModel):
    """raw/normalized/unit/scale_multiplier/scale_source mirror the value
    JSONB keys in the C3 facts contract. scale_context (the exact phrase that
    justified a column_header/page_header multiplier) and flags are not wire
    contract fields themselves -- DS-7 decides where that provenance detail is
    surfaced (e.g. the structured flag log / the fact's flags array)."""

    raw: str
    normalized: float
    unit: str | None
    scale_multiplier: float
    scale_source: ScaleSource
    scale_context: str | None = None
    flags: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Step 1: inline scale -- a value that states its own multiplier.
#
# Ported from the MVP's normalizeFinancialTokens (server/pipeline.ts) -- the
# three suffix regexes and the percent regex below are direct translations of
# that function's replace() patterns, verified against it line for line. The
# MVP version returned a mutated prose string (it fed TF-IDF tokenization,
# which DS-3 replaces); this port keeps only the recognition grammar and
# returns a structured (number, multiplier, unit) instead, since DS-4 needs a
# multiplier, not a token.
# --------------------------------------------------------------------------- #

# Each pattern: an optional sign ("-" or an opening "(" for accounting
# negative notation), an optional "$", the number, optional whitespace, the
# suffix letter, an optional spelled-out/doubled continuation, a word
# boundary, and an optional closing ")" pairing with the opening one.
#
# The continuation group is deliberately narrow -- for the M/B patterns it
# only completes "illion" or repeats the same letter ("MM", "BB", mixed
# case), matching the ported source exactly. It does NOT accept "Bn"/"Mn":
# "$3Bn" fails the trailing \b (two word characters "B" and "n" touch, so
# there is no boundary), same as in the MVP. Extending the vocabulary here
# would silently diverge from the "well-tested" function this ticket says to
# port.
_MILLION_RE = re.compile(r"(?P<sign>[-(])?\$?(?P<number>[\d,.]+)\s*[Mm](?:illion|M|m)?\b\)?")
_BILLION_RE = re.compile(r"(?P<sign>[-(])?\$?(?P<number>[\d,.]+)\s*[Bb](?:illion|B|b)?\b\)?")
_THOUSAND_RE = re.compile(r"(?P<sign>[-(])?\$?(?P<number>[\d,.]+)\s*[Kk](?:thousand)?\b\)?")

# Checked in this order (matching the MVP's M -> B -> K replace() sequence):
# for an isolated single-value token this only matters in that the first
# pattern to match wins, which mirrors the source's behavior for any string
# that contains just one financial figure.
_SUFFIX_PATTERNS: list[tuple[re.Pattern[str], float]] = [
    (_MILLION_RE, 1_000_000.0),
    (_BILLION_RE, 1_000_000_000.0),
    (_THOUSAND_RE, 1_000.0),
]

# A bare percentage. The "%" sign is itself a complete, self-scaling unit
# statement and must never be multiplied by a page/column scale header --
# a 27.3% Gross Margin figure on a "CAD (in Thousands)" page is still 27.3%,
# not 27,300%. Checked before the suffix patterns so "%" is never reached by
# them.
_PERCENT_RE = re.compile(r"(?P<sign>[-(])?(?P<number>[\d,.]+)%\)?")

_BARE_NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def _signed_number(match: re.Match[str]) -> float:
    number = float(match.group("number").replace(",", ""))
    return -number if match.group("sign") else number


def _parse_number(text: str) -> float | None:
    match = _BARE_NUMBER_RE.search(text)
    if match is None:
        return None
    return float(match.group(0).replace(",", ""))


def parse_bare_number(text: str) -> float | None:
    """The first plain number in `text` (commas stripped), or None. Public
    for reuse by other parse paths (DS-W3-5's XLSX cells that hold a numeric
    string with no inline scale marker) that need the same fallback numeric
    parse this module uses once `normalize_financial_token` finds nothing."""
    return _parse_number(text)


def normalize_financial_token(text: str) -> tuple[float, float, str | None] | None:
    """Port of the MVP's normalizeFinancialTokens, narrowed to the inline case
    this ticket owns: given a value token that states its own scale -- a
    K/M/B-family suffix ("$4.8M", "500K", "4.8 million"), accounting negative
    notation ("-$4.8M", "($4.8M)"), or a percent sign ("27.3%") -- return
    (base_number, multiplier, unit).

    Returns None when the token carries no inline scale marker (a bare
    "$15,295" or "3,817"): the multiplier is not "1", it is *unknown until
    resolved from context*, and the caller must fall through to the
    column/page header lookup (or assumed_1x) rather than treating "no
    suffix" as an implicit multiplier of 1.
    """
    percent = _PERCENT_RE.search(text)
    if percent is not None:
        return _signed_number(percent), 1.0, "%"

    for pattern, multiplier in _SUFFIX_PATTERNS:
        match = pattern.search(text)
        if match is not None:
            return _signed_number(match), multiplier, None

    return None


# --------------------------------------------------------------------------- #
# Steps 2 & 3: column-header / page-header scale phrases.
# --------------------------------------------------------------------------- #

_SCALE_WORD_MULTIPLIERS: dict[str, float] = {
    "thousand": 1_000.0,
    "million": 1_000_000.0,
    "billion": 1_000_000_000.0,
}

# "CAD (in Thousands)", "(in millions)": an optional leading ISO-style
# currency code, then the "(in <word>)" scale phrase itself. Case-sensitivity
# is scoped with (?i:...) to the phrase only -- if it covered the whole
# pattern, a lowercase 3-letter word ("the (in thousands)") could be
# misread as a currency code.
_SCALE_PHRASE_RE = re.compile(
    r"(?:(?P<currency>[A-Z]{3})\s+)?"
    r"(?i:\(\s*in\s+(?P<word>thousands?|millions?|billions?)\s*\))"
)

# "(000s)" / "(000's)": always thousands, no currency slot of its own.
_PAREN_000_RE = re.compile(r"\(\s*000(?:'s|s)?\s*\)", re.IGNORECASE)

_ScalePhraseMatch = tuple[int, int, float, str | None]


def _scale_word_multiplier(word: str) -> float:
    return _SCALE_WORD_MULTIPLIERS[word.lower().rstrip("s")]


def _find_scale_phrases(text: str) -> list[_ScalePhraseMatch]:
    """Every scale phrase in `text` as (start, end, multiplier, currency),
    in source order. `end` never includes trailing whitespace -- both
    patterns close on a literal ")" -- so a phrase that renders with a
    trailing space in the source ("CAD (in Thousands) ") still yields the
    stripped form "CAD (in Thousands)" with no extra .strip() needed at the
    regex boundary itself; callers still strip defensively before use."""
    matches: list[_ScalePhraseMatch] = [
        (m.start(), m.end(), _scale_word_multiplier(m.group("word")), m.group("currency"))
        for m in _SCALE_PHRASE_RE.finditer(text)
    ]
    matches += [(m.start(), m.end(), 1_000.0, None) for m in _PAREN_000_RE.finditer(text)]
    matches.sort(key=lambda t: t[0])
    return matches


def scale_phrase_in_text(text: str) -> tuple[float, str | None, str] | None:
    """The rightmost (nearest/most-recent) scale phrase in `text`, or None.

    Public on purpose: DS-W3-5's XLSX path reuses this same phrase grammar
    for its own column/sheet-header scan ("Scale headers -- same capture as
    DS-4") without needing a PDF PageIndex -- XLSX header text is scanned
    cell-by-cell rather than as one flat positioned string.
    """
    phrases = _find_scale_phrases(text)
    if not phrases:
        return None
    start, end, multiplier, currency = phrases[-1]
    return multiplier, currency, text[start:end].strip()


def _column_header_scale(
    table: TableRecord, cell: TableCellRecord
) -> tuple[float, str | None, str] | None:
    """Walk the value cell's own column upward, row by row (DS-W3-2 table
    structure), looking for a scale phrase in a header cell above it. Stops
    at the nearest match -- a column-specific scale phrase takes precedence
    over anything found later at the page level."""
    col_cells = {c.row: c for c in table.cells if c.col == cell.col}
    for row in range(cell.row - 1, -1, -1):
        header_cell = col_cells.get(row)
        if header_cell is None:
            continue
        found = scale_phrase_in_text(header_cell.text_normalized)
        if found is not None:
            return found
    return None


def _page_header_scale(page_text: str, before: int) -> tuple[float, str | None, str] | None:
    """The nearest scale phrase anywhere in `page_text` strictly before
    `before` (the value's char_start). "Nearest" because a later scale
    declaration earlier on the same page supersedes an even-earlier one
    (e.g. a document that switches from thousands to millions partway down)."""
    candidates = [m for m in _find_scale_phrases(page_text) if m[1] <= before]
    if not candidates:
        return None
    start, end, multiplier, currency = max(candidates, key=lambda m: m[1])
    return multiplier, currency, page_text[start:end].strip()


# --------------------------------------------------------------------------- #
# Entry point.
# --------------------------------------------------------------------------- #


def determine_scale(
    raw: str,
    page: PageIndex,
    char_start: int,
    *,
    table: TableRecord | None = None,
    cell: TableCellRecord | None = None,
) -> ScaleResult:
    """Resolve one numeric fact's scale, per the module's resolution order.

    `char_start` is the value's position in `page.text` (from the DS-W3-3
    resolved Span) -- the page-header search only looks at text strictly
    before it, matching the ticket's "page-level scale phrase before the
    value". `table`/`cell` are the DS-W3-2 table record and the value's own
    cell, when the value came from a table; omit both for prose values, which
    skip straight from the inline check to the page-level one.
    """
    inline = normalize_financial_token(raw)
    if inline is not None:
        number, multiplier, unit = inline
        return ScaleResult(
            raw=raw,
            normalized=number * multiplier,
            unit=unit,
            scale_multiplier=multiplier,
            scale_source="explicit_in_value",
        )

    number = _parse_number(raw)
    if number is None:
        raise ValueError(f"raw value {raw!r} has no numeric content to scale")

    if table is not None and cell is not None:
        column_match = _column_header_scale(table, cell)
        if column_match is not None:
            multiplier, currency, context = column_match
            return ScaleResult(
                raw=raw,
                normalized=number * multiplier,
                unit=currency,
                scale_multiplier=multiplier,
                scale_source="column_header",
                scale_context=context,
            )

    page_match = _page_header_scale(page.text, char_start)
    if page_match is not None:
        multiplier, currency, context = page_match
        return ScaleResult(
            raw=raw,
            normalized=number * multiplier,
            unit=currency,
            scale_multiplier=multiplier,
            scale_source="page_header",
            scale_context=context,
        )

    return ScaleResult(
        raw=raw,
        normalized=number * 1.0,
        unit=None,
        scale_multiplier=1.0,
        scale_source="assumed_1x",
        flags=["scale_assumed"],
    )

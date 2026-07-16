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

Only a *currency* figure's magnitude is declared by a "(in thousands)"
header. A percent, ratio, count, or date is read at face value -- a headcount
of 1,200 on a "CAD (in Thousands)" income statement is 1,200 people, not
1,200,000. determine_scale therefore takes the fact's value_type and applies
the header lookup to currency alone; every other type self-scales at a known
1.0. Forcing a non-currency value through the header lookup is exactly the
silent 1000x error this module exists to prevent.

Resolution order for a currency value (first match wins):
  1. Inline in the value itself ("$4.8M", "500K") -> explicit_in_value.
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

# Mirrors the value.value_type enum in contracts/facts.schema.json. Kept here
# with its sole current consumer; promote to schemas.py once the extractor and
# the fact emitter share it.
ValueType = Literal["currency", "percent", "count", "ratio", "date", "text"]


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
# three suffix regexes and the percent regex below are translations of that
# function's replace() patterns. The MVP version returned a mutated prose
# string (it fed TF-IDF tokenization, which DS-3 replaces); this port keeps
# only the recognition grammar and returns a structured (number, multiplier,
# unit) instead, since DS-4 needs a multiplier, not a token.
# --------------------------------------------------------------------------- #

# Each pattern: an optional sign ("-" or an opening "(" for accounting
# negative notation), an optional "$", the number, optional whitespace, the
# suffix letter, an optional spelled-out/abbreviated continuation, a word
# boundary, and an optional closing ")" pairing with the opening one.
#
# The continuation group accepts the spelled-out form ("illion"), the doubled
# letter ("MM"/"BB", the accounting shorthand for millions/billions), and the
# "n" abbreviation ("Mn"/"Bn"). The MVP's port stopped at the first two, which
# dropped "$3Bn"/"$5Mn" -- a common CIM notation. That is a real recall gap,
# not a precision risk: no non-magnitude token ends a number with "bn"/"mn",
# so the abbreviation is included here.
_MILLION_RE = re.compile(r"(?P<sign>[-(])?\$?(?P<number>[\d,.]+)\s*[Mm](?:illion|M|m|n|N)?\b\)?")
_BILLION_RE = re.compile(r"(?P<sign>[-(])?\$?(?P<number>[\d,.]+)\s*[Bb](?:illion|B|b|n|N)?\b\)?")
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

# A bare number with optional sign. The sign group matches a leading "-" or an
# accounting-negative "(" so a bracketed negative ("($15,295)") is not read as
# positive; a leading "$" is stripped. Used for values that carry no inline
# scale marker, once the multiplier is settled from context.
_SIGNED_NUMBER_RE = re.compile(r"(?P<sign>[-(])?\$?(?P<number>\d[\d,]*(?:\.\d+)?)")


def _signed_number(match: re.Match[str]) -> float:
    number = float(match.group("number").replace(",", ""))
    return -number if match.group("sign") else number


def _parse_number(text: str) -> float | None:
    match = _SIGNED_NUMBER_RE.search(text)
    if match is None:
        return None
    return _signed_number(match)


def normalize_financial_token(text: str) -> tuple[float, float, str | None] | None:
    """Port of the MVP's normalizeFinancialTokens, narrowed to the inline case
    this ticket owns: given a value token that states its own scale -- a
    K/M/B-family suffix ("$4.8M", "500K", "4.8 million", "$3Bn"), accounting
    negative notation ("-$4.8M", "($4.8M)"), or a percent sign ("27.3%") --
    return (base_number, multiplier, unit).

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
# Self-scaling value types: percent / ratio / count / date.
# --------------------------------------------------------------------------- #

# Unit carried by a self-scaling value read at face value. A percent keeps
# "%"; a ratio is the contract's "ratio" unit; a count or date carries no unit
# from scale capture (a currency unit, when present, comes from a header, and
# these types never take one).
_SELF_SCALING_UNIT: dict[ValueType, str | None] = {
    "percent": "%",
    "ratio": "ratio",
    "count": None,
    "date": None,
}


def _self_scaling(raw: str, value_type: ValueType) -> ScaleResult:
    """A percent, ratio, count, or date is read at face value: its magnitude
    is not declared by a page/column scale header, so the multiplier is a
    KNOWN 1.0. scale_source is explicit_in_value, not assumed_1x -- there is
    nothing to assume and nothing to flag; the value's own type settles it.

    (Known limitation: a count genuinely printed "in thousands", e.g. shares
    outstanding, is read at face value here rather than scaled. That fails
    safe -- it never 1000x-inflates a headcount -- and is revisited if the
    corpus shows a scaled-count column.)
    """
    number = _parse_number(raw)
    if number is None:
        raise ValueError(f"raw value {raw!r} has no numeric content to scale")
    return ScaleResult(
        raw=raw,
        normalized=number,
        unit=_SELF_SCALING_UNIT[value_type],
        scale_multiplier=1.0,
        scale_source="explicit_in_value",
    )


# --------------------------------------------------------------------------- #
# Steps 2 & 3: column-header / page-header scale phrases (currency only).
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
    """The rightmost (nearest/most-recent) scale phrase in `text` as
    (multiplier, currency, context), or None.

    Public on purpose: the downstream parse paths that share this module's
    scale grammar -- the XLSX header scan (DS-W3-5) and table/chart element
    processing (DS-W3-6) -- reuse it directly rather than re-deriving the
    phrase lookup, so a scale phrase is recognized identically no matter which
    path is scanning. "Rightmost" mirrors _page_header_scale's nearest-wins
    rule: a later declaration in the same text supersedes an earlier one.
    """
    phrases = _find_scale_phrases(text)
    if not phrases:
        return None
    start, end, multiplier, currency = phrases[-1]
    return multiplier, currency, text[start:end].strip()


def _cell_covering_column(table: TableRecord, row: int, col: int) -> TableCellRecord | None:
    """The cell in `row` whose column span covers `col`. Honors col_span so a
    merged banner ("CAD (in Thousands)" spanning several value columns, stored
    at its start column) is found from any column it covers -- not only its
    start column, which the exact-column match would miss."""
    for cell in table.cells:
        if cell.row == row and cell.col <= col < cell.col + cell.col_span:
            return cell
    return None


def _column_header_scale(
    table: TableRecord, cell: TableCellRecord
) -> tuple[float, str | None, str] | None:
    """Walk the value cell's own column upward, row by row (DS-W3-2 table
    structure), looking for a scale phrase in a header cell above it. Stops
    at the nearest match -- a column-specific scale phrase takes precedence
    over anything found later at the page level."""
    for row in range(cell.row - 1, -1, -1):
        header_cell = _cell_covering_column(table, row, cell.col)
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
    value_type: ValueType,
    table: TableRecord | None = None,
    cell: TableCellRecord | None = None,
) -> ScaleResult:
    """Resolve one numeric fact's scale, per the module's resolution order.

    `value_type` (the C3 contract's value.value_type) gates the header lookup:
    only "currency" is scaled by a column/page "(in thousands)" header; every
    other numeric type self-scales at a known 1.0, and "text" has no magnitude
    to scale. It is required, with no default, so a caller can never silently
    fall into currency scaling for a value that is not currency.

    `char_start` is the value's position in `page.text` (from the DS-W3-3
    resolved Span) -- the page-header search only looks at text strictly
    before it, matching the ticket's "page-level scale phrase before the
    value". `table`/`cell` are the DS-W3-2 table record and the value's own
    cell, when the value came from a table; omit both for prose values, which
    skip straight from the inline check to the page-level one.
    """
    if value_type == "text":
        raise ValueError("determine_scale does not apply to value_type='text' (no magnitude)")
    if value_type != "currency":
        return _self_scaling(raw, value_type)

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

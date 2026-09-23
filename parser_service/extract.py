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
  - map a period to a fiscal-year END-MONTH convention: resolve_period reads
    (year, A/E/P) straight off the column-header suffix, which is document-
    local and unambiguous on its own (SIM-345), but a stated "fiscal year
    ending <month>" -- needed only to compare across documents with different
    FY ends -- is a separate, still-deferred field

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

import re
from collections import Counter

from .emit import Claim, FlagLog, PeriodKind, emit_pdf_table_cell_claim
from .scale import ValueType, has_parseable_magnitude
from .schemas import PageIndex, TableCellRecord, TableRecord

_LABEL_COL = 0


def _is_a_figure(text: str) -> bool:
    """Whether a cell holds a magnitude, asked of the parser that will read it.

    Gates claim emission, so it must agree with scale.py or the disagreement
    lands on determine_scale as a raise. It used to be
    `any(ch.isdigit() for ch in text)`, and str.isdigit() is true for
    superscripts and circled digits where `\\d` is not: a cell reading "m2" with
    a superscript two passed, was typed currency by the default below, and
    reached determine_scale -- which raises on it, uncaught. The docstring on
    infer_value_type_for claimed that was unreachable. It was reachable; the two
    functions simply disagreed about what a digit is.
    """
    return has_parseable_magnitude(text)


def _carries_cell_content(text: str) -> bool:
    """Whether a row is a data row rather than a section banner.

    Deliberately NOT _is_a_figure. A banner is a label row with nothing in its
    value columns; "Floor area | m2" has something there, so it is a data row
    whose value happens not to be a magnitude. Reading it as a banner makes it
    govern every row beneath it, and "Revenue" two rows down comes back as
    "Floor area | Revenue | 2019F".

    So the looser digit test is right here and wrong at the claim gate: this
    question is about layout, and that one is about whether scale.py can read
    the value.
    """
    return any(ch.isdigit() for ch in text)


def _cell_at(table: TableRecord, row: int, col: int) -> TableCellRecord | None:
    return next((c for c in table.cells if c.row == row and c.col == col), None)


def _covering_cell_at(table: TableRecord, row: int, col: int) -> TableCellRecord | None:
    """The cell in `row` whose column span covers `col`, honouring col_span so a
    merged header label is found from every column it spans, not only its start."""
    return next(
        (c for c in table.cells if c.row == row and c.col <= col < c.col + c.col_span), None
    )


def _header_rows(table: TableRecord) -> list[int]:
    """The header BLOCK: the inferred header row and any continuation rows below it."""
    if table.header_row is None:
        return []
    return [table.header_row, *table.header_continuation]


def _infer_label_column(table: TableRecord, header_block: set[int]) -> int:
    """The column that holds the row labels.

    Column 0 is the label column in the normal case and is returned whenever it
    carries content in a majority of the data rows -- so for an ordinary table
    this is byte-identical to the old hardcoded _LABEL_COL=0.

    It only shifts right when column 0 is a near-empty LEADING SPACER, which
    Docling routinely emits for indented line items. With a hardcoded label
    column that spacer is read as the label (blank -> the row falls back to its
    banner) AND the real label column beside it is treated as a value column, so
    a stray digit sitting in a label cell (a footnote marker, a reference, a bare
    count) is emitted as a figure bound to the wrong metric -- the "grabbed a
    stray table number" failure. When column 0 is a spacer this returns the
    leftmost column that is majority TEXT (label-like, not figures) instead.

    It never shifts onto a genuine value column: a value column is majority
    figures, so it fails the text test, and if there is no text label column at
    all (a spacer followed only by value columns) it falls back to column 0 --
    exactly the old behaviour, so recall never regresses relative to today."""
    data_rows = {c.row for c in table.cells if c.row not in header_block}
    if not data_rows:
        return _LABEL_COL
    threshold = len(data_rows) // 2 + 1  # a STRICT majority of the data rows

    def _filled(col: int) -> int:
        return sum(
            1
            for row in data_rows
            if (c := _cell_at(table, row, col)) is not None and c.text_normalized.strip()
        )

    def _texty(col: int) -> int:
        # Non-empty AND not a parseable figure -- a label, not a value. Uses the
        # same _is_a_figure gate emission uses, so "text" here means exactly
        # "would not be emitted as a value".
        return sum(
            1
            for row in data_rows
            if (c := _cell_at(table, row, col)) is not None
            and c.text_normalized.strip()
            and not _is_a_figure(c.text_normalized.strip())
        )

    cols = sorted({c.col for c in table.cells})
    if not cols:
        return _LABEL_COL
    # Normal case: column 0 carries labels in most rows -> it IS the label column.
    if _filled(cols[0]) >= threshold:
        return cols[0]
    # Column 0 is a near-empty leading spacer: pick the leftmost majority-text
    # column (a real label column), never a value column. If none exists, keep 0.
    for col in cols[1:]:
        if _texty(col) >= threshold:
            return col
    return _LABEL_COL


def _column_header(table: TableRecord, col: int) -> str:
    """This column's label: the header block's cells stacked top to bottom
    ("Hotel" + "Rooms" -> "Hotel Rooms"), col_span honoured so a merged label
    reaches every column it spans. Empty when the table has no header."""
    parts: list[str] = []
    seen: set[int] = set()
    for row in _header_rows(table):
        cell = _covering_cell_at(table, row, col)
        if cell is None:
            continue
        text = cell.text_normalized.strip()
        if text and id(cell) not in seen:
            seen.add(id(cell))
            parts.append(text)
    return " ".join(parts)


# An optional "FY" marker, stripped separately from the year/kind match below so
# a bare two-digit year ("19") -- too easy to collide with an unrelated count or
# footnote number -- is only trusted with that marker in front of it. "FY2019E"
# and "FY 2019E" both strip to "2019E" here; a plain "2019E" never had a prefix
# to strip.
_FY_PREFIX = re.compile(r"^\s*FY\s*", re.IGNORECASE)

# year (2 digit, only trustworthy once FY has been stripped by the caller, or 4
# digit standing alone) + an optional actual/estimate/projected suffix letter +
# an optional trailing footnote marker ("2008E (1)"). Anchored to the WHOLE
# (post-FY-strip) string: this is asked only of a column header, never a row
# label, so "2019F" resolves and a row label that happens to end in a year does
# not silently gain a period.
_PERIOD_RE = re.compile(
    r"^(?P<year>(?:19|20)\d{2}|\d{2})(?P<kind>[AEFP])?\s*(?:\(\d+\))?$",
    re.IGNORECASE,
)

# F and P both spell a forward-looking figure ("2019F", "2019P") -- the ticket's
# landed decision (SIM-345) reads them as the same period_kind. LTM/TTM name a
# trailing period that has no slot in the contract's A/E/P enum (a Postgres
# String(1) CHECK constraint on the backend) and are deliberately left
# unresolved rather than guessing one; see resolve_period's docstring.
_PERIOD_KIND_BY_LETTER: dict[str, PeriodKind] = {"A": "A", "E": "E", "F": "P", "P": "P"}

# A "fiscal" marker: a filed statement often heads its columns "Fiscal 2026" /
# "Fiscal year 2026" rather than a bare year. Word-bounded so it never fires on
# an unrelated label, and only trusted alongside a four-digit year (below).
_FISCAL_MARK = re.compile(r"\bfiscal\b", re.IGNORECASE)

# A trailing-twelve-month / interim window names a period the A/E/P contract has
# no slot for (SIM-345 deferral) -- it must stay unresolved even when it carries
# a month + year (e.g. "LTM September 2020"), or it would be corroborated against
# EDGAR's ANNUAL fact for that calendar year, which the LTM window is not.
_LTM_MARK = re.compile(r"\b(?:LTM|TTM)\b", re.IGNORECASE)

# A variance / growth column ("... vs ...", "Change", "Growth") names no single
# period, so a month+year inside it must not be read as one.
_DELTA_MARK = re.compile(r"\b(?:vs|versus|change|variance|growth|delta)\b", re.IGNORECASE)


def resolve_period(column_header: str) -> tuple[int | None, PeriodKind | None]:
    """(period_year, period_kind) parsed from a column header -- SIM-345.

    Document-local and unambiguous, so this needs no fiscal-year convention:
    A -> actual, E -> estimate, F/P -> projected. A bare year ("2019") resolves
    the year with no kind, since the header does not qualify it.

    Two header shapes carry a period:
    - A pitch-deck-style bare year, optionally FY-prefixed and A/E/F/P-suffixed
      ("2024E", "FY2019", "2008E (1)") -- read by _PERIOD_RE.
    - A FILED statement's fiscal-year-END DATE ("January 25, 2026", "Year Ended
      September 27, 2025") or "Fiscal 2026" label -- the norm in a 10-K/10-Q,
      where a bare year almost never appears. The period_year is the CALENDAR
      YEAR of the stated end date, which is exactly what the filer's own FY label
      uses (NVIDIA's FY2026 ends Jan 25, 2026; Apple's FY2025 ends Sep 27, 2025)
      and what SEC EDGAR keys its annual facts on -- so a claim stamped this way
      lines up with EDGAR for corroboration. Without it, every 10-K statement
      figure loses its period and can never be corroborated.

    Returns (None, None), never a guess, when the header is not period-shaped at
    all; when it is LTM/TTM (a trailing period this contract has no slot for yet,
    deferred -- see the module docstring and _PERIOD_RE); when it is a
    variance/growth column; or when it is ambiguous (a merged header naming more
    than one distinct year -- decline rather than pick one).
    """
    text = column_header.strip()
    had_fy_prefix = bool(_FY_PREFIX.match(text))
    text = _FY_PREFIX.sub("", text, count=1)

    match = _PERIOD_RE.match(text)
    if match is not None:
        year_text = match.group("year")
        if len(year_text) == 2 and not had_fy_prefix:
            return None, None
        year = int(year_text)
        if len(year_text) == 2:
            year += 2000 if year < 69 else 1900
        kind_letter = match.group("kind")
        kind = _PERIOD_KIND_BY_LETTER.get(kind_letter.upper()) if kind_letter else None
        return year, kind

    # Not a bare year. Only a header that actually names a month (a date column)
    # or says "fiscal" is read as a period, so a "2025 vs 2024 Change" column
    # stays unresolved. kind stays None, as for a bare year: the header carries no
    # A/E/P marker to trust. (Numeric-only dates like "12/31/2024" -- no spelled
    # month -- are deliberately not read here, to avoid pulling a year out of a
    # footnote/ratio; a spelled month is the reliable date signal.)
    if _MONTH_NAME.search(text) or _FISCAL_MARK.search(text):
        # An LTM/TTM window or a variance/growth column names no annual period.
        if _LTM_MARK.search(text) or _DELTA_MARK.search(text):
            return None, None
        # Exactly one distinct four-digit year -> that IS the period_year (the
        # calendar year of the fiscal-year-end date). Two or more (a merged
        # two-date column, "Jan 2026 vs Jan 2025", "... 2026 and 2025") is
        # ambiguous -- decline rather than guess which one, never silently picking
        # the older/newer.
        years = set(_YEAR.findall(text))
        if len(years) == 1:
            return int(next(iter(years))), None

    return None, None


# --------------------------------------------------------------------------- #
# value_type vocabularies.
#
# Every one of these is matched against WORD-SPLIT tokens, never as a substring.
# Substring matching is what the earlier version did, and it silently stripped
# the scale header off real money: "count" is a substring of "ac-count-s", so
# "Accounts payable" typed as a count, DS-W3-4 refused the "($ in millions)"
# header, and a $5.2M payable shipped as 5.2 next to a correctly-scaled
# "Total current assets" a million times larger. "date" hid the same way inside
# "consoli-date-d", and "unit" inside "comm-unit-y".
#
# The lists are ordered by tier below, and the tier order is the whole design:
# a metric noun beats a count noun, because a label naming an amount settles the
# question however many countable things it also mentions.
# --------------------------------------------------------------------------- #

# A metric noun names a measured amount, so its presence anywhere in the label
# makes the value money. Measured against the 852-claim reference set built over
# a real CIM: 0 of 172 count-typed attributes contain one, while 392 of 414
# currency-typed attributes do. That separation is what lets this run ahead of
# the count tier at no cost -- it is why "Member's Equity" stays money rather
# than being read as a headcount of members.
_METRIC_NOUNS = frozenset(
    """
    revenue revenues sales turnover expense expenses cost costs spend spending income
    loss losses profit profits earnings margin margins ebitda ebit noi opex capex
    depreciation amortization amortisation impairment interest tax taxes dividend
    dividends distribution distributions debt borrowings equity capital contribution
    contributions asset assets liability liabilities receivable receivables payable
    payables payroll compensation wages salaries rent insurance marketing
    advertising royalty royalties cash flow flows proceeds investment investments
    purchase price value budget fee fees charge charges goodwill inventory
    inventories equipment premises benefits allowance allowances reserve reserves
    """.split()  # noqa: SIM905 -- a word list reads better than 200 literal lines
)

# "net" only names an amount when it TRAILS, which is the accounting qualifier:
# "Property and equipment, net", "Customer list, net" -- net of depreciation or
# allowance, and money either way. Leading, it qualifies whatever noun follows,
# and that noun is often countable: "Net rooms added", "Net new subscribers",
# "Net units shipped" are counts, and typing them currency would let a
# "($ in millions)" header multiply a room count by a million. Position is the
# whole signal here, so it cannot live in the set above.
_TRAILING_METRIC_NOUN = "net"

# A count noun names something you can count. "property" is deliberately absent:
# "Property and Equipment, Net" is money, and a genuine property count reads
# "number of properties", which the number-of pair below catches instead.
_COUNT_NOUNS = frozenset(
    """
    slot slots machine machines seat seats room rooms suite suites bed beds table
    tables game games employee employees headcount staff fte ftes visitor visitors
    customer customers subscriber subscribers member members guest guests resident
    residents store stores location locations outlet outlets branch branches
    restaurant restaurants shop shops well wells rig rigs vehicle vehicles aircraft
    patent patents contract contracts license licenses position positions space
    spaces unit units attendee attendees convention conventions population admission
    admissions passenger passengers household households
    """.split()  # noqa: SIM905 -- a word list reads better than 200 literal lines
)

# A span of time, counted rather than priced: "ACEP Tenure (In Years)" -> 7 is
# seven years. Kept apart from the count nouns because these same words spell a
# PERIOD CAPTION -- "Year Ended December 31," , "Twelve Months Ended" -- and a
# caption names when a figure was measured, not what it counts. A period caption
# arrives here as a column header appended to the attribute, so treating its
# words as counts silently strips the scale header off every money row beneath
# it: "Total | Year Ended December 31," shipped 2,400 where the truth was
# $2.4bn, beside a correctly-scaled sibling. _is_period_caption settles which
# sense is meant.
_DURATION_NOUNS = frozenset(
    """
    years yrs tenure months days
    """.split()  # noqa: SIM905 -- a word list reads better than 200 literal lines
)

# The words that mark a label as naming a reporting period rather than a span.
_PERIOD_CAPTION_WORDS = frozenset(
    """
    ended ending fiscal quarter interim trailing
    """.split()  # noqa: SIM905 -- a word list reads better than 200 literal lines
)

# A physical extent. Same tier as a count -- a currency scale header never
# applies to it -- but kept separate because these name a dimension rather than
# a countable thing, and the distinction matters to anything reading this list.
_DIMENSION_NOUNS = frozenset(
    """
    footage feet foot acre acres acreage mile miles metre metres meter meters hectare
    hectares square capacity megawatt megawatts ton tons tonne tonnes barrel barrels
    gallon gallons litre litres
    """.split()  # noqa: SIM905 -- a word list reads better than 200 literal lines
)

# A date word in the label licenses reading the value as a period -- but only
# when the value is date-SHAPED as well. The label alone is not enough:
# "Acq. of Flamingo Laughlin, net of cash acquired" -> "(109.4 )" matches
# "acquired" at a genuine word start, so anchoring does not save it, and a
# $109.4M cash outflow would be recorded as a negative year.
_DATE_WORDS = frozenset(
    """
    date dated dates acquired opened opening openings completion completions completed
    expires expiration expirations expiry commenced commencement founded established
    incorporated renovated renovation renovations
    """.split()  # noqa: SIM905 -- a word list reads better than 200 literal lines
)

_CAMEL_BOUNDARY = re.compile(r"([a-z0-9])([A-Z])")
_NON_ALPHA = re.compile(r"[^A-Za-z]+")

_CURRENCY_MARK = re.compile(
    r"US\$|C\$|A\$|NZ\$|HK\$|[$£€¥]|\b(?:USD|CAD|AUD|EUR|GBP|JPY|CHF)\b|\bdollars?\b",
    re.IGNORECASE,
)
_PERCENT_MARK = re.compile(r"%|\b\d+(?:\.\d+)?\s*(?:bps|basis\s+points?)\b", re.IGNORECASE)
# An EBITDA multiple, leverage or coverage figure: "7.5x", "4.2 x".
_RATIO_MARK = re.compile(r"\b\d+(?:\.\d+)?\s*x\b", re.IGNORECASE)
_MONTH_NAME = re.compile(
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)"
    r"(?:uary|ruary|ch|il|e|y|ust|tember|ober|ember)?\b",
    re.IGNORECASE,
)
_APOSTROPHE_YEAR = re.compile(r"'\d{2}\b")
_YEAR = re.compile(r"(?:19|20)\d{2}")
# A unit printed inside the value itself ("1,300 square feet", "6-seat"). More
# reliable than any label, because the document is naming the unit outright.
_UNIT_IN_VALUE = re.compile(
    r"\b(?:square\s+(?:feet|foot|footage|metres?|meters?)|sq\.?\s?ft\.?|"
    r"acres?|miles?|seats?|rooms?|suites?|slots?|machines?|table\s+games?|"
    r"beds?|units?|years?|yrs?|months?|days?)\b",
    re.IGNORECASE,
)
# A struck-out or not-meaningful cell. Carries no magnitude to scale.
_SENTINEL = re.compile(r"\s*(?:N/?A|NM|[-–—]+)\s*", re.IGNORECASE)

# Beyond this many words, the cell holds a sentence rather than a value.
_PROSE_WORD_COUNT = 4


def _label_tokens(attribute: str) -> list[str]:
    """The attribute's words in order, lowercased, with camelCase split apart.

    Reads the WHOLE attribute -- row label and column header both -- because
    which axis names the metric varies by table. A financial statement puts it
    in the row ("Cash and cash equivalents | 2003"); a property summary puts it
    in the column ("Stratosphere | Slot Machines"). Reading only the row label
    would miss every count in the second shape.

    Order is preserved because one word's meaning depends on it: see
    _TRAILING_METRIC_NOUN.
    """
    expanded = _CAMEL_BOUNDARY.sub(r"\1 \2", attribute)
    return [token for token in _NON_ALPHA.split(expanded.lower()) if token]


def _names_an_amount(tokens: list[str]) -> bool:
    """Whether the label names a measured amount rather than a countable thing."""
    if set(tokens) & _METRIC_NOUNS:
        return True
    # "net" counts only when it trails: "Customer list, net" is money, while
    # "Net rooms added" is a room count.
    return _TRAILING_METRIC_NOUN in tokens[1:]


def _is_period_caption(vocabulary: set[str]) -> bool:
    """Whether the label names a reporting period ("Year Ended December 31,")
    rather than a span of time ("Tenure (In Years)").

    A period caption is a column header, so it is appended to every attribute in
    its table. Reading its "Year"/"Months" as a count would type every money row
    under it as a count, and a count never takes the scale header.
    """
    return bool(vocabulary & _PERIOD_CAPTION_WORDS)


def _is_date_shaped(raw: str) -> bool:
    """Whether the value itself could be a date: a month name, an apostrophe
    year ("'07"), or a four-digit year anywhere in it."""
    return bool(_MONTH_NAME.search(raw) or _APOSTROPHE_YEAR.search(raw) or _YEAR.search(raw))


def infer_value_type_for(raw: str, attribute: str) -> ValueType:
    """The value_type for a cell, judged from the value and its attribute.

    This gates DS-W3-4: only `currency` is multiplied by the table's
    "($ in millions)" scale header, so a mistype is a 1000x error in one
    direction or the other. The two directions are NOT equally dangerous, and
    that asymmetry sets the default at the bottom of this function. A value
    wrongly typed currency fails visibly -- it ends up assumed_1x carrying a
    `scale_assumed` flag, or takes a header multiplier and an `ambiguous_unit`
    flag. A value wrongly typed count or date fails SILENTLY: scale.py's
    _self_scaling returns a KNOWN 1.0 with scale_source="not_applicable" and
    no flag at all, which is indistinguishable from a verified scale. So where
    the signals run out, currency is the answer whose failure can be audited.

    Rules are ordered and first match wins; the order is the design:

      0. No digit, or a sentinel ("-", "N/A") -> text. emit.py gives `text` a
         null normalized, so nothing is invented from a value that has no
         magnitude -- and determine_scale is never reached with an unparseable
         value, which today raises ValueError uncaught.
      1. The value declares its own unit ("%", "7.5x", "$") -> that unit wins.
         A "%" is a complete scale statement and must never take a header.
      2. Four or more words is a sentence, not a value -> text, so no number is
         invented from a fragment whose leftmost digits happen to parse.
      3. A unit inside the value ("1,300 square feet") outranks any label.
      4. A date word in the label, but ONLY with a date-shaped value.
      5. A metric noun anywhere in the label -> currency, ahead of the count
         tier. This is what keeps "Member's Equity" money.
      6. A count or dimension noun -> count.
      7. A bare four-digit year -> date.
      8. Otherwise currency, per the asymmetry above.
    """
    tokens = _label_tokens(attribute)
    vocabulary = set(tokens)
    stripped = raw.strip()

    if not any(character.isdigit() for character in stripped):
        return "text"
    if _SENTINEL.fullmatch(stripped):
        return "text"

    if _PERCENT_MARK.search(stripped):
        return "percent"
    if _RATIO_MARK.search(stripped):
        return "ratio"
    if _CURRENCY_MARK.search(stripped):
        return "currency"

    words = [word for word in _NON_ALPHA.split(stripped) if len(word) > 1]
    if len(words) >= _PROSE_WORD_COUNT:
        return "text"

    if _UNIT_IN_VALUE.search(stripped):
        return "count"

    if vocabulary & _DATE_WORDS and _is_date_shaped(stripped):
        return "date"
    if _MONTH_NAME.search(stripped):
        return "date"

    if _names_an_amount(tokens):
        return "currency"
    if vocabulary & _COUNT_NOUNS or vocabulary & _DIMENSION_NOUNS:
        return "count"
    if vocabulary & _DURATION_NOUNS and not _is_period_caption(vocabulary):
        return "count"
    if "number" in vocabulary and "of" in vocabulary:
        return "count"

    if _YEAR.fullmatch(stripped):
        return "date"

    return "currency"


def is_confident_currency(raw: str, attribute: str) -> bool:
    """Whether a currency value earned that type by a POSITIVE signal rather than
    by infer_value_type_for's fallthrough default (its last line).

    That default is defended by the claim that a wrong currency fails VISIBLY --
    it ends up assumed_1x carrying a scale_assumed flag. The defense holds only
    while no banner applies: a page-level "(in millions)" banner turns the default
    into a SILENT 1000x, which is how a whole property-summary table of slot / room
    / square-foot COUNTS came to be stored a million-fold too large from a financial
    banner bleeding across it (SIM-323). So the table caller passes this to
    determine_scale(page_header_ok=...): a value confident by an inline currency
    mark or a metric-noun label may bind a page banner; everything else declines it
    and falls to a flagged assumed_1x. The two positive signals mirror the two
    positive currency branches of infer_value_type_for exactly.
    """
    if _CURRENCY_MARK.search(raw.strip()):
        return True
    return _names_an_amount(_label_tokens(attribute))


def section_banners(table: TableRecord) -> dict[int, str]:
    """Each row's governing in-table section banner, by row index.

    A financial statement groups its rows under banners that are themselves rows:
    a label with no figures beside it.

        row 1   TURNOVER                                      <- banner
        row 4     Coffee Shop        41    92    94
        row 11  COST OF SALES                                 <- banner
        row 14    Coffee Shop        31    71    72

    Those banners carry meaning that nothing else in the table does, and without
    them two of these rows are the same claim. "Coffee Shop | YEAR 1" appears
    twice with different numbers and no way to tell which is revenue and which is
    the cost of earning it.

    They also name the rows the table leaves unlabelled. A section subtotal is
    printed as figures with an empty label cell, so `attribute_for` had nothing to
    call it and the value was dropped -- 431 cells on one CIM, including every
    turnover and cost-of-sales total.

    A row counts as a banner when its label column is filled and no cell beside it
    holds a digit. That is deliberately strict: a row with any figure in it is
    data, whatever it is called.
    """
    rows: dict[int, dict[int, str]] = {}
    for cell in table.cells:
        rows.setdefault(cell.row, {})[cell.col] = cell.text_normalized.strip()

    governing: dict[int, str] = {}
    current: str | None = None
    header_block = set(_header_rows(table))
    for row in sorted(rows):
        if row in header_block:
            continue
        cells = rows[row]
        label = cells.get(_LABEL_COL, "")
        has_figures = any(
            col >= 1 and text and _carries_cell_content(text) for col, text in cells.items()
        )
        if label and not has_figures:
            current = label
            continue
        if current is not None:
            governing[row] = current
    return governing


def attribute_for(
    table: TableRecord,
    cell: TableCellRecord,
    banner: str | None = None,
    *,
    label_col: int = _LABEL_COL,
) -> str | None:
    """The table's own name for this cell:
    "<section banner> | <row label> | <column header>".

    Deliberately the source document's words rather than a product attribute
    (`revenueLatestUsd`). Mapping a CIM's phrasing onto a fixed vocabulary is not
    guessed here -- inventing a canonical name from a string match would be a
    silent, confident error of exactly the kind this codebase refuses elsewhere.
    That mapping is SIM-344's job instead: a downstream proposer-with-code-gate
    pass (parser_service.propose.canonicalize_attributes, wired in by
    extract_service.extract_claims when canonicalize_attributes=True) reads
    whatever this function returns as the raw label, maps it onto the closed
    core-financial-statement enum or the operating_metric bucket, and moves it
    into Claim.attribute_raw while Claim.attribute becomes the canonical value.
    This function's job stays exactly what it always was: produce the document's
    own words, honestly and without inventing structure the table doesn't have.

    The banner is included because without it the name is not unique inside its
    own table: "Coffee Shop | YEAR 1" is both a revenue line and a cost line on
    the same page. It also makes the name true to what the document says, which
    is what lets value typing get these right -- a row under TURNOVER is money
    however countable its own noun sounds.

    Returns None when the row has no label AND no banner governs it, because a
    value nothing in the table can name has no attribute to claim.
    """
    label_cell = _cell_at(table, cell.row, label_col)
    label = label_cell.text_normalized.strip() if label_cell else ""
    if not label:
        # An unlabelled row under a banner is that section's own line -- a
        # subtotal, in every case seen so far. The banner is the table's word
        # for it, so it is a name the document supplies rather than one invented
        # here.
        if not banner:
            return None
        label = banner
        banner = None

    # header_row is None only when the columns are genuinely unlabeled; the row
    # label alone is then the whole attribute -- weaker, but honest.
    #
    # Deliberately NOT gated on column_headers_reliable. That flag reports
    # whether Docling's per-cell column_header markers agree with the structural
    # inference -- a diagnostic about Docling, not a verdict on header_row.
    # Gating on it discarded the column header on every financial statement in a
    # CIM, because Docling's markers disagree there. The cost was severe: all
    # four years of "Cash and cash equivalents" collapsed onto one attribute,
    # four rows with the same name and different numbers, none of them usable.
    # DS-W3-2's _infer_header_row now validates the row structurally -- it must
    # span the value columns and read as labels or periods -- and that inference
    # is the signal worth trusting.
    parts = [banner, label] if banner else [label]

    # The column label is the whole header BLOCK stacked, not one cell: a header
    # that wrapped onto a second line ("Hotel"/"Rooms") keeps both, so the count
    # noun on the continuation line survives into the attribute and its column of
    # figures types as a count rather than defaulting to currency.
    column_header = _column_header(table, cell.col)
    if not column_header:
        return " | ".join(parts)
    return " | ".join([*parts, column_header])


# --------------------------------------------------------------------------- #
# Column ROLE: a primary figure vs a derived/auxiliary RE-EXPRESSION of one.
#
# Beside a metric's primary figure a financial table often prints columns that
# only re-express it: a percentage OF another column ("% of revenue"), a
# period-over-period CHANGE in it ("$ change", "Increase / (Decrease)", "YoY %"),
# a growth rate or a variance. Each such cell passes _is_a_figure, so without a
# role check every one is emitted as its own claim under the same row label. Once
# canonicalize_attributes (propose.py) strips the period/scale qualifier, a
# change or percentage cell can canonicalize onto the SAME metric as the primary
# column beside it and be surfaced in its place -- the Total-Liabilities-514M
# class, where a component/change figure stands in for the true total.
#
# The classifier keys on the column HEADER, never on counting value columns:
# table_extract._value_columns needs >=2 figures per row in >=2 rows, so it drops
# single-value-column tables outright -- a recall loss the follow-up brief
# forbids. A header-semantic test instead skips a column only on a POSITIVE
# derived signal, and only when a primary column survives to carry the metric
# (the guard in _value_column_roles). A single unlabeled or plainly-labelled
# value column is therefore always kept, and a table whose columns are ALL
# percentages is its own data and is kept whole.
#
# Per-share is deliberately NOT derived: EPS and other per-share figures are
# first-class facts a consumer wants, and canonicalize_attributes already keeps
# "<metric> per share" distinct from "<metric>" (it strips only period/basis/
# scale qualifiers, never "per share"), so per-share never collapses onto the
# total the way a bare "% of" or "change" column does.
#
# The verbs are anchored against the ONE recall risk a header-semantic filter
# carries: a transposed/matrix CIM disclosure where a genuine primary metric
# heads a column. "change"/"difference" are matched only when NOT continued by
# "in <noun>", so "Change in fair value" and "Net change in cash" survive while a
# bare "Change" / "% Change" / "Change (%)" column is dropped; "increase"/
# "decrease" only as the standalone PAIR ("Increase / (Decrease)") and again not
# when it runs into "in <noun>", so "Net increase (decrease) in cash" survives;
# "growth" only as an explicit RATE ("growth rate", "% growth"), so "Growth
# capital" survives. Word boundaries already spare "Exchange rate",
# "Interchange fees" and the like.
# The "not continued by in <noun>" guard tolerates an intervening ")" and
# whitespace, so it still spares "Net increase (decrease) in cash" where a close
# paren sits between the verb and "in".
_NOT_INTO_NOUN = r"(?![\s)]*\bin\b)"
_DERIVED_COLUMN_RE = re.compile(
    rf"""
      \bchange\b{_NOT_INTO_NOUN} | \bchg\b | \bvariance\b | \bdifference\b{_NOT_INTO_NOUN}
    | \bincrease\b[\s/()]*\bdecrease\b{_NOT_INTO_NOUN}
    | \byoy\b | \bqoq\b | \by\s*/\s*y\b | \by-o-y\b
    | \bcagr\b | \bgrowth\s+rate\b | \bgrowth\s*% | %\s*growth\b
    | \bperiod[\s-]to[\s-]period\b
    | %\s*of\b | \bpercent(?:age)?\s+of\b
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _is_derived_column(table: TableRecord, col: int) -> bool:
    """Whether `col` re-expresses another column (a %/change/growth OF it) rather
    than carrying a primary figure, judged from its stacked column header."""
    header = _column_header(table, col)
    return bool(header and _DERIVED_COLUMN_RE.search(header))


def _value_column_roles(
    table: TableRecord, label_col: int, header_block: set[int]
) -> tuple[set[int], set[int]]:
    """Partition the table's value columns into (emitted, skipped).

    A value column carries at least one figure in a data row. A column is skipped
    only when its header reads as a derived re-expression AND at least one primary
    (non-derived) value column survives to carry the metric. When every value
    column is derived-headed they ARE the table's data, so none is skipped -- the
    recall guard that keeps a percentages-only or single-column table whole."""
    data_rows = {c.row for c in table.cells if c.row not in header_block}
    value_cols = {
        c.col
        for c in table.cells
        if c.col != label_col and c.row in data_rows and _is_a_figure(c.text_normalized.strip())
    }
    derived = {col for col in value_cols if _is_derived_column(table, col)}
    primary = value_cols - derived
    if not primary:
        return value_cols, set()
    return primary, derived


def _unresolved_header_columns(table: TableRecord, emitted_cols: set[int]) -> set[int]:
    """The emitted value columns whose period/column qualifier cannot be trusted
    because the header failed to SEPARATE them.

    The danger a header carries is cross-column collapse: two value columns in one
    table that resolve to the SAME (row label x column header) attribute cannot be
    told apart, so their claims fold onto one metric-period and a consumer picks
    between them blind. That is what produces the impossible orderings when the
    header lands on a title/scale/caption row, or on nothing at all -- three years
    of one line item wearing one indistinguishable name.

    A column is unresolved when the table has two or more emitted value columns
    and this column's header is either SHARED with another emitted column (the two
    collapse onto one attribute) or ABSENT (the value carries no period qualifier
    at all, so its period is untrustworthy even though it does not collapse). A
    lone value column has nothing to collapse against and is always trusted, so
    single-value-column recall is untouched.

    This is a header-STRING signal, not a resolved-period one: two headers that
    differ as text but map to the same period downstream ("2019" vs "FY2019" vs
    "2019 (1)") collapse after canonicalize_attributes strips the qualifier yet
    are not flagged here. That residue is rare and belongs to the period resolver,
    not to this structural check."""
    if len(emitted_cols) < 2:
        return set()
    header_by_col = {col: _column_header(table, col) for col in emitted_cols}
    counts = Counter(header_by_col.values())
    return {col for col, header in header_by_col.items() if not header or counts[header] > 1}


def claims_from_table(
    table: TableRecord,
    page: PageIndex,
    *,
    entity: str,
    file: str,
    flag_log: FlagLog,
    section: str | None = None,
    inherited_scale: tuple[float, str | None, str] | None = None,
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

    period_year/period_kind (SIM-345) are resolved from the cell's own column
    header, independent of whether the value resolves -- a missing claim still
    carries the period it was searched under.

    Rows are read under their in-table section banner (see section_banners), so a
    line keeps the heading the document filed it under. That is what separates
    the two "Coffee Shop" rows on a P&L, and what names the unlabelled subtotal
    rows that were previously dropped for having nothing to call them.

    `inherited_scale` is the nearest preceding page's scale banner, supplied by the
    caller for a table on a page that carries no banner of its own (a statement
    whose "(in millions)" caption sits on the page before the figures, or a table
    that continues onto this page). It is passed to determine_scale as the lowest-
    priority scale source; an own-page column/page header always wins.
    """
    banners = section_banners(table)
    header_block = set(_header_rows(table))
    label_col = _infer_label_column(table, header_block)
    emitted_cols, skipped_cols = _value_column_roles(table, label_col, header_block)
    unresolved_cols = _unresolved_header_columns(table, emitted_cols)
    claims: list[Claim] = []
    for cell in sorted(table.cells, key=lambda c: (c.row, c.col)):
        if cell.col == label_col or cell.row in header_block:
            continue
        # A derived/auxiliary column (a %/change/growth re-expression of a primary
        # column) is dropped so it cannot canonicalize onto the metric beside it
        # and be surfaced in place of the true figure.
        if cell.col in skipped_cols:
            continue
        raw = cell.text_normalized.strip()
        if not raw or not _is_a_figure(raw):
            continue

        attribute = attribute_for(table, cell, banners.get(cell.row), label_col=label_col)
        if attribute is None:
            continue

        period_year, period_kind = resolve_period(_column_header(table, cell.col))
        # A value whose column the header could not tell apart from another is
        # still emitted -- a dropped cell is invisible -- but flagged, so a
        # consumer never silently trusts its (missing or collapsed) period
        # qualifier as if the columns had been cleanly separated.
        extra_flags = ["header_unresolved"] if cell.col in unresolved_cols else None

        claims.append(
            emit_pdf_table_cell_claim(
                entity,
                attribute,
                table,
                cell,
                page,
                value_type=infer_value_type_for(raw, attribute),
                file=file,
                flag_log=flag_log,
                section=section,
                page_header_ok=is_confident_currency(raw, attribute),
                inherited_scale=inherited_scale,
                period_year=period_year,
                period_kind=period_kind,
                extra_flags=extra_flags,
            )
        )
    return claims

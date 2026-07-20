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
  - know a period from a fiscal-year convention (period_year/period_kind are
    left for a later pass; the year, where present, is in the column header
    the attribute string quotes)

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

from .emit import Claim, FlagLog, emit_pdf_table_cell_claim
from .scale import ValueType, has_parseable_magnitude
from .schemas import PageIndex, TableCellRecord, TableRecord

_LABEL_COL = 0


def _is_a_figure(text: str) -> bool:
    """Whether a cell holds a number, asked of the parser that will read it.

    This was `any(ch.isdigit() for ch in text)`, which answers a different
    question than scale.py does: str.isdigit() is true for superscripts and
    circled digits, `\\d` is not. A cell reading "m2" with a superscript two
    therefore passed this gate, was typed currency by the default below, and
    reached determine_scale -- which raised on it, uncaught. The docstring in
    infer_value_type_for claimed that could not happen. It could; the two
    functions simply disagreed about what a digit is.
    """
    return has_parseable_magnitude(text)


def _cell_at(table: TableRecord, row: int, col: int) -> TableCellRecord | None:
    return next((c for c in table.cells if c.row == row and c.col == col), None)


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
    date dated acquired opened opening completion completed expires expiration expiry
    commenced commencement founded established incorporated renovated renovation
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
    for row in sorted(rows):
        if row == table.header_row:
            continue
        cells = rows[row]
        label = cells.get(_LABEL_COL, "")
        has_figures = any(col >= 1 and text and _is_a_figure(text) for col, text in cells.items())
        if label and not has_figures:
            current = label
            continue
        if current is not None:
            governing[row] = current
    return governing


def attribute_for(
    table: TableRecord, cell: TableCellRecord, banner: str | None = None
) -> str | None:
    """The table's own name for this cell:
    "<section banner> | <row label> | <column header>".

    Deliberately the source document's words rather than a product attribute
    (`revenueLatestUsd`). Mapping a CIM's phrasing onto a fixed vocabulary is the
    extractor's real job and is not guessed here -- inventing a canonical name
    from a string match would be a silent, confident error of exactly the kind
    this codebase refuses elsewhere.

    The banner is included because without it the name is not unique inside its
    own table: "Coffee Shop | YEAR 1" is both a revenue line and a cost line on
    the same page. It also makes the name true to what the document says, which
    is what lets value typing get these right -- a row under TURNOVER is money
    however countable its own noun sounds.

    Returns None when the row has no label AND no banner governs it, because a
    value nothing in the table can name has no attribute to claim.
    """
    label_cell = _cell_at(table, cell.row, _LABEL_COL)
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

    if table.header_row is None:
        return " | ".join(parts)

    header_cell = _cell_at(table, table.header_row, cell.col)
    if header_cell is None or not header_cell.text_normalized.strip():
        return " | ".join(parts)
    return " | ".join([*parts, header_cell.text_normalized.strip()])


def claims_from_table(
    table: TableRecord,
    page: PageIndex,
    *,
    entity: str,
    file: str,
    flag_log: FlagLog,
    section: str | None = None,
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

    Rows are read under their in-table section banner (see section_banners), so a
    line keeps the heading the document filed it under. That is what separates
    the two "Coffee Shop" rows on a P&L, and what names the unlabelled subtotal
    rows that were previously dropped for having nothing to call them.
    """
    banners = section_banners(table)
    claims: list[Claim] = []
    for cell in sorted(table.cells, key=lambda c: (c.row, c.col)):
        if cell.col == _LABEL_COL or cell.row == table.header_row:
            continue
        raw = cell.text_normalized.strip()
        if not raw or not _is_a_figure(raw):
            continue

        attribute = attribute_for(table, cell, banners.get(cell.row))
        if attribute is None:
            continue

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
            )
        )
    return claims

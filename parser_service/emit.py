"""DS-W3-7 claim emission -- the JSON seam from parse output to the
deterministic claims spine (contracts/claims.schema.json, the frozen C3
contract).

Given a claim candidate (entity, attribute, and the quote or cell it should
come from) plus the parse-time records it is read against, this module
resolves it through DS-W3-3 (exact-span resolver) and DS-W3-4 / DS-W3-5
(scale capture) into one Claim JSON row. All-or-nothing provenance: a claim
either resolves to an exact span/cell, or is written `status="missing"` with no
span at all -- never a partial citation, and never a zero span standing in for
one.

A claim is an assertion pending verification, not a verified truth. Trust is
carried by `status` plus `verification_method` (how that trust was earned),
never by a confidence score -- a number conflates "the extractor felt good about
this" with "this was checked", and only the second means anything downstream.

What this service can honestly emit:

    PDF prose or table cell -> proposed           (citation attached, not yet checked)
    XLSX literal cell       -> cited/direct_read  (reading the bytes IS the check)
    XLSX formula cell       -> proposed           (cell exact, value pending re-execution)
    nothing resolved        -> missing            (no span; the page is still recorded)

`cited` is reachable at parse time only for a literal XLSX cell. A formula stays
`proposed`: this service never executes formulas, and the workbook's own cached
result is never trusted as a substitute (the exact anti-pattern DS-W3-5
replaces). Re-execution happens outside this service and is what later earns
cited/formula_reexecution -- the emitter must not pre-empt a check it never ran.

Pydantic (`extra="forbid"`, closed Literal enums) is the boundary validation
the ticket calls for: a key-name or enum-value drift raises here, before the
JSON ever crosses the seam.

Every flag raised while emitting a claim is both attached to that claim's
`flags` array AND appended to a `FlagLog` (run_id, stage, element_id,
flag_type, detail) -- the structured per-run capture the ticket asks for,
without building the aggregation/dashboard surface that's explicitly out of
scope for alpha.
"""

from __future__ import annotations

import uuid
from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict, Field

from .elements import ChartElement, TableElement
from .resolver import resolve, resolve_in_cell
from .scale import ScaleSource, ValueOrigin, ValueType, determine_scale, scale_invariant_holds
from .schemas import BBox, PageIndex, TableCellRecord, TableRecord, XlsxCellRecord, XlsxSheetRecord
from .xlsx_parser import determine_xlsx_scale

# The subset of the contract's 8-value status enum this service can reach.
# rejected / conflicted / inconclusive / partially_verified / verified are earned
# by later passes; an extractor asserting them would be claiming a check it never
# ran.
Status = Literal["proposed", "cited", "missing"]

# How citation-level trust was earned. None until something has actually been
# checked -- the contract requires a method once status is cited or later.
VerificationMethod = Literal["direct_read"]

# A=Actual, E=Estimate, P=Projected -- mirrors the contract's period_kind enum
# exactly (contracts/claims.schema.json), which is itself a Postgres
# String(1) CHECK constraint on the backend (app/models/claim.py). There is no
# fourth slot for a trailing period (LTM/TTM): that needs a schema + DB
# migration on the backend side and is out of scope here (SIM-345).
PeriodKind = Literal["A", "E", "P"]

# SIM-364 / FinGround (arXiv:2604.23588): the TYPE of assertion a claim makes,
# orthogonal to the binding-audit verdicts (which are failure modes). Verification
# routes on it -- a `computational` claim to formula reconstruction, a `numerical`
# claim to exact-span, and so on. Assigned once at extraction by the tier best
# placed to know: the table/XLSX tiers set it in code, the prose/qualitative tiers
# via the model. `unknown` is the explicit fallback for an un-typeable claim --
# never defaulted to a wrong type, which would mis-route it downstream.
ClaimType = Literal[
    "numerical",
    "temporal",
    "entity_attribute",
    "comparative",
    "regulatory",
    "computational",
    "unknown",
]
CLAIM_TYPES: frozenset[str] = frozenset(get_args(ClaimType))

# Mirrors contracts/claims.schema.json's flags enum exactly. Kept here (rather
# than re-derived from the schema file at import time) so an unknown flag string
# raises immediately, at the call site that produced it.
FLAG_TYPES = frozenset(
    {
        "ragged_table_rows",
        "chart_data_not_extracted",
        "scale_assumed",
        "quote_unresolved",
        "ambiguous_location",
        "ambiguous_region_bounds",
        "ambiguous_unit",
        "table_touches_page_edge",
        "zero_text_page",
        "empty_section",
        "formula_mismatch",
        "external_reference_unresolved",
        # A claim the extractor typed numeric whose value text held no single
        # readable number, so it was emitted as text. NOT a qualitative claim --
        # without this the two text populations are indistinguishable in the store.
        "magnitude_unparseable",
        # The resolved span lies wholly inside repetition-tagged page furniture.
        "cites_boilerplate",
        # The quote resolved, but the entity or asserting clause the model bound
        # to it is not readable inside it. A precision signal, deliberately not
        # quote_unresolved -- that one means the resolver could not find the text
        # at all, and merging the two makes each unreadable.
        "binding_unsupported",
        # SIM-344: a label the attribute-mapping proposer judged core-adjacent
        # (or produced a string outside the closed enum for) but could not map to
        # one of CORE_ATTRIBUTES. Never guessed past -- see gate_canonical_attribute.
        "attribute_unmapped",
        # SIM-369: the dumb-consumer guard, set by alpha's 3a reconciliation on the
        # non-canonical claim in a same_fact merge so an edge-ignorant reader does not
        # count it as independent corroboration. The parser never writes it
        # (reconciliation is alpha-only), but FLAG_TYPES and the contract's flags enum
        # are kept identical (test_all_flag_types_constant_matches_schema_enum), so it
        # lives here too.
        "superseded_by_same_fact",
    }
)


# --------------------------------------------------------------------------- #
# SIM-344 / E2: attribute vocabulary mapping.
#
# The closed, versioned enum for the financial-statement core -- recurring line
# items that feed consistency (revenue, margin, ...). The ticket calls for
# seeding this from a recall-audit's actual observed attributes; no such
# dataset is checked into this repo, so this list is seeded from the ticket's
# own examples plus the standard recurring CIM financial-statement line items
# instead. Extend it the same way FLAG_TYPES/AssertionClass get extended: from
# an observed gap on a real document, never from imagination.
#
# Everything that is NOT one of these is NOT invented a synthetic core name --
# it lands in the OPERATING_METRIC bucket below, raw label retained via
# Claim.attribute_raw. A fully-closed enum covering sector/operating metrics
# (occupancy, ARPU, same-store sales, ...) would flag-storm; this two-tier
# split is the ticket's landed decision, not a shortcut.
#
# "margin" was originally one bucket for every margin figure. A single core
# fact-slot review found this too coarse: a gross-margin claim and a net-margin
# claim on the same CIM page both canonicalized to "margin", so the extract
# reducer's `contradicts` pass (keyed on page/entity/attribute/value_type) read
# two different real facts as one fact-slot disagreeing with itself. Split into
# gross/net/EBITDA margin so each keeps its own fact-slot identity; the
# reducer needs no change; the attribute-mapping prompt already lists whatever
# is in this enum, so the model sees the split automatically.
# --------------------------------------------------------------------------- #

CoreAttribute = Literal[
    "revenue",
    "cogs",
    "gross_profit",
    "opex",
    "ebitda",
    "ebit",
    "net_income",
    "gross_margin",
    "net_margin",
    "ebitda_margin",
    "depreciation_and_amortization",
    "interest_expense",
    "tax_expense",
    "capex",
    "total_assets",
    "total_liabilities",
    "total_equity",
    "cash_and_equivalents",
    "total_debt",
    "net_debt",
    "working_capital",
    "accounts_receivable",
    "accounts_payable",
    "inventory",
    "operating_cash_flow",
    "free_cash_flow",
]

CORE_ATTRIBUTES: frozenset[str] = frozenset(get_args(CoreAttribute))

# The escape valve for everything outside the closed core -- a sector metric,
# an operating KPI, or any label the core enum was never meant to cover.
OPERATING_METRIC = "operating_metric"

# SIM-384: the OTHER escape valve -- "financial but unplaceable" (an ambiguous
# subtotal, a nonstandard label the proposer cannot confidently name) as
# distinct from OPERATING_METRIC's "real sector metric". The two used to
# collapse onto one value, which destroyed the distinction 3a needs to
# reconcile core_unmapped claims among themselves rather than against
# unrelated sector metrics. See propose.py's mapping prompt, case 3.
CORE_UNMAPPED = "core_unmapped"

CANONICAL_ATTRIBUTES: frozenset[str] = CORE_ATTRIBUTES | {OPERATING_METRIC, CORE_UNMAPPED}


def gate_canonical_attribute(proposed: str) -> tuple[str, list[str]]:
    """The code half of SIM-344's proposer-with-code-gate: a proposed canonical
    string earns that exact value only by literal membership in CORE_ATTRIBUTES,
    or by being the OPERATING_METRIC or CORE_UNMAPPED escape valves. Anything
    else -- a hallucinated near-miss, a paraphrase outside all three answers --
    falls to OPERATING_METRIC and is flagged attribute_unmapped rather than
    trusted. Nothing calling this function can end up with a canonical
    attribute outside CANONICAL_ATTRIBUTES.
    """
    if proposed in CORE_ATTRIBUTES:
        return proposed, []
    if proposed == OPERATING_METRIC:
        return OPERATING_METRIC, []
    if proposed == CORE_UNMAPPED:
        return CORE_UNMAPPED, ["attribute_unmapped"]
    return OPERATING_METRIC, ["attribute_unmapped"]


class ClaimValue(BaseModel):
    """Mirrors the C3 contract's `value` object. `scale_source` has no null in
    its JSON-schema type (it's a bare enum), so it must be omitted -- never
    emitted as null -- whenever it isn't known; see `to_json`."""

    model_config = ConfigDict(extra="forbid")

    raw: str
    normalized: float | None
    unit: str | None
    value_type: ValueType
    scale_multiplier: float | None = None
    scale_source: ScaleSource | None = None

    def to_json(self) -> dict:
        out: dict = {
            "raw": self.raw,
            "normalized": self.normalized,
            "unit": self.unit,
            "value_type": self.value_type,
        }
        if self.scale_multiplier is not None:
            out["scale_multiplier"] = self.scale_multiplier
        if self.scale_source is not None:
            out["scale_source"] = self.scale_source
        return out


class PdfLocation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["pdf"] = "pdf"
    file: str
    page: int
    # Optional because a `missing` claim has no span: it records the page it
    # searched and stops there. The contract enforces both halves -- a claim that
    # is not missing MUST carry the span, and a missing one must NOT.
    # `paragraph` is deliberately absent: it belongs to the docx location. A PDF
    # is paginated, not a flow document, so its locator is the page.
    char_start: int | None = None
    char_end: int | None = None
    bbox: list[BBox] = Field(default_factory=list)
    document_id: str | None = None
    document_name: str | None = None

    def to_json(self) -> dict:
        out: dict = {
            "kind": "pdf",
            "file": self.file,
            "page": self.page,
        }
        if self.char_start is not None and self.char_end is not None:
            out["char_start"] = self.char_start
            out["char_end"] = self.char_end
        if self.bbox:
            out["bbox"] = [[b.x0, b.top, b.x1, b.bottom] for b in self.bbox]
        if self.document_id is not None:
            out["document_id"] = self.document_id
        if self.document_name is not None:
            out["document_name"] = self.document_name
        return out


class XlsxLocation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["xlsx"] = "xlsx"
    file: str
    sheet: str
    cell_ref: str
    document_id: str | None = None
    document_name: str | None = None

    def to_json(self) -> dict:
        out: dict = {
            "kind": "xlsx",
            "file": self.file,
            "sheet": self.sheet,
            "cell_ref": self.cell_ref,
        }
        if self.document_id is not None:
            out["document_id"] = self.document_id
        if self.document_name is not None:
            out["document_name"] = self.document_name
        return out


Location = PdfLocation | XlsxLocation


class Claim(BaseModel):
    """One claims-spine row, per contracts/claims.schema.json. `id`/`deal_id`/
    `session_id`/`org_id`/`data_source_id` are the backend's to fill in at
    persistence time -- the parser doesn't know them, so they're omitted here
    rather than populated with placeholders."""

    model_config = ConfigDict(extra="forbid")

    entity: str
    attribute: str
    # SIM-344: the document's own words for this attribute, before canonical
    # mapping. None for a claim the canonicalization pass never reached (e.g.
    # qualitative assertions, which keep their own attribute vocabulary via
    # assertion_class) -- absence here means "not in scope", not "unmapped".
    attribute_raw: str | None = None
    value: ClaimValue
    location: Location
    status: Status
    # SIM-365: a stable, positional, attribute-independent id for this claim.
    # Assigned once at extraction by extract_service._assign_claim_refs (None
    # until that pass runs; the pipeline always runs it before emit). Edges
    # reference it instead of the run-scoped element_id_for, and the backend keys
    # idempotent re-ingest on it (unique per org + data_source).
    claim_ref: str | None = None
    period_year: int | None = None
    period_kind: PeriodKind | None = None
    verification_method: VerificationMethod | None = None
    section: str | None = None
    flags: list[str] = Field(default_factory=list)
    # Which extraction contract produced this row. Absent means quantitative, so
    # every claim written before this field existed stays valid and unchanged.
    claim_kind: Literal["quantitative", "qualitative"] | None = None
    assertion_class: str | None = None
    # SIM-364: the assertion type (see ClaimType), assigned at extraction.
    # Non-null with an `unknown` fallback so an un-typeable claim is visibly
    # untyped and skipped by type-routing, never mis-routed as the wrong type.
    claim_type: ClaimType = "unknown"

    def to_json(self) -> dict:
        out: dict = {
            "entity": self.entity,
            "attribute": self.attribute,
            "value": self.value.to_json(),
            "location": self.location.to_json(),
            "status": self.status,
            "claim_type": self.claim_type,
        }
        if self.claim_ref is not None:
            out["claim_ref"] = self.claim_ref
        if self.attribute_raw is not None:
            out["attribute_raw"] = self.attribute_raw
        # Omitted rather than emitted null when unresolved, matching every other
        # optional field here -- the contract treats absence and null the same,
        # and this keeps the JSON free of noise for the (currently common) case
        # where the source doesn't qualify the period at all.
        if self.period_year is not None:
            out["period_year"] = self.period_year
        if self.period_kind is not None:
            out["period_kind"] = self.period_kind
        # Emitted only when trust was actually earned. The contract requires it
        # once status is cited or later, and forbids inventing one before that.
        if self.verification_method is not None:
            out["verification_method"] = self.verification_method
        if self.claim_kind is not None:
            out["claim_kind"] = self.claim_kind
        if self.assertion_class is not None:
            out["assertion_class"] = self.assertion_class
        if self.section is not None:
            out["section"] = self.section
        if self.flags:
            out["flags"] = self.flags
        return out


class SkippedPage(BaseModel):
    """One page a tier could not extract (a table that would not parse, a prose
    or qualitative model call that raised) -- the run-level record that makes a
    caught-and-skipped page auditable instead of a stderr line lost when the run
    ends. `tier` names which pass skipped it; `reason` is its exception."""

    model_config = ConfigDict(extra="forbid")

    page: int
    tier: str
    reason: str

    def to_json(self) -> dict:
        return {"page": self.page, "tier": self.tier, "reason": self.reason}


def element_id_for(claim: Claim) -> str:
    """A per-claim id, stable within one parse run, for cross-referencing
    claims outside the C3 contract itself (SIM-341's `edges`). Not persisted
    and not the backend's eventual `id` -- the backend assigns that at
    ingest and maps these run-scoped ids onto it.

    Deliberately distinct from the page+attribute id FlagLog entries key on
    (built inline in emit_pdf_claim): that one exists to bucket flags raised
    before a quote resolves, so it intentionally collides across claims that
    share a page and attribute. An edge instead names ONE specific claim, so
    this includes the resolved span (or cell ref) too -- two claims with the
    same page and attribute but different quotes must not collapse onto one
    id, or a `contradicts` edge would point from a claim to itself.
    """
    location = claim.location
    if isinstance(location, PdfLocation):
        base = f"pdf:{location.file}:p{location.page}:{claim.attribute}"
        if location.char_start is not None and location.char_end is not None:
            return f"{base}:c{location.char_start}-{location.char_end}"
        return base
    return f"xlsx:{location.file}:{location.sheet}!{location.cell_ref}:{claim.attribute}"


def claim_ref_base(claim: Claim) -> str:
    """The positional stem of a claim's `claim_ref` (SIM-365), before the
    disambiguating ordinal `_assign_claim_refs` appends.

    PDF: `{page}:{char_start}-{char_end}`, or `{page}:none` for a claim with no
    resolved span (a `missing` claim, or a spanless qualitative one). XLSX:
    `{sheet}!{cell_ref}`. Deliberately excludes `attribute`: E2 canonicalizes
    attribute, so an id built on it would not be stable across runs -- exactly
    what element_id_for above does and why edges must not use it.
    """
    location = claim.location
    if isinstance(location, PdfLocation):
        if location.char_start is not None and location.char_end is not None:
            return f"{location.page}:{location.char_start}-{location.char_end}"
        return f"{location.page}:none"
    return f"{location.sheet}!{location.cell_ref}"


# The two within-page relationships the fan-in reducer
# (parser_service/extract_service.py) can find between claims two different
# tiers produced on the same page. Kept
# here, mirrored in the C3 contract's edge definition, same pattern as
# FLAG_TYPES above.
EdgeType = Literal["same_fact", "contradicts"]

EDGE_TYPES = frozenset({"same_fact", "contradicts"})


class Edge(BaseModel):
    """One within-page relationship between two claims, emitted by the SIM-341
    fan-in reducer. The parser only emits the edge -- it never drops or merges
    the claims it connects, so both provenances survive; the backend maps
    `from`/`to` (element_id_for) onto real claim rows at ingest and is where
    cross-document edges get added, not here.

    `same_fact`: `to` is the higher-precision source (table beats prose on a
    number) and `from` is kept as corroboration of the same real-world fact.
    `contradicts`: `from`/`to` name what looks like the same fact but disagree
    on value -- neither wins, both stay, and the disagreement is the signal.
    """

    # No alias for `from_`: this model is only ever constructed in Python (the
    # reducer) and serialized outward via to_json(), never parsed FROM JSON --
    # so there is no need for pydantic to accept the reserved-word key "from"
    # as input, only to emit it, which to_json() does directly.
    model_config = ConfigDict(extra="forbid")

    type: EdgeType
    from_: str
    to: str
    basis: str

    def to_json(self) -> dict:
        return {"type": self.type, "from": self.from_, "to": self.to, "basis": self.basis}


class FlagLogEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    stage: str
    element_id: str
    flag_type: str
    detail: str | None = None


class FlagLog:
    """Structured per-run flag log -- alpha-appropriate monitoring capture,
    not a dashboard. Every flag raised anywhere in a run is recorded here, in
    addition to being attached to the fact/element it belongs to, so the
    history exists once the aggregation/dashboard surface is built (Pilot).
    One FlagLog per parse run.
    """

    def __init__(self, run_id: str | None = None) -> None:
        self.run_id = run_id or str(uuid.uuid4())
        self.entries: list[FlagLogEntry] = []

    def log(self, stage: str, element_id: str, flag_type: str, detail: str | None = None) -> None:
        if flag_type not in FLAG_TYPES:
            raise ValueError(f"unknown flag_type {flag_type!r}; not in the C3 contract's flag enum")
        self.entries.append(
            FlagLogEntry(
                run_id=self.run_id,
                stage=stage,
                element_id=element_id,
                flag_type=flag_type,
                detail=detail,
            )
        )

    def log_all(
        self, stage: str, element_id: str, flag_types: list[str], detail: str | None = None
    ) -> None:
        for flag_type in flag_types:
            self.log(stage, element_id, flag_type, detail)

    def to_json(self) -> list[dict]:
        return [e.model_dump(exclude_none=True) for e in self.entries]


_STAGE_CLAIM_EMISSION = "claim_emission"
_STAGE_ELEMENT_PROCESSING = "element_processing"


def _missing_pdf_claim(
    entity: str,
    attribute: str,
    raw: str,
    value_type: ValueType,
    page: PageIndex,
    file: str,
    flags: list[str],
    *,
    document_id: str | None,
    document_name: str | None,
    section: str | None,
    period_year: int | None = None,
    period_kind: PeriodKind | None = None,
    claim_kind: Literal["quantitative", "qualitative"] | None = None,
    assertion_class: str | None = None,
    attribute_raw: str | None = None,
) -> Claim:
    """A claim we looked for and did not find.

    The location carries the page that was searched and NO span. Not a zero
    span -- char_start=0/char_end=0 is not an absence, it is a citation to the
    first character of the page, and a highlight UI would draw it over unrelated
    text while the claim says nothing was found. The contract rejects it.

    period_year/period_kind are carried through even here: the period was read
    off the column header, not the resolved span, so a cell whose value fails
    to resolve still keeps the period it was searched under.
    """
    return Claim(
        entity=entity,
        attribute=attribute,
        attribute_raw=attribute_raw,
        value=ClaimValue(raw=raw, normalized=None, unit=None, value_type=value_type),
        location=PdfLocation(
            file=file,
            page=page.page,
            document_id=document_id,
            document_name=document_name,
        ),
        status="missing",
        period_year=period_year,
        period_kind=period_kind,
        section=section,
        flags=flags,
        claim_kind=claim_kind,
        assertion_class=assertion_class,
    )


def emit_pdf_claim(
    entity: str,
    attribute: str,
    quote: str,
    page: PageIndex,
    *,
    value_type: ValueType,
    origin: ValueOrigin,
    file: str,
    flag_log: FlagLog,
    table: TableRecord | None = None,
    cell: TableCellRecord | None = None,
    section: str | None = None,
    value_text: str | None = None,
    period_year: int | None = None,
    period_kind: PeriodKind | None = None,
    claim_kind: Literal["quantitative", "qualitative"] | None = None,
    assertion_class: str | None = None,
    extra_flags: list[str] | None = None,
    document_id: str | None = None,
    document_name: str | None = None,
    page_header_ok: bool = True,
    stage: str = _STAGE_CLAIM_EMISSION,
    attribute_raw: str | None = None,
    claim_type: ClaimType = "unknown",
) -> Claim:
    """Emit one PDF-sourced claim for `quote` on `page`. Fails closed to
    `status="missing"` on a zero-text page or an unresolved/ambiguous
    quote (DS-W3-3) -- never a fabricated or partially-cited value. `table`/
    `cell` are passed through to DS-W3-4's column-header scale lookup when the
    quote came from a table cell; omit both for prose.

    `value_text` separates the two jobs the quote otherwise does at once. A table
    cell's quote IS its value, so parsing the magnitude out of the quote is right
    there. A prose quote is a SENTENCE CONTAINING the value, and parsing that the
    same way takes the leftmost number in the sentence: "In 2003, 18,454 students
    attended" yields 2003, and a parenthetical yields an accounting negative. Both
    were observed on the first real page this ran against. Pass the value token
    itself as `value_text` -- the quote still carries the citation, and the
    magnitude comes from the token. Callers whose quote is already the value (every
    table caller) omit it.

    When `cell` is given, the quote is resolved inside that cell's own region
    first: the caller knows which cell it read, so asking the page instead
    throws that away and cannot cite a value the table repeats (see
    resolver.resolve_in_cell). Page-wide search remains the fallback, so a cell
    whose box does not enclose its own text is no worse off than before.
    """
    element_id = f"pdf:{file}:p{page.page}:{attribute}"

    if not page.text.strip():
        flag_log.log(stage, element_id, "zero_text_page")
        return _missing_pdf_claim(
            entity,
            attribute,
            quote,
            value_type,
            page,
            file,
            [*(extra_flags or []), "zero_text_page"],
            document_id=document_id,
            document_name=document_name,
            section=section,
            period_year=period_year,
            period_kind=period_kind,
            claim_kind=claim_kind,
            assertion_class=assertion_class,
            attribute_raw=attribute_raw,
        )

    # Cell-scoped first when we know the cell, page-wide otherwise. Both go
    # through the same exact-span core, so this widens what is citable without
    # loosening what counts as a citation.
    span = resolve_in_cell(quote, page, cell) if cell is not None else None
    if span is None:
        span = resolve(quote, page)
    if span is None:
        flag_log.log(stage, element_id, "quote_unresolved", detail=quote)
        return _missing_pdf_claim(
            entity,
            attribute,
            quote,
            value_type,
            page,
            file,
            [*(extra_flags or []), "quote_unresolved"],
            document_id=document_id,
            document_name=document_name,
            section=section,
            period_year=period_year,
            period_kind=period_kind,
            claim_kind=claim_kind,
            assertion_class=assertion_class,
            attribute_raw=attribute_raw,
        )

    # Seeded, not empty: a caller-supplied flag (e.g. magnitude_unparseable from
    # the prose guard) must survive into the claim. The scaled branch below used
    # to REASSIGN this list, which would have discarded every seeded flag
    # silently -- it extends now.
    flags: list[str] = list(extra_flags or [])

    # The one chokepoint every PDF claim from every path crosses, which is why
    # the check lives here rather than in a caller that a future path could
    # route around. `all` rather than `any` mirrors inspect.is_boilerplate_token:
    # a partially-tagged span is a real claim sitting beside furniture, and
    # over-excluding would hide a real miss. Flag-only for now -- tag_boilerplate
    # keys on repetition, so a repeated section header could be caught, and
    # dropping on it would gamble measured recall on a heuristic never built to
    # gate emission.
    span_chars = page.char_map[span.char_start : span.char_end]
    if span_chars and all(c.is_boilerplate for c in span_chars):
        flags.append("cites_boilerplate")

    scale_detail: str | None = None
    if value_type == "text":
        # Same choice the scaled branch makes below: the quote is the citation
        # and may be a whole clause, so where the caller named the value token
        # it is the raw. Both are verbatim page text; this keeps a text claim
        # from recording a sentence where it means "five years".
        value = ClaimValue(
            raw=value_text if value_text is not None else quote,
            normalized=None,
            unit=None,
            value_type=value_type,
        )
    else:
        scale_result = determine_scale(
            value_text if value_text is not None else quote,
            page,
            span.char_start,
            value_type=value_type,
            origin=origin,
            table=table,
            cell=cell,
            page_header_ok=page_header_ok,
        )
        flags.extend(scale_result.flags)
        if (
            value_type == "currency"
            and scale_result.unit is None
            and scale_result.scale_source
            in (
                "column_header",
                "page_header",
            )
        ):
            # A real scale header was found and applied, but it carried no
            # currency code ("(in Thousands)" with no "CAD"/"USD" prefix) --
            # distinct from assumed_1x (no header at all), so flagged
            # separately: the multiplier is trusted, the currency is not.
            flags.append("ambiguous_unit")
        scale_detail = scale_result.scale_context
        # The magnitude must be what the multiplier says it is. This cannot be a
        # contract rule -- JSON Schema has no arithmetic across fields -- so it is
        # asserted where the claim is MADE, not merely where it is validated. It
        # holds by construction today; the assertion is here so that a future
        # scale path cannot quietly stop satisfying it, which is precisely how
        # every 1000x defect in the July 2026 audit reached the store.
        if not scale_invariant_holds(
            scale_result.raw, scale_result.normalized, scale_result.scale_multiplier
        ):
            raise AssertionError(
                f"scale invariant violated: {scale_result.raw!r} x "
                f"{scale_result.scale_multiplier} != {scale_result.normalized}"
            )
        value = ClaimValue(
            raw=scale_result.raw,
            normalized=scale_result.normalized,
            unit=scale_result.unit,
            value_type=value_type,
            scale_multiplier=scale_result.scale_multiplier,
            scale_source=scale_result.scale_source,
        )

    # Logged once for both branches. It used to sit inside the scaled branch,
    # so a flag raised on a text claim was attached but never logged.
    if flags:
        flag_log.log_all(stage, element_id, flags, detail=scale_detail)

    bbox = span.line_bboxes or [span.bbox]
    return Claim(
        entity=entity,
        attribute=attribute,
        attribute_raw=attribute_raw,
        claim_type=claim_type,
        value=value,
        location=PdfLocation(
            file=file,
            page=span.page,
            char_start=span.char_start,
            char_end=span.char_end,
            bbox=bbox,
            document_id=document_id,
            document_name=document_name,
        ),
        status="proposed",
        period_year=period_year,
        period_kind=period_kind,
        section=section,
        flags=flags,
        claim_kind=claim_kind,
        assertion_class=assertion_class,
    )


def emit_pdf_table_cell_claim(
    entity: str,
    attribute: str,
    table: TableRecord,
    cell: TableCellRecord,
    page: PageIndex,
    *,
    value_type: ValueType,
    file: str,
    flag_log: FlagLog,
    section: str | None = None,
    document_id: str | None = None,
    document_name: str | None = None,
    page_header_ok: bool = True,
    period_year: int | None = None,
    period_kind: PeriodKind | None = None,
    stage: str = _STAGE_CLAIM_EMISSION,
    attribute_raw: str | None = None,
) -> Claim:
    """Emit a fact for one table cell. A cell with no resolvable bbox (neither
    Docling-native nor DS-2's reconstruction fallback located it) is written
    missing outright -- citing a cell whose own region is unknown would be a
    partial citation, which all-or-nothing provenance forbids.

    period_year/period_kind (SIM-345) are the caller's to supply: this function
    reads one table cell and has no view of the column header its value sits
    under, so the caller (extract.claims_from_table, which does) resolves the
    period and passes it through."""
    if cell.bbox_source is None:
        element_id = f"pdf:{file}:p{page.page}:table:r{cell.row}c{cell.col}:{attribute}"
        flag_log.log(
            stage,
            element_id,
            "ambiguous_region_bounds",
            detail=f"cell ({cell.row},{cell.col}) has no resolvable source bbox",
        )
        return _missing_pdf_claim(
            entity,
            attribute,
            cell.text_normalized,
            value_type,
            page,
            file,
            ["ambiguous_region_bounds"],
            document_id=document_id,
            document_name=document_name,
            section=section,
            attribute_raw=attribute_raw,
            period_year=period_year,
            period_kind=period_kind,
        )

    return emit_pdf_claim(
        entity,
        attribute,
        cell.text_normalized,
        page,
        value_type=value_type,
        # A table cell's origin is settled by construction: this entry point
        # cannot be called without the table it came from.
        origin="table",
        # A printed table figure is a FinGround `numerical` claim (SIM-364).
        claim_type="numerical",
        file=file,
        flag_log=flag_log,
        table=table,
        cell=cell,
        section=section,
        document_id=document_id,
        document_name=document_name,
        page_header_ok=page_header_ok,
        period_year=period_year,
        period_kind=period_kind,
        stage=stage,
        attribute_raw=attribute_raw,
    )


def emit_xlsx_claim(
    entity: str,
    attribute: str,
    sheet: XlsxSheetRecord,
    cell: XlsxCellRecord,
    *,
    value_type: ValueType,
    file: str,
    flag_log: FlagLog,
    section: str | None = None,
    document_id: str | None = None,
    document_name: str | None = None,
    stage: str = _STAGE_CLAIM_EMISSION,
) -> Claim:
    """Emit one XLSX-sourced fact. Provenance always resolves to
    `cell.merged_anchor_ref`, per DS-W3-5, so a non-anchor merged cell still
    points somewhere real. openpyxl stores a merged range's value only on its
    anchor cell -- a non-anchor cell's own `value`/`formula` are always None --
    so a non-anchor `cell` is resolved to its anchor's record before reading
    any value data; only the anchor ever actually holds one.

    Status per cell kind:
      literal -> cited/direct_read. Reading the bytes IS the verification;
                 there is no span to resolve and nothing left to check.
      formula -> proposed. The cell reference is exact, but the value is only
                 knowable once HyperFormula re-executes it outside this service,
                 and the file's own cached result is never trusted as a
                 substitute (the exact anti-pattern DS-W3-5 replaces). Calling it
                 cited would assert a check this service never ran; re-execution
                 is what later earns cited/formula_reexecution.
    """
    if not cell.is_merged_anchor:
        anchor = next(
            (c for c in sheet.cells if c.cell_ref == cell.merged_anchor_ref),
            None,
        )
        if anchor is None:
            # A merged range whose anchor is absent from the sheet record means
            # the parse output is internally inconsistent. Fail closed rather
            # than raise StopIteration out of a generator: there is no value to
            # read, so there is no honest claim to make about this cell.
            raise ValueError(
                f"merged cell {cell.cell_ref!r} names anchor {cell.merged_anchor_ref!r}, "
                f"which is not present in sheet {sheet.name!r}"
            )
        cell = anchor

    element_id = f"xlsx:{file}:{sheet.name}!{cell.cell_ref}:{attribute}"
    location = XlsxLocation(
        file=file,
        sheet=sheet.name,
        cell_ref=cell.merged_anchor_ref,
        document_id=document_id,
        document_name=document_name,
    )

    if cell.formula is not None:
        return Claim(
            entity=entity,
            attribute=attribute,
            value=ClaimValue(raw=cell.formula, normalized=None, unit=None, value_type=value_type),
            location=location,
            status="proposed",
            # A workbook formula is recomputed from its operands -> FinGround
            # `computational` (SIM-364); this is the type the consistency pass routes on.
            claim_type="computational",
            section=section,
            flags=[],
        )

    if value_type == "text":
        raw = "" if cell.value is None else str(cell.value)
        found = cell.value is not None
        return Claim(
            entity=entity,
            attribute=attribute,
            value=ClaimValue(raw=raw, normalized=None, unit=None, value_type=value_type),
            location=location,
            status="cited" if found else "missing",
            verification_method="direct_read" if found else None,
            section=section,
            flags=[] if found else ["quote_unresolved"],
        )

    scale_result = determine_xlsx_scale(sheet, cell, value_type=value_type)
    flags = list(scale_result.flags)
    if (
        value_type == "currency"
        and scale_result.unit is None
        and scale_result.scale_source
        in (
            "column_header",
            "page_header",
        )
    ):
        flags.append("ambiguous_unit")
    if flags:
        flag_log.log_all(stage, element_id, flags, detail=scale_result.scale_context)

    value = ClaimValue(
        raw=scale_result.raw,
        normalized=scale_result.normalized,
        unit=scale_result.unit,
        value_type=value_type,
        scale_multiplier=scale_result.scale_multiplier,
        scale_source=scale_result.scale_source,
    )
    return Claim(
        entity=entity,
        attribute=attribute,
        value=value,
        location=location,
        status="cited",
        verification_method="direct_read",
        # A directly-read numeric cell -> FinGround `numerical` (SIM-364).
        claim_type="numerical",
        section=section,
        flags=flags,
    )


def log_table_element_flags(
    flag_log: FlagLog,
    file: str,
    table_element: TableElement,
    stage: str = _STAGE_ELEMENT_PROCESSING,
) -> None:
    """Record a TableElement's own flags (e.g. ragged_table_rows) to the
    structured per-run log, independent of whether any cell in it became a
    claim -- the monitoring scope is the interpreted element, not just what
    got cited."""
    element_id = f"pdf:{file}:p{table_element.page}:table"
    flag_log.log_all(stage, element_id, table_element.flags)


def log_chart_element_flags(
    flag_log: FlagLog,
    file: str,
    chart_element: ChartElement,
    stage: str = _STAGE_ELEMENT_PROCESSING,
) -> None:
    """Record a ChartElement's own flags (always chart_data_not_extracted)."""
    element_id = f"pdf:{file}:p{chart_element.page}:chart"
    flag_log.log_all(stage, element_id, chart_element.flags)

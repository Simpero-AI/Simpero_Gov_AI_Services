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
from typing import Literal

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
    }
)


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
    value: ClaimValue
    location: Location
    status: Status
    verification_method: VerificationMethod | None = None
    section: str | None = None
    flags: list[str] = Field(default_factory=list)
    # Which extraction contract produced this row. Absent means quantitative, so
    # every claim written before this field existed stays valid and unchanged.
    claim_kind: Literal["quantitative", "qualitative"] | None = None
    assertion_class: str | None = None

    def to_json(self) -> dict:
        out: dict = {
            "entity": self.entity,
            "attribute": self.attribute,
            "value": self.value.to_json(),
            "location": self.location.to_json(),
            "status": self.status,
        }
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
) -> Claim:
    """A claim we looked for and did not find.

    The location carries the page that was searched and NO span. Not a zero
    span -- char_start=0/char_end=0 is not an absence, it is a citation to the
    first character of the page, and a highlight UI would draw it over unrelated
    text while the claim says nothing was found. The contract rejects it.
    """
    return Claim(
        entity=entity,
        attribute=attribute,
        value=ClaimValue(raw=raw, normalized=None, unit=None, value_type=value_type),
        location=PdfLocation(
            file=file,
            page=page.page,
            document_id=document_id,
            document_name=document_name,
        ),
        status="missing",
        section=section,
        flags=flags,
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
    claim_kind: Literal["quantitative", "qualitative"] | None = None,
    assertion_class: str | None = None,
    extra_flags: list[str] | None = None,
    document_id: str | None = None,
    document_name: str | None = None,
    stage: str = _STAGE_CLAIM_EMISSION,
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
            ["zero_text_page"],
            document_id=document_id,
            document_name=document_name,
            section=section,
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
            ["quote_unresolved"],
            document_id=document_id,
            document_name=document_name,
            section=section,
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
    stage: str = _STAGE_CLAIM_EMISSION,
) -> Claim:
    """Emit a fact for one table cell. A cell with no resolvable bbox (neither
    Docling-native nor DS-2's reconstruction fallback located it) is written
    missing outright -- citing a cell whose own region is unknown would be a
    partial citation, which all-or-nothing provenance forbids."""
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
        file=file,
        flag_log=flag_log,
        table=table,
        cell=cell,
        section=section,
        document_id=document_id,
        document_name=document_name,
        stage=stage,
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

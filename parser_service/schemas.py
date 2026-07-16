from typing import Literal

from pydantic import BaseModel, Field


class CharBox(BaseModel):
    char: str
    x0: float
    top: float
    x1: float
    bottom: float
    page: int = Field(ge=1)
    is_boilerplate: bool = False
    # "char": a real per-glyph rectangle from Docling. "word": Docling does not
    # expose per-character geometry under this parse configuration, so this box
    # is the character's containing word's full bounding box, not an estimate
    # of the glyph's own extent — consumers must not treat "word"-precision
    # boxes as exact-span highlighting coordinates. No default: every CharBox
    # must state which one it actually is.
    precision: Literal["char", "word"]


class PageIndex(BaseModel):
    page: int = Field(ge=1)
    text: str
    char_map: list[CharBox]


class ParagraphIndex(BaseModel):
    """The DOCX analog of PageIndex: one paragraph of a flow document.

    A Word file has no page layout, so there is deliberately no char_map and no
    geometry here — Docling reports no prov at all for DOCX text items, because
    there is none to report. The exact citation for DOCX is therefore
    (paragraph, char_start, char_end), which is precisely what the claims
    contract's docx location carries. Same trust bar as PDF, different shape:
    a claim still resolves to an exact span or it is not citable.
    """

    paragraph: int = Field(ge=0, description="0-based index in document body order.")
    text: str
    # Docling's own label for the item ("section_header", "text", ...). Advisory
    # context for the extractor; never a citation.
    label: str | None = None


class TableCellRecord(BaseModel):
    row: int = Field(ge=0)
    col: int = Field(ge=0)
    row_span: int = Field(ge=1)
    col_span: int = Field(ge=1)
    # Verbatim cell text exactly as Docling read it off the page — may contain
    # a split numeric token ("3 ,817"). Kept unmodified so it still matches the
    # source for provenance/debugging.
    text: str
    # `text` with split numeric tokens collapsed ("3 ,817" -> "3,817"), using
    # the same rule as the flat page index (normalize.py). Parse numeric claims
    # from THIS field, not `text` — see DS-W3-2 finding F2.
    text_normalized: str
    # Docling's own flag. ADVISORY ONLY — on tables with unlabeled columns it
    # marks the first *data* row instead of a real header row (verified on
    # Pitchbook PDF-page 17). Use TableRecord.header_row to locate the header.
    column_header: bool
    row_header: bool
    page: int = Field(ge=1)
    # None only when neither the Docling-native bbox nor the reconstruction
    # fallback resolved a coordinate — see TableRecord.cell_provenance_ok.
    x0: float | None
    top: float | None
    x1: float | None
    bottom: float | None
    # How the coordinate above was obtained. "docling_native": Docling exposed a
    # valid cell bbox. "reconstructed": it did not, so the bbox was recovered
    # from the page's positioned index by exact-span match of the cell text
    # (fail closed on absent/ambiguous). None: neither resolved, so the cell has
    # no citable location. A reconstructed box must never be treated as native.
    bbox_source: Literal["docling_native", "reconstructed"] | None = None


class TableRecord(BaseModel):
    page: int = Field(ge=1)
    num_rows: int = Field(ge=0)
    num_cols: int = Field(ge=0)
    cells: list[TableCellRecord]
    # False if any cell still lacks a resolvable source bbox after both the
    # Docling-native path and the positioned-index reconstruction fallback — per
    # DS-2, such a table must not be treated as citable until this is true.
    cell_provenance_ok: bool
    # Count of cells whose coordinates came from the reconstruction fallback
    # rather than a Docling-native bbox. > 0 means the table is citable but part
    # of its geometry was recovered from the flat index, not read from the
    # layout — surfaced so it is never silently trusted as native.
    reconstructed_cells: int = Field(ge=0, default=0)
    # Header row inferred from table *structure*, not from Docling's
    # column_header flag. None means this table has unlabeled columns (no
    # header row exists to key off) — consumers must then fall back to
    # positional/section context rather than inventing column names.
    header_row: int | None
    # True only when Docling's column_header flags agree with the structurally
    # inferred header row. False means the flags are untrustworthy for this
    # table and must be ignored.
    column_headers_reliable: bool


class BBox(BaseModel):
    # TOPLEFT origin, matching CharBox: top < bottom numerically.
    x0: float
    top: float
    x1: float
    bottom: float
    page: int = Field(ge=1)


class XlsxCellRecord(BaseModel):
    sheet: str
    cell_ref: str = Field(description='e.g. "B14".')
    row: int = Field(ge=1)
    col: int = Field(ge=1)
    # The literal value Excel stores for this cell. None for a formula cell —
    # the file's cached formula result is never trusted as the claim's value
    # (the exact anti-pattern DS-W3-5 replaces); see `formula`/`cached_value`.
    value: float | str | bool | None
    # The formula text (e.g. "=B2+B3"), when this cell computes rather than
    # states its value. None for a literal cell. Re-execution happens via
    # HyperFormula on the TS side (per the locked architecture) — this module
    # only surfaces the formula itself, never evaluates it.
    formula: str | None = None
    # The workbook's own last-cached result for a formula cell, kept only for
    # debugging/comparison. NEVER authoritative: a claim built from this field
    # instead of a HyperFormula re-execution is not formula-verified — that is
    # the exact trust gap ("recompute...rather than trusting the file's
    # cached value") this ticket exists to close.
    cached_value: float | str | bool | None = None
    number_format: str
    # True when Excel's own number format is a percentage ("0.00%", "0%", ...).
    # A native-percentage cell stores the fraction (0.285 for 28.5%), not the
    # whole number — the SIM-17 bug this ticket folds in and fixes. scale.py's
    # inline "%" detection can't see this: the "%" lives in the cell's
    # formatting metadata, not in any printed text.
    is_percentage: bool = False
    # False when this cell is a non-anchor member of a merged range — openpyxl
    # stores the value only on the anchor (top-left) cell, so a non-anchor
    # cell's `value`/`formula` are always None. Consumers must resolve
    # provenance to `merged_anchor_ref`, per the ticket: "resolve to the
    # anchor cell so provenance points somewhere real."
    is_merged_anchor: bool = True
    # The anchor cell's ref for the merged range this cell belongs to; equals
    # `cell_ref` when the cell is unmerged or is itself the anchor.
    merged_anchor_ref: str


class XlsxSheetRecord(BaseModel):
    name: str
    cells: list[XlsxCellRecord]
    # openpyxl's sheet visibility: "visible", "hidden", or "veryHidden". A claim
    # sourced from a non-visible sheet is materially different in diligence
    # (adjustments/assumptions are often parked on hidden sheets), so the state
    # is preserved on the record rather than silently flattened to visible.
    sheet_state: Literal["visible", "hidden", "veryHidden"] = "visible"
    # True if openpyxl found at least one embedded chart on this sheet. Flag
    # chart_data_not_extracted downstream — the chart object itself is not
    # read, though its underlying series data is usually already present in
    # cells (and thus already extracted separately).
    has_chart: bool = False


class XlsxParseResult(BaseModel):
    sha256: str
    sheets: list[XlsxSheetRecord]


class Span(BaseModel):
    # Character offsets into PageIndex.text; char_end is exclusive, so
    # page.text[char_start:char_end] == the resolved quote.
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    page: int = Field(ge=1)
    # Union of every matched character's box. On a quote that wraps across
    # lines this is a coarse envelope that also covers text between the lines —
    # use it only to locate the claim on the page, not to draw the highlight.
    bbox: BBox
    # One box per visual line the quote covers (the matched region split on the
    # newline characters the flat page index inserts between lines).
    # Single-element for a single-line quote. This is the correct primitive to
    # highlight from; a single union box would cover unrelated text on any
    # quote that wraps.
    line_bboxes: list[BBox]


class ParagraphSpan(BaseModel):
    """The DOCX analog of Span: an exact citation with no geometry.

    Deliberately a separate type rather than a Span with an optional bbox. A PDF
    Span *guarantees* geometry; weakening that to satisfy DOCX would force every
    PDF consumer to None-check a box that is always present, and would let a
    geometry-free span leak into a code path that assumes one. The formats differ
    legitimately — the claims contract already models them as separate location
    kinds — so the types differ too. What they share is the matching rule, not
    the shape: both resolve through the same exact, unambiguous, fail-closed core.
    """

    # Character offsets into ParagraphIndex.text; char_end is exclusive, so
    # paragraph.text[char_start:char_end] == the resolved quote.
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    paragraph: int = Field(ge=0)

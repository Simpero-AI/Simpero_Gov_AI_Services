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
    # the same rule as the flat page index (normalize.py). Parse numeric facts
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
    # (fail closed on absent/ambiguous). "vision_resolved": neither of the above
    # worked, so DS-W3-6 escalated to Claude Vision for a candidate reading,
    # which was then independently verified via the DS-W3-3 exact-span resolver
    # (fail closed) -- vision proposes text, never a coordinate, so this source
    # carries the same fail-closed guarantee as "reconstructed". None: nothing
    # resolved, so the cell has no citable location. A reconstructed or
    # vision-resolved box must never be treated as native.
    bbox_source: Literal["docling_native", "reconstructed", "vision_resolved"] | None = None


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


class Span(BaseModel):
    # Character offsets into PageIndex.text; char_end is exclusive, so
    # page.text[char_start:char_end] == the resolved quote.
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    page: int = Field(ge=1)
    # Union of every matched character's box. On a quote that wraps across
    # lines this is a coarse envelope that also covers text between the lines —
    # use it only to locate the fact on the page, not to draw the highlight.
    bbox: BBox
    # One box per visual line the quote covers (the matched region split on the
    # newline characters the flat page index inserts between lines).
    # Single-element for a single-line quote. This is the correct primitive to
    # highlight from; a single union box would cover unrelated text on any
    # quote that wraps.
    line_bboxes: list[BBox]

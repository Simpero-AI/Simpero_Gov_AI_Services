"""DS-W3-6 vision escalation -- Claude Vision for tables DS-2 (docling_parser.py
/ table_extract.py) couldn't resolve deterministically (locked: vision for
complex tables).

Vision never supplies a citable coordinate on its own say-so. It only
proposes cell text; that text must then resolve to a real page location via
the DS-W3-3 exact-span resolver (fail closed: found exactly once, or not
citable) before a cell counts as resolved. A vision guess that doesn't
resolve stays uncoordinated and flagged -- never silently accepted as if it
were a real geometry read.
"""

import base64
from collections.abc import Callable

from pydantic import BaseModel

from .resolver import resolve
from .schemas import PageIndex, TableCellRecord, TableRecord

# Per the "vision for complex tables" lock: always Opus, never downgraded for
# cost -- a wrong table reading here would otherwise be quietly cheaper to get
# wrong.
_VISION_MODEL = "claude-opus-4-8"


class VisionTableCell(BaseModel):
    row: int
    col: int
    text: str


class VisionTableReading(BaseModel):
    cells: list[VisionTableCell]


# A vision client reads a cropped table image and returns its best-effort
# per-cell reading. Injectable so escalate_table_to_vision's resolve-or-flag
# logic is testable without a real Claude API call; defaults to the real
# claude_vision_client for production callers.
VisionClient = Callable[[bytes], VisionTableReading]


def needs_vision_escalation(table: TableRecord) -> bool:
    """True when DS-2's deterministic paths (Docling-native + reconstruction)
    left at least one cell without a resolvable source bbox."""
    return not table.cell_provenance_ok


def claude_vision_client(image_bytes: bytes) -> VisionTableReading:
    """The real vision client. Claude reads a cropped table image and
    proposes each cell's text by (row, col) -- a proposal only; see
    escalate_table_to_vision for why it is never trusted as a coordinate on
    its own. `anthropic` is imported lazily so importing this module never
    requires the SDK/credentials unless this specific function is called."""
    import anthropic

    client = anthropic.Anthropic()
    image_b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
    response = client.messages.parse(
        model=_VISION_MODEL,
        max_tokens=16000,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": (
                            "Read this table image cell by cell. For every non-empty "
                            "cell, report its 0-indexed row and column and its exact "
                            "text as printed -- verbatim, do not normalize, round, or "
                            "reformat numbers. Omit empty cells."
                        ),
                    },
                ],
            }
        ],
        output_format=VisionTableReading,
    )
    if response.parsed_output is None:
        # A refusal or a max_tokens cutoff leaves no parsed structured output.
        # Fail loudly rather than let a None flow into the resolve-or-flag
        # logic as if it were an (empty) reading -- this is a hard failure of
        # the escalation attempt, not "no cells found".
        raise ValueError(
            f"Claude Vision returned no parsed output (stop_reason={response.stop_reason!r})"
        )
    return response.parsed_output


def escalate_table_to_vision(
    table: TableRecord,
    page: PageIndex,
    table_image: bytes,
    vision_client: VisionClient = claude_vision_client,
) -> TableRecord:
    """Re-resolve a table's unresolved cells via vision + exact-span matching.

    Only cells DS-2 could not place (x0 is None) are touched -- a cell DS-2
    already resolved (Docling-native or reconstructed) is never overwritten
    by a vision guess, since deterministic geometry always outranks a vision
    reading. For each unresolved cell, ask the vision client for its
    best-effort text at that (row, col), then try to resolve THAT text
    against the page via DS-W3-3's resolve() (exact, unambiguous substring
    match). Resolved -> a real bbox, bbox_source="vision_resolved". Not
    resolved (not found, or ambiguous) -> the cell stays uncoordinated and
    the table stays not cell_provenance_ok.

    A short-circuit if nothing is actually unresolved: `needs_vision_escalation`
    is the intended caller-side gate, but a real Claude API call is billed, so
    this function never makes one for a table that has nothing to resolve
    even if a caller forgets to check the gate first.
    """
    if not needs_vision_escalation(table):
        return table

    reading = vision_client(table_image)
    guesses = {(cell.row, cell.col): cell.text for cell in reading.cells}

    updated_cells: list[TableCellRecord] = []
    provenance_ok = True
    for cell in table.cells:
        if cell.x0 is not None:
            updated_cells.append(cell)
            continue

        guess = guesses.get((cell.row, cell.col))
        span = resolve(guess, page) if guess else None
        if span is not None:
            updated_cells.append(
                cell.model_copy(
                    update={
                        "x0": span.bbox.x0,
                        "top": span.bbox.top,
                        "x1": span.bbox.x1,
                        "bottom": span.bbox.bottom,
                        "bbox_source": "vision_resolved",
                    }
                )
            )
        else:
            updated_cells.append(cell)  # stays uncoordinated -- flagged, not trusted
            provenance_ok = False

    return table.model_copy(update={"cells": updated_cells, "cell_provenance_ok": provenance_ok})

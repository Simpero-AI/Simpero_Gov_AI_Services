"""DS-W3-6 vision escalation tests.

Fast, CI-portable tests only -- no real Claude API call is ever made here.
escalate_table_to_vision is tested against an injected fake vision_client, so
the resolve-or-flag logic (the actual trust boundary) is exercised
deterministically. claude_vision_client (the real integration) is tested
separately with anthropic.Anthropic mocked out, to pin the request shape
without hitting the network.
"""

import base64
from types import SimpleNamespace
from typing import Literal
from unittest.mock import MagicMock, patch

import pytest

from services.parser.parser_service.schemas import CharBox, PageIndex, TableCellRecord, TableRecord
from services.parser.parser_service.vision_escalation import (
    VisionTableCell,
    VisionTableReading,
    claude_vision_client,
    escalate_table_to_vision,
    needs_vision_escalation,
)


def _page(text: str, page_no: int = 1) -> PageIndex:
    char_map = [
        CharBox(
            char=ch,
            x0=float(i),
            top=0.0,
            x1=float(i + 1),
            bottom=10.0,
            page=page_no,
            precision="word",
        )
        for i, ch in enumerate(text)
    ]
    return PageIndex(page=page_no, text=text, char_map=char_map)


def _cell(
    row: int,
    col: int,
    text: str,
    *,
    x0: float | None = None,
    bbox_source: Literal["docling_native", "reconstructed", "vision_resolved"] | None = None,
) -> TableCellRecord:
    coords = (
        {"top": 0.0, "x1": x0 + 1.0, "bottom": 10.0}
        if x0 is not None
        else {"top": None, "x1": None, "bottom": None}
    )
    return TableCellRecord(
        row=row,
        col=col,
        row_span=1,
        col_span=1,
        text=text,
        text_normalized=text,
        column_header=False,
        row_header=False,
        page=1,
        x0=x0,
        bbox_source=bbox_source,
        **coords,
    )


def _table(
    cells: list[TableCellRecord],
    *,
    num_rows: int = 1,
    num_cols: int = 1,
    cell_provenance_ok: bool = False,
) -> TableRecord:
    return TableRecord(
        page=1,
        num_rows=num_rows,
        num_cols=num_cols,
        cells=cells,
        cell_provenance_ok=cell_provenance_ok,
        header_row=None,
        column_headers_reliable=False,
    )


# --------------------------------------------------------------------------- #
# needs_vision_escalation
# --------------------------------------------------------------------------- #


def test_needs_vision_escalation_true_when_provenance_not_ok() -> None:
    table = _table([_cell(0, 0, "x")], cell_provenance_ok=False)
    assert needs_vision_escalation(table) is True


def test_needs_vision_escalation_false_when_provenance_ok() -> None:
    table = _table(
        [_cell(0, 0, "x", x0=0.0, bbox_source="docling_native")], cell_provenance_ok=True
    )
    assert needs_vision_escalation(table) is False


# --------------------------------------------------------------------------- #
# escalate_table_to_vision -- resolve-or-flag, the actual trust boundary.
# --------------------------------------------------------------------------- #


def test_escalate_short_circuits_when_nothing_needs_resolving() -> None:
    # Regression: a table with cell_provenance_ok already True used to
    # trigger a vision_client call anyway (a real, billed Claude API call)
    # even though needs_vision_escalation would have said no. A caller that
    # forgets the gate must not still pay for the call.
    page = _page("100")
    already_resolved = _cell(0, 0, "100", x0=0.0, bbox_source="docling_native")
    table = _table([already_resolved], num_rows=1, num_cols=1, cell_provenance_ok=True)

    calls: list[bytes] = []

    def spy_vision_client(image_bytes: bytes) -> VisionTableReading:
        calls.append(image_bytes)
        return VisionTableReading(cells=[])

    result = escalate_table_to_vision(table, page, b"img", spy_vision_client)

    assert calls == []
    assert result == table


def test_escalate_resolves_unresolved_cell_via_matching_vision_guess() -> None:
    page = _page("Revenue 3,817 total")
    unresolved = _cell(1, 1, "3 ,817")  # verbatim source text kept; x0=None
    table = _table([unresolved], num_rows=2, num_cols=2, cell_provenance_ok=False)

    def fake_vision_client(image_bytes: bytes) -> VisionTableReading:
        return VisionTableReading(cells=[VisionTableCell(row=1, col=1, text="3,817")])

    result = escalate_table_to_vision(table, page, b"fake-image-bytes", fake_vision_client)

    resolved_cell = result.cells[0]
    assert resolved_cell.bbox_source == "vision_resolved"
    assert resolved_cell.x0 is not None
    assert resolved_cell.x1 is not None and resolved_cell.x1 > resolved_cell.x0
    assert result.cell_provenance_ok is True


def test_escalate_leaves_cell_unresolved_when_vision_guess_not_found_on_page() -> None:
    page = _page("Revenue 3,817 total")
    unresolved = _cell(1, 1, "whatever")
    table = _table([unresolved], num_rows=2, num_cols=2, cell_provenance_ok=False)

    def fake_vision_client(image_bytes: bytes) -> VisionTableReading:
        return VisionTableReading(cells=[VisionTableCell(row=1, col=1, text="not-on-page")])

    result = escalate_table_to_vision(table, page, b"img", fake_vision_client)

    assert result.cells[0].x0 is None
    assert result.cells[0].bbox_source is None
    assert result.cell_provenance_ok is False


def test_escalate_leaves_cell_unresolved_when_vision_guess_is_ambiguous() -> None:
    page = _page("$15,295 here and $15,295 again")
    unresolved = _cell(0, 0, "whatever")
    table = _table([unresolved], num_rows=1, num_cols=1, cell_provenance_ok=False)

    def fake_vision_client(image_bytes: bytes) -> VisionTableReading:
        return VisionTableReading(cells=[VisionTableCell(row=0, col=0, text="$15,295")])

    result = escalate_table_to_vision(table, page, b"img", fake_vision_client)

    assert result.cells[0].x0 is None
    assert result.cell_provenance_ok is False


def test_escalate_never_overwrites_a_cell_docling_already_resolved() -> None:
    page = _page("Revenue $99,999 total")
    already_resolved = _cell(0, 0, "$5,000", x0=42.0, bbox_source="docling_native")
    table = _table([already_resolved], num_rows=1, num_cols=1, cell_provenance_ok=True)

    def fake_vision_client(image_bytes: bytes) -> VisionTableReading:
        # Deliberately conflicting guess -- must be ignored: deterministic
        # geometry always outranks a vision reading.
        return VisionTableReading(cells=[VisionTableCell(row=0, col=0, text="$99,999")])

    result = escalate_table_to_vision(table, page, b"img", fake_vision_client)

    assert result.cells[0].x0 == 42.0
    assert result.cells[0].bbox_source == "docling_native"


def test_escalate_never_overwrites_a_reconstructed_cell() -> None:
    page = _page("Revenue $99,999 total")
    reconstructed = _cell(0, 0, "$5,000", x0=7.0, bbox_source="reconstructed")
    table = _table([reconstructed], num_rows=1, num_cols=1, cell_provenance_ok=True)

    def fake_vision_client(image_bytes: bytes) -> VisionTableReading:
        return VisionTableReading(cells=[VisionTableCell(row=0, col=0, text="$99,999")])

    result = escalate_table_to_vision(table, page, b"img", fake_vision_client)

    assert result.cells[0].x0 == 7.0
    assert result.cells[0].bbox_source == "reconstructed"


def test_escalate_mixed_outcomes_across_multiple_cells() -> None:
    page = _page("A 100 B not-found C")
    cell_a = _cell(0, 0, "100")  # unresolved, vision guess resolves
    cell_b = _cell(0, 1, "200")  # unresolved, vision guess does not resolve
    already = _cell(0, 2, "999", x0=1.0, bbox_source="docling_native")  # untouched
    table = _table([cell_a, cell_b, already], num_rows=1, num_cols=3, cell_provenance_ok=False)

    def fake_vision_client(image_bytes: bytes) -> VisionTableReading:
        return VisionTableReading(
            cells=[
                VisionTableCell(row=0, col=0, text="100"),
                VisionTableCell(row=0, col=1, text="does-not-exist"),
            ]
        )

    result = escalate_table_to_vision(table, page, b"img", fake_vision_client)

    resolved, unresolved, untouched = result.cells
    assert resolved.bbox_source == "vision_resolved"
    assert unresolved.x0 is None
    assert untouched.bbox_source == "docling_native"
    assert result.cell_provenance_ok is False  # one cell still unresolved


def test_escalate_all_resolved_sets_provenance_ok_true() -> None:
    page = _page("100 200")
    cell_a = _cell(0, 0, "100")
    cell_b = _cell(0, 1, "200")
    table = _table([cell_a, cell_b], num_rows=1, num_cols=2, cell_provenance_ok=False)

    def fake_vision_client(image_bytes: bytes) -> VisionTableReading:
        return VisionTableReading(
            cells=[
                VisionTableCell(row=0, col=0, text="100"),
                VisionTableCell(row=0, col=1, text="200"),
            ]
        )

    result = escalate_table_to_vision(table, page, b"img", fake_vision_client)

    assert result.cell_provenance_ok is True


def test_escalate_missing_guess_for_a_cell_leaves_it_unresolved() -> None:
    page = _page("100")
    cell_a = _cell(0, 0, "100")
    table = _table([cell_a], num_rows=1, num_cols=1, cell_provenance_ok=False)

    def fake_vision_client(image_bytes: bytes) -> VisionTableReading:
        return VisionTableReading(cells=[])  # vision proposed nothing for this cell

    result = escalate_table_to_vision(table, page, b"img", fake_vision_client)

    assert result.cells[0].x0 is None
    assert result.cell_provenance_ok is False


# --------------------------------------------------------------------------- #
# claude_vision_client -- the real integration, network mocked out.
# --------------------------------------------------------------------------- #


def test_claude_vision_client_sends_image_block_and_returns_parsed_output() -> None:
    fake_reading = VisionTableReading(cells=[VisionTableCell(row=0, col=0, text="42")])
    fake_response = SimpleNamespace(parsed_output=fake_reading, stop_reason="end_turn")

    mock_client_instance = MagicMock()
    mock_client_instance.messages.parse.return_value = fake_response

    with patch("anthropic.Anthropic", return_value=mock_client_instance) as mock_anthropic:
        result = claude_vision_client(b"raw-image-bytes")

    assert result is fake_reading
    mock_anthropic.assert_called_once()

    _, kwargs = mock_client_instance.messages.parse.call_args
    assert kwargs["model"] == "claude-opus-4-8"
    assert kwargs["output_format"] is VisionTableReading

    content = kwargs["messages"][0]["content"]
    image_block = next(b for b in content if b["type"] == "image")
    assert image_block["source"]["type"] == "base64"
    assert image_block["source"]["media_type"] == "image/png"
    assert base64.standard_b64decode(image_block["source"]["data"]) == b"raw-image-bytes"


def test_claude_vision_client_raises_when_no_parsed_output() -> None:
    fake_response = SimpleNamespace(parsed_output=None, stop_reason="refusal")
    mock_client_instance = MagicMock()
    mock_client_instance.messages.parse.return_value = fake_response

    with (
        patch("anthropic.Anthropic", return_value=mock_client_instance),
        pytest.raises(ValueError, match="refusal"),
    ):
        claude_vision_client(b"img")

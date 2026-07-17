"""DS-W3-8 visual inspection harness tests.

Two layers, matching the rest of the parser suite:
- Fast, CI-portable tests against small synthetic PDFs generated in-process
  via reportlab (same convention as test_pdf_parser.py's make_text_pdf) --
  Docling parses these reliably in this environment; only larger multi-page
  real corpora hit the known memory-constrained local crash.
- A local_corpus-marked acceptance test against the real PTL PDF-page 11 (the
  ticket's own acceptance example: every income-statement figure green, no
  red boxes on real values).
"""

from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path
from typing import cast

import pytest
from PIL import Image
from PIL.Image import Image as PILImage
from reportlab.pdfgen import canvas

from services.parser.parser_service.inspect import (
    BoxAnnotation,
    classify_token,
    detect_numeric_tokens,
    inspect_pdf,
    is_boilerplate_token,
    main,
    render_page_overlay,
)
from services.parser.parser_service.schemas import BBox, CharBox, PageIndex


def make_pdf(lines: list[str]) -> bytes:
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=(612, 792))
    y = 760
    for line in lines:
        pdf.drawString(72, y, line)
        y -= 24
    pdf.save()
    return buffer.getvalue()


# --------------------------------------------------------------------------- #
# detect_numeric_tokens
# --------------------------------------------------------------------------- #


def test_detects_dollar_amount() -> None:
    tokens = detect_numeric_tokens("Revenue $15,295 total")
    assert [t.text for t in tokens] == ["$15,295"]
    assert tokens[0].value_type == "currency"


def test_detects_percent() -> None:
    tokens = detect_numeric_tokens("Gross margin 27.3% this year")
    assert [t.text for t in tokens] == ["27.3%"]
    assert tokens[0].value_type == "percent"


def test_detects_thousands_bare_number() -> None:
    # No "$" but has thousands-comma grouping -- still financially meaningful.
    tokens = detect_numeric_tokens("Gross Margin 3,817 thousand")
    assert [t.text for t in tokens] == ["3,817"]
    assert tokens[0].value_type == "currency"


def test_ignores_bare_small_integers_and_years() -> None:
    # Page numbers, years, footnote markers: no $/%/comma/decimal/suffix.
    tokens = detect_numeric_tokens("See page 3 of the 2026 annual report, note 1")
    assert tokens == []


def test_detects_accounting_negative_currency() -> None:
    tokens = detect_numeric_tokens("Net loss ($15,295) this year")
    assert [t.text for t in tokens] == ["($15,295)"]


def test_detects_suffix_scaled_currency() -> None:
    tokens = detect_numeric_tokens("Valuation of $4.8M implied")
    assert [t.text for t in tokens] == ["$4.8M"]


def test_no_overlapping_tokens_for_dollar_prefixed_thousands() -> None:
    # "$15,295" must be one token, not "$15,295" plus a separately-matched
    # "15,295" substring.
    tokens = detect_numeric_tokens("Total: $15,295 and 27.3% margin")
    assert len(tokens) == 2
    assert tokens[0].text == "$15,295"
    assert tokens[1].text == "27.3%"


def test_detects_decimal_without_dollar_or_percent() -> None:
    tokens = detect_numeric_tokens("Leverage ratio of 1.8 times EBITDA")
    assert [t.text for t in tokens] == ["1.8"]


# --------------------------------------------------------------------------- #
# classify_token
#
# These build synthetic PageIndex fixtures directly (same convention as
# test_resolver.py's make_page) rather than running real pdf bytes through
# Docling -- classify_token's logic only cares about PageIndex.text/char_map,
# not where they came from, and this keeps the fast tier from repeatedly
# invoking Docling's heavy ML layout model (real-PDF Docling runs are reserved
# for the render_page_overlay/inspect_pdf/main and local_corpus tests below).
# --------------------------------------------------------------------------- #


def make_page(text: str, page_no: int = 1) -> PageIndex:
    char_map: list[CharBox] = []
    x = 0.0
    top = 0.0
    line_height = 10.0
    for ch in text:
        if ch == "\n":
            char_map.append(
                CharBox(
                    char="\n",
                    x0=x,
                    top=top,
                    x1=x,
                    bottom=top + line_height,
                    page=page_no,
                    precision="word",
                )
            )
            x = 0.0
            top += 20.0
        else:
            char_map.append(
                CharBox(
                    char=ch,
                    x0=x,
                    top=top,
                    x1=x + 5.0,
                    bottom=top + line_height,
                    page=page_no,
                    precision="word",
                )
            )
            x += 5.0
    return PageIndex(page=page_no, text=text, char_map=char_map)


def test_boilerplate_numbers_are_excluded_not_boxed_red() -> None:
    # Acceptance: page furniture is not a claim, so a number inside it is not a
    # miss. The shape heuristic already drops a bare page number; this covers
    # what it cannot -- furniture that happens to look financial. A running
    # total in a footer boxed red would report a recall gap that does not exist.
    page = make_page("Confidential -- do not distribute $9,999")
    for char in page.char_map:
        char.is_boilerplate = True

    tokens = detect_numeric_tokens(page.text)
    assert [t.text for t in tokens] == ["$9,999"], "the token must still be detected"
    assert all(is_boilerplate_token(t, page) for t in tokens)


def test_real_figures_are_not_mistaken_for_boilerplate() -> None:
    # The mirror of the above, and the more important direction: over-excluding
    # would hide a real miss, which is the one failure this harness must not
    # have. Identical token, ordinary body text, must not be excluded.
    page = make_page("Revenue $9,999")
    tokens = detect_numeric_tokens(page.text)
    assert [t.text for t in tokens] == ["$9,999"]
    assert not any(is_boilerplate_token(t, page) for t in tokens)


def test_bare_table_figure_is_scaled_like_its_dollar_marked_row() -> None:
    # PTL PDF-page 11 prints "Revenue $15,295" but "Gross Margin 4,171" -- money
    # either way, and 4,171 means 4,171,000 under the CAD (in Thousands) header.
    # Typing an unmarked token as `count` would refuse the scale and show an
    # income-statement figure a thousand times too small, in a green box.
    page = make_page("CAD (in Thousands)\nRevenue $15,295\nGross Margin 4,171")

    marked = next(t for t in detect_numeric_tokens(page.text) if t.text == "$15,295")
    bare = next(t for t in detect_numeric_tokens(page.text) if t.text == "4,171")
    assert bare.value_type == "currency"

    assert classify_token(marked, page).label.startswith("15,295,000")
    assert classify_token(bare, page).label.startswith("4,171,000")


def test_classify_green_when_scale_confidently_resolved() -> None:
    page = make_page("CAD (in Thousands)\nRevenue $15,295 total")
    token = detect_numeric_tokens(page.text)[0]
    assert token.text == "$15,295"

    ann = classify_token(token, page)

    assert ann.color == "green"
    assert "15,295,000" in ann.label
    assert "page_header" in ann.label


def test_classify_yellow_when_scale_assumed() -> None:
    page = make_page("Unscaled figure $4,000 here, no header anywhere")
    token = detect_numeric_tokens(page.text)[0]

    ann = classify_token(token, page)

    assert ann.color == "yellow"
    assert "assumed_1x" in ann.label


def test_classify_red_when_ambiguous() -> None:
    page = make_page("Ambiguous $9,999 value appears twice: $9,999 again")
    tokens = detect_numeric_tokens(page.text)
    assert len(tokens) == 2

    ann = classify_token(tokens[0], page)

    assert ann.color == "red"
    assert "MISS" in ann.label
    # The box still has a real, correctly-positioned bbox even though the
    # citation itself failed -- drawn from the token's own known offsets.
    assert ann.bbox.x1 > ann.bbox.x0


def test_classify_percent_is_always_explicit_in_value() -> None:
    page = make_page("Gross margin 27.3% this year")
    token = detect_numeric_tokens(page.text)[0]

    ann = classify_token(token, page)

    assert ann.color == "green"
    assert "explicit_in_value" in ann.label


def test_classify_red_box_position_matches_resolver_span_when_available() -> None:
    # For a genuinely ambiguous token, the red box's bbox still comes from
    # union_bbox over the token's own char_map slice (not a resolved Span,
    # since resolve() returned None) -- confirm it lines up with the first
    # occurrence's real position on the page, not (0, 0).
    page = make_page("Value $9,999 here and $9,999 there")
    token = detect_numeric_tokens(page.text)[0]

    ann = classify_token(token, page)

    expected_chars = page.char_map[token.start : token.end]
    assert ann.bbox.x0 == min(c.x0 for c in expected_chars)
    assert ann.bbox.x1 == max(c.x1 for c in expected_chars)


# --------------------------------------------------------------------------- #
# render_page_overlay / inspect_pdf
# --------------------------------------------------------------------------- #


def test_render_page_overlay_produces_image_at_requested_scale() -> None:
    pdf_bytes = make_pdf(["Revenue $15,295 total"])
    ann = BoxAnnotation(
        bbox=BBox(x0=100, top=20, x1=160, bottom=35, page=1), color="green", label="15,295,000"
    )

    image = render_page_overlay(pdf_bytes, 1, [ann], scale=2.0)

    # US-letter page (612x792 pts) at scale=2.0 -> 1224x1584 px.
    assert image.size == (1224, 1584)


def test_render_page_overlay_draws_the_correct_color_pixel() -> None:
    pdf_bytes = make_pdf(["Revenue $15,295 total"])
    ann = BoxAnnotation(
        bbox=BBox(x0=100, top=20, x1=160, bottom=35, page=1), color="green", label="x"
    )

    image = render_page_overlay(pdf_bytes, 1, [ann], scale=2.0)

    # The box's left edge, at its vertical midpoint, should be the green outline.
    x = round(100 * 2.0)
    y = round((20 + 35) / 2 * 2.0)
    pixel = image.getpixel((x, y))
    assert pixel == (0, 153, 0)


def test_inspect_pdf_writes_one_png_per_page(tmp_path: Path) -> None:
    pdf_bytes = make_pdf(["CAD (in Thousands)", "Revenue $15,295 total"])

    written = inspect_pdf(pdf_bytes, "demo", tmp_path)

    assert written == [tmp_path / "demo_page_01.png"]
    assert written[0].exists()


def _colors_in(image: PILImage) -> set[tuple[int, int, int]]:
    # render_page_overlay always produces "RGB"-mode images, so every color
    # here is genuinely a 3-int tuple; PIL's stub is generic over image mode.
    colors = image.getcolors(maxcolors=2_000_000)
    assert colors is not None, "test image has more distinct colors than maxcolors"
    return {cast("tuple[int, int, int]", color) for _, color in colors}


def test_inspect_pdf_ambiguous_value_renders_as_red_miss_not_silent(tmp_path: Path) -> None:
    # The ticket's acceptance criterion: "A deliberately dropped fact shows
    # as a red box (miss is visible, not silent)".
    pdf_bytes = make_pdf(["Duplicate $9,999 figure appears twice: $9,999 again"])

    written = inspect_pdf(pdf_bytes, "miss-demo", tmp_path)
    image = Image.open(written[0])
    assert (220, 0, 0) in _colors_in(image)  # the red outline color must appear somewhere.


def test_inspect_pdf_assumed_1x_renders_yellow_distinct_from_green(tmp_path: Path) -> None:
    pdf_bytes = make_pdf(["No header on this page.", "Unscaled figure $4,000 here"])

    written = inspect_pdf(pdf_bytes, "yellow-demo", tmp_path)
    image = Image.open(written[0])
    colors = _colors_in(image)
    assert (204, 153, 0) in colors  # yellow present
    assert (0, 153, 0) not in colors  # and no green box on this page


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def test_main_cli_writes_overlay_and_prints_path(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(make_pdf(["Revenue $15,295 total"]))
    out_dir = tmp_path / "out"

    main([str(pdf_path), "--out-dir", str(out_dir)])

    captured = capsys.readouterr()
    expected = out_dir / "sample_page_01.png"
    assert expected.exists()
    assert str(expected) in captured.out


# --------------------------------------------------------------------------- #
# Real-corpus acceptance (the ticket's DS-W3-8 acceptance example).
# --------------------------------------------------------------------------- #


def _ptl_pdf_path() -> Path | None:
    root = os.environ.get("PARSER_LOCAL_CORPUS_DIR")
    if not root:
        return None
    path = Path(root) / "1st-app-h-ptl/1st-App-H-PTL-Group-CIM.pdf"
    return path if path.exists() else None


@pytest.mark.local_corpus
def test_ptl_page_11_renders_green_no_red_on_verified_values(tmp_path: Path) -> None:
    pdf_path = _ptl_pdf_path()
    if not pdf_path:
        pytest.skip(
            "Real PTL CIM not available on this machine (confidential document, "
            "never committed to this repo)."
        )
    from services.parser.parser_service.docling_parser import parse_pdf_bytes

    result = parse_pdf_bytes(pdf_path.read_bytes())
    page = result.pages[10]  # PDF page index 11 (0-indexed 10): income statement.

    for value in ["$15,295", "27.3%", "3,817"]:
        assert page.text.count(value) == 1, f"expected {value!r} to be unique on PDF-page 11"
        token = next(t for t in detect_numeric_tokens(page.text) if t.text == value)
        ann = classify_token(token, page)
        assert ann.color != "red", f"{value!r} should not be a miss"

    written = inspect_pdf(pdf_path.read_bytes(), "ptl", tmp_path)
    assert any(p.name == "ptl_page_11.png" for p in written)

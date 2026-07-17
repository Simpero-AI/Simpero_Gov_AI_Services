"""DS-W3-8 visual inspection harness -- validation by inspection, with misses
made visible.

For the prototype/build phase, parsing is validated by a human looking at
rendered pages, not a full automated eval harness (that's a Pilot
deliverable). Plain inspection has a blind spot: it confirms what you look
at, not what got dropped. This module renders each page with every detected
numeric value boxed and color-coded, so a reviewer catches both wrong
citations (precision) and dropped claims (recall) at a glance:

    green  = resolved to an exact span (DS-W3-3) with a confidently captured
             scale (DS-W3-4) -- no flags.
    yellow = resolved, but the scale carries a flag (assumed_1x or otherwise
             flagged) -- still cited, but the citation says so honestly.
    red    = a numeric-looking token detected on the page that produced NO
             citable claim -- most commonly DS-W3-3's ambiguity fail-close (the
             same literal value appears more than once on the page, so which
             instance is unknowable). The miss signal.

Detection scope, deliberately: this harness does not know any real claim's
entity/attribute/value_type (that's supplied by whatever extraction step
identifies a specific claim candidate) -- it sweeps the page's own text for
numeric-looking tokens and classifies each one through the same DS-W3-3/DS-4
machinery a real claim would use, inferring value_type from the token's own
shape (see `_infer_value_type`, which guesses conservatively because DS-W3-4
only header-scales currency).

A bare integer with no "$", no thousands grouping, no decimal point, and no
M/B/K suffix (a page number, a year, a footnote marker) is treated as noise
and left unboxed -- boxing every stray digit on a page would bury the real
misses in noise, which is exactly what this tool exists to prevent. Numbers
inside furniture DS-W3-1 tagged `is_boilerplate` (running headers/footers,
confidentiality banners) are excluded for the same reason: they are not claims,
so they are not misses.

Run as: `uv run python -m services.parser.parser_service.inspect <cim.pdf>`
Writes `<out_dir>/<doc>_page_NN.png` per page (default out_dir: the `out/`
directory next to this module).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Literal

import pypdfium2 as pdfium
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel

from .docling_parser import parse_pdf_bytes
from .resolver import resolve, union_bbox
from .scale import ValueType, determine_scale
from .schemas import BBox, PageIndex

Color = Literal["green", "yellow", "red"]

_COLOR_RGB: dict[Color, tuple[int, int, int]] = {
    "green": (0, 153, 0),
    "yellow": (204, 153, 0),
    "red": (220, 0, 0),
}

DEFAULT_RENDER_SCALE = 2.0  # pixels per PDF point (2.0 -> 144 DPI).
_OUT_DIR = Path(__file__).parent / "out"

# Any digit run, with its optional sign/currency/scale-suffix/percent
# trimmings captured so callers can decide which of them made the token
# "financially meaningful" (see `detect_numeric_tokens`). Thousands commas
# are matched inline (`\d[\d,]*`), not as a separate group, so "3,817"
# resolves as one token rather than three.
_NUMERIC_TOKEN_RE = re.compile(
    r"(?P<sign>[-(])?"
    r"(?P<dollar>\$)?"
    r"(?P<number>\d[\d,]*(?:\.\d+)?)"
    r"(?P<suffix>\s?(?:[MmBb](?:illion|M|m|n|N)?|[Kk](?:thousand)?))?"
    r"(?P<percent>%)?"
    r"(?P<close>\))?"
)


class NumericToken(BaseModel):
    start: int
    end: int
    text: str
    value_type: ValueType


def detect_numeric_tokens(text: str) -> list[NumericToken]:
    """Every financially-meaningful numeric span in `text`, left to right,
    non-overlapping. "Financially meaningful" means the match carries a "$",
    a "%", an M/B/K scale suffix, a decimal point, or thousands-comma
    grouping -- a bare small integer (a page number, a year, a footnote
    marker) has none of these and is skipped."""
    tokens: list[NumericToken] = []
    last_end = -1
    for match in _NUMERIC_TOKEN_RE.finditer(text):
        if match.start() < last_end:
            continue  # overlaps the previous (longer, earlier) token.
        has_dollar = match.group("dollar") is not None
        has_suffix = match.group("suffix") is not None
        has_percent = match.group("percent") is not None
        number = match.group("number")
        meaningful = has_dollar or has_suffix or has_percent or "," in number or "." in number
        if not meaningful:
            continue
        # Everything that isn't a percent is read as currency, including bare
        # tokens with no "$". That looks too eager -- DS-W3-4 only header-scales
        # currency, so this is what lets "(in Thousands)" multiply a bare number
        # by 1000 -- but the corpus is unambiguous. PTL PDF-page 11:
        #     CAD (in Thousands)
        #     Revenue       $15,295  $17,146  $18,003  $18,903
        #     Gross Margin    4,171    3,631    3,817    4,007
        # Only Revenue carries "$". Gross Margin is money too, and 3,817 means
        # 3,817,000 there. Typing bare tokens as `count` would refuse the scale
        # and box an income-statement figure green at a thousandth of its value
        # -- the silent miss this tool exists to surface. The cost is that a real
        # headcount under a scale header reads as 3,817,000, which is loud enough
        # for a reviewer to catch. Only the overlay guesses: emit.py takes
        # value_type from its caller and never infers.
        value_type: ValueType = "percent" if has_percent else "currency"
        tokens.append(
            NumericToken(
                start=match.start(), end=match.end(), text=match.group(0), value_type=value_type
            )
        )
        last_end = match.end()
    return tokens


def is_boilerplate_token(token: NumericToken, page: PageIndex) -> bool:
    """True when the whole token sits in page furniture DS-W3-1 tagged.

    Running headers/footers, page-number furniture and confidentiality banners
    are not claims, so a number inside one is not a miss -- boxing it red would
    report a recall gap that does not exist, and a tool whose red boxes cry wolf
    stops being read. The shape heuristic in `detect_numeric_tokens` already
    drops most of this (a bare page number has no $, comma or decimal), but it
    cannot catch furniture that happens to look financial -- a running total in
    a footer. DS-W3-1 already knows which chars are furniture; this asks it.

    Requires EVERY char to be boilerplate, deliberately. Over-excluding would
    hide a real miss, which is the one failure this harness must not have.
    """
    chars = page.char_map[token.start : token.end]
    return bool(chars) and all(char.is_boilerplate for char in chars)


class BoxAnnotation(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    bbox: BBox
    color: Color
    label: str


def classify_token(token: NumericToken, page: PageIndex) -> BoxAnnotation:
    """Run one detected token through the real DS-W3-3/DS-W3-4 machinery and
    return the box + label a reviewer sees. Never raises: an unexpected
    scale-resolution failure is surfaced as a red box with the error message
    rather than aborting the whole page -- a debug tool that crashes on one
    bad token hides every miss after it, the opposite of this ticket's goal.
    """
    span = resolve(token.text, page)
    if span is None:
        # Most commonly DS-W3-3's ambiguity fail-close: this exact literal
        # value appears more than once on the page, so no single instance is
        # safely citable. The box still needs a real position, so it's drawn
        # from the token's own known offsets rather than a resolved Span.
        chars = page.char_map[token.start : token.end]
        return BoxAnnotation(bbox=union_bbox(chars), color="red", label="MISS (unresolved)")

    try:
        scale_result = determine_scale(
            token.text, page, span.char_start, value_type=token.value_type
        )
    except ValueError as exc:
        return BoxAnnotation(bbox=span.bbox, color="red", label=f"MISS ({exc})")

    color: Color = "yellow" if scale_result.flags else "green"
    if token.value_type == "percent":
        label = f"{scale_result.normalized:g}% {scale_result.scale_source}"
    else:
        label = f"{scale_result.normalized:,.0f} {scale_result.scale_source}"
    return BoxAnnotation(bbox=span.bbox, color=color, label=label)


def render_page_overlay(
    pdf_bytes: bytes,
    page_no: int,
    annotations: list[BoxAnnotation],
    *,
    scale: float = DEFAULT_RENDER_SCALE,
) -> Image.Image:
    """Rasterize `page_no` (1-indexed) and draw every annotation's box +
    label. Docling's bbox coordinates are TOPLEFT-origin PDF points, which
    line up directly with a pypdfium2-rendered raster under a flat
    `pixel = point * scale` -- no axis flip (verified empirically)."""
    pdf = pdfium.PdfDocument(pdf_bytes)
    try:
        page = pdf[page_no - 1]
        # pypdfium2's stub types `scale` as int; the real implementation
        # accepts (and this module relies on) a float scale -- verified
        # empirically against a real render.
        bitmap = page.render(scale=scale)  # type: ignore[arg-type]
        image = bitmap.to_pil().convert("RGB")
    finally:
        pdf.close()

    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=14)
    for ann in annotations:
        rgb = _COLOR_RGB[ann.color]
        box = ann.bbox
        coords = (box.x0 * scale, box.top * scale, box.x1 * scale, box.bottom * scale)
        draw.rectangle(coords, outline=rgb, width=2)
        draw.text((coords[0], coords[3] + 2), ann.label, fill=rgb, font=font)
    return image


def inspect_pdf(
    pdf_bytes: bytes, doc_name: str, out_dir: Path, *, scale: float = DEFAULT_RENDER_SCALE
) -> list[Path]:
    """Parse `pdf_bytes`, render every page with its detected-number overlay,
    and write `<out_dir>/<doc_name>_page_NN.png`. Returns the written paths."""
    result = parse_pdf_bytes(pdf_bytes)
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for page in result.pages:
        tokens = [
            token
            for token in detect_numeric_tokens(page.text)
            if not is_boilerplate_token(token, page)
        ]
        annotations = [classify_token(token, page) for token in tokens]
        image = render_page_overlay(pdf_bytes, page.page, annotations, scale=scale)
        out_path = out_dir / f"{doc_name}_page_{page.page:02d}.png"
        image.save(out_path)
        written.append(out_path)
    return written


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf_path", type=Path, help="Path to the CIM PDF to inspect.")
    parser.add_argument(
        "--out-dir", type=Path, default=_OUT_DIR, help="Directory to write page overlays into."
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=DEFAULT_RENDER_SCALE,
        help="Render scale (pixels per PDF point).",
    )
    args = parser.parse_args(argv)

    pdf_bytes = args.pdf_path.read_bytes()
    written = inspect_pdf(pdf_bytes, args.pdf_path.stem, args.out_dir, scale=args.scale)
    for path in written:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()

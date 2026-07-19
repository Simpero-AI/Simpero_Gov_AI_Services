"""Non-table text block extraction.

Fast synthetic-fixture tests, in the same convention as test_table_extract.py, plus a
local_corpus-marked run over a real CIM.

What is deliberately NOT tested here: whether a block is worth reading, or what it
asserts. This module surfaces what Docling found and decides nothing else. What IS tested
is that it never reports a coordinate it did not derive, never silently drops a block, and
converts the coordinate origin -- the one error here that would look plausible rather than
obviously wrong.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from docling_core.types.doc.base import BoundingBox, CoordOrigin

from parser_service.schemas import CharBox, PageIndex
from parser_service.text_extract import blocks_on_page, extract_text_blocks

PAGE_HEIGHT = 792.0


def _page_index(text: str, page_no: int = 1) -> PageIndex:
    char_map: list[CharBox] = []
    x = 0.0
    for character in text:
        char_map.append(
            CharBox(
                char=character,
                x0=x,
                top=100.0,
                x1=x + 5.0,
                bottom=110.0,
                page=page_no,
                precision="word",
            )
        )
        x += 5.0
    return PageIndex(page=page_no, text=text, char_map=char_map)


def _text_item(text: str, *, label: str = "text", bbox: BoundingBox | None, page_no: int = 1):
    """A Docling TextItem stand-in: only prov/text/label are read."""
    prov = [SimpleNamespace(page_no=page_no, bbox=bbox)] if bbox is not None or page_no else []
    return SimpleNamespace(text=text, label=label, prov=prov)


def _doc(items: list, *, page_height: float | None = PAGE_HEIGHT):
    size = SimpleNamespace(height=page_height) if page_height is not None else None
    return cast(
        "object",
        SimpleNamespace(texts=items, pages={1: SimpleNamespace(size=size)}),
    )


def _bottom_left(left: float, top_from_bottom: float, right: float, bottom_from_bottom: float):
    return BoundingBox(
        l=left, t=top_from_bottom, r=right, b=bottom_from_bottom, coord_origin=CoordOrigin.BOTTOMLEFT
    )


# --------------------------------------------------------------------------- #
# Coordinate origin -- the failure that would not look like one.
# --------------------------------------------------------------------------- #


def test_a_prov_box_is_converted_to_the_origin_the_citation_surface_uses() -> None:
    # Docling reports a text item's prov bbox BOTTOM-LEFT, while char_map -- what every
    # citation is measured against -- is TOP-LEFT. A heading 112pt from the top of a
    # 792pt page arrives as t=680. Storing that verbatim puts the citation in the wrong
    # half of the page at a coordinate well inside the plausible range.
    item = _text_item("Investment Considerations", bbox=_bottom_left(94.0, 680.0, 285.0, 667.0))
    blocks = extract_text_blocks(cast("object", _doc([item])))  # pyright: ignore[reportArgumentType]

    assert len(blocks) == 1
    block = blocks[0]
    assert block.top == pytest.approx(PAGE_HEIGHT - 680.0)
    assert block.bottom == pytest.approx(PAGE_HEIGHT - 667.0)
    assert block.top < block.bottom, "top-left origin: top is above bottom"
    assert block.bbox_source == "docling_native"


def test_a_block_keeps_its_page_and_document_order() -> None:
    items = [
        _text_item("First", label="section_header", bbox=_bottom_left(0, 700, 100, 690)),
        _text_item("Second", bbox=_bottom_left(0, 680, 100, 670)),
        _text_item("Third", bbox=_bottom_left(0, 660, 100, 650)),
    ]
    blocks = extract_text_blocks(cast("object", _doc(items)))  # pyright: ignore[reportArgumentType]

    assert [b.order for b in blocks] == [0, 1, 2]
    assert [b.text for b in blocks] == ["First", "Second", "Third"]
    assert blocks[0].label == "section_header"


# --------------------------------------------------------------------------- #
# Provenance: recovered, or absent -- never invented.
# --------------------------------------------------------------------------- #


def test_a_block_with_no_usable_box_is_recovered_from_the_page_index() -> None:
    # No page height means to_top_left_origin cannot be trusted, so the native path is
    # refused and the text is resolved against the positioned index instead.
    item = _text_item("Revenue 3,817 total", bbox=_bottom_left(0, 700, 100, 690))
    blocks = extract_text_blocks(
        cast("object", _doc([item], page_height=None)),  # pyright: ignore[reportArgumentType]
        [_page_index("Revenue 3,817 total")],
    )

    assert blocks[0].bbox_source == "reconstructed"
    assert blocks[0].top == pytest.approx(100.0)


def test_a_block_that_cannot_be_located_is_kept_without_a_coordinate() -> None:
    # It must not vanish: a dropped block is invisible, an uncitable one is a gap you
    # can see. And it must not carry a fabricated box.
    item = _text_item("nowhere on the page", bbox=_bottom_left(0, 700, 100, 690))
    blocks = extract_text_blocks(
        cast("object", _doc([item], page_height=None)),  # pyright: ignore[reportArgumentType]
        [_page_index("entirely different text")],
    )

    assert len(blocks) == 1
    assert blocks[0].bbox_source is None
    assert (blocks[0].x0, blocks[0].top, blocks[0].x1, blocks[0].bottom) == (None, None, None, None)


def test_an_ambiguous_block_is_not_located_by_guessing() -> None:
    # resolve() fails closed on a repeated quote, so the fallback must too.
    item = _text_item("3,817", bbox=_bottom_left(0, 700, 100, 690))
    blocks = extract_text_blocks(
        cast("object", _doc([item], page_height=None)),  # pyright: ignore[reportArgumentType]
        [_page_index("3,817 versus 3,817 again")],
    )
    assert blocks[0].bbox_source is None


def test_split_numeric_tokens_are_normalized_like_table_cells() -> None:
    item = _text_item("Revenue of 3 ,817 this year", bbox=_bottom_left(0, 700, 100, 690))
    blocks = extract_text_blocks(cast("object", _doc([item])))  # pyright: ignore[reportArgumentType]

    assert blocks[0].text == "Revenue of 3 ,817 this year", "verbatim text is preserved"
    assert blocks[0].text_normalized == "Revenue of 3,817 this year"


def test_an_item_with_no_provenance_is_skipped() -> None:
    # Every record must carry a real page number; an item with no prov has no page to
    # cite it on.
    orphan = SimpleNamespace(text="floating", label="text", prov=[])
    good = _text_item("anchored", bbox=_bottom_left(0, 700, 100, 690))
    blocks = extract_text_blocks(cast("object", _doc([orphan, good])))  # pyright: ignore[reportArgumentType]

    assert [b.text for b in blocks] == ["anchored"]


def test_blocks_on_page_filters_by_page() -> None:
    items = [
        _text_item("one", bbox=_bottom_left(0, 700, 100, 690), page_no=1),
        _text_item("two", bbox=_bottom_left(0, 700, 100, 690), page_no=2),
    ]
    doc = cast("object", SimpleNamespace(texts=items, pages={1: SimpleNamespace(size=None)}))
    blocks = extract_text_blocks(doc)  # pyright: ignore[reportArgumentType]
    assert [b.text for b in blocks_on_page(blocks, 1)] == ["one"]


# --------------------------------------------------------------------------- #
# Real corpus.
# --------------------------------------------------------------------------- #


def _corpus_pdf() -> Path | None:
    root = os.environ.get("PARSER_LOCAL_CORPUS_DIR")
    if not root:
        return None
    path = Path(root) / "1st-App-H-PTL-Group-CIM.pdf"
    return path if path.exists() else None


@pytest.mark.local_corpus
def test_real_cim_text_blocks_are_citable_and_top_left() -> None:
    pdf_path = _corpus_pdf()
    if pdf_path is None:
        pytest.skip("PARSER_LOCAL_CORPUS_DIR not set or the CIM is not present")

    from parser_service.docling_parser import parse_pdf_bytes

    result = parse_pdf_bytes(pdf_path.read_bytes())
    assert result.document is not None
    blocks = extract_text_blocks(result.document, result.pages)

    assert blocks, "a real CIM has non-table text"
    located = [b for b in blocks if b.bbox_source is not None]
    assert located, "at least some blocks must be citable"
    for block in located:
        assert block.top is not None and block.bottom is not None
        assert block.top < block.bottom, f"top-left origin violated on page {block.page}"
        assert 0 <= block.top, "a top-left coordinate is never negative"

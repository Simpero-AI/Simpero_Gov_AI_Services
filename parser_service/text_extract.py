"""Non-table text blocks from a parsed DoclingDocument, with citable coordinates.

The claim path reads doc.tables and nothing else. Measured against reference sets over
two real CIMs, that leaves 44% and 59% of their facts unreachable -- not badly extracted,
never visited. Docling already separates the rest of the page, and the parse result
discarded it: on a 102-page CIM it reports 1,426 text items across 78 pages, against the
19 pages that carry a table at all.

This module is the smallest honest thing that makes those blocks available. It surfaces
what Docling found, gives each block a top-left box comparable with the citation surface,
and decides nothing else. It does NOT judge which blocks are worth reading, attach a
section heading, or extract a claim -- a sentence carries no row label and no column
header, so deciding what it asserts is a different problem with a different design.

WHAT IS DELIBERATELY NOT TRUSTED
================================
Docling's per-item `label` is carried through as advisory context and nothing more. This
codebase has twice found those labels unreliable on real CIMs and replaced them with its
own inference: DS-W3-2 ignores `column_header` because it marks data rows on tables with
unlabeled columns, and tag_boilerplate ignores `page_header`/`page_footer` because they
are not reliably assigned. Re-measured here, `page_header` was assigned to a reproduced
press clipping in an appendix. Any consumer that gates behaviour on a label class should
measure that class first.
"""

import logging

from docling_core.types.doc.document import (
    DoclingDocument,
    TextItem,  # pyright: ignore[reportPrivateImportUsage]
)

from .normalize import normalize_numeric_text
from .schemas import PageIndex, TextBlockRecord
from .table_extract import bbox_is_valid, reconstruct_bbox

logger = logging.getLogger(__name__)


def _top_left_box(
    item: TextItem, page_height: float | None
) -> tuple[float, float, float, float] | None:
    """The item's box in TOP-LEFT origin, or None when Docling gave no usable one.

    A text item's prov bbox is reported in BOTTOM-LEFT origin, while the table CELL
    boxes this codebase already stores are top-left, and PageIndex.char_map -- the
    surface every citation is measured against -- is top-left too. Assigning prov.t
    straight to `top`, as the table path legitimately does for cells, would put a
    prose citation in the wrong half of the page at a coordinate plausible enough to
    survive review: on a 792pt page, a heading 112pt from the top reports as 680.

    Docling's own to_top_left_origin does the conversion, so the page height it needs
    is the only thing that has to be right here.
    """
    if not item.prov:
        return None
    bbox = item.prov[0].bbox
    if bbox is None or page_height is None:
        return None
    box = bbox.to_top_left_origin(page_height)
    if not bbox_is_valid(box):
        return None
    return (box.l, box.t, box.r, box.b)


def extract_text_blocks(
    doc: DoclingDocument, page_indices: list[PageIndex] | None = None
) -> list[TextBlockRecord]:
    """Every non-table text item in `doc`, in document body order.

    Consumes the in-memory document (DoclingParseResult.document), like
    extract_tables, so this runs in the same request as the parse.

    When page_indices are supplied, a block Docling gives no usable box is recovered
    from that page's positioned index by exact-span match, the same fallback the table
    path uses -- and it fails closed the same way, leaving bbox_source None rather than
    inventing a coordinate.
    """
    by_page = {index.page: index for index in (page_indices or [])}
    heights = {
        number: page.size.height if page.size else None for number, page in doc.pages.items()
    }

    blocks: list[TextBlockRecord] = []
    uncitable = 0
    reconstructed = 0

    for order, item in enumerate(doc.texts):
        if not item.prov:
            # No provenance at all means no page to cite it on. Skipping keeps the
            # invariant that every record here carries a page number that is real.
            continue
        page_no = item.prov[0].page_no
        text = item.text or ""
        text_normalized = normalize_numeric_text(text)

        box = _top_left_box(item, heights.get(page_no))
        if box is not None:
            x0, top, x1, bottom = box
            source = "docling_native"
        else:
            page_index = by_page.get(page_no)
            rebuilt = reconstruct_bbox(text_normalized, page_index) if page_index else None
            if rebuilt is not None:
                x0, top, x1, bottom = rebuilt
                source = "reconstructed"
                reconstructed += 1
            else:
                x0 = top = x1 = bottom = None
                source = None
                uncitable += 1

        blocks.append(
            TextBlockRecord(
                page=page_no,
                order=order,
                label=str(getattr(item, "label", "text")),
                text=text,
                text_normalized=text_normalized,
                x0=x0,
                top=top,
                x1=x1,
                bottom=bottom,
                bbox_source=source,
            )
        )

    # Provenance is audit-critical, so say where geometry came from rather than
    # letting a recovered or absent box pass as native.
    if reconstructed:
        logger.info(
            "text blocks: recovered %d of %d box(es) via the reconstruction fallback",
            reconstructed,
            len(blocks),
        )
    if uncitable:
        logger.warning(
            "text blocks: %d of %d have no resolvable coordinate and are not citable",
            uncitable,
            len(blocks),
        )
    return blocks


def blocks_on_page(blocks: list[TextBlockRecord], page_no: int) -> list[TextBlockRecord]:
    return [block for block in blocks if block.page == page_no]

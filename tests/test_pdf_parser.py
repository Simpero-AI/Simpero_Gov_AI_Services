import shutil
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pypdf import PdfWriter
from reportlab.pdfgen import canvas

from services.parser.parser_service.config import get_settings
from services.parser.parser_service.docling_parser import (
    normalize_numeric_tokens,
    parse_known_hashes,
    parse_pdf_bytes,
    tag_boilerplate,
)
from services.parser.parser_service.main import app
from services.parser.parser_service.schemas import CharBox, PageIndex

client = TestClient(app)


def make_page_index(page_no: int, lines: list[str]) -> PageIndex:
    text = "\n".join(lines)
    char_map = [
        CharBox(char=ch, x0=0.0, top=0.0, x1=1.0, bottom=1.0, page=page_no, precision="word")
        for ch in text
    ]
    return PageIndex(page=page_no, text=text, char_map=char_map)


def make_char_map(text: str, page_no: int = 1) -> list[CharBox]:
    return [
        CharBox(char=ch, x0=0.0, top=0.0, x1=1.0, bottom=1.0, page=page_no, precision="word")
        for ch in text
    ]


def make_text_pdf(lines: list[str]) -> bytes:
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer)
    y = 760
    for line in lines:
        pdf.drawString(72, y, line)
        y -= 24
    pdf.save()
    return buffer.getvalue()


def make_multi_page_pdf_with_boilerplate(num_pages: int = 4) -> bytes:
    """A synthetic, generated-at-test-time multi-page PDF reproducing the
    structural properties DS-W3-1 validates against real CIM documents — a
    repeating confidentiality banner, a page-number footer, and a numeric
    token split by a literal space — without depending on a real (and
    potentially confidential) source document. Generated in-process rather
    than committed as a binary fixture: no repo bloat, no binary diffs to
    review, and the exact content under test is visible as code.
    """
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer)
    for page_no in range(1, num_pages + 1):
        pdf.drawString(72, 760, "CONFIDENTIAL - SAMPLE TEST DOCUMENT")
        pdf.drawString(72, 730, f"Section {page_no}: Revenue was 3 ,817 thousand dollars.")
        pdf.drawString(72, 50, str(page_no))
        pdf.showPage()
    pdf.save()
    return buffer.getvalue()


def make_blank_pdf(page_count: int = 1) -> bytes:
    writer = PdfWriter()
    for _ in range(page_count):
        writer.add_blank_page(width=612, height=792)
    buffer = BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def copy_test_pdf_if_needed() -> Path | None:
    """Locate the real PTL CIM in a sibling repo checkout, if this machine
    has one. Only used by the `local_corpus`-marked supplementary test below
    — never relied on for CI coverage, since this confidential document is
    never committed to this repository.
    """
    dest_dir = Path(__file__).parent.parent / "test_data"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / "1st-App-H-PTL-Group-CIM.pdf"

    if not dest_path.exists():
        src_path = Path(
            "p:/simpero_GOV_AI/scripts/examples/1st-app-h-ptl/1st-App-H-PTL-Group-CIM.pdf"
        )
        if src_path.exists():
            shutil.copy(src_path, dest_path)

    return dest_path if dest_path.exists() else None


def copy_pitchbook_pdf_if_needed() -> Path | None:
    """Same as copy_test_pdf_if_needed, for the Sell-Side Pitchbook CIM."""
    dest_dir = Path(__file__).parent.parent / "test_data"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / "Sell-Side-Pitchbook-CIM-Sample.pdf"

    if not dest_path.exists():
        src_path = Path(
            "p:/simpero_GOV_AI/scripts/examples/sell-side-pitchbook/"
            "Sell-Side-Pitchbook-CIM-Sample.pdf"
        )
        if src_path.exists():
            shutil.copy(src_path, dest_path)

    return dest_path if dest_path.exists() else None


def test_parse_pdf_returns_page_index_and_char_coordinates() -> None:
    pdf_bytes = make_text_pdf(
        [
            "Revenue increased to $10 million.",
            "EBITDA margin was 25%.",
        ]
    )

    result = parse_pdf_bytes(pdf_bytes)

    assert result.sha256 == sha256(pdf_bytes).hexdigest()
    assert len(result.pages) == 1
    page = result.pages[0]

    assert page.page == 1
    assert "Revenue increased to $10 million." in page.text
    assert "EBITDA margin was 25%." in page.text

    # Assert hard invariants
    assert len(page.text) == len(page.char_map)
    for i, char in enumerate(page.text):
        assert char == page.char_map[i].char
        # Bbox coordinates should be valid floats
        assert isinstance(page.char_map[i].x0, float)
        assert isinstance(page.char_map[i].top, float)
        assert isinstance(page.char_map[i].x1, float)
        assert isinstance(page.char_map[i].bottom, float)

    # Docling does not expose per-character geometry for native (non-OCR)
    # text extraction, so every box must honestly declare word-level
    # precision rather than a fabricated per-glyph estimate.
    assert all(cb.precision == "word" for cb in page.char_map)


def test_parse_endpoint_accepts_pdf_bytes() -> None:
    pdf_bytes = make_text_pdf(["The company serves enterprise customers."])

    response = client.post(
        "/parse",
        content=pdf_bytes,
        headers={"Content-Type": "application/pdf"},
    )

    assert response.status_code == 200
    assert response.headers["X-Content-SHA256"] == sha256(pdf_bytes).hexdigest()

    pages = response.json()
    assert len(pages) == 1
    assert "The company serves enterprise customers." in pages[0]["text"]
    assert len(pages[0]["text"]) == len(pages[0]["char_map"])


def test_zero_byte_pdf_is_rejected() -> None:
    response = client.post(
        "/parse",
        content=b"",
        headers={"Content-Type": "application/pdf"},
    )

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "zero_byte_pdf"


def test_corrupt_pdf_is_rejected() -> None:
    response = client.post(
        "/parse",
        content=b"not a pdf",
        headers={"Content-Type": "application/pdf"},
    )

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "corrupt_pdf"


def test_duplicate_pdf_is_rejected_by_known_sha_header() -> None:
    pdf_bytes = make_text_pdf(["Duplicate source document."])
    digest = sha256(pdf_bytes).hexdigest()

    response = client.post(
        "/parse",
        content=pdf_bytes,
        headers={
            "Content-Type": "application/pdf",
            "X-Known-SHA256": digest,
        },
    )

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "duplicate_pdf"


def test_page_limit_guard_rejects_over_110_pages() -> None:
    response = client.post(
        "/parse",
        content=make_blank_pdf(page_count=111),
        headers={"Content-Type": "application/pdf"},
    )

    assert response.status_code == 413
    assert response.json()["detail"]["code"] == "pdf_too_large"


def test_blank_pdf_without_text_is_rejected() -> None:
    # A completely blank PDF generated without any text cells
    response = client.post(
        "/parse",
        content=make_blank_pdf(page_count=1),
        headers={"Content-Type": "application/pdf"},
    )

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "no_extractable_text"


def test_normalize_numeric_tokens_collapses_space_before_separator() -> None:
    text = "Revenue was 3 ,817 thousand."
    char_map = make_char_map(text)

    normalized_text, normalized_char_map = normalize_numeric_tokens(text, char_map)

    assert "3,817" in normalized_text
    assert "3 ,817" not in normalized_text
    assert len(normalized_text) == len(normalized_char_map)
    for i, ch in enumerate(normalized_text):
        assert ch == normalized_char_map[i].char


def test_normalize_numeric_tokens_handles_multiple_groups_in_one_number() -> None:
    text = "Total: 1 ,234 ,567 units."
    char_map = make_char_map(text)

    normalized_text, normalized_char_map = normalize_numeric_tokens(text, char_map)

    assert "1,234,567" in normalized_text
    assert " ," not in normalized_text
    assert len(normalized_text) == len(normalized_char_map)


def test_normalize_numeric_tokens_handles_decimal_point_split() -> None:
    text = "Margin was 27 .3 percent."
    char_map = make_char_map(text)

    normalized_text, normalized_char_map = normalize_numeric_tokens(text, char_map)

    assert "27.3" in normalized_text
    assert "27 .3" not in normalized_text
    assert len(normalized_text) == len(normalized_char_map)


def test_normalize_numeric_tokens_is_noop_without_split_tokens() -> None:
    text = "Revenue was 3,817 thousand, already normalized."
    char_map = make_char_map(text)

    normalized_text, normalized_char_map = normalize_numeric_tokens(text, char_map)

    assert normalized_text == text
    assert len(normalized_char_map) == len(char_map)
    assert [c.char for c in normalized_char_map] == list(text)


def _indexed_char_map(text: str) -> list[CharBox]:
    """char_map where each box's x0 == its position in `text`, so a test can
    assert a surviving char still points at its true source glyph after
    normalization (not merely that the character value matches)."""
    return [
        CharBox(
            char=ch, x0=float(i), top=0.0, x1=float(i) + 1.0, bottom=1.0, page=1, precision="word"
        )
        for i, ch in enumerate(text)
    ]


def test_normalize_preserves_exact_source_coordinates_of_kept_chars() -> None:
    # The provenance guarantee that matters for citations: when "3 ,817"
    # collapses, the surviving comma must carry the *comma glyph's* real
    # coordinate, not the dropped space's. Char-equality alone can't catch a
    # coordinate mix-up; distinct per-index coordinates can.
    text = "X 3 ,817 Y"
    normalized_text, normalized_char_map = normalize_numeric_tokens(text, _indexed_char_map(text))

    assert normalized_text == "X 3,817 Y"
    # Every surviving char's x0 still equals its ORIGINAL index in the source.
    survivors = {"X": 0, "3": 2, ",": 4, "8": 5, "1": 6, "7": 7, "Y": 9}
    # Walk normalized output and confirm the collapsed comma points at index 4
    # (the real comma), never index 3 (the dropped space).
    comma_boxes = [c for c in normalized_char_map if c.char == ","]
    assert len(comma_boxes) == 1
    assert comma_boxes[0].x0 == float(survivors[","])
    # Spot-check the digits flanking the dropped space keep their real glyphs.
    three_box = next(c for c in normalized_char_map if c.char == "3")
    eight_box = next(c for c in normalized_char_map if c.char == "8")
    assert three_box.x0 == 2.0
    assert eight_box.x0 == 5.0


def test_normalize_split_token_at_string_boundaries() -> None:
    # Split token as the very first characters, and as the very last, must
    # normalize and keep the invariant (off-by-one guard on the sliding match).
    for text, expected in [
        ("3 ,817 trailing", "3,817 trailing"),
        ("leading 3 ,817", "leading 3,817"),
    ]:
        normalized_text, normalized_char_map = normalize_numeric_tokens(text, make_char_map(text))
        assert expected in normalized_text
        assert "3 ,817" not in normalized_text
        assert len(normalized_text) == len(normalized_char_map)
        for i, ch in enumerate(normalized_text):
            assert ch == normalized_char_map[i].char


def test_normalize_tolerates_multiple_spaces_before_separator() -> None:
    text = "Revenue 3  ,817 total"
    normalized_text, normalized_char_map = normalize_numeric_tokens(text, make_char_map(text))
    assert "3,817" in normalized_text
    assert len(normalized_text) == len(normalized_char_map)


def test_normalize_does_not_touch_number_followed_by_non_digit() -> None:
    # "3 . Next" is a sentence boundary, NOT a split decimal — the trailing
    # \d in the pattern must prevent collapse. Regression guard against a
    # future loosening of the regex that would merge sentence punctuation.
    text = "See item 3 . Next section begins."
    normalized_text, _ = normalize_numeric_tokens(text, make_char_map(text))
    assert normalized_text == text


def test_normalize_does_not_collapse_spaced_lists_or_section_numbers() -> None:
    # Regression guard for QA finding F1. The rule must NOT merge a spaced list
    # ("pages 3 , 4") or a sub-section number ("Section 3 . 2") — corrupting a
    # figure on a financial due-diligence platform is worse than leaving a rare
    # split token unmerged. A thousands comma needs exactly 3 following digits;
    # a decimal point needs the digit immediately after it (no space).
    for text in [
        "pages 3 , 4 today",  # one digit != a thousands group
        "Section 3 . 2 rules",  # space after the point => not a decimal
        "See item 3 . Next section",  # sentence punctuation
        "value 3 ,8175 here",  # four digits != a thousands group
        "spread 27 . 3 range",  # spaces both sides => ambiguous, left alone
    ]:
        normalized, char_map = normalize_numeric_tokens(text, make_char_map(text))
        assert normalized == text, f"must not alter {text!r}"
        assert len(normalized) == len(char_map)


def test_normalize_still_collapses_genuine_split_numbers() -> None:
    # The other half of F1: tightening the rule must not break the real cases.
    for text, expected in [
        ("Revenue 3 ,817 thousand", "Revenue 3,817 thousand"),  # verified corpus token
        ("Total 1 ,234 ,567 units", "Total 1,234,567 units"),  # chained groups
        ("Margin 27 .3 percent", "Margin 27.3 percent"),  # decimal
        ("Revenue 3  ,817 total", "Revenue 3,817 total"),  # multiple spaces
        ("Cash 1 , 234 total", "Cash 1,234 total"),  # space after comma, 3 digits
    ]:
        normalized, char_map = normalize_numeric_tokens(text, make_char_map(text))
        assert normalized == expected
        assert len(normalized) == len(char_map)
        for i, ch in enumerate(normalized):
            assert ch == char_map[i].char


def test_normalize_is_linear_on_large_pages() -> None:
    # Regression guard for QA finding F4. The old implementation re-sliced the
    # page text on every character (quadratic); a large page made this crawl.
    # This asserts the linear behavior holds — a generous bound, so it fails only
    # on a genuine complexity regression, not on a slow machine.
    import time

    text = ("Revenue was 3 ,817 thousand and margin 27 .3 percent. " * 4000)[:200_000]
    char_map = make_char_map(text)

    start = time.perf_counter()
    normalized, normalized_char_map = normalize_numeric_tokens(text, char_map)
    elapsed = time.perf_counter() - start

    assert "3,817" in normalized
    assert len(normalized) == len(normalized_char_map)
    assert elapsed < 5.0, f"200k-char page took {elapsed:.2f}s — complexity regression?"


def test_parse_known_hashes_handles_none_empty_and_messy_input() -> None:
    assert parse_known_hashes(None) == set()
    assert parse_known_hashes("") == set()
    assert parse_known_hashes("   ") == set()
    # Whitespace trimmed, lowercased, empty items between commas dropped.
    assert parse_known_hashes("  ABC , def ,, GHI ") == {"abc", "def", "ghi"}


def test_synthetic_multi_page_document_acceptance() -> None:
    """CI-portable equivalent of the ticket's real-corpus acceptance test.

    Exercises the full parse pipeline — the text/char_map invariant, numeric-token
    normalization, and boilerplate tagging — against a fixture generated at test
    time rather than a real (and confidential) CIM document. Always runs, on any
    platform, in CI; no skip condition.
    """
    pdf_bytes = make_multi_page_pdf_with_boilerplate(num_pages=4)

    result = parse_pdf_bytes(pdf_bytes)

    assert len(result.pages) == 4

    # Hard invariant across all pages
    for page in result.pages:
        assert len(page.text) == len(page.char_map)
        for i, char in enumerate(page.text):
            assert char == page.char_map[i].char

    # Numeric normalization: "3 ,817" -> "3,817" on every page
    for page in result.pages:
        assert "3,817" in page.text
        assert "3 ,817" not in page.text

    # Boilerplate: the repeating banner and page-number footer are tagged on
    # every page (4 >= the ticket's >= 3 threshold), without being stripped.
    pages_with_boilerplate = [
        page for page in result.pages if any(c.is_boilerplate for c in page.char_map)
    ]
    assert len(pages_with_boilerplate) >= 3
    for page in pages_with_boilerplate:
        boilerplate_text = "".join(c.char for c in page.char_map if c.is_boilerplate)
        assert boilerplate_text, "boilerplate chars must retain their text, not be stripped"


@pytest.mark.local_corpus
def test_cim_pdf_acceptance_and_normalization() -> None:
    pdf_path = copy_test_pdf_if_needed()
    if not pdf_path or not pdf_path.exists():
        pytest.skip(
            "Real PTL CIM not available on this machine (confidential document, "
            "never committed to this repo — see test_synthetic_multi_page_document_"
            "acceptance for the CI-portable equivalent coverage)."
        )

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    result = parse_pdf_bytes(pdf_bytes)
    digest = sha256(pdf_bytes).hexdigest()

    # 1st-App-H-PTL-Group-CIM.pdf is 19 pages
    assert len(result.pages) == 19

    # The raw DoclingDocument is cached as a single whole-document file for
    # DS-W3-2/DS-W3-6.
    cache_file = get_settings().cache_dir / f"{digest}.json"
    assert cache_file.exists(), "expected the DoclingDocument cache file"

    # Hard invariant test across all pages
    for page in result.pages:
        assert len(page.text) == len(page.char_map)
        for i, char in enumerate(page.text):
            assert char == page.char_map[i].char

    # PDF Page index 11 (0-indexed page 10) is the income statement page.
    # In 1st-App-H-PTL-Group-CIM.pdf, page index 11 has the value 3,817 which originally renders as "3 ,817"
    page_11 = result.pages[10]

    # Verify the space has been collapsed and normalized
    assert "3,817" in page_11.text
    assert "3 ,817" not in page_11.text


def test_boilerplate_tagging_flags_repeating_banner_and_page_numbers() -> None:
    # A confidentiality banner repeats verbatim on every page; a page number
    # varies but occupies the same structural position; body text is unique
    # per page and must not be tagged.
    pages = [
        make_page_index(
            page_no,
            [
                "CONFIDENTIAL - DO NOT DISTRIBUTE",
                f"Revenue for period {page_no} was $10 million.",
                str(page_no),
            ],
        )
        for page_no in range(1, 5)
    ]

    tag_boilerplate(pages)

    for page in pages:
        lines = page.text.split("\n")
        banner_len = len(lines[0])
        body_start = banner_len + 1
        body_len = len(lines[1])
        page_number_start = body_start + body_len + 1

        assert all(c.is_boilerplate for c in page.char_map[:banner_len]), (
            "repeating banner should be tagged is_boilerplate"
        )
        assert not any(
            c.is_boilerplate for c in page.char_map[body_start : body_start + body_len]
        ), "unique body text must not be tagged is_boilerplate"
        assert all(c.is_boilerplate for c in page.char_map[page_number_start:]), (
            "page-number line should be tagged is_boilerplate"
        )


def test_boilerplate_tagging_does_not_strip_text() -> None:
    pages = [
        make_page_index(page_no, ["CONFIDENTIAL", f"Body text {page_no}"])
        for page_no in range(1, 4)
    ]
    original_texts = [p.text for p in pages]

    tag_boilerplate(pages)

    assert [p.text for p in pages] == original_texts
    for page in pages:
        assert len(page.text) == len(page.char_map)
        for i, char in enumerate(page.text):
            assert char == page.char_map[i].char


def test_boilerplate_not_tagged_below_repeat_threshold() -> None:
    # A banner repeating on only 2 pages is under MIN_BOILERPLATE_REPEAT_PAGES
    # (3) and must NOT be tagged — guards against tagging furniture in very
    # short documents where a "repeat" isn't yet established.
    pages = [
        make_page_index(page_no, ["CONFIDENTIAL BANNER", f"Body {page_no}", str(page_no)])
        for page_no in range(1, 3)
    ]
    tag_boilerplate(pages)
    assert not any(c.is_boilerplate for p in pages for c in p.char_map)


def test_boilerplate_page_number_needs_recurrence_across_pages() -> None:
    # Page-number furniture is tagged by structural recurrence, not exact text.
    # A digit-only footer present on only 2 of 4 pages is below threshold and
    # must not be tagged; the repeating banner on all 4 still is.
    pages = []
    for page_no in range(1, 5):
        footer = str(page_no) if page_no <= 2 else f"Footnote {page_no} continues"
        pages.append(make_page_index(page_no, ["CONFIDENTIAL BANNER", f"Body {page_no}", footer]))
    tag_boilerplate(pages)

    for page in pages[:2]:
        lines = page.text.split("\n")
        page_number_start = len(lines[0]) + 1 + len(lines[1]) + 1
        assert not any(c.is_boilerplate for c in page.char_map[page_number_start:]), (
            "digit footer appearing on only 2/4 pages must not be tagged"
        )
    # Banner (line 0) still tagged on every page.
    for page in pages:
        banner_len = len(page.text.split("\n")[0])
        assert all(c.is_boilerplate for c in page.char_map[:banner_len])


def test_boilerplate_ignores_repeating_line_outside_header_footer_zone() -> None:
    # A line identical across pages but sitting in the middle of a long page
    # (outside the top/bottom BOILERPLATE_ZONE_LINES window) is body content,
    # not furniture, and must not be tagged.
    middle_marker = "REPEATED MIDDLE LINE"
    pages = []
    for page_no in range(1, 4):
        lines = [f"top-{page_no}-{i}" for i in range(6)]
        lines.append(middle_marker)  # index 6 — middle of a 13-line page
        lines.extend(f"bot-{page_no}-{i}" for i in range(6))
        pages.append(make_page_index(page_no, lines))
    tag_boilerplate(pages)

    for page in pages:
        spans = []
        start = 0
        for line in page.text.split("\n"):
            spans.append((start, start + len(line), line))
            start += len(line) + 1
        for s, e, line in spans:
            if line == middle_marker:
                assert not any(c.is_boilerplate for c in page.char_map[s:e]), (
                    "a repeating line outside the header/footer zone is body text"
                )


def test_boilerplate_short_page_does_not_over_tag_unique_body() -> None:
    # On a 3-line page the top-5 and bottom-5 zones overlap (every line is in
    # both). Verify the overlap does not cause unique body text to be tagged —
    # only the genuinely-repeating banner and page-number are.
    pages = [
        make_page_index(
            page_no, ["CONFIDENTIAL BANNER", f"Unique sentence {page_no}", str(page_no)]
        )
        for page_no in range(1, 4)
    ]
    tag_boilerplate(pages)
    for page in pages:
        lines = page.text.split("\n")
        body_start = len(lines[0]) + 1
        body_end = body_start + len(lines[1])
        assert not any(c.is_boilerplate for c in page.char_map[body_start:body_end]), (
            "unique body text on a short page must not be tagged despite zone overlap"
        )


@pytest.mark.local_corpus
def test_pitchbook_cim_boilerplate_and_acceptance() -> None:
    pdf_path = copy_pitchbook_pdf_if_needed()
    if not pdf_path or not pdf_path.exists():
        pytest.skip(
            "Real Pitchbook CIM not available on this machine (confidential document, "
            "never committed to this repo — see test_synthetic_multi_page_document_"
            "acceptance for the CI-portable equivalent coverage)."
        )

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    result = parse_pdf_bytes(pdf_bytes)

    # Sell-Side-Pitchbook-CIM-Sample.pdf is 21 pages
    assert len(result.pages) == 21

    # Hard invariant test across all pages
    for page in result.pages:
        assert len(page.text) == len(page.char_map)
        for i, char in enumerate(page.text):
            assert char == page.char_map[i].char

    # The confidentiality banner and page-number header must be tagged
    # is_boilerplate on multiple pages, per DS-W3-1's acceptance test, without
    # having been stripped from the text.
    pages_with_boilerplate = [
        page for page in result.pages if any(c.is_boilerplate for c in page.char_map)
    ]
    assert len(pages_with_boilerplate) >= 3

    for page in pages_with_boilerplate:
        boilerplate_text = "".join(c.char for c in page.char_map if c.is_boilerplate)
        assert boilerplate_text, "boilerplate chars must retain their text, not be stripped"

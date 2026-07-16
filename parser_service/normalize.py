"""Numeric-token normalization — the single source of truth for the rule.

Some PDFs render a real space inside a number: the PTL CIM's income statement
renders 3,817 as the literal string "3 ,817" (verified with PyMuPDF rawdict).
Left alone, a quote of "3,817" fails exact-match and a correct claim is dropped.

Both the flat page index (DS-W3-1) and table cell values (DS-W3-2) must apply
the *same* rule, so it lives here rather than being duplicated — tightening the
rule in one place fixes every consumer at once.

## The rule, and why it is this narrow

A naive `(\\d)\\s+([.,])\\s*(\\d)` over-collapses: it cannot tell a genuine split
thousands separator from a spaced list or a sub-section number, so it silently
corrupted "pages 3 , 4" into "pages 3,4" and "Section 3 . 2" into "Section 3.2".
On a financial due-diligence platform that is a real (if low-frequency)
correctness risk, so each separator now carries a distinguishing constraint:

* Thousands comma — requires **exactly three** following digits
  (`,\\s*(?=\\d{3}(?!\\d))`). "3 ,817" collapses; "3 , 4" does not, because one
  digit is not a thousands group.
* Decimal point — requires the digit to follow the point with **no space**
  (`\\.(?=\\d)`). "27 .3" collapses; "Section 3 . 2" does not, because the space
  after the point marks it as sentence/section punctuation, not a decimal.

The trailing digits are matched with a lookahead, not consumed, so consecutive
groups still chain: "1 ,234 ,567" -> "1,234,567" (the "4" remains available as
the next group's leading digit).

Deliberately conservative: "27 . 3" (spaces on both sides of a point) is left
alone. It is indistinguishable from a section number, and wrongly merging a
financial figure is worse than leaving a rare split token unmerged.
"""

import re

# Matches the leading digit, the whitespace, and the separator (plus any space
# after a comma). The digits *after* the separator are asserted by lookahead and
# left unconsumed so adjacent groups chain.
SPLIT_NUMERIC_RE = re.compile(r"\d\s+(?:,\s*(?=\d{3}(?!\d))|\.(?=\d))")


def kept_indices(text: str) -> list[int]:
    """Indices of `text` that survive normalization, in order.

    Everything inside a matched split-number span except its whitespace is
    kept; everything outside a match is kept verbatim. Returning indices (rather
    than a new string) is what lets the char_map be filtered in lockstep, so a
    surviving character keeps pointing at its own source glyph.

    Uses `re.match(text, pos)` rather than matching against `text[i:]`: slicing
    per character made this quadratic in page length. `match` already anchors at
    `pos`, so the pattern needs no `^` and no slice is allocated.
    """
    kept: list[int] = []
    i = 0
    n = len(text)
    while i < n:
        match = SPLIT_NUMERIC_RE.match(text, i)
        if match:
            kept.extend(j for j in range(match.start(), match.end()) if not text[j].isspace())
            i = match.end()
        else:
            kept.append(i)
            i += 1
    return kept


def normalize_numeric_text(text: str) -> str:
    """Normalize split numeric tokens in a bare string (no coordinates).

    Used for table cell values, which carry a bbox per cell rather than a
    per-character map.
    """
    return "".join(text[i] for i in kept_indices(text))

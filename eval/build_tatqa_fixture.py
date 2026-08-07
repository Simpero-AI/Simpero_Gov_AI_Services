"""Rebuild eval/fixtures/tatqa_sample.json from the TAT-QA dev set.

SIM-374. Committed so the fixture has an auditable, reproducible provenance --
the previous fixture was made by an uncommitted script, and that opacity is how
a silent "keep only the rows determine_scale already passes" filter slipped in
and pinned the harness at a meaningless 100%. This builder keeps ALL sampled
rows, labelled with TAT-QA's gold `scale`, and never consults determine_scale's
output when deciding what to keep.

Source : https://github.com/NExTplusplus/TAT-QA  (dataset_raw/tatqa_dataset_dev.json)
License : MIT (c) 2021 Fengbin Zhu -- redistribution of the sampled rows is
          permitted with this notice; the raw file itself is gitignored, only
          the small derived sample is committed.

Run `python -m eval.build_tatqa_fixture` to regenerate the fixture. The raw
dataset is downloaded to eval/_datasets/ (gitignored) on first run. CI never
runs this -- it reads the committed sample -- so the urllib download is a
developer-only path.

Scope of the harness this feeds (eval/tatqa.py): it exercises determine_scale's
PAGE-BANNER path -- origin="table" with no TableRecord/cell, the configuration a
caller is in when it holds page text but not the value's own cell. TAT-QA's
structure-free grid cannot faithfully rebuild the parser's TableRecord (no
col-spans, no header rows, no geometry), so the COLUMN-HEADER walk is not
testable here without testing the reconstruction instead. Rows whose scale is
carried only by column structure are therefore tagged `column_header` and
reported as out-of-path, never as determine_scale failures -- counting them as
failures would understate the parser, which resolves them via the column-header
walk in production.
"""

from __future__ import annotations

import json
import random
import re
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

_HERE = Path(__file__).parent
_CACHE = _HERE / "_datasets" / "tatqa_dataset_dev.json"
_FIXTURE = _HERE / "fixtures" / "tatqa_sample.json"
_SOURCE_URL = (
    "https://raw.githubusercontent.com/NExTplusplus/TAT-QA/master/"
    "dataset_raw/tatqa_dataset_dev.json"
)

# TAT-QA's per-answer `scale` gold -> the multiplier a correct reader applies.
_SCALE_MULT = {
    "": 1.0,
    "thousand": 1000.0,
    "million": 1_000_000.0,
    "billion": 1_000_000_000.0,
    "percent": 1.0,
}

_HAS_DIGIT = re.compile(r"\d")
# A bare four-digit year answer ("2019") is a date, not a scaled magnitude --
# excluded so the harness never grades scale resolution on a column heading.
_YEAR = re.compile(r"^\(?(?:19|20)\d{2}\)?$")
# An answer that already spells its own magnitude ("$85.1 million") self-scales
# via the inline path, not the banner path this harness tests, and TAT-QA labels
# it scale="" -- excluded so its expected multiplier is unambiguous.
_INLINE_MAGNITUDE = re.compile(r"\b(?:thousand|million|billion|trillion)\b", re.IGNORECASE)

# Independent scale-caption detector for the `scale_locus` tag ONLY. Deliberately
# NOT parser_service.scale's _find_scale_phrases: if locus were derived from the
# very function under test, banner-locus rows would be tautologically correct.
# This looser reading ("in <magnitude>" in any surrounding phrasing) is what a
# human sees as a page caption; where determine_scale's stricter parenthesised
# grammar fails to bind one this reading finds, that surfaces as an honest miss.
_BANNER = re.compile(
    r"(?i)(?:\bin\s+|\bdollars?\s+in\s+|amounts?\s+in\s+|\(\s*(?:\$|us\$|dollars?|amounts?)?\s*in\s+)"
    r"(thousand|million|billion)s?"
    r"|(thousand|million|billion)s?\s+of\s+dollars"
)
_WORD_MULT = {"thousand": 1000.0, "million": 1_000_000.0, "billion": 1_000_000_000.0}

_SEED = 374  # SIM-374; fixed so the sample is reproducible byte-for-byte.
_TARGET_SIZE = 55


def _download_if_absent() -> None:
    if _CACHE.exists():
        return
    _CACHE.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading TAT-QA dev set -> {_CACHE} (gitignored) ...")
    urllib.request.urlretrieve(_SOURCE_URL, _CACHE)  # noqa: S310  (fixed https GitHub raw URL)


def _render_page(doc: dict) -> tuple[str, int]:
    """The synthetic page a filing would present: paragraphs in reading order,
    then the table, so any "(in thousands)" note sits ahead of its values -- the
    same order the page-header search assumes. Returns (page_text, table_offset)
    where table_offset is where the table begins, used to locate a table value
    past the prose that may repeat the same digits."""
    paragraphs = sorted(doc["paragraphs"], key=lambda p: p["order"])
    prose = "\n".join(p["text"] for p in paragraphs)
    table = "\n".join("\t".join(str(cell) for cell in row) for row in doc["table"]["table"])
    return f"{prose}\n\n{table}", len(prose) + 2


def _independent_banner_multiplier(text: str, before: int) -> float | None:
    """Multiplier of the nearest independent scale caption strictly before
    `before`, or None. Nearest-wins mirrors the page-header rule."""
    found = None
    for match in _BANNER.finditer(text):
        if match.start() >= before:
            break
        found = _WORD_MULT[(match.group(1) or match.group(2)).lower()]
    return found


def _clean_candidates(dev: list[dict]) -> list[dict]:
    """Every table value with a determinable gold scale: single-value span
    answers drawn from a table, excluding years and inline-magnitude answers.
    No filtering by whether determine_scale happens to agree."""
    rows: list[dict] = []
    for doc in dev:
        page_text, table_offset = _render_page(doc)
        for q in doc["questions"]:
            if q.get("answer_type") != "span":
                continue
            answer = q.get("answer")
            if not (isinstance(answer, list) and len(answer) == 1):
                continue
            value = answer[0]
            if not isinstance(value, str) or not _HAS_DIGIT.search(value):
                continue
            if q.get("answer_from") not in ("table", "table-text"):
                continue
            if _YEAR.match(value.strip()) or _INLINE_MAGNITUDE.search(value):
                continue
            scale = q.get("scale", "")
            if scale not in _SCALE_MULT:
                continue
            char_start = page_text.find(value, table_offset)
            if char_start < 0:
                continue
            rows.append(
                {
                    "id": q["uid"],
                    "page_text": page_text,
                    "value_raw": value,
                    "char_start": char_start,
                    "value_type": "percent" if scale == "percent" else "currency",
                    "gold_scale": scale,
                    "expected_multiplier": _SCALE_MULT[scale],
                }
            )
    return rows


def _scale_locus(row: dict) -> str:
    """Where TAT-QA's scale for this value lives -- a fact about the data,
    decided by the independent caption reader, not by determine_scale.

    banner        : a page/section caption of the right magnitude precedes it;
                    the page-banner path should resolve it.
    face_value    : gold is 1x (no magnitude). In-path: a spurious nearby banner
                    that over-scales it is a real miss, so these are graded.
    percent       : self-scaling; the "never currency-scale a percent" check.
    column_header : gold is a magnitude but no page caption carries it -- the
                    scale is in column structure. Out-of-path (see module doc).
    """
    scale = row["gold_scale"]
    if scale == "percent":
        return "percent"
    if scale == "":
        return "face_value"
    banner = _independent_banner_multiplier(row["page_text"], row["char_start"])
    return "banner" if banner == row["expected_multiplier"] else "column_header"


def _sample(rows: list[dict], seed: int, target: int) -> list[dict]:
    """Deterministic, stratified by locus so every bucket is represented and the
    sample mirrors the full-pool mix. All `percent` rows are kept (only a handful
    exist and each is a distinct never-currency-scale case); the rest are drawn
    proportionally at a rate that lands near `target`."""
    by_locus: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_locus[row["scale_locus"]].append(row)

    percent = sorted(by_locus.pop("percent", []), key=lambda r: r["id"])
    remaining = sum(len(v) for v in by_locus.values())
    rate = max(0.0, (target - len(percent))) / remaining if remaining else 0.0

    picked = list(percent)
    for locus in sorted(by_locus):
        group = sorted(by_locus[locus], key=lambda r: r["id"])
        random.Random(f"{seed}:{locus}").shuffle(group)
        picked.extend(group[: round(rate * len(group))])
    picked.sort(key=lambda r: (r["scale_locus"], r["id"]))
    return picked


def _print_full_pool_stats(rows: list[dict]) -> None:
    """The honest headline numbers, computed over the FULL pool (not the sample)
    so the PR can cite them. Imports determine_scale lazily -- the builder's own
    correctness never depends on it."""
    from parser_service.scale import determine_scale
    from parser_service.schemas import PageIndex

    by_locus: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for row in rows:
        got = determine_scale(
            row["value_raw"],
            PageIndex(page=1, text=row["page_text"], char_map=[]),
            row["char_start"],
            value_type=row["value_type"],
            origin="table",
        ).scale_multiplier
        ok = got == row["expected_multiplier"]
        by_locus[row["scale_locus"]][0] += ok
        by_locus[row["scale_locus"]][1] += 1

    in_path = [locus for locus in by_locus if locus != "column_header"]
    correct = sum(by_locus[locus][0] for locus in in_path)
    total = sum(by_locus[locus][1] for locus in in_path)
    print(f"\nfull clean pool: n={len(rows)}")
    for locus, (c, t) in sorted(by_locus.items()):
        print(f"  {locus:14} {c}/{t} = {c / t:.0%}")
    print(f"IN-PATH scale_accuracy: {correct}/{total} = {correct / total:.1%}")
    print(f"OUT-OF-PATH (column_header, excluded): {by_locus['column_header'][1]}")


def main() -> None:
    _download_if_absent()
    dev = json.loads(_CACHE.read_text(encoding="utf-8"))
    rows = _clean_candidates(dev)
    for row in rows:
        row["scale_locus"] = _scale_locus(row)

    _print_full_pool_stats(rows)

    sample = _sample(rows, _SEED, _TARGET_SIZE)
    _FIXTURE.write_text(json.dumps(sample, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nwrote {len(sample)} rows -> {_FIXTURE}")
    print("  sample locus mix:", dict(Counter(r["scale_locus"] for r in sample)))


if __name__ == "__main__":
    main()

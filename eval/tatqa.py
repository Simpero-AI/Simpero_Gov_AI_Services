"""SIM-374: public-dataset eval harness for scale resolution.

TAT-QA's per-question `scale` gold ("", "thousand", "million", "billion",
"percent") answers exactly what `scale.py::determine_scale` exists to answer:
what a printed number actually means once its "(in thousands)"-style caption is
accounted for. A wrong power-of-1000 is the `scale_absurd` failure class the
July 2026 parser audit was built around.

This harness exercises determine_scale's PAGE-BANNER path -- origin="table" with
no TableRecord/cell, the configuration a caller is in when it holds page text
but not the value's own cell. TAT-QA's structure-free grid cannot faithfully
rebuild the parser's TableRecord (no col-spans, header rows, or geometry), so
the column-header walk is not testable here without grading the reconstruction
instead. Values whose scale is carried only by column structure are tagged
`column_header` by the builder and reported as OUT-OF-PATH, never as failures --
production resolves them via the column-header walk, so counting them against
determine_scale would understate the parser.

The score is honest and un-cherry-picked. The fixture is a stratified sample of
ALL clean TAT-QA table values (see eval/build_tatqa_fixture.py) -- rows where
determine_scale disagrees with the gold scale are KEPT, so `scale_accuracy` can
and does fall below 100%. The fixture's `scale_locus` comes from an independent
caption reader in the builder, never from determine_scale's own output, so the
banner bucket is a real judgement rather than a snapshot of the function grading
itself.

Necessary-but-not-sufficient, like every harness in this ticket.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from parser_service.scale import determine_scale
from parser_service.schemas import PageIndex

_FIXTURES = Path(__file__).parent / "fixtures"

# Buckets the page-banner path is responsible for; `column_header` is excluded
# from the accuracy denominator and reported on its own.
_IN_PATH = ("banner", "face_value", "percent")


@dataclass
class EvalResult:
    dataset: str
    n: int  # total fixture rows
    in_path_n: int  # rows the page-banner path is responsible for
    scale_accuracy: float  # correct / in_path_n -- the honest, un-cherry-picked score
    out_of_path_n: int  # column_header rows, excluded from accuracy
    by_locus: dict[str, tuple[int, int]]  # locus -> (correct, total)


def run_tatqa() -> EvalResult:
    examples = json.loads((_FIXTURES / "tatqa_sample.json").read_text(encoding="utf-8"))

    tally: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for ex in examples:
        result = determine_scale(
            ex["value_raw"],
            PageIndex(page=1, text=ex["page_text"], char_map=[]),
            ex["char_start"],
            value_type=ex["value_type"],
            origin="table",
        )
        correct = result.scale_multiplier == ex["expected_multiplier"]
        bucket = tally[ex["scale_locus"]]
        bucket[0] += correct
        bucket[1] += 1

    in_path_correct = sum(tally[locus][0] for locus in _IN_PATH)
    in_path_n = sum(tally[locus][1] for locus in _IN_PATH)
    return EvalResult(
        dataset="tatqa",
        n=len(examples),
        in_path_n=in_path_n,
        scale_accuracy=in_path_correct / in_path_n if in_path_n else 0.0,
        out_of_path_n=tally["column_header"][1],
        by_locus={locus: (c, t) for locus, (c, t) in tally.items()},
    )


def main() -> None:
    result = run_tatqa()
    print(
        f"{result.dataset}: n={result.n} "
        f"scale_accuracy={result.scale_accuracy:.1%} (in-path n={result.in_path_n}) "
        f"out_of_path={result.out_of_path_n}"
    )
    for locus in (*_IN_PATH, "column_header"):
        if locus in result.by_locus:
            c, t = result.by_locus[locus]
            print(f"  {locus:14} {c}/{t}")
    print(
        "\nNecessary-but-not-sufficient: grades the page-banner path against "
        "TAT-QA's own scale annotations, un-cherry-picked. column_header rows "
        "need the column-header walk (not testable from TAT-QA's grid) and are "
        "reported, not scored. Never a substitute for a CIM golden-set score."
    )


if __name__ == "__main__":
    main()

"""SIM-374: public-dataset eval harness for scale resolution.

TAT-QA's per-question `scale` gold label ("", "thousand", "million",
"billion", "percent") answers exactly the question `scale.py::determine_scale`
exists to answer: what a printed number actually means once its "(in
millions)"-style header is accounted for. That's the direct `scale_absurd`
check -- a wrong power-of-1000 is exactly the failure class documented in
`scale.py`'s own module docstring (the July 2026 parser audit's real 1000x
defects).

`eval/fixtures/tatqa_sample.json` was built by a TAT-QA-specific fixture
builder that constructed a synthetic page (paragraphs first, then the
table -- the order a real filing places a scale note ahead of its table)
and used the REAL `determine_scale` as the ground-truth oracle: only rows
where our own reference run agreed with TAT-QA's gold scale label were kept.
As with the FinanceBench fixture, today's harness run is a regression
snapshot of already-correct behavior, not a fresh judgment call.

A TAT-QA `scale: "percent"` question is deliberately included and typed
`value_type="percent"` here -- that's the module docstring's own warning
case (a per-share/percent figure must never take a currency page banner),
and this dataset happens to carry real examples of it.

Necessary-but-not-sufficient, like every harness in this ticket.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from parser_service.scale import determine_scale
from parser_service.schemas import PageIndex

_FIXTURES = Path(__file__).parent / "fixtures"


@dataclass
class EvalResult:
    dataset: str
    n: int
    scale_accuracy: float


def run_tatqa() -> EvalResult:
    examples = json.loads((_FIXTURES / "tatqa_sample.json").read_text(encoding="utf-8"))

    n_correct = 0
    for ex in examples:
        page = PageIndex(page=1, text=ex["page_text"], char_map=[])
        result = determine_scale(
            ex["value_raw"],
            page,
            ex["char_start"],
            value_type=ex["value_type"],
            origin="table",
        )
        if result.scale_multiplier == ex["expected_multiplier"]:
            n_correct += 1

    n = len(examples)
    return EvalResult(dataset="tatqa", n=n, scale_accuracy=n_correct / n if n else 0.0)


def main() -> None:
    result = run_tatqa()
    print(f"{result.dataset}: n={result.n} scale_accuracy={result.scale_accuracy:.2%}")
    print(
        "\nNecessary-but-not-sufficient: validates scale resolution against "
        "TAT-QA's own scale-of-the-number annotations. Never a substitute "
        "for a CIM golden-set score."
    )


if __name__ == "__main__":
    main()

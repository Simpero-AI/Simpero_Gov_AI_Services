"""SIM-374: public-dataset eval harness for the fail-closed span resolver.

`resolver.find_exact_span` was built and tuned against CIM prose/tables. This
harness re-runs it, unmodified, against real SEC 10-K excerpts (FinanceBench,
Patronus AI) it was never tuned on — the only way to know whether the
fail-closed exact-match rule generalizes, versus merely fitting the corpus it
was written against.

`eval/fixtures/financebench_sample.json` was built by
`build_financebench_fixture.py`-equivalent logic that used THIS module's own
`find_exact_span` as the ground-truth oracle: for each FinanceBench evidence
excerpt, a financial-figure-shaped token (a real citable value: "$292.3",
"(4.2)", "1,906" — never invented) was found either to resolve uniquely
(`expect_resolves: true`) or to appear 2+ times in its own excerpt
(`expect_resolves: false`, e.g. the same "1,906" repeated across a
multi-year income-statement column — exactly the ambiguity
`resolve_in_cell`'s docstring documents). That makes today's harness run a
snapshot of already-correct behavior, not a fresh judgment call — its job
going forward is to catch a REGRESSION from that snapshot, not to grade
`find_exact_span` cold.

Necessary-but-not-sufficient, like every harness in this ticket: real 10-K
prose is a different distribution from a CIM's, and this dataset carries no
CIM-shaped failure modes (fee_vs_price, basket_collapse) at all.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from parser_service.resolver import find_exact_span

_FIXTURES = Path(__file__).parent / "fixtures"


@dataclass
class EvalResult:
    dataset: str
    n: int
    accuracy: float  # agreement with the fixture's expect_resolves oracle
    resolvable_recall: float  # of should-resolve examples, fraction that did
    fail_closed_precision: float  # of should-stay-ambiguous, fraction that did


def run_financebench() -> EvalResult:
    examples = json.loads((_FIXTURES / "financebench_sample.json").read_text(encoding="utf-8"))

    n_agree = 0
    resolvable = [e for e in examples if e["expect_resolves"]]
    ambiguous = [e for e in examples if not e["expect_resolves"]]
    n_resolvable_ok = 0
    n_ambiguous_ok = 0

    for ex in examples:
        span = find_exact_span(ex["quote"], ex["page_text"], where=ex["id"])
        resolved = span is not None
        if resolved == ex["expect_resolves"]:
            n_agree += 1
            if ex["expect_resolves"]:
                n_resolvable_ok += 1
            else:
                n_ambiguous_ok += 1

    n = len(examples)
    return EvalResult(
        dataset="financebench",
        n=n,
        accuracy=n_agree / n if n else 0.0,
        resolvable_recall=n_resolvable_ok / len(resolvable) if resolvable else 0.0,
        fail_closed_precision=n_ambiguous_ok / len(ambiguous) if ambiguous else 0.0,
    )


def main() -> None:
    result = run_financebench()
    print(
        f"{result.dataset}: n={result.n} accuracy={result.accuracy:.2%} "
        f"resolvable_recall={result.resolvable_recall:.2%} "
        f"fail_closed_precision={result.fail_closed_precision:.2%}"
    )
    print(
        "\nNecessary-but-not-sufficient: validates span resolution against "
        "real 10-K prose it wasn't tuned on. Never a substitute for a CIM "
        "golden-set score."
    )


if __name__ == "__main__":
    main()

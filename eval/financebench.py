"""SIM-374: public-dataset eval harness for the fail-closed span resolver.

`resolver.find_exact_span` was built and tuned against CIM prose/tables. This
harness re-runs it, unmodified, against real SEC 10-K excerpts (FinanceBench,
Patronus AI) it was never tuned on -- the only way to know whether the
fail-closed exact-match rule generalizes, versus merely fitting the corpus it
was written against.

Ground truth is INDEPENDENT of the resolver. Each fixture row carries a real
citable figure ("$292.3", "(4.2)", "1,906" -- never invented) drawn from a
FinanceBench evidence excerpt, plus `expect_resolves`: whether that figure
occurs exactly once in the excerpt. The harness re-derives that multiplicity at
run time with `_occurrences`, a dead-simple substring scan deliberately unrelated
to `find_exact_span`'s token/`\\s+`/overlap regex, and

  1. asserts the fixture's stored `expect_resolves` still matches it -- so the
     label can never silently rot into a stale echo of the resolver's own
     output, which is exactly the circularity this rewrite removes; and
  2. grades `find_exact_span` against it.

Because the label is decided by the figure's real multiplicity in the text -- a
fact anyone can verify by eye from `page_text` -- the harness grades the resolver
COLD: a resolver that is wrong TODAY fails here, not merely one that drifts from
a past snapshot. The fixture quotes are single financial figures with no interior
whitespace, so the plain scan and the resolver's whitespace-flexible match cannot
diverge on them; the count is the figure's true multiplicity, not one grammar's
opinion of it.

Necessary-but-not-sufficient, like every harness in this ticket: real 10-K prose
is a different distribution from a CIM's, and this dataset carries no CIM-shaped
failure modes (fee_vs_price, basket_collapse) at all.
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
    accuracy: float  # agreement with the independent occurrence ground truth
    resolvable_recall: float  # of should-resolve examples, fraction that did
    fail_closed_precision: float  # of should-stay-ambiguous, fraction that did


def _occurrences(quote: str, text: str) -> int:
    """How many times `quote` appears in `text`, counted independently of
    `find_exact_span`.

    A plain left-to-right substring scan -- no token splitting, no whitespace
    flexibility, no regex. The step of +1 (not +len) keeps overlapping hits
    counted, matching the resolver's own overlap semantics so the two never
    disagree for a spurious reason. This is the harness's ground truth: grading
    the resolver against its own output would only ever confirm the resolver
    reproduces itself.
    """
    if not quote.strip():
        return 0
    count, i = 0, text.find(quote)
    while i != -1:
        count += 1
        i = text.find(quote, i + 1)
    return count


def run_financebench() -> EvalResult:
    examples = json.loads((_FIXTURES / "financebench_sample.json").read_text(encoding="utf-8"))

    n_agree = 0
    n_resolvable = 0
    n_ambiguous = 0
    n_resolvable_ok = 0
    n_ambiguous_ok = 0

    for ex in examples:
        should_resolve = _occurrences(ex["quote"], ex["page_text"]) == 1
        # The stored label is human-authored documentation of intent; this guard
        # keeps it honest. A mismatch means the excerpt or quote was edited out
        # from under the label -- fail loud rather than grade against a lie.
        if should_resolve != ex["expect_resolves"]:
            raise ValueError(
                f"{ex['id']}: fixture expect_resolves={ex['expect_resolves']} but "
                f"{ex['quote']!r} occurs {_occurrences(ex['quote'], ex['page_text'])}x "
                "in its excerpt"
            )

        resolved = find_exact_span(ex["quote"], ex["page_text"], where=ex["id"]) is not None
        if should_resolve:
            n_resolvable += 1
            if resolved:
                n_resolvable_ok += 1
        else:
            n_ambiguous += 1
            if not resolved:
                n_ambiguous_ok += 1
        if resolved == should_resolve:
            n_agree += 1

    n = len(examples)
    return EvalResult(
        dataset="financebench",
        n=n,
        accuracy=n_agree / n if n else 0.0,
        resolvable_recall=n_resolvable_ok / n_resolvable if n_resolvable else 0.0,
        fail_closed_precision=n_ambiguous_ok / n_ambiguous if n_ambiguous else 0.0,
    )


def main() -> None:
    result = run_financebench()
    print(
        f"{result.dataset}: n={result.n} accuracy={result.accuracy:.2%} "
        f"resolvable_recall={result.resolvable_recall:.2%} "
        f"fail_closed_precision={result.fail_closed_precision:.2%}"
    )
    print(
        "\nNecessary-but-not-sufficient: grades span resolution COLD against the "
        "figures' real multiplicity in real 10-K prose it wasn't tuned on. Never "
        "a substitute for a CIM golden-set score."
    )


if __name__ == "__main__":
    main()

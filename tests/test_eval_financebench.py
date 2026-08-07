"""SIM-374: public-dataset eval harness for span resolution (eval/financebench.py).
Fast, CI-portable, no fixtures beyond the committed JSON sample -- no docling,
no local_corpus marker needed, since this exercises find_exact_span directly
against pre-extracted text."""

from __future__ import annotations

from eval.financebench import run_financebench


def test_financebench_fixture_regression_floor() -> None:
    result = run_financebench()
    assert result.n >= 30
    assert result.resolvable_recall >= 0.9
    assert result.fail_closed_precision >= 0.9

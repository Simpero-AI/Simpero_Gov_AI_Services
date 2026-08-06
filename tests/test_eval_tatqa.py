"""SIM-374: public-dataset eval harness for scale resolution (eval/tatqa.py).
Fast, CI-portable, no docling -- exercises determine_scale directly against
a synthetic PageIndex built from the committed TAT-QA fixture."""

from __future__ import annotations

from eval.tatqa import run_tatqa


def test_tatqa_fixture_regression_floor() -> None:
    result = run_tatqa()
    assert result.n >= 25
    assert result.scale_accuracy >= 0.9

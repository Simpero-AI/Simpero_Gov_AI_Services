"""SIM-374: public-dataset eval harness for scale resolution (eval/tatqa.py).
Fast, CI-portable, no docling -- exercises determine_scale directly against
a synthetic PageIndex built from the committed TAT-QA fixture."""

from __future__ import annotations

from eval.tatqa import run_tatqa


def test_tatqa_scale_resolution_floor() -> None:
    result = run_tatqa()
    assert result.n >= 40
    assert result.in_path_n >= 30
    # A regression FLOOR, not a target: the sample is un-cherry-picked, so this
    # sits below 1.0 today (determine_scale misses SEC "(in thousands, except
    # per share)"-style captions). Fixing that recognition raises the score and
    # still passes; a drop below the floor fails.
    assert 0.55 <= result.scale_accuracy <= 1.0


def test_tatqa_fixture_is_not_re_cherry_picked() -> None:
    # Structural guard against the flaw this rewrite fixed: the fixture must keep
    # rows the page-banner path cannot resolve (column_header) and the graded
    # banner rows, so the score reflects real behaviour rather than a set of
    # pre-selected passes.
    result = run_tatqa()
    assert result.out_of_path_n >= 1
    assert "banner" in result.by_locus

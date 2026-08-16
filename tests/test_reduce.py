"""SIM-341: per-page tier reducer (same-fact fan-in) tests.

Fixtures build Claim objects directly rather than through the real resolver
pipeline (test_emit.py's job) -- the reducer only ever looks at entity,
attribute, page, status and value.normalized/raw, so a minimal Claim is
enough to exercise its grouping logic in isolation.
"""

from __future__ import annotations

from typing import Literal

from parser_service import extract_service
from parser_service.emit import Claim, ClaimValue, FlagLog, PdfLocation, PeriodKind, element_id_for
from parser_service.extract_service import _canonicalize_quantitative_claims, _reduce_same_fact
from parser_service.scale import ValueType


def _claim(
    *,
    entity: str = "ACME",
    attribute: str,
    raw: str,
    normalized: float | None,
    value_type: ValueType | None = None,
    unit: str | None = None,
    period_year: int | None = None,
    period_kind: PeriodKind | None = None,
    page: int = 1,
    char_start: int = 0,
    status: Literal["proposed", "cited", "missing"] = "proposed",
) -> Claim:
    missing = status == "missing"
    resolved_type: ValueType = value_type or ("text" if normalized is None else "currency")
    return Claim(
        entity=entity,
        attribute=attribute,
        value=ClaimValue(
            raw=raw,
            normalized=normalized,
            unit=unit,
            value_type=resolved_type,
        ),
        location=PdfLocation(
            file="cim.pdf",
            page=page,
            char_start=None if missing else char_start,
            char_end=None if missing else char_start + len(raw),
        ),
        period_year=period_year,
        period_kind=period_kind,
        status=status,
    )


def test_same_fact_links_prose_to_table_and_table_wins() -> None:
    table_claim = _claim(attribute="Revenue | 2024F", raw="$15,295", normalized=15295000, page=3)
    prose_claim = _claim(
        attribute="total revenue for fiscal 2024",
        raw="fifteen million two hundred ninety-five thousand",
        normalized=15295000,
        page=3,
        char_start=500,
    )

    edges = _reduce_same_fact([("table", [table_claim]), ("prose", [prose_claim])])

    assert len(edges) == 1
    edge = edges[0]
    assert edge.type == "same_fact"
    assert edge.from_ == prose_claim.claim_ref
    assert edge.to == table_claim.claim_ref


def test_same_fact_does_not_collapse_across_value_types() -> None:
    # 40% (a margin), 40 (a headcount) and $40 (a price) share one page and one
    # normalized magnitude (40.0), yet they are three different facts. Keying
    # same_fact on the bare value alone fused them into a bogus corroboration;
    # value_type in the key keeps them apart.
    pct = _claim(
        attribute="gross margin", raw="40%", normalized=40.0, value_type="percent", unit="%"
    )
    headcount = _claim(
        attribute="employees", raw="40", normalized=40.0, value_type="count", char_start=20
    )
    price = _claim(
        attribute="Revenue", raw="$40", normalized=40.0, value_type="currency", char_start=40
    )

    edges = _reduce_same_fact([("prose", [pct]), ("qualitative", [headcount]), ("table", [price])])

    assert [edge for edge in edges if edge.type == "same_fact"] == []


def test_same_tier_repeating_a_value_is_not_a_duplicate() -> None:
    # Two table cells legitimately citing the same figure (e.g. a subtotal
    # repeated elsewhere) is not the cross-tier collision this reducer exists
    # to catch.
    claim_a = _claim(attribute="Subtotal A", raw="$5,000", normalized=5000, char_start=10)
    claim_b = _claim(attribute="Subtotal B", raw="$5,000", normalized=5000, char_start=50)

    edges = _reduce_same_fact([("table", [claim_a, claim_b])])

    assert edges == []


def test_contradicts_when_tiers_disagree_on_the_same_attribute() -> None:
    table_claim = _claim(
        attribute="Revenue | 2024F", raw="$15,000,000", normalized=15_000_000, page=5
    )
    prose_claim = _claim(
        attribute="Revenue | 2024F",
        raw="$12,000,000 excluding settlement",
        normalized=12_000_000,
        page=5,
        char_start=200,
    )

    edges = _reduce_same_fact([("table", [table_claim]), ("prose", [prose_claim])])

    assert len(edges) == 1
    edge = edges[0]
    assert edge.type == "contradicts"
    assert {edge.from_, edge.to} == {table_claim.claim_ref, prose_claim.claim_ref}


def test_contradicts_excludes_the_operating_metric_catch_all_bucket() -> None:
    # Two unrelated operating metrics (an occupancy rate and an ARPU figure)
    # both canonicalize into OPERATING_METRIC -- SIM-344's catch-all bucket,
    # not a fact-slot. Grouping `contradicts` on attribute alone fused them
    # into a false "these disagree" edge even though they are not the same
    # fact. Reproduces the review's example: a percent and a currency, both
    # operating_metric, on the same page.
    occupancy = _claim(
        attribute="operating_metric", raw="95%", normalized=95.0, value_type="percent", page=7
    )
    arpu = _claim(
        attribute="operating_metric",
        raw="$52",
        normalized=52.0,
        value_type="currency",
        page=7,
        char_start=30,
    )

    edges = _reduce_same_fact([("table", [occupancy]), ("prose", [arpu])])

    assert edges == []


def test_contradicts_excludes_the_core_unmapped_catch_all_bucket() -> None:
    # SIM-384 split core_unmapped out of operating_metric as its own catch-all
    # bucket. Like operating_metric, two unrelated figures that both land in
    # core_unmapped (a label the core enum maps to nothing) share only the
    # bucket, not a fact-slot -- they must not fuse into a false `contradicts`.
    # Two currency subtotals, same page/entity/value_type, different tiers and
    # values: without the exclusion this would be a `contradicts` edge.
    table_subtotal = _claim(attribute="core_unmapped", raw="$1,000", normalized=1000.0, page=7)
    prose_subtotal = _claim(
        attribute="core_unmapped", raw="$2,000", normalized=2000.0, page=7, char_start=30
    )

    edges = _reduce_same_fact([("table", [table_subtotal]), ("prose", [prose_subtotal])])

    assert edges == []


def test_contradicts_requires_matching_value_type() -> None:
    # A core attribute stated once as a percent and once as a currency (e.g. a
    # mis-canonicalized label) is not the same fact-slot disagreeing with
    # itself -- value_type belongs in the key the same way it already does
    # for same_fact.
    pct = _claim(attribute="revenue", raw="15%", normalized=15.0, value_type="percent", page=9)
    currency = _claim(
        attribute="revenue",
        raw="$15",
        normalized=15.0,
        value_type="currency",
        page=9,
        char_start=30,
    )

    edges = _reduce_same_fact([("table", [pct]), ("prose", [currency])])

    assert edges == []


def test_contradicts_requires_matching_period() -> None:
    # Canonicalization strips the year off the label (SIM-344: "Revenue | 2019F"
    # and "Revenue | 2020F" both -> "revenue"), so two years of the same metric
    # reach the reducer sharing attribute AND value_type. Without period in the
    # key they fuse into a false "these disagree" edge -- revenue was $100 in
    # 2019 and $120 in 2020, two different fact-slots, not a contradiction. E3
    # (SIM-345) makes the period the structured field the key can carry.
    fy2019 = _claim(
        attribute="revenue",
        raw="$100",
        normalized=100.0,
        value_type="currency",
        period_year=2019,
        period_kind="A",
        page=5,
    )
    fy2020 = _claim(
        attribute="revenue",
        raw="$120",
        normalized=120.0,
        value_type="currency",
        period_year=2020,
        period_kind="A",
        page=5,
        char_start=30,
    )

    edges = _reduce_same_fact([("table", [fy2019]), ("prose", [fy2020])])

    assert edges == []


def test_contradicts_still_fires_within_one_period() -> None:
    # The period key must not over-split: two tiers disagreeing on the SAME
    # metric in the SAME period is the genuine contradiction this pass exists to
    # catch, and adding period to the key must leave it intact.
    table_claim = _claim(
        attribute="revenue",
        raw="$15,000,000",
        normalized=15_000_000,
        value_type="currency",
        period_year=2024,
        period_kind="P",
        page=5,
    )
    prose_claim = _claim(
        attribute="revenue",
        raw="$12,000,000 excluding settlement",
        normalized=12_000_000,
        value_type="currency",
        period_year=2024,
        period_kind="P",
        page=5,
        char_start=200,
    )

    edges = _reduce_same_fact([("table", [table_claim]), ("prose", [prose_claim])])

    assert len(edges) == 1
    assert edges[0].type == "contradicts"
    assert {edges[0].from_, edges[0].to} == {
        table_claim.claim_ref,
        prose_claim.claim_ref,
    }


def test_missing_and_qualitative_claims_take_no_part() -> None:
    missing_claim = _claim(attribute="X", raw="", normalized=None, status="missing")
    qualitative_claim = _claim(attribute="X", raw="growing steadily", normalized=None)

    edges = _reduce_same_fact([("table", [missing_claim]), ("qualitative", [qualitative_claim])])

    assert edges == []


def test_different_pages_do_not_collide() -> None:
    claim_a = _claim(attribute="Revenue", raw="$100", normalized=100, page=1)
    claim_b = _claim(attribute="Revenue", raw="$100", normalized=100, page=2, char_start=50)

    edges = _reduce_same_fact([("table", [claim_a]), ("prose", [claim_b])])

    assert edges == []


def test_element_id_distinguishes_claims_sharing_page_and_attribute() -> None:
    claim_a = _claim(attribute="Revenue", raw="$100", normalized=100_000, char_start=0)
    claim_b = _claim(attribute="Revenue", raw="$200", normalized=200_000, char_start=50)

    assert element_id_for(claim_a) != element_id_for(claim_b)


# --------------------------------------------------------------------------- #
# SIM-344: _canonicalize_quantitative_claims -- must run BEFORE
# _reduce_same_fact, since edges are addressed by element_id_for(claim), which
# reads claim.attribute.
# --------------------------------------------------------------------------- #


def test_canonicalization_mutates_attribute_and_sets_attribute_raw(monkeypatch) -> None:
    claim = _claim(attribute="Revenue | 2019F", raw="$1", normalized=1.0)
    monkeypatch.setattr(
        extract_service, "canonicalize_attributes", lambda labels: {labels[0]: ("revenue", [])}
    )

    _canonicalize_quantitative_claims([("table", [claim])], FlagLog())

    assert claim.attribute == "revenue"
    assert claim.attribute_raw == "Revenue | 2019F"
    assert claim.flags == []


def test_canonicalization_attaches_and_logs_the_unmapped_flag(monkeypatch) -> None:
    claim = _claim(attribute="Adjusted Whatever", raw="$1", normalized=1.0)
    monkeypatch.setattr(
        extract_service,
        "canonicalize_attributes",
        lambda labels: {labels[0]: ("operating_metric", ["attribute_unmapped"])},
    )
    flag_log = FlagLog()

    _canonicalize_quantitative_claims([("table", [claim])], flag_log)

    assert claim.attribute == "operating_metric"
    assert claim.attribute_raw == "Adjusted Whatever"
    assert claim.flags == ["attribute_unmapped"]
    assert [entry.flag_type for entry in flag_log.entries] == ["attribute_unmapped"]
    assert flag_log.entries[0].detail == "Adjusted Whatever"


def test_canonicalization_skips_the_qualitative_tier(monkeypatch) -> None:
    qualitative_claim = _claim(
        attribute="on-site dry cleaning availability", raw="x", normalized=None
    )

    def _must_not_run(labels):
        raise AssertionError("qualitative attributes are out of scope for SIM-344")

    monkeypatch.setattr(extract_service, "canonicalize_attributes", _must_not_run)

    _canonicalize_quantitative_claims([("qualitative", [qualitative_claim])], FlagLog())

    assert qualitative_claim.attribute == "on-site dry cleaning availability"
    assert qualitative_claim.attribute_raw is None


def test_canonicalization_makes_no_call_when_nothing_is_in_scope(monkeypatch) -> None:
    def _must_not_run(labels):
        raise AssertionError("empty scope should cost nothing")

    monkeypatch.setattr(extract_service, "canonicalize_attributes", _must_not_run)

    _canonicalize_quantitative_claims([], FlagLog())
    _canonicalize_quantitative_claims(
        [("qualitative", [_claim(attribute="x", raw="y", normalized=None)])], FlagLog()
    )


def test_gross_and_net_margin_no_longer_false_contradict_after_canonicalization(
    monkeypatch,
) -> None:
    # Reproduces the review's residual: before CORE_ATTRIBUTES split "margin"
    # into gross_margin/net_margin/ebitda_margin, a gross-margin claim and a
    # net-margin claim on the same page both canonicalized to the single
    # "margin" bucket and false-contradicted (contradicts is keyed on
    # attribute, and a coarse bucket is not one fact-slot). With the split,
    # each keeps its own canonical name, so the reducer no longer collapses
    # them.
    gross = _claim(
        attribute="Gross Margin", raw="40%", normalized=40.0, value_type="percent", page=7
    )
    net = _claim(
        attribute="Net Margin",
        raw="12%",
        normalized=12.0,
        value_type="percent",
        page=7,
        char_start=30,
    )
    monkeypatch.setattr(
        extract_service,
        "canonicalize_attributes",
        lambda labels: {"Gross Margin": ("gross_margin", []), "Net Margin": ("net_margin", [])},
    )
    tier_claims = [("table", [gross]), ("prose", [net])]

    _canonicalize_quantitative_claims(tier_claims, FlagLog())
    edges = _reduce_same_fact(tier_claims)

    assert edges == []


def test_canonicalization_before_reduction_keeps_edges_consistent_with_final_attributes(
    monkeypatch,
) -> None:
    # SIM-365: edges now anchor on claim_ref, which is positional and
    # attribute-independent, so canonicalization (which rewrites attribute) can no
    # longer desync an edge endpoint from the claim it names -- the drift the old
    # attribute-keyed element_id risked if canonicalization ran after the reducer.
    # Guards that the same_fact edge is still produced and its endpoints are the
    # claims' claim_ref, not their (canonicalized) attribute.
    table_claim = _claim(attribute="Revenue | 2024F", raw="$15,295", normalized=15295000, page=3)
    prose_claim = _claim(
        attribute="total revenue for fiscal 2024",
        raw="fifteen million two hundred ninety-five thousand",
        normalized=15295000,
        page=3,
        char_start=500,
    )
    monkeypatch.setattr(
        extract_service,
        "canonicalize_attributes",
        lambda labels: {label: ("revenue", []) for label in labels},
    )
    tier_claims = [("table", [table_claim]), ("prose", [prose_claim])]

    _canonicalize_quantitative_claims(tier_claims, FlagLog())
    edges = _reduce_same_fact(tier_claims)

    assert len(edges) == 1
    edge = edges[0]
    assert edge.from_ == prose_claim.claim_ref
    assert edge.to == table_claim.claim_ref
    # Both canonicalized to "revenue" before the reducer grouped them, but the
    # edge id itself carries no attribute -- the inverse of the old assertion.
    assert table_claim.attribute == "revenue"
    assert "revenue" not in edge.to

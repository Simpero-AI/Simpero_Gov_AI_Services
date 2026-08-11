"""SIM-359/SIM-388: the binding-audit wiring in extract_service -- routing
into the three families (scale_absurd / bound_as_point / semantic sample) and
_audit_claims' flag-only application, including the semantic sample's
escalate-to-Opus-on-a-flag behavior. The auditor itself (verify.audit_claim)
is mocked; these test the wiring, not the model."""

from __future__ import annotations

import parser_service.extract_service as extract_service
from parser_service.emit import Claim, ClaimValue, FlagLog, PdfLocation, Status
from parser_service.scale import ScaleSource
from parser_service.schemas import PageIndex
from parser_service.verify import DEFAULT_MODEL, AuditVerdict

_SEMANTIC_MODEL = extract_service._SEMANTIC_AUDIT_MODEL


def _claim(
    *,
    scale_source: ScaleSource = "column_header",
    normalized: float | None = 15_295_000.0,
    status: Status = "cited",
    char_start: int = 10,
    char_end: int = 20,
):
    return Claim(
        entity="Target Co",
        attribute="revenue",
        value=ClaimValue(
            raw="$15,295",
            normalized=normalized,
            unit="USD",
            value_type="currency",
            scale_multiplier=1000.0,
            scale_source=scale_source,
        ),
        location=PdfLocation(file="cim.pdf", page=1, char_start=char_start, char_end=char_end),
        status=status,
    )


def _page(text: str = "x" * 200) -> PageIndex:
    return PageIndex(page=1, text=text, char_map=[])


# --------------------------------------------------------------------------- #
# Routing predicates.
# --------------------------------------------------------------------------- #


def test_scale_absurd_candidate_routes_only_page_header_magnitudes() -> None:
    assert extract_service._is_scale_absurd_candidate(_claim(scale_source="page_header")) is True
    # a value carrying its own/column scale is trusted -- not the scale_absurd risk
    assert extract_service._is_scale_absurd_candidate(_claim(scale_source="column_header")) is False
    assert (
        extract_service._is_scale_absurd_candidate(_claim(scale_source="explicit_in_value"))
        is False
    )
    # no magnitude -> nothing to sanity-check
    assert (
        extract_service._is_scale_absurd_candidate(
            _claim(scale_source="page_header", normalized=None)
        )
        is False
    )
    # a missing claim has no resolved span to read
    assert (
        extract_service._is_scale_absurd_candidate(
            _claim(scale_source="page_header", status="missing")
        )
        is False
    )


def test_bound_as_point_candidate_matches_a_bound_phrase_or_symbol() -> None:
    assert extract_service._is_bound_as_point_candidate("revenue was greater than $5M") is True
    assert extract_service._is_bound_as_point_candidate("occupancy was at least 90%") is True
    assert extract_service._is_bound_as_point_candidate("capacity up to 1,200 rooms") is True
    assert extract_service._is_bound_as_point_candidate("EBITDA margin > 20%") is True
    assert extract_service._is_bound_as_point_candidate("leverage < 3.5x") is True
    assert extract_service._is_bound_as_point_candidate("revenue was $15,295 total") is False


# --------------------------------------------------------------------------- #
# _audit_claims routing + flagging.
# --------------------------------------------------------------------------- #


def test_scale_absurd_route_is_audited_on_opus_directly(monkeypatch) -> None:
    seen_models: list[str] = []

    def fake(*, model, **_kwargs):
        seen_models.append(model)
        return AuditVerdict(verdict="scale_absurd", evidence="a regional casino")

    monkeypatch.setattr(extract_service, "audit_claim", fake)

    claim = _claim(scale_source="page_header")
    flag_log = FlagLog(run_id="t")
    extract_service._audit_claims([claim], [_page()], flag_log, workers=1)

    assert seen_models == [DEFAULT_MODEL]
    assert "binding_unsupported" in claim.flags
    assert len(flag_log.entries) == 1
    entry = flag_log.entries[0]
    assert entry.flag_type == "binding_unsupported"
    # the seven verdict modes collapse onto one flag; the mode rides in the detail
    assert entry.detail is not None and "scale_absurd" in entry.detail


def test_bound_as_point_routes_regardless_of_scale_source(monkeypatch) -> None:
    seen_models: list[str] = []

    def fake(*, model, **_kwargs):
        seen_models.append(model)
        return AuditVerdict(verdict="bound_as_point", evidence="greater than")

    monkeypatch.setattr(extract_service, "audit_claim", fake)

    # column_header, not page_header -- would never reach scale_absurd's gate.
    claim = _claim(scale_source="column_header", char_start=0, char_end=25)
    page = _page("revenue was greater than $5M more text padding out the page")
    flag_log = FlagLog(run_id="t")
    extract_service._audit_claims([claim], [page], flag_log, workers=1)

    assert seen_models == [DEFAULT_MODEL]
    assert "binding_unsupported" in claim.flags


def test_an_ordinary_claim_reaches_the_semantic_sample_on_the_cheap_model(monkeypatch) -> None:
    seen_models: list[str] = []

    def fake(*, model, **_kwargs):
        seen_models.append(model)
        return AuditVerdict(verdict="supported", evidence="")

    monkeypatch.setattr(extract_service, "audit_claim", fake)

    # Neither scale_absurd (column_header) nor bound_as_point (no bound phrase)
    # -- SIM-388's whole point is that this claim now reaches an audit call at
    # all, on the cheap model, instead of being silently unrouted.
    claim = _claim(scale_source="column_header", char_start=0, char_end=5)
    page = _page("no bound phrase here, just an ordinary revenue figure")
    flag_log = FlagLog(run_id="t")
    extract_service._audit_claims([claim], [page], flag_log, workers=1)

    assert seen_models == [_SEMANTIC_MODEL]
    assert claim.flags == []
    assert flag_log.entries == []


def test_a_flagged_semantic_verdict_escalates_to_opus_and_only_the_escalation_is_written(
    monkeypatch,
) -> None:
    calls: list[str] = []

    def fake(*, model, **_kwargs):
        calls.append(model)
        if model == _SEMANTIC_MODEL:
            # The cheap pass finds a candidate...
            return AuditVerdict(verdict="attribute_mismatch", evidence="cheap-model guess")
        # ...but only the Opus confirmation is ever written.
        return AuditVerdict(verdict="attribute_mismatch", evidence="opus-confirmed")

    monkeypatch.setattr(extract_service, "audit_claim", fake)

    claim = _claim(scale_source="column_header", char_start=0, char_end=5)
    page = _page("no bound phrase here, just an ordinary revenue figure")
    flag_log = FlagLog(run_id="t")
    extract_service._audit_claims([claim], [page], flag_log, workers=1)

    assert calls == [_SEMANTIC_MODEL, DEFAULT_MODEL]
    assert "binding_unsupported" in claim.flags
    assert len(flag_log.entries) == 1
    assert flag_log.entries[0].detail == "attribute_mismatch: opus-confirmed"


def test_a_supported_semantic_verdict_never_escalates(monkeypatch) -> None:
    calls: list[str] = []

    def fake(*, model, **_kwargs):
        calls.append(model)
        return AuditVerdict(verdict="supported", evidence="")

    monkeypatch.setattr(extract_service, "audit_claim", fake)

    claim = _claim(scale_source="column_header", char_start=0, char_end=5)
    page = _page("no bound phrase here, just an ordinary revenue figure")
    flag_log = FlagLog(run_id="t")
    extract_service._audit_claims([claim], [page], flag_log, workers=1)

    assert calls == [_SEMANTIC_MODEL]


def test_semantic_sample_is_capped_and_deterministic(monkeypatch) -> None:
    monkeypatch.setattr(extract_service, "_SEMANTIC_SAMPLE_CAP", 3)

    # 10 distinct spans (distinct char_start/char_end), so 10 distinct
    # element_ids and 10 distinct sample-key hashes to choose among.
    claims = [_claim(scale_source="column_header", char_start=i, char_end=i + 1) for i in range(10)]
    page = _page("0123456789" * 5)

    def make_fake(log: list[str]):
        def fake(*, model, quote, **_kwargs):
            log.append(quote)
            return AuditVerdict(verdict="supported", evidence="")

        return fake

    seen_first: list[str] = []
    monkeypatch.setattr(extract_service, "audit_claim", make_fake(seen_first))
    extract_service._audit_claims(claims, [page], FlagLog(run_id="t1"), workers=1)

    # Re-running against the same claims audits the same sample, in the same
    # order -- deterministic because it is hashed on element_id_for, not
    # incidental list or thread-completion ordering.
    seen_second: list[str] = []
    monkeypatch.setattr(extract_service, "audit_claim", make_fake(seen_second))
    extract_service._audit_claims(claims, [page], FlagLog(run_id="t2"), workers=1)

    assert len(seen_first) == 3
    assert seen_first == seen_second


def test_a_missing_claim_is_never_routed(monkeypatch) -> None:
    def fake(*, model, **_kwargs):  # pragma: no cover - must not be called
        raise AssertionError("a claim with no resolved span must not reach an audit call")

    monkeypatch.setattr(extract_service, "audit_claim", fake)

    missing = _claim(status="missing")
    page = _page()
    flag_log = FlagLog(run_id="t")
    extract_service._audit_claims([missing], [page], flag_log, workers=1)

    assert missing.flags == []
    assert flag_log.entries == []

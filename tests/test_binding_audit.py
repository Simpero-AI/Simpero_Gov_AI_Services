"""SIM-359: the binding-audit wiring in extract_service -- _should_audit routing
and _audit_claims flag-only application. The auditor itself (verify.audit_claim)
is mocked; these test the wiring, not the model."""

from __future__ import annotations

import parser_service.extract_service as extract_service
from parser_service.emit import Claim, ClaimValue, FlagLog, PdfLocation, Status
from parser_service.scale import ScaleSource
from parser_service.schemas import PageIndex
from parser_service.verify import AuditVerdict


def _claim(
    *, scale_source: ScaleSource, normalized: float | None = 15_295_000.0, status: Status = "cited"
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
        location=PdfLocation(file="cim.pdf", page=1, char_start=10, char_end=20),
        status=status,
    )


def test_should_audit_routes_only_page_header_magnitudes() -> None:
    assert extract_service._should_audit(_claim(scale_source="page_header")) is True
    # a value carrying its own/column scale is trusted -- not the scale_absurd risk
    assert extract_service._should_audit(_claim(scale_source="column_header")) is False
    assert extract_service._should_audit(_claim(scale_source="explicit_in_value")) is False
    # no magnitude -> nothing to sanity-check
    assert (
        extract_service._should_audit(_claim(scale_source="page_header", normalized=None)) is False
    )
    # a missing claim has no resolved span to read
    assert (
        extract_service._should_audit(_claim(scale_source="page_header", status="missing")) is False
    )


def test_audit_flags_binding_unsupported_and_carries_the_mode(monkeypatch) -> None:
    monkeypatch.setattr(
        extract_service,
        "audit_claim",
        lambda **_: AuditVerdict(verdict="scale_absurd", evidence="a regional casino"),
    )
    claim = _claim(scale_source="page_header")
    page = PageIndex(page=1, text="x" * 50, char_map=[])
    flag_log = FlagLog(run_id="t")
    extract_service._audit_claims([claim], [page], flag_log, workers=1)

    assert "binding_unsupported" in claim.flags
    assert len(flag_log.entries) == 1
    entry = flag_log.entries[0]
    assert entry.flag_type == "binding_unsupported"
    # the six verdict modes collapse onto one flag; the mode rides in the detail
    assert entry.detail is not None and "scale_absurd" in entry.detail


def test_audit_leaves_supported_and_unrouted_claims_untouched(monkeypatch) -> None:
    monkeypatch.setattr(
        extract_service,
        "audit_claim",
        lambda **_: AuditVerdict(verdict="supported", evidence=""),
    )
    routed = _claim(scale_source="page_header")  # audited, but supported
    unrouted = _claim(scale_source="column_header")  # never routed
    page = PageIndex(page=1, text="x" * 50, char_map=[])
    flag_log = FlagLog(run_id="t")
    extract_service._audit_claims([routed, unrouted], [page], flag_log, workers=1)

    assert routed.flags == []
    assert unrouted.flags == []
    assert flag_log.entries == []

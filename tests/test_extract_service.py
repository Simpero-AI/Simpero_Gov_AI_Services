"""parser_service.extract_service.extract_claims -- the shared extraction
entry point scripts/emit_claims.py and POST /extract both call (SIM-340/E5).

The acceptance bar from the ticket: the CLI and the callable entry point must
be provably the same code path, and the callable form must return a payload
identical to what the CLI would print for the same input.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Literal

import pytest

from parser_service import extract_service
from parser_service.emit import ClaimValue, PdfLocation
from parser_service.scale import ValueType
from scripts import emit_claims


class _Doc:
    pass


class _Result:
    document = _Doc()
    pages: list = []
    sha256 = "0" * 64


class _Page:
    def __init__(self, page: int) -> None:
        self.page = page


def _table_claim(
    page: int, attribute: str = "revenue", value_type: ValueType = "currency"
) -> extract_service.Claim:
    return extract_service.Claim(
        entity="ACME",
        attribute=attribute,
        value=ClaimValue(raw="$1", normalized=1.0, unit="$", value_type=value_type),
        location=PdfLocation(file="cim.pdf", page=page, char_start=0, char_end=2),
        status="proposed",
    )


def _page_index(page: int) -> extract_service.PageIndex:
    # A minimal real PageIndex for the _prose_tiers unit tests: those stub the
    # extractors and blocks_on_page, so only `.page` is ever read, but the helper
    # is typed as PageIndex to match the signature the fan-out actually takes.
    return extract_service.PageIndex(page=page, text="", char_map=[])


def test_prose_without_a_key_raises_before_any_parsing(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)

    def _must_not_run(*_a, **_k):
        raise AssertionError("parse must not start when the key is absent")

    monkeypatch.setattr(extract_service, "parse_pdf_bytes", _must_not_run)

    with pytest.raises(extract_service.ProseCredentialMissing):
        extract_service.extract_claims(
            b"%PDF-1.4 stub",
            entity="ACME",
            run_id="run-1",
            correlation_id="doc-1",
            source_file="cim.pdf",
            prose=True,
        )


def test_qualitative_implies_prose_for_the_credential_check(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.setattr(
        extract_service, "parse_pdf_bytes", lambda *_a, **_k: AssertionError("unreachable")
    )

    with pytest.raises(extract_service.ProseCredentialMissing):
        extract_service.extract_claims(
            b"%PDF-1.4 stub",
            entity="ACME",
            run_id="run-1",
            correlation_id="doc-1",
            source_file="cim.pdf",
            qualitative=True,
        )


def test_table_only_tier_runs_without_touching_the_credential_check(monkeypatch) -> None:
    def _fake_key_check() -> bool:  # pragma: no cover - must not be called
        raise AssertionError("the table tier must not consult the credential")

    monkeypatch.setattr(extract_service, "api_key_present", _fake_key_check)
    monkeypatch.setattr(extract_service, "parse_pdf_bytes", lambda _b: _Result())
    monkeypatch.setattr(extract_service, "extract_tables", lambda *_a, **_k: [])

    payload = extract_service.extract_claims(
        b"%PDF-1.4 stub",
        entity="ACME",
        run_id="run-1",
        correlation_id="doc-1",
        source_file="cim.pdf",
    )
    assert payload["claims"] == []


def test_correlation_id_is_not_written_into_the_returned_payload(monkeypatch) -> None:
    # correlation_id/run_id identify the caller's run for logging/correlation --
    # the C3 claims contract has no slot for either, so only run_id (which the
    # contract's payload shape already carries) should appear. Named
    # correlation_id, not document_id, because that term already means the
    # content hash elsewhere (emit_chunks sets document_id = sha256).
    monkeypatch.setattr(extract_service, "parse_pdf_bytes", lambda _b: _Result())
    monkeypatch.setattr(extract_service, "extract_tables", lambda *_a, **_k: [])

    payload = extract_service.extract_claims(
        b"%PDF-1.4 stub",
        entity="ACME",
        run_id="run-1",
        correlation_id="a-correlation-id-that-must-not-leak",
        source_file="cim.pdf",
    )
    assert "correlation_id" not in payload
    assert "document_id" not in payload
    assert "a-correlation-id-that-must-not-leak" not in json.dumps(payload)


def test_source_file_stays_none_when_the_caller_has_none(monkeypatch) -> None:
    # A caller with no real filename must not fall back to correlation_id --
    # that value is a correlation token, not a filename, and stamping it into
    # every claim's debug `file` field would misrepresent it as one.
    monkeypatch.setattr(extract_service, "parse_pdf_bytes", lambda _b: _Result())
    monkeypatch.setattr(extract_service, "extract_tables", lambda *_a, **_k: [])

    payload = extract_service.extract_claims(
        b"%PDF-1.4 stub",
        entity="ACME",
        run_id="run-1",
        correlation_id="run-token-not-a-filename",
    )
    assert payload["source_file"] is None
    assert "run-token-not-a-filename" not in json.dumps(payload)


def test_a_bad_table_does_not_abort_the_document_and_is_reported_skipped(monkeypatch) -> None:
    # One table's failure must not take its sibling tables, other pages, or the
    # whole document down -- and the affected page is still named in
    # skipped_pages so an HTTP caller (no stderr to read) can see it.
    class _MultiPageResult:
        document = _Doc()
        pages = [_Page(1), _Page(2)]
        sha256 = "0" * 64

    def _fake_tables_on_page(_tables, page_no: int) -> list[str]:
        return {1: ["t1-bad", "t1-good"], 2: ["t2-good"]}[page_no]

    def _fake_claims_from_table(table, page, *, entity, file, flag_log):
        if table == "t1-bad":
            raise ValueError("a malformed table")
        return [_table_claim(page.page, attribute=table)]

    monkeypatch.setattr(extract_service, "parse_pdf_bytes", lambda _b: _MultiPageResult())
    monkeypatch.setattr(extract_service, "extract_tables", lambda *_a, **_k: [])
    monkeypatch.setattr(extract_service, "tables_on_page", _fake_tables_on_page)
    monkeypatch.setattr(extract_service, "claims_from_table", _fake_claims_from_table)

    payload = extract_service.extract_claims(
        b"%PDF-1.4 stub",
        entity="ACME",
        run_id="run-1",
        correlation_id="doc-1",
        source_file="cim.pdf",
    )
    assert payload["skipped_pages"] == [
        {"page": 1, "tier": "tables", "reason": "ValueError: a malformed table"}
    ]
    attributes = {c["attribute"] for c in payload["claims"]}
    # t1-good survives its sibling t1-bad's failure; t2-good is untouched.
    assert attributes == {"t1-good", "t2-good"}


def test_skipped_pages_is_empty_when_nothing_failed(monkeypatch) -> None:
    monkeypatch.setattr(extract_service, "parse_pdf_bytes", lambda _b: _Result())
    monkeypatch.setattr(extract_service, "extract_tables", lambda *_a, **_k: [])

    payload = extract_service.extract_claims(
        b"%PDF-1.4 stub",
        entity="ACME",
        run_id="run-1",
        correlation_id="doc-1",
        source_file="cim.pdf",
    )
    assert payload["skipped_pages"] == []


def test_canonicalize_attributes_without_a_key_raises_before_any_parsing(monkeypatch) -> None:
    # SIM-344's pass calls the Anthropic API just as the prose tiers do, and it
    # is independent of them -- so it must fail closed at the door the same way
    # even when prose/complete/qualitative are all off.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)

    def _must_not_run(*_a, **_k):
        raise AssertionError("parse must not start when the key is absent")

    monkeypatch.setattr(extract_service, "parse_pdf_bytes", _must_not_run)

    with pytest.raises(extract_service.ProseCredentialMissing):
        extract_service.extract_claims(
            b"%PDF-1.4 stub",
            entity="ACME",
            run_id="run-1",
            correlation_id="doc-1",
            source_file="cim.pdf",
            canonicalize_attributes=True,
        )


def test_canonicalize_attributes_maps_table_claims_end_to_end(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    class _OnePageResult:
        document = _Doc()
        pages = [_Page(1)]
        sha256 = "0" * 64

    monkeypatch.setattr(extract_service, "parse_pdf_bytes", lambda _b: _OnePageResult())
    monkeypatch.setattr(extract_service, "extract_tables", lambda *_a, **_k: [])
    monkeypatch.setattr(extract_service, "tables_on_page", lambda *_a, **_k: ["t1"])
    monkeypatch.setattr(
        extract_service,
        "claims_from_table",
        lambda table, page, *, entity, file, flag_log: [
            _table_claim(page.page, attribute="Revenue | 2019F")
        ],
    )
    monkeypatch.setattr(
        extract_service,
        "canonicalize_attributes",
        lambda labels: {"Revenue | 2019F": ("revenue", [])},
    )

    payload = extract_service.extract_claims(
        b"%PDF-1.4 stub",
        entity="ACME",
        run_id="run-1",
        correlation_id="doc-1",
        source_file="cim.pdf",
        canonicalize_attributes=True,
    )

    assert len(payload["claims"]) == 1
    claim = payload["claims"][0]
    assert claim["attribute"] == "revenue"
    assert claim["attribute_raw"] == "Revenue | 2019F"


def test_canonicalize_attributes_skips_text_and_date_claims(monkeypatch) -> None:
    # SIM-384: a date ("Opening Date: March 2024") or a magnitude-guard-downgraded
    # text claim (a bare section-header row like "Current assets:", which carries
    # no parseable number) is not a financial-statement or sector metric, so its
    # label must never reach the attribute-mapping prompt -- doing so is what
    # flooded core_unmapped/operating_metric with dates and section headers on a
    # real CIM. Only the currency claim's label should be sent to the model; the
    # other two keep their document-supplied label untouched.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    class _OnePageResult:
        document = _Doc()
        pages = [_Page(1)]
        sha256 = "0" * 64

    monkeypatch.setattr(extract_service, "parse_pdf_bytes", lambda _b: _OnePageResult())
    monkeypatch.setattr(extract_service, "extract_tables", lambda *_a, **_k: [])
    monkeypatch.setattr(extract_service, "tables_on_page", lambda *_a, **_k: ["t1"])
    monkeypatch.setattr(
        extract_service,
        "claims_from_table",
        lambda table, page, *, entity, file, flag_log: [
            _table_claim(page.page, attribute="Revenue | 2019F", value_type="currency"),
            _table_claim(page.page, attribute="Opening Date | 2019F", value_type="date"),
            _table_claim(page.page, attribute="Current assets:", value_type="text"),
        ],
    )

    def _fake_canonicalize(labels):
        assert labels == ["Revenue | 2019F"], "only the currency claim's label may be mapped"
        return {"Revenue | 2019F": ("revenue", [])}

    monkeypatch.setattr(extract_service, "canonicalize_attributes", _fake_canonicalize)

    payload = extract_service.extract_claims(
        b"%PDF-1.4 stub",
        entity="ACME",
        run_id="run-1",
        correlation_id="doc-1",
        source_file="cim.pdf",
        canonicalize_attributes=True,
    )

    by_attribute = {c.get("attribute_raw", c["attribute"]): c for c in payload["claims"]}
    assert by_attribute["Revenue | 2019F"]["attribute"] == "revenue"
    assert by_attribute["Opening Date | 2019F"]["attribute"] == "Opening Date | 2019F"
    assert "attribute_raw" not in by_attribute["Opening Date | 2019F"]
    assert by_attribute["Current assets:"]["attribute"] == "Current assets:"
    assert "attribute_raw" not in by_attribute["Current assets:"]


def test_canonicalize_attributes_failure_does_not_abort_the_document(monkeypatch) -> None:
    # Unlike the sibling tiers (table/prose/completeness), the canonicalization
    # pass had no try/except -- a transient API error discarded every
    # already-emitted claim, including pure-table claims that never needed the
    # API. It must degrade the same way a bad table does: report the failure
    # and keep what was already extracted.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    class _OnePageResult:
        document = _Doc()
        pages = [_Page(1)]
        sha256 = "0" * 64

    monkeypatch.setattr(extract_service, "parse_pdf_bytes", lambda _b: _OnePageResult())
    monkeypatch.setattr(extract_service, "extract_tables", lambda *_a, **_k: [])
    monkeypatch.setattr(extract_service, "tables_on_page", lambda *_a, **_k: ["t1"])
    monkeypatch.setattr(
        extract_service,
        "claims_from_table",
        lambda table, page, *, entity, file, flag_log: [
            _table_claim(page.page, attribute="Revenue | 2019F")
        ],
    )

    def _fake_canonicalize(labels):
        raise RuntimeError("transient API error")

    monkeypatch.setattr(extract_service, "canonicalize_attributes", _fake_canonicalize)

    payload = extract_service.extract_claims(
        b"%PDF-1.4 stub",
        entity="ACME",
        run_id="run-1",
        correlation_id="doc-1",
        source_file="cim.pdf",
        canonicalize_attributes=True,
    )

    assert len(payload["claims"]) == 1
    claim = payload["claims"][0]
    assert claim["attribute"] == "Revenue | 2019F"
    assert "attribute_raw" not in claim
    assert payload["skipped_pages"] == [
        {"page": 0, "tier": "attribute_mapping", "reason": "RuntimeError: transient API error"}
    ]


def test_canonicalize_attributes_defaults_to_off(monkeypatch) -> None:
    # Table-only extraction stays credential-free unless the caller opts in --
    # this is the regression this ticket must not introduce.
    def _fake_key_check() -> bool:  # pragma: no cover - must not be called
        raise AssertionError("table-only extraction must not consult the credential")

    monkeypatch.setattr(extract_service, "api_key_present", _fake_key_check)
    monkeypatch.setattr(extract_service, "parse_pdf_bytes", lambda _b: _Result())
    monkeypatch.setattr(extract_service, "extract_tables", lambda *_a, **_k: [])

    payload = extract_service.extract_claims(
        b"%PDF-1.4 stub",
        entity="ACME",
        run_id="run-1",
        correlation_id="doc-1",
        source_file="cim.pdf",
    )
    assert payload["claims"] == []


def test_cli_and_direct_call_produce_an_identical_payload_for_the_same_input(
    monkeypatch, tmp_path, capsys
) -> None:
    # The ticket's core acceptance criterion: emit_claims.py and the shared
    # entry point share one code path, so a caller of either gets the same
    # claims payload for the same document/entity/run_id.
    monkeypatch.setattr(extract_service, "parse_pdf_bytes", lambda _b: _Result())
    monkeypatch.setattr(extract_service, "extract_tables", lambda *_a, **_k: [])

    pdf = tmp_path / "cim.pdf"
    pdf.write_bytes(b"%PDF-1.4 stub")

    emit_claims.main([str(pdf), "--entity", "ACME", "--run-id", "run-shared"])
    cli_payload = json.loads(capsys.readouterr().out)

    direct_payload = extract_service.extract_claims(
        pdf.read_bytes(),
        entity="ACME",
        run_id="run-shared",
        correlation_id="cim",
        source_file="cim.pdf",
    )

    assert cli_payload == direct_payload


def test_is_systemic_prose_loss_large_share_and_total_loss():
    fn = extract_service._is_systemic_prose_loss
    # A large share of a big prose set escalates; a one-off does not.
    assert fn(20, 100) is True  # 20% of 100
    assert fn(1, 10) is False  # one-off (1 < max(3, 2))
    assert fn(3, 10) is True  # hits the floor of 3
    # Total prose loss escalates even on a set too small to clear the floor.
    assert fn(2, 2) is True  # 100% of 2 prose pages, though 2 < max(3, 0)
    # Nothing lost, or no prose pages -> never systemic.
    assert fn(0, 100) is False
    assert fn(0, 0) is False
    # lost is a subset of prose_pages by construction; be defensive anyway.
    assert fn(5, 5) is True


def _dashboard_claim(
    attribute: str,
    *,
    entity: str = "ACME",
    claim_kind: Literal["quantitative", "qualitative"] | None = None,
    value_type: ValueType = "currency",
) -> extract_service.Claim:
    return extract_service.Claim(
        entity=entity,
        attribute=attribute,
        value=ClaimValue(raw="x", normalized=None, unit=None, value_type=value_type),
        location=PdfLocation(file="cim.pdf", page=1, char_start=0, char_end=1),
        status="proposed",
        claim_kind=claim_kind,
    )


def test_dashboard_metrics_present_excludes_catchall_and_qualitative() -> None:
    # metric_order must list only canonical quantitative metrics: the catch-all
    # buckets and qualitative assertions (open noun phrases, never canonicalized)
    # must not leak their raw attributes into the dashboard.
    claims = [
        _dashboard_claim("revenue"),
        _dashboard_claim("ebitda", claim_kind="quantitative"),
        _dashboard_claim(extract_service.OPERATING_METRIC),
        _dashboard_claim(extract_service.CORE_UNMAPPED),
        _dashboard_claim(
            "on-site dry cleaning availability",
            claim_kind="qualitative",
            value_type="text",
        ),
    ]

    assert extract_service._dashboard_metrics_present(claims) == ["ebitda", "revenue"]


def test_dashboard_entity_counts_excludes_qualitative_entities() -> None:
    # organize_claims ranks entities by fact count and caps the list, so a
    # qualitative tier's free-text entities (a market, a competitor) must not be
    # counted -- they would compete for slots and evict real financial entities.
    claims = [
        _dashboard_claim("revenue", entity="AcmeCo"),
        _dashboard_claim("ebitda", entity="AcmeCo"),
        _dashboard_claim(
            "outsources cleaning",
            entity="A Competitor",
            claim_kind="qualitative",
            value_type="text",
        ),
        _dashboard_claim(
            "target market note", entity="The Market", claim_kind="qualitative", value_type="text"
        ),
    ]

    assert extract_service._dashboard_entity_counts(claims) == {"AcmeCo": 2}


# --------------------------------------------------------------------------- #
# The numeric prose tier and the qualitative tier share ONE fan-out pool (they
# read only `blocks`, never each other's claims). The merge is a pure wall-clock
# win -- same calls, same prompts -- so these guard the invariants that make it
# safe: peak concurrency stays at EXTRACT_WORKERS, per-tier failure accounting
# survives, and completeness still sees exactly table+prose (never qualitative).
# --------------------------------------------------------------------------- #


def _prose_extractor(
    kind: str, *, barrier: threading.Barrier | None = None, counter: dict | None = None
):
    """A fake prose/qualitative extractor: returns one claim tagged with `kind`.
    Optionally rendezvous on `barrier` (to prove two tiers run at once) or record
    live/peak concurrency in `counter` (to prove the worker cap holds)."""

    def run(_blocks, page, *, entity_hint, file, flag_log, client):
        if barrier is not None:
            barrier.wait()  # completes only if a task from the OTHER tier is also live
        if counter is not None:
            with counter["lock"]:
                counter["live"] += 1
                counter["peak"] = max(counter["peak"], counter["live"])
            time.sleep(0.02)  # hold the slot so overlap is actually exercised
            with counter["lock"]:
                counter["live"] -= 1
        return [_table_claim(page.page, attribute=kind)]

    return run


def test_prose_tiers_runs_the_two_passes_concurrently_in_one_pool(monkeypatch) -> None:
    # The point of the merge: a numeric page and a qualitative page run at the SAME
    # time. Two pools drained back-to-back could never both reach this barrier, so
    # the rendezvous completing (no skip) is what proves the single-pool behaviour.
    barrier = threading.Barrier(2, timeout=5)
    monkeypatch.setattr(extract_service, "blocks_on_page", lambda _b, _p: None)
    monkeypatch.setattr(
        extract_service, "claims_from_prose", _prose_extractor("prose", barrier=barrier)
    )
    monkeypatch.setattr(
        extract_service, "assertions_from_prose", _prose_extractor("qualitative", barrier=barrier)
    )

    tiers = extract_service._prose_tiers(
        ["prose", "qualitative"],
        [_page_index(1)],
        blocks=None,
        entity="ACME",
        file="cim.pdf",
        flag_log=extract_service.FlagLog(run_id="run-1"),
        workers=2,
        client=None,
    )

    # A timed-out barrier (sequential pools) would surface as skipped pages, not claims.
    assert tiers["prose"][1] == [] and tiers["qualitative"][1] == []
    assert len(tiers["prose"][0]) == 1 and len(tiers["qualitative"][0]) == 1


def test_prose_tiers_never_exceeds_the_worker_cap_across_both_passes(monkeypatch) -> None:
    # 2 * len(pages) tasks share one pool, but peak concurrency stays at `workers`
    # -- the merge must not double the Anthropic calls in flight, since that width
    # is what the account's rate-limit headroom (EXTRACT_WORKERS) is sized for.
    counter = {"lock": threading.Lock(), "live": 0, "peak": 0}
    monkeypatch.setattr(extract_service, "blocks_on_page", lambda _b, _p: None)
    monkeypatch.setattr(
        extract_service, "claims_from_prose", _prose_extractor("prose", counter=counter)
    )
    monkeypatch.setattr(
        extract_service, "assertions_from_prose", _prose_extractor("qualitative", counter=counter)
    )

    extract_service._prose_tiers(
        ["prose", "qualitative"],
        [_page_index(n) for n in (1, 2, 3, 4)],  # 8 tasks
        blocks=None,
        entity="ACME",
        file="cim.pdf",
        flag_log=extract_service.FlagLog(run_id="run-1"),
        workers=3,
        client=None,
    )

    assert counter["peak"] <= 3, "one merged pool must not exceed EXTRACT_WORKERS"


def test_prose_tiers_attributes_each_failure_to_its_own_tier(monkeypatch) -> None:
    # A page that fails the numeric pass but not the qualitative one is a skip
    # record under "prose" only -- the per-tier accounting the single pool must
    # preserve, since skipped_pages is the only loss signal an HTTP caller can read.
    def _boom(_blocks, page, *, entity_hint, file, flag_log, client):
        raise RuntimeError("numeric boom")

    monkeypatch.setattr(extract_service, "blocks_on_page", lambda _b, _p: None)
    monkeypatch.setattr(extract_service, "claims_from_prose", _boom)
    monkeypatch.setattr(extract_service, "assertions_from_prose", _prose_extractor("qualitative"))

    tiers = extract_service._prose_tiers(
        ["prose", "qualitative"],
        [_page_index(7)],
        blocks=None,
        entity="ACME",
        file="cim.pdf",
        flag_log=extract_service.FlagLog(run_id="run-1"),
        workers=4,
        client=None,
    )

    assert tiers["prose"][0] == []
    assert [(s.page, s.tier) for s in tiers["prose"][1]] == [(7, "prose")]
    assert "numeric boom" in tiers["prose"][1][0].reason
    assert len(tiers["qualitative"][0]) == 1
    assert tiers["qualitative"][1] == []


def test_prose_tiers_a_page_failing_both_passes_is_one_skip_per_tier(monkeypatch) -> None:
    # The other half of per-tier accounting: a page whose numeric AND qualitative calls
    # both fail is TWO SkippedPage records -- one under each tier -- not one merged loss.
    # skipped_pages is keyed per-(page, tier), so both entry points can name each gap.
    def _boom(message: str):
        def run(_blocks, page, *, entity_hint, file, flag_log, client):
            raise RuntimeError(message)

        return run

    monkeypatch.setattr(extract_service, "blocks_on_page", lambda _b, _p: None)
    monkeypatch.setattr(extract_service, "claims_from_prose", _boom("numeric boom"))
    monkeypatch.setattr(extract_service, "assertions_from_prose", _boom("qualitative boom"))

    tiers = extract_service._prose_tiers(
        ["prose", "qualitative"],
        [_page_index(5)],
        blocks=None,
        entity="ACME",
        file="cim.pdf",
        flag_log=extract_service.FlagLog(run_id="run-1"),
        workers=4,
        client=None,
    )

    assert tiers["prose"][0] == [] and tiers["qualitative"][0] == []
    assert [(s.page, s.tier) for s in tiers["prose"][1]] == [(5, "prose")]
    assert [(s.page, s.tier) for s in tiers["qualitative"][1]] == [(5, "qualitative")]
    assert "numeric boom" in tiers["prose"][1][0].reason
    assert "qualitative boom" in tiers["qualitative"][1][0].reason


def test_prose_tiers_returns_only_the_requested_kinds(monkeypatch) -> None:
    # Qualitative off -> the pool runs the numeric pass alone (kinds == ["prose"]),
    # and the qualitative extractor is never called.
    def _must_not_run(*_a, **_k):
        raise AssertionError("the qualitative extractor must not run when it is not requested")

    monkeypatch.setattr(extract_service, "blocks_on_page", lambda _b, _p: None)
    monkeypatch.setattr(extract_service, "claims_from_prose", _prose_extractor("prose"))
    monkeypatch.setattr(extract_service, "assertions_from_prose", _must_not_run)

    tiers = extract_service._prose_tiers(
        ["prose"],
        [_page_index(1)],
        blocks=None,
        entity="ACME",
        file="cim.pdf",
        flag_log=extract_service.FlagLog(run_id="run-1"),
        workers=2,
        client=None,
    )

    assert set(tiers) == {"prose"}
    assert len(tiers["prose"][0]) == 1


def test_completeness_sees_prose_but_never_qualitative(monkeypatch) -> None:
    # Completeness measures coverage over the numeric claims and must see exactly
    # what it saw when it ran BETWEEN the two tiers: table + prose, never qualitative.
    # The merge fans prose+qualitative out together but appends qualitative to
    # `claims` only AFTER completeness runs, so this ordering invariant is preserved.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    class _OnePageResult:
        document = _Doc()
        pages = [_Page(1)]
        sha256 = "0" * 64

    monkeypatch.setattr(extract_service, "parse_pdf_bytes", lambda _b: _OnePageResult())
    monkeypatch.setattr(extract_service, "extract_tables", lambda *_a, **_k: [])
    monkeypatch.setattr(extract_service, "tables_on_page", lambda *_a, **_k: [])
    monkeypatch.setattr(extract_service, "make_client", lambda: None)
    monkeypatch.setattr(extract_service, "extract_text_blocks", lambda *_a, **_k: [])
    monkeypatch.setattr(extract_service, "_prose_pages", lambda pages, _blocks: list(pages))
    monkeypatch.setattr(extract_service, "blocks_on_page", lambda _b, _p: None)
    monkeypatch.setattr(
        extract_service,
        "claims_from_prose",
        lambda _b, page, **_k: [_table_claim(page.page, attribute="from-prose")],
    )
    monkeypatch.setattr(
        extract_service,
        "assertions_from_prose",
        lambda _b, page, **_k: [
            _table_claim(page.page, attribute="from-qualitative", value_type="text")
        ],
    )

    seen: dict[str, list[str]] = {}

    def _spy_completeness(pages, prior_claims, *, entity, file, flag_log, workers, client):
        seen["prior"] = [c.attribute for c in prior_claims]
        return [], []

    monkeypatch.setattr(extract_service, "_completeness_claims", _spy_completeness)

    extract_service.extract_claims(
        b"%PDF-1.4 stub",
        entity="ACME",
        run_id="run-1",
        correlation_id="doc-1",
        source_file="cim.pdf",
        prose=True,
        complete=True,
        qualitative=True,
    )

    assert seen["prior"] == ["from-prose"], (
        "completeness must see table+prose only -- qualitative is appended to claims after it"
    )

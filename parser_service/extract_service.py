"""Claim extraction as a callable stage, shared by scripts/emit_claims.py (the
CLI) and POST /extract (main.py) so the two entry points can never drift --
the same reasoning as dispatch.parse_bytes for POST /parse and worker.py.

Stateless transform: document bytes in, the C3-contract claims payload out.
No DB access and no persistence here. `run_id` and `correlation_id` identify
the caller's run for logging/correlation only -- the claims contract has no
slot for either (the backend assigns `id`/`document_id` at persistence time,
per emit.Claim's docstring), so neither is written into the returned payload.
`correlation_id` is deliberately not named `document_id`: that term already
means the content hash elsewhere in this codebase (emit_chunks sets
`document_id = sha256`), and a caller-supplied token is not that.
"""

from __future__ import annotations

import logging
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from .coverage import NumberMiss, document_coverage
from .docling_parser import parse_pdf_bytes
from .emit import (
    OPERATING_METRIC,
    Claim,
    Edge,
    FlagLog,
    PdfLocation,
    SkippedPage,
    claim_ref_base,
    element_id_for,
)
from .extract import claims_from_table
from .propose import (
    api_key_present,
    assertions_from_prose,
    canonicalize_attributes,
    claims_from_completeness,
    claims_from_prose,
    prose_text,
)
from .schemas import PageIndex
from .table_extract import extract_tables, tables_on_page
from .text_extract import blocks_on_page, extract_text_blocks
from .verify import AuditVerdict, audit_claim

logger = logging.getLogger(__name__)


class ProseCredentialMissing(RuntimeError):
    """The prose/qualitative tiers were requested without an Anthropic
    credential in the environment. Raised before any parsing happens, so a
    caller never pays for a parse it can't finish -- the CLI maps this to an
    argparse SystemExit, the HTTP endpoint maps it to a 503."""


def _prose_claims(
    kind: str,
    pages: list[PageIndex],
    blocks,
    *,
    entity: str,
    file: str,
    flag_log: FlagLog,
    workers: int,
) -> tuple[list[Claim], list[SkippedPage]]:
    """Run one prose tier over every page that has prose, concurrently.

    `kind` selects the extractor: "prose" for the numeric pass, "qualitative"
    for the assertion pass. A page whose model call fails is reported on
    stderr AND returned as a SkippedPage(page, tier, reason) -- a partial
    result that names its gaps is more useful than none, and both entry points
    need that name: the CLI caller can re-run it, and an HTTP caller has no
    stderr to read at all (see extract_claims' `skipped_pages`).
    """
    extractor = claims_from_prose if kind == "prose" else assertions_from_prose

    def run(page: PageIndex) -> tuple[int, list[Claim]]:
        return page.page, extractor(
            blocks_on_page(blocks, page.page),
            page,
            entity_hint=entity,
            file=file,
            flag_log=flag_log,
        )

    with_prose = [p for p in pages if prose_text(blocks_on_page(blocks, p.page), p).strip()]
    claims: list[Claim] = []
    failed: list[tuple[int, str]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run, p): p.page for p in with_prose}
        for future in as_completed(futures):
            try:
                _, page_claims = future.result()
                claims += page_claims
            except Exception as exc:  # noqa: BLE001 -- one bad page must not lose the run
                failed.append((futures[future], f"{type(exc).__name__}: {exc}"))
    resolved = sum(1 for c in claims if c.status != "missing")
    print(
        f"tier {kind}: {len(claims)} claims over {len(with_prose)} prose pages "
        f"({resolved} resolved)",
        file=sys.stderr,
    )
    if failed:
        print(
            f"  {len(failed)} page(s) failed and were skipped: {[p for p, _ in failed]}",
            file=sys.stderr,
        )
    return claims, [
        SkippedPage(page=page_no, tier=kind, reason=reason) for page_no, reason in failed
    ]


def _pdf_span(claim: Claim) -> tuple[int, int, int] | None:
    """This claim's (page, char_start, char_end), or None for a `missing` claim
    or a non-PDF location. Isolated in its own function so the char_start/
    char_end None-checks narrow cleanly for the type checker, which a single
    list-comprehension condition does not do across two different attributes.
    """
    location = claim.location
    if (
        isinstance(location, PdfLocation)
        and location.char_start is not None
        and location.char_end is not None
    ):
        return (location.page, location.char_start, location.char_end)
    return None


def _completeness_claims(
    pages: list[PageIndex],
    prior_claims: list[Claim],
    *,
    entity: str,
    file: str,
    flag_log: FlagLog,
    workers: int,
) -> tuple[list[Claim], list[SkippedPage]]:
    """Recover Gate-1 coverage misses: numbers a page printed that no claim
    gathered so far covers, but that resolve to a unique span (parser_service/
    coverage.py). Coverage is measured over every claim gathered so far, not
    just the prose tier, since a number any tier already cited is not a miss.

    Only a page with at least one addressable miss costs a model call -- a
    fully-covered page is skipped outright (see claims_from_completeness).
    Numeric only, by design; see coverage.py's module docstring.

    A prior `missing` claim (e.g. a table cell with no resolvable bbox) has no
    span, so it does not suppress its own number here -- that number gets a
    second recovery attempt from this pass. Benign by construction: recovery
    can only add a resolved claim, never remove one, so the worst case is a
    second `missing` claim for the same figure, not a false cover. Not
    deduped against the earlier miss -- `_reduce_same_fact` below matches on
    a magnitude a `missing` claim never carries (see `_has_magnitude`).

    A page whose recovery call fails is reported on stderr AND returned as a
    SkippedPage(page, "complete", reason), the same failure model the prose
    tiers use, so the payload's `skipped_pages` names it for an HTTP caller.
    """
    claim_spans = [span for c in prior_claims if (span := _pdf_span(c)) is not None]
    coverage = document_coverage(pages, claim_spans)
    misses_by_page: dict[int, list[NumberMiss]] = defaultdict(list)
    for miss in coverage.gate1_misses:
        misses_by_page[miss.page].append(miss)
    pages_by_no = {p.page: p for p in pages}

    def run(page_no: int) -> tuple[int, list[Claim]]:
        page = pages_by_no[page_no]
        missed = [(m.text, m.context) for m in misses_by_page[page_no]]
        return page_no, claims_from_completeness(
            page, missed, entity_hint=entity, file=file, flag_log=flag_log
        )

    claims: list[Claim] = []
    failed: list[tuple[int, str]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run, page_no): page_no for page_no in misses_by_page}
        for future in as_completed(futures):
            try:
                _, page_claims = future.result()
                claims += page_claims
            except Exception as exc:  # noqa: BLE001 -- one bad page must not lose the run
                failed.append((futures[future], f"{type(exc).__name__}: {exc}"))
    recovered = sum(1 for c in claims if c.status != "missing")
    print(
        f"tier complete: coverage {coverage.coverage:.1%} "
        f"({coverage.covered}/{coverage.total}), {coverage.gate1} gate-1 miss(es) "
        f"over {len(misses_by_page)} page(s), {recovered} recovered",
        file=sys.stderr,
    )
    if failed:
        print(
            f"  {len(failed)} page(s) failed and were skipped: {[p for p, _ in failed]}",
            file=sys.stderr,
        )
    return claims, [
        SkippedPage(page=page_no, tier="complete", reason=reason) for page_no, reason in failed
    ]


def _has_magnitude(claim: Claim) -> bool:
    """Whether `claim` carries a number the fan-in reducer can match on --
    excludes `missing` (no value was found) and qualitative/text claims
    (normalized is null by construction, never a magnitude to compare)."""
    return claim.status != "missing" and claim.value.normalized is not None


def _page_of(claim: Claim) -> int:
    """The page a numeric claim was found on. This service is PDF-only, so
    every claim reaching the reducer carries a PdfLocation -- asserted here,
    once, so every other reducer line can read `.page` without re-narrowing
    the union each time a Claim comes back out of a dict/list.
    """
    assert isinstance(claim.location, PdfLocation)
    return claim.location.page


_STAGE_ATTRIBUTE_MAPPING = "attribute_mapping"

# SIM-344 scope: quantitative claims only. "qualitative" is deliberately
# excluded -- assertions_from_prose's assertion_class already categorizes
# those, and their attributes are open noun phrases ("on-site dry cleaning
# availability") that do not fit a financial-core-or-operating_metric enum.
_CANONICALIZABLE_TIERS = frozenset({"table", "prose", "complete"})


def _canonicalize_quantitative_claims(
    tier_claims: list[tuple[str, list[Claim]]], flag_log: FlagLog
) -> None:
    """SIM-344: map each quantitative claim's document-label attribute onto the
    closed canonical vocabulary, in place.

    Must run before _reduce_same_fact: edges are addressed by
    emit.element_id_for(claim), which reads claim.attribute, so an edge built
    from the pre-canonicalization attribute would point at an element_id no
    claim carries by the time the payload is returned.

    One batched call across every distinct raw label in the in-scope tiers --
    see propose.canonicalize_attributes -- never one call per claim.
    """
    raw_labels = [
        claim.attribute
        for tier, claims in tier_claims
        if tier in _CANONICALIZABLE_TIERS
        for claim in claims
    ]
    if not raw_labels:
        return
    mapping = canonicalize_attributes(raw_labels)
    for tier, claims in tier_claims:
        if tier not in _CANONICALIZABLE_TIERS:
            continue
        for claim in claims:
            raw = claim.attribute
            canonical, extra_flags = mapping[raw]
            claim.attribute_raw = raw
            claim.attribute = canonical
            if extra_flags:
                claim.flags = [*claim.flags, *extra_flags]
                flag_log.log_all(
                    _STAGE_ATTRIBUTE_MAPPING, element_id_for(claim), extra_flags, detail=raw
                )


def _assign_claim_refs(claims: list[Claim]) -> None:
    """SIM-365: give every claim a stable, positional `claim_ref`.

    Numbers claims within their (page, span) / (sheet, cell) bucket in emit order,
    so two runs over one document produce identical refs and claims sharing a span
    are disambiguated by the `[#n]` ordinal. Must run BEFORE the reducer (edges
    reference `claim_ref`) and before emit (the payload carries it). Positional and
    attribute-independent, so E2 canonicalization never shifts a ref.
    """
    counts: dict[str, int] = {}
    for claim in claims:
        base = claim_ref_base(claim)
        n = counts.get(base, 0)
        claim.claim_ref = f"{base}[#{n}]"
        counts[base] = n + 1


def _ref(claim: Claim) -> str:
    """The claim's assigned `claim_ref`, for edge endpoints. Asserts it was set --
    _assign_claim_refs runs before the reducer, so a None here is a wiring bug,
    not a data case."""
    assert claim.claim_ref is not None, "claim_ref must be assigned before reduction"
    return claim.claim_ref


def _reduce_same_fact(tiers: list[tuple[str, list[Claim]]]) -> list[Edge]:
    """SIM-341: the per-page extraction fan-in. Reconciles a fact two tiers
    both saw on the same page, without dropping either claim -- the parser
    only emits the edge, per the ticket's landed decision; a store-level
    merge is the backend's job, not this one.

    Two independent passes, both scoped to claims that carry a magnitude
    (see _has_magnitude) and to pairs from DIFFERENT tiers -- a tier citing
    the same figure twice on one page is not the cross-tier duplication this
    reducer exists to catch.

    same_fact: group by (page, entity, value_type, normalized value),
    attribute-blind -- two tiers may name the identical number differently
    (attribute-vocabulary mapping is E2, not this ticket), so the magnitude is
    the fingerprint. value_type is part of that fingerprint: a bare 40.0 is not
    one fact but potentially three -- a 40% margin, 40 employees, a $40 price --
    so the type keeps them from fusing into a false corroboration. unit is
    deliberately NOT in the key: a currency unit is header-derived and usually
    absent on a bare "$" (scale.py), so keying on it would split the very
    table-header-vs-bare-prose duplicate this exists to catch. Period lives
    inside the attribute string today; separating two periods that coincide in
    value waits on E2/E3's canonical attribute + period.
    A cross-tier collision keeps every claim, but names a winner (table wins
    on numbers; otherwise the first tier in pipeline order) and links every
    other claim in the group to it with a `same_fact` edge.

    contradicts: group by (page, entity, attribute, value_type, period_year,
    period_kind) instead -- the same labeled fact-slot, stated with different
    numbers by different tiers ("$15M" table vs "$12M excluding settlement"
    prose). value_type is in the key for the same reason it is in same_fact's:
    without it, a percent and a currency that both canonicalized to the same
    attribute look like one fact-slot disagreeing with itself. period is in the
    key because canonicalization strips the year off the attribute (SIM-344:
    "Revenue | 2019F" and "Revenue | 2020F" both -> "revenue"), so without it a
    2019 figure and a 2020 figure of the same metric would fuse into a false
    `contradicts` edge -- E3 (SIM-345) is what makes the period the structured
    field this keys on. OPERATING_METRIC claims are excluded from this pass
    entirely -- it is SIM-344's catch-all bucket for every sector/operating
    metric the core enum does not cover, so two claims landing there share
    nothing but the bucket, not a fact-slot (occupancy vs ARPU would otherwise
    fuse into a false `contradicts` edge). That disagreement is signal, not
    noise, so it is flagged and neither claim is preferred.
    """
    # SIM-365: stamp every claim (all tiers, not only numeric) with its stable
    # claim_ref first. The edges below name their endpoints by claim_ref, and the
    # emitted payload reads these same objects -- so this is also what puts a
    # claim_ref on every emitted claim. Positional and attribute-independent, so
    # canonicalize-vs-reduce order never shifts a ref.
    _assign_claim_refs([claim for _tier, tier_claims in tiers for claim in tier_claims])

    tier_of: dict[int, str] = {}
    numeric_claims: list[Claim] = []
    for tier, tier_claims in tiers:
        for claim in tier_claims:
            if not _has_magnitude(claim):
                continue
            tier_of[id(claim)] = tier
            numeric_claims.append(claim)

    edges: list[Edge] = []

    same_fact_groups: dict[tuple[int, str, str, float], list[Claim]] = defaultdict(list)
    for claim in numeric_claims:
        assert claim.value.normalized is not None  # narrowed by _has_magnitude above
        same_fact_groups[
            (_page_of(claim), claim.entity, claim.value.value_type, claim.value.normalized)
        ].append(claim)
    for group in same_fact_groups.values():
        if len({tier_of[id(c)] for c in group}) < 2:
            continue  # same tier, not a cross-tier duplicate
        winner = next((c for c in group if tier_of[id(c)] == "table"), group[0])
        for claim in group:
            if claim is winner:
                continue
            edges.append(
                Edge(
                    type="same_fact",
                    from_=_ref(claim),
                    to=_ref(winner),
                    basis=(
                        f"page {_page_of(winner)}: {tier_of[id(claim)]} and "
                        f"{tier_of[id(winner)]} tiers agree on entity, value type "
                        "+ normalized value"
                    ),
                )
            )

    attribute_groups: dict[tuple[int, str, str, str, int | None, str | None], list[Claim]] = (
        defaultdict(list)
    )
    for claim in numeric_claims:
        if claim.attribute == OPERATING_METRIC:
            continue
        attribute_groups[
            (
                _page_of(claim),
                claim.entity,
                claim.attribute,
                claim.value.value_type,
                claim.period_year,
                claim.period_kind,
            )
        ].append(claim)
    seen_pairs: set[frozenset[int]] = set()
    for group in attribute_groups.values():
        for i, a in enumerate(group):
            for b in group[i + 1 :]:
                if tier_of[id(a)] == tier_of[id(b)] or a.value.normalized == b.value.normalized:
                    continue
                pair = frozenset((id(a), id(b)))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                edges.append(
                    Edge(
                        type="contradicts",
                        from_=_ref(a),
                        to=_ref(b),
                        basis=(
                            f"page {_page_of(a)}: {tier_of[id(a)]} ({a.value.raw!r}) and "
                            f"{tier_of[id(b)]} ({b.value.raw!r}) disagree on the same attribute"
                        ),
                    )
                )

    return edges


_STAGE_BINDING_AUDIT = "binding_audit"
# Source text shown around the resolved span so the auditor reads the binding in
# context (a list heading, a "greater than", a services rate card) -- the span
# alone is often too tight to see the failure mode.
_AUDIT_CONTEXT = 400


def _should_audit(claim: Claim) -> bool:
    """SIM-359 provisional routing: audit only the scale-risk bindings -- a value
    a PAGE BANNER scaled, which is the exact `scale_absurd` mechanism (a
    "(in thousands)" header bleeding onto a value it does not govern, SIM-323/377).
    A value with no magnitude (qualitative/text) or one carrying its own or column
    scale is left to the GS-7-tuned routing; keeping the per-claim model audit
    narrow is the point of routing.
    """
    value = claim.value
    return (
        claim.status != "missing"
        and value.normalized is not None
        and value.scale_source == "page_header"
    )


def _audit_claims(
    claims: list[Claim], pages: list[PageIndex], flag_log: FlagLog, *, workers: int
) -> None:
    """SIM-359: flag claims whose cited span does not justify their
    (entity, attribute, value) binding, via verify.audit_claim -- an independent
    reader that only flags, never mutates a value/span/status. Routed claims run
    concurrently, like the prose tiers; every verdict mode collapses onto the
    existing `binding_unsupported` flag (mode + evidence in the FlagLog detail),
    with the per-mode flag taxonomy deferred to GS-7.
    """
    page_text = {p.page: p.text for p in pages}
    routed = [(c, span) for c in claims if _should_audit(c) and (span := _pdf_span(c)) is not None]
    if not routed:
        return

    def run(item: tuple[Claim, tuple[int, int, int]]) -> tuple[Claim, AuditVerdict]:
        claim, (page_no, start, end) = item
        text = page_text.get(page_no, "")
        return claim, audit_claim(
            entity=claim.entity,
            attribute=claim.attribute_raw or claim.attribute,
            value_raw=claim.value.raw,
            value_normalized=claim.value.normalized,
            unit=claim.value.unit,
            quote=text[start:end],
            context=text[max(0, start - _AUDIT_CONTEXT) : end + _AUDIT_CONTEXT],
            page=page_no,
        )

    flagged = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run, item) for item in routed]
        for future in as_completed(futures):
            try:
                claim, verdict = future.result()
            except Exception as exc:  # noqa: BLE001 -- one bad audit must not lose the run
                print(
                    f"tier binding_audit: a claim's audit failed and was skipped: "
                    f"{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
                continue
            if verdict.verdict == "supported":
                continue
            if "binding_unsupported" not in claim.flags:
                claim.flags = [*claim.flags, "binding_unsupported"]
            flag_log.log(
                _STAGE_BINDING_AUDIT,
                element_id_for(claim),
                "binding_unsupported",
                detail=f"{verdict.verdict}: {verdict.evidence}",
            )
            flagged += 1
    print(
        f"tier binding_audit: {len(routed)} claim(s) audited, "
        f"{flagged} flagged binding_unsupported",
        file=sys.stderr,
    )


def extract_claims(
    data: bytes,
    *,
    entity: str,
    run_id: str,
    correlation_id: str,
    source_file: str | None = None,
    prose: bool = False,
    complete: bool = False,
    qualitative: bool = False,
    canonicalize_attributes: bool = False,
    audit: bool = False,
    workers: int = 8,
) -> dict:
    """Parse `data` and emit its claims as the payload that crosses the C3 seam.

    The one code path both entry points call: scripts/emit_claims.py (CLI) and
    POST /extract (main.py). Raises ProseCredentialMissing at the door, before
    any parsing, if `prose`/`complete`/`qualitative`/`canonicalize_attributes`
    are set without a credential in the environment -- fail closed, not
    partway through a document. `complete` and `qualitative` each imply the
    prose tier. `canonicalize_attributes` (SIM-344) does not -- it maps every
    quantitative claim's attribute onto the closed canonical vocabulary
    regardless of which tiers ran, table-only included, since a table-only run
    has attributes just as much as a prose run does.

    `source_file` is the real filename when the caller has one (the CLI
    always does); an HTTP caller that genuinely has none should leave it
    unset rather than pass a stand-in like `correlation_id` -- the C3 contract
    documents each claim's `file` as "debug/human-readable only", and a
    correlation token dressed as a filename is a false debug signal. Every
    claim still needs a string for `file`, so an unset source_file becomes ""
    there (never a guess); the top-level `source_file` in the returned
    payload stays `None` so a caller can tell "no filename given" apart from
    "the filename was an empty string".
    """
    want_prose = prose or complete or qualitative
    if (want_prose or canonicalize_attributes or audit) and not api_key_present():
        raise ProseCredentialMissing(
            "the prose tiers, attribute canonicalization, and the binding audit call "
            "the Anthropic API and need ANTHROPIC_API_KEY (or ANTHROPIC_AUTH_TOKEN) in "
            "the environment; set it, or drop prose/complete/qualitative/"
            "canonicalize_attributes/audit to emit table claims only."
        )

    logger.info(
        "extract_claims start: run_id=%s correlation_id=%s entity=%s prose=%s "
        "complete=%s qualitative=%s canonicalize_attributes=%s audit=%s",
        run_id,
        correlation_id,
        entity,
        prose,
        complete,
        qualitative,
        canonicalize_attributes,
        audit,
    )

    result = parse_pdf_bytes(data)
    assert result.document is not None, "a successful parse always carries the DoclingDocument"
    tables = extract_tables(result.document, result.pages)
    flag_log = FlagLog(run_id=run_id)
    file = source_file or ""

    claims: list[Claim] = []
    tier_claims: list[tuple[str, list[Claim]]] = []
    skipped_pages: list[SkippedPage] = []
    table_claims: list[Claim] = []
    for page in result.pages:
        for table in tables_on_page(tables, page.page):
            try:
                table_claims += claims_from_table(
                    table,
                    page,
                    entity=entity,
                    file=file,
                    flag_log=flag_log,
                )
            except Exception as exc:  # noqa: BLE001 -- one bad table must not abort the document
                # Guarded per table, not per page: a page with several tables
                # keeps every sibling table's claims when only one is
                # malformed. The page is still named in skipped_pages -- it
                # lost something real, even if not everything.
                skipped_pages.append(
                    SkippedPage(
                        page=page.page,
                        tier="tables",
                        reason=f"{type(exc).__name__}: {exc}",
                    )
                )
                print(
                    f"tier tables: page {page.page} table failed and was skipped: "
                    f"{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
    claims += table_claims
    tier_claims.append(("table", table_claims))
    print(f"tier tables: {len(claims)} claims", file=sys.stderr)

    if want_prose:
        blocks = extract_text_blocks(result.document, result.pages)
        prose_claims, prose_failed = _prose_claims(
            "prose",
            result.pages,
            blocks,
            entity=entity,
            file=file,
            flag_log=flag_log,
            workers=workers,
        )
        claims += prose_claims
        skipped_pages.extend(prose_failed)
        tier_claims.append(("prose", prose_claims))
        if complete:
            complete_claims, complete_failed = _completeness_claims(
                result.pages,
                claims,
                entity=entity,
                file=file,
                flag_log=flag_log,
                workers=workers,
            )
            claims += complete_claims
            skipped_pages.extend(complete_failed)
            tier_claims.append(("complete", complete_claims))
        if qualitative:
            qualitative_claims, qualitative_failed = _prose_claims(
                "qualitative",
                result.pages,
                blocks,
                entity=entity,
                file=file,
                flag_log=flag_log,
                workers=workers,
            )
            claims += qualitative_claims
            skipped_pages.extend(qualitative_failed)
            tier_claims.append(("qualitative", qualitative_claims))

    if canonicalize_attributes:
        try:
            _canonicalize_quantitative_claims(tier_claims, flag_log)
        except Exception as exc:  # noqa: BLE001 -- one bad API call must not abort the document
            # Document-wide, not page-scoped (one batched call over every
            # distinct label) -- page=0 is the same sentinel propose.py's
            # _parse_with_retry already uses for this call's retries. Claims
            # already emitted survive with their raw (pre-canonicalization)
            # attribute rather than being discarded.
            skipped_pages.append(
                SkippedPage(
                    page=0,
                    tier="attribute_mapping",
                    reason=f"{type(exc).__name__}: {exc}",
                )
            )
            print(
                f"tier attribute_mapping: failed and was skipped, claims keep their raw "
                f"attribute labels: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )

    if audit:
        _audit_claims(claims, result.pages, flag_log, workers=workers)

    edges = _reduce_same_fact(tier_claims)
    same_fact_count = sum(1 for e in edges if e.type == "same_fact")
    contradicts_count = sum(1 for e in edges if e.type == "contradicts")
    print(
        f"reducer: {same_fact_count} same_fact edge(s), {contradicts_count} contradicts edge(s)",
        file=sys.stderr,
    )

    payload = {
        "run_id": flag_log.run_id,
        "sha256": result.sha256,
        "source_file": source_file,
        "claims": [c.to_json() for c in claims],
        "edges": [e.to_json() for e in edges],
        "flag_log": flag_log.to_json(),
        "skipped_pages": [
            s.to_json() for s in sorted(skipped_pages, key=lambda s: (s.page, s.tier))
        ],
    }
    print(
        f"\n\nemitted {len(claims)} claims "
        f"({sum(1 for c in claims if c.status != 'missing')} cited, "
        f"{sum(1 for c in claims if c.status == 'missing')} missing), "
        f"{len(edges)} edges, {len(flag_log.entries)} flags, {len(skipped_pages)} skip(s)",
        file=sys.stderr,
    )
    return payload

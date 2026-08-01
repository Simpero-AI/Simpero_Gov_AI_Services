"""The emit_claims CLI: the committed entry point for the parse-to-claims seam.

scripts/emit_claims.py is now a thin argparse wrapper over
parser_service.extract_service.extract_claims (SIM-340/E5) -- that shared
function is where parsing actually happens, so these tests patch it there,
not on the CLI module. The behaviour under test is still the CLI's own: the
prose tiers must refuse to start without a credential rather than failing
part way through a document, and the table tier needs neither key nor network.
"""

from __future__ import annotations

import json

import pytest

from parser_service import extract_service
from parser_service.emit import Claim, ClaimValue, PdfLocation, element_id_for
from parser_service.schemas import CharBox, PageIndex
from scripts import emit_claims


def _page(text: str, page_no: int = 1) -> PageIndex:
    char_map: list[CharBox] = []
    x = 0.0
    for character in text:
        char_map.append(
            CharBox(
                char=character,
                x0=x,
                top=100.0,
                x1=x + 5.0,
                bottom=110.0,
                page=page_no,
                precision="char",
            )
        )
        x += 5.0
    return PageIndex(page=page_no, text=text, char_map=char_map)


def _proposed_claim(entity: str, attribute: str, text: str, page: PageIndex) -> Claim:
    start = page.text.index(text)
    end = start + len(text)
    return Claim(
        entity=entity,
        attribute=attribute,
        value=ClaimValue(raw=text, normalized=None, unit=None, value_type="currency"),
        location=PdfLocation(file="cim.pdf", page=page.page, char_start=start, char_end=end),
        status="proposed",
    )


def test_prose_without_a_key_fails_closed_before_any_work(monkeypatch, tmp_path) -> None:
    # A missing credential must stop the run at the door, not after a parse and
    # not part way through a document. It surfaces as a SystemExit (the CLI's
    # argparse.error translation of ProseCredentialMissing), and it must fire
    # before parse_pdf_bytes is ever called.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)

    called = False

    def _must_not_run(*_a, **_k):
        nonlocal called
        called = True
        raise AssertionError("parse must not start when the key is absent")

    monkeypatch.setattr(extract_service, "parse_pdf_bytes", _must_not_run)
    pdf = tmp_path / "cim.pdf"
    pdf.write_bytes(b"%PDF-1.4 stub")

    with pytest.raises(SystemExit):
        emit_claims.main([str(pdf), "--entity", "ACME", "--prose"])
    assert called is False


def test_qualitative_implies_prose_and_also_needs_a_key(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.setattr(
        extract_service, "parse_pdf_bytes", lambda *_a, **_k: AssertionError("unreachable")
    )
    pdf = tmp_path / "cim.pdf"
    pdf.write_bytes(b"%PDF-1.4 stub")

    with pytest.raises(SystemExit):
        emit_claims.main([str(pdf), "--entity", "ACME", "--qualitative"])


def test_complete_implies_prose_and_also_needs_a_key(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.setattr(
        extract_service, "parse_pdf_bytes", lambda *_a, **_k: AssertionError("unreachable")
    )
    pdf = tmp_path / "cim.pdf"
    pdf.write_bytes(b"%PDF-1.4 stub")

    with pytest.raises(SystemExit):
        emit_claims.main([str(pdf), "--entity", "ACME", "--complete"])


def test_canonicalize_attributes_also_needs_a_key(monkeypatch, tmp_path) -> None:
    # SIM-344's pass is independent of --prose but still calls the Anthropic
    # API, so it must fail closed at the door the same way.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.setattr(
        extract_service, "parse_pdf_bytes", lambda *_a, **_k: AssertionError("unreachable")
    )
    pdf = tmp_path / "cim.pdf"
    pdf.write_bytes(b"%PDF-1.4 stub")

    with pytest.raises(SystemExit):
        emit_claims.main([str(pdf), "--entity", "ACME", "--canonicalize-attributes"])


def test_canonicalize_attributes_flag_maps_a_table_claim(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    class _Doc:
        pass

    class _Page:
        def __init__(self, page: int) -> None:
            self.page = page

    class _Result:
        document = _Doc()
        pages = [_Page(1)]
        sha256 = "0" * 64

    monkeypatch.setattr(extract_service, "parse_pdf_bytes", lambda *_a, **_k: _Result())
    monkeypatch.setattr(extract_service, "extract_tables", lambda *_a, **_k: ["a-table"])
    monkeypatch.setattr(extract_service, "tables_on_page", lambda tables, _page_no: tables)
    monkeypatch.setattr(
        extract_service,
        "claims_from_table",
        lambda table, page, *, entity, file, flag_log: [
            _proposed_claim(entity, "Revenue | 2019F", "$1", _page("$1 total", page_no=page.page))
        ],
    )
    monkeypatch.setattr(
        extract_service,
        "canonicalize_attributes",
        lambda labels: {"Revenue | 2019F": ("revenue", [])},
    )

    pdf = tmp_path / "cim.pdf"
    pdf.write_bytes(b"%PDF-1.4 stub")

    emit_claims.main([str(pdf), "--entity", "ACME", "--canonicalize-attributes"])

    payload = json.loads(capsys.readouterr().out)
    assert len(payload["claims"]) == 1
    assert payload["claims"][0]["attribute"] == "revenue"
    assert payload["claims"][0]["attribute_raw"] == "Revenue | 2019F"


def test_tables_only_needs_no_key(monkeypatch, tmp_path) -> None:
    # The default tier calls no model and must run with the environment stripped.
    # A present key must not be consulted for a table-only run, so the guard is
    # never reached -- proven by leaving parse stubbed and asserting it ran.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)

    ran = False

    def _fake_key_check() -> bool:  # pragma: no cover - must not be called
        raise AssertionError("the table tier must not consult the credential")

    monkeypatch.setattr(extract_service, "api_key_present", _fake_key_check)

    class _Doc:
        pass

    class _Result:
        document = _Doc()
        pages: list = []
        sha256 = "0" * 64

    def _fake_parse(_bytes):
        nonlocal ran
        ran = True
        return _Result()

    monkeypatch.setattr(extract_service, "parse_pdf_bytes", _fake_parse)
    monkeypatch.setattr(extract_service, "extract_tables", lambda *_a, **_k: [])
    pdf = tmp_path / "cim.pdf"
    pdf.write_bytes(b"%PDF-1.4 stub")

    emit_claims.main([str(pdf), "--entity", "ACME"])
    assert ran is True


def test_cli_stdout_is_pure_json_with_no_stderr_chatter_mixed_in(
    monkeypatch, tmp_path, capsys
) -> None:
    # json.dump goes to stdout; every diagnostic line is on stderr (see
    # extract_service.extract_claims) -- a downstream consumer piping stdout
    # into a JSON parser must never see a stray print land in the payload.
    class _Doc:
        pass

    class _Result:
        document = _Doc()
        pages: list = []
        sha256 = "0" * 64

    monkeypatch.setattr(extract_service, "parse_pdf_bytes", lambda _b: _Result())
    monkeypatch.setattr(extract_service, "extract_tables", lambda *_a, **_k: [])
    pdf = tmp_path / "cim.pdf"
    pdf.write_bytes(b"%PDF-1.4 stub")

    emit_claims.main([str(pdf), "--entity", "ACME", "--run-id", "run-xyz"])
    captured = capsys.readouterr()

    payload = json.loads(captured.out)
    assert payload["run_id"] == "run-xyz"
    assert payload["source_file"] == "cim.pdf"
    assert payload["claims"] == []
    assert "tier tables" in captured.err


def test_a_failed_prose_page_is_recorded_and_the_run_still_succeeds(
    monkeypatch, tmp_path, capsys
) -> None:
    # A page whose model call raises must not lose the run: the good page's
    # claim still comes out, and the bad page is named in the payload's
    # `skipped_pages` (the E6 record) as a SkippedPage(page, tier, reason), not
    # just printed to stderr and forgotten. The CLI runs it through
    # extract_service.extract_claims, so the tiers are patched there.
    monkeypatch.setattr(extract_service, "api_key_present", lambda: True)

    class _Doc:
        pass

    class _Page:
        def __init__(self, page: int) -> None:
            self.page = page

    class _Result:
        document = _Doc()
        pages = [_Page(1), _Page(2)]
        sha256 = "0" * 64

    class _Value:
        normalized = None  # no magnitude -> the fan-in reducer skips this claim

    class _FakeClaim:
        status = "proposed"
        value = _Value()

        def to_json(self) -> dict:
            return {"entity": "ACME", "status": "proposed"}

    def _fake_claims_from_prose(_blocks, page, **_kwargs):
        if page.page == 2:
            raise ValueError("model call failed")
        return [_FakeClaim()]

    monkeypatch.setattr(extract_service, "parse_pdf_bytes", lambda _bytes: _Result())
    monkeypatch.setattr(extract_service, "extract_tables", lambda *_a, **_k: [])
    monkeypatch.setattr(extract_service, "extract_text_blocks", lambda *_a, **_k: [])
    monkeypatch.setattr(extract_service, "blocks_on_page", lambda *_a, **_k: [])
    monkeypatch.setattr(extract_service, "prose_text", lambda *_a, **_k: "some prose")
    monkeypatch.setattr(extract_service, "claims_from_prose", _fake_claims_from_prose)

    pdf = tmp_path / "cim.pdf"
    pdf.write_bytes(b"%PDF-1.4 stub")

    emit_claims.main([str(pdf), "--entity", "ACME", "--prose", "--workers", "1"])

    payload = json.loads(capsys.readouterr().out)
    assert len(payload["claims"]) == 1
    assert payload["skipped_pages"] == [
        {"page": 2, "tier": "prose", "reason": "ValueError: model call failed"}
    ]


def test_complete_recovers_a_gate1_miss_the_prose_tier_left_uncovered(
    monkeypatch, tmp_path, capsys
) -> None:
    # The prose tier claims $100 and walks past $200 on the same page. Both are
    # financially-meaningful and $200 resolves to a unique span, so it is a
    # Gate-1 miss -- exactly what --complete exists to recover. The recovery
    # runs inside extract_service now, so the tiers are patched there.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    page = _page("Revenue was $100 in year one and $200 in year two.")

    class _Doc:
        pass

    class _Result:
        document = _Doc()
        pages = [page]
        sha256 = "0" * 64

    monkeypatch.setattr(extract_service, "parse_pdf_bytes", lambda *_a, **_k: _Result())
    monkeypatch.setattr(extract_service, "extract_tables", lambda *_a, **_k: [])
    monkeypatch.setattr(extract_service, "extract_text_blocks", lambda *_a, **_k: [])
    monkeypatch.setattr(extract_service, "blocks_on_page", lambda _blocks, _page_no: [])
    monkeypatch.setattr(extract_service, "prose_text", lambda _blocks, page_: page_.text)
    monkeypatch.setattr(
        extract_service,
        "claims_from_prose",
        lambda *_a, **_k: [_proposed_claim("ACME", "revenue | year one", "$100", page)],
    )

    completeness_calls: list[list[tuple[str, str]]] = []

    def _fake_completeness(page_, missed, *, entity_hint, file, flag_log):
        completeness_calls.append(missed)
        return [_proposed_claim(entity_hint, "revenue | year two", "$200", page_)]

    monkeypatch.setattr(extract_service, "claims_from_completeness", _fake_completeness)

    pdf = tmp_path / "cim.pdf"
    pdf.write_bytes(b"%PDF-1.4 stub")

    emit_claims.main([str(pdf), "--entity", "ACME", "--complete"])

    stdout = capsys.readouterr().out
    payload = json.loads(stdout)
    attributes = {c["attribute"] for c in payload["claims"]}
    assert "revenue | year one" in attributes
    assert "revenue | year two" in attributes

    # Only the real Gate-1 miss ($200) was handed to the recovery pass -- $100
    # was already covered by the prose tier and must not be re-proposed.
    assert len(completeness_calls) == 1
    missed_numbers = {number for number, _context in completeness_calls[0]}
    assert missed_numbers == {"$200"}


def test_edges_link_a_table_and_prose_duplicate_on_the_same_page(
    monkeypatch, tmp_path, capsys
) -> None:
    # SIM-341: the table tier and the prose tier both see the same figure on
    # the same page. Both claims must survive in `claims` -- the reducer only
    # emits a `same_fact` edge, it never drops or merges one -- and the table
    # claim must be the edge's `to` (table wins on numbers).
    monkeypatch.setattr(extract_service, "api_key_present", lambda: True)

    page = _page("Revenue was $15,295 in fiscal 2024, driven by strong demand.")

    table_claim = Claim(
        entity="ACME",
        attribute="Revenue | 2024F",
        value=ClaimValue(raw="$15,295", normalized=15_295_000, unit="USD", value_type="currency"),
        location=PdfLocation(file="cim.pdf", page=1, char_start=13, char_end=20),
        status="proposed",
    )
    prose_claim = Claim(
        entity="ACME",
        attribute="total revenue in fiscal 2024",
        value=ClaimValue(raw="$15,295", normalized=15_295_000, unit="USD", value_type="currency"),
        location=PdfLocation(file="cim.pdf", page=1, char_start=13, char_end=20),
        status="proposed",
    )

    class _Doc:
        pass

    class _Result:
        document = _Doc()
        pages = [page]
        sha256 = "0" * 64

    monkeypatch.setattr(extract_service, "parse_pdf_bytes", lambda *_a, **_k: _Result())
    monkeypatch.setattr(extract_service, "extract_tables", lambda *_a, **_k: ["a-table"])
    monkeypatch.setattr(extract_service, "tables_on_page", lambda tables, _page_no: tables)
    monkeypatch.setattr(extract_service, "claims_from_table", lambda *_a, **_k: [table_claim])
    monkeypatch.setattr(extract_service, "extract_text_blocks", lambda *_a, **_k: [])
    monkeypatch.setattr(extract_service, "blocks_on_page", lambda *_a, **_k: [])
    monkeypatch.setattr(extract_service, "prose_text", lambda *_a, **_k: page.text)
    monkeypatch.setattr(extract_service, "claims_from_prose", lambda *_a, **_k: [prose_claim])

    pdf = tmp_path / "cim.pdf"
    pdf.write_bytes(b"%PDF-1.4 stub")

    emit_claims.main([str(pdf), "--entity", "ACME", "--prose", "--workers", "1"])

    payload = json.loads(capsys.readouterr().out)
    assert len(payload["claims"]) == 2
    assert payload["edges"] == [
        {
            "type": "same_fact",
            "from": element_id_for(prose_claim),
            "to": element_id_for(table_claim),
            "basis": (
                "page 1: prose and table tiers agree on entity, value type + normalized value"
            ),
        }
    ]

"""Deal-profile classification (Path B).

Hermetic: a stub stands in for the Anthropic client, so these assert the
contract around the model, not the model. The classifier must pass through only
what the document states (null otherwise), feed a bounded overview window, judge
mandate fit only against options we actually supplied, never trust an off-list
fit option, and fail soft to an all-null profile rather than raise.
"""

from __future__ import annotations

from types import SimpleNamespace

from parser_service.deal_profile import (
    _OVERVIEW_CHARS,
    _OVERVIEW_PAGES,
    DealProfile,
    MandateFit,
    classify_deal_profile,
)


class _StubClient:
    """Returns a fixed DealProfile and records the parse kwargs."""

    def __init__(self, profile: DealProfile) -> None:
        self._profile = profile
        self.calls: list[dict] = []
        self.messages = SimpleNamespace(parse=self._parse)

    def _parse(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(parsed_output=self._profile)


class _GrammarTimeout(Exception):
    """A transient server-side grammar-compilation 400 (see is_grammar_timeout)."""

    status_code = 400

    def __init__(self) -> None:
        super().__init__("grammar compilation timed out")


class _FlakyGrammarClient:
    """Raises a transient grammar-400 on the first parse, then succeeds."""

    def __init__(self, profile: DealProfile) -> None:
        self._profile = profile
        self.calls = 0
        self.messages = SimpleNamespace(parse=self._parse)

    def _parse(self, **_kwargs):
        self.calls += 1
        if self.calls == 1:
            raise _GrammarTimeout()
        return SimpleNamespace(parsed_output=self._profile)


def test_a_transient_grammar_400_is_retried_not_dropped(monkeypatch) -> None:
    # Regression: classify_deal_profile must route through parse_with_retry so a
    # transient grammar-compilation 400 is retried rather than propagating (and
    # being silently swallowed by extract_claims, dropping the whole stage).
    monkeypatch.setattr("parser_service.llm_client._GRAMMAR_BACKOFF_S", 0.0)
    client = _FlakyGrammarClient(DealProfile(sector="fintech", hq_geography="London, UK"))
    out = classify_deal_profile(["Company overview."], entity="Acme", client=client)
    assert client.calls == 2  # first raised the grammar-400; the retry succeeded
    assert out.sector == "fintech"
    assert out.hq_geography == "London, UK"


def test_passes_through_stated_sector_and_hq() -> None:
    profile = DealProfile(
        sector="enterprise SaaS",
        sector_evidence="a vertical SaaS platform for dental clinics",
        hq_geography="Toronto, Ontario, Canada",
        hq_evidence="Headquartered in Toronto, Ontario",
    )
    client = _StubClient(profile)
    out = classify_deal_profile(["Company overview page."], entity="Acme", client=client)
    assert out.sector == "enterprise SaaS"
    assert out.hq_geography == "Toronto, Ontario, Canada"
    assert out.hq_evidence == "Headquartered in Toronto, Ontario"


def test_null_field_when_document_is_silent() -> None:
    client = _StubClient(DealProfile(sector="fintech lending", hq_geography=None))
    out = classify_deal_profile(["Some prose."], entity="Acme", client=client)
    assert out.sector == "fintech lending"
    assert out.hq_geography is None
    assert out.hq_evidence == ""


def test_no_options_means_no_fit_even_if_the_model_returns_one() -> None:
    # A model that volunteers a fit with no options supplied must be ignored.
    profile = DealProfile(
        sector="fintech",
        hq_geography="Canada",
        sector_fit=MandateFit(status="match", option="Fintech"),
        hq_fit=MandateFit(status="match", option="Canada"),
    )
    out = classify_deal_profile(["prose"], entity="Acme", client=_StubClient(profile))
    assert out.sector_fit is None
    assert out.hq_fit is None


def test_match_returns_the_supplied_option_verbatim() -> None:
    # Model echoes a differently-cased/spaced option; we return OUR verbatim string
    # so the backend writes an exact fold-match for approves_sector.
    profile = DealProfile(
        sector="B2B SaaS for clinics",
        hq_geography="Toronto",
        sector_fit=MandateFit(status="match", option="healthcare  IT"),
        hq_fit=MandateFit(status="match", option="canada"),
    )
    out = classify_deal_profile(
        ["prose"],
        entity="Acme",
        sector_options=["Fintech", "Healthcare IT", "Enterprise Software"],
        geo_options=["Canada", "United States"],
        client=_StubClient(profile),
    )
    assert out.sector_fit == MandateFit(status="match", option="Healthcare IT")
    assert out.hq_fit == MandateFit(status="match", option="Canada")


def test_off_list_match_option_is_downgraded_to_unknown() -> None:
    profile = DealProfile(
        sector="defense",
        hq_geography="France",
        sector_fit=MandateFit(status="match", option="Defense Manufacturing"),  # not offered
        hq_fit=MandateFit(status="match", option="France"),  # not offered
    )
    out = classify_deal_profile(
        ["prose"],
        entity="Acme",
        sector_options=["Fintech", "Healthcare IT"],
        geo_options=["Canada", "United States"],
        client=_StubClient(profile),
    )
    assert out.sector_fit == MandateFit(status="unknown", option=None)
    assert out.hq_fit == MandateFit(status="unknown", option=None)


def test_outside_and_unknown_pass_through_with_no_option() -> None:
    profile = DealProfile(
        sector="cannabis retail",
        hq_geography="somewhere",
        sector_fit=MandateFit(status="outside", option=None),
        hq_fit=MandateFit(status="unknown", option=None),
    )
    out = classify_deal_profile(
        ["prose"],
        entity="Acme",
        sector_options=["Fintech"],
        geo_options=["Canada"],
        client=_StubClient(profile),
    )
    assert out.sector_fit == MandateFit(status="outside", option=None)
    assert out.hq_fit == MandateFit(status="unknown", option=None)


def test_match_without_an_option_collapses_to_unknown() -> None:
    profile = DealProfile(
        sector="fintech",
        hq_geography="Canada",
        sector_fit=MandateFit(status="match", option=None),  # inconsistent
    )
    out = classify_deal_profile(
        ["prose"], entity="Acme", sector_options=["Fintech"], client=_StubClient(profile)
    )
    assert out.sector_fit == MandateFit(status="unknown", option=None)
    assert out.hq_fit is None  # no geo options supplied


def test_empty_overview_returns_all_null_without_calling_the_model() -> None:
    client = _StubClient(DealProfile(sector="x", hq_geography="y"))
    out = classify_deal_profile(["", "   ", "\n"], entity="Acme", client=client)
    assert out.sector is None and out.hq_geography is None
    assert client.calls == []  # no page had text -> no API spend


def test_only_the_first_pages_are_sent() -> None:
    # Short pages so the page cap, not the char cap, is what bounds the window.
    pages = [f"PAGE-{i}" for i in range(_OVERVIEW_PAGES + 4)]
    client = _StubClient(DealProfile(sector=None, hq_geography=None))
    classify_deal_profile(pages, entity="Acme", client=client)

    sent = client.calls[0]["messages"][0]["content"]
    assert f"PAGE-{_OVERVIEW_PAGES - 1}" in sent  # last in-window page reached
    assert f"PAGE-{_OVERVIEW_PAGES}" not in sent  # first out-of-window page dropped


def test_overview_is_char_capped() -> None:
    client = _StubClient(DealProfile(sector=None, hq_geography=None))
    classify_deal_profile(["x" * (_OVERVIEW_CHARS * 3)], entity="Acme", client=client)

    sent = client.calls[0]["messages"][0]["content"]
    # The overview slice is char-capped; the user message adds only a small preamble.
    assert sent.count("x") == _OVERVIEW_CHARS


def test_supplied_options_are_rendered_into_the_prompt() -> None:
    client = _StubClient(DealProfile(sector=None, hq_geography=None))
    classify_deal_profile(
        ["prose"],
        entity="Acme",
        sector_options=["Fintech", "Healthcare IT"],
        geo_options=["Canada"],
        client=client,
    )
    sent = client.calls[0]["messages"][0]["content"]
    assert "- Fintech" in sent and "- Healthcare IT" in sent
    assert "- Canada" in sent


def test_retries_once_on_an_unparseable_body_then_succeeds() -> None:
    good = DealProfile(sector="medical devices", hq_geography="Boston, MA, USA")

    class _FlakyClient:
        def __init__(self) -> None:
            self.n = 0
            self.messages = SimpleNamespace(parse=self._parse)

        def _parse(self, **kwargs):
            self.n += 1
            if self.n == 1:
                DealProfile.model_validate({})  # missing required fields -> ValidationError
            return SimpleNamespace(parsed_output=good)

    client = _FlakyClient()
    out = classify_deal_profile(["Overview."], entity="Acme", client=client)
    assert client.n == 2
    assert out.sector == "medical devices"


def test_none_parsed_output_falls_back_to_all_null() -> None:
    class _NullClient:
        def __init__(self) -> None:
            self.messages = SimpleNamespace(parse=self._parse)

        def _parse(self, **kwargs):
            return SimpleNamespace(parsed_output=None)

    out = classify_deal_profile(["Overview."], entity="Acme", client=_NullClient())
    assert out.sector is None and out.hq_geography is None

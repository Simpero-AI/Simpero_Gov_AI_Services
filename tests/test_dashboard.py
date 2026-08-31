"""Per-document dashboard organization (Pipeline Inspector).

Hermetic: a stub stands in for the Anthropic client. These assert the contract
around the model -- above all the grounding trust boundary: the model may only
arrange entities/metrics that were supplied, and every failure degrades to an
empty structure (the Inspector then falls back to frequency grouping).
"""

from __future__ import annotations

from types import SimpleNamespace

from parser_service.dashboard import DashboardStructure, Subject, organize_claims

_ENTITIES = {"Acme Corp": 200, "North Region": 40, "South Region": 30, "One-off Mention": 1}
_METRICS = ["revenue", "ebitda", "net_income", "total_assets"]


class _StubClient:
    def __init__(self, structure: DashboardStructure) -> None:
        self._structure = structure
        self.calls: list[dict] = []
        self.messages = SimpleNamespace(parse=self._parse)

    def _parse(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(parsed_output=self._structure)


def test_groups_entities_and_orders_metrics() -> None:
    client = _StubClient(
        DashboardStructure(
            subjects=[
                Subject(name="Consolidated", kind="consolidated", entities=["Acme Corp"]),
                Subject(name="North", kind="segment", entities=["North Region"]),
                Subject(name="South", kind="segment", entities=["South Region"]),
            ],
            metric_order=["revenue", "ebitda", "net_income", "total_assets"],
        )
    )
    out = organize_claims(_ENTITIES, _METRICS, company="Acme", client=client)
    assert [s.name for s in out.subjects] == ["Consolidated", "North", "South"]
    assert sum(1 for s in out.subjects if s.kind == "consolidated") == 1
    assert out.metric_order == ["revenue", "ebitda", "net_income", "total_assets"]


def test_drops_entities_not_supplied() -> None:
    client = _StubClient(
        DashboardStructure(
            subjects=[
                Subject(
                    name="Consolidated",
                    kind="consolidated",
                    entities=["Acme Corp", "Fabricated Inc"],
                ),
            ],
            metric_order=["revenue"],
        )
    )
    out = organize_claims(_ENTITIES, _METRICS, company="Acme", client=client)
    assert out.subjects[0].entities == ["Acme Corp"]  # invented entity dropped


def test_drops_invented_metrics_and_appends_omitted() -> None:
    client = _StubClient(
        DashboardStructure(
            subjects=[Subject(name="Consolidated", kind="consolidated", entities=["Acme Corp"])],
            metric_order=["revenue", "made_up_metric"],  # ebitda/net_income/total_assets omitted
        )
    )
    out = organize_claims(_ENTITIES, _METRICS, company="Acme", client=client)
    assert "made_up_metric" not in out.metric_order
    assert out.metric_order[0] == "revenue"
    assert set(out.metric_order) == set(_METRICS)  # omitted supplied metrics still kept


def test_only_one_consolidated_survives() -> None:
    client = _StubClient(
        DashboardStructure(
            subjects=[
                Subject(name="A", kind="consolidated", entities=["Acme Corp"]),
                Subject(name="B", kind="consolidated", entities=["North Region"]),
            ],
            metric_order=["revenue"],
        )
    )
    out = organize_claims(_ENTITIES, _METRICS, company="Acme", client=client)
    kinds = [s.kind for s in out.subjects]
    assert kinds == ["consolidated", "segment"]  # the second is demoted


def test_promotes_a_consolidated_when_model_marked_none() -> None:
    client = _StubClient(
        DashboardStructure(
            subjects=[Subject(name="North", kind="segment", entities=["North Region"])],
            metric_order=["revenue"],
        )
    )
    out = organize_claims(_ENTITIES, _METRICS, company="Acme", client=client)
    assert out.subjects[0].kind == "consolidated"  # grouping needs an anchor


def test_empty_vocabulary_makes_no_call() -> None:
    client = _StubClient(DashboardStructure())
    assert organize_claims({}, _METRICS, company="Acme", client=client).subjects == []
    assert organize_claims(_ENTITIES, [], company="Acme", client=client).subjects == []
    assert client.calls == []


def test_retries_once_on_an_unparseable_body() -> None:
    good = DashboardStructure(
        subjects=[Subject(name="Consolidated", kind="consolidated", entities=["Acme Corp"])],
        metric_order=["revenue"],
    )

    class _FlakyClient:
        def __init__(self) -> None:
            self.n = 0
            self.messages = SimpleNamespace(parse=self._parse)

        def _parse(self, **kwargs):
            self.n += 1
            if self.n == 1:
                Subject.model_validate({})  # missing required -> ValidationError
            return SimpleNamespace(parsed_output=good)

    client = _FlakyClient()
    out = organize_claims(_ENTITIES, _METRICS, company="Acme", client=client)
    assert client.n == 2
    assert out.subjects[0].name == "Consolidated"


def test_a_shared_entity_is_filed_under_exactly_one_subject() -> None:
    # The model lists "Acme Corp" under both the consolidated total and a segment.
    # The whole-company total must keep it; the segment must not also carry it.
    client = _StubClient(
        DashboardStructure(
            subjects=[
                Subject(name="North", kind="segment", entities=["Acme Corp", "North Region"]),
                Subject(name="Consolidated", kind="consolidated", entities=["Acme Corp"]),
            ],
            metric_order=["revenue"],
        )
    )
    out = organize_claims(_ENTITIES, _METRICS, company="Acme", client=client)
    consolidated = next(s for s in out.subjects if s.kind == "consolidated")
    assert consolidated.entities == ["Acme Corp"]
    north = next(s for s in out.subjects if s.name == "North")
    assert north.entities == ["North Region"]  # the shared entity was not double-filed
    # every entity appears under at most one subject
    seen: list[str] = []
    for s in out.subjects:
        seen += s.entities
    assert len(seen) == len(set(seen))


def test_never_raises_on_a_non_validation_error() -> None:
    class _BoomClient:
        def __init__(self) -> None:
            self.messages = SimpleNamespace(parse=self._parse)

        def _parse(self, **kwargs):
            raise RuntimeError("api exploded")

    out = organize_claims(_ENTITIES, _METRICS, company="Acme", client=_BoomClient())
    assert out.subjects == []  # degrades to empty, does not propagate

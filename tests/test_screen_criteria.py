"""Document search for qualitative screening criteria (Path B).

Hermetic: a stub stands in for the Anthropic client. These assert the contract
around the model, not the model -- above all the anti-hallucination gate: a Y/N
verdict is honored only when its evidence quote is actually present in the
document; anything else, or a silent document, resolves to unknown.
"""

from __future__ import annotations

from types import SimpleNamespace

from parser_service.screen_criteria import (
    CriteriaAssessment,
    CriterionFinding,
    assess_criteria,
)

_DOC = ["The founders are full-time on the business.", "All IP is owned by the company."]
_CRITERIA = [
    {"rule_id": "gs_01", "question": "Founder(s) full-time on the business"},
    {"rule_id": "db_03", "question": "Founder seeking full exit within 24 months"},
]


class _StubClient:
    def __init__(self, findings: list[CriterionFinding]) -> None:
        self._assessment = CriteriaAssessment(findings=findings)
        self.calls: list[dict] = []
        self.messages = SimpleNamespace(parse=self._parse)

    def _parse(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(parsed_output=self._assessment)


def test_grounded_verdict_is_honored() -> None:
    client = _StubClient(
        [
            CriterionFinding(
                rule_id="gs_01",
                verdict="Y",
                evidence="The founders are full-time on the business.",
            )
        ]
    )
    out = assess_criteria(_DOC, entity="Acme", criteria=_CRITERIA, client=client)
    assert out["gs_01"] == {
        "verdict": "Y",
        "evidence": "The founders are full-time on the business.",
    }
    # db_03 was requested but the model said nothing -> unknown.
    assert out["db_03"] == {"verdict": "unknown", "evidence": ""}


def test_ungrounded_verdict_is_downgraded_to_unknown() -> None:
    # A confident "Y" whose quote is nowhere in the document must not be trusted.
    client = _StubClient(
        [
            CriterionFinding(
                rule_id="gs_01", verdict="Y", evidence="Founders work part-time elsewhere."
            )
        ]
    )
    out = assess_criteria(_DOC, entity="Acme", criteria=_CRITERIA, client=client)
    assert out["gs_01"] == {"verdict": "unknown", "evidence": ""}


def test_grounded_negative_is_honored() -> None:
    doc = ["The IP is licensed from a third party, not owned by the company."]
    client = _StubClient(
        [
            CriterionFinding(
                rule_id="db_06",
                verdict="Y",
                evidence="The IP is licensed from a third party, not owned by the company.",
            )
        ]
    )
    out = assess_criteria(
        doc,
        entity="Acme",
        criteria=[{"rule_id": "db_06", "question": "IP owned by third party"}],
        client=client,
    )
    assert out["db_06"]["verdict"] == "Y"


def test_explicit_unknown_passes_through() -> None:
    client = _StubClient([CriterionFinding(rule_id="gs_01", verdict="unknown", evidence="")])
    out = assess_criteria(_DOC, entity="Acme", criteria=_CRITERIA, client=client)
    assert out["gs_01"] == {"verdict": "unknown", "evidence": ""}


def test_model_cannot_introduce_an_unrequested_rule() -> None:
    client = _StubClient(
        [
            CriterionFinding(
                rule_id="db_99", verdict="Y", evidence="The founders are full-time on the business."
            )
        ]
    )
    out = assess_criteria(_DOC, entity="Acme", criteria=_CRITERIA, client=client)
    assert "db_99" not in out
    assert set(out) == {"gs_01", "db_03"}  # only the requested rules


def test_empty_criteria_makes_no_call() -> None:
    client = _StubClient([])
    assert assess_criteria(_DOC, entity="Acme", criteria=[], client=client) == {}
    assert client.calls == []


def test_empty_document_is_all_unknown_without_a_call() -> None:
    client = _StubClient([CriterionFinding(rule_id="gs_01", verdict="Y", evidence="x")])
    out = assess_criteria(["", "  "], entity="Acme", criteria=_CRITERIA, client=client)
    assert out == {
        "gs_01": {"verdict": "unknown", "evidence": ""},
        "db_03": {"verdict": "unknown", "evidence": ""},
    }
    assert client.calls == []


def test_retries_once_on_an_unparseable_body() -> None:
    good = CriteriaAssessment(
        findings=[
            CriterionFinding(
                rule_id="gs_01", verdict="Y", evidence="The founders are full-time on the business."
            )
        ]
    )

    class _FlakyClient:
        def __init__(self) -> None:
            self.n = 0
            self.messages = SimpleNamespace(parse=self._parse)

        def _parse(self, **kwargs):
            self.n += 1
            if self.n == 1:
                CriterionFinding.model_validate({})  # missing required -> ValidationError
            return SimpleNamespace(parsed_output=good)

    client = _FlakyClient()
    out = assess_criteria(_DOC, entity="Acme", criteria=_CRITERIA, client=client)
    assert client.n == 2
    assert out["gs_01"]["verdict"] == "Y"

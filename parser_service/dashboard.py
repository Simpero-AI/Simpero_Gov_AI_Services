"""Per-document dashboard organization (Pipeline Inspector).

A CIM's extracted facts vary immensely by sector -- a casino has properties, a
SaaS company has product lines, a biotech has programs -- so the dashboard's
structure cannot be hardcoded. This pass reads the document's own extracted
vocabulary (which entities appear, and how often; which canonical metrics are
present) and returns a STRUCTURE the Inspector renders: how to fold the noisy
free-text entities into a few business subjects, and which metrics matter most
for this company.

Grounding is the whole discipline (handover C-10/C-11): the model only GROUPS
entities that exist and ORDERS metrics that exist -- it never sees or invents a
value. Every number on the dashboard stays a real, cited claim; this pass only
decides how to arrange them. A subject entity the model returns that we did not
supply is dropped; a metric it invents is dropped; on any failure the structure
is empty and the Inspector falls back to deterministic frequency grouping.
"""

import logging
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from .propose import DEFAULT_MODEL

logger = logging.getLogger(__name__)

# Cap the entity vocabulary sent to the model: the long singleton tail carries no
# grouping signal and just costs tokens. The most-mentioned entities are the ones
# that form real subjects.
_MAX_ENTITIES = 120

_SYSTEM = """\
You organize a company's already-extracted facts into a dashboard structure. You \
are given the ENTITIES those facts are about (with how many facts mention each) \
and the canonical financial METRICS present. Return two things:

- subjects: fold the entities into a FEW business subjects a reader would expect \
for this company -- the consolidated whole, plus its real segments / business \
units / properties / products. Exactly one subject has kind "consolidated" (the \
whole company); the rest are kind "segment". List, under each subject, the exact \
entity strings (copied verbatim from the supplied list) that belong to it. A \
one-off entity that fits no segment may be left out -- the dashboard files \
unassigned entities under "Other" itself.
- metric_order: the canonical metric names (copied exactly from the supplied \
list) ordered by how central they are to THIS company's story (revenue and \
profitability first, balance-sheet items later).

Rules: use ONLY entities and metrics from the supplied lists -- never invent \
either. You are ARRANGING facts, never creating them. Prefer a small number of \
meaningful subjects over many tiny ones.
"""


class Subject(BaseModel):
    """One business subject -- the consolidated whole or a segment -- and the raw
    entity strings that belong to it."""

    name: str = Field(description="Display name, e.g. 'Consolidated' or a segment name")
    kind: Literal["consolidated", "segment"] = Field(
        description='"consolidated" for the whole company (exactly one), else "segment"'
    )
    entities: list[str] = Field(
        default_factory=list, description="Entity strings from the supplied list, copied verbatim"
    )


class DashboardStructure(BaseModel):
    """How the Inspector should organize a deal's facts. Empty when the model
    could not organize them; the Inspector then falls back to frequency grouping."""

    subjects: list[Subject] = Field(default_factory=list)
    metric_order: list[str] = Field(default_factory=list)


def _validate(
    structure: DashboardStructure, entities: set[str], metrics: set[str]
) -> DashboardStructure:
    """Trust boundary: keep only supplied entities/metrics, drop empty subjects,
    and guarantee exactly one consolidated subject."""
    subjects: list[Subject] = []
    consolidated_seen = False
    for s in structure.subjects:
        kept = [e for e in s.entities if e in entities]
        if not kept and s.kind != "consolidated":
            continue
        kind = s.kind
        if kind == "consolidated":
            if consolidated_seen:
                kind = "segment"  # the model can only crown one whole company
            else:
                consolidated_seen = True
        subjects.append(Subject(name=s.name.strip() or "Consolidated", kind=kind, entities=kept))
    if subjects and not consolidated_seen:
        subjects[0].kind = "consolidated"  # promote the first so grouping has an anchor
    ordered = [m for m in structure.metric_order if m in metrics]
    ordered += [m for m in metrics if m not in ordered]  # metrics the model dropped keep a slot
    return DashboardStructure(subjects=subjects, metric_order=ordered)


def organize_claims(
    entities: dict[str, int],
    metrics: list[str],
    *,
    company: str,
    model: str = DEFAULT_MODEL,
    client=None,
) -> DashboardStructure:
    """Organize a document's entities + canonical metrics into a dashboard
    structure. `entities` maps each entity string to its fact count; `metrics` is
    the canonical attributes present. Never raises: an empty vocabulary or any
    error yields an empty structure (the Inspector falls back to frequency
    grouping)."""
    if not entities or not metrics:
        return DashboardStructure()

    top = sorted(entities.items(), key=lambda kv: (-kv[1], kv[0]))[:_MAX_ENTITIES]
    entity_set = {e for e, _ in top}
    metric_set = set(metrics)

    if client is None:
        import anthropic

        client = anthropic.Anthropic()

    rendered_entities = "\n".join(f"- {e} ({n} facts)" for e, n in top)
    rendered_metrics = "\n".join(f"- {m}" for m in metrics)
    user = (
        f"COMPANY: {company}\n\n"
        f"ENTITIES (with fact counts):\n{rendered_entities}\n\n"
        f"CANONICAL METRICS PRESENT:\n{rendered_metrics}"
    )

    def call():
        return client.messages.parse(
            model=model,
            max_tokens=2000,
            system=[{"type": "text", "text": _SYSTEM, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
            output_format=DashboardStructure,
        )

    try:
        response = call()
    except ValidationError:
        logger.warning("dashboard: unparseable body for %s; retrying once", company)
        response = call()

    parsed = response.parsed_output
    if parsed is None:
        return DashboardStructure()
    return _validate(parsed, entity_set, metric_set)

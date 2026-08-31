"""Document search for qualitative screening criteria (Path B, "search just in
case").

Some screening rules are qualitative facts a CIM only *sometimes* states outright
-- "founder(s) full-time on the business", "founder seeking full exit within 24
months", "IP owned by the company". They don't fall out of the numeric/table
extraction, and the deterministic engine has no evaluator for them. Rather than
give up on them, this pass searches the parsed document for each selected
criterion and returns a grounded verdict: Y or N when the document actually says
so (with the exact supporting quote), or "unknown" when it is silent -- which
routes the deal to human review, never a guess.

Posture (handover C-10/C-11): AI-at-the-edge PROPOSAL only, and nothing it says
is trusted verbatim. A Y/N verdict is honored ONLY when its evidence quote is
actually found in the document text (see `_grounds`); an ungrounded or
unsupported verdict is downgraded to "unknown" here, deterministically. The
backend persists these findings and its evaluators turn them into rule verdicts;
the recommendation stays deterministic. Only criteria we were asked about are
returned, and only for the rule ids supplied -- the model cannot introduce a
rule.
"""

import logging
import unicodedata
from typing import Literal

from pydantic import BaseModel, Field

from .llm_client import make_client, parse_with_retry
from .propose import DEFAULT_MODEL

logger = logging.getLogger(__name__)

# Qualitative facts (team commitment, cap table, IP, exit intent) are scattered
# through a CIM, not just the overview, so this pass reads a far wider window than
# the sector/HQ classifier -- bounded only to keep a single call sane.
_MAX_CHARS = 120_000

_SYSTEM = """\
You are screening a company's information memorandum against a fund's diligence \
criteria. For EACH criterion you are given, decide whether the DOCUMENT supports \
it, about the TARGET COMPANY itself (never an investor, advisor, or comparable):

- verdict "Y": the document affirmatively supports the criterion.
- verdict "N": the document affirmatively contradicts it.
- verdict "unknown": the document does not address it either way.

Rules:
- Base every "Y"/"N" on the document ALONE, and put the EXACT verbatim source \
sentence that settles it in `evidence` (copied character-for-character from the \
text). If you cannot quote a supporting sentence, the verdict is "unknown" with \
evidence "". Never infer, assume, or use outside knowledge.
- When in doubt, choose "unknown" -- a wrong "Y"/"N" is worse than "unknown", \
which simply sends the question to a human.
- Return exactly one finding per supplied criterion, echoing its `rule_id`.
"""


class CriterionFinding(BaseModel):
    """One criterion's grounded verdict. `evidence` is the verbatim supporting
    quote for a Y/N, or "" when unknown."""

    rule_id: str = Field(description="The rule id from the supplied criterion, echoed exactly")
    verdict: Literal["Y", "N", "unknown"] = Field(description='"Y", "N", or "unknown"')
    evidence: str = Field(
        default="", description="Verbatim source sentence supporting a Y/N, or '' for unknown"
    )


class CriteriaAssessment(BaseModel):
    findings: list[CriterionFinding] = Field(default_factory=list)


def _document_text(page_texts: list[str]) -> str:
    return "\n".join(page_texts)[:_MAX_CHARS]


def _normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split()).casefold()


def _grounds(evidence: str, haystack_norm: str) -> bool:
    """Whether `evidence` is actually present in the document (whitespace/case
    tolerant). The anti-hallucination gate: a Y/N with no locatable quote cannot
    be trusted, so it is not honored."""
    needle = _normalize(evidence)
    return len(needle) >= 8 and needle in haystack_norm


def _unknown() -> dict:
    return {"verdict": "unknown", "evidence": ""}


def assess_criteria(
    page_texts: list[str],
    *,
    entity: str,
    criteria: list[dict],
    model: str = DEFAULT_MODEL,
    client=None,
) -> dict[str, dict]:
    """Assess each supplied criterion against the document. Returns
    {rule_id: {"verdict": "Y"|"N"|"unknown", "evidence": str}} for EVERY supplied
    rule id (a criterion the model omits, or whose Y/N quote is not found in the
    document, resolves to unknown). Never raises for a normal empty document.

    `criteria` is a list of {"rule_id": str, "question": str}. `page_texts` is the
    per-page text in document order (result.pages[*].text)."""
    wanted = {
        c["rule_id"]: c["question"] for c in criteria if c.get("rule_id") and c.get("question")
    }
    if not wanted:
        return {}

    document = _document_text(page_texts)
    if not document.strip():
        return {rule_id: _unknown() for rule_id in wanted}

    if client is None:
        client = make_client()

    rendered = "\n".join(f"- rule_id={rid}: {question}" for rid, question in wanted.items())
    user = (
        f"TARGET COMPANY: {entity}\n\n"
        f"CRITERIA:\n{rendered}\n\n"
        f'DOCUMENT (verbatim):\n"""\n{document}\n"""'
    )

    def call():
        return client.messages.parse(
            model=model,
            max_tokens=2000,
            system=[{"type": "text", "text": _SYSTEM, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
            output_format=CriteriaAssessment,
        )

    # Route through the shared retry so a transient grammar-compilation 400 is
    # retried here too (same structured-output path as the prose tiers), instead
    # of propagating and silently dropping this stage. page_no=0 is the sentinel
    # for a whole-document (non per-page) call.
    response = parse_with_retry(call, page_no=0, what="screen_criteria")

    parsed = response.parsed_output
    results = {rule_id: _unknown() for rule_id in wanted}
    if parsed is None:
        return results

    haystack = _normalize(document)
    for finding in parsed.findings:
        if finding.rule_id not in wanted:
            continue  # the model may not introduce a rule we did not ask about
        if finding.verdict in ("Y", "N") and _grounds(finding.evidence, haystack):
            results[finding.rule_id] = {"verdict": finding.verdict, "evidence": finding.evidence}
        elif finding.verdict in ("Y", "N"):
            logger.info(
                "screen_criteria: %s verdict %s had no locatable quote; -> unknown",
                finding.rule_id,
                finding.verdict,
            )
    return results

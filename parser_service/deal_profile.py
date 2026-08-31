"""Deal-profile classification (Path B, mandate-fit screening).

Reads the company's own overview text and reports two facts *about the target
company itself* -- its primary sector and its headquarters geography -- and,
when given the fund's approved options, judges whether the company falls inside
them. This is what lets the backend screen a deal against the analyst's Mandate
Builder selections (gs_07 = HQ in an approved geography, gs_08 = operates in an
approved sector). These are qualitative, no-magnitude facts, so they don't fall
out of the numeric/table extraction the way revenue or burn do; a small
classification pass over the parsed prose is the cheapest accurate way to derive
them.

Why the fit judgement lives here and not in the backend: the approved-option
vocabulary is admin-managed runtime data (mandate_options), not a fixed enum, and
the backend deliberately makes no model calls. Matching a company's free-text
sector ("a vertical SaaS platform for dental clinics") onto an admin option
("Healthcare IT") is a semantic comparison -- exactly the AI-at-the-edge seam the
handover sanctions (C-10/C-11) -- so it belongs where the model already runs.

Posture: this is a PROPOSAL only, and nothing the model says is trusted verbatim.
A returned raw value must be grounded in a quoted span or it is null; a returned
fit option must be one of the options we supplied (fold-equal) or it is
downgraded to "unknown" here, deterministically. The backend maps the fit onto
`deal.sector`/`deal.hq_geography` and the gs_07/gs_08 verdict stays deterministic.
Nothing here decides approval.
"""

import logging
import unicodedata
from typing import Literal

from pydantic import BaseModel, Field

from .llm_client import make_client, parse_with_retry
from .propose import DEFAULT_MODEL

logger = logging.getLogger(__name__)

# Sector/HQ live in the company overview at the front of a CIM/deck, so a bounded
# window off the top is both cheaper and higher-signal than the whole document.
_OVERVIEW_PAGES = 6
_OVERVIEW_CHARS = 12_000

_SYSTEM = """\
You read a company's information memorandum / pitch deck and report facts ABOUT \
THE TARGET COMPANY ITSELF (never an investor, advisor, customer, or comparable \
named in passing).

Always report, grounded in the document:
- sector: the company's primary industry / sector, in the document's own words \
(e.g. "enterprise SaaS", "medical devices", "fintech lending").
- hq_geography: where the company is headquartered -- the country, plus the \
region/state/city if the document gives it (e.g. "Toronto, Ontario, Canada").
Report a value ONLY if the document states it; quote the exact source text in the \
matching `_evidence` field. If the document does not state it, return null and "".
Never infer or guess a value you cannot ground in a quote.

Mandate fit. You are given the fund's approved sector options and approved \
geography options. For each of sector and HQ, decide `*_fit`:
- status "match" + `option` = the single approved option (copied EXACTLY from the \
supplied list) the company falls under. Match on meaning, generously: a specialized \
description still matches a broader approved option it clearly belongs to.
- status "outside" + `option` null: the company's sector/HQ IS determinable from \
the document and clearly does not fall under ANY supplied option.
- status "unknown" + `option` null: you cannot tell, or no options were supplied. \
When unsure between "outside" and "unknown", choose "unknown" -- never guess a miss.
Never invent an option that is not in the supplied list.
"""


class MandateFit(BaseModel):
    """Whether the company falls under one of the fund's approved options for a
    dimension. `option` is set (verbatim from the supplied list) only when
    status is "match"."""

    status: Literal["match", "outside", "unknown"] = Field(
        description='"match", "outside", or "unknown" (see the rules)'
    )
    option: str | None = Field(
        default=None,
        description="The matched approved option, copied exactly, when status is match; else null",
    )


class DealProfile(BaseModel):
    """The target company's sector + HQ as stated in its own document, plus its
    fit against the fund's approved options. Raw fields are None when the
    document is silent; a `*_fit` is None when no options were supplied."""

    sector: str | None = Field(description="Primary sector/industry as stated, or null")
    sector_evidence: str = Field(
        default="", description="Verbatim quote that settles the sector, or '' if none"
    )
    hq_geography: str | None = Field(
        description="Headquarters location as stated (country + region if given), or null"
    )
    hq_evidence: str = Field(
        default="", description="Verbatim quote that settles the HQ, or '' if none"
    )
    sector_fit: MandateFit | None = Field(
        default=None,
        description="Fit against the approved sector options, or null if none supplied",
    )
    hq_fit: MandateFit | None = Field(
        default=None,
        description="Fit against the approved geography options, or null if none supplied",
    )


def _overview_text(page_texts: list[str]) -> str:
    return "\n".join(page_texts[:_OVERVIEW_PAGES])[:_OVERVIEW_CHARS]


def _normalize_label(value: str) -> str:
    """The backend's normalize_label, replicated: NFKC-normalize, collapse
    whitespace, casefold. A fit option validated against this is guaranteed to
    fold-match the same option on the backend's approves_* check."""
    return " ".join(unicodedata.normalize("NFKC", value).split()).casefold()


def _validated_fit(fit: MandateFit | None, options: list[str] | None) -> MandateFit | None:
    """Trust boundary for the model's fit. No options supplied -> no fit (None).
    A "match" is honored only when its option is one we supplied (fold-equal), and
    the returned option is OUR verbatim string, not the model's echo -- so the
    backend writes the exact approved label. Anything else collapses to
    "unknown", never a fabricated or mismatched option."""
    if not options:
        return None
    if fit is None or fit.status != "match":
        return MandateFit(status=fit.status if fit is not None else "unknown", option=None)
    if fit.option is None:
        return MandateFit(status="unknown", option=None)
    wanted = _normalize_label(fit.option)
    canonical = next((opt for opt in options if _normalize_label(opt) == wanted), None)
    if canonical is None:
        logger.warning("deal_profile: model returned an off-list fit option; downgraded to unknown")
        return MandateFit(status="unknown", option=None)
    return MandateFit(status="match", option=canonical)


def classify_deal_profile(
    page_texts: list[str],
    *,
    entity: str,
    sector_options: list[str] | None = None,
    geo_options: list[str] | None = None,
    model: str = DEFAULT_MODEL,
    client=None,
) -> DealProfile:
    """One structured call over the company overview -> the target's sector + HQ
    (each grounded in a quoted span or null) and, when options are supplied, their
    fit against the fund's approved options. Never raises for a normal empty/short
    document: an empty overview returns an all-null profile without an API call.

    `page_texts` is the per-page text in document order (result.pages[*].text).
    `sector_options` / `geo_options` are the org's approved option display strings
    (post sub-tree expansion) -- pass None to skip the fit judgement entirely."""
    overview = _overview_text(page_texts)
    if not overview.strip():
        return DealProfile(sector=None, hq_geography=None)

    if client is None:
        client = make_client()

    def _render(options: list[str] | None) -> str:
        if not options:
            return "(none supplied)"
        return "\n".join(f"- {opt}" for opt in options)

    user = (
        f"TARGET COMPANY: {entity}\n\n"
        f"APPROVED SECTOR OPTIONS:\n{_render(sector_options)}\n\n"
        f"APPROVED GEOGRAPHY OPTIONS:\n{_render(geo_options)}\n\n"
        f'DOCUMENT OVERVIEW (verbatim):\n"""\n{overview}\n"""'
    )

    def call():
        return client.messages.parse(
            model=model,
            max_tokens=1200,
            system=[{"type": "text", "text": _SYSTEM, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
            output_format=DealProfile,
        )

    # Route through the shared retry so the grammar-compilation 400 this PR
    # narrows is retried here too -- messages.parse(output_format=...) is the
    # exact structured-output path that triggers it. page_no=0 is the sentinel
    # for a whole-document (non per-page) call.
    response = parse_with_retry(call, page_no=0, what="deal_profile")

    parsed = response.parsed_output
    if parsed is None:
        # Nothing groundable -> all-null, same fail-safe as the binding audit.
        return DealProfile(sector=None, hq_geography=None)

    # Enforce the trust boundary on the fit regardless of what the model returned.
    parsed.sector_fit = _validated_fit(parsed.sector_fit, sector_options)
    parsed.hq_fit = _validated_fit(parsed.hq_fit, geo_options)
    return parsed

# Simpero Gov AI Services

The document parse service. Takes untrusted bytes — PDF, XLSX or DOCX — and returns an **exactly-citable index**: the structure needed to prove that a claim came from a specific place in a specific document.

Split out of `Simpero_AI_Gov_Alpha` with history preserved (`git filter-repo`); `git log` and `git blame` still reach the original commits.

## Why this is its own repo

The parse service and the backend have genuinely different shapes, and the monorepo was charging both of them for the difference:

- **Dependencies.** This service needs `docling`, and `docling` needs PyTorch. The backend needs none of it. In the monorepo, every backend PR installed the full ML tree to run tests that never touched it. `docling`, `pypdf`, `openpyxl` and `reportlab` appear nowhere outside the parser — the backend now sheds torch entirely.
- **Trust boundary.** This service holds no tenant data and talks to no database. Its whole CI runs on a bare runner with no Postgres, no Valkey, no auth secrets. That is a property worth protecting: if a job here starts needing a database, the boundary has been crossed by mistake.

The parse service imports nothing from the backend's `app/` package, which is what made the split mechanical rather than a rewrite.

## Layout

```
parser_service/       the service
  main.py             POST /parse — dispatches on the format sniffed from bytes
  docling_parser.py   PDF  -> positioned page index (page, char span, bbox)
  xlsx_parser.py      XLSX -> addressed cells (sheet, cell_ref)
  docx_parser.py      DOCX -> paragraph index (paragraph, char span)
  resolver.py         the citation trust boundary — one exact-span rule
  ooxml.py            zip-bomb guard shared by the XLSX and DOCX lanes
  scale.py            "(in thousands)" header scaling, gated on value_type
  table_extract.py    table reconstruction
contracts/            the C3 seam: claims.schema.json + its tests
tests/                pytest suite covering all three lanes
```

## The seam

`contracts/claims.schema.json` is the contract between this service and the backend. This repo **owns** it; the backend pins it and runs a drift check against the pinned copy. A change to the schema is a change to both sides, and CI fails on either side if they disagree — a silent drift at the seam is the exact failure mode the product exists to prevent.

## Contract rules that are not negotiable

- **A claim is an assertion pending verification, not a verified truth.** Hence `status` + `verification_method`, never a `confidence` float that conflates "the extractor felt good" with "this was checked".
- **Provenance is all-or-nothing.** A quote resolves to an exact span, or the claim is emitted with `status=missing`. There is no "closest sentence" and no partial citation — an approximate span looks correct while pointing at unrelated text, which is worse than no claim at all.
- **The three lanes share a trust bar, not a shape.** A PDF span guarantees a bbox; a DOCX span has no geometry, because a Word file is a flow document with no page layout to report. The shapes differ honestly rather than flattening to a lowest common denominator.
- **Format is sniffed from bytes**, never from a filename or a client-supplied `Content-Type`. Dispatching the wrong lane on a lie is how a parser gets fed what it never guards against.

## Development

```bash
uv sync --all-extras --dev

uv run ruff check . && uv run ruff format --check .
uv run pyright
uv run pytest tests/ -q
uv run --with jsonschema pytest -q contracts/test_claims_contract.py
```

`local_corpus`-marked tests read real corpus documents that are not committed, and skip explicitly when the fixture is absent. They are supplementary — not CI coverage.

## Running it

### The parse service — deterministic, no key

```bash
uv run uvicorn parser_service.main:app --reload --port 8001

# or the deployable image
docker build -t simpero-parser .
docker run --rm -p 8001:8001 simpero-parser
```

`POST /parse` takes raw bytes as the request body and returns a tagged result: `kind` names the lane, and exactly one of `pages` / `sheets` / `paragraphs` is populated. `GET /health` is the liveness probe. This layer reads structure only — it calls no model and needs no credential.

### Parse to claims — the seam's left half

`scripts/emit_claims.py` turns a PDF into the claims JSON that crosses the C3 seam, in three tiers, each a strict superset of the last:

```bash
# tables only — deterministic, no network, no key
uv run python scripts/emit_claims.py <cim.pdf> --entity "PTL Group" > claims.json

# + numeric facts stated in prose (one model call per prose page)
ANTHROPIC_API_KEY=... uv run python scripts/emit_claims.py <cim.pdf> \
    --entity "PTL Group" --prose > claims.json

# + claims that carry no number (implies --prose)
ANTHROPIC_API_KEY=... uv run python scripts/emit_claims.py <cim.pdf> \
    --entity "PTL Group" --qualitative > claims.json
```

`--prose` and `--qualitative` call the Anthropic API once per prose page and require `ANTHROPIC_API_KEY` (or `ANTHROPIC_AUTH_TOKEN`) in the environment; without it the script fails closed at the door rather than part way through a document. The table tier needs neither. Do not pass a key on the command line for a real CIM — export it, or use a secret manager; the prose tiers send page text to the API, so treat a confidential document accordingly.

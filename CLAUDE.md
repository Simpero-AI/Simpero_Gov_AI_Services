# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Document parse service: takes untrusted bytes (PDF, XLSX, DOCX) and returns an exactly-citable index — the structure needed to prove a claim came from a specific place in a specific document. Split out of `Simpero_AI_Gov_Alpha` with history preserved. It holds no tenant data and talks to no database (it does hold a Valkey connection for job queueing — see `worker.py` below — but that's a queue, not a database); CI runs on a bare runner and must stay that way — a job needing Postgres/secrets means the trust boundary was crossed by mistake.

## Commands

```bash
uv sync --all-extras --dev          # install (uv-managed, Python >=3.11)

uv run ruff check . && uv run ruff format --check .
uv run pyright
uv run pytest tests/ -q             # full suite
uv run pytest tests/test_resolver.py -q            # one file
uv run pytest tests/test_resolver.py::test_name -q # one test
uv run --with jsonschema pytest -q contracts/test_claims_contract.py  # contract check

uv run uvicorn parser_service.main:app --reload --port 8001  # run locally
```

Tests marked `local_corpus` read real corpus documents (confidential, never committed, live in `tests/test_data/`) and skip explicitly when absent — they are supplementary, not CI coverage.

## Architecture

Pipeline: `dispatch.py` sniffs format **from bytes** (never filename or Content-Type) and dispatches to one of three lanes — `docling_parser.py` (PDF → pages with char-level geometry), `xlsx_parser.py` (XLSX → addressed cells), `docx_parser.py` (DOCX → paragraphs, no geometry). `ParseResponse` is tagged by `kind`; exactly one of `pages`/`sheets`/`paragraphs` is populated. The lanes share a trust bar, not a shape — do not flatten them into a common denominator. `main.py`'s `POST /parse` is the HTTP entry point onto this dispatch logic; `worker.py` (below) is the other one.

Downstream of parsing:
- `resolver.py` — the citation trust boundary. A quote resolves to an **exact, unambiguous** span (whitespace-flexible, otherwise literal) or returns None. Found more than once = None. No fuzzy matching, no "closest sentence" — a wrong span looks correct, which is worse than no claim.
- `emit.py` — turns claim candidates + parse records into Claim JSON rows per `contracts/claims.schema.json`. Status semantics: PDF/table → `proposed`; literal XLSX cell → `cited/direct_read`; formula cell → `proposed` (this service never executes formulas and never trusts cached results); unresolved → `missing` with **no span at all**.
- `normalize.py` — the single source of truth for numeric-token normalization (spaced thousands separators inside PDFs). Both the flat page index and table cells apply it; never duplicate the rule.
- `scale.py` — "(in thousands)" header scaling, gated on `value_type`.
- `elements.py` / `table_extract.py` — table/chart regions as citable elements; charts are flagged `chart_data_not_extracted`, never guessed from pixels.
- `ooxml.py` — zip-bomb guard shared by XLSX and DOCX.
- `document_cache.py` — best-effort DoclingDocument cache in object storage (Spaces/S3); errors never fail a parse; disabled until configured (no confidential content on unencrypted local disk).
- `inspect.py` — developer-only visual harness (`python -m parser_service.inspect`, renders to `parser_service/out/`); its deps (pypdfium2, pillow) are dev-group only and must stay out of `[project].dependencies` so the Docker image doesn't carry them.
- `worker.py` — a SAQ worker consuming the shared `"parse"` Valkey queue that `Simpero_AI_Gov_Alpha`'s `app/jobs/parse_client.py` enqueues onto. Reuses `dispatch.parse_bytes`, the same logic `POST /parse` uses, and writes results to Spaces as a bucket+key pointer rather than returning raw parsed data through the queue. Currently dead-end infrastructure — nothing in Alpha calls the enqueue function yet.

## The contract seam

`contracts/claims.schema.json` is owned by this repo; the backend pins a copy and drift-checks it. A schema change is a change to both sides — CI fails on either side if they disagree.

Non-negotiable contract rules:
- A claim is an assertion pending verification. Trust is `status` + `verification_method`, never a confidence float.
- Provenance is all-or-nothing: exact span or `status=missing`. Never a partial or approximate citation.

## Conventions

- Pyright is load-bearing for the PageIndex/char_map invariants — keep it clean.
- Ruff: line length 100, rules E/W/F/I/B/UP/SIM (E501 ignored in `tests/` and `contracts/`).
- pytest runs with `asyncio_mode = "auto"`.
- `CharBox.precision` (`"char"` vs `"word"`) has no default on purpose: word-precision boxes are the containing word's bbox, not glyph geometry — consumers must not treat them as exact highlight coordinates.

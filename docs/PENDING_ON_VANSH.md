# Pending on Vansh — parser-service deploy/destroy pipeline

Everything the code needs is written once Phases 1–4 of the plan land
(`.github/workflows/deploy.yml`, `.github/workflows/destroy.yml`, `terraform/`,
`docker-compose.prod.yml`, `Caddyfile`). None of it can run end-to-end until
the steps below are done — they're all manual, DO-console/GitHub-console
actions that only someone with account access can do. Full rationale for each
decision lives in `docs/plans/parser-droplet-deployment.md` (§3, §5); this doc
is just the walkthrough. It deliberately mirrors the sibling repo's own
`Simpero_AI_Gov_Alpha/docs/PENDING_ON_VANSH.md` — same shape, same DO account,
a second independent set of credentials.

**Status as of 2026-07-29: steps 1–8 and 10 are done for `staging`.**
`staging` is fully populated and ready for a first `deploy.yml` dispatch —
see the status table below for exact confirmation of what's set. `production`
is deliberately still scaffolded-only (SSH keys set, everything else blank)
per the production-deferral note further down — its Spaces bucket hasn't
been decided yet, so nothing else was guessed. Step 9 (DNS access) is the
only genuinely open item for staging's own bring-up.

**A rule for this handoff, not just a suggestion:** never paste raw `.env`
file contents (this repo's or Alpha's) into a message to Claude or any other
tool while working through this list. Where a step needs you to confirm a
value exists, confirm it (e.g. "yes, `CLERK_SECRET_KEY` is set on the staging
droplet") — don't paste the value or the file itself. GitHub Environment
secrets are the only place these values should live going forward.

---

## 1. Generate 2 new SSH deploy keypairs

Staging and production each need their own keypair, and — separately —
**neither can reuse Alpha's existing keypairs.** DigitalOcean deduplicates
`digitalocean_ssh_key` resources by the key's fingerprint (the actual key
material), not by the name you give it in the console or in Terraform. This
repo's Terraform state is completely independent from Alpha's. If you
register a public key DO has already seen from Alpha's state, whichever
`apply` runs second either silently no-ops against the wrong resource or
collides — not a risk worth taking to save two `ssh-keygen` calls.

**Steps (run twice, once per environment):**
```bash
ssh-keygen -t ed25519 -f simpero-services-deploy-staging -C "simpero-services-deploy-staging"
ssh-keygen -t ed25519 -f simpero-services-deploy-production -C "simpero-services-deploy-production"
```
Save both pairs somewhere secure (password manager, not this repo). For each
pair you'll use:
- The **public** key content (`.pub` file) → that environment's
  `TF_VAR_ssh_public_key` GitHub secret (step 7).
- The **private** key content → that environment's `DROPLET_SSH_PRIVATE_KEY`
  GitHub secret (step 7).

Once both are stored as GitHub secrets, delete the local copies or keep them
in a password manager — no workflow reads them from disk.

**Status: Pending.**

---

## 2. Confirm/create the Spaces access key(s) for Terraform state

This repo's Terraform state lives in the **same two buckets** Alpha and the
frontend already use — `simpero-tf-state-staging` and
`simpero-tf-state-production` — just under a new `services/` key prefix
(`services/staging.tfstate`, `services/production.tfstate`) rather than
`backend/` or `frontend/`. Nothing new needs creating at the bucket level.
What needs confirming is the **access key**.

**The bucket-scoped-key nuance (this is the part that trips people up):**
DigitalOcean limited-access Spaces keys are scoped to exactly **one bucket**
each, fixed permanently at key-creation time — a key can't be re-scoped to
cover a second bucket later. That means staging and production almost
certainly need **two separate key pairs**, one granted to each bucket, not
one key that magically spans both. Alpha's own `PENDING_ON_VANSH.md` (step 4)
already worked through this and produced exactly two such pairs, shared with
the frontend repo. This repo doesn't need brand-new keys — it needs the
**same two pairs Alpha already has**, reused as-is (same Access Key ID /
Secret Access Key, copied into this repo's GitHub secrets in step 7). Do not
request or create a third pair "for this repo" — that would just be a third
credential with the exact same blast radius as the existing two.

**What you're confirming, per bucket:**
1. DO console → **Spaces → API → Spaces access keys**.
2. Find the limited-access key already granted to `simpero-tf-state-staging`.
   Confirm its permission column shows **Read, Write, Delete** — not just
   Read/Write. Delete matters because `use_lockfile` (this repo's Terraform
   backend config, matching Alpha's) creates *and removes* a lock object on
   every single `terraform plan`, not just `apply`. A delete-less grant fails
   at plan time, cleanly but every time.
3. Repeat for the key granted to `simpero-tf-state-production`.
4. If either key is missing delete permission, it can't be edited in place —
   delete it and create a new limited-access key scoped to that one bucket,
   then update whichever GitHub secret(s) (in this repo and Alpha's/the
   frontend's) referenced the old one.
5. Get the Access Key ID and Secret Access Key values for both pairs ready
   for step 7 (`SPACES_ACCESS_KEY_ID` / `SPACES_SECRET_ACCESS_KEY`, per
   environment).

**Status: Pending.**

---

## 3. Confirm object versioning is enabled on both state buckets

This is the **only recovery mechanism** if a leaked credential, a bad
`terraform destroy`, or a stray delete from any of the three repos now
sharing these two buckets (Alpha, the frontend, and this one) ever touches
state it shouldn't. There's no way to fence the repos apart at the DO level —
a key that can touch the bucket can touch every prefix in it, including this
repo's `services/` state.

Alpha's own `PENDING_ON_VANSH.md` records this as already confirmed done for
both buckets as part of that repo's setup — but **re-verify it here rather
than trusting that note**, since a versioning setting could in principle have
been changed since. DigitalOcean's console doesn't expose this toggle
directly; check via the S3-compatible API with your account's **full-access**
Spaces key (not either bucket-scoped key from step 2 — bucket-configuration
reads are blocked for limited-access keys by design):

```bash
export AWS_ACCESS_KEY_ID=<your account's full-access Spaces key>
export AWS_SECRET_ACCESS_KEY=<its secret>

aws s3api get-bucket-versioning \
  --bucket simpero-tf-state-staging \
  --endpoint-url https://tor1.digitaloceanspaces.com

aws s3api get-bucket-versioning \
  --bucket simpero-tf-state-production \
  --endpoint-url https://tor1.digitaloceanspaces.com
```
Each should return `Status: Enabled`. If either doesn't, enable it before
going any further — this is a **go/no-go gate**, not a nice-to-have: do not
run a real `terraform apply` for this repo against either bucket until both
come back enabled.

**Status: Pending (re-verification only expected — should already be true).**

---

## 4. Mint a new DigitalOcean API token for this repo

**Do not reuse Alpha's `TF_VAR_do_token` value here.** Same DO account,
therefore the same underlying power either way, but a **separate token gives
independent revocation** — if this repo's token ever leaks (a compromised
Actions run, a misconfigured log, whatever), you rotate exactly one token
without also having to update Alpha's and the frontend's pipelines in the
same panic.

**Steps:**
1. DO console → **API → Tokens → Generate New Token**.
2. Name it clearly and consistently, e.g. `simpero-services-tf-token` — so a
   year from now it's obvious which repo owns it.
3. If DigitalOcean's granular token scopes are available on this account,
   scope it to **droplet, firewall, ssh_key, project — read/write** only
   (this repo's Terraform never touches databases, DNS, or anything else).
   If granular scopes aren't offered, use full-scope — same as Alpha did —
   but keep the naming convention so it's identifiable in the token list.
4. Copy the token value immediately (DO only shows it once) and hold it for
   step 8.

**Status: Pending.**

---

## 5. Confirm the DO Project names exist and are spelled exactly right

Terraform looks these up by a **data source**, not by creating them — a
misspelling or case mismatch fails cleanly at `terraform plan`, but it still
blocks the run. The names this repo's plan expects: **`Simpero-Staging`** and
**`Simpero-Prod`**, case-sensitive.

**Flag before you take this at face value:** Alpha's own `PENDING_ON_VANSH.md`
(§5b) records its staging droplet as living in a project literally named
`"Simpero"` — not `"Simpero-Staging"`. This repo's plan doc asserts
`Simpero-Staging` "matching Alpha's `.tfvars`," but that doesn't match what
Alpha's own runbook says is actually in the DO console. **Check the DO
console directly (Projects list) before filling in `terraform/staging.tfvars`
— don't trust either document's name over the other.** Whatever you find
there is the name that goes in `do_project_name` in this repo's
`staging.tfvars`.

**Steps:**
1. DO console → **Projects**.
2. Confirm the exact staging project name (resolve the `Simpero` vs.
   `Simpero-Staging` discrepancy above) and the exact production project name
   (expected `Simpero-Prod`).
3. If either doesn't exist yet, decide whether to create it or point this
   repo's droplet at an existing project instead — don't create a
   fourth/fifth ad hoc project without checking what's already there, since
   these are meant to be shared across the frontend, Alpha, and this repo.

**Status: Pending — includes resolving a naming discrepancy, not just a
lookup.**

---

## 6. Generate 2 `PARSER_API_KEY` values

This is the single most important credential in this whole setup. `POST
/parse` has no other protection — no IP allowlist, no rate limit — so this
key is the entire access control standing between the internet and an
unauthenticated ML pipeline running on untrusted document bytes. Treat it
with the weight that implies.

**Steps:**
```bash
openssl rand -hex 32   # staging
openssl rand -hex 32   # production — run again, do not reuse the staging value
```
Each environment gets its **own** value — one per environment, never shared,
never reused across staging and production (a staging leak shouldn't hand
someone production access for free). Hold both for step 7.

**Status: Pending.**

---

## 7. Create 4 GitHub Environments and populate their secrets

GitHub → this repo → **Settings → Environments → New environment**. Create
all four:

| Environment name | Required reviewers? |
|---|---|
| `staging-plan` | No |
| `production-plan` | No |
| `staging` | **Yes — add yourself** |
| `production` | **Yes — add yourself** |

For `staging` and `production`, after creating them, open each one's
settings and add **"Required reviewers"** with yourself. Leave `staging-plan`
and `production-plan` unprotected — the whole point of the plan/apply split
is that the plan step is reviewable *before* it's gated, and Terraform plans
against these credentials can't mutate infrastructure on their own.

Deployment branches: match Alpha's pattern — `staging`/`staging-plan` allow
`main` + `staging`; `production`/`production-plan` allow `main` only.

### Secrets — reproduced from the plan doc's §3 table

For each Environment: that Environment's page → **Environment secrets → Add
secret**. `TF_VAR_ssh_public_key` and the two Spaces credentials need to be
set in **both** an environment's `-plan` and its matching gated Environment
(GitHub doesn't support one Environment inheriting another's secrets — this
duplication is unavoidable).

**`staging-plan`:**

| Secret | Value | Source |
|---|---|---|
| `TF_VAR_ssh_public_key` | Staging keypair's public key content | Fresh — step 1 |
| `SPACES_ACCESS_KEY_ID` | Staging state-bucket key's Access Key ID | **Shared with Alpha — same value, just copy it** (step 2) |
| `SPACES_SECRET_ACCESS_KEY` | Staging state-bucket key's secret | **Shared with Alpha — same value** (step 2) |

**`staging`:** (same three secrets/values as `staging-plan`, plus:)

| Secret | Value | Source |
|---|---|---|
| `DROPLET_HOST` | *(leave empty — fill in after first successful apply, step 9)* | — |
| `DROPLET_SSH_PRIVATE_KEY` | Staging keypair's private key content | Fresh — step 1 |
| `PARSER_API_KEY` | Staging `PARSER_API_KEY` value | Fresh — step 6 |
| `PARSER_HOSTNAME` | `services-staging.simpero.com` | Fresh — fixed value |
| `PARSER_SPACES_BUCKET` | Doc-cache bucket name (`simpero-cim-xlsx-upload`) | **Shared with Alpha — same value** |
| `PARSER_SPACES_REGION` | Doc-cache bucket's region | **Shared with Alpha — same value** |
| `PARSER_SPACES_ENDPOINT_URL` | Doc-cache Spaces endpoint URL | **Shared with Alpha — same value** |
| `PARSER_SPACES_ACCESS_KEY_ID` | Doc-cache Spaces key ID | **Shared with Alpha — same value** |
| `PARSER_SPACES_SECRET_ACCESS_KEY` | Doc-cache Spaces secret | **Shared with Alpha — same value** |
| `PARSER_SPACES_KEY_PREFIX` | `parser/document-cache/staging` | Fresh — new, per-environment (§1.10 of the plan — this is what stops staging and production sharing one cache namespace) |
| `VALKEY_URL` | Staging Valkey connection string | **Shared with Alpha — same value, same instance, just copy it** (step 10 — this repo's worker consumes a different queue name on the SAME instance Alpha uses, not a new one) |
| `PARSER_RESULTS_KEY_PREFIX` | `parser/parse-results/staging` | Fresh — new, per-environment (step 10) |

**`production-plan`:** same shape as `staging-plan`, with production's
keypair (step 1) and production's state-bucket key (step 2 — this one is
also shared with Alpha, just the production-bucket-scoped pair rather than
staging's).

**`production`:** same shape as `staging`, with production's values:

| Secret | Value | Source |
|---|---|---|
| `DROPLET_HOST` | *(leave empty — fill in after first successful apply, step 9)* | — |
| `DROPLET_SSH_PRIVATE_KEY` | Production keypair's private key content | Fresh — step 1 |
| `PARSER_API_KEY` | Production `PARSER_API_KEY` value | Fresh — step 6 |
| `PARSER_HOSTNAME` | `services.simpero.com` | Fresh — fixed value |
| `PARSER_SPACES_BUCKET` | Doc-cache bucket name (`simpero-cim-xlsx-upload`) | **Shared with Alpha — same value** |
| `PARSER_SPACES_REGION` | Doc-cache bucket's region | **Shared with Alpha — same value** |
| `PARSER_SPACES_ENDPOINT_URL` | Doc-cache Spaces endpoint URL | **Shared with Alpha — same value** |
| `PARSER_SPACES_ACCESS_KEY_ID` | Doc-cache Spaces key ID | **Shared with Alpha — same value** |
| `PARSER_SPACES_SECRET_ACCESS_KEY` | Doc-cache Spaces secret | **Shared with Alpha — same value** |
| `PARSER_SPACES_KEY_PREFIX` | `parser/document-cache/production` | Fresh — new, per-environment |
| `VALKEY_URL` | Production Valkey connection string | **Shared with Alpha — same value, same instance** (step 10) |
| `PARSER_RESULTS_KEY_PREFIX` | `parser/parse-results/production` | Fresh — new, per-environment (step 10) |

**Quick reference — what's fresh for this repo vs. what's a straight copy
from Alpha's existing secrets:**
- **Fresh, generate/create specifically for this repo:** both
  `TF_VAR_ssh_public_key` values and both `DROPLET_SSH_PRIVATE_KEY` values —
  one keypair per environment (step 1) — both `PARSER_API_KEY` values
  (step 6), both `PARSER_HOSTNAME` values, both `DROPLET_HOST` values (filled
  in later), both `PARSER_SPACES_KEY_PREFIX` values, both
  `PARSER_RESULTS_KEY_PREFIX` values (step 10), the DO API token (step 4).
- **Shared, just copy the same value across from Alpha's existing secrets:**
  `SPACES_ACCESS_KEY_ID` / `SPACES_SECRET_ACCESS_KEY` (state buckets, per
  environment), all five `PARSER_SPACES_BUCKET` / `_REGION` /
  `_ENDPOINT_URL` / `_ACCESS_KEY_ID` / `_SECRET_ACCESS_KEY` (doc-cache
  bucket credentials), both `VALKEY_URL` values (step 10 — same Valkey
  instance Alpha already connects to, per environment).

Copy these values directly between GitHub's secret-entry UIs (or via a
password manager) — do not have Claude read either repo's raw secret values
or `.env` files to do this.

**Status: Pending.**

---

## 8. Set the repo-level secret `TF_VAR_do_token`

GitHub → this repo → **Settings → Secrets and variables → Actions → New
repository secret**. Name: `TF_VAR_do_token`. Value: the token minted in
step 4 (this repo's own token — not Alpha's).

**Status: Pending.**

---

## 9. Confirm DNS access for `services.simpero.com` / `services-staging.simpero.com`

Nothing to create yet — the A records only get added **after** each
environment's first successful `terraform apply`, once `droplet_ip` is known
from that apply's output (same sequencing as Alpha's own droplet rollout:
apply first, then DNS, then re-run the deploy job so Caddy can complete its
Let's Encrypt issuance against a hostname that actually resolves). The action
item right now is just:

1. Confirm you (or whoever handles it) still has access to whichever
   provider actually hosts `simpero.com`'s DNS (not DigitalOcean).
2. Note it down for later — the real record-creation step happens as part of
   Phase 5/6 of the deploy runbook (dispatch `deploy.yml` → apply → set
   `DROPLET_HOST` → add the A record → re-dispatch), not now.

**Status: Pending (access-confirmation only; record creation deferred to
first deploy).**

---

## 10. Add `VALKEY_URL` and `PARSER_RESULTS_KEY_PREFIX` to the `staging` and `production` Environments

`parser_service/worker.py` runs a SAQ worker that consumes a queue named
`"parse"` on the **SAME DigitalOcean Managed Valkey instance Alpha's app
already connects to** — not a new, separately provisioned instance. This is
just a different queue name within an instance that already exists, so there
is nothing to provision here, only two secrets to add.

**Only the gated `staging` and `production` Environments need this** — not
`staging-plan`/`production-plan`, which only ever run Terraform and never
touch Valkey.

**Steps (per environment):**
1. `VALKEY_URL` — open that environment's existing `VALKEY_URL` secret in
   Alpha's repo (Alpha → Settings → Environments → that environment →
   Environment secrets) and copy the **exact same value** into this repo's
   matching Environment. Do not request or provision a new Valkey instance —
   this is a straight copy, same instance, same credential.
2. `PARSER_RESULTS_KEY_PREFIX` — a fresh, per-environment value:
   `parser/parse-results/staging` for `staging`,
   `parser/parse-results/production` for `production`.

Copy `VALKEY_URL` directly between GitHub's secret-entry UIs (or via a
password manager) — do not have Claude read either repo's raw secret values
or `.env` files to do this, same rule as step 7.

**Status: Pending.**

---

## Status table

| # | Item | Status |
|---|---|---|
| 1 | Generate 2 new SSH deploy keypairs (staging + production) | **Done** (2026-07-29) — both keypairs created in `~/.ssh`, matching the naming convention |
| 2 | Confirm/reuse Spaces access keys for both TF-state buckets (readwrite+delete) | **Done** (2026-07-29) |
| 3 | Confirm object versioning enabled on both TF-state buckets | **Done** (2026-07-29) |
| 4 | Mint a new DO API token for this repo | **Done** — repo-level `TF_VAR_DO_TOKEN` secret set |
| 5 | Confirm DO Project names (`Simpero-Staging`/`Simpero-Prod` vs. actual console names) | **Done** (2026-07-29) |
| 6 | Generate 2 `PARSER_API_KEY` values | **Staging done** (2026-07-29). **Production intentionally left blank** — see the production-deferral note below. |
| 7 | Create 4 GitHub Environments + populate all secrets | **Done for `staging`** (2026-07-29, confirmed via secret-update timestamps — every staging secret now holds a real value, including `PARSER_SPACES_ACCESS_KEY_ID/SECRET_ACCESS_KEY`, `SPACES_ACCESS_KEY_ID/SECRET_ACCESS_KEY`, `VALKEY_URL`, `PARSER_SPACES_BUCKET/REGION/ENDPOINT_URL` — only `DROPLET_HOST` is still blank, which is expected until the first `terraform apply`). Environments themselves all exist with correct protection rules (gated pair has required reviewers — staging: `vanshkhanna17`+`kpal002`, production: `vanshkhanna17`; `-plan` pair ungated). **`production` intentionally still scaffolded-only** — real values only for `TF_VAR_ssh_public_key`/`DROPLET_SSH_PRIVATE_KEY`, everything else blank on purpose (see production-deferral note below). |
| 8 | Set repo-level `TF_VAR_do_token` secret | **Done** (see row 4) |
| 9 | Confirm DNS access (record creation deferred to first deploy) | Pending |
| 10 | Add `VALKEY_URL` (copied from Alpha) + `PARSER_RESULTS_KEY_PREFIX` (fresh) to `staging`/`production` | **Staging done** (2026-07-29, both secrets hold real values). **Production intentionally left blank** — see below. |

**Production deferral (2026-07-29, Vansh's call):** production's Spaces
bucket will be a **different bucket than staging's** — staging uses the
shared `simpero-cim-xlsx-upload` bucket (same one Alpha's doc cache uses),
but production's bucket has not been decided/created yet. Given that, Vansh
asked to defer **every production secret except the SSH keypair** until
production actually gets deployed, rather than guessing values now that
might be wrong. Concretely: `PARSER_API_KEY`, `PARSER_HOSTNAME`,
`PARSER_SPACES_KEY_PREFIX`, `PARSER_RESULTS_KEY_PREFIX` were set with real
values in `production` earlier in this session and have been **reverted to
blank** — only `TF_VAR_ssh_public_key` and `DROPLET_SSH_PRIVATE_KEY` should
hold real values in `production`/`production-plan` right now. **Staging is
unaffected by this — its values stand.** When production is actually ready
to deploy, revisit every row in this doc for production specifically (not
just re-copy staging's values), including confirming the production Spaces
bucket name/region/endpoint/credentials, which may differ from staging's
shared bucket entirely.

**Note on production sequencing:** unlike Alpha, this repo's production
droplet does **not** wait on any database cluster — this service opens no DB
connection. It does, however, now hold a Valkey connection: `worker.py`
connects to the **same Valkey instance Alpha's production uses**, under a
different queue name (`"parse"`), so it is not independent of Alpha's
Valkey status the way it's independent of Alpha's Postgres. Vansh has
confirmed staging and production point at **separate** Valkey instances, so
it's safe to enable the worker in both environments without cross-environment
job collisions. That said, per the production-deferral note above, staging
and production are no longer intended to be brought up back-to-back — staging
can proceed on its own, and production's remaining secrets (including its own
Spaces bucket) get filled in only when production deployment actually starts.

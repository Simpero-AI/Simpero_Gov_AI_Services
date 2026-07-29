# Plan: Deploy the parse service to DigitalOcean Droplets via GitHub Actions

Status: planned, not implemented. Approved topology (2026-07-29): dedicated
droplet per environment, mirroring `Simpero_AI_Gov_Alpha`'s pattern. A sidecar
alternative was proposed and overruled by Vansh — see §1.0 for what was
considered and why the droplet path won, since that reasoning constrains what
this plan is allowed to simplify later.

Companion document: `Simpero_AI_Gov_Alpha/docs/plans/do-droplet-deployment.md`.
That plan is the source of this one's shape. Where the two disagree, Alpha's
**implemented files** are authoritative over Alpha's **document** — the doc has
known stale sections (§4's job graph, specifically). See §6.

## 0. What this covers

Three manually-triggered workflows for this repo, covering **two independent
environments** — staging and production, each its own droplet, its own
Terraform state, its own hostname:

1. Reuse the existing lint/typecheck/test/contract/image/audit suite from
   `ci.yml` (via `workflow_call`, not duplicated).
2. Build the parse-service image and push to GHCR.
3. Optionally (input toggle) provision/update that environment's droplet —
   Terraform plan, then a gated apply — against the DigitalOcean Spaces remote
   state buckets **already shared with the backend and frontend repos**, under a
   new `services/` key prefix.
4. Deploy to that environment's droplet — SSH, write `.env`, pull, restart.
5. Tear down an environment (`destroy.yml`), same plan-then-gated-apply shape.

No Kubernetes, no load balancer, no blue/green. A brief container restart during
deploy is acceptable at this scale, for both environments.

**Explicitly not in scope:** wiring a caller. Nothing in `Simpero_AI_Gov_Alpha`
calls this service today, in either direction — see §1.0. Building that
integration is the AI engineer's territory and is provisional until Gate G1.

### What this service is, and what that removes from Alpha's plan

This service holds no tenant data, opens no database connection, and has no
Valkey/SAQ dependency. Everything DB-shaped in Alpha's plan is therefore deleted
here rather than adapted: no `pgbouncer`, no `alembic upgrade head` step, no
`digitalocean_database_firewall` resources, no `postgres_cluster_id` /
`valkey_cluster_id` variables, and none of Alpha's §5-item-8 "audit existing
trusted sources before every apply" operational burden. That hazard class does
not exist here.

`.github/workflows/ci.yml` is barely touched — it gains a `workflow_call`
trigger so `deploy.yml` can reuse its jobs. Its bare-runner property (CLAUDE.md:
"a job needing Postgres/secrets means the trust boundary was crossed by
mistake") is unchanged and must stay that way.

## 1. Architecture decisions

### 1.0 Topology: dedicated droplet per environment (Vansh's call, overruling the recommendation)

The recommendation put to Vansh was a **sidecar container in Alpha's existing
compose stack** — no droplet, no Terraform, no DNS, no certificate, no public
exposure, reachable only at `http://parser:8001` on an internal docker network.
The supporting facts, all verified against both codebases, are recorded here
because they remain true and shape the risks in §6:

- **Alpha's droplet already provisions this service's environment.** Alpha's
  `deploy.yml` writes `PARSER_SPACES_BUCKET/REGION/ENDPOINT_URL/ACCESS_KEY_ID/
  SECRET_ACCESS_KEY` into `/opt/simpero/.env` on every deploy, and
  `Alpha/.env.example:45` says out loud these are "read by ParserSettings, not
  the app." A slot was left for a parser process that was never placed in it.
- **There is no caller in either direction.** `Alpha/CLAUDE.md:160-173` states
  `app/jobs/parse_client.py` "currently enqueues jobs nothing will ever consume"
  and is "scaffolding for a decision that hasn't been made." This repo has no
  queue, no SAQ, no Valkey, no worker — `parser_service/main.py` exposes exactly
  `GET /health` and `POST /parse`. Nothing in Alpha's `app/api/` calls either.
  The design doc's claim that this service is "queue-based, not browser-facing"
  (Alpha §1, "Informational") is **out of date with both codebases.**
- **Gate G1 is still open**, so parse-pipeline decisions are provisional.

Vansh's decision, 2026-07-29: **dedicated droplet, both environments, this
pass.** Recorded as a deliberate trade — paying for host-level isolation of the
untrusted-bytes boundary and independent scaling ahead of need, rather than
container-level isolation on a shared host. The consequences that decision
brings with it (a public hostname, therefore authentication; a second host per
environment; an idle production droplet) are handled below rather than deferred.

### 1.1 Authentication — resolved, not deferred

`POST /parse` has no authentication of any kind today. It accepts arbitrary
bytes and runs a torch/opencv/docling pipeline on them. Putting that on
`services.simpero.com` unauthenticated is free compute for anyone who finds it,
and an unauthenticated attack surface against a large ML dependency tree fed
hostile input by design. **This blocks the first image push, not the first
deploy.**

**Decision: a shared-secret header checked in `parser_service/main.py`.**
`ParserSettings` gains `api_key: str | None = None` (`PARSER_API_KEY`).
`POST /parse` compares an `X-Parser-Key` header against it with
`hmac.compare_digest` (stdlib, constant-time). `GET /health` stays open — it
returns no data and both Caddy and the deploy health check need it.

Two alternatives were considered and rejected, with reasons, so they aren't
re-litigated later:

- **DO Cloud Firewall restricting 443 to Alpha's droplet.** Rejected on three
  counts. (a) It cannot be expressed in this repo's Terraform — Alpha's droplet
  ID/IP lives in a different state file, so it becomes a hand-copied value in
  `{env}.tfvars`. (b) It breaks silently on every Alpha droplet replacement, and
  Alpha's own §6 records that any `user_data` change forces replacement and a
  new IP — parse would sever with no signal from either repo's pipeline. (c)
  Port 80 must stay open to the world regardless for Let's Encrypt HTTP-01
  validation, so "locked down by firewall" is never fully true without moving
  Caddy to DNS-01 and handing it a DO API token — a bigger moving part than the
  thing it replaces.
- **Caddyfile `header` matcher → 401.** Genuinely fewer lines, and rejected
  anyway: the protection would exist only in this deployment topology, not in
  the artifact, so the service stays naked anywhere else it runs (a future
  sidecar, a local compose, a misconfigured port publish). Caddy's matcher also
  does an ordinary string comparison rather than a constant-time one. The
  service that calls itself the citation trust boundary should own its own front
  door.

### 1.2 Enforcement is unconditional and fail-closed

`/parse` requires a matching header on **every** request:

- `PARSER_API_KEY` unset → **503** (misconfigured, don't serve).
- Header absent or mismatched → **401**.

An "enforce only when the secret happens to be set" design means a droplet
deployed with the key missing from the `.env` heredoc is silently a public parse
farm. Alpha's §6 already records heredoc omission as a live, sharp failure mode
("the secret can exist in GitHub and still never reach the droplet"). Fail-open
is not acceptable at this boundary. **This is not a knob and not negotiable at
implementation time.**

Test cost is smaller than it looks. The three test modules each build
`TestClient(app)` **once at module level** — `tests/test_pdf_parser.py:22`,
`tests/test_docx_parser.py:187`, `tests/test_xlsx_parser.py:660` — so the ~15
`client.post("/parse", ...)` call sites need no edits at all. Three constructor
lines gain a default `headers=`, plus one `conftest.py` autouse fixture that
sets `PARSER_API_KEY` and clears `get_settings`'s `lru_cache`.

### 1.3 The deploy pipeline asserts the control is live

The `deploy` job's health check makes two calls, not one:

- `GET /health` → expect **200**.
- `POST /parse` with **no** key → expect **exactly 401**.

If the secret never reached the droplet, that second call returns 503 and the
deploy fails red. This turns §1.2's fail-closed property from an assertion into
a deploy-time gate. Assert the exact status code — "not 200" would let a 503
through and defeat the point.

**Rate limiting is deliberately skipped.** The key stops the internet; it does
not stop a leaked key hammering the endpoint. There is no caller yet, therefore
no traffic shape against which to size a limit. Add a Caddy `rate_limit` if
abuse is ever observed.

### 1.4 Non-root container user

This repo's `Dockerfile` currently runs as root; Alpha's does not
(`useradd -m -u 1000 appuser`). Add one here. `HF_HOME` and `TORCH_HOME` already
point at `/tmp/`, so no writable-path change is needed. Do not touch the
pip-not-uv install block or its comment — that comment documents a resolved
transformers/torch platform-pinning conflict and is deliberate.

### 1.5 `ci.yml` gains only a `workflow_call` trigger

Required, not stylistic: GitHub rejects a `uses:`-called workflow that lacks the
trigger. No job gains a service, a secret, or an Environment; no `secrets:
inherit` at the call site; no `inputs:`/`secrets:` block under the trigger. An
empty `workflow_call:` is the whole change, and CLAUDE.md's bare-runner property
stays literally true. **If any change makes a CI job need a credential or a
database, the boundary has been crossed by mistake — stop and escalate.**

### 1.6 Two environments, one Terraform config

Copied from Alpha's shape, with the DB resources deleted:

- `main.tf` parameterized by a validated `environment` variable, suffixing every
  DO resource name — `simpero-parser-${var.environment}`,
  `simpero-parser-deploy-${var.environment}`,
  `simpero-parser-firewall-${var.environment}`. Required, not cosmetic: DO
  resource names must stay unique account-wide, now across three repos' worth of
  droplets.
- `environment` flows in as `TF_VAR_environment` from `inputs.environment`, never
  duplicated into the `.tfvars` files.
- Separate `.tfvars` per environment carrying `region`, `droplet_size`,
  `do_project_name`.
- **Separate deploy SSH keypairs per environment, and new keypairs distinct from
  Alpha's.** DO dedupes `digitalocean_ssh_key` by key material, not name — two
  independent states registering the same public key risk a fingerprint
  collision on whichever apply runs second.
- `digitalocean_firewall` needs explicit **outbound** rules (tcp/udp/icmp
  allow-all). DO Cloud Firewalls deny-by-default in both directions; without
  them the droplet cannot reach GHCR, Let's Encrypt, Spaces, or DNS.
- Droplet assigned into the pre-existing DO Project by data-source lookup
  (`Simpero-Staging` / `Simpero-Prod`, matching Alpha's `.tfvars`). Only the
  droplet is assignable — DO Projects support a fixed list of nine resource
  types, and neither firewalls nor SSH keys are on it.

### 1.7 Remote state: shared buckets, new `services/` prefix

- Buckets: `simpero-tf-state-staging` and `simpero-tf-state-production`, region
  **`tor1`**, created manually outside any Terraform state. **Not**
  `simpero-cim-xlsx-upload` — that is a different bucket entirely
  (`PARSER_SPACES_BUCKET`, the doc cache). Conflating them would put Terraform
  state inside the confidential document cache.
- Key prefix: **`services/`**, alongside Alpha's `backend/` and the frontend's
  `frontend/`. So `services/staging.tfstate` and `services/production.tfstate`.
- Partial backend config resolved per environment at `terraform init` time via
  `-backend-config=backend-<environment>.hcl`. Environment-invariant flags
  (`skip_*`, `use_path_style`, `use_lockfile`) live in `versions.tf`.
- Credentials via `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` env at init
  time, never in files.
- **The Spaces key needs readwrite + delete on both buckets.** `use_lockfile`
  creates *and removes* a lock object during `plan`, so a read-only or
  no-delete grant fails at plan time. Bucket grants are fixed at key creation —
  if the existing shared key lacks delete, a new key is required.
- **Inherited accepted risk, now spanning three repos:** DO Spaces keys scope to
  an entire bucket, not a prefix, and DO bucket policies are incompatible with
  limited-access keys. Any credential touching these buckets can read, write and
  delete all three repos' state. Object versioning is the sole recovery
  mechanism and remains unconfirmed per Alpha's §5 item 3.

### 1.8 Droplet-level setup

- Image: DO's "Docker on Ubuntu" marketplace image, slug `docker-20-04`.
- Size: **`s-2vcpu-4gb`, both environments.** Docling's layout models will not
  run in 1GB. Production gets the same size as staging — it has no load profile
  yet, and it is a one-line `.tfvars` change when it does.
- SSH open to all source IPs, key-only auth — matching Alpha's posture
  (confirmed 2026-07-29).
- Reserved IPs deferred this pass (confirmed 2026-07-29). Consequence: droplet
  recreation means redoing DNS and `DROPLET_HOST` by hand, same as Alpha.
- **`cloud-init.yaml.tpl` is copied from Alpha verbatim**, with the swapfile
  raised from 1G to 2G (this image loads torch models; the droplet is bigger but
  so is the peak). Every line in that file is a scar from Alpha's §8 and must be
  reproduced exactly:
  - **Pure ASCII, comments included.** An em dash (`E2 80 94`) in a comment cost
    two days of misdiagnosis; the cloud-init parser choked on the middle byte.
  - `chown deploy:deploy /opt/simpero` immediately after `mkdir` — runcmd runs
    as root and SCP runs as `deploy`.
  - `ufw allow 22/80/443` **before** any `ufw delete`, so a failure in cleanup
    leaves the droplet over-permissive and reachable rather than locked out.
  - `ufw --force delete limit 22/tcp` — the marketplace image rate-limits SSH to
    ~6 connections/30s, which silently drops `drone-scp`'s multiple rapid
    connections as an "i/o timeout."
  - Close Docker API ports 2375/2376 in UFW as defense-in-depth.
  - Trailing `ufw status verbose` so the resulting rule state lands in
    `/var/log/cloud-init-output.log` on every boot.
  - `ssh_pwauth: false` last, after the user exists.
- Any edit to `cloud-init.yaml.tpl` forces droplet replacement — new IP, stale
  `DROPLET_HOST`, stale DNS. `.env` is not a casualty (it is regenerated from
  GitHub secrets on every deploy).

### 1.9 Caddy and compose

- **`Caddyfile` is a single shared file**, using Caddy's parse-time env
  interpolation in the site-address position (`{$VAR}`, not the runtime
  `{env.VAR}` form which does not work there):
  ```
  {$PARSER_HOSTNAME} {
      reverse_proxy parser:8001
  }
  ```
  `PARSER_HOSTNAME` comes from each environment's `.env` —
  `services.simpero.com` (production) / `services-staging.simpero.com`
  (staging), both confirmed 2026-07-29.
- **`docker-compose.prod.yml` has two services: `parser` and `caddy`.** No
  pgbouncer, no worker, no migration step. `parser` publishes **no host ports**
  — only Caddy publishes 80/443. Network named `simpero-parser-network`, which
  must not collide with Alpha's `simpero-network`.
- Image owner hardcoded lowercase (`ghcr.io/simpero-ai/simpero-gov-ai-services`)
  rather than dynamic, matching Alpha's deliberate post-org-transfer choice: a
  real owner change needs a look at both this file and `deploy.yml`'s image
  computation, not a silent auto-follow.
- **No repo checkout on the droplet.** Only `docker-compose.prod.yml`,
  `Caddyfile` and `.env` live in `/opt/simpero/`.
- `.env` is generated by the `deploy` job from Environment secrets on every run,
  via a quoted heredoc, `chmod 600`. The GitHub secrets are the single source of
  truth; the file on disk is disposable.

### 1.10 Per-environment doc-cache prefix (new finding)

`parser_service/config.py:49` defaults `spaces_key_prefix` to
`parser/document-cache`, and Alpha's `deploy.yml` heredoc never overrides it.
With both environments pointing at the same bucket (`simpero-cim-xlsx-upload`),
**staging and production would share one content-addressed cache namespace** — a
staging parse, possibly running an unreviewed G1 bake-off variant, writes a
`sha256`-keyed entry that production later reads as authoritative.
`config.py`'s own comment already flags that these keys are shared across
tenants; sharing them across environments too is a strictly worse version of the
same problem, and it is the one place in this plan where a defect could reach
customer-facing output.

**Fix: `PARSER_SPACES_KEY_PREFIX` becomes a per-environment secret** —
`parser/document-cache/staging` and `parser/document-cache/production`. One env
var, zero code change. Alpha's `deploy.yml` should get the same treatment
eventually; cross-repo, not blocking this work.

### 1.11 No separate `publish.yml`

`deploy.yml`'s `docker-publish` job already builds and pushes, as it does in
Alpha. A second workflow publishing the same image puts the image name and the
GHCR login in two files that must be kept in sync — the exact drift class that
produced Alpha's `digitallick`/`simpero-ai` incident (§8). A published tag that
no deploy pulls is not a deliverable either. If a build-without-deploy button is
wanted, it becomes a `publish_only` boolean input on `deploy.yml`, not a second
file. **Pending Vansh's confirmation — see §9.**

### 1.12 Production proceeds independently of Alpha's deferred production DB

Confirmed true, and checked for reasons it might not be: this service opens no
DB connection, has no Valkey/SAQ dependency, imports nothing from Alpha, and its
only external dependency is the optional Spaces doc cache, which self-disables
when unset (`document_cache.build_document_cache`). Nothing about it needs
Alpha's production Postgres to exist.

Two honest caveats, neither blocking: production runs with **zero callers**
until Alpha's production exists (~$24/mo for an idle host, and no traffic
against which to notice a problem), and its first real traffic will therefore
also be its first real test.

### 1.13 GHCR authentication

Built-in `GITHUB_TOKEN` on both sides — `packages: write` on `docker-publish`,
`packages: read` on `deploy` for the droplet's pull. No PAT, no bot account, no
GitHub App: **GHCR does not accept GitHub App installation tokens** (confirmed
platform limitation — `docker login` succeeds and `docker pull` is denied). No
new secret to create or rotate. `github.repository_owner` is `Simpero-AI` and
GHCR rejects uppercase, so the image name must be explicitly lowercased in the
workflow, not just interpolated.

## 2. File layout

```
parser_service/
  config.py               # + api_key: str | None (PARSER_API_KEY)
  main.py                 # + X-Parser-Key check on POST /parse (401/503)

tests/
  conftest.py             # NEW - autouse fixture: set PARSER_API_KEY,
                          #       get_settings.cache_clear()
  test_auth.py            # NEW - 401 no header / 401 wrong header /
                          #       200 correct header / 503 unset
  test_pdf_parser.py:22   # TestClient(app) gains default headers=
  test_docx_parser.py:187 # same
  test_xlsx_parser.py:660 # same

Dockerfile                # + non-root user, USER before CMD

terraform/
  versions.tf             # required_version >= 1.11.0; digitalocean ~> 2.0;
                          # partial backend "s3": skip_credentials_validation,
                          # skip_region_validation, skip_requesting_account_id,
                          # skip_metadata_api_check, skip_s3_checksum,
                          # use_path_style, use_lockfile
  variables.tf             # environment (validated), do_token (sensitive),
                          # ssh_public_key, region (tor1), droplet_size
                          # (s-2vcpu-4gb), do_project_name.
                          # NO postgres_cluster_id / valkey_cluster_id
  main.tf                  # ssh_key + droplet + firewall (in 22/80/443,
                          # out tcp/udp/icmp) + project data source +
                          # project_resources. NO database_firewall
  outputs.tf               # droplet_ip
  cloud-init.yaml.tpl      # verbatim from Alpha, swap 2G, PURE ASCII
  backend-staging.hcl      # simpero-tf-state-staging, services/staging.tfstate,
                          # tor1, endpoints.s3
  backend-production.hcl   # simpero-tf-state-production,
                          # services/production.tfstate, tor1, endpoints.s3
  staging.tfvars           # tor1, s-2vcpu-4gb, do_project_name "Simpero-Staging"
  production.tfvars        # tor1, s-2vcpu-4gb, do_project_name "Simpero-Prod"

docker-compose.prod.yml   # NEW - parser + caddy only
Caddyfile                 # NEW - {$PARSER_HOSTNAME} -> reverse_proxy parser:8001
.env.example               # NEW - warranted now that there is required config

.github/workflows/
  ci.yml                  # + workflow_call: in on:. Nothing else.
  deploy.yml               # NEW
  destroy.yml               # NEW
```

`.env.example` is new because `PARSER_API_KEY` is the first genuinely required
config this service has ever had. Contents: `PARSER_API_KEY`,
`PARSER_HOSTNAME`, the five `PARSER_SPACES_*`, `PARSER_SPACES_KEY_PREFIX`, and a
comment noting `IMAGE_TAG` is supplied by the deploy job rather than this file.

## 3. GitHub repo configuration required

**Four GitHub Environments.** The plan/gated split is not tidiness: any job
referencing a gated Environment inherits its approval gate, and `terraform-plan`
needs the Spaces credential while staying reviewable *before* approval — which
is the entire point of a plan step. Do not collapse this to two.

| Environment | Gated (required reviewer) | Secrets |
|---|---|---|
| `staging-plan` | No | `TF_VAR_ssh_public_key`, `SPACES_ACCESS_KEY_ID`, `SPACES_SECRET_ACCESS_KEY` |
| `production-plan` | No | same three, production values |
| `staging` | Yes (Vansh) | the three above, **plus** `DROPLET_HOST`, `DROPLET_SSH_PRIVATE_KEY`, `PARSER_API_KEY`, `PARSER_HOSTNAME`, `PARSER_SPACES_BUCKET`, `PARSER_SPACES_REGION`, `PARSER_SPACES_ENDPOINT_URL`, `PARSER_SPACES_ACCESS_KEY_ID`, `PARSER_SPACES_SECRET_ACCESS_KEY`, `PARSER_SPACES_KEY_PREFIX` |
| `production` | Yes (Vansh) | same shape, production values |

**Repo-level secret:**

| Secret | Holds |
|---|---|
| `TF_VAR_do_token` | A **new DO API token minted for this repo** (confirmed 2026-07-29) — deliberately not Alpha's token value. Same DO account, therefore the same power, but independent revocation: a leak here doesn't force rotating Alpha's and the frontend's pipelines simultaneously. If DO's granular token scopes are available on the account, scope it to droplet / firewall / ssh_key / project read-write. |

No GHCR credential secret (§1.13). `DROPLET_SSH_USER` is not a secret — the
literal `deploy`, hardcoded in the workflow.

**Shared with the other repos (same values, nothing new to provision):**
`SPACES_ACCESS_KEY_ID` / `SPACES_SECRET_ACCESS_KEY` (state buckets),
`PARSER_SPACES_*` credentials and bucket name (doc cache), GHCR auth.

**New and specific to this repo:** the DO API token, both SSH keypairs, both
`PARSER_API_KEY` values, both `PARSER_HOSTNAME` values, both `DROPLET_HOST`
values, and both `PARSER_SPACES_KEY_PREFIX` values.

**Two approval prompts per run when `run_terraform` is true** (`terraform-apply`,
then `deploy`) — GitHub evaluates Environment protection per job, not
deduplicated per Environment per run. One prompt when it's false.

## 4. Workflow behavior

### `deploy.yml`

Inputs: `environment` (choice `staging` | `production`, default `staging` —
biases a forgotten selection toward lower blast radius), `run_terraform`
(boolean, default `false`). `concurrency: group: infra-${{ inputs.environment }}`
— shared name with `destroy.yml`, deliberately not keyed on `github.workflow`
(which differs per file), so a deploy and a destroy against the same environment
can never race. No `cancel-in-progress`: a half-applied Terraform run should
finish, not be killed.

```
job ci:
  uses: ./.github/workflows/ci.yml

job docker-publish (needs: ci; permissions: packages: write, contents: read):
  checkout @ github.sha
  lowercase the owner, then build + push
    ghcr.io/simpero-ai/simpero-gov-ai-services:${{ github.sha }} and :latest
  cache-from/cache-to: type=gha,mode=max   # docling pulls PyTorch

job terraform-plan (needs: docker-publish; if: inputs.run_terraform;
                    environment: { name: '${{ inputs.environment }}-plan' }):
  terraform init -backend-config=backend-${{ inputs.environment }}.hcl
  terraform plan -var-file=${{ inputs.environment }}.tfvars -out=tfplan
  upload tfplan artifact (name includes environment)

job terraform-apply (needs: terraform-plan; if: inputs.run_terraform;
                     environment: { name: '${{ inputs.environment }}' }):
  download tfplan, re-init, terraform apply tfplan
  # the exact downloaded plan, never a fresh one

job deploy (needs: [docker-publish, terraform-apply];
            environment: { name: '${{ inputs.environment }}' };
            permissions: packages: read):
  if: always() && needs.docker-publish.result == 'success'
      && (inputs.run_terraform == false
          || needs.terraform-apply.result == 'success')
  scp docker-compose.prod.yml + Caddyfile -> /opt/simpero
  ssh: write .env from Environment secrets (quoted heredoc, chmod 600)
       keys: PARSER_API_KEY, PARSER_HOSTNAME, PARSER_SPACES_BUCKET,
             PARSER_SPACES_REGION, PARSER_SPACES_ENDPOINT_URL,
             PARSER_SPACES_ACCESS_KEY_ID, PARSER_SPACES_SECRET_ACCESS_KEY,
             PARSER_SPACES_KEY_PREFIX
  ssh: docker login ghcr.io (GITHUB_TOKEN)
       IMAGE_TAG=${{ github.sha }} docker compose -f docker-compose.prod.yml pull
       IMAGE_TAG=${{ github.sha }} docker compose -f docker-compose.prod.yml up -d
       # NO migration step - this service has no database
  health check from the runner, --resolve pinning PARSER_HOSTNAME to
  DROPLET_HOST on both 443 and 80:
       GET  /health          -> expect 200   (bare path, no /api prefix here)
       POST /parse (no key)  -> expect exactly 401  (see §1.3)
```

Three details that are easy to get wrong and are load-bearing:

- **`always()` in the `deploy` `if:`** — without it GitHub implicitly ANDs in
  `success()`, blocking the job whenever `terraform-apply` is skipped (the
  routine `run_terraform: false` case). Check `inputs.run_terraform` **directly**
  rather than tolerating any `skipped` result: `terraform-apply` shows `skipped`
  both when `run_terraform` was false (fine) and when `terraform-plan` actually
  failed (not fine), and those look identical from here.
- **Dynamic `environment:` needs the object form** — `environment:\n  name: ${{
  ... }}`. The string shorthand does not reliably resolve a dynamic value in
  that position.
- **`--resolve` on the health check** — Caddy matches purely on Host header /
  TLS SNI, so a request to the bare `DROPLET_HOST` IP gets a 404 regardless of
  DNS. `--resolve` pins the hostname to the IP for that one call. It still needs
  DNS to have been live at some point, because Let's Encrypt's own validation
  request travels the real internet and is unaffected by `--resolve` — so the
  first attempt after a fresh droplet fails with a TLS/cert error, which is the
  *correct* failure, not a misconfiguration.

**Where Alpha's document and Alpha's implementation disagree, copy the
implementation.** Specifically: the doc's §4 says `terraform-plan` has
`needs: ci` and gives an older form of the `deploy` `if:` condition; the actual
`Alpha/.github/workflows/deploy.yml` (lines 86 and 184) is correct and is what
this plan mirrors.

### `destroy.yml`

Mirrors the plan-then-gated-apply pattern exactly. Inputs: `environment`
(choice) and `confirm_environment` (string, required). Same
`infra-${{ inputs.environment }}` concurrency group. No new secrets or
Environments.

```
job terraform-destroy-plan (environment: '${{ inputs.environment }}-plan'):
  FIRST STEP, before checkout (working-directory: .):
    if confirm_environment != environment -> ::error:: and exit 1
  checkout, init, terraform plan -destroy -out=tfplan, upload artifact

job terraform-destroy-apply (needs: above;
                             environment: '${{ inputs.environment }}'):
  download, re-init, terraform apply tfplan
  ::warning:: DROPLET_HOST and the DNS record for this environment are now
              stale - update both after the next successful apply
```

The confirmation check must run **before checkout** and must fail loudly (red
job) rather than skip — a skipped, easy-to-miss job is the wrong failure mode
for a typo on something this destructive. It needs `working-directory: .`
because the job-level `terraform` default doesn't exist on disk yet.

Nothing in this pipeline holds the repo-admin credentials needed to auto-clear a
GitHub secret after a destroy, deliberately — same narrow-credential posture as
everywhere else. Clearing `DROPLET_HOST` and DNS is a manual follow-up.

## 5. Dependencies from Vansh's side

> Blocks Phase 5 (first real apply). Phases 1–4 can be implemented and reviewed
> without any of it.

1. **Generate two new SSH keypairs** — `simpero-services-deploy-staging` and
   `simpero-services-deploy-production`. Must be new, not Alpha's (§1.6).
2. **Confirm the shared Spaces key has readwrite + delete on both state
   buckets.** Delete is required for `use_lockfile` (§1.7). Bucket grants are
   fixed at key creation, so a new key may be needed.
3. **Confirm object versioning is enabled on both state buckets** before either
   environment's first real apply. Sole recovery mechanism for the shared-bucket
   risk; still unconfirmed per Alpha's §5 item 3. Treat as a go/no-go gate.
4. **Mint the new DO API token** for this repo and set `TF_VAR_do_token` at repo
   level (§3).
5. **Create the 4 GitHub Environments** and populate every secret in §3's table,
   with Vansh as required reviewer on `staging` and `production`.
6. **Generate two `PARSER_API_KEY` values** (e.g. `openssl rand -hex 32`), one
   per environment.
7. **Confirm the DO Project names** — `Simpero-Staging` and `Simpero-Prod`,
   case-sensitive, taken from Alpha's `.tfvars`. Looked up by data source, so a
   mismatch fails at plan time with a clear error.
8. **DNS access** for `services.simpero.com` and `services-staging.simpero.com`.
   A records get added after each environment's first successful apply, once
   `droplet_ip` is known.

## 6. Known risks

- **`PARSER_API_KEY` is now the entire access control.** No IP restriction, no
  rate limit, no automated rotation. A leaked key is unauthenticated public
  compute until someone edits the secret and redeploys. Accepted deliberately;
  the alternatives are argued in §1.1. Rotation is cheap but nothing prompts it.
- **Long parses vs. proxy timeouts, untested.** Caddy's `reverse_proxy` has no
  default response timeout, so a multi-minute docling parse should survive — but
  this has never been exercised over a real proxy. Watch the first large-PDF
  production parse rather than assuming.
- **Idle production droplet** (§1.12) — ~$24/mo with no caller, and its first
  real traffic is also its first real test.
- **Third repo on the shared state buckets** (§1.7) — blast radius grows again;
  versioning is the only recovery path and remains unconfirmed.
- **`.env` heredoc drift, now with a security consequence.** A new required var
  that never gets added to `deploy.yml`'s heredoc silently never reaches the
  droplet. §1.3's 401 assertion catches this specifically for `PARSER_API_KEY`;
  nothing catches it for the others.
- **Droplet replacement breaks DNS and `DROPLET_HOST`**, per environment. Any
  `cloud-init.yaml.tpl` edit forces it. Reserved IPs, which would decouple this,
  are deferred this pass.
- **Every cloud-init hazard from Alpha's §8 is inherited** — non-ASCII bytes,
  missing `chown`, UFW ordering, SCP rate-limiting. §1.8 says copy verbatim
  precisely because each line is already paid for.
- **DO API token is account-level power** even as a separate token, reachable by
  anyone who can both dispatch with `run_terraform: true` and clear the gated
  Environment. Document who has both.
- **Doc/implementation drift in Alpha** — copy Alpha's *files*, not Alpha's
  *doc*, for the `deploy` job's `if:` and `terraform-plan`'s `needs:` (§4).
- **Cross-boundary: there is still no caller.** Wiring one — synchronous
  `POST /parse` vs. reviving the `parse_client.py` queue model — is the AI
  engineer's territory and provisional until **G1**. Whoever writes it must send
  `X-Parser-Key`; that is now a hard contract, and it needs `PARSER_SERVICE_URL`
  and `PARSER_API_KEY` added to Alpha's Environment secrets and `.env` heredoc.
  Not in this plan's scope.
- **G1 generally:** this builds infrastructure around a parse pipeline whose
  internals may still change. The infrastructure is deliberately shaped to be
  indifferent — one stateless container behind Caddy, no DB, no migrations — so
  a G1 outcome should change the image, not the topology.

## 7. Implementation phases

### Phase 0 — Vansh's prerequisites
Everything in §5. Blocks Phase 5 only.

### Phase 1 — service changes (must land before any image is published)
1. `parser_service/config.py` — add `api_key`.
2. `parser_service/main.py` — `X-Parser-Key` check per §1.1/§1.2.
3. `tests/conftest.py` — new autouse fixture.
4. Three `TestClient(app)` constructor lines gain default `headers=`.
5. `tests/test_auth.py` — new.
6. `Dockerfile` — non-root user.
7. `.github/workflows/ci.yml` — add `workflow_call:`.

Order matters: auth (1–5) precedes the image work so the first image ever pushed
to GHCR is already protected. `ci.yml` must precede Phase 4.

### Phase 2 — deploy artifacts (repo root)
8. `docker-compose.prod.yml`
9. `Caddyfile`
10. `.env.example`

### Phase 3 — `terraform/`
11–19. `versions.tf`, `variables.tf`, `main.tf`, `outputs.tf`,
`cloud-init.yaml.tpl`, `backend-staging.hcl`, `backend-production.hcl`,
`staging.tfvars`, `production.tfvars` — per §2's layout.

### Phase 4 — workflows
20. `.github/workflows/deploy.yml`
21. `.github/workflows/destroy.yml`

### Phase 5 — staging bring-up
22. Dispatch `deploy.yml` (staging, `run_terraform: true`). Approve
    `terraform-apply`.
23. Read `droplet_ip` from the apply output; set `DROPLET_HOST` in the `staging`
    Environment.
24. Add the A record for `services-staging.simpero.com`.
25. Re-dispatch with `run_terraform: false`. Expect the health check to fail on
    the first attempt until Caddy's first Let's Encrypt issuance completes —
    a TLS/cert error is the correct failure here.
26. Verify: `/health` 200; `/parse` without key 401; `/parse` with key returns a
    parse. Check `cloud-init status --long` is clean and that
    `/var/log/cloud-init-output.log`'s trailing `ufw status verbose` shows
    22/80/443 allowed.

### Phase 6 — production
27. Repeat 22–26 with `environment: production`, `services.simpero.com`, and the
    production keypair and secrets. No dependency on Alpha's production DB
    (§1.12).

## 8. Handoff instructions for the implementer

Work Phases 1 → 4 in the numbered order. Do not start Phase 5 until Vansh
confirms Phase 0 is complete. **Never run `terraform apply` locally** — the
pipeline is the only path to state.

Non-negotiables, not open to re-decision at implementation time:

1. **Auth is unconditional and fail-closed** (§1.2). Unset key → 503;
   mismatched or absent header → 401; `hmac.compare_digest` for the comparison;
   `/health` stays open. If the tests get inconvenient, fix the fixture, not the
   enforcement.
2. **`ci.yml` gets `workflow_call:` and nothing else** (§1.5). If any change
   makes a CI job need Postgres, Valkey, or a credential, stop and escalate —
   that is the trust boundary CLAUDE.md protects, and crossing it is a design
   error, not a config detail.
3. **`cloud-init.yaml.tpl` must be pure ASCII**, comments included. Verify with
   a byte-level scan before committing.
4. **`terraform-plan` and `terraform-destroy-plan` stay ungated** on the `-plan`
   Environments; `terraform-apply`, `terraform-destroy-apply` and `deploy` stay
   gated. Apply the **exact downloaded plan artifact**, never re-plan in an
   apply job.
5. **No migration step, no database resources, no shared `.env` with Alpha.**
   This service has no database and must not grow a dependency on one.
6. **Image name `ghcr.io/simpero-ai/simpero-gov-ai-services`**, computed with an
   explicit lowercase step in the workflow and hardcoded lowercase in
   `docker-compose.prod.yml`. One place each.
7. **Copy disputed logic from Alpha's files, not Alpha's doc** (§4).
8. Do not touch the `Dockerfile`'s pip-not-uv block or its comment; do not add
   `pypdfium2`/`pillow` to the image (dev-group only, deliberately).

Validation before handing back: `uv run pytest tests/ -q` green including the
new `test_auth.py`; `uv run ruff check .` and `uv run pyright` clean;
`terraform validate` passes against both `-var-file`s; a `deploy.yml` dispatch
with `run_terraform: false` reaches the `deploy` job's approval gate without
error.

## 9. Still open

- **Confirm §1.11** — no separate `publish.yml`; `deploy.yml`'s `docker-publish`
  covers it, as it does in Alpha. Say so if a build-without-deploy button is
  wanted and it becomes a `publish_only` input instead.
- **Confirm §1.10** — per-environment `PARSER_SPACES_KEY_PREFIX`, one extra
  secret per environment, closing the staging→production doc-cache bleed.
- Rate limiting on `/parse` — deliberately skipped (§1.3); revisit only if abuse
  is observed.
- Reserved IPs — deferred this pass (§1.8); revisit if droplet recreation
  becomes routine.
- Org-scoping the doc-cache key prefix per tenant — pre-existing follow-up noted
  in `config.py`, unchanged by this plan.

## 10. Settled decisions log

| Date | Decision |
|---|---|
| 2026-07-29 | Topology: dedicated droplet per environment, overruling the sidecar recommendation (§1.0) |
| 2026-07-29 | Image name: `ghcr.io/simpero-ai/simpero-gov-ai-services` |
| 2026-07-29 | Droplet size: `s-2vcpu-4gb`, both environments |
| 2026-07-29 | Scope: both staging and production this pass |
| 2026-07-29 | Hostnames: `services.simpero.com` / `services-staging.simpero.com` |
| 2026-07-29 | DO API token: mint a new one for this repo, not Alpha's |
| 2026-07-29 | SSH: open to all IPs, key-only auth |
| 2026-07-29 | Reserved IPs: deferred |
</content>

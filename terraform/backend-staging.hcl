# Partial S3-backend config for the staging environment — passed to
# `terraform init -backend-config=terraform/backend-staging.hcl`.
# Environment-invariant settings (skip_*, use_path_style, use_lockfile) live
# in versions.tf instead. See docs/plans/parser-droplet-deployment.md §1.7.
#
# This bucket is shared with the backend repo (Simpero_AI_Gov_Alpha) and the
# frontend repo (Simpero_AI_Gov_Web) — this repo's state lives under the
# `services/` prefix, alongside their `backend/` and `frontend/` prefixes.
# Neither bucket is created/imported by this repo's Terraform; both are
# created manually by Vansh.
bucket = "simpero-tf-state-staging"
key    = "services/staging.tfstate"

# Confirmed real Spaces region for the shared state buckets (matches the
# droplet region default, but was confirmed independently — see plan §5).
region = "tor1"
endpoints = {
  s3 = "https://tor1.digitaloceanspaces.com"
}

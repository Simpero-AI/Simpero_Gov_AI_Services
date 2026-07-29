terraform {
  # >= 1.11.0: needed for the `endpoints` block syntax below and for
  # `use_lockfile` (native S3-backend locking, GA in 1.11).
  required_version = ">= 1.11.0"

  required_providers {
    digitalocean = {
      source  = "digitalocean/digitalocean"
      version = "~> 2.0"
    }
  }

  # Remote state on DigitalOcean Spaces (S3-compatible) — a GitHub Actions
  # runner is ephemeral and can't hold local state between runs. The buckets
  # themselves are shared with the backend and frontend repos, created
  # manually, once, outside any Terraform state (see
  # docs/plans/parser-droplet-deployment.md §1.7).
  #
  # This is a *partial* backend config — only the settings that don't vary by
  # environment live here. bucket/key/region/endpoints are supplied per
  # environment at `terraform init` time via
  # `-backend-config=terraform/backend-<environment>.hcl`. Access keys
  # come from AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY env vars at init time —
  # never here.
  backend "s3" {
    skip_credentials_validation = true
    skip_region_validation      = true
    skip_requesting_account_id  = true
    skip_metadata_api_check     = true
    skip_s3_checksum            = true
    use_path_style              = true
    # Native S3-backend state locking (GA since 1.11) — three repos now share
    # these buckets, so real locking closes a gap the `concurrency:` group in
    # deploy.yml can't (it doesn't protect a local emergency `terraform` run).
    use_lockfile = true
  }
}

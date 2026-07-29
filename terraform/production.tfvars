# Per-environment values for production. do_token, ssh_public_key, and
# environment are NOT here — they come from TF_VAR_* env vars in the
# workflow (secrets + inputs.environment), one source of truth. See
# docs/plans/parser-droplet-deployment.md §1.6.
region       = "tor1"
droplet_size = "s-2vcpu-4gb"

# Must exactly match the DigitalOcean Project name (case-sensitive) —
# confirm this against whatever you actually named it in the DO console.
do_project_name = "Simpero-Prod"

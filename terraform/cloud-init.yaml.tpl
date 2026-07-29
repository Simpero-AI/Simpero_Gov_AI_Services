#cloud-config
# Rendered via Terraform's templatefile() (main.tf) so the deploy keypair's
# public half can be injected - cloud-init can't reference a TF variable
# directly. Ordering matters: the deploy user + its authorized_keys must be
# confirmed written before password auth is disabled (last step below), so a
# partial run never locks out SSH entirely (digitalocean_droplet.ssh_keys is
# the redundant fallback login path for that same reason).
users:
  - name: deploy
    sudo: ALL=(ALL) NOPASSWD:ALL
    shell: /bin/bash
    groups: docker
    ssh_authorized_keys:
      - ${ssh_public_key}

runcmd:
  # runcmd runs as root, so /opt/simpero is created root-owned by default -
  # the deploy user (via SCP in deploy.yml) needs write access to it.
  - mkdir -p /opt/simpero
  - chown deploy:deploy /opt/simpero
  # 2G swap: this droplet loads docling/torch layout models, and a brief
  # memory spike (Docker pull + container restart, all during an active
  # SSH/SCP session in deploy.yml) can trigger an instant OOM kill instead
  # of being absorbed. Persisted via fstab so it's still active after a
  # reboot, not just this first boot.
  - fallocate -l 2G /swapfile
  - chmod 600 /swapfile
  - mkswap /swapfile
  - swapon /swapfile
  - echo '/swapfile none swap sw 0 0' >> /etc/fstab
  # Explicit allow rules for everything the app needs, added BEFORE any
  # deletes below. Ordering matters: if a delete step has any issue
  # (observed once - a delete+allow sequence for port 22 left UFW active
  # with zero rules and its default-deny-incoming policy, blocking
  # everything including 80/443), this way the droplet ends up
  # over-permissive and reachable, not under-permissive and locked out.
  # 80/443 were never explicitly opened before this fix - the marketplace
  # image's default UFW config only ever had 22/tcp (rate-limited) and
  # the Docker API ports, so Caddy's ports were never actually reachable
  # at the UFW layer on any droplet, independent of anything else fixed
  # elsewhere (DNS, Host-header matching, etc.).
  - ufw allow 22/tcp
  - ufw allow 80/tcp
  - ufw allow 443/tcp
  # Now clean up what's not needed. The marketplace image's default UFW
  # config rate-limits port 22 (max ~6 connections per 30s per source
  # IP) - fine for interactive SSH, but deploy.yml's SCP/SSH steps open
  # several connections in quick succession (create folder, untar,
  # cleanup), and the later ones get silently dropped once the limit
  # trips, producing an "i/o timeout" unrelated to auth or the deploy
  # pipeline itself. Real access control already happens one layer out,
  # at the DO cloud firewall (digitalocean_firewall.parser in main.tf) - a
  # local rate limit on top of that isn't adding meaningful protection,
  # just breaking our own deploy tool. --force skips the interactive
  # confirmation prompt, needed since this runs unattended.
  - ufw --force delete limit 22/tcp
  # Also close the Docker remote API ports the marketplace image leaves
  # open in UFW by default. Not currently reachable in practice - the DO
  # cloud firewall only allows 22/80/443 inbound, sitting in front of UFW -
  # but closing them locally too is cheap defense-in-depth against an
  # unauthenticated Docker API (root-equivalent remote code execution)
  # ever being exposed by a future firewall misconfiguration.
  - ufw --force delete allow 2375/tcp
  - ufw --force delete allow 2376/tcp
  - ufw status verbose

# Last: disable SSH password auth via cloud-init's native directive, not a
# hand-rolled sshd_config sed.
ssh_pwauth: false

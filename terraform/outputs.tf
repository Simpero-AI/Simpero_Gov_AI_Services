output "droplet_ip" {
  description = "Droplet public IPv4 address. Use to set the DROPLET_HOST GitHub secret after apply."
  value       = digitalocean_droplet.parser.ipv4_address
}

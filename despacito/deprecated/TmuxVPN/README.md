# TmuxVPN

Deprecated VPN management combining openvpn, ssh, and tmux. Encryption disabled.

## Installation

```bash
apt-get update
apt-get install openvpn tmux
```

Edit `ip.txt` with server addresses. SSH key authentication required.

## Architecture

**Client:** Multiple VPN server connections
**Server:** OpenVPN only, no tmux required

## Security

All encryption and authentication features disabled. Use at own risk.

## Status

DEPRECATED. Do not use in production.

# Security Policy

## Supported versions

Only the latest `main` branch is maintained.

## Reporting

Please report a suspected vulnerability privately through the repository owner's GitHub security advisory page. Do not include board credentials, SSH keys, camera frames, local network addresses or proprietary model files in a public issue.

## Deployment defaults

The web UI binds to `127.0.0.1` by default and is intended to be reached through SSH port forwarding. It has no authentication or TLS. Do not bind it to `0.0.0.0` on an untrusted network without placing an authenticated reverse proxy or equivalent access control in front of it.

The HTTP interface changes only in-memory target and SAM state, but it exposes camera imagery through MJPEG and snapshot endpoints. Treat access to the port as access to the camera feed.

# Security

Report credentials or security issues privately to **dlutzzb@gmail.com**. Include affected paths, the revision, and a minimal description; do not post API keys or other sensitive material in public issues.

The inference transport is intended for trusted research networks. It uses Python pickle for both observations and responses; deserializing data from an untrusted peer can execute code. Bind model servers and the per-arm proxy to loopback, use SSH tunnels for remote machines, and connect only to servers you control. The supplied `example.pkl` is a protocol fixture; do not substitute an untrusted pickle file.

The rule-based evaluation workflow requires no cloud API key. Optional VLM experiments read credentials from environment variables. Store local configuration in an ignored `.env` file and never commit it. If a key is committed, revoke it with the service provider: deleting the current line does not invalidate the credential or remove it from earlier Git objects.

For public releases, use a clean source snapshot when the development repository contains sensitive history. See [publishing](docs/publishing.md). Static release checks are a useful guard but cannot prove the absence of every secret.

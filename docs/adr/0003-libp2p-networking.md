# libp2p for P2P Networking

All peer-to-peer communication runs through libp2p v0.56: node discovery (Kademlia DHT), encrypted transport (Noise handshake), NAT traversal (STUN), and peer identity verification (Ed25519).

**Why libp2p and not raw TCP or gRPC:** Building a secure P2P stack from scratch is a full-time project — DHT, encrypted transport, NAT traversal, peer identity — all of it needs to be correct and audited. libp2p provides all of these in a production-grade stack used by IPFS (exabyte-scale), Filecoin, and Polkadot.

**Key mappings:**
- `expert://<model-hash>/<expert-id> → [NodeInfo]` maps to Kademlia DHT keys
- Noise handshake provides DTLS-grade encryption without CA infrastructure
- STUN for NAT traversal (no TURN in V1 — ~10% of users behind symmetric NATs excluded)
- Rust-libp2p 0.56 pinned in Cargo.toml to prevent accidental upgrades (API changes significantly between versions)

**What we don't use from libp2p:** WebRTC transport (immature in rust-libp2p, deferred to V2+), PubSub/gossip (not needed in V1), Identify protocol (deferred).

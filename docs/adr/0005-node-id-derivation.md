# Node Identity via Ed25519 Public Key + SHA256

Every node in the swarm is identified by `NodeId([u8; 32])`, which is `SHA256(ed25519_public_key)`. The Ed25519 key pair lives in the domain `KeyPair` entity; signing/verification lives in the `Ed25519Signer` infrastructure adapter.

**Why not just use the raw public key as the ID?** The one-way hash provides privacy — NodeIds can be listed publicly (DHT, catalog) without exposing the node's public key. It also creates a clean separation between identity (NodeId) and authentication material (KeyPair) — future key rotation won't change the NodeId.

**Why not UUID?** Not verifiable — anyone can claim any UUID. A P2P network needs cryptographic identity.

**Why not X.509 certificates?** libp2p uses Noise handshake, not TLS. Certificates add CA management, expiration, and revocation complexity for zero benefit over Ed25519+Noise.

**Consequences:**
- `NodeId::from_public_key(pk)` is pure SHA256 — no I/O, no crypto deps in domain
- Public key must be transmitted alongside signatures for verification (can't derive public key from NodeId)
- 32 bytes: fixed size, Copy semantics, efficient for DHT keys and HashMaps
- NodeId display is 64-char lowercase hex

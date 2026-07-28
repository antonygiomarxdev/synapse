# Single Binary Design

The entire Synapse protocol ships as a single binary: `cargo build --release` produces `synapse-node`. This one binary contains the HTTP gateway (axum), the swarm orchestration, and the DHT (libp2p Kademlia). All three run in the same tokio runtime.

**Why not separate gateway + swarm processes?** The gateway, swarm, and DHT share in-memory state (node registry, expert routes, reputation). Splitting them would require IPC (gRPC or message queue) with serialization overhead, deployment complexity (two binaries, port coordination, health monitoring for both), and latency penalties for request routing. The gateway is lightweight — it doesn't need independent scaling.

**Why not a shared library + microservice pattern?** Same problem — if they're separate processes, they still need IPC. If they're in-process, it's a library call, which is what a single binary does anyway. Premature modularity.

**Fault tolerance:** The DHT is the source of truth. If the binary crashes and restarts, it rejoins the DHT with zero data loss — no persistent state to recover. The gateway is stateless (catalog is cached, routes computed from DHT). This is the Bitcoin model: minimal bootstrap infrastructure, restart is recovery.

**V2+ consideration:** If scaling demands a split (e.g., multiple gateways behind a load balancer), the trait boundaries already exist. The DHT remains the shared truth — a split is deployment-only, not architectural.

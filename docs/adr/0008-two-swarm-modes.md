# Two Swarm Modes: Speculative + DAG

The swarm serves two modes, selected per-request via the `priority` field:

- **Speculative Swarm** (`priority: "realtime"`): N nodes run the full model independently with different random seeds. The coordinator votes per token and returns the majority result. Latency is approximately single-node latency — no network hops per token. Used for chat, autocomplete, and CLI agents.

- **Swarm DAG** (`priority: "batch"`): True expert distribution — each node holds 2-5 experts. Requests flow through the expert graph. The gateway assembles the cheapest valid route. Throughput scales with node count, but individual latency is seconds to minutes. Used for CI/CD, codebase analysis, and dataset processing.

**Why two modes:** Interactive and batch workloads have fundamentally conflicting requirements. Interactive demands low latency — the fastest path is running the full model on one node and verifying by ensemble. Batch demands throughput — the fastest path is distributing experts across many nodes and pipelining requests through the expert graph. A single mode would compromise both.

**Speculative mode details:** All nodes load the full model. Each generates with a different seed. The coordinator collects results and votes per token — majority wins. Minority nodes are flagged for re-sync. This is effectively an ensemble method that trades compute cost (N×) for reliability and auditability.

**DAG mode details:** The gateway consults the DHT's co-activation heat map to build the cheapest expert route. Multiple requests pipeline through the expert graph simultaneously. Mid-request node failures trigger replica fallback — the failing node loses payment, surviving nodes are paid normally.

**Both modes share:** DHT, node registry, reputation system, pricing, and stake/slashing. They're execution strategies, not separate protocols.

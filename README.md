<p align="center">
  <img src="https://img.shields.io/badge/status-mvp--in--development-orange" alt="Status: MVP">
  <img src="https://img.shields.io/badge/license-Apache%202.0-blue" alt="License">
  <img src="https://img.shields.io/badge/rust-1.97%2B-orange" alt="Rust 1.97+">
  <img src="https://img.shields.io/badge/python-3.12%2B-blue" alt="Python 3.12+">
</p>

# Synapse

**Decentralized inference protocol for Mixture-of-Experts models.**

Synapse turns thousands of consumer GPUs into a single distributed inference swarm. Miners contribute idle hardware. Clients consume frontier AI. No datacenter. No gatekeeper. No rate limits.

> *"Any MoE model, served by a swarm of consumer GPUs. Never says no."*

---

## Why Synapse?

MoE (Mixture-of-Experts) models like **Kimi K2.7 Code**, DeepSeek-V2, and Mixtral are the future of LLM architecture. They work by activating only a small fraction of their total parameters per token — typically 1-2%. This means 98% of the model sits idle at any moment.

**Synapse exploits this property.** Instead of cramming the entire model into one datacenter GPU, the swarm distributes experts across hundreds of consumer GPUs. Each node holds just a handful of experts. Requests flow through the swarm, activating only the experts they need.

### Example: Kimi K2.7 Code on Synapse

Kimi K2.7 Code is a ~1 trillion parameter MoE model from Moonshot AI, optimized for coding. It has 384 experts, with 32 active per token. Running the full model requires datacenter hardware.

With Synapse:
- 192 consumer GPUs each hold 2 experts (~11GB VRAM in 4-bit)
- Each token activates only 32 experts across ~16 nodes
- The swarm self-organizes: hot experts auto-replicate, cold experts stay dormant
- A developer in any country can use Kimi K2.7 Code without an API key, rate limit, or regional restriction

### The Pitch

| | Centralized Providers | Synapse |
|---|---|---|
| **Availability** | Rate limits, downtime, regional blocks | Always on while nodes exist |
| **Model selection** | Only what they choose to serve | Any open-weight MoE model |
| **Access control** | API keys, waitlists, KYC | Open protocol, no permission needed |
| **Hardware** | Datacenter H100s ($30K each) | Consumer GPUs (your gaming PC) |
| **Censorship resistance** | Can be shut down, blocked | P2P mesh, no central authority |

Synapse doesn't compete on speed. It competes on **absence of gatekeepers.**

---

## How It Works

### Two Swarm Modes

| Mode | How | Latency | Use Case |
|---|---|---|---|
| **Speculative Swarm** | N nodes run the full model in parallel. Majority vote on each token. | ~1 node latency | Chat, IDEs, CLI agents |
| **Swarm DAG** | True expert distribution. Requests flow through expert nodes. | Not latency-sensitive | CI/CD, codebase analysis, batch eval |

### Speculative Swarm (Realtime)

```
Client: "Write a Rust function..."

     ┌─────────────────────┐
     │    Gateway (Rust)   │
     │    axum + tokio     │
     └──┬───┬───┬───┬───┬──┘
        │   │   │   │   │
   ┌────▼┐ ┌▼────┐ ┌▼────┐ ┌▼────┐ ┌▼────┐
   │Node1│ │Node2│ │Node3│ │Node4│ │Node5│
   │K2.7 │ │K2.7 │ │K2.7 │ │K2.7 │ │K2.7 │
   └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘
      │       │       │       │       │
   "fn "   "fn "   "fn "   "pub"   "fn "
      │       │       │       │       │
      └───────┴───┬───┴───────┴───────┘
                  │
            CONSENSUS: "fn " (4/5)
```

- 5 nodes generate independently with different seeds
- Majority token wins; minority nodes re-sync
- **Latency = single-node latency** — no cross-node communication overhead

### Swarm DAG (Batch)

```
Kimi K2.7 Code: 384 experts, 32 active per token

  Expert #12 held by: Node A ($0.08/1M tokens), Node B ($0.11)
  Expert #47 held by: Node C ($0.09), Node D ($0.14)

  Gateway assembles: A (#12) + C (#47) + ... = cheapest route
  Client pays catalog price (gateway fee included)

  Batch of 100 requests flows through expert DAG simultaneously
```

- True expert distribution — each node only loads 2-5 experts
- Gateway is the market maker: miners compete on price
- Zero barrier to entry: any GPU can mine

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│              SYNAPSE NODE (single Rust binary)           │
│                                                          │
│  ┌──────────────────────┐  ┌──────────────────────────┐ │
│  │  Gateway (axum)      │  │  Swarm Core              │ │
│  │  /v1/models          │  │  Consensus · DAG · KAD   │ │
│  │  /v1/chat/completions│  │  DHT · Pricing · Stake   │ │
│  └──────────┬───────────┘  └──────────┬───────────────┘ │
│             │                         │                  │
│             └─────────┬───────────────┘                  │
│                       │                                  │
│              Unix Socket + protobuf                      │
│                       │                                  │
│  ┌────────────────────▼──────────────────────────────┐  │
│  │  Python Runtime (vLLM)                             │  │
│  │  Weight Loader · Expert Management · Determinism   │  │
│  └───────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

**Single binary.** Gateway + P2P core in Rust (axum + libp2p). Python only for vLLM inference runtime — communicates via Unix socket.

### Tech Stack

| Layer | Language | Why |
|---|---|---|
| **Gateway + P2P Core** | Rust | Zero-cost abstractions, tokio async, <2ms p99, single binary |
| **Inference Runtime** | Python | vLLM is Python-only. Isolated subprocess. |
| **Smart Contracts** | Solidity | L2 standard (Solana programs also planned) |

---

## Quick Start

### Prerequisites
- Rust 1.97+
- Python 3.12+ (for vLLM runtime)
- CUDA-capable GPU (for inference; DHT relay nodes can run CPU-only)

### Build & Run

```bash
git clone https://github.com/antonygiomarxdev/synapse.git
cd synapse

# Build (single binary: gateway + swarm core)
cargo build --release

# Run tests
cargo test

# Start node (gateway + DHT + miner in one process)
./target/release/synapse-node
# → Gateway on http://0.0.0.0:8000
# → DHT peer active
# → GPU detected, experts auto-assigned
```

### Inference Request

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "kimi-k27-code",
    "priority": "realtime",
    "swarm_size": 5,
    "messages": [{"role": "user", "content": "Write a Rust function to merge two sorted arrays"}]
  }'
```

Response:
```json
{
  "id": "chatcmpl-0001",
  "model": "kimi-k27-code",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "fn merge_sorted<T: Ord>(a: &[T], b: &[T]) -> Vec<T> { ..."
    },
    "finish_reason": "stop"
  }]
}
```

---

## Project Structure

```
synapse/
├── synapse-core/            # Rust — everything in one crate
│   ├── src/
│   │   ├── main.rs          #   Binary entrypoint
│   │   ├── gateway/         #   axum HTTP server, catalog, routing
│   │   ├── identity/        #   NodeId, KeyPair, Node
│   │   ├── model/           #   ModelId, ExpertId, Catalog
│   │   ├── swarm/           #   Consensus, Speculative, DAG engines
│   │   ├── economic/        #   Reputation, Pricing, Stake
│   │   ├── transport/       #   WebRTC, Signalling
│   │   └── dht/             #   Kademlia, Expert registry
│   └── proto/               #   Protobuf schemas
├── synapse-runtime/         # Python — vLLM adapter (subprocess)
├── contracts/stake/         # Solidity — StakeManager
├── config/
│   ├── models.toml          #   Curated model catalog
│   └── default.toml         #   Node defaults
└── docs/                    #   Spec + implementation plan
```

---

## Model Catalog

Curated and verified. Community proposals via PR.

| Model | Params | Experts | Active | 🔥 For | License |
|---|---|---|---|---|---|
| **Kimi K2.7 Code** | ~1T | 384 | 32 | Software dev | MIT |
| Mixtral 8x7B | 46.7B | 8 | 2 | General purpose | Apache 2.0 |
| Mixtral 8x22B | 141B | 8 | 2 | Complex reasoning | Apache 2.0 |
| DeepSeek-V2 Lite | 16B | 64 | 6 | Lightweight edge | MIT |
| Qwen2.5-MoE | 57B | 64 | 8 | Multilingual | Apache 2.0 |

---

## How Miners Earn

```
Client pays $0.40/1M tokens → Gateway keeps 15-20% → Miner receives remainder

Example (Kimi K2.7 Code, expert #12):
  $0.08/1M tokens × 1000 requests × 1000 tokens = $0.08/batch
  Average: $2-12/day depending on GPU, demand, and uptime

You earn for verified work only.
Garbage output → no payment + reputation flag → slashing.
```

---

## Security

### Consensus
- **Realtime:** Ensemble voting. N nodes. Majority wins. Divergent → re-sync.
- **Batch:** Statistical audit. ~5% verified by second expert set.

### Slashing (fully automatic)
1. Single divergence → no payment (re-sync)
2. 3+ per request → expelled + flag
3. 10+ flags/24h → stake frozen 48h
4. 50+ flags/7d → 20% slashed
5. Fraud → full slashing + ban

---

## Roadmap

**V1 — Swarm MVP (Q3-Q4 2026)**
- [x] Spec + architecture design
- [x] Repository scaffold (Rust + Python + Solidity)
- [ ] Kademlia DHT + expert registry
- [ ] Speculative Swarm (realtime consensus)
- [ ] Swarm DAG (batch expert distribution)
- [ ] vLLM runtime adapter
- [ ] axum gateway (OpenAI-compatible)
- [ ] USDC payments on L2

**V2 — Scale & Privacy (2027)**
- [ ] Kimi K2.7 Code (~192 node swarm)
- [ ] Split Inference privacy
- [ ] Client SDKs (Python, TypeScript, Rust)

**V3 — Kimi K3 Era (2028+)**
- [ ] 3T+ class models
- [ ] KDA-aware DAG parallelism
- [ ] Governance DAO

---

## Contributing

Open source (Apache 2.0). Protocol, node software, and launcher are free.

See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## License

- Protocol: [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- Core, node, launcher: [Apache 2.0](LICENSE)
- Reference gateway: BSL

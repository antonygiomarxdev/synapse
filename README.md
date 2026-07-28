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

MoE (Mixture-of-Experts) models like Mixtral, DeepSeek-V2, and Kimi K2 are the future of LLM architecture. They work by activating only a small fraction of their total parameters per token — typically 1-2%. This means 98% of the model sits idle at any moment.

**Synapse exploits this property.** Instead of cramming the entire model into one datacenter GPU, the swarm distributes experts across hundreds of consumer GPUs. Each node holds just a handful of experts. Requests flow through the swarm, activating only the experts they need.

### The Pitch

| | Centralized Providers (OpenAI, Anthropic, Groq) | Synapse |
|---|---|---|
| **Availability** | Rate limits, downtime, regional blocks | Always on while nodes exist |
| **Model selection** | Only what they choose to serve | Any open-weight MoE model |
| **Access control** | API keys, waitlists, KYC | Open protocol, no permission needed |
| **Hardware** | Datacenter H100s ($30K each) | Consumer GPUs (your gaming PC) |
| **Censorship resistance** | Can be shut down, blocked, restricted | P2P mesh, no central authority |

Synapse doesn't compete on speed. It competes on **absence of gatekeepers.**

---

## How It Works

### Two Swarm Modes

| Mode | How | Latency | Use Case |
|---|---|---|---|
| **Speculative Swarm** | N nodes run the full model in parallel. Results merged by majority vote. | ~1 node latency | Chat, IDEs, CLI agents |
| **Swarm DAG** | True expert distribution. Requests flow through expert nodes. | Not latency-sensitive | CI/CD, codebase analysis, batch evaluation |

### Speculative Swarm (Realtime)

```
Client: "Write a Python function..."

     ┌─────────────────────┐
     │    Gateway           │
     │  (Coordinator)       │
     └──┬───┬───┬───┬───┬──┘
        │   │   │   │   │
   ┌────▼┐ ┌▼────┐ ┌▼────┐ ┌▼────┐ ┌▼────┐
   │Node1│ │Node2│ │Node3│ │Node4│ │Node5│
   └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘
      │       │       │       │       │
   "def"   "def"   "def"   "fun"   "def"
      │       │       │       │       │
      └───────┴───┬───┴───────┴───────┘
                  │
            CONSENSUS: "def" (4/5)
```

- 5 nodes generate independently
- Majority token wins
- Divergent nodes are re-synced or flagged
- **Latency = single-node latency** (no cross-node communication overhead)

### Swarm DAG (Batch)

```
Model: Mixtral 8x7B (8 experts, 2 active per token)

  Expert #3 held by: Node A ($0.08/1M tokens), Node B ($0.11)
  Expert #7 held by: Node C ($0.09), Node D ($0.14)

  Gateway assembles: A (#3) + C (#7) = $0.17
  Client pays: $0.25 (gateway fee included)

  Request 1 → activates experts [3, 7] → routed through A + C
  Request 2 → activates experts [1, 5] → routed through different nodes
```

- True expert distribution across the swarm
- Gateway assembles the cheapest valid route
- Miners compete on price — expensive nodes get no work

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│              B2B API Gateway (FastAPI)               │
│     OpenAI-compatible  ·  Catalog  ·  Pricing        │
└─────────────────────┬────────────────────────────────┘
                      │
            Encrypted (WebRTC / Noise)
                      │
                      ▼
┌──────────────────────────────────────────────────────┐
│              Swarm Orchestration (Rust)              │
│     Kademlia DHT  ·  Consensus  ·  Routing           │
│     Speculative Engine  ·  DAG Engine                │
└─────────────────────┬────────────────────────────────┘
                      │
              Hidden states / tensors
                      │
                      ▼
┌──────────────────────────────────────────────────────┐
│              Compute Node (Python + vLLM)            │
│     Weight Loader  ·  Expert Management  ·  Runtime  │
└──────────────────────────────────────────────────────┘
```

### Tech Stack

- **Core (P2P, DHT, consensus):** Rust + libp2p + protobuf
- **Inference runtime:** Python + vLLM (communicates with core via Unix socket)
- **Gateway:** Python + FastAPI (OpenAI-compatible endpoints)
- **Payments:** USDC on L2 (Solana/Base) via Solidity smart contracts

---

## Quick Start

### Prerequisites

- Rust 1.97+
- Python 3.12+
- CUDA-capable GPU (for inference; DHT-only nodes can run CPU)

### Build

```bash
git clone https://github.com/antonygiomarxdev/synapse.git
cd synapse

# Rust core
cargo build --release
cargo test

# Python runtime
cd synapse-runtime && pip install -e ".[dev]"

# Gateway
cd ../synapse-gateway && pip install -e ".[dev]"

# Smart contracts
cd ../contracts/stake && npm install && npx hardhat compile
```

### Run a Miner Node

```bash
# 1. Install Synapse
cargo build --release

# 2. Stake USDC (required for eligibility)
#    Send ≥100 USDC to the StakeManager contract on L2

# 3. Start mining (auto-assigns experts based on your VRAM)
./target/release/synapse-node
# Output: "Serving Mixtral 8x7B — Experts #3, #7 — Earning ~$0.12/hr"
```

### Run the Gateway

```bash
cd synapse-gateway
uvicorn synapse_gateway.api:app --reload
# Gateway available at http://localhost:8000
```

### Make an Inference Request

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mixtral-8x7b",
    "priority": "realtime",
    "swarm_size": 5,
    "messages": [{"role": "user", "content": "Write a Python function to sort a list"}]
  }'
```

---

## Project Structure

```
synapse/
├── synapse-core/            # Rust — P2P core, DHT, domain logic
│   ├── src/
│   │   ├── identity/        #   NodeId, KeyPair, Node aggregate
│   │   ├── model/           #   ModelId, ExpertId, Catalog
│   │   ├── swarm/           #   Consensus, Speculative engine, DAG engine
│   │   ├── economic/        #   Reputation, Pricing, Stake management
│   │   ├── transport/       #   WebRTC, Signalling
│   │   └── dht/             #   Kademlia, Expert registry, Bootstrap
│   └── proto/               #   Protobuf schemas
├── synapse-runtime/         # Python — vLLM adapter, weight loader
├── synapse-gateway/         # Python — FastAPI, B2B API, catalog
├── contracts/stake/         # Solidity — StakeManager + graduated slashing
├── config/
│   ├── models.toml          #   Curated model catalog
│   └── default.toml         #   Node defaults
├── docs/superpowers/        #   Design spec + implementation plan
└── scripts/                 #   Dev tooling
```

---

## Model Catalog

Synapse uses a curated catalog. Models are manually verified (SHA256 hash, architecture, license) before listing. Community proposals welcome via GitHub PR.

| Model | Params | Experts | Active/Token | VRAM/node (2 experts) | License |
|---|---|---|---|---|---|
| Mixtral 8x7B | 46.7B | 8 | 2 | ~9GB | Apache 2.0 |
| Mixtral 8x22B | 141B | 8 | 2 | ~24GB | Apache 2.0 |
| DeepSeek-V2 Lite | 16B | 64 | 6 | ~1.3GB | MIT |
| Qwen2.5-MoE | 57B | 64 | 8 | ~3GB | Apache 2.0 |
| Kimi K2.7 Code | ~1T | ~384 | ~32 | ~11GB | MIT (modified) |

*VRAM includes shared parameters (embeddings, attention, gate). 4-bit quantization assumed.*

---

## How Miners Earn

```
Client pays $0.25/1M tokens → Gateway keeps 15-20% → Miner receives remainder

Example (Mixtral 8x7B, expert #3):
  You serve Expert #3 at $0.08/1M tokens
  1000 requests, 1000 tokens each = 1M tokens
  You earn $0.08 for that batch

  Average earnings: $2-10/day depending on GPU tier, demand, and uptime.

Key principle: you earn for verified work only.
If you produce garbage → no payment + reputation flag → slashing.
```

---

## Security

### Consensus

- **Realtime:** Ensemble voting. N nodes generate independently. Majority wins.
- **Batch:** Statistical audit. ~5% of requests verified by second expert set.

### Slashing

Fully automatic. Graduated penalties:
1. Single divergence → no payment for that token (re-sync)
2. 3+ divergences per request → expelled + flag
3. 10+ flags in 24h → stake frozen 48h
4. 50+ flags in 7 days → 20% stake slashed + score reset
5. Fraud pattern → full slashing + permanent ban

### Attack Resistance

| Attack | Defense |
|---|---|
| Garbage output | Ensemble voting rejects. Audit detects. |
| Sybil (fake nodes) | Stake required. New nodes have low priority. |
| Free-riding (charge without work) | No valid response → no payment. |
| Prompt theft | WebRTC DTLS encryption + economic slashing. |
| Price cartel | Zero barrier to entry. New miners undercut. |

---

## Roadmap

**V1 (2026 Q3-Q4) — Swarm MVP**
- [x] Protocol design & spec
- [x] Repository scaffold
- [ ] Kademlia DHT + expert registry
- [ ] Speculative Swarm (realtime)
- [ ] Swarm DAG (batch)
- [ ] vLLM runtime adapter
- [ ] FastAPI gateway (OpenAI-compatible)
- [ ] USDC payments on L2

**V2 (2027) — Scale & Privacy**
- [ ] Kimi K2.7 Code (~192 node swarm)
- [ ] Split Inference privacy mode
- [ ] Full reputation system (Platinum tiers)
- [ ] Client SDKs (Python, TypeScript, Rust)

**V3 (2028+) — Kimi K3 Era**
- [ ] KDA-aware DAG parallelism
- [ ] Speculative expert prediction
- [ ] 3T+ class model support
- [ ] Governance token / DAO

---

## Contributing

Synapse is open source (Apache 2.0). The protocol, node software, and launcher are free to use, modify, and redistribute. The reference gateway is source-available (BSL).

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines (coming soon).

### Ways to Contribute

- **Code:** Check the [open issues](https://github.com/antonygiomarxdev/synapse/issues)
- **Models:** Propose new models for the catalog via PR to `config/models.toml`
- **Run a node:** Help grow the swarm
- **Research:** The [design spec](docs/superpowers/specs/2026-07-27-synapse-design.md) has open questions for V2+

---

## License

- Protocol specification: [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- Core, node software, launcher: [Apache 2.0](LICENSE)
- Reference gateway: Business Source License (BSL)

---

## Acknowledgments

Synapse draws inspiration from:
- **Bitcoin** — bootstrap and self-sustaining P2P network model
- **Helium / Filecoin / Render** — open protocol with reference commercial implementation
- **Moonshot AI** — Kimi K2/K3 architecture proves MoE is the future
- **Predictive coding (neuroscience)** — speculative expert activation
- **Immune system** — hot/cold expert replication

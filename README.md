<p align="center">
  <img src="https://img.shields.io/badge/license-Apache%202.0-blue" alt="License">
  <img src="https://img.shields.io/badge/status-validating%20thesis-orange" alt="Status: Validating thesis">
  <img src="https://img.shields.io/badge/rust-1.97%2B-orange" alt="Rust 1.97+">
  <img src="https://img.shields.io/badge/python-3.12%2B-blue" alt="Python 3.12+">
  <img src="https://img.shields.io/badge/coverage-80%25%2B-brightgreen" alt="Coverage 80%+">
  <img src="https://img.shields.io/badge/CI-gauntlet%20passed-brightgreen" alt="Gauntlet passed">
</p>

# Synapse

**Decentralized inference for Mixture-of-Experts models — validating the thesis.**

> *Can a small network of heterogeneous GPUs complete inference jobs reliably, verifiably, and more affordably than a centralized alternative? We're testing that.*

---

## Where we are (July 2026)

Synapse is in **thesis validation**. We're not building a global P2P marketplace yet. We're answering one question first: does the core technical idea work?

### ✅ Validated

| Claim | Evidence |
|---|---|
| Rust ↔ Python worker via Unix socket + protobuf works | [Spike: vLLM viability](docs/superpowers/spikes/2026-07-28-vllm-viability-spike.md) — 100% success, <2ms overhead |
| Real GPU inference works through the pipeline | Qwen3 8B + Ollama backend — 100% success, 40 tok/s |
| **MoE model runs through the pipeline** | **IBM Granite 3.1 MoE 3B — 40 experts, 8 active, 100% success** |
| InferencePort abstraction is real | 3 backends (vLLM, Ollama, Mock) swapped without changing protocol |
| NVML driver issues on Linux | [Documented workaround](docs/superpowers/spikes/2026-07-28-vllm-viability-spike.md#72-intento-con-gpu-real--bloqueado-por-driver-mismatch-2026-07-29) (reboot fixes it) |
| vLLM + MoE needs >8 GB VRAM | [Documented](docs/superpowers/spikes/2026-07-28-vllm-viability-spike.md#73-gpu-real--ollama--qwen3-8b-q4_k_m-2026-07-29) — Qwen-MoE-A2.7B in FP16 = 7.1 GiB weights alone |
| llama.cpp + GGUF fits MoE in 8 GB | Granite MoE 3B Q4_K_M = ~2 GB, works perfectly |
| Same principle as ESP32-AI | [Slava S ran 28.9M params on 512KB SRAM](https://www.tomshardware.com/tech-industry/artificial-intelligence/ai-developer-runs-28-9-million-parameter-model-on-usd10-esp32-s3-microcontroller-uses-googles-per-layer-embeddings-technique-stores-table-on-16mb-flash-memory) using mmap + 4-bit — same technique, 16,000x more RAM here |

### 🔄 In progress

| What | Status |
|---|---|
| Multi-worker coordination | Spike designed, pending |
| Crash recovery / fault tolerance | Spike designed, pending |
| Job model + async batch API | Next (MVP design phase) |
| 2+ node real network | Needs more hardware or cloud GPUs |

### ⏳ Post-MVP (not built yet)

Everything below is **design intent, not working code**:
- DHT / Kademlia node discovery
- P2P expert distribution
- Consensus / slashing / staking
- On-chain payments (L2)
- WebRTC transport
- OpenAI-compatible chat streaming

---

## The vision (where we're going)

MoE (Mixture-of-Experts) models like Kimi K3, DeepSeek-V2, and Mixtral activate only ~1-2% of their parameters per token. The other 98% sits idle.

**Synapse exploits this.** Instead of cramming the entire model into datacenter GPUs, the swarm distributes experts across consumer hardware. Each node holds a handful of experts. Requests flow through the swarm, activating only the experts they need.

The pitch:
- No API keys, no rate limits, no regional blocks
- Any open-weight MoE model
- Consumer GPUs instead of datacenter H100s
- Censorship-resistant P2P mesh

But first: **prove the core works.** See [pivot plan](docs/superpowers/spikes/2026-07-28-vllm-viability-spike.md).

---

## What actually runs today

```bash
git clone https://github.com/antonygiomarxdev/synapse.git
cd synapse

# Build
cargo build --release

# Run all tests (gauntlet)
make gauntlet

# Start the gateway (health + model list endpoints)
cargo run --release
# → Gateway on http://0.0.0.0:8000

# Run the viability spike (Rust → Python → GPU inference)
cargo run --bin spike -- --test=smoke --model=ollama:granite3.1-moe:3b
```

### What `cargo run` actually does

The gateway binary currently serves:
- `GET /health` → `{"status": "ok"}`
- `GET /v1/models` → hardcoded Kimi K3 catalog entry

The `/v1/chat/completions` endpoint is a mock — it echoes back a static response. Real inference goes through the `spike` binary, which is a separate experiment (see spike doc).

---

## Current architecture

```
┌──────────────────────────────────────────────────────────┐
│              SYNAPSE NODE (Rust binary)                  │
│                                                          │
│  ┌──────────────────────┐                                │
│  │  Gateway (axum)      │  ← /health, /v1/models (real) │
│  │  /v1/chat/completions│  ← mock, pending batch API     │
│  └──────────────────────┘                                │
│                                                          │
│  ┌──────────────────────┐                                │
│  │  Spike (experiment)  │  ← Rust → Python worker        │
│  │  src/bin/spike.rs    │     Unix socket + protobuf     │
│  └──────────┬───────────┘     validates the pipeline      │
│             │                                              │
│    Unix socket + protobuf (spike.proto)                   │
│             │                                              │
│  ┌──────────▼───────────┐                                │
│  │  Python Runtime      │  ← vLLM / Ollama / Mock        │
│  │  synapse_runtime/    │     interchangeable backends   │
│  └──────────────────────┘                                │
└──────────────────────────────────────────────────────────┘
```

---

## Tech stack (all implemented)

| Layer | Language | What's built |
|---|---|---|
| Gateway | Rust (axum) | Health, model list, chat mock |
| Spike coordinator | Rust (tokio) | Worker spawn, dispatch, metrics, crash test |
| Inference runtime | Python | vLLM, Ollama, Mock backends via Unix socket |
| Identity | Rust | NodeId (SHA-256 of Ed25519) |
| Model catalog | Rust | ModelId, ModelEntity, hardcoded list |
| Protocol | Protobuf | spike.proto (SpikeRequest/SpikeResponse) |
| Stubs (not built) | Rust | DHT, WebRTC, Swarm, Economic, Slashing |

---

## Project structure

```
synapse/
├── synapse-core/            # Rust — single binary + spike
│   ├── src/
│   │   ├── main.rs          #   Gateway binary (health + models)
│   │   ├── bin/spike.rs     #   Viability spike (Rust → Python → GPU)
│   │   ├── gateway/         #   axum HTTP: api, catalog, router (stubs)
│   │   ├── identity/        #   NodeId (real), rest pending
│   │   ├── model/           #   ModelId, ModelEntity (real)
│   │   ├── swarm/           #   Stubs (consensus, DAG, speculative)
│   │   ├── economic/        #   Stubs (reputation, pricing, stake)
│   │   ├── transport/       #   Stubs (WebRTC, signalling)
│   │   └── dht/             #   Stubs (Kademlia, registry)
│   └── proto/               #   spike.proto + synapse.proto
├── synapse-runtime/         # Python — worker with 3 backends
│   └── synapse_runtime/
│       └── worker.py        #   vLLM / Ollama / Mock engines
├── contracts/stake/         # Solidity — StakeManager (prototype)
├── config/
│   ├── models.toml          #   Model catalog (aspirational)
│   └── default.toml         #   Node defaults
├── docs/superpowers/
│   └── spikes/              #   Thesis validation evidence
├── features/                #   BDD specs (behavioral contracts)
└── scripts/
    └── run_spike.sh         #   Spike runner
```

---

## The quality gauntlet

Every line of code — whether written by humans, Claude, Kimi, or a hamster — must pass the same gauntlet before merging:

```bash
make gauntlet   # Runs: fmt, lint, test, coverage, mutants, audit, BDD
```

| Gate | Tool | Threshold |
|---|---|---|
| Formatting | rustfmt, ruff | Must match exactly |
| Linting | clippy -D warnings, ruff | Zero warnings |
| Unit tests | cargo test, pytest | All green |
| Coverage | cargo-llvm-cov | ≥80% lines, ≥80% functions |
| Mutation testing | cargo-mutants | All mutants killed |
| Security audit | cargo-audit, cargo-deny | Zero CVEs |
| BDD specs | Gherkin (features/) | All scenarios pass |
| Smart contracts | hardhat test, solhint | All green |

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). All changes go through PR with gauntlet passing.

---

## License

Apache 2.0. See [LICENSE](LICENSE).

---

*Synapse is validating a thesis. The README will update as evidence accumulates. Last updated: 2026-07-29 after [vLLM viability spike](docs/superpowers/spikes/2026-07-28-vllm-viability-spike.md).*

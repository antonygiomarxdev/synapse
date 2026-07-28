# SYNAPSE: Decentralized Inference Protocol for MoE Models

**Version:** 2.2 — Swarm MVP Design (Final)  
**Date:** 2026-07-27  
**Status:** Complete — ready for implementation planning

---

## 1. OVERVIEW

### 1.1 Product Definition

Synapse is a **swarm-based P2P inference protocol for Mixture-of-Experts (MoE) models**. It coordinates thousands of consumer GPUs — each holding a fraction of a model's experts — into a distributed inference mesh that operates as a single logical compute surface.

**Synapse is not a model API. It is a compute swarm.** Think of it as a decentralized GPU cluster where anyone can contribute hardware and anyone can consume inference, without a central operator.

### 1.2 The Swarm Thesis

MoE models are inherently swarm-friendly: 896 experts, only 16 active per token. This means ~98% of the model sits idle for any given token. Synapse turns this property into infrastructure:

- **Each node holds a handful of experts** (not the full model)
- **Nodes self-organize into a swarm** via DHT discovery
- **Requests flow through the swarm** activating only the experts they need
- **The swarm is the computer**

This is genuinely novel. No existing inference service works this way. Centralized providers cram the entire model into a rack of H100s. Synapse distributes it across the internet.

### 1.3 Two Operational Modes

Synapse operates in two modes from day one, selected per-request by the client:

| Mode | How It Works | Use Case | Latency | Compute Cost |
|---|---|---|---|---|
| **Swarm DAG** (batch) | True expert distribution. Request flows through expert graph. Many requests pipelined simultaneously. Gateway assembles cheapest viable expert route. Mid-request node failures handled transparently. | CI/CD, codebase analysis, benchmarks, dataset processing | Not latency-sensitive | 1× (standard) |
| **Speculative Swarm** (realtime) | Ensemble generation. N nodes run the FULL model independently with different seeds. Results merged by coordinator. | IDE autocomplete, chat, CLI agents | ~1 node latency | N× (higher) |

### 1.4 The Pitch

> *"Any MoE model, served by a swarm of consumer GPUs. No datacenter. No gatekeeper. No rate limits. Never says no."*

### 1.5 Market Dynamics

**Synapse is a marketplace, not a subsidized service.** No entity pays miners out of pocket. The economic model is:

- **Miners compete on price.** Each publishes their per-token rate to the DHT.
- **Gateway is the market maker.** It assembles the cheapest valid expert route and quotes a single fixed price to the client.
- **Prices self-regulate.** If a model gets expensive, new miners join to capture margin → supply increases → price drops. Zero barrier to entry (any GPU can mine).
- **No cartel risk.** To control a model's price, you'd need 100% of its expert replicas. With open participation, that's impossible.
- **Client always knows the price upfront.** The catalog shows final prices. No surprises.

---

## 2. TARGET MODELS

### 2.1 Philosophy: Protocol Designed for MoE

Synapse is purpose-built for MoE architectures. The protocol's primitives — expert routing, DAG execution, speculative parallel generation — assume sparse expert activation. Dense models are not supported by design.

### 2.2 V1 Swarm-Compatible Models

These models are the V1 catalog. All are open-weight and fit the swarm architecture.

| Model | Total Params | Experts | Active/Token | Expert Size (4-bit) | Shared Params (4-bit) | Nodes for Full Coverage (2 experts/node) |
|---|---|---|---|---|---|---|
| **Mixtral 8x7B** | 46.7B | 8 | 2 | ~3GB | ~3GB | 4 nodes |
| **Mixtral 8x22B** | 141B | 8 | 2 | ~9GB | ~6GB | 4 nodes (Tier Full) |
| **Qwen2.5-MoE** | 57B | 64 | 8 | ~0.5GB | ~2GB | 32 nodes |
| **DeepSeek-V2 Lite** | 16B | 64 | 6 | ~0.15GB | ~1GB | 32 nodes |
| **Kimi K3** | 2.8T | 896 | 16 | ~1.5GB | ~12GB | ~448 nodes |

> **Note on VRAM:** In addition to expert weights, every node must load shared parameters (embeddings, attention layers, gate network — loaded once regardless of expert count). Actual VRAM = `(num_experts × expert_size) + shared_params`.

### 2.3 Why MoE-Only

Dense models with frontier quality require >70GB VRAM — datacenter territory. MoE models achieve frontier quality with far less active memory per token, making consumer-GPU distribution viable.

### 2.4 The Scaling Path

```
V1 (2026): Mixtral 8x7B — 4 nodes, 2 experts each
V2 (2027): Kimi K3 — ~448 nodes, 2 experts each
V3 (2028+): Next-gen MoE — protocol scales with model expert count
```

The protocol is identical. Only the swarm size changes.

---

## 3. ARCHITECTURE

```
┌──────────────────────────────────────────────────────────────┐
│              INTERFACE LAYER (B2B API Gateway)               │
│  - OpenAI-compatible API    - Curated catalog                │
│  - Per-token metering       - B2B billing                    │
│  - Auto mode (routing)      - Manual model selection         │
│  - priority: "realtime" | "batch"                           │
│  - Gateway redundancy (active/passive + DNS failover)       │
└──────────────────────────┬───────────────────────────────────┘
                           │
              Encrypted (WebRTC / Noise Protocol)
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│               SWARM ORCHESTRATION LAYER                      │
│                                                              │
│  ┌─────────────────────┐  ┌──────────────────────────────┐  │
│  │  Kademlia DHT       │  │  Execution Engines           │  │
│  │  Expert -> [NodeIDs]│  │                              │  │
│  │  Co-activation map  │  │  Swarm DAG (batch mode)      │  │
│  │  Reputation scores  │  │  Speculative Swarm (realtime)│  │
│  │  Node pricing data  │  │                              │  │
│  └─────────────────────┘  └──────────────────────────────┘  │
│                                                              │
│  Consensus: ensemble voting (realtime)                       │
│           + statistical audit (batch)                        │
│  Resilience: partition tolerance, re-sync, failure recovery  │
└──────────────────────────┬───────────────────────────────────┘
                           │
       ┌───────────────────┤
       │  InferencePort    │  Rust trait (runtime-agnostic)
       │  load / generate  │  protobuf over Unix socket
       │  verify / detect  │
       └───────────────────┤
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                COMPUTE NODE LAYER                            │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐               │
│  │  vLLM     │  │ llama.cpp │  │  SGLang   │  ...more      │
│  │  Adapter  │  │  Adapter  │  │  Adapter  │               │
│  │  (V1)     │  │  (V2+)    │  │  (V2+)    │               │
│  └───────────┘  └───────────┘  └───────────┘               │
│  - Experts + shared params loaded in VRAM (4-bit)            │
│  - Weights identified by SHA256 hash                         │
│  - Auto-assignment: system chooses experts + runtime         │
│  - Launcher: install -> connect GPU -> click Start -> earn   │
└──────────────────────────────────────────────────────────────┘
```

---

## 4. INTERFACE LAYER

### 4.1 Endpoints

```
GET  /v1/models                          → Curated catalog
POST /v1/chat/completions                → Inference
```

### 4.2 Request Format

```json
POST /v1/chat/completions
{
  "model": "mixtral-8x7b",
  "priority": "realtime",
  "swarm_size": 8,
  "messages": [{"role": "user", "content": "..."}]
}
```

| Parameter | Values | Description |
|---|---|---|
| `model` | model ID or `"auto"` | Model to use. `"auto"` lets gateway pick cheapest available. |
| `priority` | `"realtime"` or `"batch"` | Swarm mode. |
| `swarm_size` | 2-32 or `"auto"` | Nodes in speculative swarm. `"auto"` picks optimal. |

### 4.3 Catalog Curation

**V1: Synapse Inc. curates the catalog.** Models are manually verified before listing:
- Weight hash verified against official source (HuggingFace, model author)
- Architecture validated (must be MoE, must support deterministic inference)
- License checked (must be open-weight, commercially usable)

Community can propose models via GitHub PR. Synapse Inc. reviews and merges.

**V2+:** Community voting via node reputation. Platinum-tier nodes vote to approve/reject model proposals.

### 4.4 Catalog Response

```json
GET /v1/models
{
  "models": [
    {
      "id": "mixtral-8x7b",
      "total_params": "46.7B",
      "active_params": "12.9B",
      "experts": {"total": 8, "active_per_token": 2},
      "shared_params_size": "3GB",
      "swarm": {
        "expert_nodes": 4,
        "expert_replicas": {"expert_0": 3, "expert_1": 2, "expert_2": 4},
        "min_full_replicas": 2
      },
      "pricing": {
        "batch_per_1M_input": 0.10,
        "batch_per_1M_output": 0.15,
        "realtime_per_1M_input": 0.50,
        "realtime_per_1M_output": 0.75
      },
      "status": "healthy"
    }
  ]
}
```

Pricing in the catalog is the **final client price** (gateway fee included). Miners compete underneath.

### 4.5 Gateway Redundancy

The gateway is the only centralized component in V1. To avoid SPOF:

- **Multiple gateway instances** behind a load balancer
- **DNS failover:** If primary region fails, clients resolve to secondary
- **Gateway state is soft:** DHT is the source of truth. A failed gateway restarts and re-joins with zero data loss.
- **Future (V2+):** Client SDKs can query the DHT directly for basic routing, bypassing the gateway entirely for non-billing operations.

### 4.6 Privacy

| Mode | V1 | V2+ |
|---|---|---|
| `"standard"` | WebRTC encrypted + economic trust | Same |
| `"private"` | Not available | Split Inference |

---

## 5. SWARM ORCHESTRATION

### 5.1 Bootstrap & Initial Infrastructure

**Problem:** How does Node #1 join an empty network? How are the first model weights distributed? Who runs STUN/TURN?

**Solution (Bitcoin model): minimal seed infrastructure, network self-sustains after bootstrap.**

| Resource | V1 (Synapse Inc. seeds) | V2+ (network self-sustains) |
|---|---|---|
| **Bootstrap nodes** | 3-5 VPS (~$50/mo total) with hardcoded IPs in node software | Any stable node auto-promotes to bootstrap. DNS seeds. |
| **Weight distribution** | Launcher downloads from HuggingFace. Once ≥1 node has weights, P2P takes over. | Entirely P2P. HF remains as eternal fallback. |
| **STUN** | Free (Google STUN servers). Works for ~80% of users. | Same. Free, no cost. |
| **TURN** | None in V1. ~10% of users behind restrictive NATs cannot participate. | 1-2 volunteer or Synapse-run TURN servers. |
| **Gateway** | Synapse Inc. runs reference gateway. | Anyone can run a gateway (open protocol). |

This is exactly how Bitcoin bootstrapped: Satoshi ran the first node, DNS seeds came later.

### 5.2 Kademlia DHT — The Swarm's Nervous System

The DHT maintains the expert registry:

```
Key: expert://<model-hash>/<expert-id>
Value: [NodeID(latency, reputation, stake, price_per_token), ...]

Example:
expert://sha256(mixtral-8x7b)/3 → [
  {node: "abc123", latency_ms: 45, reputation: 720, stake: 500, price: 0.08},
  {node: "def456", latency_ms: 120, reputation: 890, stake: 2000, price: 0.11}
]
```

**Co-activation heat map:**
```
coact://<model-hash>/<expert-id> → [expert-id, ...]
coact://sha256(mixtral-8x7b)/3 → [3, 7, 1, 5]
```

This data drives expert placement: co-activated experts are preferentially placed on the same node.

### 5.3 Transport: WebRTC

- Direct connections between nodes for tensor transfer
- Native NAT traversal (STUN)
- Point-to-point DTLS encryption
- Payload: hidden states (~8-16KB)
- **No TURN in V1.** ~10% of potential miners behind symmetric NATs are excluded. Acceptable for MVP.

### 5.4 Swarm DAG Mode (Batch)

**How it works:** True expert distribution. Each node holds 2-5 experts. Requests flow through the expert graph. Gateway assembles the cheapest valid route.

```
Gateway receives batch of 5 independent requests
  │
  ├─ Request 1 needs experts [3, 7] → route via Miner A (expert 3, $0.08) + Miner D (expert 7, $0.09)
  ├─ Request 2 needs experts [1, 5] → route via Miner B (expert 1, $0.07) + Miner E (expert 5, $0.10)
  ├─ Request 3 needs experts [3, 7] → same route as Request 1 (pipeline them)
  └─ ...
```

- Throughput scales with swarm size
- Individual latency: not optimized (seconds)
- Use case: CI/CD, codebase analysis, benchmarks

**Mid-request failure recovery:** If a node fails during a batch request, the gateway handles recovery transparently. The failing node loses its payment; the surviving nodes are unaffected. See §5.7.

### 5.5 Speculative Swarm Mode (Realtime) + Re-Sync Protocol

**How it works:**

```
Gateway sends prompt to N nodes simultaneously
  │
  ├─ Node 1 (seed=A) → "def fibo" ✓
  ├─ Node 2 (seed=B) → "def fib"  ✗ (diverged)
  ├─ Node 3 (seed=C) → "def fibo" ✓
  ├─ Node 4 (seed=D) → "def fibo" ✓
  └─ Node 5 (seed=E) → "def fib"  ✗ (diverged)
       │
       ▼
  CONSENSUS: "def fibo" (3/5)
```

**Re-Sync Protocol:**

When a node diverges from consensus:

```
1. Divergent node does NOT get paid for the diverged token
2. Coordinator sends current consensus state + KV cache to divergent node
3. Divergent node re-syncs and continues from the consensus token
4. If same node diverges 3+ times in one request:
   → Expelled from that request (no further payment)
   → Flag on reputation score
5. Chronic divergers (10+ flags in 24h) → automatic partial slashing
```

Nodes are incentivized to converge: they only earn on tokens that match consensus.

**Key properties:**
- Each node has the COMPLETE model (speculative mode only)
- Nodes generate independently — no cross-node communication
- Latency = latency of slowest node
- Built-in consensus replaces post-hoc audit

**Swarm size effects:**

| Swarm Size | Consensus Strength | Compute Cost | Latency |
|---|---|---|---|
| 3 | Ok for non-critical | 3× | ~1 node |
| 5 | Good default | 5× | ~1 node |
| 8 | Strong | 8× | ~1 node |
| 16 | Very strong | 16× | ~1 node |

**Mid-request node failure:** If a node in the speculative swarm disconnects mid-generation, the gateway simply ignores that node. Consensus is computed from remaining nodes. If too many nodes drop (no majority possible), the request returns 503. Surviving nodes are paid normally. See §5.7.

### 5.6 Consensus and Verification

**Realtime mode (ensemble voting):**
- All N nodes generate independently
- Coordinator compares token-by-token
- Majority token wins
- Minority nodes: re-sync or flag (see 5.5)

**Batch mode (statistical audit):**
- ~5% of requests duplicated to a second set of expert nodes
- Log-probability matrices compared (same model, same seed → must be identical)
- Mismatch → slashing of divergent nodes

### 5.7 Mid-Request Failure Recovery

**Principle:** The failing node pays the cost. No other party is penalized.

**Batch mode (Swarm DAG):**

```
Request using Expert #3 (Node A) + Expert #7 (Node B)
  → Node B crashes mid-computation
  
  Gateway handles:
  1. Node A: paid normally (completed its work correctly)
  2. Node B: no payment + mid-request timeout flag on reputation
  3. Gateway re-routes Expert #7 portion to Node C (replica)
  4. Request completes, client receives normal response
  5. If NO replica of Expert #7 exists:
     → Model was already marked "degraded" in catalog
     → Client chose to use it knowing the risk
     → Error 503 returned
```

**Speculative mode:**

```
5-node swarm, Node 3 disconnects mid-generation
  
  Gateway handles:
  1. Removes Node 3 from the consensus group
  2. Continues with 4 remaining nodes (consensus = 3/4 needed)
  3. If too many nodes drop (< 3 remaining for swarm_size=5):
     → Error 503 returned to client
  4. Surviving nodes: paid normally
  5. Disconnected nodes: no payment, timeout flag
```

**The key guarantee:** A single node failure never causes an error visible to the client — as long as at least one replica exists (batch) or majority remains (speculative).

### 5.8 Network Partition Handling (Split-Brain)

**Problem:** If the swarm splits in two (e.g., transatlantic cable cut), each half could operate independently.

**Strategy: Availability over consistency (CAP theorem — same as Bitcoin, DNS).**

**During partition:**
- Each half continues operating independently with the experts it has
- If a half lacks full expert coverage → model marked `"degraded"` in that region
- On-chain payments are the source of truth (blockchain is globally consistent)
- No slashing for timeouts during suspected partition (5-min grace window after reconnection)

**After reconnection:**
- DHT merges via Kademlia's native longest-prefix reconciliation
- Reputation scores reconcile via last-write-wins (timestamped)
- Grace window prevents false slashing

**What we DON'T do:** halt the network to "stay consistent." Degraded > dead.

### 5.9 Hot/Cold Expert Replication (Immune System Model)

- Hot experts (frequently activated) auto-replicate to 8+ nodes
- Cold experts: 1-2 nodes, latent
- Activation spike → auto-replication within 60-120s
- Zero-touch: swarm self-regulates

### 5.10 Co-Activation Heat Map

The swarm learns which experts activate together and places them on the same node. This reduces cross-node communication in batch mode by 60-80%.

---

## 6. COMPUTE NODE

### 6.1 Miner Hardware (VRAM Accounting Includes Shared Params)

Every node must load shared parameters once (embeddings, attention, gate network). Expert weights are additional. All sizes in 4-bit quantization.

| Tier | VRAM | Example Config (Mixtral 8x7B) | GPUs |
|---|---|---|---|
| **Light** | 8-12GB | 1 expert (3GB) + shared (3GB) = **6GB** | RTX 3060/4060, RX 6700 XT |
| **Light+** | 12-16GB | 2 experts (6GB) + shared (3GB) = **9GB** | RTX 3060 12GB, RX 6800 |
| **Full** | 16-24GB | 4 experts (12GB) + shared (3GB) = **15GB** | RTX 3090/4080, RX 7900 XT |
| **Speculative** | ≥24GB | 8 experts (24GB) + shared (3GB) = **27GB** | RTX 4090, multi-GPU |

Light/Full tiers participate in batch mode (partial experts). Speculative tier also participates in realtime mode (full model).

Estimated income: $2-10/day net depending on tier, uptime, and model demand.

### 6.2 Zero-Decision Miner Experience

**The miner should never have to choose anything.** The system optimizes for maximum network utility automatically.

```
Miner experience:
  1. Install Synapse node software
  2. Stake USDC on L2 (one-time, small amount)
  3. Click "Start Mining"
  
  System auto-detects:
  - Available VRAM
  - GPU model and compute capability
  - Network latency and bandwidth
  
  System auto-assigns:
  - Which model to serve (based on network demand)
  - Which experts to load (based on scarcity + co-activation)
  - Price (competitive default, adjustable)
  
  Miner sees:
  - "Serving Mixtral 8x7B — Experts #3, #7 — Earning ~$0.12/hr"
  - That's it. No decisions. No config files.
```

**Power user mode (optional):** Advanced miners can manually select model, experts, and set custom pricing. But the default is fully automatic.

### 6.3 Weight Distribution

**Problem:** When a new model joins the catalog, Node #1 has no peers to download from.

**Solution: HuggingFace as eternal seed.**

1. Node downloads weights from HuggingFace (or any URL the model author provides)
2. Node verifies SHA256 hash against the catalog entry
3. Node registers in DHT: `expert://<hash>/<expert-id> → <NodeID>`
4. Node seeds weights via BitTorrent to subsequent nodes
5. Subsequent nodes download P2P from existing peers (faster, no HF bandwidth cost)

HF is the fallback forever. It costs nothing (HF hosts weights for free). Once ≥1 node has the model, P2P takes over.

### 6.4 Determinism and Verification

- **Weight hash:** SHA256 of full model weights
- **Expert hash:** SHA256 of per-expert weights (sub-hash for partial loading verification)
- **Fixed seed:** `seed=0` for batch mode (auditable)
- **Variable seed:** `seed=X` for speculative mode (diversity is desired)
- **Precision:** Model-native, enforced at protocol level

---

## 7. ECONOMIC LAYER

### 7.1 Payment Flow

```
B2B Client → fiat/stablecoin → Gateway → 15-20% fee →
  ├─ Remainder → Miners (proportional to verified tokens)
  └─ Auditor bonus → Audit nodes (batch mode only)
```

Nobody subsidizes anything. Gateway margin covers operations. Miners earn from client payments.

### 7.2 Pricing Model

**How pricing works:**

1. **Miners set ask price** (per 1M tokens) published to DHT with their expert registration
2. **Gateway is the market maker:**
   - Batch mode: selects cheapest viable expert route, publishes single model price
   - Speculative mode: selects N cheapest nodes with full model, price = N × node price
3. **Client pays the catalog price** — fixed, predictable, no surprises
4. **Miners compete:** overpriced miners get no work, drop price or exit
5. **Market self-corrects:** high prices → new miners enter → supply increases → price drops

```
Batch mode pricing walkthrough:

  Expert #3 miners:  A ($0.08)  B ($0.11)  C ($0.09)
  Expert #7 miners:  D ($0.09)  E ($0.14)  F ($0.10)

  Gateway assembles: A ($0.08) + D ($0.09) = $0.17/1M tokens cost
  Gateway publishes:  $0.25/1M tokens (client price, $0.08 margin)
  
  B, C, E, F get no work unless A or D go offline or their prices drop.
```

**What if all miners collude to raise prices?**
- Barrier to entry is zero: anyone with a GPU can mine
- New miners immediately undercut the cartel
- To maintain a cartel, you'd need 100% of expert replicas actively colluding
- With open participation, this is economically impossible

**Pricing tiers:**

| | Batch (Swarm DAG) | Realtime (Speculative Swarm) |
|---|---|---|
| Price to client | Low ($0.10-0.15/1M tokens) | Higher ($0.50-0.75/1M tokens) |
| Cost driver | Expert activations × tokens | Swarm size × tokens |
| Consensus cost | ~5% (audit overhead) | 0% (built into ensemble) |
| Scarcity effect | Under-replicated models cost more → attracts miners | More swarm nodes = higher cost but stronger consensus |

### 7.3 Slashing Mechanism (Tiered & Automatic)

**Design principle:** V1 is fully automatic. No human governance. Code is law.

| Violation | Detection | Penalty |
|---|---|---|
| Single token divergence (speculative) | Real-time consensus check | No payment for that token. Re-sync. |
| 3+ divergences in one request | Real-time, per-request counter | Expelled from request. Flag on reputation. |
| Mid-request disconnect/timeout | Gateway deadline (30s) | No payment for that request. Reputation flag. |
| 10+ flags in 24h | Automated tally | Stake frozen 48h. Cooling-off period. |
| 50+ flags in 7 days | Automated tally | Partial slashing (20% of stake). Score reset. |
| Audit failure (batch mode) | Statistical audit detection | Slashing proportional to audit divergence. |
| Sybil / fraud pattern | Heuristic detection | Full slashing + permanent ban. |

**No appeals in V1.** False positives are possible but rare. A 48h freeze on 10 flags gives the miner time to debug their setup. V2+ can add multi-sig arbitration for ambiguous cases.

### 7.4 Reputation System

- **Score (0-1000):** Composite of consensus matches, uptime, latency
- **Tiers:** Bronze (new, limited) → Silver → Gold → Platinum (priority, best rates)
- **Temporal decay:** Inactive nodes lose score
- **Scarcity bonus:** Nodes serving under-replicated experts earn bonus score

---

## 8. REQUEST LIFECYCLE

### 8.1 Batch Mode (Swarm DAG)

```
1.  Client → POST /v1/chat/completions {priority: "batch"}
2.  Gateway places request in batch queue
3.  When batch fills (or timeout):
4.    Gateway queries DHT for expert placement + pricing
5.    Gateway assembles cheapest valid expert route
6.    Gateway routes each request through the expert DAG
7.    Multiple requests flow simultaneously through different expert paths
8.    If a node fails mid-request → re-route to replica (§5.7)
9.    ~5% requests: parallel audit with different expert set
10. Gateway returns responses
11. Gateway distributes payments, updates reputation
```

### 8.2 Realtime Mode (Speculative Swarm)

```
1.  Client → POST /v1/chat/completions {priority: "realtime", swarm_size: 5}
2.  Gateway selects N cheapest nodes with full model
3.  Gateway sends prompt to ALL N nodes simultaneously
4.  Each node generates independently (different seed)
5.  If a node disconnects mid-generation → ignore, continue with N-1 (§5.7)
6.  Coordinator merges: token-level majority vote
7.  Divergent nodes: re-sync (see 5.5)
8.  Gateway returns consensus response
9.  Gateway distributes payments, flags chronic divergers
```

---

## 9. SECURITY MODEL

### 9.1 Attack Surface and Defenses

| Attack | Mode Affected | Defense |
|---|---|---|
| Single node produces garbage | Both | Ensemble voting rejects (realtime). Audit detects (batch). Score reduction → slashing. |
| N/2+1 collusion (majority attack) | Realtime | Requires compromising majority of swarm. Stake cost >> benefit. Random node selection. |
| Sybil (fake nodes) | Both | Stake per node. Reputation makes new nodes low-priority. |
| Free-riding | Both | No valid response → no payment. No consensus match → no payment + flag. |
| Prompt theft | Both | WebRTC DTLS encryption. Economic deterrence via slashing. |
| Eclipse attack | Both | DHT replication. Bootstrap diversity. |
| Price cartel | Both | Zero barrier to entry. New miners undercut. Market self-corrects. |
| Network partition | Both | Each half operates degraded. Reconnects gracefully. Grace window prevents false slashing. |
| Mid-request node failure | Both | §5.7: automatic recovery via replicas or consensus majority. |

### 9.2 Swarm Size vs Security

| Swarm Size | Nodes to Corrupt | Cost to Attack (at $500 stake/node) |
|---|---|---|
| 3 | 2/3 | $1,000 |
| 5 | 3/5 | $1,500 |
| 8 | 5/8 | $2,500 |
| 16 | 9/16 | $4,500 |

Stake amounts can be adjusted per-model based on value at risk.

---

## 10. LICENSING AND BUSINESS MODEL

| Component | License | Monetization |
|---|---|---|
| Protocol specification | CC-BY 4.0 | Free to implement |
| Reference DHT + node software | Apache 2.0 | Open source |
| Launcher | MIT | Open source |
| B2B Gateway | BSL (Business Source License) | Synapse Inc. reference gateway. 15-20% fee. |

Helium / Filecoin model: open protocol, reference commercial implementation.

---

## 11. ROADMAP

### V1 — Swarm MVP (2026 Q3-Q4)
- [ ] Kademlia DHT with expert registry, pricing, co-activation heat map
- [ ] WebRTC transport (STUN only, no TURN)
- [ ] Speculative Swarm mode (realtime ensemble + re-sync protocol)
- [ ] Swarm DAG mode (batch expert distribution)
- [ ] Mid-request failure recovery (§5.7)
- [ ] Consensus: ensemble voting + statistical audit
- [ ] Slashing: fully automatic, tiered (§7.3)
- [ ] Reputation: numeric score + basic tiers
- [ ] Gateway: redundant instances, market maker pricing
- [ ] Bootstrap: 3-5 VPS seed nodes, HF weight fallback
- [ ] Curated catalog: Synapse Inc. verified models (community PRs welcome)
- [ ] Zero-decision miner: auto-assignment of experts (§6.2)
- [ ] Models: Mixtral 8x7B, DeepSeek-V2 Lite, Qwen2.5-MoE
- [ ] USDC payments on L2 (Solana/Base)
- [ ] OpenAI-compatible gateway API
- [ ] Network partition tolerance

### V2 — Scale & Privacy (2027)
- [ ] Kimi K3 (~448 nodes)
- [ ] Split Inference privacy mode
- [ ] Full reputation system (Platinum, graduated penalties)
- [ ] Community catalog voting
- [ ] TURN server support (volunteer-run)
- [ ] Client SDKs (Python, TypeScript, Rust)
- [ ] Direct DHT query from clients (gateway bypass)

### V3 — Kimi K3 Era (2028+)
- [ ] KDA-aware DAG parallelism
- [ ] Speculative expert prediction
- [ ] Swarm auto-scaling
- [ ] Governance token / DAO for protocol upgrades

---

## 12. TECHNICAL APPENDIX: Why MoE Enables This

Dense 70B model: every token activates all 70B parameters. One GPU holds them all.

MoE 47B (Mixtral 8x7B): 12.9B active per token. The other 34B sit idle. Synapse asks: why load 47B on one GPU when 4 GPUs can hold 12B each?

As models add more experts (896 in Kimi K3), Synapse's advantage grows. Centralized providers buy bigger GPU clusters. Synapse adds more consumer nodes.

---

## 13. RISKS AND MITIGATIONS

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Swarm DAG latency too high | Medium | High | Co-activation placement. Fallback to speculative mode. |
| Insufficient miner pool | Medium | Critical | Scarcity pricing. Zero-decision launcher. Open source. |
| Collusion in speculative swarm | Low | High | Random node selection. Stake cost. Swarm size tuning. |
| Expert "hot spots" (no replicas) | Medium | Medium | Auto-replication incentive. Auto-balancing. |
| Gateway SPOF | Medium | High | Redundant instances. DNS failover. V2: client-side DHT queries. |
| False slashing | Low | Medium | 48h cooling-off period. V2: multi-sig appeals. |
| 10% miners excluded (no TURN) | Low | Low | Acceptable for V1. TURN in V2. |
| Network partition | Low | Medium | Degraded > dead. Automatic reconciliation. |
| Mid-request node failure | Medium | Low | §5.7: automatic retry/recovery. Client-transparent. |

---

## 14. OPEN QUESTIONS FOR V2+

1. Governance token for protocol DAO?
2. Multi-sig slashing appeals?
3. Public model benchmarks on the network?
4. Distributed fine-tuning (not just inference)?
5. Cross-chain L1 payment integration?
6. Split Inference: which models support this natively?

---

## 15. REFERENCES

- **Kimi K3:** Moonshot AI (2026). 2.8T MoE, 896 experts, KDA, LatentMoE. [HF: moonshotai/Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3)
- **Mixtral 8x7B:** Mistral AI. 8 experts, 2 active. [HF: mistralai/Mixtral-8x7B](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1)
- **DeepSeek-V2 Lite:** 16B MoE, 64 experts. [HF: deepseek-ai/DeepSeek-V2-Lite](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite)
- **vLLM:** High-throughput inference engine.
- **Kademlia DHT:** Maymounkov & Mazières (2002).
- **WebRTC:** W3C P2P communication standard.
- **Predictive Coding:** Rao & Ballard (1999). Cortical processing model.

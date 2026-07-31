# Research: Similar Projects — Patterns for Synapse Core V1

> **Ticket #43** — What installation, networking, model support, and observability patterns do projects like Ollama, vLLM, llama.cpp, Petals, and Ray use? What can we learn for Synapse Core V1?

**Date:** 2026-07-31
**Status:** Complete

---

## Executive Summary

We analyzed five major open-source projects that overlap with Synapse Core's goals across installation, networking, model formats, observability, GPU support, and dependency management. The key findings:

1. **Ollama proves that zero-friction installation wins adoption.** One-line install scripts, bundled binaries, and a REST API on `localhost:11434` made it the most-starred project in this space (177k stars). Synapse should match this simplicity.

2. **vLLM shows the gold standard for distributed inference.** Tensor parallelism, pipeline parallelism, expert parallelism, Ray integration for multi-node, and OpenAI-compatible API — all battle-tested at scale. Synapse should learn from its parallelism strategies for MoE.

3. **llama.cpp demonstrates the power of zero-dependency, cross-platform design.** Plain C/C++, GGUF as a universal format, 18+ hardware backends, and CPU fallback built in. This is what "usable in any place" actually means.

4. **Petals is the closest architectural analog to Synapse.** It splits models across peers (BitTorrent-style), uses DHT (hivemind/Kademlia) for discovery, and handles heterogeneous consumer GPUs. Its limitations (NAT traversal, reliability) are warnings for Synapse's P2P plans.

5. **Ray is the reference for distributed cluster management.** Multi-node orchestration, observability dashboard, metrics export, and fault tolerance. vLLM uses Ray for multi-node; Synapse should evaluate whether to integrate or build its own lightweight alternative.

---

## Detailed Project Profiles

### 1. Ollama (177k ★)

**What it is:** A local LLM runner with Docker-like UX. Wraps llama.cpp with a polished developer experience.

#### Installation
| Method | Command / Notes |
|--------|----------------|
| macOS | `curl -fsSL https://ollama.com/install.sh \| sh` or DMG download |
| Linux | `curl -fsSL https://ollama.com/install.sh \| sh` |
| Windows | `irm https://ollama.com/install.ps1 \| iex` or EXE installer |
| Docker | `docker run ollama/ollama` (official Docker Hub image) |
| Homebrew | `brew install ollama` |
| Nix | `nix-env -iA nixpkgs.ollama` |
| Arch (pacman) | `pacman -S ollama` |
| Helm Chart | Available on ArtifactHub for Kubernetes |

**Key insight:** Every major package manager is covered. The install.sh script auto-detects GPU hardware and configures accordingly. This is the adoption bar Synapse must match.

#### Networking
- **Protocol:** HTTP REST API on `localhost:11434`
- **API style:** Custom API + OpenAI-compatible `/v1/chat/completions` endpoint
- **No clustering:** Ollama is single-node by design. No built-in multi-node distribution.
- **Client libraries:** Official Python (`ollama-python`) and JavaScript (`ollama-js`) SDKs
- **Streaming:** Server-Sent Events (SSE) for token streaming

#### Model Management
- **Format:** GGUF (via llama.cpp backend). Models are pulled from `ollama.com/library` registry.
- **UX:** `ollama pull gemma4`, `ollama run gemma4` — Docker-like model management
- **Modelfile:** Dockerfile-inspired format for model configuration (system prompt, parameters, template)
- **Auto-quantization:** Models available in multiple quantization levels (Q4_K_M, Q8_0, etc.)
- **Registry:** Centralized model registry at ollama.com/library with curated, tested models

#### Observability
- **Built-in:** Minimal built-in observability (logs to stdout/stderr)
- **Community integrations:** OpenLIT (OpenTelemetry-native), Langfuse, Lunary, HoneyHive, MLflow Tracing, Opik
- **Pattern:** Ollama delegates observability to the ecosystem rather than building it in

#### GPU Support
- **NVIDIA:** CUDA (auto-detected)
- **AMD:** ROCm (auto-detected)
- **Apple Silicon:** Metal (first-class)
- **CPU fallback:** Yes, automatic when no GPU detected
- **Dependencies:** Bundled — no need to install CUDA/ROCm separately

#### Dependencies
- **Zero external dependencies** for end users — everything is bundled in the single binary
- **Internally:** Go (server), llama.cpp (inference via vendored C/C++), MLX (Apple Silicon)

---

### 2. vLLM (88k ★)

**What it is:** High-throughput LLM serving engine. The production standard for serving large models.

#### Installation
| Method | Command / Notes |
|--------|----------------|
| pip | `pip install vllm` (with CUDA extra-index-url) |
| uv (recommended) | `uv pip install vllm --torch-backend=auto` |
| Docker | `docker run --gpus all vllm/vllm-openai:latest` |
| Build from source | `git clone && uv pip install -e .` |
| Nightly | `uv pip install -U vllm --extra-index-url https://wheels.vllm.ai/nightly` |

**Key insight:** vLLM's installation is heavy — requires Python, CUDA toolkit, PyTorch. It targets datacenter operators, not end users. Ollama-style bundling is not attempted.

#### Networking
- **Protocol:** HTTP REST API (OpenAI-compatible), gRPC, Anthropic Messages API
- **Distributed inference:**
  - **Tensor parallelism:** Split model layers across GPUs on same node (NCCL)
  - **Pipeline parallelism:** Split model across nodes (NCCL + Ray or multiprocessing)
  - **Expert parallelism:** For MoE models — expert layers distributed separately
  - **Data parallelism:** Multiple model replicas with shared attention for MoE
  - **Disaggregated prefill/decode:** Separate prefill and decode stages
- **Multi-node backend:** Ray (default for multi-node) or native multiprocessing
- **Communication:** NCCL (NVIDIA), GPUDirect RDMA for InfiniBand

#### Model Formats
- **Primary:** HuggingFace format (safetensors) — seamless HF integration
- **Also supports:** GGUF, GPTQ, AWQ, FP8, MXFP8/MXFP4, NVFP4, INT8, INT4, compressed-tensors, TorchAO
- **200+ model architectures** supported
- **MoE models:** Mixtral, DeepSeek-V3, Qwen-MoE, GPT-OSS natively supported

#### Observability
- **Built-in:** Detailed logging with structured output
- **Metrics:** Prometheus metrics endpoint for throughput, latency, queue depth
- **Integration:** OpenAI-compatible API enables standard LLM observability tools
- **NCCL debugging:** `NCCL_DEBUG=TRACE` for distributed communication inspection
- **KV cache metrics:** Reports GPU KV cache size and max concurrency in logs

#### GPU Support
| Vendor | Hardware | Notes |
|--------|----------|-------|
| NVIDIA | T4, RTX20xx, A100, L4, H100, B200 | Compute capability 7.5+, first-class |
| AMD | MI200s, MI300, MI350, RX 7900, RX 9000 | ROCm 6.3+ |
| Intel | Data Center GPU, ARC GPU | SYCL backend |
| Apple | M1/M2/M3/M4 | Community plugin (vLLM-Metal via MLX) |
| Google | TPU | Plugin support |
| Others | Intel Gaudi, IBM Spyre, Huawei Ascend, Rebellions NPU | Plugin architecture |
| CPU | x86/ARM/PowerPC | Supported but not primary |

#### Dependencies
- **Heavy:** Python 3.10-3.13, PyTorch, CUDA toolkit (or ROCm), many C++ extensions
- **Linux only** natively (WSL for Windows)
- **Build requirements:** GCC ≥ 11.3, extensive compilation

---

### 3. llama.cpp (122k ★)

**What it is:** The foundational LLM inference engine in plain C/C++. Zero dependencies, maximum portability.

#### Installation
| Method | Command / Notes |
|--------|----------------|
| Web app | https://llama.app — browser-based |
| Docker | Official Docker images with GPU support |
| Pre-built binaries | GitHub Releases for macOS, Linux, Windows |
| Build from source | `cmake -B build && cmake --build build` |
| Homebrew | `brew install llama.cpp` |
| Nix | Available in nixpkgs |
| Android | Build guide provided |
| iOS | XCFramework build support |

**Key insight:** llama.cpp is the gold standard for portability. Plain C/C++ means it compiles everywhere. Ollama, LM Studio, and dozens of other tools use it as their backend.

#### Networking
- **Built-in server:** `llama serve` — OpenAI-compatible REST API
- **RPC backend:** Distributed inference via RPC — offload computation to remote machines
- **No clustering:** Single-process, single-model. Distribution handled at the application layer (Ollama, etc.)
- **Protocol:** HTTP for API, custom RPC for remote backend operations

#### Model Formats
- **GGUF (primary):** The universal LLM format. Single-file, self-contained, metadata embedded.
  - Created by llama.cpp, now adopted by entire ecosystem
  - Supports 1.5-bit through 8-bit integer quantization
  - CPU-optimized, memory-mapped loading
- **Conversion:** `convert_hf_to_gguf.py` — converts HuggingFace models to GGUF
- **LoRA:** GGUF format for adapters via `convert_lora_to_gguf.py`
- **No external model registry needed** — direct HuggingFace download with `-hf` flag

#### Observability
- **Built-in web UI:** Visual interface for `llama serve`
- **Logging:** Structured logs for request handling, token generation, performance
- **Benchmarks:** Built-in benchmarking tools (`llama-bench`)
- **Performance tips:** Documented token generation performance troubleshooting
- **Minimal:** Focused on inference correctness, not operational metrics

#### GPU Support (18 backends!)
| Backend | Devices |
|---------|---------|
| CUDA | NVIDIA GPUs |
| HIP | AMD GPUs |
| Metal | Apple Silicon |
| Vulkan | Broad GPU support |
| SYCL | Intel GPUs |
| MUSA | Moore Threads GPUs |
| CANN | Ascend NPUs |
| OpenCL | Adreno GPUs |
| OpenVINO | Intel CPUs, GPUs, NPUs |
| WebGPU | All (browser) |
| RPC | All (remote) |
| BLAS | All (CPU) |
| VirtGPU | VirtGPU APIR |
| Hexagon | Snapdragon (in progress) |
| zDNN | IBM Z & LinuxONE |
| ZenDNN | AMD CPUs |

**CPU+GPU hybrid:** Can partially accelerate models larger than VRAM by running some layers on CPU.

#### Dependencies
- **Zero external dependencies** — plain C/C++ with vendored libraries
- **Optional:** CUDA toolkit for NVIDIA, ROCm for AMD, Metal for Apple
- **Build tools only:** CMake, C/C++ compiler (GCC, Clang, MSVC)

---

### 4. Petals (10.5k ★)

**What it is:** BitTorrent-style distributed LLM inference. The closest architectural analog to Synapse's distributed MoE vision.

#### Installation
| Method | Command / Notes |
|--------|----------------|
| pip (client) | `pip install petals` |
| pip (server) | `pip install git+https://github.com/bigscience-workshop/petals` |
| Docker | `docker run learningathome/petals:main` |
| Conda + pip | Standard PyTorch + pip install |
| Google Colab | One-click notebook |
| macOS | `brew install python && pip install petals` |

**Key insight:** Simple pip install, but heavy PyTorch dependency. Server requires GPU with specific VRAM minimums.

#### Networking — The Most Relevant for Synapse
- **Architecture:** Peer-to-peer (BitTorrent-style)
  - Each server hosts a subset of model layers (blocks)
  - Clients request inference from a chain of servers
  - Pipeline parallelism across the Internet
- **Discovery:** [Hivemind](https://github.com/learning-at-home/hivemind) library
  - **DHT (Kademlia):** Distributed hash table for peer discovery and model metadata
  - **NAT traversal:** Uses hivemind's P2P networking (libp2p-based)
  - **Health monitoring:** Public swarm monitor at health.petals.dev
- **Protocol:** Custom P2P protocol over TCP (via hivemind)
- **Private swarms:** Supported — for trusted networks without public exposure
- **Security model:** Hosting a server does NOT allow code execution on your machine
- **Throughput:** Up to 6 tok/s for Llama 2 (70B), 4 tok/s for Falcon (180B)

**Key insight for Synapse:** Petals proves that P2P inference over the Internet works, but with significant latency tradeoffs. For Synapse's "Network DAG" mode, the hivemind DHT approach is directly relevant.

#### Model Formats
- **HuggingFace format:** Direct loading from HF Hub (safetensors/PyTorch)
- **Block-based splitting:** Models split into transformer blocks, each hosted independently
- **Supported models:** Llama 3.1 (405B), Mixtral (8x22B), Falcon (40B+), BLOOM (176B)
- **No GGUF:** Works with standard HF weights, not quantized formats

#### Observability
- **Swarm monitor:** Public dashboard at health.petals.dev
  - Shows online servers, their blocks, throughput
  - Contributor leaderboard
- **Logging:** Standard Python logging
- **No metrics export:** No Prometheus/OpenTelemetry integration

#### GPU Support
- **NVIDIA:** Primary target (CUDA via PyTorch)
- **AMD:** Supported via ROCm (documented guide)
- **Apple Silicon:** Supported (M1/M2 via MPS)
- **CPU fallback:** Not practical — model blocks need GPU for reasonable performance

#### Dependencies
- **Heavy:** PyTorch, transformers, hivemind (P2P library)
- **Python 3.8+**
- **Hivemind:** Handles DHT, NAT traversal, gradient compression

---

### 5. Ray (35k+ ★)

**What it is:** Distributed computing framework for Python. Powers vLLM's multi-node inference and many production ML systems.

#### Installation
| Method | Command / Notes |
|--------|----------------|
| pip | `pip install ray` |
| pip (with extras) | `pip install "ray[cgraph]"` for collective graphs |
| Docker | `rayproject/ray` images on Docker Hub |
| KubeRay | Kubernetes operator for Ray clusters |
| Helm | Ray cluster Helm charts |

#### Networking
- **Cluster architecture:**
  - **Head node:** Runs GCS (Global Control Store), driver, dashboard
  - **Worker nodes:** Run tasks/actors, connect to head via GCS
  - **Autoscaler:** Can add/remove nodes based on demand
- **Communication:** gRPC for control plane, shared memory / NCCL for data plane
- **Multi-node setup:** `ray start --head` on leader, `ray start --address=<head_ip>` on workers
- **Kubernetes:** KubeRay operator for production deployments
- **Security:** No built-in encryption — must use private network or overlay

#### Observability (Gold Standard)
| Tool | Description |
|------|-------------|
| **Ray Dashboard** | Web UI: jobs, tasks, actors, nodes, logs, flame graphs |
| **Ray Metrics** | Prometheus-compatible metrics export |
| **Ray Logs** | Centralized log aggregation across cluster |
| **Ray State API** | CLI and REST API for cluster state inspection |
| **OpenTelemetry** | Integration for distributed tracing |
| **Grafana** | Pre-built dashboards for Ray metrics |
| **Task/Actor profiling** | CPU/GPU flame graphs per task |
| **Events** | Structured event system for lifecycle events |

**Key insight:** Ray's observability is the gold standard. For Synapse's distributed architecture, we should adopt a similar approach: built-in metrics export (Prometheus), structured logging, and a status dashboard.

#### GPU Support
- Ray is GPU-aware — tracks GPU resources per node
- `@ray.remote(num_gpus=1)` for GPU task scheduling
- Integrates with CUDA, ROCm via underlying frameworks (PyTorch, etc.)
- No direct GPU inference — delegates to frameworks like vLLM

#### Dependencies
- **Python 3.9+**
- **Optional:** Redis (for larger clusters), various cloud SDKs for autoscaling
- **Relatively lightweight** core, but grows with extras

---

## Comparison Matrix

### Installation

| Dimension | Ollama | vLLM | llama.cpp | Petals | Ray |
|-----------|--------|------|-----------|--------|-----|
| One-line install | ✅ `curl \| sh` | ❌ Complex | ❌ Build from source | ❌ pip + PyTorch | ✅ `pip install ray` |
| Docker | ✅ | ✅ | ✅ | ✅ | ✅ |
| Package managers | brew, apt, pacman, nix, helm | pip, uv | brew, nix, winget | pip | pip |
| Binary distribution | ✅ Single binary | ❌ Python wheel | ✅ Pre-built binaries | ❌ pip only | ❌ pip only |
| Auto GPU detection | ✅ | ❌ Manual | ✅ (build-time) | ❌ Manual | N/A |
| Zero dependencies | ✅ | ❌ Heavy | ✅ | ❌ Heavy | ❌ Moderate |

### Networking

| Dimension | Ollama | vLLM | llama.cpp | Petals | Ray |
|-----------|--------|------|-----------|--------|-----|
| Protocol | HTTP REST | HTTP/gRPC | HTTP + RPC | P2P (hivemind) | gRPC + SHM |
| Multi-node | ❌ | ✅ (Ray/mp) | ✅ (RPC backend) | ✅ (P2P) | ✅ (native) |
| P2P | ❌ | ❌ | ❌ | ✅ (DHT) | ❌ |
| OpenAI-compatible API | ✅ | ✅ | ✅ | ❌ (HF-style) | N/A |
| Streaming | ✅ SSE | ✅ SSE | ✅ SSE | ✅ | N/A |
| NAT traversal | N/A | ❌ | ❌ | ✅ (hivemind) | ❌ |
| Discovery | N/A | Manual/Ray | Manual | DHT (Kademlia) | GCS |

### Model Formats

| Dimension | Ollama | vLLM | llama.cpp | Petals | Ray |
|-----------|--------|------|-----------|--------|-----|
| GGUF | ✅ (primary) | ✅ (limited) | ✅ (created it) | ❌ | N/A |
| Safetensors/HF | ❌ | ✅ (primary) | Via conversion | ✅ (primary) | N/A |
| GPTQ/AWQ | ❌ | ✅ | ❌ | ❌ | N/A |
| FP8/INT4/INT8 | Via quantized GGUF | ✅ | ✅ (1.5-8 bit) | ❌ | N/A |
| Model registry | ollama.com | HuggingFace | HuggingFace | HuggingFace | N/A |
| MoE support | Via llama.cpp | ✅ Native expert parallel | Partial | ✅ Native | N/A |

### Observability

| Dimension | Ollama | vLLM | llama.cpp | Petals | Ray |
|-----------|--------|------|-----------|--------|-----|
| Built-in metrics | ❌ | ✅ Prometheus | ❌ | ❌ | ✅ Prometheus |
| Dashboard | ❌ | ❌ (external) | ✅ (basic web UI) | ✅ (health monitor) | ✅ (full dashboard) |
| Structured logging | stdout | ✅ | ✅ | Python logging | ✅ Centralized |
| Distributed tracing | ❌ (community) | ❌ | ❌ | ❌ | ✅ OpenTelemetry |
| Token metrics | ❌ | ✅ (throughput, latency) | ✅ (bench tools) | ❌ | N/A |
| Cluster status | N/A | Via Ray | N/A | ✅ (health.petals.dev) | ✅ (full state API) |

### GPU & Hardware

| Dimension | Ollama | vLLM | llama.cpp | Petals | Ray |
|-----------|--------|------|-----------|--------|-----|
| NVIDIA | ✅ | ✅ (cc ≥ 7.5) | ✅ | ✅ | ✅ (aware) |
| AMD | ✅ | ✅ (ROCm 6.3+) | ✅ (HIP) | ✅ (ROCm) | ✅ (aware) |
| Apple Silicon | ✅ (Metal) | ✅ (Metal plugin) | ✅ (Metal, 1st class) | ✅ (MPS) | N/A |
| Intel GPU | ❌ | ✅ (SYCL) | ✅ (SYCL) | ❌ | N/A |
| CPU fallback | ✅ Auto | ✅ | ✅ (native, fast) | ❌ | N/A |
| CPU+GPU hybrid | ❌ | ❌ | ✅ | ❌ | N/A |
| Vulkan | ❌ | ❌ | ✅ | ❌ | N/A |
| WebGPU | ❌ | ❌ | ✅ | ❌ | N/A |

---

## Recommendations for Synapse Core V1

### ADOPT — Patterns We Should Copy

#### 1. **Ollama-style Installation UX** ⭐ HIGH PRIORITY
```
curl -fsSL https://synapse.sh/install.sh | sh
```
- Single-line install script that auto-detects GPU, downloads the right binary, configures defaults
- Docker as first-class: `docker run synapse/synapse`
- Package managers: brew, apt, cargo install, nix, helm
- **Why:** Ollama has 177k stars largely because of installation simplicity. Synapse's "usable in any place" mandate requires this.

#### 2. **GGUF as Primary Model Format** ⭐ HIGH PRIORITY
- GGUF is already used in Synapse's native MoE forward pass — lean into it
- Single-file, self-contained, memory-mappable, quantization-aware
- llama.cpp ecosystem compatibility (conversion tools, quantization tools, model hub)
- **Why:** It's the universal format for local inference. Every consumer hardware project uses it. Safetensors should be a conversion source, not the primary format.

#### 3. **OpenAI-Compatible API** ⭐ HIGH PRIORITY
- Ollama, vLLM, and llama.cpp all converge on the same API
- Synapse already has `/v1/jobs` — extend to include `/v1/chat/completions` and `/v1/completions`
- **Why:** Ecosystem compatibility. Every LLM tool, SDK, and framework already speaks this protocol.

#### 4. **Ray-style Observability from Day One**
- Prometheus metrics endpoint (tokens/sec, latency p50/p95/p99, expert dispatch time, worker health)
- Structured JSON logging with request tracing
- Simple cluster status endpoint (which workers are online, what experts they hold)
- **Why:** Distributed systems without observability are impossible to debug. Synapse's architecture (coordinator + workers) makes this especially critical.

#### 5. **llama.cpp's Zero-Dependency Philosophy**
- Synapse is already Rust — good. Keep external dependencies minimal.
- Bundle what can be bundled. Ship pre-built binaries for all major platforms.
- CPU fallback should work out of the box (even if slower)
- **Why:** "Usable in any place" means no "install CUDA toolkit first" requirements for basic functionality.

#### 6. **Petals's DHT Discovery Model (adapted)**
- Synapse's `dht/` module already has Kademlia — this is the right direction
- For LAN: mDNS/peer discovery (zero configuration)
- For WAN: DHT with optional bootstrap nodes
- **Why:** Users should be able to `synapse start` on two machines and have them find each other automatically on a LAN.

#### 7. **vLLM's Expert Parallelism Strategy**
- Data Parallel attention + Expert/Tensor Parallel MoE layers
- This maps directly to Synapse's architecture: coordinator runs attention, workers hold expert shards
- **Why:** vLLM validated this pattern for MoE at production scale. Synapse's V0 already implements a simpler version of this.

### AVOID — Anti-Patterns to Reject

#### 1. **vLLM's Heavy Installation Footprint**
- Don't require Python, CUDA toolkit, or PyTorch as prerequisites for the core binary
- Don't require compilation from source for end users
- **Instead:** Ship pre-built Rust binaries. Use the inference runtime (Python/vLLM) as an optional backend, not a requirement.

#### 2. **Petals's Reliability Issues from Open P2P**
- Pure P2P over the Internet has fundamental reliability problems (NAT, churn, latency variance)
- Petals achieves only 4-6 tokens/sec — too slow for interactive use
- **Instead:** Focus on LAN/trusted network clusters for V1. P2P over Internet is a V2+ feature. Use WebRTC for NAT traversal when the time comes (Synapse already has `transport/` with WebRTC scaffolding).

#### 3. **vLLM's Linux-Only Restriction**
- vLLM requires Linux natively (WSL for Windows)
- **Instead:** Follow llama.cpp's approach — cross-platform from day one. Synapse (Rust) naturally supports this.

#### 4. **Petals's No-Observability Approach**
- Petals has only a basic health dashboard, no metrics export
- **Instead:** Build observability into the protocol, not as an afterthought. Every worker should report metrics. The coordinator should aggregate and expose them.

#### 5. **Ray's Complexity for Simple Deployments**
- Ray requires head node setup, cluster configuration, autoscaler tuning
- **Instead:** For Synapse V1, aim for "zero-config LAN cluster" — workers announce themselves, coordinator discovers them. No explicit cluster setup needed for the common case.

### CONSIDER — Evaluate for V1/V2

#### 1. **Ollama Registry Pattern**
- A curated model registry at `synapse.sh/models` with pre-sharded MoE models
- Users could `synapse pull mixtral-8x7b` and get optimally-sharded GGUF experts
- **V2 consideration** — V1 should work with local GGUF files first

#### 2. **vLLM's Kubernetes/Helm Deployment**
- Helm chart for production multi-node deployment
- KubeRay-style operator for Synapse clusters
- **V2 consideration** — V1 focuses on bare-metal/Docker

#### 3. **Hybrid CPU+GPU (llama.cpp pattern)**
- Run some expert shards on CPU when VRAM is limited
- llama.cpp proves this works for models larger than VRAM
- **V1 consideration** — Could significantly expand the "usable in any place" story

#### 4. **gRPC for Worker Communication**
- vLLM and Ray both use gRPC for high-performance inter-node communication
- Synapse currently uses HTTP between coordinator and workers
- **V1 consideration** — HTTP is simpler but gRPC could improve expert dispatch latency

---

## Synthesis: What "Usable in Any Place" Means

Based on this analysis, "usable in any place" for Synapse Core V1 means:

| Requirement | Learned From | Implementation |
|-------------|-------------|----------------|
| **Install anywhere** | Ollama | One-line install, Docker, package managers, pre-built binaries |
| **Run on any GPU** | llama.cpp | NVIDIA + AMD + Apple Silicon + CPU fallback, auto-detect |
| **Use any model** | llama.cpp + vLLM | GGUF primary, HF conversion, MoE-native |
| **Cluster easily** | Petals + Ray | LAN auto-discovery (mDNS), zero-config for 2-4 nodes |
| **See what's happening** | Ray | Prometheus metrics, structured logging, health dashboard |
| **API compatibility** | Ollama + vLLM | OpenAI-compatible `/v1/chat/completions` |
| **Trust the output** | V0 proven | Identical logits to monolithic (cosine sim 1.000000) |

### Priority Stack for V1

1. **Installation UX** — Must be as simple as Ollama. This is the #1 adoption driver.
2. **GGUF + OpenAI API** — Ecosystem compatibility. Users bring their own models and tools.
3. **LAN auto-discovery** — Zero-config clustering for the common case (2-4 nodes on a network).
4. **Prometheus metrics** — Every coordinator and worker exposes metrics. Non-negotiable for distributed systems.
5. **Cross-platform binaries** — macOS, Linux, Windows. Pre-built, no compilation.
6. **CPU fallback** — Synapse should degrade gracefully, not fail, when no GPU is available.

---

## Sources

- [Ollama GitHub](https://github.com/ollama/ollama) — 177k stars, MIT license
- [vLLM GitHub](https://github.com/vllm-project/vllm) — 88k stars, Apache 2.0 license
- [vLLM Parallelism & Scaling Docs](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/)
- [vLLM Installation Docs](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/)
- [llama.cpp GitHub](https://github.com/ggml-org/llama.cpp) — 122k stars, MIT license
- [Petals GitHub](https://github.com/bigscience-workshop/petals) — 10.5k stars, MIT license
- [Petals Website](https://petals.dev/)
- [Petals Paper: Collaborative Inference (ACL 2023)](https://arxiv.org/abs/2209.01188)
- [Petals Paper: Distributed Inference over Internet (NeurIPS 2023)](https://arxiv.org/abs/2312.08361)
- [Ray Observability Docs](https://docs.ray.io/en/latest/ray-observability/index.html)
- [Synapse README](../../README.md) — V0 status, architecture, benchmarks

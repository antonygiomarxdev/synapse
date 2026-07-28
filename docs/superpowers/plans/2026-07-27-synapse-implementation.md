# Synapse Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Synapse reference implementation — a P2P swarm protocol for MoE model inference — in 6 phases, producing a working V1 MVP.

**Architecture:** Kademlia DHT for node/expert discovery, WebRTC for tensor transport, vLLM for inference runtime, FastAPI for the B2B gateway, Solana/Base L2 for USDC payments. Two swarm modes: Speculative (realtime ensemble) and DAG (batch expert distribution).

**Tech Stack:** Python 3.12+, vLLM, aiortc (WebRTC), protobuf, FastAPI, web3.py, pytest, Solidity (stake contract)

## Global Constraints

- All node communication via WebRTC (DTLS encrypted). No plaintext transport.
- All models identified by SHA256 weight hash. Catalog verifies hashes before listing.
- Inference seed=0 for batch mode (auditable). Variable seed for speculative mode.
- Gateway fee: 15-20%. Miners compete on ask price.
- V1: no TURN servers (STUN only). ~10% miner exclusion acceptable.
- Catalog: Synapse Inc. curated. Community PR welcome.
- Miner UX: zero-decision default (auto-assign experts). Power user mode optional.
- Bootstrap: 3-5 VPS seed nodes, HuggingFace as weight fallback.
- Slashing: fully automatic. No human governance in V1.

---

# Synapse

> Decentralized inference protocol for MoE models.  
> Any model, served by a swarm of consumer GPUs. No datacenter. No gatekeeper.

## Architecture

```
synapse/
├── synapse-core/         # Rust — P2P core, DHT, domain logic
├── synapse-runtime/      # Python — vLLM adapter, weight loader
├── synapse-gateway/      # Python — FastAPI, B2B API
├── contracts/            # Solidity — stake + slashing
├── config/               # Model catalog, node defaults
└── docs/                 # Specs, plans
```

## Quick Start

### Rust Core

```bash
cargo build --release
cargo test
```

### Python Runtime

```bash
pip install -e synapse-runtime
```

### Gateway

```bash
pip install -e synapse-gateway
uvicorn synapse_gateway.api:app --reload
```

### Smart Contracts

```bash
cd contracts/stake
npx hardhat compile
npx hardhat test
```

## License

Apache 2.0 — see [LICENSE](LICENSE)

## Docs

- [Design Spec](docs/superpowers/specs/2026-07-27-synapse-design.md)
- [Implementation Plan](docs/superpowers/plans/2026-07-27-synapse-implementation.md)

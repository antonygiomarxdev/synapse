# Contributing to Synapse

Thanks for your interest in contributing! Synapse is an open protocol — we welcome all forms of contribution.

## Code of Conduct

Be respectful. Be constructive. Assume good intent. This is an open-source community project.

## How to Contribute

### Reporting Bugs

Open an issue with:
- What you expected to happen
- What actually happened
- Steps to reproduce
- Environment (OS, GPU, Rust/Python version, `cargo --version`, `python --version`)

### Proposing Features

Open an issue labeled `enhancement`. Describe the problem you're solving, not just the solution. Reference the [design spec](docs/superpowers/specs/2026-07-27-synapse-design.md) if relevant.

### Proposing a Model for the Catalog

1. Fork the repo
2. Add your model to `config/models.toml`:
   ```toml
   [[models]]
   id = "your-model-id"
   name = "Display Name"
   hf_repo = "org/model-name"
   sha256 = "<verified hash>"
   experts = 64
   active_per_token = 8
   expert_size_gb = 0.5
   shared_params_gb = 2.0
   context = 32768
   license = "MIT"
   ```
3. Verify the SHA256 hash matches the official HuggingFace checkpoint
4. Open a PR with `catalog:` prefix in the title
5. The Synapse maintainers will verify and merge

Requirements for catalog models:
- Must be open-weight with a commercially usable license
- Must be Mixture-of-Experts (MoE) architecture
- Must support deterministic inference (fixed seed produces identical outputs)
- SHA256 must be verified against the official source

### Contributing Code

1. **Find an issue.** Check the [open issues](https://github.com/antonygiomarxdev/synapse/issues) or propose your own.
2. **Fork and branch.** Create a branch with a descriptive name: `feat/consensus-voting`, `fix/reputation-overflow`.
3. **Follow conventions:**
   - Rust: `cargo fmt`, `cargo clippy -- -D warnings`, TDD
   - Python: `ruff format`, `ruff check`, `pytest`
   - Solidity: `npx hardhat compile`, `npx hardhat test`
4. **Write tests.** All new code must have test coverage. Domain logic should be pure and fully tested.
5. **Open a PR.** Reference the issue you're solving. Keep PRs focused — one concern per PR.

### Development Setup

```bash
# Clone
git clone https://github.com/antonygiomarxdev/synapse.git
cd synapse

# Rust
cargo build
cargo test
cargo clippy -- -D warnings

# Python
cd synapse-runtime && pip install -e ".[dev]"
cd ../synapse-gateway && pip install -e ".[dev]"

# Solidity
cd contracts/stake && npm install && npx hardhat test
```

### Commit Conventions

Follow [Conventional Commits](https://www.conventionalcommits.org/):
```
feat: add token-level consensus voting
fix: prevent reputation overflow at boundary
test: add property tests for pricing route assembly
docs: update model catalog with Kimi K2.7
chore: bump libp2p to 0.56
```

### PR Review Process

1. All PRs require at least one approving review
2. CI must pass (`cargo test`, `cargo clippy`, `pytest`, `hardhat test`)
3. Maintainers may request changes for design alignment with the spec
4. Once approved, squash-merge to `main`

## Project Structure

```
synapse-core/src/    → Rust. P2P core. Domain logic is pure, I/O in infrastructure layer.
synapse-runtime/     → Python. vLLM adapter. Bridge to Rust core via Unix socket.
synapse-gateway/     → Python. FastAPI. B2B endpoints.
contracts/           → Solidity. Deployed on L2.
config/              → Model catalog, node defaults.
docs/superpowers/    → Design spec, implementation plan.
```

## Domain-Driven Design

We follow DDD with Clean Architecture:
- **Domain layer** (no deps): pure value objects, entities, aggregates, domain services
- **Application layer** (domain deps only): use cases, ports (traits)
- **Infrastructure layer** (domain + app deps): adapters (libp2p, vLLM, L2)

This means domain logic is fully testable without network, database, or GPU access.

## Questions?

Open a [discussion](https://github.com/antonygiomarxdev/synapse/discussions) or comment on a relevant issue.

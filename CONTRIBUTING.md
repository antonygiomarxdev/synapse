# Contributing to Synapse

Thanks for your interest in contributing! Synapse is distributed inference infrastructure for Mixture-of-Experts models — we welcome all forms of contribution.

## What's easy to get merged

| Type | Difficulty | Notes |
|---|---|---|
| **Bug reports** | Low | Clear reproduction steps = fast fix |
| **Model catalog additions** | Low | Add a MoE model to `config/models.toml` (see below) |
| **Documentation** | Low | Typos, clarifications, examples |
| **Tests** | Low | Coverage gaps, property tests, edge cases |
| **Performance** | Medium | Inference speed, memory usage — benchmarks required |
| **New features** | Medium-High | Open an issue first to discuss design alignment |
| **Architecture changes** | High | Must reference design spec, needs maintainer discussion |

## Good first issues

Look for issues labeled [`good first issue`](https://github.com/antonygiomarxdev/synapse/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22). These are scoped to be solvable in 1-2 days and provide a good introduction to the codebase.

If you're new, we recommend:
1. Start with a bug fix or test addition — lowest barrier
2. Read the [design spec](docs/superpowers/specs/2026-07-27-synapse-design.md) for context on where the project is headed
3. Join the discussion on an issue before starting work

## How to contribute

### Reporting bugs

Open an issue using the **Bug Report** template. Include:
- What you expected to happen
- What actually happened
- Steps to reproduce
- Environment (OS, GPU, Rust/Python version)

### Proposing a model for the catalog

MoE models only. See existing entries in `config/models.toml` for the format.

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

Requirements: open-weight, commercially usable license, MoE architecture, verified SHA256.

### Contributing code

1. **Find or open an issue.** Check [open issues](https://github.com/antonygiomarxdev/synapse/issues) or propose your own. Features should begin with an issue, not a PR.
2. **Fork and branch.** Descriptive names: `feat/consensus-voting`, `fix/reputation-overflow`.
3. **Write the failing test first.** We follow TDD — red, green, refactor.
4. **Follow conventions:**
   - Rust: `cargo fmt`, `cargo clippy -- -D warnings`
   - Python: `ruff format`, `ruff check`, `pytest`
   - Solidity: `npx hardhat compile`, `npx hardhat test`
5. **Keep PRs focused.** One concern per PR. New contributors: limit to 1 open PR.
6. **Run the gauntlet locally** before opening a PR:
   ```bash
   make gauntlet   # fmt, lint, test, coverage, mutants, audit, BDD
   ```

### Development setup

```bash
git clone https://github.com/antonygiomarxdev/synapse.git
cd synapse

# Rust
cargo build
cargo test
cargo clippy -- -D warnings

# Python
cd synapse-runtime && pip install -e ".[dev]"

# Solidity
cd contracts/stake && npm install && npx hardhat test
```

## Commit conventions

Follow [Conventional Commits](https://www.conventionalcommits.org/):
```
feat: add token-level consensus voting
fix: prevent reputation overflow at boundary
test: add property tests for pricing route assembly
docs: update model catalog with Kimi K3
chore: bump libp2p to 0.56
```

## PR review process

1. All PRs require at least one approving review
2. CI must pass (`make gauntlet`)
3. Maintainers may request changes for design alignment with the spec
4. Once approved, squash-merge to `main`
5. Expect a status update within 3 days. If you haven't heard back, ping the issue.

## AI-generated code policy

AI-generated code (Copilot, Claude, etc.) is allowed. You are responsible for every line.

- **Disclose** how AI was used in the PR description
- **Review** every line yourself before submitting
- **Explain** any line when asked by a reviewer
- Do not use AI to write bug reports, feature requests, or discussion messages

## Architecture overview

We follow DDD with Clean Architecture:

```
synapse-core/src/    → Rust. Single crate, single binary. Domain logic is pure.
synapse-runtime/     → Python. vLLM adapter via Unix socket + protobuf.
contracts/stake/     → Solidity. Staking on L2.
config/              → Model catalog (models.toml), node defaults.
docs/superpowers/    → Design spec, implementation plans.
```

**Dependencies point inward:** Presentation (axum) → Ports (traits) → Infrastructure (adapters) → Domain. Domain never imports infrastructure.

## Questions?

Open an issue or comment on a relevant one. For general questions, use the [`discussion` label](https://github.com/antonygiomarxdev/synapse/labels).

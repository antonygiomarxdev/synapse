.PHONY: all build test lint lint-fix audit clean dev

all: build

# ─── Rust ───────────────────────────────────────────────────

SRC = synapse-core

build:
	cargo build --release

test:
	cargo test

test-gateway:
	cargo test --test '*' gateway::

lint:
	cargo fmt --check
	cargo clippy -- -D warnings

lint-fix:
	cargo fmt
	cargo clippy --fix --allow-dirty --allow-staged

audit:
	cargo deny check
	cargo audit

# ─── Python (vLLM Runtime) ─────────────────────────────────

runtime-dev:
	cd synapse-runtime && pip install -e ".[dev]"

runtime-lint:
	cd synapse-runtime && ruff check .
	cd synapse-runtime && ruff format --check .

runtime-lint-fix:
	cd synapse-runtime && ruff check --fix .
	cd synapse-runtime && ruff format .

runtime-test:
	cd synapse-runtime && python -m pytest tests/ -v

runtime-audit:
	cd synapse-runtime && pip-audit

# ─── Solidity ───────────────────────────────────────────────

contracts-test:
	cd contracts/stake && npx hardhat test

contracts-lint:
	cd contracts/stake && npx solhint 'src/**/*.sol'

# ─── Full Suite ─────────────────────────────────────────────

test-all: test runtime-test contracts-test

lint-all: lint runtime-lint contracts-lint

audit-all: audit runtime-audit

# ─── Dev Server ─────────────────────────────────────────────

dev:
	cargo run --release

# ─── Clean ──────────────────────────────────────────────────

clean:
	cargo clean
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true

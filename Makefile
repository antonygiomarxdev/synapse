.PHONY: all build test lint lint-fix audit clean dev gauntlet

all: build

# ─── Rust ───────────────────────────────────────────────────

build:
	cargo build --release

test:
	cargo test

test-coverage:
	cargo llvm-cov --fail-under-lines 80 --fail-under-functions 80

test-mutants:
	cargo mutants -- --workspace

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

# ─── The Quality Gauntlet ───────────────────────────────────
# Every PR must pass this before merge.
# Idea: Uncle Bob — surround agents with extreme constraints.

gauntlet: lint test test-coverage audit runtime-lint contracts-test contracts-lint
	@echo "============================================"
	@echo "  GAUNTLET PASSED — code is ready to merge"
	@echo "============================================"

# ─── Full Suite ─────────────────────────────────────────────

test-all: test runtime-test contracts-test

lint-all: lint runtime-lint contracts-lint

audit-all: audit runtime-audit

# ─── Dev ────────────────────────────────────────────────────

dev:
	cargo run --release

# ─── Clean ──────────────────────────────────────────────────

clean:
	cargo clean
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true

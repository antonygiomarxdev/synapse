.PHONY: all build test lint clean

all: build

# ─── Rust Core ───────────────────────────────────────────────

build:
	cargo build --release

test:
	cargo test

lint:
	cargo clippy -- -D warnings
	cargo fmt --check

bench:
	cargo bench

# ─── Python Runtime ──────────────────────────────────────────

runtime-dev:
	cd synapse-runtime && pip install -e ".[dev]"

test-runtime:
	cd synapse-runtime && python -m pytest tests/ -v

# ─── Python Gateway ──────────────────────────────────────────

gateway-dev:
	cd synapse-gateway && pip install -e ".[dev]"

test-gateway:
	cd synapse-gateway && python -m pytest tests/ -v

# ─── Contracts ───────────────────────────────────────────────

contracts-test:
	cd contracts/stake && npx hardhat test

# ─── Full Suite ──────────────────────────────────────────────

test-all: test test-runtime test-gateway contracts-test

# ─── Clean ───────────────────────────────────────────────────

clean:
	cargo clean
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true

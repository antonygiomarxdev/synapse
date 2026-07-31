#!/bin/bash
# V0 Benchmark — runs 3 inference scenarios against Ollama
# Usage: ./scripts/bench.sh

set -euo pipefail

echo "=== V0 Benchmark ==="
echo ""

# Build
echo "Building..."
cargo build --release --bin bench_inference

# Run
echo ""
cargo run --release --bin bench_inference

# Show report
DATE=$(date +%Y-%m-%d)
REPORT="docs/benchmarks/v0-${DATE}.md"

if [ -f "$REPORT" ]; then
    echo ""
    echo "=== Report ==="
    cat "$REPORT"
fi

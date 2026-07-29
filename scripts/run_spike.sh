#!/usr/bin/env bash
# ─── Synapse Spike: vLLM Viability Test Runner ───────────────────────────────
#
# Usage:
#   ./scripts/run_spike.sh [--test=TYPE] [--model=MODEL_ID] [--workers=N] [--prompts=N]
#
# Examples:
#   ./scripts/run_spike.sh --test=smoke
#   ./scripts/run_spike.sh --test=sequential --prompts=50
#   ./scripts/run_spike.sh --test=multi --workers=2
#   ./scripts/run_spike.sh --test=crash --model=Qwen/Qwen1.5-MoE-A2.7B
#
# Prerequisites:
#   - Rust toolchain (cargo)
#   - Python 3.12+ with vllm installed (or mock fallback for protocol testing)
#   - protoc (for generating Python protobuf stubs)
#
# The script:
#   1. Generates Python protobuf stubs from proto/spike.proto
#   2. Compiles the Rust spike binary
#   3. Runs the selected test
#   4. Reports results
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "═══ Synapse Spike Runner ═══"
echo "Project dir: $PROJECT_DIR"

# ── 1. Generate Python protobuf stubs ────────────────────────────────────────
echo ""
echo "─── Generating Python protobuf stubs ───"
PYTHON_PROTO_DIR="$PROJECT_DIR/synapse-runtime/synapse_runtime"

if command -v protoc &>/dev/null; then
    protoc \
        --python_out="$PYTHON_PROTO_DIR" \
        --proto_path="$PROJECT_DIR/proto" \
        "$PROJECT_DIR/proto/spike.proto"
    echo "✓ protobuf stubs generated: $PYTHON_PROTO_DIR/spike_pb2.py"
else
    echo "✗ protoc not found. Install it: apt install protobuf-compiler"
    echo "  Or: pip install grpcio-tools"
    exit 1
fi

# ── 2. Install Python dependencies (if needed) ──────────────────────────────
echo ""
echo "─── Checking Python environment ───"
if ! python3 -c "import synapse_runtime.spike_pb2" 2>/dev/null; then
    echo "Installing synapse-runtime in development mode..."
    pip install -e "$PROJECT_DIR/synapse-runtime" --quiet
fi

if python3 -c "from vllm import LLM" 2>/dev/null; then
    echo "✓ vLLM available — will use real GPU inference"
else
    echo "⚠ vLLM not installed — will use mock engine for protocol testing"
    echo "  Install with: pip install vllm"
fi

# ── 3. Compile Rust spike binary ────────────────────────────────────────────
echo ""
echo "─── Compiling Rust spike binary ───"
cargo build --bin spike --release 2>&1 | tail -5
echo "✓ spike binary compiled"

# ── 4. Run the spike ────────────────────────────────────────────────────────
echo ""
echo "─── Running spike ───"
RUST_LOG=${RUST_LOG:-info} cargo run --bin spike --release -- "$@"

echo ""
echo "═══ Spike complete ═══"

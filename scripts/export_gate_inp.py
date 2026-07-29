#!/usr/bin/env python3
"""Export gate_inp weights as raw binary F32 for Rust coordinator.

Format: [n_layers:u32][n_experts:u32][d_model:u32]
         then for each layer: n_experts * d_model f32 values

Total: 32 * 40 * 1536 * 4 = 7.3 MB

Usage:
    python export_gate_inp.py <model.gguf> -o gate_inp.bin
"""
import argparse, struct
from pathlib import Path
import numpy as np
from gguf import GGUFReader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("-o", "--output", default="gate_inp.bin")
    a = ap.parse_args()

    reader = GGUFReader(a.input)
    total_exp = int(reader.fields["granitemoe.expert_count"].parts[-1].item())
    d_model = int(reader.fields["granitemoe.embedding_length"].parts[-1].item())

    layers = {}
    for t in reader.tensors:
        if "ffn_gate_inp" in t.name and ".weight" in t.name:
            layer = int(t.name.split(".")[1])
            data = np.array(t.data).astype(np.float32)
            layers[layer] = data

    n = len(layers)
    print(f"Exporting: {n} layers × ({total_exp}, {d_model}) = {n*total_exp*d_model*4/1024:.1f} KB")

    with open(Path(a.output), "wb") as f:
        f.write(struct.pack("<III", n, total_exp, d_model))
        for idx in sorted(layers.keys()):
            layers[idx].tofile(f)

    sz = Path(a.output).stat().st_size
    print(f"Written: {a.output} ({sz/1024:.1f} KB)")


if __name__ == "__main__":
    main()

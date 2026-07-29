#!/usr/bin/env python3
"""split_gguf — GGUFWriter + fixed KV reading.

Found bugs: parts[3]=string length (not content), FLOAT32=6 (not 10).
"""
import argparse, os, struct
from pathlib import Path
import numpy as np
from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType, GGUFValueType


def is_expert(n):
    return any(k in n for k in
               ["ffn_gate_inp","ffn_gate_exps","ffn_up_exps","ffn_down_exps"])


def read_string(field):
    """parts[3]=len, parts[4]=bytes"""
    v = field.parts[4]
    if hasattr(v, "tobytes"):
        return v.tobytes().decode("utf-8", "replace")
    return str(v.item())


def read_scalar(field):
    return field.parts[-1].item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input"); ap.add_argument("-n","--num-shards",type=int,default=2)
    ap.add_argument("-o","--output-dir",default="./shards")
    a = ap.parse_args()
    out = Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)

    reader = GGUFReader(a.input)
    total_exp = int(read_scalar(reader.fields["granitemoe.expert_count"]))
    arch = read_string(reader.fields["general.architecture"])
    orig_used = int(read_scalar(reader.fields["granitemoe.expert_used_count"]))
    orig_tensors = list(reader.tensors)

    print(f"Arch:{arch} Experts:{total_exp} Tensors:{len(orig_tensors)}")

    per = total_exp // a.num_shards
    stem = Path(a.input).stem

    for si in range(a.num_shards):
        s = si * per
        ne = total_exp - per * (a.num_shards - 1) if si == a.num_shards - 1 else per
        print(f"\nShard {si}: experts {s}-{s+ne-1}")

        path = str(out / f"{stem}-shard-{si}.gguf")
        writer = GGUFWriter(path, arch)

        # Write KV
        for key, field in reader.fields.items():
            try: fk = field.parts[1].tobytes().decode()
            except: continue
            if fk in ("granitemoe.expert_count","granitemoe.expert_used_count",
                      "general.architecture","tokenizer.ggml.token_type"): continue
            vt = int(field.types[0])
            try:
                if vt == GGUFValueType.STRING:
                    writer.add_string(fk, read_string(field))
                elif vt == GGUFValueType.UINT32:
                    writer.add_uint32(fk, int(read_scalar(field)))
                elif vt == GGUFValueType.UINT64:
                    writer.add_uint64(fk, int(read_scalar(field)))
                elif vt == GGUFValueType.FLOAT32:
                    writer.add_float32(fk, float(read_scalar(field)))
                elif vt == GGUFValueType.FLOAT64:
                    writer.add_float64(fk, float(read_scalar(field)))
                elif vt == GGUFValueType.BOOL:
                    writer.add_bool(fk, bool(read_scalar(field)))
                elif vt == GGUFValueType.ARRAY:
                    parts = field.parts
                    # String arrays: [hdr, key, etype, nelem, len1, str1, len2, str2, ...]
                    # nelem from parts[3] can be wrong for large arrays.
                    # Derive from pair count: (len(parts) - 5) // 2
                    pair_count = (len(parts) - 5) // 2
                    nelem = max(int(parts[3].item()), pair_count) if pair_count > 0 else int(parts[3].item())
                    strs, idx = [], 5
                    while idx < len(parts)-1 and len(strs) < nelem:
                        try:
                            slen = int(parts[idx].item())
                            sdata = parts[idx+1].tobytes().decode("utf-8","replace")
                            strs.append(sdata)
                            idx += 2
                        except: break
                    if strs:
                        writer.add_array(fk, strs)
            except: pass

        writer.add_expert_count(ne)
        writer.add_expert_used_count(min(orig_used, ne))

        # Write tensors
        for t in orig_tensors:
            d = np.array(t.data)
            if not is_expert(t.name):
                writer.add_tensor(t.name, d, raw_shape=list(d.shape), raw_dtype=t.tensor_type)
            else:
                sliced = d[s:s+ne].copy()
                writer.add_tensor(t.name, sliced, raw_shape=list(sliced.shape), raw_dtype=t.tensor_type)

        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

        sz = os.path.getsize(path) / 1024 / 1024
        print(f"  {sz:.1f} MB")

    print("Done.")

if __name__ == "__main__": main()

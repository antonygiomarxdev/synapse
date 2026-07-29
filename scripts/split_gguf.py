#!/usr/bin/env python3
"""split_gguf.py — Split a GGUF MoE model by expert index.

The GGUF stores expert tensors with expert dimension transposed to axis 0.
Slicing is pure numpy indexing along axis 0. No dequantization needed.

Usage:
    python split_gguf.py <input.gguf> -n 2 -o ./shards/
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from gguf import GGUFReader, GGUFWriter
from gguf.constants import GGMLQuantizationType, Keys


_EXPERT_KW = ["ffn_down_exps", "ffn_gate_exps", "ffn_up_exps", "ffn_gate_inp"]


def is_expert_tensor(name: str) -> bool:
    return any(k in name for k in _EXPERT_KW)


def _get_scalar(field):
    """Extract scalar value from a GGUF field robustly."""
    raw = field.parts[-1]
    if hasattr(raw, '__len__') and hasattr(raw, 'item'):
        return raw.item()
    if hasattr(raw, 'item'):
        return raw.item()
    if isinstance(raw, (int, float, bool, str, bytes)):
        return raw
    return raw


def _get_string(field):
    """Extract string value from a GGUF field."""
    raw = field.parts[-1]
    if isinstance(raw, bytes):
        return raw.decode()
    if hasattr(raw, 'tobytes'):
        return raw.tobytes().decode()
    return str(raw)


def _get_array(field):
    """Extract array value from a GGUF field."""
    raw = field.parts[-1]
    if hasattr(raw, '__iter__') and not isinstance(raw, (str, bytes)):
        vals = []
        for v in raw:
            if hasattr(v, 'tobytes'):
                vals.append(v.tobytes().decode())
            elif hasattr(v, 'item'):
                vals.append(v.item())
            else:
                vals.append(v)
        return vals
    return []


def copy_metadata(reader, writer, experts_per_shard):
    """Copy all metadata using GGUFWriter's typed add_* methods."""
    fields = reader.fields
    updated = {"granitemoe.expert_count", "granitemoe.expert_used_count"}

    # Map of known keys to their writer methods
    key_writers = {
        # General
        "general.architecture": lambda v: writer.add_architecture(v),
        "general.name": lambda v: writer.add_name(v),
        "general.basename": lambda v: writer.add_basename(v),
        "general.finetune": lambda v: writer.add_finetune(v),
        "general.license": lambda v: writer.add_license(v),
        "general.type": lambda v: writer.add_type(v),
        "general.size_label": lambda v: writer.add_size_label(v),
        "general.file_type": lambda v: writer.add_file_type(int(v)),
        "general.quantization_version": lambda v: writer.add_quantization_version(int(v)),
        # Architecture params (granitemoe)
        "granitemoe.block_count": lambda v: writer.add_block_count(int(v)),
        "granitemoe.context_length": lambda v: writer.add_context_length(int(v)),
        "granitemoe.embedding_length": lambda v: writer.add_embedding_length(int(v)),
        "granitemoe.feed_forward_length": lambda v: writer.add_feed_forward_length(int(v)),
        "granitemoe.attention.head_count": lambda v: writer.add_head_count(int(v)),
        "granitemoe.attention.head_count_kv": lambda v: writer.add_head_count_kv(int(v)),
        "granitemoe.expert_count": lambda v: writer.add_expert_count(int(v)),
        "granitemoe.expert_used_count": lambda v: writer.add_expert_used_count(int(v)),
        "granitemoe.vocab_size": lambda v: writer.add_vocab_size(int(v)),
        "granitemoe.logit_scale": lambda v: writer.add_logit_scale(float(v)),
        "granitemoe.embedding_scale": lambda v: writer.add_embedding_scale(float(v)),
        "granitemoe.residual_scale": lambda v: writer.add_residual_scale(float(v)),
        "granitemoe.rope.dimension_count": lambda v: writer.add_rope_dimension_count(int(v)),
        "granitemoe.rope.freq_base": lambda v: writer.add_rope_freq_base(float(v)),
        "granitemoe.attention.layer_norm_rms_epsilon": lambda v: writer.add_layer_norm_rms_eps(float(v)),
        "granitemoe.attention.scale": lambda v: writer.add_attention_scale(float(v)),
        # Tokenizer
        "tokenizer.ggml.model": lambda v: writer.add_string("tokenizer.ggml.model", v),
        "tokenizer.ggml.pre": lambda v: writer.add_string("tokenizer.ggml.pre", v),
        "tokenizer.ggml.tokens": lambda v: writer.add_token_list(v),
        "tokenizer.ggml.merges": lambda v: writer.add_token_merges(v),
        "tokenizer.ggml.bos_token_id": lambda v: writer.add_bos_token_id(int(v)),
        "tokenizer.ggml.eos_token_id": lambda v: writer.add_eos_token_id(int(v)),
        "tokenizer.ggml.unknown_token_id": lambda v: writer.add_unk_token_id(int(v)),
        "tokenizer.ggml.padding_token_id": lambda v: writer.add_pad_token_id(int(v)),
        "tokenizer.ggml.add_bos_token": lambda v: writer.add_add_bos_token(bool(v)),
        "tokenizer.ggml.add_space_prefix": lambda v: writer.add_add_space_prefix(bool(v)),
        "tokenizer.chat_template": lambda v: writer.add_chat_template(v),
    }

    for key, field in fields.items():
        if len(field.parts) < 2:
            continue
        try:
            field_key = field.parts[1].tobytes().decode()
        except (IndexError, UnicodeDecodeError):
            continue

        if field_key in updated:
            continue

        val_type = field.types[0]

        if field_key in key_writers:
            try:
                if val_type == 8:  # STRING
                    key_writers[field_key](_get_string(field))
                elif val_type == 9:  # ARRAY
                    key_writers[field_key](_get_array(field))
                else:
                    key_writers[field_key](_get_scalar(field))
            except Exception as e:
                print(f"  WARNING: failed to copy '{field_key}': {e}")
        else:
            # Generic fallback for unknown keys
            try:
                if val_type == 8:
                    writer.add_string(field_key, _get_string(field))
                elif val_type == 4:
                    writer.add_uint32(field_key, int(_get_scalar(field)))
                elif val_type == 6:
                    writer.add_uint64(field_key, int(_get_scalar(field)))
                elif val_type == 10:
                    writer.add_float32(field_key, float(_get_scalar(field)))
                elif val_type == 12:
                    writer.add_bool(field_key, bool(_get_scalar(field)))
                elif val_type == 9:
                    writer.add_array(field_key, _get_array(field))
            except Exception:
                pass  # unknown field type, skip

    # Update expert counts for this shard
    writer.add_expert_count(experts_per_shard)
    original_active = int(_get_scalar(fields["granitemoe.expert_used_count"]))
    writer.add_expert_used_count(min(original_active, experts_per_shard))


def _write_tensor(writer, name, data, logical_shape, tensor_type):
    """Write tensor, preserving quantization type via raw_dtype."""
    if tensor_type in (GGMLQuantizationType.F32, GGMLQuantizationType.F16):
        writer.add_tensor(name, data, raw_shape=logical_shape)
    else:
        writer.add_tensor(name, data, raw_shape=list(data.shape), raw_dtype=tensor_type)


def build_shard(reader, num_shards, shard_idx, output_path):
    total = int(_get_scalar(reader.fields["granitemoe.expert_count"]))
    per = total // num_shards
    experts_this = total - per * (num_shards - 1) if shard_idx == num_shards - 1 else per
    start = shard_idx * per
    end = start + experts_this
    print(f"Shard {shard_idx}: experts {start}-{end - 1} ({experts_this} experts)")

    arch = _get_string(reader.fields["general.architecture"])
    writer = GGUFWriter(output_path, arch)
    copy_metadata(reader, writer, experts_this)

    shared_n, expert_n = 0, 0
    total_bytes = 0

    for tensor in reader.tensors:
        name = tensor.name
        data = np.array(tensor.data)
        logical_shape = list(tensor.shape)
        ttype = tensor.tensor_type

        if not is_expert_tensor(name):
            _write_tensor(writer, name, data, logical_shape, ttype)
            shared_n += 1
            total_bytes += data.nbytes
        else:
            n_exp = data.shape[0]
            assert n_exp == total, f"{name}: expected {total} experts axis 0, got {n_exp}"
            sliced = data[start:end]
            new_shape = logical_shape[:-1] + [sliced.shape[0]]
            _write_tensor(writer, name, sliced, new_shape, ttype)
            expert_n += 1
            total_bytes += sliced.nbytes

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"  Shared: {shared_n}, Expert: {expert_n}, Data: {total_bytes/1024/1024:.1f} MB, File: {size_mb:.1f} MB")


def main():
    p = argparse.ArgumentParser(description="Split GGUF MoE model by experts")
    p.add_argument("input")
    p.add_argument("--num-shards", "-n", type=int, default=2)
    p.add_argument("--output-dir", "-o", default="./shards")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {args.input}")
    reader = GGUFReader(args.input)

    arch = _get_string(reader.fields["general.architecture"])
    total = int(_get_scalar(reader.fields["granitemoe.expert_count"]))
    print(f"Architecture: {arch} | Experts: {total} | Tensors: {len(reader.tensors)}")
    print()

    stem = Path(args.input).stem
    for i in range(args.num_shards):
        build_shard(reader, args.num_shards, i, str(out / f"{stem}-shard-{i}.gguf"))
        print()

    print("Done.")


if __name__ == "__main__":
    main()

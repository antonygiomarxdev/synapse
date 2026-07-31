#!/usr/bin/env python3
"""Check if data_offset is absolute or relative."""
import struct
import numpy as np
from gguf import GGUFReader

MODEL = "/home/ksante/.ollama/models/blobs/sha256-4cbc52994d8ce56d58f3ecadcd451a5dbb2a4f1142098c6b9f030d18ee5e052b"

def main():
    reader = GGUFReader(MODEL)
    
    # Find attn_v tensor
    v_tensor = None
    for t in reader.tensors:
        if t.name == 'blk.0.attn_v.weight':
            v_tensor = t
            break
    
    # Get raw bytes through gguf library
    raw_gguf = v_tensor.data.tobytes()
    
    # Read raw bytes at different offsets
    with open(MODEL, 'rb') as f:
        # Test 1: data_offset as absolute
        f.seek(v_tensor.data_offset)
        raw_abs = f.read(len(raw_gguf))
        match_abs = raw_gguf == raw_abs
        print(f"data_offset as ABSOLUTE ({v_tensor.data_offset}): match={match_abs}")
        
        # Test 2: data_offset + data_region_start
        # Find data region start
        f.seek(0)
        magic = f.read(4)
        version = struct.unpack('<I', f.read(4))[0]
        n_tensors = struct.unpack('<Q', f.read(8))[0]
        n_kv = struct.unpack('<Q', f.read(8))[0]
        
        # Skip KV pairs
        for _ in range(n_kv):
            key_len = struct.unpack('<Q', f.read(8))[0]
            f.read(key_len)  # key
            val_type = struct.unpack('<I', f.read(4))[0]
            if val_type == 0: f.read(1)
            elif val_type == 1: f.read(1)
            elif val_type == 4: f.read(4)
            elif val_type == 5: f.read(4)
            elif val_type == 6: f.read(4)
            elif val_type == 7: f.read(1)
            elif val_type == 8:
                s_len = struct.unpack('<Q', f.read(8))[0]
                f.read(s_len)
            elif val_type == 9:
                elem_type = struct.unpack('<I', f.read(4))[0]
                arr_len = struct.unpack('<Q', f.read(8))[0]
                for _ in range(arr_len):
                    if elem_type == 0: f.read(1)
                    elif elem_type == 1: f.read(1)
                    elif elem_type == 4: f.read(4)
                    elif elem_type == 5: f.read(4)
                    elif elem_type == 6: f.read(4)
                    elif elem_type == 7: f.read(1)
                    elif elem_type == 8:
                        s_len = struct.unpack('<Q', f.read(8))[0]
                        f.read(s_len)
                    elif elem_type == 10: f.read(8)
                    elif elem_type == 11: f.read(8)
                    elif elem_type == 12: f.read(8)
            elif val_type == 10: f.read(8)
            elif val_type == 11: f.read(8)
            elif val_type == 12: f.read(8)
        
        # Skip tensor infos
        for _ in range(n_tensors):
            name_len = struct.unpack('<Q', f.read(8))[0]
            f.read(name_len)
            n_dims = struct.unpack('<I', f.read(4))[0]
            for _ in range(n_dims): f.read(8)
            f.read(4)  # type
            f.read(8)  # offset
        
        pos = f.tell()
        align = 32
        data_start = pos if pos % align == 0 else (pos + align - 1) // align * align
        print(f"Data region start: {data_start}")
        
        # Test 2: data_offset + data_start
        f.seek(data_start + v_tensor.data_offset)
        raw_rel = f.read(len(raw_gguf))
        match_rel = raw_gguf == raw_rel
        print(f"data_offset + data_start ({data_start + v_tensor.data_offset}): match={match_rel}")
        
        # Test 3: Just data_start (first tensor)
        first_tensor = reader.tensors[0]
        print(f"\nFirst tensor: {first_tensor.name}, data_offset={first_tensor.data_offset}")
        f.seek(first_tensor.data_offset)
        raw_first = f.read(min(20, len(raw_gguf)))
        print(f"First 20 bytes at data_offset: {list(raw_first)}")
        
        f.seek(data_start)
        raw_ds = f.read(min(20, len(raw_gguf)))
        print(f"First 20 bytes at data_start: {list(raw_ds)}")
        
        # Check if data_offset == data_start for first tensor
        print(f"\ndata_offset == data_start? {first_tensor.data_offset == data_start}")

if __name__ == '__main__':
    main()

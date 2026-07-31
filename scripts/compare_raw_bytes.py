#!/usr/bin/env python3
"""Compare raw bytes of attn_v tensor between our GGUF reader and llama.cpp."""
import struct
import numpy as np

MODEL = "/home/ksante/.ollama/models/blobs/sha256-4cbc52994d8ce56d58f3ecadcd451a5dbb2a4f1142098c6b9f030d18ee5e052b"

def main():
    # Read raw bytes at the attn_v offset
    # From gguf Python: blk.0.attn_v.weight at data_offset=85074592
    # But data_offset is relative to the data region start, not the file start
    
    from gguf import GGUFReader
    reader = GGUFReader(MODEL)
    
    v_tensor = None
    for t in reader.tensors:
        if t.name == 'blk.0.attn_v.weight':
            v_tensor = t
            break
    
    # Get raw bytes through gguf library
    raw_gguf = v_tensor.data.tobytes()
    print(f"GGUF raw bytes: len={len(raw_gguf)}, first 20={list(raw_gguf[:20])}")
    
    # Read raw bytes directly from file at the reported offset
    # The data_offset from gguf library is relative to data region start
    # We need to find the absolute file offset
    
    # Method 1: Use gguf library's data property (already gives us the raw bytes)
    # Method 2: Read directly from file
    
    # Find the data region start by looking at the first tensor
    first_tensor = reader.tensors[0]
    print(f"First tensor: {first_tensor.name}, data_offset={first_tensor.data_offset}")
    
    # The data_offset is the offset within the data region
    # The data region starts after the header + tensor infos
    
    # Read the file directly
    with open(MODEL, 'rb') as f:
        # Read the GGUF header to find data region start
        magic = f.read(4)
        print(f"Magic: {magic}")
        
        version = struct.unpack('<I', f.read(4))[0]
        print(f"Version: {version}")
        
        n_tensors = struct.unpack('<Q', f.read(8))[0]
        n_kv = struct.unpack('<Q', f.read(8))[0]
        print(f"n_tensors={n_tensors}, n_kv={n_kv}")
        
        # Skip KV pairs
        for _ in range(n_kv):
            key_len = struct.unpack('<Q', f.read(8))[0]
            key = f.read(key_len).decode('utf-8')
            val_type = struct.unpack('<I', f.read(4))[0]
            # Skip value based on type
            if val_type == 0: f.read(1)  # u8
            elif val_type == 1: f.read(1)  # i8
            elif val_type == 4: f.read(4)  # u32
            elif val_type == 5: f.read(4)  # i32
            elif val_type == 6: f.read(4)  # f32
            elif val_type == 7: f.read(1)  # bool
            elif val_type == 8:  # string
                s_len = struct.unpack('<Q', f.read(8))[0]
                f.read(s_len)
            elif val_type == 9:  # array
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
            elif val_type == 10: f.read(8)  # u64
            elif val_type == 11: f.read(8)  # i64
            elif val_type == 12: f.read(8)  # f64
        
        # Skip tensor infos
        for _ in range(n_tensors):
            name_len = struct.unpack('<Q', f.read(8))[0]
            name = f.read(name_len).decode('utf-8')
            n_dims = struct.unpack('<I', f.read(4))[0]
            for _ in range(n_dims):
                f.read(8)  # ne
            f.read(4)  # type
            f.read(8)  # offset
        
        # Current position is end of tensor infos
        pos = f.tell()
        align = 32
        data_start = pos if pos % align == 0 else (pos + align - 1) // align * align
        print(f"Data region start: {data_start}")
        
        # Read attn_v raw bytes
        v_offset = data_start + v_tensor.data_offset
        f.seek(v_offset)
        raw_file = f.read(len(raw_gguf))
        
        print(f"\nattn_v at file offset: {v_offset}")
        print(f"File raw bytes: len={len(raw_file)}, first 20={list(raw_file[:20])}")
        
        # Compare
        match = raw_gguf == raw_file
        print(f"\nBytes match: {match}")
        
        if not match:
            # Find first difference
            for i in range(min(len(raw_gguf), len(raw_file))):
                if raw_gguf[i] != raw_file[i]:
                    print(f"First diff at byte {i}: gguf={raw_gguf[i]} file={raw_file[i]}")
                    break

if __name__ == '__main__':
    main()

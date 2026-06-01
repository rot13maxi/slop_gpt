#!/usr/bin/env python3
"""
Test script to load .slop file and generate text.
Verifies quantized model works correctly.

Format v2:
  zlib-compressed payload wrapping:
  [4 bytes] magic 0x534C4F50
  [4 bytes] tensor count
  per tensor:
    [4 bytes] name_len
    [N bytes] name
    [4 bytes] ndim
    [4*ndim bytes] shape
    [4 bytes] scale (float32)
    [1 byte]  bits (2 or 4)
    [4 bytes] packed_len
    [N bytes] packed data
"""

import os
import struct
import sys
import zlib

import numpy as np
import torch

from model import (
    DEFAULT_FREQUENCY_PENALTY,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    DEVICE,
    SlopGPT,
)
from tokenizer import decode, encode


def unpack_quantized(packed_bytes, n_values, bits):
    """Unpack bytes back to int8 tensor for given bit width."""
    packed = np.frombuffer(packed_bytes, dtype=np.uint8)
    levels = 1 << bits
    if bits >= 4:
        offset = 7
    else:
        offset = levels // 2 - 1
    bits_per_byte = 8 // bits
    mask = levels - 1

    flat = np.empty(packed.shape[0] * bits_per_byte, dtype=np.int8)
    for i in range(bits_per_byte):
        flat[i::bits_per_byte] = ((packed >> (i * bits)) & mask).astype(np.int8)

    flat = flat[:n_values] - offset
    return torch.tensor(flat, dtype=torch.int8)


def dequantize_tensor(quantized, scale):
    """Reconstruct float32 tensor from quantized int8 + scale"""
    return quantized.float() * scale


def load_slop(slop_path):
    """Load model from .slop file. Auto-detects compression."""
    print(f"Loading {slop_path}...")

    with open(slop_path, "rb") as f:
        data = f.read()

    # Try zlib decompression (v2 format)
    try:
        data = zlib.decompress(data)
        print("  ✓ Decompressed (zlib)")
    except zlib.error:
        pass  # not compressed, continue with raw data

    pos = 0

    # Check magic number
    magic = struct.unpack("<I", data[pos : pos + 4])[0]
    pos += 4
    if magic != 0x534C4F50:
        raise ValueError(f"Invalid magic number: {hex(magic)}")
    print("  ✓ Magic number OK")

    # Read tensor count
    num_tensors = struct.unpack("<I", data[pos : pos + 4])[0]
    pos += 4
    print(f"  {num_tensors} tensors")

    state_dict = {}
    for i in range(num_tensors):
        # Name
        name_len = struct.unpack("<I", data[pos : pos + 4])[0]
        pos += 4
        name = data[pos : pos + name_len].decode("utf-8")
        pos += name_len

        # Shape
        ndim = struct.unpack("<I", data[pos : pos + 4])[0]
        pos += 4
        shape = tuple(struct.unpack("<" + "I" * ndim, data[pos : pos + 4 * ndim]))
        pos += 4 * ndim

        # Scale
        scale = struct.unpack("<f", data[pos : pos + 4])[0]
        pos += 4

        # Per-tensor bits (always present in v2)
        t_bits = data[pos]
        pos += 1

        # Packed data
        packed_len = struct.unpack("<I", data[pos : pos + 4])[0]
        pos += 4
        packed = data[pos : pos + packed_len]
        pos += packed_len

        # Unpack and dequantize
        n_values = int(np.prod(shape))
        quantized = unpack_quantized(packed, n_values, t_bits)
        dequantized = dequantize_tensor(quantized, scale)
        tensor = dequantized.reshape(shape)
        state_dict[name] = tensor

        if i < 5 or i >= num_tensors - 2:
            print(f"  {name:50s} {t_bits}bit {shape}")
        elif i == 5:
            print("  ...")

    print(f"  ✓ Loaded {len(state_dict)} tensors")
    return state_dict


def main():
    slop_path = sys.argv[1] if len(sys.argv) > 1 else "pleb.slop"
    bits = int(os.environ.get("BITS", "4"))

    # Load .slop file
    state_dict = load_slop(slop_path)

    # Create model and load weights
    print("\nCreating model...")
    model = SlopGPT().to(DEVICE)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print(f"  ✓ Model loaded on {DEVICE}")

    # Generate text
    print("\n" + "=" * 60)
    print("TEXT GENERATION TEST")
    print("=" * 60)

    test_prompts = [
        ("bitcoin is", DEFAULT_TEMPERATURE, DEFAULT_TOP_K, DEFAULT_TOP_P, DEFAULT_FREQUENCY_PENALTY),
        ("the problem with fiat", DEFAULT_TEMPERATURE, DEFAULT_TOP_K, DEFAULT_TOP_P, DEFAULT_FREQUENCY_PENALTY),
        ("i think that", DEFAULT_TEMPERATURE, DEFAULT_TOP_K, DEFAULT_TOP_P, DEFAULT_FREQUENCY_PENALTY),
        ("you know what i mean", DEFAULT_TEMPERATURE, DEFAULT_TOP_K, DEFAULT_TOP_P, DEFAULT_FREQUENCY_PENALTY),
        ("spam is", DEFAULT_TEMPERATURE, DEFAULT_TOP_K, DEFAULT_TOP_P, DEFAULT_FREQUENCY_PENALTY),
    ]

    for seed, temp, top_k, top_p, freq_pen in test_prompts:
        print(f'\nSeed: "{seed}" (t={temp}, k={top_k}, p={top_p}, freq={freq_pen})')
        print("-" * 40)
        tokens = torch.tensor([encode(seed)], dtype=torch.long).to(DEVICE)
        output = model.generate(tokens, max_new_tokens=48, temperature=temp, top_k=top_k, top_p=top_p, frequency_penalty=freq_pen)
        decoded = decode(output[0].tolist())
        print(f"{decoded}")

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

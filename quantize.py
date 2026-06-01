import torch
import json
import os
import struct
import zlib
import numpy as np
from pathlib import Path
from model import SlopGPT, DEVICE

# Set BITS=2 for 2-bit quantization (4 values per byte)
BITS = int(os.environ.get("BITS", "4"))

# MIXED=1 for mixed precision: 4-bit embeddings+output, BITS for everything else
MIXED = os.environ.get("MIXED", "0") == "1"

# ── quantization ──────────────────────────────────────────────────────────────

def get_quant_range(bits):
    """Return (levels, min_val, max_val) for given bit width.
    4-bit: 16 levels, range [-7, 7], offset 7
    2-bit: 4 levels, range [-1, 2], offset 1
    """
    levels = 1 << bits
    if bits >= 4:
        return levels, -7, 7
    else:
        offset = levels // 2 - 1
        return levels, -offset, offset + 1

def quantize_tensor_with_bits(tensor, bits):
    """Quantize a float32 tensor to given bit width, return (quantized as int8, scale)"""
    t = tensor.detach().float().cpu()

    max_abs = t.abs().max().item()

    if max_abs == 0:
        return torch.zeros_like(t, dtype=torch.int8), 1.0

    levels, q_min, q_max = get_quant_range(bits)

    max_bound = max(abs(q_min), q_max)
    scale = max_abs / max_bound
    scaled = t / scale

    quantized = scaled.round().clamp(q_min, q_max).to(torch.int8)

    return quantized, scale

def quantize_tensor(tensor):
    """Quantize using global BITS setting."""
    return quantize_tensor_with_bits(tensor, BITS)

def dequantize_tensor(quantized, scale):
    """Reconstruct float32 tensor from quantized int8 + scale"""
    return quantized.float() * scale

def quantization_error(original, quantized, scale):
    """How much did we lose? Returns mean absolute error"""
    reconstructed = dequantize_tensor(quantized, scale)
    error = (original.float().cpu() - reconstructed).abs().mean().item()
    return error

# ── packing ───────────────────────────────────────────────────────────────────

def pack_quantized(int8_tensor, bits):
    """Pack quantized int8 values into bytes."""
    flat = int8_tensor.flatten().numpy()

    levels, q_min, q_max = get_quant_range(bits)
    offset = abs(q_min)
    bits_per_byte = 8 // bits
    mask = levels - 1

    flat_unsigned = (flat + offset).astype(np.uint8)

    remainder = len(flat_unsigned) % bits_per_byte
    if remainder != 0:
        flat_unsigned = np.append(flat_unsigned, np.zeros(bits_per_byte - remainder, dtype=np.uint8))

    packed = np.zeros(len(flat_unsigned) // bits_per_byte, dtype=np.uint8)
    for i in range(bits_per_byte):
        packed |= (flat_unsigned[i::bits_per_byte] & mask) << (i * bits)
    return packed.tobytes()

def unpack_quantized(packed_bytes, n_values, bits):
    """Unpack bytes back to int8 tensor."""
    packed = np.frombuffer(packed_bytes, dtype=np.uint8)

    levels, q_min, q_max = get_quant_range(bits)
    offset = abs(q_min)
    bits_per_byte = 8 // bits
    mask = levels - 1

    flat = np.empty(packed.shape[0] * bits_per_byte, dtype=np.int8)
    for i in range(bits_per_byte):
        flat[i::bits_per_byte] = ((packed >> (i * bits)) & mask).astype(np.int8)

    flat = flat[:n_values] - offset
    return torch.tensor(flat, dtype=torch.int8)


# Backward compat aliases
pack_nibbles = lambda t: pack_quantized(t, BITS)
unpack_nibbles = lambda p, n: unpack_quantized(p, n, BITS)

# ── binary format ─────────────────────────────────────────────────────────────
#
# Format v2 (current):
#
# zlib-compressed payload wrapping:
#
# [4 bytes]  magic number: 0x534C4F50 ("SLOP" in ASCII)
# [4 bytes]  number of tensors (uint32)
#
# then for each tensor:
# [4 bytes]  length of name string (uint32)
# [N bytes]  name string (utf-8)
# [4 bytes]  number of dimensions (uint32)
# [4*ndim bytes]  shape (uint32 each)
# [4 bytes]  scale factor (float32)
# [1 byte]  bits (2 or 4)
# [4 bytes]  number of packed bytes (uint32)
# [N bytes]  packed data
#
# ─────────────────────────────────────────────────────────────────────────────

def serialize_model(weights_path, output_path):
    model = SlopGPT().to('cpu')
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model.eval()

    # skip non-learned buffers
    tensors = [(name, tensor) for name, tensor in model.state_dict().items()
               if 'mask' not in name]

    print(f"tensors to quantize: {len(tensors)}")
    print(f"mode: {'mixed (4b emb+out, 2b rest)' if MIXED else f'{BITS}-bit uniform'}")

    buf = bytearray()
    buf += struct.pack('<I', 0x534C4F50)  # "SLOP"
    buf += struct.pack('<I', len(tensors))

    total_original = 0
    total_packed   = 0

    for name, tensor in tensors:
        original_size = tensor.numel() * 4
        total_original += original_size

        # Determine bits for this tensor
        if MIXED:
            t_bits = 4 if ('embedding' in name or 'output' in name) else BITS
        else:
            t_bits = BITS

        quantized, scale = quantize_tensor_with_bits(tensor, t_bits)
        error            = quantization_error(tensor, quantized, scale)
        packed           = pack_quantized(quantized, t_bits)
        total_packed     += len(packed)

        name_bytes = name.encode('utf-8')
        buf += struct.pack('<I', len(name_bytes))
        buf += name_bytes
        buf += struct.pack('<I', len(tensor.shape))
        for dim in tensor.shape:
            buf += struct.pack('<I', dim)
        buf += struct.pack('<f', scale)
        buf += struct.pack('<B', t_bits)
        buf += struct.pack('<I', len(packed))
        buf += packed

        print(f"  {name:50s} {t_bits}bit {list(tensor.shape)} "
              f"scale={scale:.4f} err={error:.4f} "
              f"{len(packed):>6} bytes")

    # Zlib-compress the entire buffer for on-chain storage
    compressed = zlib.compress(buf, 9)

    Path(output_path).write_bytes(compressed)

    print(f"\noriginal: {total_original/1024:.1f} kb")
    print(f"packed:   {total_packed/1024:.1f} kb")
    print(f"uncompressed: {len(buf)/1024:.1f} kb")
    print(f"compressed:   {len(compressed)/1024:.1f} kb")
    print(f"written:  {output_path} ({len(compressed)/1024:.1f} kb)")

if __name__ == '__main__':
    bits_per_byte = 8 // BITS
    print(f"testing round-trip ({BITS}-bit quantization)...")
    original = torch.randn(128, 64)
    quantized, scale = quantize_tensor(original)
    packed           = pack_quantized(quantized, BITS)
    unpacked         = unpack_quantized(packed, original.numel(), BITS)
    reconstructed    = dequantize_tensor(unpacked.reshape(original.shape), scale)
    error = (original - reconstructed).abs().mean().item()
    expected = (original.numel() + bits_per_byte - 1) // bits_per_byte
    print(f"  mean absolute error: {error:.6f}")
    print(f"  packed size:         {len(packed)} bytes")
    print(f"  expected size:       {expected} bytes")
    print(f"  round-trip ok:       {len(packed) == expected}")
    print()

    # serialize real weights
    serialize_model('weights_best.pt', 'pleb.slop')

"""Shared on-disk format for layer activation traces.

One file per (case, layer). Layout is fixed, little-endian, and mirrored
byte-for-byte by forge/trace_io.hpp so that the C++ and PyTorch sides cannot
silently disagree about what they wrote.

  offset  0 : char   magic[8]      = "FTRC0001"
  offset  8 : uint32 layer
  offset 12 : uint32 n_positions
  offset 16 : uint32 d_mlp
  offset 20 : uint32 dtype_code    = 0  (float32, little-endian)
  offset 24 : char   case_id[32]   (NUL-padded)
  offset 56 : float32 payload, row-major [position][neuron]

Payload is the post-GELU / pre-c_proj MLP activation:
  HF    : model.transformer.h[layer].mlp.act   output, batch index 0
  Forge : the `gelu` tensor inside the transformer block
"""

from __future__ import annotations

import hashlib
import struct
from pathlib import Path

import numpy as np

MAGIC = b"FTRC0001"
HEADER_FMT = "<8sIIII32s"
HEADER_SIZE = struct.calcsize(HEADER_FMT)
assert HEADER_SIZE == 56, HEADER_SIZE

DTYPE_F32_LE = 0


class TraceError(RuntimeError):
    pass


def trace_filename(case_id: str, layer: int) -> str:
    return f"{case_id}_L{layer:02d}.ftrc"


def write_trace(path, case_id: str, layer: int, array: np.ndarray) -> Path:
    """array: shape [n_positions, d_mlp], any float dtype (cast to float32)."""
    path = Path(path)
    arr = np.ascontiguousarray(np.asarray(array, dtype="<f4"))
    if arr.ndim != 2:
        raise TraceError(f"expected 2-D [position, neuron] array, got shape {arr.shape}")
    cid = case_id.encode("ascii")
    if len(cid) > 32:
        raise TraceError(f"case_id too long (max 32 ascii chars): {case_id!r}")
    header = struct.pack(
        HEADER_FMT, MAGIC, int(layer), arr.shape[0], arr.shape[1],
        DTYPE_F32_LE, cid.ljust(32, b"\0"),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(header)
        fh.write(arr.tobytes(order="C"))
    return path


def read_trace(path):
    """Returns (meta_dict, array[n_positions, d_mlp] float32)."""
    path = Path(path)
    raw = path.read_bytes()
    if len(raw) < HEADER_SIZE:
        raise TraceError(f"{path}: file shorter than header")
    magic, layer, n_pos, d_mlp, dtype_code, cid = struct.unpack(
        HEADER_FMT, raw[:HEADER_SIZE]
    )
    if magic != MAGIC:
        raise TraceError(f"{path}: bad magic {magic!r} (expected {MAGIC!r})")
    if dtype_code != DTYPE_F32_LE:
        raise TraceError(f"{path}: unsupported dtype_code {dtype_code}")
    expected = HEADER_SIZE + n_pos * d_mlp * 4
    if len(raw) != expected:
        raise TraceError(
            f"{path}: size {len(raw)} != expected {expected} "
            f"for [{n_pos}, {d_mlp}] float32"
        )
    arr = np.frombuffer(raw, dtype="<f4", count=n_pos * d_mlp, offset=HEADER_SIZE)
    meta = {
        "path": str(path),
        "case_id": cid.rstrip(b"\0").decode("ascii"),
        "layer": int(layer),
        "n_positions": int(n_pos),
        "d_mlp": int(d_mlp),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    return meta, arr.reshape(n_pos, d_mlp).astype(np.float32)


# --- checksums shared with the C++ side -----------------------------------

FNV64_OFFSET = 0xCBF29CE484222325
FNV64_PRIME = 0x100000001B3
_MASK64 = 0xFFFFFFFFFFFFFFFF


def fnv1a64(data: bytes) -> int:
    """Layout-sensitive checksum. Implemented identically in trace_io.hpp."""
    h = FNV64_OFFSET
    for b in data:
        h = ((h ^ b) * FNV64_PRIME) & _MASK64
    return h


def tensor_invariants(arr: np.ndarray) -> dict:
    """Orientation-invariant statistics.

    Forge may store a weight transposed relative to HF's Conv1D layout, so a
    raw byte hash can differ legitimately. sum/sumsq/min/max cannot: if these
    disagree, the two programs are holding different numbers, not the same
    numbers in a different order.
    """
    a64 = np.asarray(arr, dtype=np.float64).ravel()
    a32 = np.ascontiguousarray(np.asarray(arr, dtype="<f4"))
    return {
        "shape": list(np.shape(arr)),
        "count": int(a64.size),
        "sum": float(a64.sum()),
        "sumsq": float(np.dot(a64, a64)),
        "min": float(a64.min()) if a64.size else 0.0,
        "max": float(a64.max()) if a64.size else 0.0,
        "fnv1a64_hf_layout": f"{fnv1a64(a32.tobytes(order='C')):016x}",
    }


def sha256_file(path) -> dict:
    p = Path(path)
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return {"path": str(p), "bytes": p.stat().st_size, "sha256": h.hexdigest()}

"""TORCH_LIBRARY-based access to Pyramid Forcing C++/CUDA kernels.

Exposes ``torch.ops.adahead.*`` after the JIT extension loads. Use this
module instead of ``_scatter_ext`` for code paths that need to participate
in CUDA Graph capture or torch.compile/export — TORCH_LIBRARY ops carry
schema with mutable-arg annotations and a meta kernel; the legacy
``_scatter_ext`` PYBIND11 binding does not.
"""
from __future__ import annotations

import torch

from . import _scatter_ext


def _ensure_loaded() -> bool:
    """Trigger JIT compilation; TORCH_LIBRARY blocks register on import."""
    return _scatter_ext._try_load()


def available() -> bool:
    """True iff torch.ops.adahead.* is callable on this build."""
    if not _ensure_loaded():
        return False
    return hasattr(torch.ops, "adahead") and hasattr(torch.ops.adahead, "scatter_copy")


def ops():
    """Return the ``torch.ops.adahead`` namespace, lazily loading the ext."""
    if not _ensure_loaded():
        raise RuntimeError("CUDA scatter extension not available")
    if not hasattr(torch.ops, "adahead"):
        raise RuntimeError(
            "TORCH_LIBRARY(adahead) was not registered — rebuild the extension."
        )
    return torch.ops.adahead


# Direct passthroughs for ergonomic call sites. Each thin wrapper ensures the
# extension is loaded before forwarding to the registered op. Schema and
# mutable-arg semantics live in scatter_copy_bind.cpp.

def scatter_copy(
    src_ptrs: torch.Tensor,
    lengths: torch.Tensor,
    offsets: torch.Tensor,
    dst: torch.Tensor,
    col_dim: int,
) -> None:
    ops().scatter_copy(src_ptrs, lengths, offsets, dst, int(col_dim))


_EMPTY_I64: torch.Tensor | None = None
_EMPTY_F32: torch.Tensor | None = None


def _empty_i64(device: torch.device) -> torch.Tensor:
    global _EMPTY_I64
    if _EMPTY_I64 is None or _EMPTY_I64.device != device:
        _EMPTY_I64 = torch.empty(0, dtype=torch.int64, device=device)
    return _EMPTY_I64


def _empty_f32(device: torch.device) -> torch.Tensor:
    global _EMPTY_F32
    if _EMPTY_F32 is None or _EMPTY_F32.device != device:
        _EMPTY_F32 = torch.empty(0, dtype=torch.float32, device=device)
    return _EMPTY_F32


def pyramidkv_pack(
    mgr,
    src_kind: torch.Tensor,
    src_slot_global: torch.Tensor,
    seg_lengths: torch.Tensor,
    dst_token_offsets: torch.Tensor,
    *,
    anchor_t_remap: torch.Tensor | None = None,
    freqs_ft_flat: torch.Tensor | None = None,
    freqs_fy_flat: torch.Tensor | None = None,
    freqs_fx_flat: torch.Tensor | None = None,
) -> None:
    """Ergonomic wrapper for the pyramidkv_pack op.

    The optional ``anchor_t_remap`` + ``freqs_{ft,fy,fx}_flat`` args enable
    fused-RoPE packing. Callers that just want pure memcpy can omit them;
    this helper substitutes zero-sized tensors so the schema is satisfied
    without forcing every call site to construct them.
    """
    device = src_kind.device
    e_i64 = _empty_i64(device)
    e_f32 = _empty_f32(device)
    ops().pyramidkv_pack(
        mgr,
        src_kind, src_slot_global, seg_lengths, dst_token_offsets,
        anchor_t_remap if anchor_t_remap is not None else e_i64,
        freqs_ft_flat if freqs_ft_flat is not None else e_f32,
        freqs_fy_flat if freqs_fy_flat is not None else e_f32,
        freqs_fx_flat if freqs_fx_flat is not None else e_f32,
    )


def apply_pos_override(
    pos: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    vals: torch.Tensor,
) -> None:
    ops().apply_pos_override(pos, starts, ends, vals)


def anchor_store_write_frames(
    k_seq: torch.Tensor,
    v_seq: torch.Tensor,
    pos_seq: torch.Tensor,
    frame_desc: torch.Tensor,
    store_k: torch.Tensor,
    store_v: torch.Tensor,
    store_pos: torch.Tensor,
) -> None:
    ops().anchor_store_write_frames(
        k_seq, v_seq, pos_seq, frame_desc, store_k, store_v, store_pos
    )


def refresh_readout_layout(
    src_k: list[torch.Tensor],
    src_v: list[torch.Tensor],
    src_pos: list[torch.Tensor],
    src_ptrs_k: torch.Tensor,
    src_ptrs_v: torch.Tensor,
    src_ptrs_pos: torch.Tensor,
    offsets: torch.Tensor,
    lengths: torch.Tensor,
    flags: torch.Tensor,
    dynamic_rope_t: torch.Tensor,
    dst_k_raw: torch.Tensor,
    dst_v: torch.Tensor,
    dst_rope_pos: torch.Tensor,
    dst_frame_ids: torch.Tensor,
    head_dim: int,
    current_t: int,
    history_time_mapping_mode: str,
    history_relative_t_max: int,
    history_time_soft_factor: float,
) -> None:
    ops().refresh_readout_layout(
        src_k,
        src_v,
        src_pos,
        src_ptrs_k,
        src_ptrs_v,
        src_ptrs_pos,
        offsets,
        lengths,
        flags,
        dynamic_rope_t,
        dst_k_raw,
        dst_v,
        dst_rope_pos,
        dst_frame_ids,
        int(head_dim),
        int(current_t),
        str(history_time_mapping_mode),
        int(history_relative_t_max),
        float(history_time_soft_factor),
    )

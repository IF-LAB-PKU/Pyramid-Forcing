"""mega_attention Python orchestrator.

Wires the C++ pack/plan/update ops + ``flash_attn_varlen_func`` into
per-layer attention calls. ``mega_readout`` does plan + pack +
flash_attn_varlen for a single layer; ``mega_attention_step`` chains that
with the insertion path. Callers are responsible for keeping the manager's
pools populated and the per-head states reflecting the current cache.
"""
from __future__ import annotations

import torch

from . import _ops, _mega_state_ops


def _flash_attn_varlen():
    """Lazy import — flash-attn isn't always available on CPU-only machines."""
    try:
        from flash_attn import flash_attn_varlen_func
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "flash_attn is required for mega_readout. Install flash-attn 2.8.3+."
        ) from exc
    return flash_attn_varlen_func


def mega_readout_kv_only(
    mgr,                                      # torch.classes.adahead.PyramidKVCacheManager
    per_head_states_bytes: torch.Tensor,      # uint8 [H * sizeof(PerHeadState)]
    layer_idx: int,
    current_t: int,
    H: int,
    D: int,
    *,
    freqs: torch.Tensor | None = None,
    freq_parts: tuple | None = None,
    pass_kind: int = 1,
    sink_time_mapping_mode: str = "lag",
    sink_time_clamp_min: int = 0,
    sink_time_clamp_max: int = 21,
    decoupled_sink_time_lag: int = 0,
    history_time_mapping_mode: str = "none",
    history_relative_t_max: int = 21,
    history_time_soft_factor: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run plan + pack + RoPE for ONE query chunk. Returns CLONED (k, v, cu_seqlens_k).

    Does NOT call flash_attn. Caller is responsible for FA.

    The manager's k_flat_out / v_flat_out / pos_flat_out workspaces are
    shared, so the K/V slices must be cloned out before the next chunk
    overwrites them. cu_seqlens_k is cloned for the same reason
    (mega_plan's output tensor lives in the manager).

    Returns:
        k_chunk: [n_active, 1, D] bf16  (cloned)
        v_chunk: [n_active, 1, D] bf16  (cloned)
        cu_seqlens_k: [H + 1] int32  (cloned, chunk-local offsets)
    """
    # ---- Step 1: plan ----
    (cu_seqlens_k, src_kind, src_slot_global, seg_lengths, dst_offsets,
     _anchor_t_raw, _anchor_t_remap) = _mega_state_ops.mega_plan_cuda(
        mgr, per_head_states_bytes,
        layer_idx=layer_idx,
        current_t=current_t,
        pass_kind=pass_kind,
        sink_time_mapping_mode=sink_time_mapping_mode,
        sink_time_clamp_min=sink_time_clamp_min,
        sink_time_clamp_max=sink_time_clamp_max,
        decoupled_sink_time_lag=decoupled_sink_time_lag,
        history_time_mapping_mode=history_time_mapping_mode,
        history_relative_t_max=history_relative_t_max,
        history_time_soft_factor=history_time_soft_factor,
    )

    # ---- Step 2: pack ----
    sg_adjusted = _layer_offset_slot_global(
        src_slot_global, src_kind, layer_idx,
        max_sink=int(mgr.max_sink()),
        max_middle=int(mgr.max_middle()),
        max_recent=int(mgr.max_recent()),
        num_heads=H,
        max_merge_blocks=int(mgr.max_merge_blocks()),
    )
    _ops.pyramidkv_pack(
        mgr, src_kind, sg_adjusted, seg_lengths, dst_offsets,
    )

    # ---- Step 2.5: RoPE on raw packed K ----
    # Sentinel-tolerant override: for dynamic-RoPE segments (sink, cyclic,
    # stride dynamic_rope=True, merge) plan emits tremap >= 0 → override
    # pos[:, 0]. For non-dynamic (recent, lag) plan emits -1 → keep stored pos.
    n_active = 0
    if freqs is not None:
        from .rope import apply_rope_to_flat_k
        seg_lens_long = seg_lengths.long()
        tremap_per_tok = torch.repeat_interleave(_anchor_t_remap, seg_lens_long)
        n_active = int(tremap_per_tok.shape[0])
        if n_active > 0:
            pos_3d = mgr.pos_flat_out()[:n_active].clone()
            mask = tremap_per_tok >= 0
            pos_3d[:, 0] = torch.where(mask, tremap_per_tok, pos_3d[:, 0])
            k_active = mgr.k_flat_out()[:n_active].view(n_active, D)
            apply_rope_to_flat_k(
                k_active, pos_3d,
                freqs=freqs, freq_parts=freq_parts,
                out=k_active,
            )
    else:
        n_active = int(cu_seqlens_k[-1].item())

    # ---- Clone K/V/cu_k so subsequent chunks can reuse the workspace ----
    if n_active > 0:
        k_chunk = mgr.k_flat_out()[:n_active].clone()
        v_chunk = mgr.v_flat_out()[:n_active].clone()
    else:
        k_chunk = mgr.k_flat_out()[:0].clone()
        v_chunk = mgr.v_flat_out()[:0].clone()
    return k_chunk, v_chunk, cu_seqlens_k.clone()


def _attend_one_chunk(
    q: torch.Tensor,                # [B, L_q, H, D]
    k_flat: torch.Tensor,           # [n_active, 1, D]
    v_flat: torch.Tensor,           # [n_active, 1, D]
    cu_seqlens_k: torch.Tensor,     # [H + 1] int32, chunk-local
    *,
    softmax_scale: float | None,
    causal: bool,
    max_seqlen_k: int,
) -> torch.Tensor:
    """Single-chunk FA varlen. Returns [B, L_q, H, D] in q's dtype."""
    B, L_q, H, D = q.shape
    target_dtype = q.dtype
    device = q.device
    q_flat = (
        q.permute(0, 2, 1, 3)
         .contiguous()
         .view(B * H * L_q, 1, D)
    )
    cu_seqlens_q = torch.arange(
        0, B * H * L_q + 1, L_q,
        dtype=torch.int32, device=device,
    )
    if q_flat.dtype != k_flat.dtype:
        q_flat = q_flat.to(k_flat.dtype)
    if softmax_scale is None:
        softmax_scale = D ** -0.5
    flash = _flash_attn_varlen()
    out_flat = flash(
        q_flat, k_flat, v_flat,
        cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q=L_q,
        max_seqlen_k=max_seqlen_k,
        dropout_p=0.0,
        softmax_scale=softmax_scale,
        causal=causal,
    )
    return (
        out_flat.view(B, H, L_q, D)
        .permute(0, 2, 1, 3)
        .contiguous()
        .to(target_dtype)
    )


def mega_readout(
    mgr,                                      # torch.classes.adahead.PyramidKVCacheManager
    per_head_states_bytes: torch.Tensor,      # uint8 [H * sizeof(PerHeadState)]
    q: torch.Tensor,                          # [B, L_q, H, D] bf16 — post-Q-RoPE
    layer_idx: int,
    current_t: int,
    *,
    freqs: torch.Tensor | None = None,        # [max_t, head_dim/2] complex64 RoPE table
    freq_parts: tuple | None = None,          # cached (ft, fy, fx) split — perf win
    pass_kind: int = 1,
    softmax_scale: float | None = None,
    causal: bool = False,
    max_seqlen_k: int | None = None,
    sink_time_mapping_mode: str = "lag",
    sink_time_clamp_min: int = 0,
    sink_time_clamp_max: int = 21,
    decoupled_sink_time_lag: int = 0,
    history_time_mapping_mode: str = "none",
    history_relative_t_max: int = 21,
    history_time_soft_factor: float = 1.0,
) -> torch.Tensor:
    """One-layer single-chunk attention readout from a pre-populated KV cache.

    Thin wrapper around ``mega_readout_kv_only`` + ``_attend_one_chunk``,
    retained for backward compatibility with tests and any single-chunk
    callers. Production attend path (MegaCache.attend) bypasses this to
    avoid running FA per chunk — see _mega_cache.py for the multi-chunk
    FA fusion.

    Returns: Attention output shaped [B, L_q, H, D] in q's dtype.
    """
    assert q.dim() == 4, f"q must be [B, L_q, H, D], got {tuple(q.shape)}"
    B, L_q, H, D = q.shape
    assert B == 1, "mega_readout: B>1 not yet supported"

    k_flat, v_flat, cu_seqlens_k = mega_readout_kv_only(
        mgr=mgr,
        per_head_states_bytes=per_head_states_bytes,
        layer_idx=layer_idx,
        current_t=current_t,
        H=H, D=D,
        freqs=freqs, freq_parts=freq_parts,
        pass_kind=pass_kind,
        sink_time_mapping_mode=sink_time_mapping_mode,
        sink_time_clamp_min=sink_time_clamp_min,
        sink_time_clamp_max=sink_time_clamp_max,
        decoupled_sink_time_lag=decoupled_sink_time_lag,
        history_time_mapping_mode=history_time_mapping_mode,
        history_relative_t_max=history_relative_t_max,
        history_time_soft_factor=history_time_soft_factor,
    )
    if max_seqlen_k is None:
        max_seqlen_k = int(mgr.max_total()) * int(mgr.frame_seqlen())
    return _attend_one_chunk(
        q, k_flat, v_flat, cu_seqlens_k,
        softmax_scale=softmax_scale, causal=causal,
        max_seqlen_k=max_seqlen_k,
    )


def mega_attention_step(
    mgr,                                      # PyramidKVCacheManager
    per_head_states_bytes: torch.Tensor,      # uint8 [H * sizeof(PerHeadState)]
    q: torch.Tensor,                          # [B, L_q, H, D] bf16 — post-Q-RoPE
    k_new: torch.Tensor,                      # [B, L_q, H, D] bf16 — post-K-RoPE
    v_new: torch.Tensor,                      # [B, L_q, H, D] bf16
    new_t_vals: torch.Tensor,                 # [F] int64 — t value per new frame
    layer_idx: int,
    current_t: int,
    *,
    pass_kind: int = 1,
    softmax_scale: float | None = None,
    causal: bool = False,
    max_seqlen_k: int | None = None,
    sink_time_mapping_mode: str = "lag",
    sink_time_clamp_min: int = 0,
    sink_time_clamp_max: int = 21,
    decoupled_sink_time_lag: int = 0,
    history_time_mapping_mode: str = "none",
    history_relative_t_max: int = 21,
    history_time_soft_factor: float = 1.0,
) -> torch.Tensor:
    """Full per-layer step: middle insertion + attention readout.

    Pipeline:
      - Runs ``mega_state_update`` to mutate PerHeadState + emit descriptors
        (cyclic / stride / lag → middle slot writes; merge → accum descriptors).
      - Converts per-head local slot indices to manager-global slot_global.
      - Runs ``pyramidkv_update`` to copy new K/V into the middle pool slots.
      - Runs ``mega_readout`` for the attention output.

    Out of scope (caller responsibility for now):
      - Sink insertion (only happens at cold start; pyramid forcing fills
        sink frames during pipeline init)
      - Recent insertion (sliding window; replaceable by a small Python
        scatter or by a follow-on kernel)
      - Merge K/V scatter-add (use mega_merge_accum_cuda separately when
        merge heads are present; pyramid_forcing G6 doesn't have merge)

    Returns attention output [B, L_q, H, D] in q's dtype.
    """
    assert q.shape == k_new.shape == v_new.shape, "q, k_new, v_new must agree"
    B, L_q, H, D = q.shape
    F = int(new_t_vals.shape[0])
    FSEQ = int(mgr.frame_seqlen())
    assert L_q == F * FSEQ, f"L_q ({L_q}) must equal F ({F}) * FSEQ ({FSEQ})"

    device = q.device

    # ---- Step 1: mega_state_update — mutate state, emit descriptors ----
    N = H * F
    desc_kind = torch.full((N,), -1, dtype=torch.int32, device=device)
    desc_slot_local = torch.full((N,), -1, dtype=torch.int32, device=device)
    desc_src_frame = torch.zeros(N, dtype=torch.int32, device=device)
    desc_src_head = torch.zeros(N, dtype=torch.int32, device=device)
    desc_merge_slot = torch.full((N,), -1, dtype=torch.int32, device=device)
    desc_merge_local = torch.full((N,), -1, dtype=torch.int32, device=device)
    desc_merge_finalize = torch.full((N,), -1, dtype=torch.int32, device=device)
    desc_merge_new = torch.zeros(N, dtype=torch.int32, device=device)

    _ops.ops().mega_state_update(
        per_head_states_bytes, new_t_vals,
        desc_kind, desc_slot_local, desc_src_frame, desc_src_head,
        desc_merge_slot, desc_merge_local,
        desc_merge_finalize, desc_merge_new,
        int(H), int(F), int(pass_kind),
    )

    # ---- Step 2: pyramidkv_update — write K/V into middle pool slots ----
    # mega_state_update emits per-head local slot indices (0..max_middle-1
    # for middle). pyramidkv_update expects slot_global = (l*H+h)*max_slots+slot.
    max_middle = int(mgr.max_middle())
    head_global_offset = (layer_idx * H + desc_src_head.long()) * max_middle
    desc_slot_global = (desc_slot_local.long() + head_global_offset).to(torch.int32)
    # For inactive (kind == -1), the kernel branches out so any slot_global is fine.

    # Reshape new_k/v from [B, L_q, H, D] to [F, H, FSEQ, D] (head-major frames).
    # B=1 assumed (mega_readout already enforces it). We squeeze the batch dim.
    assert B == 1, "mega_attention_step: B>1 not yet supported"
    new_k = (
        k_new.view(B, F, FSEQ, H, D)
             .permute(0, 1, 3, 2, 4)   # [B, F, H, FSEQ, D]
             .reshape(F, H, FSEQ, D)
             .contiguous()
    )
    new_v = (
        v_new.view(B, F, FSEQ, H, D)
             .permute(0, 1, 3, 2, 4)
             .reshape(F, H, FSEQ, D)
             .contiguous()
    )

    # Empty new_pos placeholder — this helper is only used by tests;
    # production path goes through MegaCache.update which threads the real
    # per-token pos required by raw-K storage.
    _empty_pos = torch.empty(0, dtype=torch.int64, device=new_k.device)
    _ops.ops().pyramidkv_update(
        mgr, new_k, new_v, _empty_pos,
        desc_kind, desc_slot_global, desc_src_frame, desc_src_head,
    )

    # ---- Step 3: readout ----
    return mega_readout(
        mgr=mgr,
        per_head_states_bytes=per_head_states_bytes,
        q=q,
        layer_idx=layer_idx,
        current_t=current_t,
        pass_kind=pass_kind,
        softmax_scale=softmax_scale,
        causal=causal,
        max_seqlen_k=max_seqlen_k,
        sink_time_mapping_mode=sink_time_mapping_mode,
        sink_time_clamp_min=sink_time_clamp_min,
        sink_time_clamp_max=sink_time_clamp_max,
        decoupled_sink_time_lag=decoupled_sink_time_lag,
        history_time_mapping_mode=history_time_mapping_mode,
        history_relative_t_max=history_relative_t_max,
        history_time_soft_factor=history_time_soft_factor,
    )


def _layer_offset_slot_global(
    src_slot_global: torch.Tensor,
    src_kind: torch.Tensor,
    layer_idx: int,
    max_sink: int,
    max_middle: int,
    max_recent: int,
    num_heads: int,
    max_merge_blocks: int = 0,
) -> torch.Tensor:
    """Shift mega_plan's per-layer slot_global into the manager's global
    [L, H, ...] pool layout that pyramidkv_pack expects.

    mega_plan emits ``slot_global = h * max_slots[kind] + slot``. The pack
    kernel computes ``src_base = slot_global * stride[kind] * head_dim``
    against the FULL pool. To address into layer ``L`` we add
    ``layer_idx * H * max_slots[kind]`` per element. ``max_slots`` differs
    by kind so the offset is kind-dependent.

    A torch.where chain over the 4 kinds handles this without a CPU sync.
    """
    if layer_idx == 0:
        return src_slot_global

    offset_per_kind = torch.tensor(
        [layer_idx * num_heads * max_sink,
         layer_idx * num_heads * max_middle,
         layer_idx * num_heads * max_recent,
         layer_idx * num_heads * max_merge_blocks],
        dtype=torch.int32, device=src_slot_global.device,
    )
    # src_kind values: 0=sink, 1=middle, 2=recent, 3=merge, -1=inactive.
    # For inactive, leave unchanged.
    kind_safe = src_kind.clamp(min=0)  # -1 → 0 (will mask out below)
    offset = offset_per_kind[kind_safe.long()]
    active_mask = (src_kind >= 0).to(offset.dtype)
    return src_slot_global + offset * active_mask

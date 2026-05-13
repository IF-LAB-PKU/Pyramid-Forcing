"""MegaCache: thin Python wrapper for the C++ mega kernel path.

Replaces ``AdaptiveKVCache`` at the model-layer interface when the C++
mega path is enabled (env ``PYRAMIDKV_USE_MEGA_ATTN=1``).

Design:
  - One ``PyramidKVCacheManager`` covers all L layers' pools (shared allocator).
  - Per-layer ``PerHeadState`` lives in a single contiguous tensor of size
    ``[L, H * sizeof(PerHeadState)]`` uint8.
  - ``MegaCache`` is instantiated per-layer (matching the existing
    ``kv_cache1[layer_idx]`` pattern), but every instance shares the same
    underlying ``(mgr, states_bytes)`` objects — only ``layer_idx`` differs.

Pipeline integration:
  - ``causal_inference.py``: if the env flag is set, call ``build_mega_caches``
    instead of constructing ``AdaptiveKVCache`` per layer.
  - ``CausalWanSelfAttention.forward``: if ``kv_cache`` is a ``MegaCache``,
    dispatch to ``mega_attention_step`` instead of ``pyramidkv_attention``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch

from . import _mega_state_ops, _mega_state_ref as ref
from .base import HeadComposition


# Strategy-name → SK_* int code (matches enum in anchor_store.cuh).
_STRATEGY_NAME_TO_KIND = {
    "RecentStrategy": ref.SK_RECENT,
    "CyclicStrategy": ref.SK_CYCLIC,
    "StrideStrategy": ref.SK_STRIDE,
    "LagStrategy":    ref.SK_LAG,
    "MergeStrategy":  ref.SK_MERGE,
}


@dataclass
class MegaCacheCtx:
    """Long-lived state shared across all per-layer MegaCache instances."""
    mgr: object                # torch.classes.adahead.PyramidKVCacheManager
    states_bytes: torch.Tensor # uint8 [L, H * sizeof(PerHeadState)]
    num_layers: int
    num_heads: int
    head_dim: int
    frame_seqlen: int
    # RoPE-remap config (per pyramid forcing yaml).
    sink_time_mapping_mode: str = "lag"
    sink_time_clamp_min: int = 0
    sink_time_clamp_max: int = 21
    decoupled_sink_time_lag: int = 0
    history_time_mapping_mode: str = "none"
    history_relative_t_max: int = 21
    history_time_soft_factor: float = 1.0
    # Spatial layout: tokens within one frame are laid out as a
    # ``spatial_height × spatial_width`` grid (row-major). Used to construct
    # the (t, y, x) position tensor passed to ``mega_merge_accum_cuda``.
    # Defaults match pyramid_forcing10 at 480×832 → 30×52 latent tokens
    # (1560 = 30*52, post-patch-embed).
    spatial_width: int = 52
    spatial_height: int = 30
    # True iff any (layer, head) composition is SK_MERGE — used to gate
    # the per-layer mega_merge_accum launch without a per-call CPU sync.
    has_merge_head: bool = False
    # Per-head sink capacity tensor [L, H] int64 on the manager's device.
    # MegaCache.update routes overflow frames (when sink_count >= cap_h)
    # to recent_pool instead of sink_pool. Set at build_mega_caches time.
    sink_caps_per_head: torch.Tensor = None
    # Committed valid_count snapshot — written at the end of each clean
    # update(), restored at the start of every update() so that noisy
    # denoising iterations overwrite the same sink/recent slot instead
    # of advancing counters monotonically.
    committed_vc: torch.Tensor = None  # [L, H, 3] int64, on mgr.valid_count's device
    # Committed recent_pool K/V snapshot. Without this, every update() re-runs
    # the FIFO shift-left, and a noisy iteration's writes contaminate the
    # shift source for the clean iteration — corrupting recent_pool slot 0
    # with a noisy K once per block. Mirror AdaptiveKVCache by restoring the
    # committed pool at update entry; only the clean pass's writes are then
    # snapshot-committed.
    committed_recent_k: torch.Tensor = None  # [L, H, max_recent, FSEQ, D] bf16
    committed_recent_v: torch.Tensor = None  # [L, H, max_recent, FSEQ, D] bf16
    # Per-kind pos pool snapshot. recent_pos_pool rides the same FIFO shift
    # as recent_k/v, so it needs the same noisy/clean snapshot to keep the
    # slot↔t mapping consistent under raw-K storage.
    committed_recent_pos: torch.Tensor = None  # [L, H, max_recent, FSEQ, 3] int64
    # 3D RoPE freq table + cached per-axis split. mega_readout's fused
    # apply_rope_to_flat_k slices freqs into (ft, fy, fx) every call; caching
    # the split avoids redundant work in the hot path.
    freqs: torch.Tensor = None              # [max_t, head_dim/2] complex64
    freq_parts: tuple = None                # (ft, fy, fx) — each [max_t, *]
    # fp32 flat (real, imag interleaved) versions of freq_parts for the
    # fused-RoPE pack kernel. Each tensor has shape [max_t, cols * 2] fp32
    # so a CUDA thread can read 2 fp32s per pair without complex32 dtype
    # quirks.
    freq_parts_flat: tuple = None            # (ft_flat, fy_flat, fx_flat)


class MegaCache:
    """Per-layer view onto the shared MegaCacheCtx.

    Mirrors the few attributes CausalWanSelfAttention.forward reads from
    a PyramidKV cache instance (layer_idx, frame_seq_length, ...). All state
    actually lives in MegaCacheCtx.mgr / MegaCacheCtx.states_bytes.

    Public methods:
      ``update(new_k, new_v, current_start)`` — insert new frames into
          the cache. Handles sink fill (Python), recent slide (Python),
          and middle insertion (C++ mega_state_update + pyramidkv_update).
      ``attend(q)`` — compute attention output from current cache state.
          Wraps mega_readout (plan + pack + flash_attn_varlen).
    """

    def __init__(self, ctx: MegaCacheCtx, layer_idx: int):
        self.ctx = ctx
        self.layer_idx = layer_idx
        # Mirror common attributes the existing pipeline reads off cache objects.
        self._frame_seqlen = ctx.frame_seqlen
        self.frame_seq_length = ctx.frame_seqlen
        self.num_heads = ctx.num_heads
        self.head_dim = ctx.head_dim
        # Used by some upstream code (don't crash if accessed).
        self.batch_size = 1

    @property
    def states_bytes_for_layer(self) -> torch.Tensor:
        """uint8 [H * sizeof(PerHeadState)] for this layer."""
        return self.ctx.states_bytes[self.layer_idx]

    # ------------------------------------------------------------------
    # Update path: sink fill (Python) + recent slide (Python) + middle (C++)
    # ------------------------------------------------------------------
    def update(
        self,
        new_k: torch.Tensor,      # [B, L_in, H, D] bf16 — RAW (pre-RoPE)
        new_v: torch.Tensor,      # [B, L_in, H, D] bf16
        current_start: int,
        *,
        cache_update_mode: str = "clean",
        freqs: torch.Tensor | None = None,  # accepted for API compat; ignored
    ) -> None:
        """Insert new frames into the per-layer cache.

        Args:
            new_k, new_v: incoming K/V for this block, shape [B, L_in, H, D]
                where L_in is a multiple of frame_seqlen. K is stored RAW
                (no ``causal_rope_apply`` upstream); the readout path applies
                a single fused 3D RoPE via ``apply_rope_to_flat_k``.
            current_start: starting token index in the absolute domain
                (= current_t * frame_seqlen).
            cache_update_mode: ``"clean"`` (or ``"default"``) commits the
                update — sink/recent counters advance, middle state machine
                runs. ``"noisy"`` writes new K/V into the tentative slot
                without advancing committed counters (`committed_vc` is the
                snapshot restored at every call), so multiple noisy
                iterations on the same block all overlay the same slot
                and the final clean pass overwrites with the canonical K.
                Middle state machine (mega_state_update) only runs on
                clean — noisy iterations don't mutate stride/cyclic/merge
                anchors.
            freqs: accepted for backward-compatible API but unused under
                raw-K storage — RoPE is applied at readout, not at update.
        """
        if cache_update_mode == "default":
            cache_update_mode = "clean"
        if cache_update_mode not in ("noisy", "clean"):
            return
        ctx = self.ctx
        FSEQ = int(ctx.frame_seqlen)
        H = int(ctx.num_heads)
        D = int(ctx.head_dim)
        assert new_k.shape == new_v.shape, "K/V shapes must match"
        B, L_in, Hk, Dk = new_k.shape
        assert B == 1, "MegaCache.update: B>1 not yet supported"
        assert Hk == H and Dk == D
        assert L_in % FSEQ == 0, f"L_in ({L_in}) not multiple of FSEQ ({FSEQ})"
        F = L_in // FSEQ

        device = new_k.device
        current_t_frame = current_start // FSEQ
        new_t_vals = torch.arange(
            current_t_frame, current_t_frame + F,
            dtype=torch.int64, device=device,
        )

        # Reshape new_k/v into per-frame views: [F, H, FSEQ, D].
        nk_frames = (
            new_k.view(1, F, FSEQ, H, D)
                 .permute(0, 1, 3, 2, 4)
                 .reshape(F, H, FSEQ, D)
                 .contiguous()
        )
        nv_frames = (
            new_v.view(1, F, FSEQ, H, D)
                 .permute(0, 1, 3, 2, 4)
                 .reshape(F, H, FSEQ, D)
                 .contiguous()
        )

        # Build per-token (t, y, x) pos tensor [F, H, FSEQ, 3] int64 once,
        # reused across sink / recent / middle / merge writes. All heads share
        # the same (y, x) within a frame; t comes from new_t_vals.
        W = int(ctx.spatial_width)
        idx_seq = torch.arange(FSEQ, dtype=torch.int64, device=device)
        ys = (idx_seq // W).view(1, 1, FSEQ)
        xs = (idx_seq %  W).view(1, 1, FSEQ)
        new_pos_frames = torch.empty(
            F, H, FSEQ, 3, dtype=torch.int64, device=device,
        )
        new_pos_frames[..., 0] = new_t_vals.view(F, 1, 1)
        new_pos_frames[..., 1] = ys
        new_pos_frames[..., 2] = xs

        # ---- Step 0: restore committed valid_count + recent_pool so noisy
        # iterations all target the same tentative slot(s) starting from
        # the same canonical pre-block state. committed_vc + committed
        # recent_k/v are updated only on clean passes, so multiple noisy
        # passes for the same block restore the same counters AND the same
        # pool contents, then overwrite the same slots. This mirrors
        # AdaptiveKVCache.fast_path which extends the ring exactly once
        # per block (on first noisy via slow path) and then overwrites the
        # tail in place on subsequent noisy/clean.
        vc = ctx.mgr.valid_count()
        sink_k = ctx.mgr.sink_k_pool()
        sink_v = ctx.mgr.sink_v_pool()
        sink_pos = ctx.mgr.sink_pos_pool()
        rec_k = ctx.mgr.recent_k_pool()
        rec_v = ctx.mgr.recent_v_pool()
        rec_pos = ctx.mgr.recent_pos_pool()
        if ctx.committed_vc is None:
            ctx.committed_vc = vc.clone()
        else:
            vc[self.layer_idx].copy_(ctx.committed_vc[self.layer_idx])
        if ctx.committed_recent_k is None:
            ctx.committed_recent_k = rec_k.clone()
            ctx.committed_recent_v = rec_v.clone()
            ctx.committed_recent_pos = rec_pos.clone()
        else:
            # Restore only this layer's slice to avoid touching unrelated layers.
            rec_k[self.layer_idx].copy_(ctx.committed_recent_k[self.layer_idx])
            rec_v[self.layer_idx].copy_(ctx.committed_recent_v[self.layer_idx])
            rec_pos[self.layer_idx].copy_(ctx.committed_recent_pos[self.layer_idx])

        # ---- Steps 1+2: per-head sink + recent fill ----
        # Each head has its own sink_capacity (M5 step 5). Frames overflowing
        # a head's sink go directly to its recent_pool with per-head FIFO.
        # This matches Python AdaptiveKVCache semantics so that osc heads
        # (sink_capacity=1) have frames 1,2 in recent instead of stranded
        # in unused sink slots.
        max_sink = int(ctx.mgr.max_sink())
        max_recent = int(ctx.mgr.max_recent())

        # Read all per-head counts in one CPU sync.
        sink_caps_l = ctx.sink_caps_per_head[self.layer_idx].cpu().tolist()
        sc_pre = vc[self.layer_idx, :, 0].cpu().tolist()
        rc_pre = vc[self.layer_idx, :, 2].cpu().tolist()

        # Updated counters to write back at the end.
        sc_post = list(sc_pre)
        rc_post = list(rc_pre)

        for h in range(H):
            cap_h = int(sink_caps_l[h])
            # Effective cap is min(cap_h, max_sink) — guard against
            # mis-configured sink_capacity > pool capacity.
            cap_h = min(cap_h, max_sink)
            sc_h = int(sc_pre[h])
            rc_h = int(rc_pre[h])

            # Per-frame routing: while sink not full → sink slot; else → recent.
            sink_writes_h = max(0, min(F, cap_h - sc_h))
            recent_writes_h = F - sink_writes_h

            # Sink writes for head h.
            for i in range(sink_writes_h):
                slot = sc_h + i
                sink_k[self.layer_idx, h, slot] = nk_frames[i, h]
                sink_v[self.layer_idx, h, slot] = nv_frames[i, h]
                sink_pos[self.layer_idx, h, slot] = new_pos_frames[i, h]
            sc_post[h] = sc_h + sink_writes_h

            if recent_writes_h == 0:
                continue

            new_total = rc_h + recent_writes_h
            if new_total <= max_recent:
                for i in range(recent_writes_h):
                    src_f = sink_writes_h + i
                    slot = rc_h + i
                    rec_k[self.layer_idx, h, slot] = nk_frames[src_f, h]
                    rec_v[self.layer_idx, h, slot] = nv_frames[src_f, h]
                    rec_pos[self.layer_idx, h, slot] = new_pos_frames[src_f, h]
                rc_post[h] = new_total
            else:
                evict = new_total - max_recent
                # FIFO shift-left (per-head).
                rec_k[self.layer_idx, h, :max_recent - evict] = (
                    rec_k[self.layer_idx, h, evict:max_recent].clone()
                )
                rec_v[self.layer_idx, h, :max_recent - evict] = (
                    rec_v[self.layer_idx, h, evict:max_recent].clone()
                )
                rec_pos[self.layer_idx, h, :max_recent - evict] = (
                    rec_pos[self.layer_idx, h, evict:max_recent].clone()
                )
                for i in range(recent_writes_h):
                    src_f = sink_writes_h + i
                    slot = max_recent - recent_writes_h + i
                    rec_k[self.layer_idx, h, slot] = nk_frames[src_f, h]
                    rec_v[self.layer_idx, h, slot] = nv_frames[src_f, h]
                    rec_pos[self.layer_idx, h, slot] = new_pos_frames[src_f, h]
                rc_post[h] = max_recent

        # Single batched counter write-back to minimize host syncs.
        vc[self.layer_idx, :, 0] = torch.tensor(
            sc_post, dtype=vc.dtype, device=vc.device
        )
        vc[self.layer_idx, :, 2] = torch.tensor(
            rc_post, dtype=vc.dtype, device=vc.device
        )

        # ---- Step 3: middle via mega_state_update + pyramidkv_update ----
        # Only runs on CLEAN passes. The middle state machine
        # (stride/cyclic/lag/merge) advances per committed frame, so we
        # must not advance it on noisy denoising iterations.
        if cache_update_mode == "clean":
            from . import _ops, _mega_attention
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
                self.states_bytes_for_layer, new_t_vals,
                desc_kind, desc_slot_local, desc_src_frame, desc_src_head,
                desc_merge_slot, desc_merge_local,
                desc_merge_finalize, desc_merge_new,
                int(H), int(F), 1,  # clean pass (pass_kind=1)
            )

            # Split descriptors: kinds 0/1/2 go to pyramidkv_update (writes to
            # sink/middle/recent pools). kind 5 (DST_KIND_MERGE_ACCUM) goes
            # to mega_merge_accum (scatter-add + finalize into merge pool).
            # We unconditionally launch merge_accum to avoid a per-layer
            # .item() sync — the kernel is a no-op when no head's state is
            # SK_MERGE.
            from . import _mega_state_ops
            has_merge_head = bool(ctx.has_merge_head)

            desc_kind_pool = torch.where(
                desc_kind == 5,
                torch.tensor(-1, dtype=desc_kind.dtype, device=desc_kind.device),
                desc_kind,
            )

            max_middle = int(ctx.mgr.max_middle())
            head_global_offset = (
                self.layer_idx * H + desc_src_head.long()
            ) * max_middle
            desc_slot_global = (
                desc_slot_local.long() + head_global_offset
            ).to(torch.int32)

            # Thread per-token pos into pyramidkv_update so the kernel writes
            # middle_pos_pool in lock-step with K/V. The per-kind kernel
            # branches on dst_kind — middle writes land in middle_pos_pool,
            # masked-out (-1) writes are no-ops.
            _ops.ops().pyramidkv_update(
                ctx.mgr, nk_frames, nv_frames, new_pos_frames,
                desc_kind_pool, desc_slot_global, desc_src_frame, desc_src_head,
            )

            # Merge accumulator: route only kind == DST_KIND_MERGE_ACCUM(=5)
            # descriptors through. Other kinds get masked to -1 (skip).
            if has_merge_head:
                desc_kind_merge = torch.where(
                    desc_kind == 5,
                    desc_kind,
                    torch.tensor(-1, dtype=desc_kind.dtype, device=desc_kind.device),
                )
                # Raw-K storage: causal_rope_apply is skipped for the MegaCache
                # branch (causal_model.py), so incoming K is raw and merge_accum
                # just does a plain sum-divide. Readout applies fresh 3D RoPE
                # via apply_rope_to_flat_k.
                _mega_state_ops.mega_merge_accum_cuda(
                    mgr=ctx.mgr,
                    layer_idx=self.layer_idx,
                    states_bytes=self.states_bytes_for_layer,
                    new_k=nk_frames, new_v=nv_frames, new_pos=new_pos_frames,
                    descriptors=(
                        desc_kind_merge,
                        desc_merge_slot,
                        desc_merge_finalize,
                        desc_merge_new,
                    ),
                )

            # Commit: snapshot the (just-advanced) valid_count + recent_pool
            # (K/V/pos) for this layer. Next noisy iteration restores from
            # here, so the FIFO shift on the next block uses the clean (not
            # noisy-stale) pool contents as source — fixing the periodic
            # noisy-leak that caused "every-few-frames" temporal oscillation.
            ctx.committed_vc[self.layer_idx].copy_(vc[self.layer_idx])
            ctx.committed_recent_k[self.layer_idx].copy_(rec_k[self.layer_idx])
            ctx.committed_recent_v[self.layer_idx].copy_(rec_v[self.layer_idx])
            ctx.committed_recent_pos[self.layer_idx].copy_(rec_pos[self.layer_idx])

    # ------------------------------------------------------------------
    # Readout path: mega_plan + pyramidkv_pack + flash_attn_varlen
    # ------------------------------------------------------------------
    def attend(
        self,
        q: torch.Tensor,                # [B, L_q, H, D]
        current_start: int,
        *,
        freqs: torch.Tensor | None = None,  # ignored if ctx.freqs is set
        softmax_scale: float | None = None,
        causal: bool = False,
        max_seqlen_k: int | None = None,
    ) -> torch.Tensor:
        """Attention output from current cache state — single-launch plan +
        pack + RoPE + FA across all query chunks.

        Per-query-frame parity with Python ``AdaptiveKVCache``: each FSEQ-sized
        query chunk uses its own sync_t for sink/cyclic/stride RoPE (mirrors
        ``wan/modules/attention/core.py``'s per-chunk loop or
        ``get_decoupled_flat_kv_and_frames_multi``).

        Pipeline (one kernel/op launch each):
          1. ``mega_plan_multi_cuda(current_t_list)`` — emits chunk-concatenated
             plan descriptors with chunk-offset dst_token_offsets and
             per-chunk anchor_t_remap.
          2. ``pyramidkv_pack`` — writes all chunks' K/V/pos to disjoint regions of
             ``mgr.k_flat_out`` / ``v_flat_out`` / ``pos_flat_out``.
          3. ``apply_rope_to_flat_k`` — single fused 3D RoPE over the entire
             packed region; per-token tremap differs by chunk so each query
             frame's sink/cyclic/stride K gets its own sync_t phase.
          4. ``flash_attn_varlen_func`` — Q reordered to (chunk, head, token);
             cu_seqlens_k is already in (chunk-major, head-major) layout
             from the plan.
        """
        from ._mega_attention import _layer_offset_slot_global, _flash_attn_varlen
        from . import _mega_state_ops, _ops
        from .rope import apply_rope_to_flat_k
        ctx = self.ctx
        FSEQ = int(ctx.frame_seqlen)
        B, L_q, H, D = q.shape
        assert B == 1, "MegaCache.attend: B>1 not yet supported"
        assert L_q % FSEQ == 0, f"L_q ({L_q}) must be a multiple of FSEQ ({FSEQ})"
        eff_freqs = ctx.freqs if ctx.freqs is not None else freqs
        device = q.device

        num_chunks = L_q // FSEQ
        current_t_list = [
            (current_start + c * FSEQ) // FSEQ for c in range(num_chunks)
        ]

        # ---- 1. Multi-chunk plan (single C++ op call) ----
        (cu_seqlens_k, src_kind, src_slot_global, seg_lengths, dst_offsets,
         _anchor_t_raw, anchor_t_remap) = _mega_state_ops.mega_plan_multi_cuda(
            ctx.mgr, self.states_bytes_for_layer,
            layer_idx=self.layer_idx,
            current_t_list=current_t_list,
            sink_time_mapping_mode=ctx.sink_time_mapping_mode,
            sink_time_clamp_min=ctx.sink_time_clamp_min,
            sink_time_clamp_max=ctx.sink_time_clamp_max,
            decoupled_sink_time_lag=ctx.decoupled_sink_time_lag,
            history_time_mapping_mode=ctx.history_time_mapping_mode,
            history_relative_t_max=ctx.history_relative_t_max,
            history_time_soft_factor=ctx.history_time_soft_factor,
        )

        # ---- 2. Pack + RoPE fused: pack kernel rotates K inline ----
        # When ctx.freq_parts_flat is set, pack writes RoPE'd K directly using
        # per-segment anchor_t_remap (-1 sentinel keeps stored pos.t). No
        # separate apply_rope_to_flat_k pass is needed.
        sg_adjusted = _layer_offset_slot_global(
            src_slot_global, src_kind, self.layer_idx,
            max_sink=int(ctx.mgr.max_sink()),
            max_middle=int(ctx.mgr.max_middle()),
            max_recent=int(ctx.mgr.max_recent()),
            num_heads=H,
            max_merge_blocks=int(ctx.mgr.max_merge_blocks()),
        )
        if eff_freqs is not None and ctx.freq_parts_flat is not None:
            ft_flat, fy_flat, fx_flat = ctx.freq_parts_flat
            _ops.pyramidkv_pack(
                ctx.mgr, src_kind, sg_adjusted, seg_lengths, dst_offsets,
                anchor_t_remap=anchor_t_remap,
                freqs_ft_flat=ft_flat,
                freqs_fy_flat=fy_flat,
                freqs_fx_flat=fx_flat,
            )
        else:
            # Fallback: separate pack + RoPE (used by older fixtures without
            # freq_parts_flat on the ctx).
            _ops.pyramidkv_pack(
                ctx.mgr, src_kind, sg_adjusted, seg_lengths, dst_offsets,
            )
            if eff_freqs is not None:
                seg_lens_long = seg_lengths.long()
                tremap_per_tok = torch.repeat_interleave(
                    anchor_t_remap, seg_lens_long
                )
                n_active = int(tremap_per_tok.shape[0])
                if n_active > 0:
                    pos_3d = ctx.mgr.pos_flat_out()[:n_active].clone()
                    mask = tremap_per_tok >= 0
                    pos_3d[:, 0] = torch.where(mask, tremap_per_tok, pos_3d[:, 0])
                    k_active = ctx.mgr.k_flat_out()[:n_active].view(n_active, D)
                    apply_rope_to_flat_k(
                        k_active, pos_3d,
                        freqs=eff_freqs, freq_parts=ctx.freq_parts,
                        out=k_active,
                    )

        # ---- 4. FA (single varlen call across all chunks) ----
        # cu_seqlens_k is already in (chunk-major, head-major) layout
        # because emit_chunk_plan writes cu[c*H + h + 1] = running_offset
        # for each chunk. Q just needs matching reorder.
        q_flat = (
            q.view(B, num_chunks, FSEQ, H, D)
             .permute(0, 1, 3, 2, 4)   # [B, num_chunks, H, FSEQ, D]
             .contiguous()
             .view(B * num_chunks * H * FSEQ, 1, D)
        )
        k_flat = ctx.mgr.k_flat_out()  # full workspace; FA reads up to cu[-1]
        v_flat = ctx.mgr.v_flat_out()
        if q_flat.dtype != k_flat.dtype:
            q_flat = q_flat.to(k_flat.dtype)

        cu_seqlens_q = torch.arange(
            0, B * num_chunks * H * FSEQ + 1, FSEQ,
            dtype=torch.int32, device=device,
        )

        if softmax_scale is None:
            softmax_scale = D ** -0.5
        if max_seqlen_k is None:
            max_seqlen_k = int(ctx.mgr.max_total()) * FSEQ

        flash = _flash_attn_varlen()
        out_flat = flash(
            q_flat, k_flat, v_flat,
            cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q=FSEQ,
            max_seqlen_k=max_seqlen_k,
            dropout_p=0.0,
            softmax_scale=softmax_scale,
            causal=causal,
        )

        # ---- Reorder output back to [B, L_q, H, D] ----
        out = (
            out_flat.view(B, num_chunks, H, FSEQ, D)
            .permute(0, 1, 3, 2, 4)   # [B, num_chunks, FSEQ, H, D]
            .contiguous()
            .view(B, L_q, H, D)
            .to(q.dtype)
        )
        return out


def _composition_to_perhead_state(
    composition: HeadComposition,
) -> ref.PerHeadState:
    """Translate a HeadComposition (Python) → PerHeadState (C++ mirror).

    The composition has 0..N middle strategies but the C++ struct supports
    one strategy_kind per head. For multi-strategy compositions, this picks
    the first middle strategy. In practice pyramid forcing composes one
    middle strategy per head (cyclic, stride, lag, or merge), so this is
    not lossy for the bench configs.

    Also threads composition.sink_frames into PerHeadState.sink_capacity
    (M5 step 5) so MegaCache.update can do per-head sink routing and
    mega_plan can do per-head sink dedup.
    """
    sink_cap = int(getattr(composition, "sink_frames", 0))

    if not composition.middle_strategies:
        state = ref.make_recent()
        state.sink_capacity = sink_cap
        return state

    middle = composition.middle_strategies[0]
    name = type(middle).__name__

    if name == "CyclicStrategy":
        state = ref.make_cyclic(
            period=int(middle.period),
            bucket_cap=int(middle.bucket_cap),
        )
    elif name == "StrideStrategy":
        # Stride's "capacity" is unbounded by default (-1). Cap at the
        # C++ kMaxTKeyed bound; runtime FIFO eviction handles overflow.
        cap = int(middle.capacity)
        if cap <= 0 or cap > ref.MAX_T_KEYED:
            cap = ref.MAX_T_KEYED
        state = ref.make_stride(
            interval=int(middle.interval),
            capacity=cap,
        )
    elif name == "LagStrategy":
        hist = int(getattr(middle, "history_frames", ref.MAX_T_KEYED))
        if hist > ref.MAX_T_KEYED:
            hist = ref.MAX_T_KEYED
        offsets = list(getattr(middle, "offsets", []) or [])
        state = ref.make_lag(history_frames=hist, offsets=offsets)
    elif name == "MergeStrategy":
        state = ref.make_merge(
            patch_size=int(getattr(middle, "patch_size", 2)),
            capacity=int(getattr(middle, "capacity", ref.MAX_MERGE_BLOCKS)),
        )
    else:
        # Fallback for unknown / Recent (already handled above).
        state = ref.make_recent()

    state.sink_capacity = sink_cap
    return state


def build_mega_caches(
    num_layers: int,
    num_heads: int,
    head_dim: int,
    frame_seqlen: int,
    max_sink_frames: int,
    max_middle_frames: int,
    max_recent_frames: int,
    compositions: Sequence[Sequence[HeadComposition]],
    *,
    max_attend_chunks: int = 8,
    device: str = "cuda:0",
    kv_dtype: str = "bfloat16",
    sink_time_mapping_mode: str = "lag",
    sink_time_clamp_min: int = 0,
    sink_time_clamp_max: int = 21,
    decoupled_sink_time_lag: int = 0,
    history_time_mapping_mode: str = "none",
    history_relative_t_max: int = 21,
    history_time_soft_factor: float = 1.0,
    spatial_width: int = 52,
    spatial_height: int = 30,
    freqs: torch.Tensor | None = None,
) -> list[MegaCache]:
    """Construct (mgr, states_bytes, MegaCache×L) from a compositions matrix.

    Args:
        compositions: [L][H] HeadComposition matrix. Each entry's first
            middle strategy is encoded into a PerHeadState. ``RecentStrategy``
            and missing-middle map to SK_RECENT (no middle anchors).
        max_*_frames: pool capacities, fed to PyramidKVCacheManager constructor.
        sink/history kwargs: dynamic-RoPE remap config; defaults match
            pyramid_forcing10.

    Returns ``num_layers`` MegaCache instances, all sharing one
    PyramidKVCacheManager and one [L, H * sizeof(PerHeadState)] state tensor.
    """
    if len(compositions) != num_layers:
        raise ValueError(
            f"compositions has {len(compositions)} layers, expected {num_layers}"
        )

    from . import _ops
    _ops._ensure_loaded()

    Cls = torch.classes.adahead.PyramidKVCacheManager
    mgr = Cls(
        num_layers, num_heads, head_dim, frame_seqlen,
        max_sink_frames, max_middle_frames, max_recent_frames,
        device, kv_dtype,
        int(max_attend_chunks),
    )

    # Build the L × (H * sizeof) state byte buffer.
    perhead_size = _mega_state_ops.PER_HEAD_STATE_DTYPE.itemsize
    states_np = np.zeros((num_layers, num_heads), dtype=_mega_state_ops.PER_HEAD_STATE_DTYPE)
    sink_caps_np = np.zeros((num_layers, num_heads), dtype=np.int64)
    any_merge = False
    for l in range(num_layers):
        if len(compositions[l]) != num_heads:
            raise ValueError(
                f"compositions[{l}] has {len(compositions[l])} heads, "
                f"expected {num_heads}"
            )
        per_head = [_composition_to_perhead_state(c) for c in compositions[l]]
        any_merge = any_merge or any(s.kind == ref.SK_MERGE for s in per_head)
        for h, s in enumerate(per_head):
            sink_caps_np[l, h] = int(s.sink_capacity)
        packed = _mega_state_ops.pack_states(per_head)
        states_np[l] = packed
    states_flat = np.ascontiguousarray(states_np).view(np.uint8).reshape(
        num_layers, num_heads * perhead_size
    )
    states_bytes = (
        torch.from_numpy(states_flat).clone().to(device=device).contiguous()
    )

    # Split freqs once into (ft, fy, fx) following the apply_rope_to_flat_k
    # convention. Cached on the ctx so per-call readout doesn't re-split. The
    # model's `self.freqs` is materialized on CPU and only moved to GPU lazily
    # on first forward — explicitly move to the manager's device here so
    # apply_rope_to_flat_k doesn't bounce CPU↔GPU on every readout call.
    freq_parts = None
    freq_parts_flat = None
    if freqs is not None:
        freqs = freqs.to(device=device)
        c = head_dim // 2
        split = [c - 2 * (c // 3), c // 3, c // 3]
        ft, fy, fx = freqs.split(split, dim=1)
        freq_parts = (ft.contiguous(), fy.contiguous(), fx.contiguous())
        # Flatten each part to fp32 [max_t, cols * 2] (real, imag interleaved)
        # for the fused-RoPE pack kernel. The flat layout lets a CUDA thread
        # load 2 fp32s per pair via plain index arithmetic.
        def _to_flat(z: torch.Tensor) -> torch.Tensor:
            # z is complex64 [max_t, cols]. view_as_real → [max_t, cols, 2] fp32.
            return torch.view_as_real(z.to(torch.complex64)).contiguous().view(
                z.shape[0], -1
            )
        freq_parts_flat = (
            _to_flat(ft), _to_flat(fy), _to_flat(fx),
        )

    ctx = MegaCacheCtx(
        mgr=mgr,
        states_bytes=states_bytes,
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        frame_seqlen=frame_seqlen,
        sink_time_mapping_mode=sink_time_mapping_mode,
        sink_time_clamp_min=sink_time_clamp_min,
        sink_time_clamp_max=sink_time_clamp_max,
        decoupled_sink_time_lag=decoupled_sink_time_lag,
        history_time_mapping_mode=history_time_mapping_mode,
        history_relative_t_max=history_relative_t_max,
        history_time_soft_factor=history_time_soft_factor,
        spatial_width=spatial_width,
        spatial_height=spatial_height,
        has_merge_head=any_merge,
        sink_caps_per_head=torch.from_numpy(sink_caps_np).to(device=device).contiguous(),
        freqs=freqs,
        freq_parts=freq_parts,
        freq_parts_flat=freq_parts_flat,
    )

    return [MegaCache(ctx, layer_idx=l) for l in range(num_layers)]

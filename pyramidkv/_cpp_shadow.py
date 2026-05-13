"""Shadow validator: mirror Python cache state into the C++ PyramidKVCacheManager
and check that ``torch.ops.adahead.pyramidkv_plan + pyramidkv_pack`` produces the
same V as Python ``PyramidKVCache.get_flat_kv()``.

Designed as a side-by-side validator with zero impact on the main code path
when ``PYRAMIDKV_USE_CPP_PACK`` is unset/0. Enable only for debugging the C++
pack path — see ``cpp_pack_output_enabled`` for an experimental perf knob
that is currently broken (do not enable in production).
"""
from __future__ import annotations

import os
from typing import Sequence

import torch


def cpp_pack_enabled() -> bool:
    return os.environ.get("PYRAMIDKV_USE_CPP_PACK", "0") == "1" or cpp_pack_output_enabled()


def cpp_pack_output_enabled() -> bool:
    """Replace Python pack V output with C++ pack V (perf path).

    Implies cpp_pack_enabled(). Run cpp pack and overwrite v_flat[:total]
    in-place at end of _materialize_readout_spec. The Python K path (RoPE)
    is untouched; only V is replaced.

    **WARNING — experimental, not production-ready.** Enabling this flag
    produces garbled video output: merge-anchor segments whose token count
    is not a multiple of ``frame_seqlen`` corrupt the V pool layout. Leave
    disabled unless you are actively debugging the C++ pack path.
    """
    return os.environ.get("PYRAMIDKV_USE_CPP_PACK_OUTPUT", "0") == "1"


def maybe_attach_shadow(
    *,
    layer_idx: int,
    num_heads: int,
    head_dim: int,
    frame_seqlen: int | None,
    max_sink: int,
    max_recent: int,
    max_middle: int = 24,
    device: torch.device | str = "cuda:0",
    dtype_str: str = "bfloat16",
) -> "_ShadowState | None":
    """Build a single-layer shadow manager when the env flag is on.

    Returns None when the env flag is unset, so callers add no overhead in the
    default path.
    """
    if not cpp_pack_enabled():
        return None
    if frame_seqlen is None or frame_seqlen <= 0:
        # frame_seqlen unknown until first update; bail and try again.
        return None
    # Ensure TORCH_LIBRARY(adahead) is registered before instantiating the
    # custom class. Production subprocesses (torchrun -> inference.py) don't
    # auto-trigger the JIT load — calling _ensure_loaded() here makes the
    # shadow path self-contained.
    from pyramidkv._ops import _ensure_loaded
    if not _ensure_loaded():
        import warnings
        warnings.warn("[PYRAMIDKV_SHADOW] extension load failed — skipping shadow")
        return None
    return _ShadowState(
        layer_idx=layer_idx,
        num_heads=num_heads,
        head_dim=head_dim,
        frame_seqlen=int(frame_seqlen),
        max_sink=max(1, max_sink),
        max_middle=max(1, max_middle),
        max_recent=max(1, max_recent),
        device=str(device),
        dtype_str=dtype_str,
    )


class _ShadowState:
    """Holds a per-layer PyramidKVCacheManager and mirrors Python cache state into
    its pools so `torch.ops.adahead.pyramidkv_plan + pyramidkv_pack` can be run as a
    parallel validator."""

    def __init__(
        self,
        *,
        layer_idx: int,
        num_heads: int,
        head_dim: int,
        frame_seqlen: int,
        max_sink: int,
        max_recent: int,
        max_middle: int = 1,
        device: str = "cuda:0",
        dtype_str: str = "bfloat16",
    ):
        Cls = torch.classes.adahead.PyramidKVCacheManager
        # L=1 because each AdaptiveKVCache is one layer; the manager is per-layer.
        # Plan B — shadow uses single-chunk plan/pack (no multi-chunk), so
        # max_attend_chunks=1 keeps the pack workspace minimal here.
        self.mgr = Cls(1, num_heads, head_dim, frame_seqlen, max_sink, max_middle, max_recent, device, dtype_str, 1)
        self.layer_idx = layer_idx
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.frame_seqlen = frame_seqlen
        self.max_sink = max_sink
        self.max_middle = max_middle
        self.max_recent = max_recent
        self.device = device
        self._mismatch_count = 0
        self._match_count = 0

    @property
    def mismatch_count(self) -> int:
        return self._mismatch_count

    @property
    def match_count(self) -> int:
        return self._match_count

    def mirror_middle_from_segments(self, segments) -> None:
        """Write middle-segment K/V from a _ReadoutSpec.segments list into
        manager middle pools, keyed by head index.

        Each segment has fields ``kind`` ('sink'|'middle'|'recent'), ``seq_idx``
        (= batch_idx * num_heads + head_idx), ``k``, ``v``, ``length``. We
        ignore non-middle kinds; the mirror_after_update path already covers
        sink+recent.
        """
        F = self.frame_seqlen
        D = self.head_dim
        H = self.num_heads
        middle_k_pool = self.mgr.middle_k_pool()
        middle_v_pool = self.mgr.middle_v_pool()
        vc = self.mgr.valid_count()

        # Collect per-head middle anchors first (segments may interleave heads).
        per_head_k: list[list[torch.Tensor]] = [[] for _ in range(H)]
        per_head_v: list[list[torch.Tensor]] = [[] for _ in range(H)]
        for seg in segments:
            if seg.kind != "anchor" or seg.length <= 0:
                continue
            h = int(seg.seq_idx)  # batch_size=1, so seq_idx == head_idx
            if h < 0 or h >= H:
                continue
            per_head_k[h].append(seg.k)
            per_head_v[h].append(seg.v)

        for h in range(H):
            if not per_head_v[h]:
                vc[0, h, 1] = 0
                continue
            v_cat = torch.cat(per_head_v[h], dim=0)
            n_tokens = v_cat.shape[0]
            n_frames = n_tokens // F
            n_frames = min(n_frames, int(middle_v_pool.shape[2]))
            if n_frames <= 0:
                vc[0, h, 1] = 0
                continue
            k_cat = torch.cat(per_head_k[h], dim=0)
            middle_k_pool[0, h, :n_frames].copy_(k_cat[: n_frames * F].reshape(n_frames, F, D))
            middle_v_pool[0, h, :n_frames].copy_(v_cat[: n_frames * F].reshape(n_frames, F, D))
            vc[0, h, 1] = n_frames

    def mirror_middle_from_spec_and_vflat(self, spec, v_flat: torch.Tensor) -> None:
        """Mirror BOTH spec.segments (kind=middle) AND spec.cpp_segments into
        the manager middle pool, sorted by offset within each head.

        spec.cpp_segments don't carry K/V tensors — they were materialized into
        v_flat at runtime by self._cpp_strategy_manager. We read them back via
        v_flat[offset:offset+length]. This unified path mirrors EVERY middle
        anchor regardless of source, enabling shadow assert on any config.

        K side: for cpp_segments we have no source K (only v_flat). We zero
        the corresponding middle_k slots since assert_v_matches only validates
        V. This is fine for the V-only assert; K validation is a follow-up.
        """
        F = self.frame_seqlen
        D = self.head_dim
        H = self.num_heads
        middle_v_pool = self.mgr.middle_v_pool()
        vc = self.mgr.valid_count()

        # Per-head ordered list of (offset, source_kind, length, k_or_None, v_or_view)
        per_head: list[list[tuple]] = [[] for _ in range(H)]
        for seg in spec.segments:
            if seg.kind != "anchor" or seg.length <= 0:
                continue
            h = int(seg.seq_idx)
            if 0 <= h < H:
                per_head[h].append((int(seg.offset), "py", int(seg.length), seg.v))
        for cpp_seg in spec.cpp_segments:
            if cpp_seg.length <= 0:
                continue
            h = int(cpp_seg.head_idx)
            if 0 <= h < H:
                v_view = v_flat[cpp_seg.offset:cpp_seg.offset + cpp_seg.length]
                per_head[h].append((int(cpp_seg.offset), "cpp", int(cpp_seg.length), v_view))

        for h in range(H):
            if not per_head[h]:
                vc[0, h, 1] = 0
                continue
            per_head[h].sort(key=lambda x: x[0])  # offset order matches Python pack
            v_chunks = [t[3] for t in per_head[h]]
            v_cat = torch.cat(v_chunks, dim=0)
            n_tokens = v_cat.shape[0]
            n_frames = n_tokens // F
            n_frames = min(n_frames, int(middle_v_pool.shape[2]))
            if n_frames <= 0:
                vc[0, h, 1] = 0
                continue
            middle_v_pool[0, h, :n_frames].copy_(v_cat[: n_frames * F].reshape(n_frames, F, D))
            vc[0, h, 1] = n_frames

    def mirror_after_update(
        self,
        *,
        static_k: Sequence[torch.Tensor | None],
        dynamic_k: Sequence[torch.Tensor | None],
        static_v: Sequence[torch.Tensor | None],
        dynamic_v: Sequence[torch.Tensor | None],
    ) -> None:
        """Copy Python state into manager pools.

        static_k[h] is shape [n_sink_tokens, head_dim] = [n_sink_frames * F, D].
        dynamic_k[h] is shape [n_recent_tokens, head_dim] = [n_recent_frames * F, D].
        manager pool slot [0, h, k] is [F, D] (one frame).
        """
        F = self.frame_seqlen
        D = self.head_dim
        H = self.num_heads
        sink_k_pool = self.mgr.sink_k_pool()
        sink_v_pool = self.mgr.sink_v_pool()
        recent_k_pool = self.mgr.recent_k_pool()
        recent_v_pool = self.mgr.recent_v_pool()
        vc = self.mgr.valid_count()

        for h in range(H):
            # --- sink ---
            sn_k = static_k[h]
            if sn_k is not None and sn_k.numel() > 0:
                n_sink = int(sn_k.shape[0]) // F
                n_sink = min(n_sink, self.max_sink)
                if n_sink > 0:
                    sink_k_pool[0, h, :n_sink].copy_(sn_k[: n_sink * F].reshape(n_sink, F, D))
                    sink_v_pool[0, h, :n_sink].copy_(static_v[h][: n_sink * F].reshape(n_sink, F, D))
                vc[0, h, 0] = n_sink
            else:
                vc[0, h, 0] = 0
            vc[0, h, 1] = 0  # middle empty — shadow validator only mirrors sink+recent

            # --- recent ---
            dn_k = dynamic_k[h]
            if dn_k is not None and dn_k.numel() > 0:
                n_recent = int(dn_k.shape[0]) // F
                n_recent = min(n_recent, self.max_recent)
                if n_recent > 0:
                    recent_k_pool[0, h, :n_recent].copy_(dn_k[: n_recent * F].reshape(n_recent, F, D))
                    recent_v_pool[0, h, :n_recent].copy_(dynamic_v[h][: n_recent * F].reshape(n_recent, F, D))
                vc[0, h, 2] = n_recent
            else:
                vc[0, h, 2] = 0

    def cpp_pack_v(self) -> torch.Tensor:
        """Run the C++ plan+pack and return the packed V tensor."""
        from . import _ops as _ops_mod
        ns = torch.ops.adahead
        cu, sk, sg, sl, dst = ns.pyramidkv_plan(self.mgr, 0, 0)
        _ops_mod.pyramidkv_pack(self.mgr, sk, sg, sl, dst)
        torch.cuda.synchronize()
        total = int(cu[-1].item())
        return self.mgr.v_flat_out()[:total, 0, :]

    def assert_v_matches(self, python_v_flat: torch.Tensor) -> None:
        """Compare C++ pack V to Python V; raise on mismatch.

        On first mismatch (per shadow instance) dumps a per-head diff summary
        to stderr — frames where V differs, magnitude of diff. Only the first
        few per-frame divergences are shown to keep output manageable.

        Known limitation: merge-strategy anchors carry token_count that is
        not a frame_seqlen multiple (pyramidkv/merge.py:182). These get
        truncated by mirror's `n_frames = n_tokens // F`, producing a small
        per-layer shape diff (~390 tokens at production frame_seqlen=1560).
        Production output is unaffected (Python pack path is bit-exact);
        this is a shadow-side limitation.
        """
        cpp_v = self.cpp_pack_v()
        if cpp_v.shape != python_v_flat.shape:
            self._mismatch_count += 1
            raise AssertionError(
                f"shape mismatch: cpp={tuple(cpp_v.shape)} python={tuple(python_v_flat.shape)}"
            )
        diff = (cpp_v - python_v_flat).abs().max().item()
        if diff > 0.0:
            self._mismatch_count += 1
            if self._mismatch_count == 1:
                # One-shot diagnostic on first mismatch
                import os
                if os.environ.get("PYRAMIDKV_SHADOW_DEBUG", "0") == "1":
                    self._dump_first_mismatch(cpp_v, python_v_flat)
            raise AssertionError(f"V mismatch: max abs diff = {diff}")
        self._match_count += 1

    def _dump_first_mismatch(self, cpp_v: torch.Tensor, py_v: torch.Tensor) -> None:
        """Per-frame diff dump — identifies which frame slot in each head differs."""
        import sys
        F, D = self.frame_seqlen, self.head_dim
        diff_abs = (cpp_v - py_v).abs()  # [N, D]
        # Per frame (chunk of F rows): max abs diff
        n_total_frames = diff_abs.shape[0] // F
        per_frame_max = diff_abs.view(n_total_frames, F, D).reshape(n_total_frames, -1).max(dim=1).values
        nonzero_frames = (per_frame_max > 0).nonzero(as_tuple=True)[0]
        print(f"[PYRAMIDKV_SHADOW_DEBUG] layer {self.layer_idx} first mismatch:", file=sys.stderr)
        print(f"  total_frames={n_total_frames}, mismatched_frames={len(nonzero_frames)}", file=sys.stderr)
        print(f"  first 10 mismatched frame indices: {nonzero_frames[:10].tolist()}", file=sys.stderr)
        print(f"  per-frame max diffs (first 10): {per_frame_max[nonzero_frames[:10]].tolist()}", file=sys.stderr)

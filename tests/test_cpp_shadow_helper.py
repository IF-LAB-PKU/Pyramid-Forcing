"""M1.6 part 2: validate `pyramidkv._cpp_shadow` helper API.

Drives `_ShadowState` with synthetic AdaptiveKVCache-shaped state (per-head
2D `[n_tokens, head_dim]` tensors mirroring `static_k[h]` / `dynamic_k[h]`)
and checks `assert_v_matches` accepts equivalent V output.

Tests both the noop path (env unset) and the active path (env set).
"""
from __future__ import annotations

import os

import pytest
import torch


def _try_load_or_skip():
    try:
        from pyramidkv import _ops
    except Exception as exc:
        pytest.skip(f"pyramidkv._ops import failed: {exc}")
        return None
    if not _ops._ensure_loaded():
        pytest.skip("Extension failed to load")
    return _ops


@pytest.mark.gpu
class TestCppShadowHelper:
    def test_noop_when_env_unset(self):
        """maybe_attach_shadow returns None when PYRAMIDKV_USE_CPP_PACK is unset."""
        from pyramidkv._cpp_shadow import maybe_attach_shadow, cpp_pack_enabled

        os.environ.pop("PYRAMIDKV_USE_CPP_PACK", None)
        assert not cpp_pack_enabled()
        s = maybe_attach_shadow(
            layer_idx=0, num_heads=4, head_dim=32, frame_seqlen=64,
            max_sink=2, max_recent=2, device="cuda:0",
        )
        assert s is None

    def test_attach_when_env_set(self):
        """maybe_attach_shadow returns a _ShadowState when env=1."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        _try_load_or_skip()
        from pyramidkv._cpp_shadow import maybe_attach_shadow

        os.environ["PYRAMIDKV_USE_CPP_PACK"] = "1"
        try:
            s = maybe_attach_shadow(
                layer_idx=0, num_heads=4, head_dim=32, frame_seqlen=64,
                max_sink=2, max_recent=2, device="cuda:0",
            )
            assert s is not None
            assert s.match_count == 0
            assert s.mismatch_count == 0
        finally:
            os.environ.pop("PYRAMIDKV_USE_CPP_PACK", None)

    def test_mirror_and_match_recent_only(self):
        """End-to-end: mirror Python recent state, run cpp pack, match."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        _try_load_or_skip()
        from pyramidkv._cpp_shadow import maybe_attach_shadow

        H, D, F = 4, 32, 64
        n_recent_frames = 2
        device = "cuda:0"

        os.environ["PYRAMIDKV_USE_CPP_PACK"] = "1"
        try:
            s = maybe_attach_shadow(
                layer_idx=0, num_heads=H, head_dim=D, frame_seqlen=F,
                max_sink=2, max_recent=4, device=device,
            )
            assert s is not None

            # Python state: per-head dynamic_k/v as [n_frames * F, D] 2D.
            torch.manual_seed(7)
            dyn_k_list, dyn_v_list = [], []
            for h in range(H):
                dk = torch.randn(n_recent_frames * F, D, dtype=torch.bfloat16, device=device)
                dv = torch.randn(n_recent_frames * F, D, dtype=torch.bfloat16, device=device)
                dyn_k_list.append(dk)
                dyn_v_list.append(dv)
            stat_k_list = [None] * H
            stat_v_list = [None] * H

            s.mirror_after_update(
                static_k=stat_k_list, dynamic_k=dyn_k_list,
                static_v=stat_v_list, dynamic_v=dyn_v_list,
            )

            # Python ref: head-major torch.cat over dynamic_v
            ref_v = torch.cat(dyn_v_list, dim=0)

            # Helper does the cpp pack + assert
            s.assert_v_matches(ref_v)
            assert s.match_count == 1
            assert s.mismatch_count == 0
        finally:
            os.environ.pop("PYRAMIDKV_USE_CPP_PACK", None)

    def test_mirror_middle_from_segments(self):
        """sink + middle + recent end-to-end via mirror_middle_from_segments."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        _try_load_or_skip()
        from pyramidkv._cpp_shadow import maybe_attach_shadow

        H, D, F = 4, 16, 32
        max_sink, max_middle, max_recent = 2, 3, 2
        device = "cuda:0"

        os.environ["PYRAMIDKV_USE_CPP_PACK"] = "1"
        try:
            s = maybe_attach_shadow(
                layer_idx=0, num_heads=H, head_dim=D, frame_seqlen=F,
                max_sink=max_sink, max_middle=max_middle, max_recent=max_recent,
                device=device,
            )
            assert s is not None

            torch.manual_seed(99)
            stat_k_list, stat_v_list, dyn_k_list, dyn_v_list = [], [], [], []
            mid_k_list, mid_v_list = [], []
            ref_chunks = []
            for h in range(H):
                sn_k = torch.randn(max_sink * F, D, dtype=torch.bfloat16, device=device)
                sn_v = torch.randn(max_sink * F, D, dtype=torch.bfloat16, device=device)
                mid_k = torch.randn(max_middle * F, D, dtype=torch.bfloat16, device=device)
                mid_v = torch.randn(max_middle * F, D, dtype=torch.bfloat16, device=device)
                rc_k = torch.randn(max_recent * F, D, dtype=torch.bfloat16, device=device)
                rc_v = torch.randn(max_recent * F, D, dtype=torch.bfloat16, device=device)
                stat_k_list.append(sn_k); stat_v_list.append(sn_v)
                mid_k_list.append(mid_k); mid_v_list.append(mid_v)
                dyn_k_list.append(rc_k); dyn_v_list.append(rc_v)
                # Python pack output order per head: sink → recent → middle
                ref_chunks.extend([sn_v, rc_v, mid_v])

            # Stage 1: mirror sink+recent (as update() would)
            s.mirror_after_update(
                static_k=stat_k_list, dynamic_k=dyn_k_list,
                static_v=stat_v_list, dynamic_v=dyn_v_list,
            )

            # Stage 2: build _ReadoutSegment-like objects for middle, mirror them
            class _Seg:
                def __init__(self, kind, seq_idx, k, v):
                    self.kind = kind; self.seq_idx = seq_idx
                    self.length = int(v.shape[0]); self.k = k; self.v = v
            segments = [_Seg("anchor", h, mid_k_list[h], mid_v_list[h]) for h in range(H)]
            s.mirror_middle_from_segments(segments)

            ref_v = torch.cat(ref_chunks, dim=0)
            s.assert_v_matches(ref_v)
            assert s.match_count == 1
            assert s.mismatch_count == 0
        finally:
            os.environ.pop("PYRAMIDKV_USE_CPP_PACK", None)

    def test_mirror_middle_from_spec_and_vflat(self):
        """Unified middle mirror covers spec.segments (py-supplied) AND
        spec.cpp_segments (read back from v_flat)."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        _try_load_or_skip()
        from pyramidkv._cpp_shadow import maybe_attach_shadow

        H, D, F = 2, 16, 32
        max_sink, max_middle, max_recent = 1, 4, 2
        device = "cuda:0"

        os.environ["PYRAMIDKV_USE_CPP_PACK"] = "1"
        try:
            s = maybe_attach_shadow(
                layer_idx=0, num_heads=H, head_dim=D, frame_seqlen=F,
                max_sink=max_sink, max_middle=max_middle, max_recent=max_recent,
                device=device,
            )
            assert s is not None

            # Build per-head sink + recent mirror via the regular path
            torch.manual_seed(13)
            stat_v = [torch.randn(max_sink * F, D, dtype=torch.bfloat16, device=device) for _ in range(H)]
            dyn_v = [torch.randn(max_recent * F, D, dtype=torch.bfloat16, device=device) for _ in range(H)]
            s.mirror_after_update(
                static_k=stat_v, dynamic_k=dyn_v,
                static_v=stat_v, dynamic_v=dyn_v,
            )

            # Build "spec.segments" (py middle) + "spec.cpp_segments" (cpp middle)
            # Per-head 4 middle frames: 2 from py-segments + 2 from cpp-segments,
            # interleaved in offset space so unified mirror must sort.
            class _Seg:
                def __init__(self, kind, seq_idx, offset, length, v):
                    self.kind = kind; self.seq_idx = seq_idx
                    self.offset = offset; self.length = length
                    self.k = v; self.v = v  # k unused by V-assert; reuse v
            class _CppSeg:
                def __init__(self, head_idx, offset, length):
                    self.head_idx = head_idx; self.offset = offset; self.length = length
            class _Spec:
                def __init__(self, segments, cpp_segments):
                    self.segments = segments; self.cpp_segments = cpp_segments

            # Python pack order per head: sink → recent → middle. Compose layout:
            # [sink | recent | mid_py | mid_cpp | mid_py | mid_cpp]
            ref_chunks = []
            py_segs, cpp_segs = [], []
            offset = 0
            for h in range(H):
                # sink
                ref_chunks.append(stat_v[h])
                offset += stat_v[h].shape[0]
                # recent (matches Python's "dynamic" segment placement)
                ref_chunks.append(dyn_v[h])
                offset += dyn_v[h].shape[0]
                # middle (anchors): 4 frames split py(1) + cpp(2) + py(1), interleaved offsets
                py1 = torch.randn(F, D, dtype=torch.bfloat16, device=device)
                cpp_data = torch.randn(2 * F, D, dtype=torch.bfloat16, device=device)
                py2 = torch.randn(F, D, dtype=torch.bfloat16, device=device)
                py_segs.append(_Seg("anchor", h, offset, F, py1))
                ref_chunks.append(py1)
                offset += F
                cpp_segs.append(_CppSeg(h, offset, 2 * F))
                ref_chunks.append(cpp_data)
                offset += 2 * F
                py_segs.append(_Seg("anchor", h, offset, F, py2))
                ref_chunks.append(py2)
                offset += F

            ref_v = torch.cat(ref_chunks, dim=0)
            spec = _Spec(py_segs, cpp_segs)
            # Mirror — pass ref_v as v_flat (already the desired layout)
            s.mirror_middle_from_spec_and_vflat(spec, ref_v)

            s.assert_v_matches(ref_v)
            assert s.match_count == 1
            assert s.mismatch_count == 0
        finally:
            os.environ.pop("PYRAMIDKV_USE_CPP_PACK", None)

    def test_mismatch_raises(self):
        """assert_v_matches raises AssertionError when V differs."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        _try_load_or_skip()
        from pyramidkv._cpp_shadow import maybe_attach_shadow

        H, D, F = 2, 16, 32
        device = "cuda:0"

        os.environ["PYRAMIDKV_USE_CPP_PACK"] = "1"
        try:
            s = maybe_attach_shadow(
                layer_idx=0, num_heads=H, head_dim=D, frame_seqlen=F,
                max_sink=1, max_recent=2, device=device,
            )
            assert s is not None

            torch.manual_seed(0)
            dyn_k_list = [torch.randn(F, D, dtype=torch.bfloat16, device=device) for _ in range(H)]
            dyn_v_list = [torch.randn(F, D, dtype=torch.bfloat16, device=device) for _ in range(H)]
            s.mirror_after_update(
                static_k=[None] * H, dynamic_k=dyn_k_list,
                static_v=[None] * H, dynamic_v=dyn_v_list,
            )

            # Wrong reference (different randn seed)
            wrong_ref = torch.randn_like(torch.cat(dyn_v_list, dim=0))
            with pytest.raises(AssertionError, match="V mismatch"):
                s.assert_v_matches(wrong_ref)
            assert s.mismatch_count == 1
        finally:
            os.environ.pop("PYRAMIDKV_USE_CPP_PACK", None)

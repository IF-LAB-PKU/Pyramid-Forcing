"""Tests for pyramidkv/cyclic.py — CyclicStrategy."""
from __future__ import annotations

import torch
import pytest

from pyramidkv.cyclic import CyclicStrategy
from tests.helpers import make_anchor_data, make_multi_frame_input

FS = 4  # frame_seqlen
HD = 8  # head_dim


class TestCyclicReset:
    def test_reset_creates_empty_buckets(self):
        cs = CyclicStrategy(period=6, bucket_cap=2)
        cs.reset(3)
        assert len(cs._buckets) == 3
        for seq_buckets in cs._buckets:
            assert len(seq_buckets) == 6
            for bucket in seq_buckets:
                assert len(bucket) == 0


class TestCyclicUpdate:
    def test_update_stores_in_correct_phase_bucket(self):
        cs = CyclicStrategy(period=6, bucket_cap=5)
        cs.reset(1)
        for t in [0, 3, 7]:
            k, v, pos = make_anchor_data(t, FS, HD)
            cs.update(0, k, v, pos, FS, t)
        # t=0 → phase 0, t=3 → phase 3, t=7 → phase 1
        assert len(cs._buckets[0][0]) == 1  # phase 0
        assert cs._buckets[0][0][0].t == 0
        assert len(cs._buckets[0][3]) == 1  # phase 3
        assert cs._buckets[0][3][0].t == 3
        assert len(cs._buckets[0][1]) == 1  # phase 1
        assert cs._buckets[0][1][0].t == 7

    def test_update_lru_eviction_on_cap_exceeded(self):
        cs = CyclicStrategy(period=6, bucket_cap=1)
        cs.reset(1)
        # t=0, 6, 12 all map to phase 0
        for t in [0, 6, 12]:
            k, v, pos = make_anchor_data(t, FS, HD)
            cs.update(0, k, v, pos, FS, t)
        assert len(cs._buckets[0][0]) == 1
        assert cs._buckets[0][0][0].t == 12  # only latest retained

    def test_update_multi_frame_input(self):
        cs = CyclicStrategy(period=6, bucket_cap=5)
        cs.reset(1)
        k, v, pos = make_multi_frame_input([0, 1, 2], FS, HD)
        cs.update(0, k, v, pos, FS, 0)
        # t=0 → phase 0, t=1 → phase 1, t=2 → phase 2
        assert len(cs._buckets[0][0]) == 1
        assert len(cs._buckets[0][1]) == 1
        assert len(cs._buckets[0][2]) == 1

    def test_update_skips_misaligned_length(self):
        cs = CyclicStrategy(period=6, bucket_cap=5)
        cs.reset(1)
        # Create tensor with length not divisible by frame_seqlen
        k = torch.randn(FS + 1, HD)
        v = torch.randn(FS + 1, HD)
        pos = torch.zeros(FS + 1, 3)
        cs.update(0, k, v, pos, FS, 0)
        for bucket in cs._buckets[0]:
            assert len(bucket) == 0


class TestCyclicCollect:
    def test_collect_returns_current_phase_anchors(self):
        cs = CyclicStrategy(period=6, bucket_cap=5)
        cs.reset(1)
        # Store anchors at t=0, 6, 12 (all phase 0)
        for t in [0, 6, 12]:
            k, v, pos = make_anchor_data(t, FS, HD)
            cs.update(0, k, v, pos, FS, t)
        # Collect at t=18 (phase 0), sink_max_t=-1, recent_min_t=100
        result = cs.collect(0, current_t=18, recent_min_t=100, sink_max_t=-1)
        t_vals = [a.t for a in result]
        assert 0 in t_vals
        assert 6 in t_vals
        assert 12 in t_vals

    def test_collect_excludes_sink_overlap(self):
        cs = CyclicStrategy(period=6, bucket_cap=5)
        cs.reset(1)
        for t in [0, 6, 12]:
            k, v, pos = make_anchor_data(t, FS, HD)
            cs.update(0, k, v, pos, FS, t)
        # sink_max_t=0 → t=0 excluded (t <= sink_max_t)
        result = cs.collect(0, current_t=18, recent_min_t=100, sink_max_t=0)
        t_vals = [a.t for a in result]
        assert 0 not in t_vals
        assert 6 in t_vals

    def test_collect_excludes_recent_overlap(self):
        cs = CyclicStrategy(period=6, bucket_cap=5)
        cs.reset(1)
        for t in [0, 6, 12]:
            k, v, pos = make_anchor_data(t, FS, HD)
            cs.update(0, k, v, pos, FS, t)
        # recent_min_t=10 → t=12 excluded (t >= recent_min_t)
        result = cs.collect(0, current_t=18, recent_min_t=10, sink_max_t=-1)
        t_vals = [a.t for a in result]
        assert 12 not in t_vals
        assert 6 in t_vals

    def test_collect_empty_for_wrong_phase(self):
        cs = CyclicStrategy(period=6, bucket_cap=5)
        cs.reset(1)
        # Only store at t=0 (phase 0)
        k, v, pos = make_anchor_data(0, FS, HD)
        cs.update(0, k, v, pos, FS, 0)
        # Collect at t=1 (phase 1) → nothing
        result = cs.collect(0, current_t=1, recent_min_t=100, sink_max_t=-1)
        assert result == []

    def test_collect_equivalent_with_contiguous_store_opt_in_on_cpu(self, monkeypatch):
        monkeypatch.setenv("PYRAMIDKV_CONTIG_ANCHOR_STORE", "1")
        baseline = CyclicStrategy(period=6, bucket_cap=5)
        opt_in = CyclicStrategy(period=6, bucket_cap=5)
        baseline.reset(1)
        opt_in.reset(1)
        frames = [make_anchor_data(t, FS, HD) for t in [0, 6, 12]]
        monkeypatch.delenv("PYRAMIDKV_CONTIG_ANCHOR_STORE", raising=False)
        for t, (k, v, pos) in zip([0, 6, 12], frames):
            baseline.update(0, k, v, pos, FS, t)
        monkeypatch.setenv("PYRAMIDKV_CONTIG_ANCHOR_STORE", "1")
        for t, (k, v, pos) in zip([0, 6, 12], frames):
            opt_in.update(0, k, v, pos, FS, t)

        expected = baseline.collect(0, current_t=18, recent_min_t=100, sink_max_t=-1)
        actual = opt_in.collect(0, current_t=18, recent_min_t=100, sink_max_t=-1)
        assert [a.t for a in actual] == [a.t for a in expected]
        for got, want in zip(actual, expected):
            assert torch.equal(got.k, want.k)
            assert torch.equal(got.v, want.v)
            assert torch.equal(got.pos, want.pos)
            assert got.source_kind == "tensor"
        stats = opt_in.pop_anchor_store_stats()
        assert stats["anchor_store_fallback_count"] > 0.0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_contiguous_store_accepts_non_contiguous_cuda_sources(self, monkeypatch):
        monkeypatch.setenv("PYRAMIDKV_CONTIG_ANCHOR_STORE", "1")
        cs = CyclicStrategy(period=6, bucket_cap=5)
        cs.reset(1)

        k = torch.randn(FS, HD * 2, device="cuda")[:, ::2]
        v = torch.randn(FS, HD * 2, device="cuda")[:, ::2]
        pos = torch.zeros(FS, 6, device="cuda", dtype=torch.long)[:, ::2]
        pos[:, 0] = 0
        pos[:, 1] = torch.arange(FS, device="cuda")

        assert not k.is_contiguous()
        assert not v.is_contiguous()
        assert not pos.is_contiguous()
        cs.update(0, k, v, pos, FS, 0)
        result = cs.collect(0, current_t=0, recent_min_t=100, sink_max_t=-1)
        stats = cs.pop_anchor_store_stats()

        assert len(result) == 1
        assert result[0].source_kind == "anchor_store"
        assert torch.equal(result[0].k, k)
        assert torch.equal(result[0].v, v)
        assert torch.equal(result[0].pos, pos)
        assert stats["anchor_store_update_count"] == 1.0
        assert stats["anchor_store_fallback_count"] == 0.0


class TestCyclicEdgeCases:
    def test_period_clamped_to_minimum_1(self):
        cs = CyclicStrategy(period=0)
        assert cs.period == 1
        cs2 = CyclicStrategy(period=-5)
        assert cs2.period == 1

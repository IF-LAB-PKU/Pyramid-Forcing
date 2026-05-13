"""Tests for headkv/stride.py — StrideStrategy."""
from __future__ import annotations

import torch
import pytest

from headkv.stride import StrideStrategy
from tests.helpers import make_anchor_data

FS = 4  # frame_seqlen
HD = 8  # head_dim


class TestStrideReset:
    def test_reset_creates_empty_dicts(self):
        ss = StrideStrategy(interval=6)
        ss.reset(2)
        assert len(ss._anchors) == 2
        for store in ss._anchors:
            assert store == {}


class TestStrideUpdate:
    def test_update_stores_only_aligned_frames(self):
        ss = StrideStrategy(interval=6)
        ss.reset(1)
        for t in range(13):  # 0..12
            k, v, pos = make_anchor_data(t, FS, HD)
            ss.update(0, k, v, pos, FS, t)
        assert set(ss._anchors[0].keys()) == {0, 6, 12}

    def test_update_overwrites_same_t(self):
        ss = StrideStrategy(interval=6)
        ss.reset(1)
        k1, v1, pos1 = make_anchor_data(6, FS, HD)
        ss.update(0, k1, v1, pos1, FS, 6)
        k2, v2, pos2 = make_anchor_data(6, FS, HD)
        ss.update(0, k2, v2, pos2, FS, 6)
        assert len(ss._anchors[0]) == 1
        # The second update should overwrite
        assert torch.equal(ss._anchors[0][6].k, k2)

    def test_update_skips_misaligned_length(self):
        ss = StrideStrategy(interval=6)
        ss.reset(1)
        k = torch.randn(FS + 1, HD)
        v = torch.randn(FS + 1, HD)
        pos = torch.zeros(FS + 1, 3)
        ss.update(0, k, v, pos, FS, 0)
        assert ss._anchors[0] == {}


class TestStrideCollect:
    def test_collect_excludes_sink_and_recent(self):
        ss = StrideStrategy(interval=6)
        ss.reset(1)
        for t in [0, 6, 12, 18]:
            k, v, pos = make_anchor_data(t, FS, HD)
            ss.update(0, k, v, pos, FS, t)
        # sink_max_t=0 → exclude t=0; recent_min_t=15 → exclude t=18
        result = ss.collect(0, current_t=20, recent_min_t=15, sink_max_t=0)
        t_vals = [a.t for a in result]
        assert t_vals == [6, 12]

    def test_collect_sorted_by_t(self):
        ss = StrideStrategy(interval=6)
        ss.reset(1)
        # Insert in reverse order
        for t in [12, 0, 6]:
            k, v, pos = make_anchor_data(t, FS, HD)
            ss.update(0, k, v, pos, FS, t)
        result = ss.collect(0, current_t=20, recent_min_t=100, sink_max_t=-1)
        t_vals = [a.t for a in result]
        assert t_vals == [0, 6, 12]

    def test_collect_equivalent_with_contiguous_store_opt_in_on_cpu(self, monkeypatch):
        baseline = StrideStrategy(interval=6, capacity=4)
        opt_in = StrideStrategy(interval=6, capacity=4)
        baseline.reset(1)
        opt_in.reset(1)
        frames = [make_anchor_data(t, FS, HD) for t in range(19)]
        monkeypatch.delenv("HEADKV_CONTIG_ANCHOR_STORE", raising=False)
        for t, (k, v, pos) in enumerate(frames):
            baseline.update(0, k, v, pos, FS, t)
        monkeypatch.setenv("HEADKV_CONTIG_ANCHOR_STORE", "1")
        for t, (k, v, pos) in enumerate(frames):
            opt_in.update(0, k, v, pos, FS, t)

        expected = baseline.collect(0, current_t=20, recent_min_t=100, sink_max_t=-1)
        actual = opt_in.collect(0, current_t=20, recent_min_t=100, sink_max_t=-1)
        assert [a.t for a in actual] == [a.t for a in expected]
        for got, want in zip(actual, expected):
            assert torch.equal(got.k, want.k)
            assert torch.equal(got.v, want.v)
            assert torch.equal(got.pos, want.pos)
            assert got.source_kind == "tensor"
        stats = opt_in.pop_anchor_store_stats()
        assert stats["anchor_store_fallback_count"] > 0.0


class TestStrideSelectFrameIds:
    def test_select_frame_ids_basic(self):
        ids = StrideStrategy.select_frame_ids(21, interval=6)
        assert ids == [0, 6, 12, 18]

    def test_select_frame_ids_zero_frames(self):
        ids = StrideStrategy.select_frame_ids(0, interval=6)
        assert ids == []


class TestStrideEdgeCases:
    def test_interval_clamped_to_minimum_1(self):
        ss = StrideStrategy(interval=0)
        assert ss.interval == 1
        ss2 = StrideStrategy(interval=-3)
        assert ss2.interval == 1

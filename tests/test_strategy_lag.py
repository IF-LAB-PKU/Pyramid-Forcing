"""Tests for headkv/lag.py — LagStrategy."""
from __future__ import annotations

import torch
import pytest

from headkv.lag import LagStrategy
from tests.helpers import make_anchor_data, make_multi_frame_input

FS = 4  # frame_seqlen
HD = 8  # head_dim


class TestLagReset:
    def test_reset_creates_empty_lists(self):
        ls = LagStrategy(offsets=[6])
        ls.reset(2)
        assert len(ls._anchors) == 2
        for anchors in ls._anchors:
            assert anchors == {}


class TestLagUpdate:
    def test_update_stores_sorted_by_t(self):
        ls = LagStrategy(offsets=[6], history_frames=10)
        ls.reset(1)
        for t in [5, 2, 8]:
            k, v, pos = make_anchor_data(t, FS, HD)
            ls.update(0, k, v, pos, FS, t)
        t_vals = sorted(ls._anchors[0].keys())
        assert t_vals == [2, 5, 8]

    def test_update_deduplicates_same_t(self):
        ls = LagStrategy(offsets=[6], history_frames=10)
        ls.reset(1)
        k1, v1, pos1 = make_anchor_data(5, FS, HD)
        ls.update(0, k1, v1, pos1, FS, 5)
        k2, v2, pos2 = make_anchor_data(5, FS, HD)
        ls.update(0, k2, v2, pos2, FS, 5)
        assert len(ls._anchors[0]) == 1
        assert list(ls._anchors[0].keys()) == [5]

    def test_update_evicts_oldest_when_history_full(self):
        ls = LagStrategy(offsets=[1], history_frames=3)
        ls.reset(1)
        for t in [0, 1, 2, 3]:
            k, v, pos = make_anchor_data(t, FS, HD)
            ls.update(0, k, v, pos, FS, t)
        # history_frames is at least max(offsets)+1=2, but set to 3
        t_vals = list(ls._anchors[0].keys())
        assert len(t_vals) == 3
        assert 0 not in t_vals  # oldest evicted

    def test_history_auto_expanded_for_large_offset(self):
        ls = LagStrategy(offsets=[10], history_frames=5)
        # history should be auto-expanded to max(offsets)+1 = 11
        assert ls.history_frames >= 11


class TestLagCollect:
    def test_collect_at_current_minus_offset(self):
        ls = LagStrategy(offsets=[6], history_frames=21)
        ls.reset(1)
        for t in range(11):
            k, v, pos = make_anchor_data(t, FS, HD)
            ls.update(0, k, v, pos, FS, t)
        # collect at t=10, offset=6 → target=4
        result = ls.collect(0, current_t=10, recent_min_t=100, sink_max_t=-1)
        assert len(result) == 1
        assert result[0].t == 4

    def test_collect_multiple_offsets(self):
        ls = LagStrategy(offsets=[3, 6], history_frames=21)
        ls.reset(1)
        for t in range(13):
            k, v, pos = make_anchor_data(t, FS, HD)
            ls.update(0, k, v, pos, FS, t)
        result = ls.collect(0, current_t=12, recent_min_t=100, sink_max_t=-1)
        t_vals = sorted(a.t for a in result)
        assert t_vals == [6, 9]  # 12-6=6, 12-3=9

    def test_collect_skips_negative_target(self):
        ls = LagStrategy(offsets=[6], history_frames=21)
        ls.reset(1)
        k, v, pos = make_anchor_data(3, FS, HD)
        ls.update(0, k, v, pos, FS, 3)
        # collect at t=3, offset=6 → target=-3 → skipped
        result = ls.collect(0, current_t=3, recent_min_t=100, sink_max_t=-1)
        assert result == []

    def test_collect_excludes_sink_overlap(self):
        ls = LagStrategy(offsets=[6], history_frames=21)
        ls.reset(1)
        for t in range(11):
            k, v, pos = make_anchor_data(t, FS, HD)
            ls.update(0, k, v, pos, FS, t)
        # collect at t=6, offset=6 → target=0; sink_max_t=0 → excluded
        result = ls.collect(0, current_t=6, recent_min_t=100, sink_max_t=0)
        assert result == []

    def test_collect_excludes_recent_overlap(self):
        ls = LagStrategy(offsets=[3], history_frames=21)
        ls.reset(1)
        for t in range(11):
            k, v, pos = make_anchor_data(t, FS, HD)
            ls.update(0, k, v, pos, FS, t)
        # collect at t=10, offset=3 → target=7; recent_min_t=7 → excluded
        result = ls.collect(0, current_t=10, recent_min_t=7, sink_max_t=-1)
        assert result == []

    def test_collect_equivalent_with_contiguous_store_opt_in_on_cpu(self, monkeypatch):
        baseline = LagStrategy(offsets=[3, 6], history_frames=21)
        opt_in = LagStrategy(offsets=[3, 6], history_frames=21)
        baseline.reset(1)
        opt_in.reset(1)
        frames = [make_anchor_data(t, FS, HD) for t in range(13)]
        monkeypatch.delenv("HEADKV_CONTIG_ANCHOR_STORE", raising=False)
        for t, (k, v, pos) in enumerate(frames):
            baseline.update(0, k, v, pos, FS, t)
        monkeypatch.setenv("HEADKV_CONTIG_ANCHOR_STORE", "1")
        for t, (k, v, pos) in enumerate(frames):
            opt_in.update(0, k, v, pos, FS, t)

        expected = baseline.collect(0, current_t=12, recent_min_t=100, sink_max_t=-1)
        actual = opt_in.collect(0, current_t=12, recent_min_t=100, sink_max_t=-1)
        assert [a.t for a in actual] == [a.t for a in expected]
        for got, want in zip(actual, expected):
            assert torch.equal(got.k, want.k)
            assert torch.equal(got.v, want.v)
            assert torch.equal(got.pos, want.pos)
            assert got.source_kind == "tensor"
        stats = opt_in.pop_anchor_store_stats()
        assert stats["anchor_store_fallback_count"] > 0.0


class TestLagEdgeCases:
    def test_offsets_deduped_and_filtered(self):
        ls = LagStrategy(offsets=[6, 3, 6, 0, -1])
        # 0 and -1 are filtered (must be > 0), duplicates removed
        assert ls.offsets == [3, 6]

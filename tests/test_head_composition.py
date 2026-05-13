"""Tests for pyramidkv/base.py — HeadComposition."""
from __future__ import annotations

import torch
import pytest

from pyramidkv.base import HeadComposition, MiddleStrategy
from pyramidkv.cyclic import CyclicStrategy
from pyramidkv.lag import LagStrategy
from pyramidkv.stride import StrideStrategy
from tests.helpers import make_anchor_data

FS = 4
HD = 8


class TestHasMiddle:
    def test_has_middle_true_with_strategies(self):
        hc = HeadComposition(
            name="test", label=-1, sink_frames=1, recent_frames=4,
            middle_strategies=[CyclicStrategy()],
        )
        assert hc.has_middle is True

    def test_has_middle_false_without_strategies(self):
        hc = HeadComposition(
            name="test", label=1, sink_frames=1, recent_frames=4,
            middle_strategies=[],
        )
        assert hc.has_middle is False


class TestResetAll:
    def test_reset_all_delegates(self):
        cyclic = CyclicStrategy(period=6)
        lag = LagStrategy(offsets=[6])
        hc = HeadComposition(
            name="test", label=-1, sink_frames=1, recent_frames=4,
            middle_strategies=[cyclic, lag],
        )
        hc.reset_all(num_seq=2)
        assert len(cyclic._buckets) == 2
        assert len(lag._anchors) == 2


class TestUpdateAll:
    def test_update_all_delegates(self):
        cyclic = CyclicStrategy(period=6, bucket_cap=5)
        stride = StrideStrategy(interval=6)
        hc = HeadComposition(
            name="test", label=-1, sink_frames=1, recent_frames=4,
            middle_strategies=[cyclic, stride],
        )
        hc.reset_all(num_seq=1)
        k, v, pos = make_anchor_data(6, FS, HD)
        hc.update_all(0, k, v, pos, FS, 6)
        # cyclic: t=6 → phase 0
        assert len(cyclic._buckets[0][0]) == 1
        # stride: t=6 % 6 == 0 → stored
        assert 6 in stride._anchors[0]


class TestCollectAll:
    def _setup_hc_with_data(self):
        """Create a HeadComposition with cyclic+lag, feed data, return it."""
        cyclic = CyclicStrategy(period=6, bucket_cap=5)
        lag = LagStrategy(offsets=[3], history_frames=21)
        hc = HeadComposition(
            name="test", label=-1, sink_frames=1, recent_frames=4,
            middle_strategies=[cyclic, lag],
        )
        hc.reset_all(num_seq=1)
        # Feed frames t=0..12
        for t in range(13):
            k, v, pos = make_anchor_data(t, FS, HD)
            hc.update_all(0, k, v, pos, FS, t)
        return hc

    def test_collect_all_deduplicates_by_t(self):
        # Both cyclic and lag might return the same t
        cyclic = CyclicStrategy(period=6, bucket_cap=5)
        lag = LagStrategy(offsets=[6], history_frames=21)
        hc = HeadComposition(
            name="test", label=-1, sink_frames=1, recent_frames=4,
            middle_strategies=[cyclic, lag],
        )
        hc.reset_all(num_seq=1)
        for t in range(13):
            k, v, pos = make_anchor_data(t, FS, HD)
            hc.update_all(0, k, v, pos, FS, t)
        # At t=12: cyclic phase=0 might have t=6; lag offset=6 → t=6
        result = hc.collect_all(0, current_t=12, recent_min_t=100, sink_max_t=-1)
        t_vals = [r.t for r in result]
        # t=6 should appear only once
        assert t_vals.count(6) <= 1

    def test_collect_all_unions_non_overlapping(self):
        hc = self._setup_hc_with_data()
        # At t=12: cyclic phase=0 → t=0,6,12; lag offset=3 → t=9
        result = hc.collect_all(0, current_t=12, recent_min_t=100, sink_max_t=-1)
        t_vals = [r.t for r in result]
        # Should include both cyclic and lag results (union)
        assert len(t_vals) >= 2  # at least something from both strategies

    def test_collect_all_sorted_by_t(self):
        hc = self._setup_hc_with_data()
        result = hc.collect_all(0, current_t=12, recent_min_t=100, sink_max_t=-1)
        t_vals = [r.t for r in result]
        assert t_vals == sorted(t_vals)

    def test_collect_all_carries_dynamic_rope_flag(self):
        hc = self._setup_hc_with_data()
        result = hc.collect_all(0, current_t=12, recent_min_t=100, sink_max_t=-1)
        for item in result:
            assert item.kind == "frame"
            assert isinstance(item.dynamic_rope, bool)

    def test_collect_all_filters_overlap_and_sorts_across_strategies(self):
        hc = self._setup_hc_with_data()
        result = hc.collect_all(0, current_t=12, recent_min_t=10, sink_max_t=0)
        assert [item.t for item in result] == [6, 9]

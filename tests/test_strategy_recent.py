"""Tests for pyramidkv/recent.py — RecentStrategy."""
from __future__ import annotations

import torch
import pytest

from pyramidkv.recent import RecentStrategy
from pyramidkv.base import MiddleStrategy
from tests.helpers import make_anchor_data

FS = 4
HD = 8


class TestRecentStrategy:
    def test_collect_returns_empty(self):
        rs = RecentStrategy()
        rs.reset(1)
        result = rs.collect(0, current_t=10, recent_min_t=5, sink_max_t=0)
        assert result == []

    def test_update_is_noop(self):
        rs = RecentStrategy()
        rs.reset(1)
        k, v, pos = make_anchor_data(5, FS, HD)
        # Should not raise
        rs.update(0, k, v, pos, FS, 5)

    def test_reset_is_noop(self):
        rs = RecentStrategy()
        # Should not raise
        rs.reset(0)
        rs.reset(100)

    def test_conforms_to_protocol(self):
        rs = RecentStrategy()
        assert isinstance(rs, MiddleStrategy)

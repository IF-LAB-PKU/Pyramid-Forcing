"""Tests for headkv/merge.py — MergeStrategy."""
from __future__ import annotations

import torch
import pytest

from headkv.merge import MergeStrategy
from tests.helpers import make_multi_frame_input

FS = 4
HD = 8


class TestMergeReset:
    def test_reset_creates_empty_block_maps(self):
        ms = MergeStrategy(patch_size=2, capacity=2)
        ms.reset(2)
        assert len(ms._blocks) == 2
        assert all(blocks == {} for blocks in ms._blocks)


class TestMergeCollect:
    def test_collect_returns_completed_block_only(self):
        ms = MergeStrategy(patch_size=2, capacity=2, dynamic_rope=True)
        ms.reset(1)
        k, v, pos = make_multi_frame_input([0, 1, 2, 3], frame_seqlen=FS, head_dim=HD)
        ms.update(0, k, v, pos, FS, 0)

        collected = ms.collect(0, current_t=8, recent_min_t=10, sink_max_t=-1)
        assert len(collected) == 1
        anchor = collected[0]
        assert anchor.kind == "merge"
        assert anchor.t == 1
        assert anchor.dynamic_rope is False
        assert anchor.token_count == 2
        assert anchor.k.shape == (2, HD)
        assert anchor.v.shape == (2, HD)
        assert anchor.pos[:, 0].tolist() == [1, 1]
        block = next(iter(ms._blocks[0].values()))
        assert block.sum_k is None
        assert block.sum_v is None
        assert all(block.seen_slots)

    def test_collect_skips_incomplete_or_recent_blocks(self):
        ms = MergeStrategy(patch_size=2, capacity=2)
        ms.reset(1)
        k, v, pos = make_multi_frame_input([0, 1, 2], frame_seqlen=FS, head_dim=HD)
        ms.update(0, k, v, pos, FS, 0)
        assert ms.collect(0, current_t=8, recent_min_t=10, sink_max_t=-1) == []

        k, v, pos = make_multi_frame_input([3], frame_seqlen=FS, head_dim=HD)
        ms.update(0, k, v, pos, FS, 3)
        assert ms.collect(0, current_t=8, recent_min_t=3, sink_max_t=-1) == []

    def test_capacity_keeps_only_newest_completed_blocks(self):
        ms = MergeStrategy(patch_size=2, capacity=1)
        ms.reset(1)
        for start_t in (0, 4):
            k, v, pos = make_multi_frame_input(
                [start_t, start_t + 1, start_t + 2, start_t + 3],
                frame_seqlen=FS,
                head_dim=HD,
            )
            ms.update(0, k, v, pos, FS, start_t)
        collected = ms.collect(0, current_t=12, recent_min_t=20, sink_max_t=-1)
        assert len(collected) == 1
        assert collected[0].t == 5

    def test_capacity_minus_one_keeps_all_completed_blocks(self):
        ms = MergeStrategy(patch_size=2, capacity=-1)
        ms.reset(1)
        for start_t in (0, 4):
            k, v, pos = make_multi_frame_input(
                [start_t, start_t + 1, start_t + 2, start_t + 3],
                frame_seqlen=FS,
                head_dim=HD,
            )
            ms.update(0, k, v, pos, FS, start_t)
        collected = ms.collect(0, current_t=12, recent_min_t=20, sink_max_t=-1)
        assert [anchor.t for anchor in collected] == [1, 5]

    def test_patch_groups_follow_row_major_order(self):
        ms = MergeStrategy(patch_size=2, capacity=1)
        source_pos = torch.tensor(
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [0, 1, 1],
                [0, 2, 0],
                [0, 2, 1],
                [0, 3, 0],
                [0, 3, 1],
            ],
            dtype=torch.long,
        )
        group_ids, output_pos = ms._build_patch_groups(source_pos, t_value=1)
        assert group_ids.tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
        assert output_pos.tolist() == [[1, 0, 0], [1, 2, 0]]

    def test_incremental_updates_match_batched_result(self):
        t_values = [0, 1, 2, 3]
        k_frames = []
        v_frames = []
        pos_frames = []
        for frame_idx, t in enumerate(t_values):
            k_frame = torch.arange(FS * HD, dtype=torch.float32).view(FS, HD) + frame_idx * 100.0
            v_frame = k_frame + 0.5
            pos_frame = torch.zeros(FS, 3, dtype=torch.long)
            pos_frame[:, 0] = t
            pos_frame[:, 1] = torch.arange(FS, dtype=torch.long)
            k_frames.append(k_frame)
            v_frames.append(v_frame)
            pos_frames.append(pos_frame)

        k = torch.cat(k_frames, dim=0)
        v = torch.cat(v_frames, dim=0)
        pos = torch.cat(pos_frames, dim=0)

        ms_batched = MergeStrategy(patch_size=2, capacity=2)
        ms_batched.reset(1)
        ms_batched.update(0, k, v, pos, FS, 0)
        anchor_batched = ms_batched.collect(0, current_t=8, recent_min_t=10, sink_max_t=-1)[0]

        ms_step = MergeStrategy(patch_size=2, capacity=2)
        ms_step.reset(1)
        for frame_idx, t in enumerate(t_values):
            start = frame_idx * FS
            end = start + FS
            ms_step.update(0, k[start:end], v[start:end], pos[start:end], FS, t)
        anchor_step = ms_step.collect(0, current_t=8, recent_min_t=10, sink_max_t=-1)[0]

        assert torch.allclose(anchor_step.k, anchor_batched.k)
        assert torch.allclose(anchor_step.v, anchor_batched.v)
        assert torch.equal(anchor_step.pos, anchor_batched.pos)

    def test_duplicate_block_slot_raises(self):
        ms = MergeStrategy(patch_size=2, capacity=2)
        ms.reset(1)
        k, v, pos = make_multi_frame_input([0], frame_seqlen=FS, head_dim=HD)
        ms.update(0, k, v, pos, FS, 0)
        with pytest.raises(ValueError, match="Duplicate merge frame slot"):
            ms.update(0, k, v, pos, FS, 0)

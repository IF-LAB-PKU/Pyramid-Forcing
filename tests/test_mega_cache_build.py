"""M4 Day 1 — MegaCache construction + composition→PerHeadState bridge.

Verifies that build_mega_caches encodes a heterogeneous L×H compositions
matrix into PerHeadState bytes correctly:
  - CyclicStrategy → kind=SK_CYCLIC with period/bucket_cap fields
  - StrideStrategy → kind=SK_STRIDE with interval/capacity
  - LagStrategy   → kind=SK_LAG with capacity (history_frames)
  - RecentStrategy / empty middle → kind=SK_RECENT
  - MergeStrategy  → kind=SK_MERGE with patch_size/capacity

Also checks the L MegaCache instances share the same mgr + states_bytes
(only layer_idx differs).
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():  # pragma: no cover
    pytest.skip("CUDA required", allow_module_level=True)

from pyramidkv import _ops, _mega_cache, _mega_state_ops as ops_mod, _mega_state_ref as ref
from pyramidkv.base import HeadComposition
from pyramidkv.cyclic import CyclicStrategy
from pyramidkv.stride import StrideStrategy
from pyramidkv.lag import LagStrategy
from pyramidkv.merge import MergeStrategy

_ops._ensure_loaded()  # register torch.classes.adahead.* before build_mega_caches


def _mk_comp(middle):
    """Build a minimal HeadComposition wrapping one middle strategy."""
    return HeadComposition(
        name="test", label=1,
        sink_frames=1, recent_frames=4,
        middle_strategies=[middle] if middle is not None else [],
    )


def _decode_perhead(states_bytes_for_layer, num_heads):
    """Read back PerHeadState dataclasses from a layer's uint8 buffer."""
    import numpy as np
    perhead_size = ops_mod.PER_HEAD_STATE_DTYPE.itemsize
    flat = states_bytes_for_layer.cpu().numpy()
    arr = flat.view(ops_mod.PER_HEAD_STATE_DTYPE).reshape(num_heads).copy()
    return ops_mod.unpack_states(arr)


def test_build_mega_caches_encodes_all_strategies():
    num_layers, num_heads = 2, 5  # 5 heads × 5 strategies
    head_dim, frame_seqlen = 16, 4

    # Layer 0: one head per strategy kind.
    compositions = [
        [
            _mk_comp(CyclicStrategy(period=6, bucket_cap=3)),
            _mk_comp(StrideStrategy(interval=6, capacity=4)),
            _mk_comp(LagStrategy(offsets=[3], history_frames=8)),
            _mk_comp(None),                                       # RecentStrategy fallback
            _mk_comp(MergeStrategy(patch_size=2, capacity=6)),
        ],
        # Layer 1: shuffle order to verify layer-independent encoding.
        [
            _mk_comp(StrideStrategy(interval=4, capacity=12)),
            _mk_comp(None),
            _mk_comp(CyclicStrategy(period=4, bucket_cap=2)),
            _mk_comp(MergeStrategy(patch_size=3, capacity=4)),
            _mk_comp(LagStrategy(offsets=[1, 2], history_frames=10)),
        ],
    ]

    caches = _mega_cache.build_mega_caches(
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        frame_seqlen=frame_seqlen,
        max_sink_frames=2,
        max_middle_frames=18,
        max_recent_frames=4,
        compositions=compositions,
    )

    assert len(caches) == num_layers
    # All caches share the same ctx (so same mgr and states_bytes object).
    base_ctx = caches[0].ctx
    for c in caches:
        assert c.ctx is base_ctx
    assert tuple(base_ctx.states_bytes.shape) == (
        num_layers, num_heads * ops_mod.PER_HEAD_STATE_DTYPE.itemsize,
    )

    # ---- Decode and verify each layer ----
    layer0 = _decode_perhead(base_ctx.states_bytes[0], num_heads)
    layer1 = _decode_perhead(base_ctx.states_bytes[1], num_heads)

    # Layer 0
    assert layer0[0].kind == ref.SK_CYCLIC
    assert layer0[0].period == 6 and layer0[0].bucket_cap == 3
    assert layer0[1].kind == ref.SK_STRIDE
    assert layer0[1].interval == 6 and layer0[1].capacity == 4
    assert layer0[2].kind == ref.SK_LAG
    assert layer0[2].capacity == 8
    assert layer0[3].kind == ref.SK_RECENT
    assert layer0[4].kind == ref.SK_MERGE
    assert layer0[4].patch_size == 2 and layer0[4].capacity == 6
    assert layer0[4].block_frames == 4  # = patch_size**2

    # Layer 1 (shuffled)
    assert layer1[0].kind == ref.SK_STRIDE
    assert layer1[0].interval == 4 and layer1[0].capacity == 12
    assert layer1[1].kind == ref.SK_RECENT
    assert layer1[2].kind == ref.SK_CYCLIC
    assert layer1[2].period == 4 and layer1[2].bucket_cap == 2
    assert layer1[3].kind == ref.SK_MERGE
    assert layer1[3].patch_size == 3 and layer1[3].block_frames == 9
    assert layer1[4].kind == ref.SK_LAG
    assert layer1[4].capacity == 10


def test_mega_cache_layer_view_attributes():
    """MegaCache exposes the cache-interface fields the model forward reads."""
    compositions = [
        [_mk_comp(None) for _ in range(3)] for _ in range(2)
    ]
    caches = _mega_cache.build_mega_caches(
        num_layers=2, num_heads=3, head_dim=16, frame_seqlen=4,
        max_sink_frames=2, max_middle_frames=4, max_recent_frames=2,
        compositions=compositions,
    )
    for i, c in enumerate(caches):
        assert c.layer_idx == i
        assert c.frame_seq_length == 4
        assert c.num_heads == 3
        assert c.head_dim == 16
        # states_bytes_for_layer returns the right slice.
        assert c.states_bytes_for_layer.shape == (
            3 * ops_mod.PER_HEAD_STATE_DTYPE.itemsize,
        )

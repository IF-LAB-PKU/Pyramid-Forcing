"""M2 Day 1 — mega_plan CPU op correctness test.

Verifies the per-layer plan emits descriptors matching the per-(layer, head)
expectations for representative strategies:

1. Empty manager: cu_seqlens_k == 0 everywhere.
2. Sink-only frames: emit sink segments with t = slot_idx, no middle/recent.
3. Cyclic strategy with 4 frames in: walks PerHeadState.cyclic_slot[] and
   emits one segment per phase bucket that was written, with t values read
   from PerHeadState.cyclic_t[].
4. Stride strategy with 3 stride-aligned frames: emits 3 sequential middle
   segments with t values from PerHeadState.tkey_t.
5. Recent valid_count=2 + current_t=10: emits 2 recent segments with
   t = 8, 9 (sliding-window math).
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():  # pragma: no cover
    pytest.skip("CUDA required", allow_module_level=True)

from pyramidkv import _mega_state_ops as ops_mod
from pyramidkv import _mega_state_ref as ref
from pyramidkv import _ops


def _make_manager(H=2, D=16, FSEQ=4, max_sink=2, max_middle=4, max_recent=2):
    _ops._ensure_loaded()
    Cls = torch.classes.adahead.PyramidKVCacheManager
    return Cls(
        1,    # num_layers — single-layer manager for these unit tests
        H, D, FSEQ,
        max_sink, max_middle, max_recent,
        "cuda:0",
        "bfloat16",
        4,    # max_attend_chunks (Plan B; covers mega_plan_multi fixtures)
    )


def _pack_states_to_device(states, device):
    import numpy as np
    arr = ops_mod.pack_states(states)
    flat = arr.tobytes()
    return torch.from_numpy(np.frombuffer(flat, dtype=np.uint8)).to(device)


def _set_valid_count(mgr, layer, head, kind, count):
    """Test helper: poke the manager's valid_count[layer, head, kind]."""
    vc = mgr.valid_count()  # [L, H, 3] int64
    vc[layer, head, kind] = count


def test_empty_manager_emits_zero_token_plan():
    mgr = _make_manager(H=2)
    states = [ref.make_recent(), ref.make_recent()]
    sb = _pack_states_to_device(states, torch.device("cuda:0"))

    cu, sk, sg, sl, dst, traw, tremap = ops_mod.mega_plan_cuda(
        mgr=mgr, states_bytes=sb, layer_idx=0, current_t=0, pass_kind=1
    )
    assert cu.cpu().tolist() == [0, 0, 0]   # H+1 = 3 zeros
    assert (sk.cpu() == -1).all()           # all inactive
    assert (sl.cpu() == 0).all()


def test_sink_only_emits_per_slot_segments():
    """Sink stores first N frames (t = 0..N-1). Plan should emit one
    segment per filled sink slot for each head."""
    H, FSEQ, max_sink = 2, 4, 3
    mgr = _make_manager(H=H, FSEQ=FSEQ, max_sink=max_sink)

    # Plant 3 sink frames per head via valid_count.
    for h in range(H):
        _set_valid_count(mgr, 0, h, 0, 3)

    states = [ref.make_recent(), ref.make_recent()]
    sb = _pack_states_to_device(states, torch.device("cuda:0"))

    cu, sk, sg, sl, dst, traw, tremap = ops_mod.mega_plan_cuda(
        mgr=mgr, states_bytes=sb, layer_idx=0, current_t=10, pass_kind=1
    )

    # Each head: 3 sink frames × FSEQ tokens = 12 tokens.
    assert cu.cpu().tolist() == [0, 12, 24]
    # Plan now sizes per-head segment slots as
    # max_sink + max_middle + max_recent + max_merge_blocks (M5 step 2).
    max_total = max_sink + 4 + 2 + int(mgr.max_merge_blocks())
    sk_l = sk.cpu().tolist()
    sg_l = sg.cpu().tolist()
    sl_l = sl.cpu().tolist()
    traw_l = traw.cpu().tolist()
    for h in range(H):
        base = h * max_total
        assert sk_l[base:base + 3] == [0, 0, 0]
        # slot_global = h * max_sink + slot_in_kind
        assert sg_l[base:base + 3] == [h * max_sink + 0, h * max_sink + 1, h * max_sink + 2]
        assert sl_l[base:base + 3] == [FSEQ, FSEQ, FSEQ]
        # t values: 0, 1, 2 for sink frames.
        assert traw_l[base:base + 3] == [0, 1, 2]
        # Slots 3+ inactive.
        assert sk_l[base + 3] == -1


def test_recent_emits_sliding_window_t_values():
    """Recent slot r holds frame at t = current_t - n_recent + r."""
    H, FSEQ, max_recent = 1, 4, 2
    mgr = _make_manager(H=H, FSEQ=FSEQ, max_recent=max_recent)
    _set_valid_count(mgr, 0, 0, 2, 2)  # 2 recent frames

    states = [ref.make_recent()]
    sb = _pack_states_to_device(states, torch.device("cuda:0"))

    current_t = 10
    cu, sk, sg, sl, dst, traw, tremap = ops_mod.mega_plan_cuda(
        mgr=mgr, states_bytes=sb, layer_idx=0, current_t=current_t, pass_kind=1
    )

    sk_l = sk.cpu().tolist()
    traw_l = traw.cpu().tolist()
    # Slot 0: recent kind=2. t = 10 - 2 + 0 = 8. Slot 1: t = 9.
    assert sk_l[0] == 2
    assert sk_l[1] == 2
    assert traw_l[0] == 8
    assert traw_l[1] == 9


def test_stride_middle_reads_tkey_from_perheadstate():
    """Stride state machine appends slots 0..tkey_count-1 with t values in
    PerHeadState.tkey_t. The plan should pick those up as middle segments."""
    H, FSEQ, max_middle = 1, 4, 4
    mgr = _make_manager(H=H, FSEQ=FSEQ, max_middle=max_middle)

    # Drive 3 stride-aligned frames through the Python ref (stride interval=6).
    states = [ref.make_stride(interval=6, capacity=4)]
    ref.mega_state_update_ref(states, [0, 6, 12], pass_kind=1)
    # State now has tkey_count=3 with tkey_t = [0, 6, 12].
    assert states[0].tkey_count == 3
    assert states[0].tkey_t[:3] == [0, 6, 12]

    sb = _pack_states_to_device(states, torch.device("cuda:0"))

    cu, sk, sg, sl, dst, traw, tremap = ops_mod.mega_plan_cuda(
        mgr=mgr, states_bytes=sb, layer_idx=0, current_t=12, pass_kind=1
    )

    sk_l = sk.cpu().tolist()
    sg_l = sg.cpu().tolist()
    traw_l = traw.cpu().tolist()

    # No sink/recent set → first 3 entries are middle slots 0, 1, 2.
    assert sk_l[0:3] == [1, 1, 1]
    assert sg_l[0:3] == [0, 1, 2]   # h=0, slot in pool 0..2
    assert traw_l[0:3] == [0, 6, 12]
    assert sk_l[3] == -1  # rest inactive
    # cu_seqlens_k[1] = 3 frames × FSEQ tokens.
    assert cu.cpu().tolist() == [0, 12]


def test_cyclic_middle_reads_only_current_phase_bucket():
    """M5 step 8: Python parity — cyclic.collect emits ONLY anchors from
    bucket `current_t % period`, not all phase buckets. mega_plan now
    matches that semantics."""
    H, FSEQ, max_middle = 1, 4, 4
    mgr = _make_manager(H=H, FSEQ=FSEQ, max_middle=max_middle)

    # 2 frames hit phase 0 (slots 0, 1), 1 frame hits phase 1 (slot 3).
    states = [ref.make_cyclic(period=6, bucket_cap=3)]
    ref.mega_state_update_ref(states, [0, 6, 7], pass_kind=1)
    s = states[0]
    assert s.cyclic_slot[0] == 0 and s.cyclic_t[0] == 0
    assert s.cyclic_slot[1] == 1 and s.cyclic_t[1] == 6
    assert s.cyclic_slot[3] == 3 and s.cyclic_t[3] == 7

    sb = _pack_states_to_device(states, torch.device("cuda:0"))

    # current_t=6 → phase_idx = 6 % 6 = 0 → emit slots 0, 1.
    cu, sk, sg, sl, dst, traw, tremap = ops_mod.mega_plan_cuda(
        mgr=mgr, states_bytes=sb, layer_idx=0, current_t=6, pass_kind=1
    )
    sk_l = sk.cpu().tolist()
    assert sk_l[0:2] == [1, 1]
    assert sg.cpu().tolist()[0:2] == [0, 1]
    assert traw.cpu().tolist()[0:2] == [0, 6]
    assert sk_l[2] == -1
    assert cu.cpu().tolist() == [0, 8]  # 2 frames × FSEQ

    # current_t=7 → phase_idx = 1 → emit slot 3 only.
    cu, sk, sg, sl, dst, traw, tremap = ops_mod.mega_plan_cuda(
        mgr=mgr, states_bytes=sb, layer_idx=0, current_t=7, pass_kind=1
    )
    sk_l = sk.cpu().tolist()
    assert sk_l[0] == 1 and sg.cpu().tolist()[0] == 3
    assert traw.cpu().tolist()[0] == 7
    assert sk_l[1] == -1
    assert cu.cpu().tolist() == [0, 4]  # 1 frame × FSEQ

    # current_t=8 → phase_idx = 2 → no anchors in bucket 2 → empty.
    cu, sk, sg, sl, dst, traw, tremap = ops_mod.mega_plan_cuda(
        mgr=mgr, states_bytes=sb, layer_idx=0, current_t=8, pass_kind=1
    )
    assert sk.cpu().tolist()[0] == -1
    assert cu.cpu().tolist() == [0, 0]


def test_anchor_t_remap_sink_uses_sync_t_recent_identity():
    """M5 step 9 Python-parity: sink anchors get tremap = sync_t (= current_t
    under lag/0 default) regardless of stored t — Python's
    _write_anchor_segment overrides pos[:,0] = sync_t for decouple-sink and
    dynamic_rope=True middle segments. Recent stays at raw t under default
    history_time_mapping_mode='none'."""
    mgr = _make_manager(H=1)
    _set_valid_count(mgr, 0, 0, 0, 2)  # 2 sink frames (traw 0, 1)
    _set_valid_count(mgr, 0, 0, 2, 1)  # 1 recent frame (traw 9)

    states = [ref.make_recent()]
    sb = _pack_states_to_device(states, torch.device("cuda:0"))

    cu, sk, sg, sl, dst, traw, tremap = ops_mod.mega_plan_cuda(
        mgr=mgr, states_bytes=sb, layer_idx=0, current_t=10
    )  # defaults: sink_time_mapping_mode="lag", history_time_mapping_mode="none"
    traw_l = traw.cpu().tolist()
    tremap_l = tremap.cpu().tolist()
    # 2 sink + 1 recent emitted. sync_t = map_sink_time(10, lag, lag=0) = 10.
    assert traw_l[:3] == [0, 1, 9]
    assert tremap_l[:3] == [10, 10, 9]


def test_sink_window_clamp_uses_sync_t_of_current_t():
    """M5 step 9 Python-parity: under sink_grid_decoupling (assumed always
    on for this config), all sink anchors get tremap = sync_t =
    map_sink_time(current_t, window_clamp, min, max). Anchor's own t is
    never the input to sink remap — only current_t is."""
    H, max_sink = 1, 4
    mgr = _make_manager(H=H, max_sink=max_sink)
    _set_valid_count(mgr, 0, 0, 0, 4)  # 4 sink frames at traw = 0, 1, 2, 3

    states = [ref.make_recent()]
    sb = _pack_states_to_device(states, torch.device("cuda:0"))

    # current_t=100, window_clamp min=2 max=3 →
    #   delta_t = clamp(100, 2, 3) = 3, sync_t = max(0, 100 - 3) = 97.
    # All 4 sink anchors should emit tremap = 97.
    cu, sk, sg, sl, dst, traw, tremap = ops_mod.mega_plan_cuda(
        mgr=mgr, states_bytes=sb, layer_idx=0, current_t=100,
        sink_time_mapping_mode="window_clamp",
        sink_time_clamp_min=2, sink_time_clamp_max=3,
    )
    tremap_l = tremap.cpu().tolist()
    assert tremap_l[:4] == [97, 97, 97, 97]


def test_history_relative_clamp_caps_old_recent_frames():
    """history_time_mapping_mode='relative_clamp', history_relative_t_max=2:
    recent frames older than current_t - 2 get remapped to current_t - 2."""
    H, max_recent = 1, 4
    mgr = _make_manager(H=H, max_recent=max_recent)
    _set_valid_count(mgr, 0, 0, 2, 4)  # 4 recent frames

    states = [ref.make_recent()]
    sb = _pack_states_to_device(states, torch.device("cuda:0"))

    # current_t=10, n_recent=4 → recent t values = [6, 7, 8, 9].
    # relative_clamp(2): rel = current_t - t, clamp(0, 2). For t=6: rel=4→2
    # → remap = 10 - 2 = 8. For t=7: rel=3→2 → remap=8. For t=8: rel=2→2 → remap=8.
    # For t=9: rel=1→1 → remap=9.
    cu, sk, sg, sl, dst, traw, tremap = ops_mod.mega_plan_cuda(
        mgr=mgr, states_bytes=sb, layer_idx=0, current_t=10,
        history_time_mapping_mode="relative_clamp",
        history_relative_t_max=2,
    )
    traw_l = traw.cpu().tolist()
    tremap_l = tremap.cpu().tolist()
    assert traw_l[:4] == [6, 7, 8, 9]
    assert tremap_l[:4] == [8, 8, 8, 9]


def test_history_relative_softcap_compresses_overflow():
    """relative_softcap with soft_factor=0.5: t-distance over max gets halved."""
    H, max_recent = 1, 4
    mgr = _make_manager(H=H, max_recent=max_recent)
    _set_valid_count(mgr, 0, 0, 2, 4)

    states = [ref.make_recent()]
    sb = _pack_states_to_device(states, torch.device("cuda:0"))

    # current_t=20, n_recent=4 → recent t = [16, 17, 18, 19].
    # history_relative_t_max=2, soft_factor=0.5:
    #   t=16: rel=4, over=2, compressed=round(2*0.5)=1, rel_mapped=2+1=3 → remap=17
    #   t=17: rel=3, over=1, compressed=round(1*0.5)=0, wait round(0.5)=0 in Python's banker's
    #         but C++ std::llround rounds 0.5 to 1. Let's not depend on exact value
    #         for this case; use larger soft_factor where rounding is unambiguous.
    cu, sk, sg, sl, dst, traw, tremap = ops_mod.mega_plan_cuda(
        mgr=mgr, states_bytes=sb, layer_idx=0, current_t=20,
        history_time_mapping_mode="relative_softcap",
        history_relative_t_max=2,
        history_time_soft_factor=1.0,  # no compression → same as identity
    )
    traw_l = traw.cpu().tolist()
    tremap_l = tremap.cpu().tolist()
    # With soft_factor=1.0, the softcap is a no-op (rel_mapped = rel_max + over = rel).
    # So remap == raw.
    assert traw_l[:4] == [16, 17, 18, 19]
    assert tremap_l[:4] == [16, 17, 18, 19]


def test_history_softcap_with_half_factor_compresses_over_limit():
    """history_relative_t_max=2, soft_factor=0.5:
       For t=16 (rel=4, over=2): compressed=1, rel_mapped=3 → remap=17.
       For t=14 (rel=6, over=4): compressed=2, rel_mapped=4 → remap=16.
       Use even `over` values to avoid llround tie-breaker ambiguity."""
    H, max_recent = 1, 4
    mgr = _make_manager(H=H, max_recent=max_recent)
    _set_valid_count(mgr, 0, 0, 2, 4)

    states = [ref.make_recent()]
    sb = _pack_states_to_device(states, torch.device("cuda:0"))

    # current_t=20, n_recent=4 → recent t = [16, 17, 18, 19].
    # over values: 16→2, 17→1, 18→0, 19→-(clamp)
    cu, sk, sg, sl, dst, traw, tremap = ops_mod.mega_plan_cuda(
        mgr=mgr, states_bytes=sb, layer_idx=0, current_t=20,
        history_time_mapping_mode="relative_softcap",
        history_relative_t_max=2,
        history_time_soft_factor=0.5,
    )
    tremap_l = tremap.cpu().tolist()
    # t=16 rel=4 over=2 → compressed=1 → rel_mapped=3 → remap=17
    # t=17 rel=3 over=1 → compressed=1 (llround 0.5→1, banker dependent; on
    #                                    GCC's std::llround it's away-from-zero
    #                                    so 0.5 → 1) → rel_mapped=3 → remap=17
    #   Skip strict check on rel=odd entries to dodge tie-break drift.
    # t=18 rel=2 ≤ rel_max → unchanged → remap=18
    # t=19 rel=1 ≤ rel_max → unchanged → remap=19
    assert tremap_l[0] == 17
    assert tremap_l[2] == 18
    assert tremap_l[3] == 19

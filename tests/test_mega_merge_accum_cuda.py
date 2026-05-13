"""M1 Day 5d — CUDA mega_merge_accum kernel correctness test.

Day 5d uses **identity grouping** (group_idx == token_idx). This means the
finalized merge anchor is the simple per-token mean across the 4 frames of
a block. Day 5e will port the spatial _build_patch_groups from
headkv/merge.py for bit-exact match.

Test plan
---------
1. Configure a small (H=2 heads) manager with merge_enabled.
2. Drive 4 frames of block 0 through mega_state_update — get descriptors.
3. Run mega_merge_accum with the descriptors + random K/V/pos.
4. Compare merge_k_pool[0] against a CPU fp64 reference computed as
   ``new_k.mean(dim=0)`` (mean across the 4 frames in the block).
5. Compare merge_token_count, merge_pos_pool against expected.
"""
from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():  # pragma: no cover
    pytest.skip("CUDA required", allow_module_level=True)

from headkv import _mega_state_ops as ops_mod
from headkv import _mega_state_ref as ref
from headkv import _ops


def _make_manager(H, D, FSEQ):
    Cls = torch.classes.adahead.HeadKVCacheManager
    return Cls(
        1,    # num_layers
        H,
        D,
        FSEQ,
        2,    # max_sink
        4,    # max_middle
        2,    # max_recent
        "cuda:0",
        "bfloat16",
        1,    # max_attend_chunks (Plan B; merge_accum test is single-chunk)
    )


def _per_head_state_bytes(states, device):
    arr = ops_mod.pack_states(states)
    flat = arr.tobytes()
    import numpy as np
    return torch.from_numpy(np.frombuffer(flat, dtype=np.uint8)).to(device)


def test_single_block_finalize_each_token_own_group():
    """4 frames, FSEQ tokens, positions chosen so each token gets its own
    spatial group (patch_size=2 + y=2*t makes (y/2)=t unique per token).
    Anchor[t] should equal the per-token mean across the 4 frames."""
    _ops._ensure_loaded()  # ensure JIT compiled

    H, D, FSEQ = 2, 16, 8
    F = 4
    device = torch.device("cuda:0")

    mgr = _make_manager(H, D, FSEQ)
    mgr.reset()

    states = [ref.make_merge(patch_size=2, capacity=6) for _ in range(H)]
    states_bytes = _per_head_state_bytes(states, device)

    new_t_tensor = torch.tensor([0, 1, 2, 3], dtype=torch.int64, device=device)
    N = H * F
    desc_kind = torch.full((N,), -1, dtype=torch.int32, device=device)
    desc_slot = torch.full((N,), -1, dtype=torch.int32, device=device)
    desc_frame = torch.zeros(N, dtype=torch.int32, device=device)
    desc_head = torch.zeros(N, dtype=torch.int32, device=device)
    desc_accum_slot = torch.full((N,), -1, dtype=torch.int32, device=device)
    desc_local_idx = torch.full((N,), -1, dtype=torch.int32, device=device)
    desc_finalize = torch.full((N,), -1, dtype=torch.int32, device=device)
    desc_new = torch.zeros(N, dtype=torch.int32, device=device)

    _ops.ops().mega_state_update(
        states_bytes, new_t_tensor,
        desc_kind, desc_slot, desc_frame, desc_head,
        desc_accum_slot, desc_local_idx, desc_finalize, desc_new,
        int(H), int(F), 1,
    )

    desc_kind_l = desc_kind.cpu().tolist()
    desc_new_l = desc_new.cpu().tolist()
    desc_finalize_l = desc_finalize.cpu().tolist()
    for h in range(H):
        base = h * F
        assert desc_kind_l[base:base + 4] == [ref.DST_KIND_MERGE_ACCUM] * 4
        assert desc_new_l[base:base + 4] == [1, 0, 0, 0]
        assert desc_finalize_l[base:base + 4] == [-1, -1, -1, 0]

    torch.manual_seed(42)
    new_k = torch.randn(F, H, FSEQ, D, dtype=torch.bfloat16, device=device)
    new_v = torch.randn(F, H, FSEQ, D, dtype=torch.bfloat16, device=device)
    # y=2*t, x=0  → patch_y=t, patch_x=0 → group_id=t (each token own group).
    new_pos = torch.zeros(F, H, FSEQ, 3, dtype=torch.int64, device=device)
    for f in range(F):
        new_pos[f, :, :, 0] = f  # t-value in pos[0]
        for t in range(FSEQ):
            new_pos[f, :, t, 1] = 2 * t
            new_pos[f, :, t, 2] = 0

    ops_mod.mega_merge_accum_cuda(
        mgr=mgr, layer_idx=0, states_bytes=states_bytes,
        new_k=new_k, new_v=new_v, new_pos=new_pos,
        descriptors=(desc_kind, desc_accum_slot, desc_finalize, desc_new),
    )

    # Each token in its own group → anchor[t] = mean across 4 frames for token t.
    expected_k = new_k.to(torch.float32).mean(dim=0)  # [H, FSEQ, D]
    expected_v = new_v.to(torch.float32).mean(dim=0)

    got_k = mgr.merge_k_pool()[0, :, 0, :FSEQ, :].to(torch.float32)
    got_v = mgr.merge_v_pool()[0, :, 0, :FSEQ, :].to(torch.float32)

    assert torch.allclose(got_k, expected_k, atol=2e-2, rtol=2e-2)
    assert torch.allclose(got_v, expected_v, atol=2e-2, rtol=2e-2)

    cnt = mgr.merge_token_count()[0, :, 0].cpu().tolist()
    assert cnt == [FSEQ] * H

    # Group g's output position should be (t=0, y=2g, x=0).
    pos = mgr.merge_pos_pool()[0, 0, 0, :FSEQ, :].cpu()
    for g in range(FSEQ):
        assert pos[g, 0].item() == 0   # captured at first-frame t
        assert pos[g, 1].item() == 2 * g
        assert pos[g, 2].item() == 0


def test_spatial_grouping_2x2_patches_match_python_ref():
    """4 frames × 8 tokens laid out as 4 rows × 2 cols. patch_size=2 collapses
    the 4×2 grid into 2 patches (each 2×2 box has 4 tokens). The 2 anchor
    rows = mean over (4 frames × 4 tokens per patch) for each head, dim."""
    _ops._ensure_loaded()

    H, D, FSEQ = 1, 16, 8
    F = 4
    device = torch.device("cuda:0")

    mgr = _make_manager(H, D, FSEQ)
    mgr.reset()

    states = [ref.make_merge(patch_size=2, capacity=6)]
    states_bytes = _per_head_state_bytes(states, device)

    new_t = torch.tensor([0, 1, 2, 3], dtype=torch.int64, device=device)
    N = H * F
    desc_kind = torch.full((N,), -1, dtype=torch.int32, device=device)
    desc_slot = torch.full((N,), -1, dtype=torch.int32, device=device)
    desc_frame = torch.zeros(N, dtype=torch.int32, device=device)
    desc_head = torch.zeros(N, dtype=torch.int32, device=device)
    desc_accum_slot = torch.full((N,), -1, dtype=torch.int32, device=device)
    desc_local_idx = torch.full((N,), -1, dtype=torch.int32, device=device)
    desc_finalize = torch.full((N,), -1, dtype=torch.int32, device=device)
    desc_new = torch.zeros(N, dtype=torch.int32, device=device)

    _ops.ops().mega_state_update(
        states_bytes, new_t,
        desc_kind, desc_slot, desc_frame, desc_head,
        desc_accum_slot, desc_local_idx, desc_finalize, desc_new,
        int(H), int(F), 1,
    )

    torch.manual_seed(11)
    new_k = torch.randn(F, H, FSEQ, D, dtype=torch.bfloat16, device=device)
    new_v = torch.randn(F, H, FSEQ, D, dtype=torch.bfloat16, device=device)

    # 8 tokens laid out as a 4×2 grid in (y, x):
    #   token 0: (0, 0)   ← patch (0, 0) = group 0
    #   token 1: (0, 1)   ← patch (0, 0) = group 0
    #   token 2: (1, 0)   ← patch (0, 0) = group 0
    #   token 3: (1, 1)   ← patch (0, 0) = group 0
    #   token 4: (2, 0)   ← patch (1, 0) = group 1
    #   token 5: (2, 1)   ← patch (1, 0) = group 1
    #   token 6: (3, 0)   ← patch (1, 0) = group 1
    #   token 7: (3, 1)   ← patch (1, 0) = group 1
    new_pos = torch.zeros(F, H, FSEQ, 3, dtype=torch.int64, device=device)
    coords = [(0, 0), (0, 1), (1, 0), (1, 1),
              (2, 0), (2, 1), (3, 0), (3, 1)]
    for f in range(F):
        new_pos[f, :, :, 0] = f
        for t, (y, x) in enumerate(coords):
            new_pos[f, :, t, 1] = y
            new_pos[f, :, t, 2] = x

    ops_mod.mega_merge_accum_cuda(
        mgr=mgr, layer_idx=0, states_bytes=states_bytes,
        new_k=new_k, new_v=new_v, new_pos=new_pos,
        descriptors=(desc_kind, desc_accum_slot, desc_finalize, desc_new),
    )

    # Expected: group 0 mean over (4 frames × tokens 0..3), group 1 over 4..7.
    # Mean is computed in fp32 then bf16-rounded.
    g0_tokens = new_k[:, 0, 0:4, :].to(torch.float32)  # [F, 4, D]
    g1_tokens = new_k[:, 0, 4:8, :].to(torch.float32)
    expected_k0 = g0_tokens.reshape(-1, D).mean(dim=0)  # [D]
    expected_k1 = g1_tokens.reshape(-1, D).mean(dim=0)

    got_k0 = mgr.merge_k_pool()[0, 0, 0, 0, :].to(torch.float32)
    got_k1 = mgr.merge_k_pool()[0, 0, 0, 1, :].to(torch.float32)

    assert torch.allclose(got_k0, expected_k0, atol=2e-2, rtol=2e-2), (
        f"group 0 K err: {(got_k0 - expected_k0).abs().max().item()}"
    )
    assert torch.allclose(got_k1, expected_k1, atol=2e-2, rtol=2e-2), (
        f"group 1 K err: {(got_k1 - expected_k1).abs().max().item()}"
    )

    # 2 groups total → merge_token_count[0, 0, 0] = 2.
    assert mgr.merge_token_count()[0, 0, 0].item() == 2
    # Output pos[0] = (0, 0, 0) [top-left of patch (0,0) at frame 0's t]
    # Output pos[1] = (0, 2, 0) [top-left of patch (1,0)]
    pos = mgr.merge_pos_pool()[0, 0, 0, :2, :].cpu()
    assert pos[0].tolist() == [0, 0, 0]
    assert pos[1].tolist() == [0, 2, 0]


def test_two_consecutive_blocks_use_distinct_completed_slots():
    """Block 0 → completed[0]; block 1 → completed[1]. Distinct y per token
    so each gets its own spatial group → mean-over-frames anchor."""
    _ops._ensure_loaded()

    H, D, FSEQ = 1, 16, 8
    F = 8  # 2 blocks of 4 frames
    device = torch.device("cuda:0")

    mgr = _make_manager(H, D, FSEQ)
    mgr.reset()

    states = [ref.make_merge(patch_size=2, capacity=6)]
    states_bytes = _per_head_state_bytes(states, device)

    new_t = torch.arange(F, dtype=torch.int64, device=device)
    N = H * F
    bufs = [torch.full((N,), -1, dtype=torch.int32, device=device) for _ in range(6)]
    bufs.append(torch.zeros(N, dtype=torch.int32, device=device))  # is_new_block
    desc_kind, desc_slot, desc_frame, desc_head, desc_accum_slot, desc_local_idx, desc_new = bufs
    desc_finalize = torch.full((N,), -1, dtype=torch.int32, device=device)

    _ops.ops().mega_state_update(
        states_bytes, new_t,
        desc_kind, desc_slot, desc_frame, desc_head,
        desc_accum_slot, desc_local_idx, desc_finalize, desc_new,
        int(H), int(F), 1,
    )

    assert desc_finalize.cpu().tolist() == [-1, -1, -1, 0, -1, -1, -1, 1]

    torch.manual_seed(7)
    new_k = torch.randn(F, H, FSEQ, D, dtype=torch.bfloat16, device=device)
    new_v = torch.randn(F, H, FSEQ, D, dtype=torch.bfloat16, device=device)
    new_pos = torch.zeros(F, H, FSEQ, 3, dtype=torch.int64, device=device)
    for f in range(F):
        new_pos[f, :, :, 0] = f
        for t in range(FSEQ):
            # y = 2*t makes patch_y = t — each token its own spatial group.
            new_pos[f, :, t, 1] = 2 * t
            new_pos[f, :, t, 2] = 0

    ops_mod.mega_merge_accum_cuda(
        mgr=mgr, layer_idx=0, states_bytes=states_bytes,
        new_k=new_k, new_v=new_v, new_pos=new_pos,
        descriptors=(desc_kind, desc_accum_slot, desc_finalize, desc_new),
    )

    # Block 0 mean: frames 0..3; block 1 mean: frames 4..7.
    expected_k0 = new_k[0:4].to(torch.float32).mean(dim=0)  # [H, FSEQ, D]
    expected_k1 = new_k[4:8].to(torch.float32).mean(dim=0)

    got_k0 = mgr.merge_k_pool()[0, :, 0, :FSEQ, :].to(torch.float32)
    got_k1 = mgr.merge_k_pool()[0, :, 1, :FSEQ, :].to(torch.float32)

    assert torch.allclose(got_k0, expected_k0, atol=2e-2, rtol=2e-2)
    assert torch.allclose(got_k1, expected_k1, atol=2e-2, rtol=2e-2)

    cnt = mgr.merge_token_count()[0, 0, :2].cpu().tolist()
    assert cnt == [FSEQ, FSEQ]

    # Block-0 anchor[0] position: (frame-0's t = 0, y=0, x=0)
    # Block-1 anchor[0] position: (frame-4's t = 4, y=0, x=0)
    assert mgr.merge_pos_pool()[0, 0, 0, 0, 0].item() == 0
    assert mgr.merge_pos_pool()[0, 0, 1, 0, 0].item() == 4

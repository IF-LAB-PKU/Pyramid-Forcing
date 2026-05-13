"""M5 — MegaCache noisy/clean parity.

Drives noisy iterations + final clean for the same block and verifies:
  1. Multiple noisy updates write into the same tentative slot — sink/recent
     counters don't advance past the committed snapshot.
  2. After the clean pass, sink/recent counters commit and the final K/V
     in the pool reflects the clean update.
  3. mega_state_update only runs on clean — middle anchors don't advance
     during noisy denoising iterations.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():  # pragma: no cover
    pytest.skip("CUDA required", allow_module_level=True)

from pyramidkv import _ops, _mega_cache, _mega_state_ops as ops_mod
from pyramidkv.base import HeadComposition
from pyramidkv.stride import StrideStrategy

_ops._ensure_loaded()


def _mk_stride_comp():
    return HeadComposition(
        name="stride", label=1,
        sink_frames=3, recent_frames=4,
        middle_strategies=[StrideStrategy(interval=1, capacity=8)],
    )


def _unpack_states_for_layer(cache, H):
    return ops_mod.unpack_states(
        cache.states_bytes_for_layer.cpu().numpy()
        .view(ops_mod.PER_HEAD_STATE_DTYPE).reshape(H).copy()
    )


def test_noisy_iterations_dont_advance_committed_vc_then_clean_commits():
    H, D, FSEQ, L = 1, 16, 4, 1
    max_sink, max_middle, max_recent = 3, 8, 4
    F = 3
    L_in = F * FSEQ
    device = torch.device("cuda:0")

    compositions = [[_mk_stride_comp()] for _ in range(L)]
    caches = _mega_cache.build_mega_caches(
        num_layers=L, num_heads=H, head_dim=D, frame_seqlen=FSEQ,
        max_sink_frames=max_sink,
        max_middle_frames=max_middle,
        max_recent_frames=max_recent,
        compositions=compositions,
        device="cuda:0", kv_dtype="bfloat16",
    )
    cache = caches[0]
    cache.ctx.mgr.reset()
    cache.ctx.committed_vc = None  # ensure fresh init

    torch.manual_seed(7)
    noisy_k = [torch.randn(1, L_in, H, D, dtype=torch.bfloat16, device=device) for _ in range(4)]
    noisy_v = [torch.randn(1, L_in, H, D, dtype=torch.bfloat16, device=device) for _ in range(4)]
    clean_k = torch.randn(1, L_in, H, D, dtype=torch.bfloat16, device=device)
    clean_v = torch.randn(1, L_in, H, D, dtype=torch.bfloat16, device=device)

    vc = cache.ctx.mgr.valid_count()

    # ---- Block 1: 4 noisy iterations, no commit. ----
    for i in range(4):
        cache.update(noisy_k[i], noisy_v[i], current_start=0, cache_update_mode="noisy")
        # After each noisy call, vc[layer] advanced to F (within sink). But
        # committed_vc should still be 0 — next noisy iteration will restore.
        assert int(vc[0, 0, 0].item()) == F, f"noisy iter {i}: vc[sink] {int(vc[0,0,0].item())} != {F}"
        assert int(cache.ctx.committed_vc[0, 0, 0].item()) == 0, \
            f"noisy iter {i}: committed_vc moved to {int(cache.ctx.committed_vc[0,0,0].item())}"

    # Middle state machine must NOT have advanced.
    states = _unpack_states_for_layer(cache, H)
    assert states[0].tkey_count == 0, \
        f"noisy passes mutated middle state: tkey_count={states[0].tkey_count}"

    # ---- Clean pass commits. ----
    cache.update(clean_k, clean_v, current_start=0, cache_update_mode="clean")
    assert int(vc[0, 0, 0].item()) == F
    assert int(cache.ctx.committed_vc[0, 0, 0].item()) == F, \
        "clean pass didn't commit committed_vc"

    # Sink pool slot 0..F-1 should now hold CLEAN K (not the last noisy K).
    sink_k = cache.ctx.mgr.sink_k_pool()
    # First sink slot, head 0, all positions: should equal clean K's first frame.
    expected = clean_k.view(1, F, FSEQ, H, D)[0, 0, :, 0, :]  # [FSEQ, D]
    got = sink_k[0, 0, 0, :, :]  # [FSEQ, D]
    assert torch.allclose(got.float(), expected.float(), atol=2e-2), \
        "clean pass didn't overwrite sink slot with canonical K"

    # Middle state machine advanced on clean.
    states_after_clean = _unpack_states_for_layer(cache, H)
    assert states_after_clean[0].tkey_count == F, \
        f"clean pass didn't advance middle state: tkey_count={states_after_clean[0].tkey_count}"


def test_block_2_noisy_uses_block_1_clean_committed_baseline():
    """After block 1 clean commits, block 2's noisy iterations should restore
    block 1's committed_vc, not block 2's prior noisy iteration state."""
    H, D, FSEQ, L = 1, 16, 4, 1
    max_sink, max_middle, max_recent = 3, 8, 4
    F = 3
    L_in = F * FSEQ
    device = torch.device("cuda:0")

    compositions = [[_mk_stride_comp()] for _ in range(L)]
    caches = _mega_cache.build_mega_caches(
        num_layers=L, num_heads=H, head_dim=D, frame_seqlen=FSEQ,
        max_sink_frames=max_sink, max_middle_frames=max_middle,
        max_recent_frames=max_recent, compositions=compositions,
        device="cuda:0", kv_dtype="bfloat16",
    )
    cache = caches[0]
    cache.ctx.mgr.reset()
    cache.ctx.committed_vc = None

    torch.manual_seed(11)
    k1 = torch.randn(1, L_in, H, D, dtype=torch.bfloat16, device=device)
    v1 = torch.randn(1, L_in, H, D, dtype=torch.bfloat16, device=device)
    k2_noisy = torch.randn(1, L_in, H, D, dtype=torch.bfloat16, device=device)
    v2_noisy = torch.randn(1, L_in, H, D, dtype=torch.bfloat16, device=device)

    # Block 1: clean. Fills sink (max_sink=3).
    cache.update(k1, v1, current_start=0, cache_update_mode="clean")
    assert int(cache.ctx.committed_vc[0, 0, 0].item()) == max_sink  # sink full
    assert int(cache.ctx.committed_vc[0, 0, 2].item()) == 0          # recent empty

    # Block 2: noisy. current_start=3*FSEQ, sink is full → 3 frames go to recent.
    cache.update(k2_noisy, v2_noisy, current_start=3 * FSEQ, cache_update_mode="noisy")
    vc = cache.ctx.mgr.valid_count()
    # vc[recent] advanced to 3 (tentative), but committed_vc[recent] still 0.
    assert int(vc[0, 0, 2].item()) == 3
    assert int(cache.ctx.committed_vc[0, 0, 2].item()) == 0

    # Second noisy iteration for block 2: restore committed → recent slot 0..2
    # gets overwritten with k2_noisy_b again, not appended.
    k2_noisy_b = torch.randn(1, L_in, H, D, dtype=torch.bfloat16, device=device)
    v2_noisy_b = torch.randn(1, L_in, H, D, dtype=torch.bfloat16, device=device)
    cache.update(k2_noisy_b, v2_noisy_b, current_start=3 * FSEQ, cache_update_mode="noisy")
    assert int(vc[0, 0, 2].item()) == 3  # still 3, didn't grow to 6
    assert int(cache.ctx.committed_vc[0, 0, 2].item()) == 0


def test_recent_fifo_no_noisy_contamination_across_blocks():
    """Phase B regression: drive 5+ blocks with noisy+clean interleaving.
    After the FIFO shift triggers (recent_pool overflow), verify that
    recent_pool slot 0 contains a CLEAN K from a previous block, not the
    noisy K from the immediately-prior noisy iteration. Without the
    committed_recent_pool snapshot fix, every block boundary would leak
    one noisy frame into slot 0 → periodic temporal oscillation.
    """
    H, D, FSEQ, L = 1, 16, 4, 1
    max_sink, max_middle, max_recent = 1, 8, 4  # osc-like: sink1 + recent4
    F = 3
    L_in = F * FSEQ
    device = torch.device("cuda:0")

    # osc head: sink_capacity=1, recent_frames=4.
    from pyramidkv.recent import RecentStrategy
    comp = HeadComposition(
        name="recent_only", label=-1,
        sink_frames=1, recent_frames=4,
        middle_strategies=[RecentStrategy()],
    )
    caches = _mega_cache.build_mega_caches(
        num_layers=L, num_heads=H, head_dim=D, frame_seqlen=FSEQ,
        max_sink_frames=max_sink, max_middle_frames=max_middle,
        max_recent_frames=max_recent, compositions=[[comp]],
        device="cuda:0", kv_dtype="bfloat16",
    )
    cache = caches[0]
    cache.ctx.mgr.reset()
    cache.ctx.committed_vc = None
    cache.ctx.committed_recent_k = None
    cache.ctx.committed_recent_v = None

    torch.manual_seed(101)
    # Pre-fill: 2 clean blocks. After block 1 (frames 0..2): sink[0]=frame0,
    # recent[0..1]=frames1,2 (since sink_cap=1). After block 2 (frames 3..5):
    # recent[2..3]=frames4,5? Actually sink is full so all of block 2 → recent.
    # → recent = [frame1, frame2, frame3, frame4, frame5] but max=4 so shift:
    # recent[0..3] = [frame2, frame3, frame4, frame5].
    block_clean_k = [torch.randn(1, L_in, H, D, dtype=torch.bfloat16, device=device) for _ in range(5)]
    block_clean_v = [torch.randn(1, L_in, H, D, dtype=torch.bfloat16, device=device) for _ in range(5)]
    for b in range(2):
        cache.update(block_clean_k[b], block_clean_v[b],
                     current_start=b * L_in, cache_update_mode="clean")

    # Snapshot the recent pool state after block 2's clean commit.
    rec_k_after_b2_clean = cache.ctx.mgr.recent_k_pool()[0].clone()

    # Block 3: drive multiple noisy iterations (denoise sweep) + final clean.
    # Each noisy K is random + distinguishable; the bug would let noisy K
    # leak into recent_pool slot 0 after the FIFO shift.
    noisy_iters = [
        (torch.randn(1, L_in, H, D, dtype=torch.bfloat16, device=device),
         torch.randn(1, L_in, H, D, dtype=torch.bfloat16, device=device))
        for _ in range(3)
    ]
    for k_n, v_n in noisy_iters:
        cache.update(k_n, v_n, current_start=2 * L_in, cache_update_mode="noisy")
    # Clean pass with canonical block 3 K.
    cache.update(block_clean_k[2], block_clean_v[2],
                 current_start=2 * L_in, cache_update_mode="clean")

    rec_k_after_b3 = cache.ctx.mgr.recent_k_pool()[0].clone()  # [H, max_recent, FSEQ, D]

    # Expected: recent_pool slot 0 after block 3 = recent_pool slot 3 BEFORE
    # block 3's writes (the value that gets shifted left during FIFO).
    # rec_k_after_b2_clean[0, 3] is the canonical "last frame of block 2"
    # which the FIFO shift moves to slot 0.
    expected_slot0 = rec_k_after_b2_clean[0, 3]  # [FSEQ, D]
    got_slot0 = rec_k_after_b3[0, 0]            # [FSEQ, D]

    assert torch.allclose(got_slot0.float(), expected_slot0.float(), atol=2e-2), (
        "Recent_pool slot 0 was contaminated by a noisy K — the noisy/clean "
        "FIFO shift bug is back. committed_recent_pool snapshot is failing."
    )

    # Sanity: slot 0 should NOT match any of the noisy K values.
    for i, (k_n, _) in enumerate(noisy_iters):
        noisy_last_frame = k_n.view(1, F, FSEQ, H, D)[0, F - 1, :, 0, :]  # last noisy frame
        assert not torch.allclose(got_slot0.float(), noisy_last_frame.float(), atol=2e-2), (
            f"slot 0 matches noisy iter {i}'s last frame — FIFO contamination active"
        )

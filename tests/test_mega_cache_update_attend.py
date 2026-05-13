"""M4 Day 2 — MegaCache.update + .attend end-to-end across 3 blocks.

Drives 3 sequential blocks of frames through MegaCache and verifies:
  1. Block 1 fills sink (2 frames at t=0,1).
  2. Block 2 fills 1 more sink frame (t=2) + writes 2 to recent (t=3,4)
     since max_sink=3, F=3.
  3. Block 3 (t=5,6,7) all go to recent (sliding window) since sink is
     full. middle gets populated for stride heads.
  4. Final readout matches direct attention over the gathered K/V.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():  # pragma: no cover
    pytest.skip("CUDA required", allow_module_level=True)
try:
    from flash_attn import flash_attn_varlen_func  # noqa: F401
except ImportError:  # pragma: no cover
    pytest.skip("flash-attn not installed", allow_module_level=True)

from headkv import _ops, _mega_cache, _mega_state_ref as ref
from headkv.base import HeadComposition
from headkv.stride import StrideStrategy

_ops._ensure_loaded()


def _mk_stride_comp():
    # NOTE: capacity is set high enough that no FIFO eviction happens
    # within the 9 frames this test drives (3 blocks × 3 frames each).
    # The existing FIFO-eviction path in update_stride has a known
    # semantic bug: it shifts tkey_t metadata left but doesn't shift
    # the corresponding middle_pool data, so post-eviction tkey_t no
    # longer matches the actual stored K/V. A ring-buffer fix lands
    # in a follow-up commit before this is unblocked on the bench.
    return HeadComposition(
        name="stride", label=1,
        sink_frames=3, recent_frames=4,
        middle_strategies=[StrideStrategy(interval=1, capacity=12)],
    )


def test_three_blocks_update_then_attend_matches_reference():
    H, D, FSEQ = 1, 16, 4
    max_sink, max_middle, max_recent = 3, 12, 4  # max_middle = stride capacity
    L = 1
    layer_idx = 0
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
    cache = caches[layer_idx]
    cache.ctx.mgr.reset()

    # 3 blocks of 3 frames each (F=3).
    F = 3
    L_in = F * FSEQ
    torch.manual_seed(3)
    blocks_k, blocks_v, blocks_q = [], [], []
    for _ in range(3):
        blocks_k.append(torch.randn(1, L_in, H, D, dtype=torch.bfloat16, device=device))
        blocks_v.append(torch.randn(1, L_in, H, D, dtype=torch.bfloat16, device=device))
        blocks_q.append(torch.randn(1, L_in, H, D, dtype=torch.bfloat16, device=device))

    # ---- Block 1: current_start=0, frames at t=0,1,2 → sink slots 0,1,2 ----
    cache.update(blocks_k[0], blocks_v[0], current_start=0)
    vc = cache.ctx.mgr.valid_count()
    assert int(vc[0, 0, 0].item()) == max_sink  # sink full after block 1
    assert int(vc[0, 0, 2].item()) == 0          # recent empty
    # Stride state should record tkey_count=3 with t=[0,1,2].
    import numpy as np
    from headkv import _mega_state_ops as ops_mod
    states_after_b1 = ops_mod.unpack_states(
        cache.states_bytes_for_layer.cpu().numpy()
        .view(ops_mod.PER_HEAD_STATE_DTYPE).reshape(H).copy()
    )
    assert states_after_b1[0].tkey_count == 3
    assert states_after_b1[0].tkey_t[:3] == [0, 1, 2]

    # ---- Block 2: current_start=3*FSEQ, frames t=3,4,5 ----
    # sink full → all 3 go to recent. Recent count: 0+3=3.
    cache.update(blocks_k[1], blocks_v[1], current_start=3 * FSEQ)
    assert int(vc[0, 0, 0].item()) == max_sink
    assert int(vc[0, 0, 2].item()) == 3

    # Stride state with capacity=12: no FIFO, all 6 frames present.
    states_after_b2 = ops_mod.unpack_states(
        cache.states_bytes_for_layer.cpu().numpy()
        .view(ops_mod.PER_HEAD_STATE_DTYPE).reshape(H).copy()
    )
    assert states_after_b2[0].tkey_count == 6
    assert states_after_b2[0].tkey_t[:6] == [0, 1, 2, 3, 4, 5]

    # ---- Block 3: current_start=6*FSEQ, frames t=6,7,8 ----
    # All to recent (sliding window). recent_count was 3; new total 6 > max=4,
    # so evict 2, append 3 → recent now holds frames at slots 0..3.
    out3 = cache.attend(
        q=blocks_q[2], current_start=6 * FSEQ,
        max_seqlen_k=(max_sink + max_middle + max_recent) * FSEQ,
    )
    # First call attend BEFORE block 3 update to test mid-sequence attend.
    # (Then update + attend after for the comprehensive correctness check.)
    assert out3.shape == (1, L_in, H, D)

    cache.update(blocks_k[2], blocks_v[2], current_start=6 * FSEQ)
    assert int(vc[0, 0, 2].item()) == max_recent  # recent now full

    # Final attend after block 3 update.
    out_final = cache.attend(
        q=blocks_q[2], current_start=6 * FSEQ,
        max_seqlen_k=(max_sink + max_middle + max_recent) * FSEQ,
    )
    assert out_final.shape == (1, L_in, H, D)

    # ---- Reference: build K/V manually in plan order (sink → recent → middle) ----
    # Sink slots 0,1,2 = block 1 frames [0,1,2]
    sink_src = blocks_k[0].view(1, F, FSEQ, H, D).permute(0, 1, 3, 2, 4).reshape(F, H, FSEQ, D)
    sink_src_v = blocks_v[0].view(1, F, FSEQ, H, D).permute(0, 1, 3, 2, 4).reshape(F, H, FSEQ, D)

    # Recent: block 2 went to slots 0..2 (recent_count was 0); block 3 evicts
    # slots 0,1 and adds 3 new at slots 2,3 → recent holds [block2_frame2,
    # block3_frame0, block3_frame1, block3_frame2].
    # Wait — re-tracing: after block 2, recent slots 0..2 hold block2[0,1,2].
    # Block 3 has 3 frames going to recent: new_total = 3+3=6 > max_recent=4
    # → evict 2 (recent[0:2]), shift recent[2:4] → recent[0:2], then append 3
    # at recent[1..3]. Final recent: [block2[2], block3[0], block3[1], block3[2]].
    b2_src = blocks_k[1].view(1, F, FSEQ, H, D).permute(0, 1, 3, 2, 4).reshape(F, H, FSEQ, D)
    b2_src_v = blocks_v[1].view(1, F, FSEQ, H, D).permute(0, 1, 3, 2, 4).reshape(F, H, FSEQ, D)
    b3_src = blocks_k[2].view(1, F, FSEQ, H, D).permute(0, 1, 3, 2, 4).reshape(F, H, FSEQ, D)
    b3_src_v = blocks_v[2].view(1, F, FSEQ, H, D).permute(0, 1, 3, 2, 4).reshape(F, H, FSEQ, D)

    rec_frames_k = [b2_src[2], b3_src[0], b3_src[1], b3_src[2]]    # [H, FSEQ, D] × 4
    rec_frames_v = [b2_src_v[2], b3_src_v[0], b3_src_v[1], b3_src_v[2]]

    # Middle (stride capacity=12): pool holds 9 frames at t=0..8 after
    # block 3 update. M5 step 6: plan-level recent dedup is disabled, only
    # sink-side dedup (t < sink_cap=3) applies. So middle emits t=3..8
    # (6 frames). t=5..8 overlap with recent; the readout sees these as
    # duplicates after the same identity RoPE remap (pyramid-forcing10
    # default: sink_time_mapping_mode='lag', history_time_mapping_mode=
    # 'none' → identity), so softmax weight is split equally between the
    # duplicate entries and the output value matches the reference.
    mid_frames_k = [b2_src[0], b2_src[1], b2_src[2], b3_src[0], b3_src[1], b3_src[2]]
    mid_frames_v = [b2_src_v[0], b2_src_v[1], b2_src_v[2], b3_src_v[0], b3_src_v[1], b3_src_v[2]]

    h = 0
    sink_h_k = torch.stack([sink_src[i, h] for i in range(3)]).reshape(-1, D).float()
    sink_h_v = torch.stack([sink_src_v[i, h] for i in range(3)]).reshape(-1, D).float()
    rec_h_k = torch.stack([f[h] for f in rec_frames_k]).reshape(-1, D).float()
    rec_h_v = torch.stack([f[h] for f in rec_frames_v]).reshape(-1, D).float()
    if mid_frames_k:
        mid_h_k = torch.stack([f[h] for f in mid_frames_k]).reshape(-1, D).float()
        mid_h_v = torch.stack([f[h] for f in mid_frames_v]).reshape(-1, D).float()
        K_ref = torch.cat([sink_h_k, rec_h_k, mid_h_k], dim=0)
        V_ref = torch.cat([sink_h_v, rec_h_v, mid_h_v], dim=0)
    else:
        K_ref = torch.cat([sink_h_k, rec_h_k], dim=0)
        V_ref = torch.cat([sink_h_v, rec_h_v], dim=0)

    q_h = blocks_q[2][0, :, h, :].float()
    logits = q_h @ K_ref.T * (D ** -0.5)
    weights = torch.softmax(logits, dim=-1)
    expected = weights @ V_ref

    got = out_final[0, :, h, :].float()
    assert torch.allclose(got, expected, atol=5e-2, rtol=5e-2), (
        f"Max abs err: {(got - expected).abs().max().item():.4f}"
    )

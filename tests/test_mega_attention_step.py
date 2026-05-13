"""M3 Day 2 — mega_attention_step end-to-end (middle insertion + readout).

Drives 2 sequential blocks through a stride strategy head and verifies:
  1. PerHeadState advances correctly (tkey_count grows; tkey_t records
     the stride-aligned t values).
  2. mid_k_pool / mid_v_pool get populated with the new K data at the
     expected slots.
  3. Final readout output matches a direct attention computation over the
     full populated cache.

Sink + recent are pre-set externally (Day 2 doesn't manage them); this
unit isolates the strategy-driven middle path.
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

from pyramidkv import _ops, _mega_state_ops as ops_mod, _mega_state_ref as ref
from pyramidkv import _mega_attention


def _pack(states, device):
    import numpy as np
    arr = ops_mod.pack_states(states)
    return torch.from_numpy(np.frombuffer(arr.tobytes(), dtype=np.uint8)).to(device)


def _unpack(states_bytes, n_heads):
    import numpy as np
    arr = (
        states_bytes.cpu().numpy()
        .view(ops_mod.PER_HEAD_STATE_DTYPE)
        .reshape(n_heads).copy()
    )
    return ops_mod.unpack_states(arr)


def test_two_blocks_stride_insertion_then_readout():
    _ops._ensure_loaded()

    H, D, FSEQ = 1, 16, 4
    max_sink, max_middle, max_recent = 2, 4, 2
    L = 1
    layer_idx = 0
    interval = 1  # every frame is stride-eligible
    device = torch.device("cuda:0")

    Cls = torch.classes.adahead.PyramidKVCacheManager
    mgr = Cls(L, H, D, FSEQ, max_sink, max_middle, max_recent,
              "cuda:0", "bfloat16", 4)
    mgr.reset()

    # ---- Pre-fill sink + recent so readout has something for those kinds ----
    torch.manual_seed(7)
    sink_k_all = torch.randn(H, max_sink, FSEQ, D, dtype=torch.bfloat16, device=device)
    sink_v_all = torch.randn(H, max_sink, FSEQ, D, dtype=torch.bfloat16, device=device)
    rec_k_all  = torch.randn(H, max_recent, FSEQ, D, dtype=torch.bfloat16, device=device)
    rec_v_all  = torch.randn(H, max_recent, FSEQ, D, dtype=torch.bfloat16, device=device)
    mgr.sink_k_pool()[0].copy_(sink_k_all)
    mgr.sink_v_pool()[0].copy_(sink_v_all)
    mgr.recent_k_pool()[0].copy_(rec_k_all)
    mgr.recent_v_pool()[0].copy_(rec_v_all)
    vc = mgr.valid_count()
    vc[0, 0, 0] = max_sink     # sink_count
    vc[0, 0, 2] = max_recent   # recent_count

    # ---- Initial state: stride strategy, no middle data yet ----
    states = [ref.make_stride(interval=interval, capacity=max_middle)]
    states_bytes = _pack(states, device)

    # ---- Block 1: 2 new stride-eligible frames at t = 0, 1 ----
    F = 2
    L_q = F * FSEQ
    q1 = torch.randn(1, L_q, H, D, dtype=torch.bfloat16, device=device)
    k1 = torch.randn(1, L_q, H, D, dtype=torch.bfloat16, device=device)
    v1 = torch.randn(1, L_q, H, D, dtype=torch.bfloat16, device=device)
    t1 = torch.tensor([0, 1], dtype=torch.int64, device=device)

    out1 = _mega_attention.mega_attention_step(
        mgr=mgr, per_head_states_bytes=states_bytes,
        q=q1, k_new=k1, v_new=v1, new_t_vals=t1,
        layer_idx=layer_idx, current_t=0, pass_kind=1,
        max_seqlen_k=(max_sink + max_middle + max_recent) * FSEQ,
    )
    assert out1.shape == (1, L_q, H, D)

    # State after block 1: tkey_count == 2, tkey_t == [0, 1, ...]
    new_states = _unpack(states_bytes, n_heads=H)
    assert new_states[0].tkey_count == 2, f"got tkey_count={new_states[0].tkey_count}"
    assert new_states[0].tkey_t[0] == 0
    assert new_states[0].tkey_t[1] == 1
    # Middle pool slots 0, 1 should now hold k1's two frames per head h=0.
    # new_k reshape from [B=1, L_q=8, H=1, D=16] → [F=2, H=1, FSEQ=4, D=16].
    k1_per_frame = (
        k1.view(1, F, FSEQ, H, D)
          .permute(0, 1, 3, 2, 4)
          .reshape(F, H, FSEQ, D)
    )
    mid_k = mgr.middle_k_pool()[0, 0, :2]  # [2, FSEQ, D]
    assert torch.equal(mid_k, k1_per_frame[:, 0])

    # ---- Block 2: 2 more stride-eligible frames at t = 2, 3 ----
    q2 = torch.randn(1, L_q, H, D, dtype=torch.bfloat16, device=device)
    k2 = torch.randn(1, L_q, H, D, dtype=torch.bfloat16, device=device)
    v2 = torch.randn(1, L_q, H, D, dtype=torch.bfloat16, device=device)
    t2 = torch.tensor([2, 3], dtype=torch.int64, device=device)

    out2 = _mega_attention.mega_attention_step(
        mgr=mgr, per_head_states_bytes=states_bytes,
        q=q2, k_new=k2, v_new=v2, new_t_vals=t2,
        layer_idx=layer_idx, current_t=2, pass_kind=1,
        max_seqlen_k=(max_sink + max_middle + max_recent) * FSEQ,
    )
    assert out2.shape == (1, L_q, H, D)

    new_states = _unpack(states_bytes, n_heads=H)
    assert new_states[0].tkey_count == 4
    assert new_states[0].tkey_t[:4] == [0, 1, 2, 3]

    # ---- Verify readout: the cache now has sink (2) + middle (4) +
    # recent (2) = 8 frames per head. Compute reference attention. ----
    k2_per_frame = (
        k2.view(1, F, FSEQ, H, D).permute(0, 1, 3, 2, 4).reshape(F, H, FSEQ, D)
    )
    v1_per_frame = (
        v1.view(1, F, FSEQ, H, D).permute(0, 1, 3, 2, 4).reshape(F, H, FSEQ, D)
    )
    v2_per_frame = (
        v2.view(1, F, FSEQ, H, D).permute(0, 1, 3, 2, 4).reshape(F, H, FSEQ, D)
    )
    # mega_plan emits sink → recent → middle. Build the reference K/V in
    # that same order, then run direct attention.
    sink_k_h = sink_k_all[0].reshape(-1, D).float()      # [max_sink*FSEQ, D]
    sink_v_h = sink_v_all[0].reshape(-1, D).float()
    rec_k_h  = rec_k_all[0].reshape(-1, D).float()
    rec_v_h  = rec_v_all[0].reshape(-1, D).float()
    # Middle: slots 0,1,2,3 in pool ← blocks 1 frames + block 2 frames.
    mid_k_full = torch.cat([k1_per_frame[:, 0], k2_per_frame[:, 0]], dim=0)  # [4, FSEQ, D]
    mid_v_full = torch.cat([v1_per_frame[:, 0], v2_per_frame[:, 0]], dim=0)
    mid_k_h = mid_k_full.reshape(-1, D).float()
    mid_v_h = mid_v_full.reshape(-1, D).float()

    K_ref = torch.cat([sink_k_h, rec_k_h, mid_k_h], dim=0)
    V_ref = torch.cat([sink_v_h, rec_v_h, mid_v_h], dim=0)

    q2_h = q2[0, :, 0, :].float()
    logits = q2_h @ K_ref.T * (D ** -0.5)
    weights = torch.softmax(logits, dim=-1)
    expected = weights @ V_ref

    got = out2[0, :, 0, :].float()
    assert torch.allclose(got, expected, atol=5e-2, rtol=5e-2), (
        f"Max abs err: {(got - expected).abs().max().item():.4f}"
    )

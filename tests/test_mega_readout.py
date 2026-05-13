"""M3 Day 1 — mega_readout end-to-end test (plan + pack + flash_attn).

Pre-populates the manager pools with random K/V data, sets the per-head
state to a stride strategy with known tkey_t values, then verifies that
mega_readout's flash_attn output matches a direct attention computation on
the same K/V slice. This pins down:

  1. mega_plan correctly enumerates the live slots → cu_seqlens_k matches
     the expected token count.
  2. headkv_pack correctly gathers pool slices into the flat workspace.
  3. flash_attn_varlen_func produces the same output as a manual
     scaled-dot-product attention on the gathered K/V.

This is a single-layer, B=1, H=2 unit test. M3 Day 2 will exercise the
insertion path (mega_state_update + headkv_update).
"""
from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():  # pragma: no cover
    pytest.skip("CUDA required for mega_readout test", allow_module_level=True)

try:  # pragma: no cover
    from flash_attn import flash_attn_varlen_func  # noqa: F401
except ImportError:  # pragma: no cover
    pytest.skip("flash-attn not installed", allow_module_level=True)

from headkv import _ops, _mega_state_ops as ops_mod, _mega_state_ref as ref
from headkv import _mega_attention


def _make_manager(H, D, FSEQ, max_sink, max_middle, max_recent, L=1,
                  max_attend_chunks=4):
    _ops._ensure_loaded()
    Cls = torch.classes.adahead.HeadKVCacheManager
    return Cls(L, H, D, FSEQ, max_sink, max_middle, max_recent,
               "cuda:0", "bfloat16", max_attend_chunks)


def _pack_states_to_device(states, device):
    import numpy as np
    arr = ops_mod.pack_states(states)
    return torch.from_numpy(np.frombuffer(arr.tobytes(), dtype=np.uint8)).to(device)


def _ref_attention(q, k, v, softmax_scale=None):
    """Plain scaled-dot-product attention. Inputs flat: q [Lq, D], k/v [Lk, D]."""
    Lq, D = q.shape
    if softmax_scale is None:
        softmax_scale = D ** -0.5
    # Use fp32 to stay above bf16 round-off.
    q = q.float()
    k = k.float()
    v = v.float()
    logits = (q @ k.T) * softmax_scale          # [Lq, Lk]
    weights = torch.softmax(logits, dim=-1)
    out = weights @ v                           # [Lq, D]
    return out


def test_mega_readout_matches_direct_attention():
    H, D, FSEQ = 2, 16, 4
    max_sink, max_middle, max_recent = 2, 4, 2
    layer_idx, current_t = 0, 100

    device = torch.device("cuda:0")
    mgr = _make_manager(H, D, FSEQ, max_sink, max_middle, max_recent)
    mgr.reset()

    # ---- Plant deterministic K/V into the pools ----
    torch.manual_seed(42)
    # sink_pool: [L, H, max_sink, F, D] bf16
    sink_k = torch.randn(H, max_sink, FSEQ, D, dtype=torch.bfloat16, device=device)
    sink_v = torch.randn(H, max_sink, FSEQ, D, dtype=torch.bfloat16, device=device)
    mid_k  = torch.randn(H, max_middle, FSEQ, D, dtype=torch.bfloat16, device=device)
    mid_v  = torch.randn(H, max_middle, FSEQ, D, dtype=torch.bfloat16, device=device)
    rec_k  = torch.randn(H, max_recent, FSEQ, D, dtype=torch.bfloat16, device=device)
    rec_v  = torch.randn(H, max_recent, FSEQ, D, dtype=torch.bfloat16, device=device)

    mgr.sink_k_pool()[0].copy_(sink_k)
    mgr.sink_v_pool()[0].copy_(sink_v)
    mgr.middle_k_pool()[0].copy_(mid_k)
    mgr.middle_v_pool()[0].copy_(mid_v)
    mgr.recent_k_pool()[0].copy_(rec_k)
    mgr.recent_v_pool()[0].copy_(rec_v)

    # ---- valid_count: 2 sink, 3 middle (stride filled 3/4), 2 recent ----
    vc = mgr.valid_count()
    for h in range(H):
        vc[0, h, 0] = max_sink
        vc[0, h, 1] = 3
        vc[0, h, 2] = max_recent

    # ---- Per-head state: stride strategy with tkey_count=3 ----
    states = [ref.make_stride(interval=1, capacity=max_middle) for _ in range(H)]
    # Mark tkey_count=3 so mega_plan walks tkey_slot[0..2].
    for s in states:
        s.tkey_count = 3
        # tkey_t values arbitrary (not used by pack, only by plan's t_remap
        # — which we don't read in mega_readout for hist=none).
        s.tkey_t = [10, 11, 12] + [0] * 15
        s.tkey_slot = [0, 1, 2] + [-1] * 15
    states_bytes = _pack_states_to_device(states, device)

    # ---- Build Q ----
    L_q = FSEQ  # 1 new frame's worth of queries
    q = torch.randn(1, L_q, H, D, dtype=torch.bfloat16, device=device)

    # ---- mega_readout ----
    out = _mega_attention.mega_readout(
        mgr=mgr,
        per_head_states_bytes=states_bytes,
        q=q,
        layer_idx=layer_idx,
        current_t=current_t,
        max_seqlen_k=(max_sink + max_middle + max_recent) * FSEQ,
    )
    assert out.shape == (1, L_q, H, D)
    assert out.dtype == torch.bfloat16

    # ---- Reference: gather K/V manually in the same order mega_plan emits ----
    # mega_plan emits per-head: sink frames 0..n_sink-1, recent 0..n_recent-1,
    # middle slots in tkey_slot[0..tkey_count-1].
    for h in range(H):
        sink_h_k = sink_k[h, :max_sink].reshape(-1, D).float()       # [n_sink*F, D]
        sink_h_v = sink_v[h, :max_sink].reshape(-1, D).float()
        rec_h_k  = rec_k[h, :max_recent].reshape(-1, D).float()
        rec_h_v  = rec_v[h, :max_recent].reshape(-1, D).float()
        # Middle: slots [0, 1, 2]
        mid_h_k = mid_k[h, :3].reshape(-1, D).float()
        mid_h_v = mid_v[h, :3].reshape(-1, D).float()

        k_h = torch.cat([sink_h_k, rec_h_k, mid_h_k], dim=0)
        v_h = torch.cat([sink_h_v, rec_h_v, mid_h_v], dim=0)

        q_h = q[0, :, h, :].float()
        expected_h = _ref_attention(q_h, k_h, v_h)

        got_h = out[0, :, h, :].float()
        # bf16 round-trip + flash_attn vs fp32 reference → tolerance ~5e-2.
        assert torch.allclose(got_h, expected_h, atol=5e-2, rtol=5e-2), (
            f"head {h} mismatch: max abs err "
            f"{(got_h - expected_h).abs().max().item():.4f}"
        )

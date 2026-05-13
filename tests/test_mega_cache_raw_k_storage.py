"""Option C parity test: MegaCache stores raw K, readout applies a single
fused 3D RoPE, and the resulting flash_attn output matches a manual path
that does the same thing in Python.

End-to-end: MegaCache.update(raw_k) → MegaCache.attend(roped_q) vs.
apply_rope_to_flat_k(raw_k, pos) → flash_attn_varlen_func.

CUDA + flash-attn gated. Skips on CPU-only machines.
"""
from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():  # pragma: no cover
    pytest.skip("CUDA required for raw-K storage parity test",
                allow_module_level=True)
try:  # pragma: no cover
    from flash_attn import flash_attn_varlen_func
except ImportError:  # pragma: no cover
    pytest.skip("flash-attn not installed", allow_module_level=True)

from headkv._mega_cache import build_mega_caches
from headkv.base import HeadComposition
from headkv.recent import RecentStrategy
from headkv.rope import apply_rope_to_flat_k


def _build_freqs(head_dim: int, max_t: int = 64) -> torch.Tensor:
    """Mirror wan.modules.causal_model freqs layout: complex64 [max_t, D/2]."""
    c_half = head_dim // 2
    ft_cols = c_half - 2 * (c_half // 3)
    fy_cols = c_half // 3
    fx_cols = c_half // 3
    parts = []
    for cols, base in [(ft_cols, 10000.0), (fy_cols, 10000.0), (fx_cols, 10000.0)]:
        if cols <= 0:
            continue
        inv = 1.0 / (base ** (torch.arange(0, cols * 2, 2, dtype=torch.float32) / (cols * 2)))
        ts = torch.arange(max_t, dtype=torch.float32)
        angles = ts[:, None] * inv[None, :]
        parts.append(torch.polar(torch.ones_like(angles), angles).to(torch.complex64))
    return torch.cat(parts, dim=1).cuda()


def test_mega_cache_raw_k_matches_manual_rope_then_attn():
    L, H, D = 1, 2, 64
    FSEQ = 8
    spatial_w, spatial_h = 4, 2  # 4*2 = 8 = FSEQ
    max_sink, max_middle, max_recent = 1, 1, 4
    F_block = 1

    # All heads use recent-only composition for simplicity. RecentStrategy
    # means storage path is just sink + recent (no merge/state machine
    # gymnastics), so the parity check focuses on the raw-K storage + readout
    # RoPE path that's the heart of Option C.
    compositions: list[list[HeadComposition]] = []
    for _ in range(L):
        row = []
        for _h in range(H):
            row.append(HeadComposition(
                name="test", label=1,
                sink_frames=1, recent_frames=max_recent,
                middle_strategies=[RecentStrategy()],
                policy_type="osc", capacity=32,
            ))
        compositions.append(row)

    freqs = _build_freqs(D, max_t=64)
    caches = build_mega_caches(
        num_layers=L, num_heads=H, head_dim=D,
        frame_seqlen=FSEQ,
        max_sink_frames=max_sink, max_middle_frames=max_middle,
        max_recent_frames=max_recent,
        compositions=compositions,
        device="cuda:0", kv_dtype="bfloat16",
        spatial_width=spatial_w, spatial_height=spatial_h,
        freqs=freqs,
    )
    cache = caches[0]

    # Emit one frame's worth of raw K/V (no upstream RoPE — that's the point).
    torch.manual_seed(0)
    raw_k = torch.randn(1, F_block * FSEQ, H, D,
                        dtype=torch.bfloat16, device="cuda:0")
    v     = torch.randn(1, F_block * FSEQ, H, D,
                        dtype=torch.bfloat16, device="cuda:0")
    q_rop = torch.randn(1, FSEQ, H, D, dtype=torch.bfloat16, device="cuda:0")

    # First call: fills the sink frame (sink_frames=1).
    cache.update(raw_k, v, current_start=0, cache_update_mode="clean")
    # Second call: fills recent slot 0 (sink already full at sink_frames=1).
    raw_k1 = torch.randn(1, FSEQ, H, D, dtype=torch.bfloat16, device="cuda:0")
    v1     = torch.randn(1, FSEQ, H, D, dtype=torch.bfloat16, device="cuda:0")
    cache.update(raw_k1, v1, current_start=FSEQ, cache_update_mode="clean")

    sync_t = 2  # current_start = 2*FSEQ ⇒ current_t_frame = 2; lag mode, lag=0
    out = cache.attend(q_rop, current_start=sync_t * FSEQ, freqs=freqs, causal=False)
    assert out.shape == (1, FSEQ, H, D)

    # ---- Reference: replicate the readout in pure Python ----
    # mega_plan emits, per head: sink anchor with tremap = sync_t (because
    # sink_grid_decoupling is always-on in pyramid_forcing10's plan path),
    # followed by recent anchor with tremap = original t. The fused readout
    # RoPE uses tremap for the temporal axis and the stored (y, x) for
    # spatial — so the reference must rotate sink K at t=sync_t (not at the
    # frame's stored t=0).
    idx = torch.arange(FSEQ, device="cuda:0", dtype=torch.int64)
    ys = (idx // spatial_w)
    xs = (idx %  spatial_w)
    ref_outs = []
    for h in range(H):
        k_sink = raw_k[0, :, h, :].contiguous()    # [FSEQ, D]
        k_rec  = raw_k1[0, :, h, :].contiguous()   # [FSEQ, D]
        v_sink = v[0,  :, h, :].contiguous()
        v_rec  = v1[0, :, h, :].contiguous()

        sink_t_eff = torch.full((FSEQ,), sync_t, dtype=torch.int64, device="cuda:0")
        rec_t_eff  = torch.full((FSEQ,), 1,      dtype=torch.int64, device="cuda:0")
        pos_sink = torch.stack([sink_t_eff, ys, xs], dim=1)
        pos_rec  = torch.stack([rec_t_eff,  ys, xs], dim=1)
        k_sink_rot = apply_rope_to_flat_k(k_sink, pos_sink, freqs=freqs)
        k_rec_rot  = apply_rope_to_flat_k(k_rec,  pos_rec,  freqs=freqs)

        k_h = torch.cat([k_sink_rot, k_rec_rot], dim=0).contiguous()
        v_h = torch.cat([v_sink, v_rec], dim=0).contiguous()

        q_h = q_rop[0, :, h, :].contiguous()           # [FSEQ, D]
        Lq = q_h.shape[0]
        Lk = k_h.shape[0]
        cu_q = torch.tensor([0, Lq], dtype=torch.int32, device="cuda:0")
        cu_k = torch.tensor([0, Lk], dtype=torch.int32, device="cuda:0")
        ref = flash_attn_varlen_func(
            q_h.view(Lq, 1, D),
            k_h.view(Lk, 1, D),
            v_h.view(Lk, 1, D),
            cu_q, cu_k,
            max_seqlen_q=Lq, max_seqlen_k=Lk,
            dropout_p=0.0,
            softmax_scale=D ** -0.5,
            causal=False,
        )  # [Lq, 1, D]
        ref_outs.append(ref.view(Lq, D))

    expected = torch.stack(ref_outs, dim=1).unsqueeze(0)  # [1, Lq, H, D]
    # bf16 numerics + flash-attn quirks → loose tolerance
    torch.testing.assert_close(
        out.float(), expected.float(), atol=2e-2, rtol=2e-2,
    )

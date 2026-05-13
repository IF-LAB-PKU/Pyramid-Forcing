"""Plan B — buffer-overflow guard test.

mega_plan_multi sizes its pack workspace at construction by max_attend_chunks.
If a caller asks for more chunks than the buffer holds, the plan op must
TORCH_CHECK and refuse — silently overrunning would corrupt k_flat_out /
v_flat_out / pos_flat_out across forward passes and is far worse than a
loud failure.

Strategy: build a manager with max_attend_chunks=1 (smallest legal cap),
fill enough state that two chunks each emit at least one segment, then
call mega_plan_multi with current_t_list of length 2. The C++ TORCH_CHECK
after the emit loop must trip.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():  # pragma: no cover
    pytest.skip("CUDA required", allow_module_level=True)

import numpy as np

from headkv import _mega_state_ops as ops_mod
from headkv import _mega_state_ref as ref
from headkv import _ops


@pytest.mark.gpu
def test_mega_plan_multi_rejects_chunk_count_exceeding_capacity():
    _ops._ensure_loaded()
    Cls = torch.classes.adahead.HeadKVCacheManager
    H, D, FSEQ = 2, 16, 4
    max_sink, max_middle, max_recent = 2, 4, 2
    # max_attend_chunks=1: pack workspace fits exactly one chunk's worth.
    mgr = Cls(1, H, D, FSEQ, max_sink, max_middle, max_recent,
              "cuda:0", "bfloat16", 1)

    # Populate sink valid_count so each chunk emits ≥ 1 segment per head.
    vc = mgr.valid_count()
    vc[0, :, 0] = max_sink  # kind=0 → sink

    states = [ref.PerHeadState() for _ in range(H)]
    arr = ops_mod.pack_states(states)
    states_bytes = torch.from_numpy(
        np.frombuffer(arr.tobytes(), dtype=np.uint8)
    ).to("cuda:0")

    # Two chunks → total tokens = 2 × H × max_sink × FSEQ = 2*2*2*4 = 32,
    # but max_pack_tokens = 1 × H × (max_total + max_merge_blocks) × FSEQ
    # = 1 × 2 × (8 + 6) × 4 = 112 — wait, this doesn't overflow yet because
    # the buffer is sized for "1 chunk worst case" which is bigger than
    # "2 chunks of sink-only". Need to push past the buffer to trigger.
    #
    # Worst case per chunk: H × (max_total + max_merge_blocks) × FSEQ tokens.
    # Buffer cap: max_attend_chunks × <same>. So to overflow we need to ask
    # for > max_attend_chunks chunks AND fill enough so the total tokens
    # exceed the cap. Setting middle full + sink full per chunk:
    vc[0, :, 1] = max_middle  # middle full
    vc[0, :, 2] = max_recent  # recent full
    # Per chunk = H * (max_sink + max_middle + max_recent) * FSEQ
    #           = 2 * 8 * 4 = 64 tokens.
    # 3 chunks = 192 tokens vs cap 112 → overflow.

    current_t_list = torch.tensor([0, 5, 10], dtype=torch.int64)
    with pytest.raises(RuntimeError, match="exceed pack-workspace capacity"):
        ops_mod.mega_plan_multi_cuda(
            mgr, states_bytes,
            layer_idx=0,
            current_t_list=current_t_list,
        )

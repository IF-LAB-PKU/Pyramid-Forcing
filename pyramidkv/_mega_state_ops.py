"""Python ↔ C++ bridge for the mega_state_update kernel.

The kernel mutates an array of ``PerHeadState`` POD structs in-place. To call
it from Python we have to serialize the dataclass mirror in
``pyramidkv/_mega_state_ref.py`` into a raw byte buffer with the **exact same
ABI** the C++ compiler used. We do this with a ``numpy`` structured dtype
mirroring the field layout in ``pyramidkv/csrc/anchor_store.cuh``.

Layout safety
-------------
The structured dtype is built with ``align=True`` so numpy inserts the same
padding the C++ compiler does for natural alignment. We cross-check that the
resulting itemsize matches ``sizeof(PerHeadState)`` reported by
``torch.ops.adahead.mega_state_perhead_size()``; if they ever drift we fail
loudly rather than corrupting state silently.

Scope
-----
Cyclic / stride / lag / recent paths are exercised here; merge updates are
handled by the separate ``mega_merge_accum_cuda`` wrapper below.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import torch

from . import _mega_state_ref as _ref
from . import _ops


# Mirror of pyramidkv/csrc/anchor_store.cuh:PerHeadState. Field order MUST match
# the C++ declaration verbatim — numpy's ``align=True`` inserts the same
# implicit padding the compiler does between mismatched-alignment fields.
PER_HEAD_STATE_DTYPE = np.dtype(
    [
        ("kind", np.int8),
        ("sink_capacity", np.int8),
        ("_pad0", np.int8, (2,)),
        ("period", np.int32),
        ("bucket_cap", np.int32),
        ("interval", np.int32),
        ("capacity", np.int32),
        ("patch_size", np.int32),
        ("block_frames", np.int32),
        ("lag_offsets", np.int32, (_ref.MAX_LAG_OFFSETS,)),
        ("lag_offset_count", np.int32),
        ("cyclic_slot", np.int32, (_ref.MAX_PHASE * _ref.MAX_BUCKET,)),
        ("cyclic_t", np.int32, (_ref.MAX_PHASE * _ref.MAX_BUCKET,)),
        ("cyclic_cursor", np.int8, (_ref.MAX_PHASE,)),
        # implicit 2 bytes pad here (align=True) before the next int32 array
        ("tkey_slot", np.int32, (_ref.MAX_T_KEYED,)),
        ("tkey_t", np.int32, (_ref.MAX_T_KEYED,)),
        ("tkey_count", np.int8),
        ("tkey_head", np.int8),
        ("_pad1", np.int8, (2,)),
        ("merge_completed_slot", np.int32, (_ref.MAX_MERGE_BLOCKS,)),
        ("merge_completed_block_id", np.int32, (_ref.MAX_MERGE_BLOCKS,)),
        ("merge_completed_start_t", np.int32, (_ref.MAX_MERGE_BLOCKS,)),
        ("merge_completed_end_t", np.int32, (_ref.MAX_MERGE_BLOCKS,)),
        ("merge_completed_median_t", np.int32, (_ref.MAX_MERGE_BLOCKS,)),
        ("merge_completed_count", np.int8),
        ("_pad2", np.int8, (3,)),
        ("merge_active_block_id", np.int32, (_ref.MAX_MERGE_ACTIVE,)),
        ("merge_active_slot", np.int32, (_ref.MAX_MERGE_ACTIVE,)),
        (
            "merge_active_seen",
            np.int8,
            (_ref.MAX_MERGE_ACTIVE, _ref.MAX_MERGE_BLOCK_FRAMES),
        ),
        ("merge_active_complete_count", np.int8, (_ref.MAX_MERGE_ACTIVE,)),
        ("_pad3", np.int8, (2,)),
        ("cached_num_groups", np.int32),
    ],
    align=True,
)


_ABI_CHECKED = False


def _check_abi() -> None:
    """Verify numpy dtype size matches sizeof(PerHeadState) on the C++ side.

    Run once per process; mismatch means the struct layout drifted and the
    raw bytes we ship to the kernel would be misaligned.
    """
    global _ABI_CHECKED
    if _ABI_CHECKED:
        return
    cpp_size = int(_ops.ops().mega_state_perhead_size())
    py_size = PER_HEAD_STATE_DTYPE.itemsize
    if cpp_size != py_size:
        raise RuntimeError(
            f"PerHeadState ABI mismatch: numpy dtype itemsize={py_size}, "
            f"sizeof(PerHeadState)={cpp_size}. The C++ struct layout in "
            f"pyramidkv/csrc/anchor_store.cuh and the dtype in "
            f"pyramidkv/_mega_state_ops.py must be kept in sync."
        )
    _ABI_CHECKED = True


def pack_states(states: Sequence[_ref.PerHeadState]) -> np.ndarray:
    """Convert a list of dataclass states into a structured numpy array."""
    arr = np.zeros(len(states), dtype=PER_HEAD_STATE_DTYPE)
    for i, s in enumerate(states):
        rec = arr[i]
        rec["kind"] = s.kind
        rec["sink_capacity"] = s.sink_capacity
        rec["period"] = s.period
        rec["bucket_cap"] = s.bucket_cap
        rec["interval"] = s.interval
        rec["capacity"] = s.capacity
        rec["patch_size"] = s.patch_size
        rec["block_frames"] = s.block_frames
        rec["lag_offsets"][:] = s.lag_offsets
        rec["lag_offset_count"] = s.lag_offset_count
        rec["cyclic_slot"][:] = s.cyclic_slot
        rec["cyclic_t"][:] = s.cyclic_t
        rec["cyclic_cursor"][:] = s.cyclic_cursor
        rec["tkey_slot"][:] = s.tkey_slot
        rec["tkey_t"][:] = s.tkey_t
        rec["tkey_count"] = s.tkey_count
        rec["tkey_head"] = s.tkey_head
        rec["merge_completed_slot"][:] = s.merge_completed_slot
        rec["merge_completed_block_id"][:] = s.merge_completed_block_id
        rec["merge_completed_count"] = s.merge_completed_count
        rec["merge_active_block_id"][:] = s.merge_active_block_id
        for a in range(_ref.MAX_MERGE_ACTIVE):
            rec["merge_active_seen"][a][:] = [
                int(b) for b in s.merge_active_seen[a]
            ]
        rec["merge_active_complete_count"][:] = s.merge_active_complete_count
        rec["cached_num_groups"] = s.cached_num_groups
    return arr


def unpack_states(arr: np.ndarray) -> list[_ref.PerHeadState]:
    """Inverse of pack_states — used by tests to inspect post-kernel state."""
    out: list[_ref.PerHeadState] = []
    for i in range(arr.shape[0]):
        rec = arr[i]
        s = _ref.PerHeadState(
            kind=int(rec["kind"]),
            sink_capacity=int(rec["sink_capacity"]),
            period=int(rec["period"]),
            bucket_cap=int(rec["bucket_cap"]),
            interval=int(rec["interval"]),
            capacity=int(rec["capacity"]),
            patch_size=int(rec["patch_size"]),
            block_frames=int(rec["block_frames"]),
            lag_offsets=[int(x) for x in rec["lag_offsets"]],
            lag_offset_count=int(rec["lag_offset_count"]),
            cyclic_slot=[int(x) for x in rec["cyclic_slot"]],
            cyclic_t=[int(x) for x in rec["cyclic_t"]],
            cyclic_cursor=[int(x) for x in rec["cyclic_cursor"]],
            tkey_slot=[int(x) for x in rec["tkey_slot"]],
            tkey_t=[int(x) for x in rec["tkey_t"]],
            tkey_count=int(rec["tkey_count"]),
            tkey_head=int(rec["tkey_head"]),
            merge_completed_slot=[int(x) for x in rec["merge_completed_slot"]],
            merge_completed_block_id=[int(x) for x in rec["merge_completed_block_id"]],
            merge_completed_count=int(rec["merge_completed_count"]),
            merge_active_block_id=[int(x) for x in rec["merge_active_block_id"]],
            merge_active_seen=[
                [bool(b) for b in rec["merge_active_seen"][a]]
                for a in range(_ref.MAX_MERGE_ACTIVE)
            ],
            merge_active_complete_count=[
                int(x) for x in rec["merge_active_complete_count"]
            ],
            cached_num_groups=int(rec["cached_num_groups"]),
        )
        out.append(s)
    return out


def _to_bytes_tensor(arr: np.ndarray, device: torch.device) -> torch.Tensor:
    """View the structured numpy array as raw uint8 bytes on ``device``."""
    flat = np.ascontiguousarray(arr).view(np.uint8).reshape(-1)
    host = torch.from_numpy(flat).clone()
    return host.to(device=device, non_blocking=False)


# Code lookups for the mode-name strings used by adaptive_cache / rope.py.
_SINK_MODE = {"lag": 0, "window_clamp": 1}
_HIST_MODE = {"none": 0, "relative_clamp": 1, "relative_softcap": 2}


def mega_plan_cuda(
    mgr,                                  # torch.classes.adahead.PyramidKVCacheManager
    states_bytes: torch.Tensor,           # uint8 [H * sizeof(PerHeadState)]
    layer_idx: int,
    current_t: int,
    pass_kind: int = 1,
    sink_time_mapping_mode: str = "lag",
    sink_time_clamp_min: int = 0,
    sink_time_clamp_max: int = 21,
    decoupled_sink_time_lag: int = 0,
    history_time_mapping_mode: str = "none",
    history_relative_t_max: int = 21,
    history_time_soft_factor: float = 1.0,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor,
]:
    """Wrapper for ``torch.ops.adahead.mega_plan``.

    Defaults match pyramid_forcing10 (sink=lag/0, hist=none) → anchor_t_remap
    is identity for that config. Pass non-default mode strings to engage the
    full map_sink_time / map_dynamic_pos_time remap from pyramidkv/rope.py.

    Returns 7 device tensors:
        cu_seqlens_k, src_kind, src_slot_global, seg_lengths,
        dst_token_offsets, anchor_t_raw, anchor_t_remap.
    """
    sink_code = _SINK_MODE.get(sink_time_mapping_mode, 0)
    hist_code = _HIST_MODE.get(history_time_mapping_mode, 0)
    return _ops.ops().mega_plan(
        mgr, states_bytes,
        int(layer_idx), int(current_t), int(pass_kind),
        int(sink_code),
        int(sink_time_clamp_min), int(sink_time_clamp_max),
        int(decoupled_sink_time_lag),
        int(hist_code),
        int(history_relative_t_max),
        float(history_time_soft_factor),
    )


def mega_plan_multi_cuda(
    mgr,                                  # torch.classes.adahead.PyramidKVCacheManager
    states_bytes: torch.Tensor,           # uint8 [H * sizeof(PerHeadState)]
    layer_idx: int,
    current_t_list,                       # int64 list/Tensor [num_chunks]
    pass_kind: int = 1,
    sink_time_mapping_mode: str = "lag",
    sink_time_clamp_min: int = 0,
    sink_time_clamp_max: int = 21,
    decoupled_sink_time_lag: int = 0,
    history_time_mapping_mode: str = "none",
    history_relative_t_max: int = 21,
    history_time_soft_factor: float = 1.0,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor,
]:
    """Multi-chunk variant of mega_plan_cuda.

    ``current_t_list`` is a list/Tensor of per-chunk query frame indices
    (length = num_chunks). Output descriptors are concatenated across
    chunks; dst_token_offsets are chunk-offset globally so that
    pyramidkv_pack can write all chunks' K/V/pos into disjoint regions of
    mgr.k_flat_out in a single launch.

    Returns 7 device tensors with shapes:
        cu_seqlens_k       [num_chunks * H + 1]
        src_kind           [num_chunks * H * max_total_segments]
        src_slot_global    [same]
        seg_lengths        [same]
        dst_token_offsets  [same] — chunk-offset globally
        anchor_t_raw       [same]
        anchor_t_remap     [same] — per-chunk sync_t for dynamic, -1 for sentinel
    """
    if not isinstance(current_t_list, torch.Tensor):
        current_t_list = torch.tensor(list(current_t_list), dtype=torch.int64)
    elif current_t_list.dtype != torch.int64:
        current_t_list = current_t_list.to(torch.int64)
    if current_t_list.dim() == 0:
        current_t_list = current_t_list.view(1)
    # The C++ op pulls current_t_list to CPU internally; pass-through works
    # for either device but a CPU tensor avoids a host sync.
    if current_t_list.device.type != "cpu":
        current_t_list = current_t_list.cpu()

    sink_code = _SINK_MODE.get(sink_time_mapping_mode, 0)
    hist_code = _HIST_MODE.get(history_time_mapping_mode, 0)
    return _ops.ops().mega_plan_multi(
        mgr, states_bytes,
        int(layer_idx), current_t_list, int(pass_kind),
        int(sink_code),
        int(sink_time_clamp_min), int(sink_time_clamp_max),
        int(decoupled_sink_time_lag),
        int(hist_code),
        int(history_relative_t_max),
        float(history_time_soft_factor),
    )


def mega_merge_accum_cuda(
    mgr,                                  # torch.classes.adahead.PyramidKVCacheManager
    layer_idx: int,
    states_bytes: torch.Tensor,           # uint8 buffer of PerHeadState[H]
    new_k: torch.Tensor,                  # [F, H, FSEQ, D] bf16 — raw K (no RoPE applied)
    new_v: torch.Tensor,                  # [F, H, FSEQ, D] bf16
    new_pos: torch.Tensor,                # [F, H, FSEQ, 3] int64
    descriptors: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    """End-to-end merge K/V scatter-add + finalize for one layer.

    Args:
        mgr: PyramidKVCacheManager (provides merge accumulator + pool views).
        layer_idx: which layer slice of manager pools to operate on.
        states_bytes: raw uint8 PerHeadState[H] buffer for this layer.
        new_k, new_v: incoming bf16 K/V, head-major [F, H, FSEQ, D].
            K is stored RAW (no RoPE applied); the kernel just sum-divides
            by tokens-per-group, and readout applies fresh 3D RoPE.
        new_pos: position tensor [F, H, FSEQ, 3] int64 (still used to derive
            spatial group_ids and store the per-anchor (median_t, py, px)
            into merge_pos_pool).
        descriptors: 4-tuple from mega_state_update output —
            (desc_dst_kind, desc_merge_accum_slot,
             desc_merge_finalize_completed_idx, desc_merge_is_new_block).
            Each [H * F] int32 on the same CUDA device.

    Side effects: mutates the manager's merge accumulator + finalized pools.
    """
    desc_kind, desc_slot, desc_finalize, desc_new = descriptors

    _ops.ops().mega_merge_accum(
        states_bytes,
        new_k, new_v, new_pos,
        desc_kind, desc_slot, desc_finalize, desc_new,
        mgr.merge_accum_sum_k()[layer_idx],
        mgr.merge_accum_sum_v()[layer_idx],
        mgr.merge_accum_pos()[layer_idx],
        mgr.merge_accum_group_ids()[layer_idx],
        mgr.merge_accum_tokens_per_group()[layer_idx],
        mgr.merge_accum_num_groups()[layer_idx],
        mgr.merge_k_pool()[layer_idx],
        mgr.merge_v_pool()[layer_idx],
        mgr.merge_pos_pool()[layer_idx],
        mgr.merge_token_count()[layer_idx],
    )


def mega_state_update_cuda(
    states: Sequence[_ref.PerHeadState],
    new_t_vals: Iterable[int],
    pass_kind: int,
    device: torch.device | str = "cuda",
) -> tuple[
    list[int], list[int], list[int], list[int],
    list[int], list[int], list[int], list[int],
    list[_ref.PerHeadState],
]:
    """End-to-end: pack states → call kernel → unpack states & descriptors.

    Returns 9-element tuple:
        (desc_dst_kind, desc_dst_slot, desc_src_frame, desc_src_head,
         desc_merge_accum_slot, desc_merge_local_idx,
         desc_merge_finalize_completed_idx, desc_merge_is_new_block,
         mutated_states)
    See _mega_state_ref.mega_state_update_ref for descriptor semantics.
    """
    _check_abi()
    dev = torch.device(device)
    H = len(states)
    new_t_list = list(new_t_vals)
    F = len(new_t_list)
    N = H * F

    arr = pack_states(states)
    states_buf = _to_bytes_tensor(arr, dev)
    new_t_tensor = torch.tensor(new_t_list, dtype=torch.int64, device=dev)
    desc_dst_kind = torch.full((N,), -1, dtype=torch.int32, device=dev)
    desc_dst_slot = torch.full((N,), -1, dtype=torch.int32, device=dev)
    desc_src_frame = torch.zeros(N, dtype=torch.int32, device=dev)
    desc_src_head = torch.zeros(N, dtype=torch.int32, device=dev)
    desc_merge_accum_slot = torch.full((N,), -1, dtype=torch.int32, device=dev)
    desc_merge_local_idx = torch.full((N,), -1, dtype=torch.int32, device=dev)
    desc_merge_finalize_completed_idx = torch.full((N,), -1, dtype=torch.int32, device=dev)
    desc_merge_is_new_block = torch.zeros(N, dtype=torch.int32, device=dev)

    _ops.ops().mega_state_update(
        states_buf,
        new_t_tensor,
        desc_dst_kind,
        desc_dst_slot,
        desc_src_frame,
        desc_src_head,
        desc_merge_accum_slot,
        desc_merge_local_idx,
        desc_merge_finalize_completed_idx,
        desc_merge_is_new_block,
        int(H),
        int(F),
        int(pass_kind),
    )

    arr_back = (
        states_buf.cpu()
        .numpy()
        .view(PER_HEAD_STATE_DTYPE)
        .reshape(H)
        .copy()
    )
    mutated = unpack_states(arr_back)

    return (
        desc_dst_kind.cpu().tolist(),
        desc_dst_slot.cpu().tolist(),
        desc_src_frame.cpu().tolist(),
        desc_src_head.cpu().tolist(),
        desc_merge_accum_slot.cpu().tolist(),
        desc_merge_local_idx.cpu().tolist(),
        desc_merge_finalize_completed_idx.cpu().tolist(),
        desc_merge_is_new_block.cpu().tolist(),
        mutated,
    )

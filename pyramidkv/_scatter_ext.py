"""JIT-compiled CUDA scatter-copy extension.

Provides ``scatter_copy``, ``apply_pos_override``, ``refresh_readout_layout``,
and ``anchor_store_write_frames`` for fused KV packing. Falls back gracefully
when CUDA compilation fails.
"""
import os
import torch

_module = None
_available: bool | None = None


def _set_cuda_arch_list_if_unset():
    """Suppress torch.utils.cpp_extension's "TORCH_CUDA_ARCH_LIST is not set"
    warning by populating the env var from the currently visible devices.
    Without this nvcc compiles for every arch supported by the toolkit,
    bloating .so size + JIT time. Honors user override if already set.
    """
    if os.environ.get("TORCH_CUDA_ARCH_LIST"):
        return
    archs = set()
    for i in range(torch.cuda.device_count()):
        major, minor = torch.cuda.get_device_capability(i)
        archs.add(f"{major}.{minor}")
    if archs:
        # PTX appended to the highest arch gives forward-compat without
        # rebuilding for future GPUs.
        sorted_archs = sorted(archs, key=lambda s: tuple(int(x) for x in s.split(".")))
        sorted_archs[-1] = sorted_archs[-1] + "+PTX"
        os.environ["TORCH_CUDA_ARCH_LIST"] = ";".join(sorted_archs)


def _try_load():
    global _module, _available
    if _available is not None:
        return _available
    if not torch.cuda.is_available():
        _available = False
        return False
    verbose = os.environ.get("PYRAMIDKV_VERBOSE_BUILD") == "1"
    try:
        _set_cuda_arch_list_if_unset()
        from torch.utils.cpp_extension import load
        _dir = os.path.dirname(os.path.abspath(__file__))
        _module = load(
            name="pyramidkv_scatter_v30",  # bumped: Plan B — manager takes max_attend_chunks; k_flat_out / v_flat_out / pos_flat_out shrunk to per-attend-call capacity
            sources=[
                os.path.join(_dir, "csrc", "scatter_copy_bind.cpp"),
                os.path.join(_dir, "csrc", "scatter_copy.cu"),
                os.path.join(_dir, "csrc", "pyramidkv_manager.cc"),
                os.path.join(_dir, "csrc", "pyramidkv_pack.cu"),
                os.path.join(_dir, "csrc", "pyramidkv_plan.cc"),
                os.path.join(_dir, "csrc", "pyramidkv_update.cu"),
                os.path.join(_dir, "csrc", "anchor_store_update.cu"),
                os.path.join(_dir, "csrc", "mega_merge_accum.cu"),
                os.path.join(_dir, "csrc", "mega_plan.cc"),
            ],
            extra_include_paths=[os.path.join(_dir, "csrc")],
            verbose=verbose,
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
        )
        _available = True
    except Exception as exc:
        if verbose:
            import traceback
            traceback.print_exc()
        else:
            import sys
            print(
                f"[pyramidkv] JIT build failed: {exc!r}. "
                "Set PYRAMIDKV_VERBOSE_BUILD=1 for full compile log.",
                file=sys.stderr,
            )
        _available = False
    return _available


def scatter_available() -> bool:
    """scatter_copy is opt-in: the per-readout H2D descriptor transfers
    (5 int64 tensors × many calls) outweigh the launch savings versus
    torch.cat(out=workspace). Set PYRAMIDKV_FORCE_SCATTER=1 to enable.
    """
    if os.environ.get("PYRAMIDKV_FORCE_SCATTER") != "1":
        return False
    return _try_load()


def cuda_refresh_available() -> bool:
    """Return whether the CUDA refresh extension can be loaded.

    Unlike scatter_copy(), refresh_readout_layout() is guarded by the caller's
    PYRAMIDKV_CUDA_REFRESH opt-in and does not use PYRAMIDKV_FORCE_SCATTER.
    """
    return _try_load()


def scatter_copy(
    src_ptrs: torch.Tensor,
    lengths: torch.Tensor,
    offsets: torch.Tensor,
    dst: torch.Tensor,
    col_dim: int,
) -> None:
    """Copy multiple source tensors into dst at specified offsets.

    Args:
        src_ptrs: [N] int64 tensor of source data_ptr() values (on CUDA)
        lengths: [N] int64 tensor of row counts per segment
        offsets: [N] int64 tensor of destination row offsets
        dst: [total_len, col_dim] output workspace tensor
        col_dim: number of columns per row
    """
    if not _try_load():
        raise RuntimeError("CUDA scatter extension not available")
    _module.scatter_copy(src_ptrs, lengths, offsets, dst, col_dim)


def apply_pos_override(
    pos: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    vals: torch.Tensor,
) -> None:
    """Override pos[:, 0] = val for each (start, end, val) range.

    Args:
        pos: [total_len, 3] int64 position tensor
        starts: [M] int64 range start indices
        ends: [M] int64 range end indices
        vals: [M] int64 override values
    """
    if not _try_load():
        raise RuntimeError("CUDA scatter extension not available")
    _module.apply_pos_override(pos, starts, ends, vals)


def anchor_store_write_frames(
    k_seq: torch.Tensor,
    v_seq: torch.Tensor,
    pos_seq: torch.Tensor,
    frame_desc: torch.Tensor,
    store_k: torch.Tensor,
    store_v: torch.Tensor,
    store_pos: torch.Tensor,
) -> None:
    """Copy frame slices into contiguous anchor-store slots.

    Args:
        k_seq/v_seq: [num_frames * frame_seqlen, head_dim] CUDA tensors
        pos_seq: [num_frames * frame_seqlen, 3] CUDA int64 tensor
        frame_desc: [N, 2] CUDA int64 tensor, columns are source frame index
            and destination slot index.
        store_k/store_v: [capacity, frame_seqlen, head_dim] CUDA tensors
        store_pos: [capacity, frame_seqlen, 3] CUDA int64 tensor
    """
    if not _try_load():
        raise RuntimeError("CUDA scatter extension not available")
    _module.anchor_store_write_frames(
        k_seq,
        v_seq,
        pos_seq,
        frame_desc,
        store_k,
        store_v,
        store_pos,
    )


def refresh_readout_layout(
    src_k: list[torch.Tensor],
    src_v: list[torch.Tensor],
    src_pos: list[torch.Tensor],
    src_ptrs_k: torch.Tensor,
    src_ptrs_v: torch.Tensor,
    src_ptrs_pos: torch.Tensor,
    offsets: torch.Tensor,
    lengths: torch.Tensor,
    flags: torch.Tensor,
    dynamic_rope_t: torch.Tensor,
    dst_k_raw: torch.Tensor,
    dst_v: torch.Tensor,
    dst_rope_pos: torch.Tensor,
    dst_frame_ids: torch.Tensor,
    head_dim: int,
    current_t: int,
    history_time_mapping_mode: str,
    history_relative_t_max: int,
    history_time_soft_factor: float,
) -> None:
    """Refresh mutable readout segments into existing workspaces.

    The C++ binding writes the Python source tensors' data_ptr() values into
    preallocated device descriptor buffers, then launches one CUDA kernel for
    K/V copy, position mapping, dynamic RoPE time overrides, and frame ids.
    """
    if not _try_load():
        raise RuntimeError("CUDA scatter extension not available")
    _module.refresh_readout_layout(
        src_k,
        src_v,
        src_pos,
        src_ptrs_k,
        src_ptrs_v,
        src_ptrs_pos,
        offsets,
        lengths,
        flags,
        dynamic_rope_t,
        dst_k_raw,
        dst_v,
        dst_rope_pos,
        dst_frame_ids,
        head_dim,
        current_t,
        history_time_mapping_mode,
        history_relative_t_max,
        history_time_soft_factor,
    )

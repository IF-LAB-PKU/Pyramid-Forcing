// mega_state_update_kernel: per-head strategy state mutation.
//
// Single-thread-per-head kernel that walks the per-(layer, head) PerHeadState
// for the current denoising block and:
//   1. Computes the destination kind/slot for each incoming frame
//   2. Mutates the state struct in place (cursors, t-keyed FIFOs, etc.)
//   3. Writes the (kind, slot) descriptor for the data-movement kernel
//      (existing pyramidkv_update_kernel) to consume
//
// The data movement itself stays in pyramidkv_update_kernel — this kernel only
// owns the strategy decision. That keeps each kernel small and lets us reuse
// the proven data-copy path while the strategy dispatch matures.
//
// Grid: (num_heads,) blocks, 1 thread each. State mutation is inherently
// sequential within a head (cursor++, FIFO eviction), but parallel across
// heads. Number of strategies × frames_per_block × heads ≈ 5 × 3 × 12 = 180
// state updates per call — trivially small.
//
// Strategies covered Day 2: cyclic, stride, lag, recent.
// Merge strategy lands Day 3 (more complex: accumulator + finalize state).
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "anchor_store.cuh"

namespace adahead {

namespace {

// Kind encoding matches existing pyramidkv_update_kernel:
//   0 = sink_pool   (sink-grid decoupling — sink writes happen elsewhere)
//   1 = middle_pool (cyclic/stride/lag map here)
//   2 = recent_pool (recent writes happen elsewhere)
//  -1 = skip (this descriptor entry is a no-op, kernel will branch out)
//   5 = merge_accum (frame contributes to accumulator pool, NOT flat slot)
constexpr int32_t DST_KIND_MIDDLE      = 1;
constexpr int32_t DST_KIND_SKIP        = -1;
constexpr int32_t DST_KIND_MERGE_ACCUM = 5;

__device__ __forceinline__ void update_cyclic(
    PerHeadState& state,
    int32_t t_val,
    int32_t& out_kind,
    int32_t& out_slot
) {
    // Phase = t mod period; slot in middle_pool = phase * bucket_cap + cursor.
    // Each phase has a ring of bucket_cap entries; cursor advances mod bucket_cap.
    const int32_t phase = t_val % state.period;
    const int32_t cursor = state.cyclic_cursor[phase];
    const int32_t slot = phase * state.bucket_cap + cursor;

    state.cyclic_slot[slot] = slot;  // mark used (identity mapping)
    state.cyclic_t[slot] = t_val;
    state.cyclic_cursor[phase] = (cursor + 1) % state.bucket_cap;

    out_kind = DST_KIND_MIDDLE;
    out_slot = slot;
}

__device__ __forceinline__ void update_stride(
    PerHeadState& state,
    int32_t t_val,
    int32_t& out_kind,
    int32_t& out_slot
) {
    // Stride only stores frames where t % interval == 0.
    if (t_val % state.interval != 0) {
        out_kind = DST_KIND_SKIP;
        out_slot = -1;
        return;
    }

    // Ring-buffer FIFO: slot = tkey_head; wraps mod capacity. This avoids
    // a metadata shift that would dangle pool slot → t-value mappings
    // (the shift-left FIFO mutated metadata but didn't move pool data).
    const int32_t slot = static_cast<int32_t>(state.tkey_head);
    state.tkey_head =
        static_cast<int8_t>((slot + 1) % state.capacity);
    if (state.tkey_count < state.capacity) {
        state.tkey_count = state.tkey_count + 1;
    }
    state.tkey_slot[slot] = slot;
    state.tkey_t[slot] = t_val;

    out_kind = DST_KIND_MIDDLE;
    out_slot = slot;
}

__device__ __forceinline__ void update_lag(
    PerHeadState& state,
    int32_t t_val,
    int32_t& out_kind,
    int32_t& out_slot
) {
    // Lag stores every frame; ring-buffer FIFO (same semantic as stride).
    const int32_t slot = static_cast<int32_t>(state.tkey_head);
    state.tkey_head =
        static_cast<int8_t>((slot + 1) % state.capacity);
    if (state.tkey_count < state.capacity) {
        state.tkey_count = state.tkey_count + 1;
    }
    state.tkey_slot[slot] = slot;
    state.tkey_t[slot] = t_val;

    out_kind = DST_KIND_MIDDLE;
    out_slot = slot;
}

// Mirrors pyramidkv/_mega_state_ref.py:_update_merge.
// Returns (kind, accum_slot, local_idx). slot stays at -1 — merge frames
// don't have a flat-pool destination; the merge_accum kernel uses
// (accum_slot, local_idx) instead to scatter-add into accumulator buffers.
//
// Important: this device function never raises. The Python ref raises
// MergeDuplicateError on a repeated local_idx; the kernel cannot raise from
// device code, so it sets out_kind = DST_KIND_SKIP and continues. Callers
// that care about duplicate detection must run the Python ref alongside in
// a debug build (Day 4 tests do this implicitly by deepcopy + compare).
__device__ __forceinline__ void update_merge(
    PerHeadState& state,
    int32_t t_val,
    int32_t& out_kind,
    int32_t& out_slot,
    int32_t& out_accum_slot,
    int32_t& out_local_idx,
    int32_t& out_finalize_completed_idx,
    int32_t& out_is_new_block
) {
    if (state.block_frames <= 0) {
        out_kind = DST_KIND_SKIP; out_slot = -1;
        out_accum_slot = -1; out_local_idx = -1;
        out_finalize_completed_idx = -1; out_is_new_block = 0;
        return;
    }
    const int32_t block_id  = t_val / state.block_frames;
    const int32_t local_idx = t_val % state.block_frames;

    // Find existing active slot for this block_id, else allocate.
    int32_t accum_slot = -1;
    for (int s = 0; s < kMaxMergeActive; ++s) {
        if (state.merge_active_block_id[s] == block_id) {
            accum_slot = s;
            break;
        }
    }
    int32_t is_new_block = 0;
    if (accum_slot == -1) {
        for (int s = 0; s < kMaxMergeActive; ++s) {
            if (state.merge_active_block_id[s] == -1) {
                accum_slot = s;
                break;
            }
        }
        if (accum_slot == -1) {
            // All active slots taken — evict slot 0 (mirrors Python ref).
            accum_slot = 0;
            state.merge_active_block_id[0] = -1;
            for (int j = 0; j < kMaxMergeBlockFrames; ++j) {
                state.merge_active_seen[0][j] = 0;
            }
            state.merge_active_complete_count[0] = 0;
        }
        state.merge_active_block_id[accum_slot] = block_id;
        for (int j = 0; j < kMaxMergeBlockFrames; ++j) {
            state.merge_active_seen[accum_slot][j] = 0;
        }
        state.merge_active_complete_count[accum_slot] = 0;
        is_new_block = 1;
    }

    // Duplicate-frame guard: device code can't throw, so on duplicate we
    // mark SKIP and bail without mutating accumulator state. Python ref
    // raises MergeDuplicateError here (covered separately by ref tests).
    if (state.merge_active_seen[accum_slot][local_idx] != 0) {
        out_kind = DST_KIND_SKIP; out_slot = -1;
        out_accum_slot = -1; out_local_idx = -1;
        out_finalize_completed_idx = -1; out_is_new_block = 0;
        return;
    }

    state.merge_active_seen[accum_slot][local_idx] = 1;
    state.merge_active_complete_count[accum_slot] += 1;

    int32_t finalize_completed_idx = -1;

    // Finalize when every local slot in the block has been seen.
    if (state.merge_active_complete_count[accum_slot] >= state.block_frames) {
        const int32_t start_t  = block_id * state.block_frames;
        const int32_t end_t    = start_t + state.block_frames - 1;
        const int32_t median_t = (start_t + end_t) / 2;

        int32_t completed_idx;
        if (state.merge_completed_count < state.capacity) {
            completed_idx = state.merge_completed_count;
            state.merge_completed_count += 1;
        } else {
            // FIFO evict oldest completed (shift left, reuse final index).
            for (int i = 1; i < state.capacity; ++i) {
                state.merge_completed_slot[i - 1]     = state.merge_completed_slot[i];
                state.merge_completed_block_id[i - 1] = state.merge_completed_block_id[i];
                state.merge_completed_start_t[i - 1]  = state.merge_completed_start_t[i];
                state.merge_completed_end_t[i - 1]    = state.merge_completed_end_t[i];
                state.merge_completed_median_t[i - 1] = state.merge_completed_median_t[i];
            }
            completed_idx = state.capacity - 1;
        }
        state.merge_completed_slot[completed_idx]     = completed_idx;
        state.merge_completed_block_id[completed_idx] = block_id;
        state.merge_completed_start_t[completed_idx]  = start_t;
        state.merge_completed_end_t[completed_idx]    = end_t;
        state.merge_completed_median_t[completed_idx] = median_t;

        // Reset the active slot so the next block_id can reuse it.
        state.merge_active_block_id[accum_slot] = -1;
        for (int j = 0; j < kMaxMergeBlockFrames; ++j) {
            state.merge_active_seen[accum_slot][j] = 0;
        }
        state.merge_active_complete_count[accum_slot] = 0;
        finalize_completed_idx = completed_idx;
    }

    out_kind = DST_KIND_MERGE_ACCUM;
    out_slot = -1;
    out_accum_slot = accum_slot;
    out_local_idx  = local_idx;
    out_finalize_completed_idx = finalize_completed_idx;
    out_is_new_block = is_new_block;
}

__global__ void mega_state_update_kernel(
    PerHeadState* __restrict__ states,                     // [H] — one layer's worth
    const int64_t* __restrict__ new_t_vals,                // [frames_in_block]
    int32_t* __restrict__ desc_dst_kind,                   // [H * frames_in_block]
    int32_t* __restrict__ desc_dst_slot,                   // [H * frames_in_block]
    int32_t* __restrict__ desc_src_frame,                  // [H * frames_in_block]
    int32_t* __restrict__ desc_src_head,                   // [H * frames_in_block]
    int32_t* __restrict__ desc_merge_accum_slot,           // [H * frames_in_block]
    int32_t* __restrict__ desc_merge_local_idx,            // [H * frames_in_block]
    int32_t* __restrict__ desc_merge_finalize_completed_idx, // [H * frames_in_block]
    int32_t* __restrict__ desc_merge_is_new_block,         // [H * frames_in_block]
    int H,
    int frames_in_block,
    int pass_kind                                          // 0=noisy 1=clean — only clean writes
) {
    const int h = blockIdx.x;
    if (h >= H) return;

    PerHeadState& state = states[h];
    const int desc_base = h * frames_in_block;

    // Noisy passes don't update the cache; they only read for attention. Only
    // the clean pass mutates state and writes data. We still emit descriptors
    // marked as SKIP so callers' descriptor count stays predictable.
    if (pass_kind != 1) {
        for (int f = 0; f < frames_in_block; ++f) {
            desc_dst_kind[desc_base + f]                    = DST_KIND_SKIP;
            desc_dst_slot[desc_base + f]                    = -1;
            desc_src_frame[desc_base + f]                   = f;
            desc_src_head[desc_base + f]                    = h;
            desc_merge_accum_slot[desc_base + f]            = -1;
            desc_merge_local_idx[desc_base + f]             = -1;
            desc_merge_finalize_completed_idx[desc_base + f] = -1;
            desc_merge_is_new_block[desc_base + f]          = 0;
        }
        return;
    }

    // Sequential frame loop (frames_in_block typically 3-4, no parallelism gain).
    for (int f = 0; f < frames_in_block; ++f) {
        const int32_t t_val = static_cast<int32_t>(new_t_vals[f]);
        int32_t kind = DST_KIND_SKIP;
        int32_t slot = -1;
        int32_t accum_slot = -1;
        int32_t local_idx  = -1;
        int32_t finalize_completed_idx = -1;
        int32_t is_new_block = 0;

        switch (state.kind) {
            case SK_RECENT:
                kind = DST_KIND_SKIP;
                break;
            case SK_CYCLIC:
                update_cyclic(state, t_val, kind, slot);
                break;
            case SK_STRIDE:
                update_stride(state, t_val, kind, slot);
                break;
            case SK_LAG:
                update_lag(state, t_val, kind, slot);
                break;
            case SK_MERGE:
                update_merge(state, t_val, kind, slot,
                             accum_slot, local_idx,
                             finalize_completed_idx, is_new_block);
                break;
            default:
                kind = DST_KIND_SKIP;
                break;
        }

        desc_dst_kind[desc_base + f]                    = kind;
        desc_dst_slot[desc_base + f]                    = slot;
        desc_src_frame[desc_base + f]                   = f;
        desc_src_head[desc_base + f]                    = h;
        desc_merge_accum_slot[desc_base + f]            = accum_slot;
        desc_merge_local_idx[desc_base + f]             = local_idx;
        desc_merge_finalize_completed_idx[desc_base + f] = finalize_completed_idx;
        desc_merge_is_new_block[desc_base + f]          = is_new_block;
    }
}

}  // namespace

// External entry; called from anchor_store_bind.cpp / mega_attention orchestrator.
// Inputs:
//   states      : [num_heads]              PerHeadState (raw bytes), mutated
//   new_t_vals  : [frames_in_block]        int64
//   desc_*      : [num_heads * frames_in_block] int32, output buffers
// The descriptor format matches the existing pyramidkv_update_kernel contract so
// the same data-movement kernel can be chained directly afterwards.
void launch_mega_state_update(
    torch::Tensor states,                              // bytes-as-tensor; cast in kernel
    torch::Tensor new_t_vals,                          // [frames_in_block] int64
    torch::Tensor desc_dst_kind,                       // [N] int32
    torch::Tensor desc_dst_slot,                       // [N] int32
    torch::Tensor desc_src_frame,                      // [N] int32
    torch::Tensor desc_src_head,                       // [N] int32
    torch::Tensor desc_merge_accum_slot,               // [N] int32
    torch::Tensor desc_merge_local_idx,                // [N] int32
    torch::Tensor desc_merge_finalize_completed_idx,   // [N] int32
    torch::Tensor desc_merge_is_new_block,             // [N] int32
    int64_t num_heads,
    int64_t frames_in_block,
    int64_t pass_kind
) {
    TORCH_CHECK(states.is_cuda(), "states must be on CUDA");
    TORCH_CHECK(new_t_vals.is_cuda() && new_t_vals.dtype() == torch::kInt64,
                "new_t_vals must be CUDA int64");
    TORCH_CHECK(desc_dst_kind.dtype() == torch::kInt &&
                desc_dst_slot.dtype() == torch::kInt &&
                desc_src_frame.dtype() == torch::kInt &&
                desc_src_head.dtype() == torch::kInt &&
                desc_merge_accum_slot.dtype() == torch::kInt &&
                desc_merge_local_idx.dtype() == torch::kInt &&
                desc_merge_finalize_completed_idx.dtype() == torch::kInt &&
                desc_merge_is_new_block.dtype() == torch::kInt,
                "all descriptor tensors must be int32");
    TORCH_CHECK(states.numel() * states.element_size() >=
                static_cast<int64_t>(num_heads) * static_cast<int64_t>(sizeof(PerHeadState)),
                "states tensor too small for ", num_heads, " heads");

    const int H = static_cast<int>(num_heads);
    const int F = static_cast<int>(frames_in_block);
    const int PK = static_cast<int>(pass_kind);
    if (H == 0 || F == 0) return;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(states.device().index());
    mega_state_update_kernel<<<H, 1, 0, stream>>>(
        reinterpret_cast<PerHeadState*>(states.data_ptr()),
        new_t_vals.data_ptr<int64_t>(),
        desc_dst_kind.data_ptr<int32_t>(),
        desc_dst_slot.data_ptr<int32_t>(),
        desc_src_frame.data_ptr<int32_t>(),
        desc_src_head.data_ptr<int32_t>(),
        desc_merge_accum_slot.data_ptr<int32_t>(),
        desc_merge_local_idx.data_ptr<int32_t>(),
        desc_merge_finalize_completed_idx.data_ptr<int32_t>(),
        desc_merge_is_new_block.data_ptr<int32_t>(),
        H, F, PK
    );
}

}  // namespace adahead

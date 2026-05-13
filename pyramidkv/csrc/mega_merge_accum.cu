///5e — mega_merge_accum_kernel: K/V scatter-add + divide-and-finalize
// for the merge strategy.
//
// Consumes the descriptor outputs of mega_state_update (Day 5c):
//   desc_dst_kind                    : DST_KIND_MERGE_ACCUM (5) for merge frames
//   desc_merge_accum_slot            : which active accumulator slot (0..MMA-1)
//   desc_merge_is_new_block          : 1 → must zero accumulator first
//   desc_merge_finalize_completed_idx: >=0 → divide+copy to merge_*_pool[idx]
//
// Performs per-(layer, head):
//   1. is_new_block frames: zero merge_accum_sum_{k,v}[head, accum_slot, :, :].
//      Day 5e: compute SPATIAL group_ids from pos: token at (y, x) maps to
//      group_id = (y/patch_size) * patch_cols + (x/patch_size), where
//      patch_cols = max_x / patch_size + 1 across this frame's tokens.
//      num_groups = (max_y/patch_size + 1) * patch_cols (matches Python
//      merge.py:_build_patch_groups). tokens_per_group counts tokens per
//      group then multiplies by block_frames. output_pos[g] uses the
//      group's top-left patch corner (g/patch_cols * patch_size,
//      g % patch_cols * patch_size).
//   2. All merge frames: scatter-add new K/V via atomicAdd into
//      accum_sum_{k,v}[head, accum_slot, group_ids[t], :].
//   3. Finalize frames: divide accumulator by tokens_per_group, convert
//      fp32 → bf16, copy to merge_{k,v}_pool[head, completed_idx, :, :],
//      copy output_pos (captured at new_block) to merge_pos_pool, set
//      merge_token_count = num_groups.
//
// Bit-exact contract with Python merge.py is enforced by
// test_mega_merge_accum_cuda_spatial.py.
//
// Layer is passed as an int; the kernel operates on one layer's slice of
// each tensor. The Python wrapper slices [layer_idx] before calling.
//
// Grid: (num_heads,) blocks, head_dim threads each. Frames within a block
// are processed sequentially with __syncthreads() so frame 0's clear can't
// race against frame 1's scatter-add.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

#include "anchor_store.cuh"

namespace adahead {

namespace {

// Mirror of constants in anchor_store_update.cu — keep in sync.
constexpr int32_t DST_KIND_MERGE_ACCUM = 5;

__global__ void mega_merge_accum_kernel(
    const PerHeadState* __restrict__ states,            // [H]
    const __nv_bfloat16* __restrict__ new_k,            // [F, H, FSEQ, D]
    const __nv_bfloat16* __restrict__ new_v,            // [F, H, FSEQ, D]
    const int64_t* __restrict__ new_pos,                // [F, H, FSEQ, 3]
    const int32_t* __restrict__ desc_dst_kind,          // [H * F]
    const int32_t* __restrict__ desc_merge_accum_slot,  // [H * F]
    const int32_t* __restrict__ desc_merge_finalize_completed_idx,  // [H * F]
    const int32_t* __restrict__ desc_merge_is_new_block,            // [H * F]
    // raw-K storage (raw-K storage) — kernel no longer needs RoPE freq tables;
    // incoming K is already raw so the accumulator is just a plain
    // bf16->fp32 sum-then-divide. Readout applies fresh 3D RoPE.
    float*   __restrict__ accum_sum_k,                  // [H, MMA, MTPA, D]
    float*   __restrict__ accum_sum_v,                  // [H, MMA, MTPA, D]
    int64_t* __restrict__ accum_pos,                    // [H, MMA, MTPA, 3]
    int32_t* __restrict__ accum_group_ids,              // [H, MMA, FSEQ]
    float*   __restrict__ accum_tokens_per_group,       // [H, MMA, MTPA]
    int32_t* __restrict__ accum_num_groups,             // [H, MMA]
    __nv_bfloat16* __restrict__ merge_k_pool,           // [H, MMB, MTPA, D]
    __nv_bfloat16* __restrict__ merge_v_pool,           // [H, MMB, MTPA, D]
    int64_t* __restrict__ merge_pos_pool,               // [H, MMB, MTPA, 3]
    int32_t* __restrict__ merge_token_count,            // [H, MMB]
    int H, int F, int FSEQ, int D,
    int MMA, int MMB, int MTPA
) {
    const int h = blockIdx.x;
    const int d = threadIdx.x;
    if (h >= H || d >= D) return;

    const int32_t block_frames = states[h].block_frames;

    // Process frames sequentially so frame 0's clear-then-add can't race
    // against frame 1's add.
    for (int f = 0; f < F; ++f) {
        const int desc_idx = h * F + f;
        const int32_t kind        = desc_dst_kind[desc_idx];
        const int32_t accum_slot  = desc_merge_accum_slot[desc_idx];
        const int32_t is_new      = desc_merge_is_new_block[desc_idx];
        const int32_t finalize_ix = desc_merge_finalize_completed_idx[desc_idx];

        if (kind != DST_KIND_MERGE_ACCUM || accum_slot < 0) {
            __syncthreads();
            continue;
        }

        // -----------------------------------------------------------------
        // Step 1 — new block: zero accumulator + compute spatial group_ids.
        // -----------------------------------------------------------------
        if (is_new) {
            const int32_t patch_size = states[h].patch_size;
            const int64_t base_kv =
                (static_cast<int64_t>(h) * MMA + accum_slot) * MTPA;

            // Every thread zeros its head_dim slice across MTPA accumulator
            // rows (cap at MTPA — boundary configs may have num_groups < MTPA
            // but the leftover slots stay zero and are simply unused).
            for (int g = 0; g < MTPA; ++g) {
                accum_sum_k[(base_kv + g) * D + d] = 0.0f;
                accum_sum_v[(base_kv + g) * D + d] = 0.0f;
            }

            // Spatial bookkeeping: single-thread within this block so the
            // tokens_per_group histogram doesn't need atomics. patch_size is
            // small (1-4) so this loop costs ~FSEQ writes — negligible.
            if (d == 0) {
                const int64_t gid_base =
                    (static_cast<int64_t>(h) * MMA + accum_slot) * FSEQ;
                const int64_t tpg_base =
                    (static_cast<int64_t>(h) * MMA + accum_slot) * MTPA;
                const int64_t pos_base_acc =
                    ((static_cast<int64_t>(h) * MMA + accum_slot) * MTPA) * 3;
                const int64_t pos_base_new =
                    ((static_cast<int64_t>(f) * H + h) * FSEQ) * 3;

                // Pass 1: find max y, max x to derive patch_cols + num_groups.
                int32_t max_y = -1;
                int32_t max_x = -1;
                for (int t = 0; t < FSEQ; ++t) {
                    const int32_t y =
                        static_cast<int32_t>(new_pos[pos_base_new + t * 3 + 1]);
                    const int32_t x =
                        static_cast<int32_t>(new_pos[pos_base_new + t * 3 + 2]);
                    if (y > max_y) max_y = y;
                    if (x > max_x) max_x = x;
                }
                const int32_t patch_cols =
                    (max_x >= 0 && patch_size > 0) ? (max_x / patch_size + 1) : 1;
                const int32_t patch_rows =
                    (max_y >= 0 && patch_size > 0) ? (max_y / patch_size + 1) : 1;
                const int32_t num_groups = patch_rows * patch_cols;
                // Defensive clamp; PerHeadState.cached_num_groups upper-bound
                // contract is the same as the MTPA tensor allocation budget.
                const int32_t safe_ng =
                    (num_groups > MTPA) ? MTPA : num_groups;

                accum_num_groups[h * MMA + accum_slot] = safe_ng;

                // Pass 2: compute group_ids + tokens-per-group histogram.
                for (int g = 0; g < MTPA; ++g) {
                    accum_tokens_per_group[tpg_base + g] = 0.0f;
                }
                for (int t = 0; t < FSEQ; ++t) {
                    const int32_t y =
                        static_cast<int32_t>(new_pos[pos_base_new + t * 3 + 1]);
                    const int32_t x =
                        static_cast<int32_t>(new_pos[pos_base_new + t * 3 + 2]);
                    const int32_t py = (patch_size > 0) ? (y / patch_size) : 0;
                    const int32_t px = (patch_size > 0) ? (x / patch_size) : 0;
                    int32_t gid = py * patch_cols + px;
                    if (gid < 0 || gid >= safe_ng) gid = 0;  // defensive
                    accum_group_ids[gid_base + t] = gid;
                    accum_tokens_per_group[tpg_base + gid] += 1.0f;
                }
                // Multiply by block_frames — divisor used at finalize time.
                for (int g = 0; g < safe_ng; ++g) {
                    accum_tokens_per_group[tpg_base + g] *=
                        static_cast<float>(block_frames);
                }

                // Pass 3: output_pos for each group — top-left patch corner.
                // pos[0] = median_t (captured later at finalize), pos[1]=y, pos[2]=x.
                // Day 5d captured per-token pos. Day 5e stores per-group pos.
                // For pos[0], reuse this frame's t value across all groups.
                const int64_t t_val = new_pos[pos_base_new + 0 + 0];
                for (int g = 0; g < safe_ng; ++g) {
                    const int32_t py = (g / patch_cols) * patch_size;
                    const int32_t px = (g % patch_cols) * patch_size;
                    accum_pos[pos_base_acc + g * 3 + 0] = t_val;
                    accum_pos[pos_base_acc + g * 3 + 1] = py;
                    accum_pos[pos_base_acc + g * 3 + 2] = px;
                }
            }
        }
        __syncthreads();

        // -----------------------------------------------------------------
        // Step 2 — scatter-add via spatial group_ids.
        // incoming K is raw (causal_rope_apply is skipped before
        // MegaCache.update), so the accumulator is just plain bf16->fp32
        // sum. Phase A.1/A.2 inverse-rotate branches are gone.
        // -----------------------------------------------------------------
        {
            const int64_t kv_base_acc =
                (static_cast<int64_t>(h) * MMA + accum_slot) * MTPA;
            const int64_t kv_base_new =
                (static_cast<int64_t>(f) * H + h) * FSEQ;
            const int64_t gid_base =
                (static_cast<int64_t>(h) * MMA + accum_slot) * FSEQ;
            for (int t = 0; t < FSEQ; ++t) {
                const int32_t gid = accum_group_ids[gid_base + t];
                const __nv_bfloat16 kb = new_k[(kv_base_new + t) * D + d];
                const __nv_bfloat16 vb = new_v[(kv_base_new + t) * D + d];
                const float kf = __bfloat162float(kb);
                const float vf = __bfloat162float(vb);
                accum_sum_k[(kv_base_acc + gid) * D + d] += kf;
                accum_sum_v[(kv_base_acc + gid) * D + d] += vf;
            }
        }
        __syncthreads();

        // -----------------------------------------------------------------
        // Step 3 — finalize: divide accumulator, write to merge_*_pool.
        // -----------------------------------------------------------------
        if (finalize_ix >= 0) {
            const int32_t num_groups = accum_num_groups[h * MMA + accum_slot];
            const int64_t kv_base_acc =
                (static_cast<int64_t>(h) * MMA + accum_slot) * MTPA;
            const int64_t kv_base_pool =
                (static_cast<int64_t>(h) * MMB + finalize_ix) * MTPA;
            for (int t = 0; t < num_groups; ++t) {
                const float divisor =
                    accum_tokens_per_group[kv_base_acc + t];
                const float ksum =
                    accum_sum_k[(kv_base_acc + t) * D + d];
                const float vsum =
                    accum_sum_v[(kv_base_acc + t) * D + d];
                const float inv = (divisor > 0.0f) ? (1.0f / divisor) : 0.0f;
                merge_k_pool[(kv_base_pool + t) * D + d] =
                    __float2bfloat16(ksum * inv);
                merge_v_pool[(kv_base_pool + t) * D + d] =
                    __float2bfloat16(vsum * inv);
            }
            if (d == 0) {
                const int64_t pos_base_acc =
                    ((static_cast<int64_t>(h) * MMA + accum_slot) * MTPA) * 3;
                const int64_t pos_base_pool =
                    ((static_cast<int64_t>(h) * MMB + finalize_ix) * MTPA) * 3;
                for (int t = 0; t < num_groups; ++t) {
                    merge_pos_pool[pos_base_pool + t * 3 + 0] =
                        accum_pos[pos_base_acc + t * 3 + 0];
                    merge_pos_pool[pos_base_pool + t * 3 + 1] =
                        accum_pos[pos_base_acc + t * 3 + 1];
                    merge_pos_pool[pos_base_pool + t * 3 + 2] =
                        accum_pos[pos_base_acc + t * 3 + 2];
                }
                merge_token_count[h * MMB + finalize_ix] = num_groups;
            }
        }
        __syncthreads();
    }
}

}  // namespace

// External entry. The Python wrapper / M3 orchestrator pre-slices the layer
// dimension from manager tensors before calling.
//
// Shape contract (single layer):
//   states         : [H * sizeof(PerHeadState)] uint8
//   new_k / new_v  : [F, H, FSEQ, D] bf16
//   new_pos        : [F, H, FSEQ, 3] int64
//   desc_*         : [H * F]            int32 (4 needed of the 8 emitted)
//   accum_*        : [H, MMA, MTPA, *]  fp32 / int32 / int64
//   merge_*_pool   : [H, MMB, MTPA, *]  bf16 / int64 / int32
//
// MMA / MMB / MTPA are the compile-time bounds from anchor_store.cuh.
void launch_mega_merge_accum(
    torch::Tensor states,
    torch::Tensor new_k,
    torch::Tensor new_v,
    torch::Tensor new_pos,
    torch::Tensor desc_dst_kind,
    torch::Tensor desc_merge_accum_slot,
    torch::Tensor desc_merge_finalize_completed_idx,
    torch::Tensor desc_merge_is_new_block,
    torch::Tensor accum_sum_k,
    torch::Tensor accum_sum_v,
    torch::Tensor accum_pos,
    torch::Tensor accum_group_ids,
    torch::Tensor accum_tokens_per_group,
    torch::Tensor accum_num_groups,
    torch::Tensor merge_k_pool,
    torch::Tensor merge_v_pool,
    torch::Tensor merge_pos_pool,
    torch::Tensor merge_token_count
) {
    TORCH_CHECK(new_k.is_cuda() && new_v.is_cuda() && new_pos.is_cuda(),
                "mega_merge_accum: new_k/v/pos must be CUDA");
    TORCH_CHECK(new_k.dtype() == torch::kBFloat16,
                "mega_merge_accum: new_k must be bf16");
    TORCH_CHECK(new_v.dtype() == torch::kBFloat16,
                "mega_merge_accum: new_v must be bf16");
    TORCH_CHECK(new_pos.dtype() == torch::kInt64,
                "mega_merge_accum: new_pos must be int64");
    TORCH_CHECK(accum_sum_k.dtype() == torch::kFloat &&
                accum_sum_v.dtype() == torch::kFloat,
                "mega_merge_accum: accumulators must be fp32");
    TORCH_CHECK(merge_k_pool.dtype() == torch::kBFloat16 &&
                merge_v_pool.dtype() == torch::kBFloat16,
                "mega_merge_accum: merge_k/v_pool must be bf16");

    const int F     = static_cast<int>(new_k.size(0));
    const int H     = static_cast<int>(new_k.size(1));
    const int FSEQ  = static_cast<int>(new_k.size(2));
    const int D     = static_cast<int>(new_k.size(3));
    const int MMA   = static_cast<int>(accum_sum_k.size(1));
    const int MMB   = static_cast<int>(merge_k_pool.size(1));
    const int MTPA  = static_cast<int>(accum_sum_k.size(2));
    // NOTE: identity grouping (patch_size=1) requires MTPA>=FSEQ. For
    // patch_size>1 the constraint is num_groups<=MTPA which holds when
    // ceil(spatial_h/patch_size) * ceil(spatial_w/patch_size) <= MTPA.
    // The caller is responsible for sizing MTPA accordingly; the kernel
    // caps num_groups_eff at MTPA defensively.
    if (H == 0 || F == 0) return;

    cudaStream_t stream =
        at::cuda::getCurrentCUDAStream(new_k.device().index());
    mega_merge_accum_kernel<<<H, D, 0, stream>>>(
        reinterpret_cast<const PerHeadState*>(states.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(new_k.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(new_v.data_ptr()),
        new_pos.data_ptr<int64_t>(),
        desc_dst_kind.data_ptr<int32_t>(),
        desc_merge_accum_slot.data_ptr<int32_t>(),
        desc_merge_finalize_completed_idx.data_ptr<int32_t>(),
        desc_merge_is_new_block.data_ptr<int32_t>(),
        accum_sum_k.data_ptr<float>(),
        accum_sum_v.data_ptr<float>(),
        accum_pos.data_ptr<int64_t>(),
        accum_group_ids.data_ptr<int32_t>(),
        accum_tokens_per_group.data_ptr<float>(),
        accum_num_groups.data_ptr<int32_t>(),
        reinterpret_cast<__nv_bfloat16*>(merge_k_pool.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(merge_v_pool.data_ptr()),
        merge_pos_pool.data_ptr<int64_t>(),
        merge_token_count.data_ptr<int32_t>(),
        H, F, FSEQ, D, MMA, MMB, MTPA
    );
}

}  // namespace adahead

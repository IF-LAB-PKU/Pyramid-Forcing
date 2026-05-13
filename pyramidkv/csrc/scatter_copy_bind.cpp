#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <vector>

#include "anchor_store.cuh"
#include "pyramidkv_manager.h"

// Forward declarations of CUDA kernels
void launch_scatter_copy(
    torch::Tensor src_ptrs,
    torch::Tensor dst,
    torch::Tensor offsets,
    torch::Tensor lengths,
    int64_t col_dim,
    int N
);

void launch_pos_override(
    torch::Tensor pos,
    torch::Tensor starts,
    torch::Tensor ends,
    torch::Tensor vals
);

void launch_refresh_readout_layout(
    torch::Tensor src_ptrs_k,
    torch::Tensor src_ptrs_v,
    torch::Tensor src_ptrs_pos,
    torch::Tensor offsets,
    torch::Tensor lengths,
    torch::Tensor flags,
    torch::Tensor dynamic_rope_t,
    torch::Tensor dst_k_raw,
    torch::Tensor dst_v,
    torch::Tensor dst_rope_pos,
    torch::Tensor dst_frame_ids,
    int64_t head_dim,
    int64_t current_t,
    int64_t mapping_mode,
    int64_t history_relative_t_max,
    double history_time_soft_factor,
    int N
);

void launch_anchor_store_write_frames(
    torch::Tensor k_seq,
    torch::Tensor v_seq,
    torch::Tensor pos_seq,
    torch::Tensor frame_desc,
    torch::Tensor store_k,
    torch::Tensor store_v,
    torch::Tensor store_pos,
    int64_t frame_seqlen,
    int64_t head_dim,
    int N
);

namespace adahead {
void launch_pyramidkv_pack(
    torch::Tensor sink_pool,
    torch::Tensor middle_pool,
    torch::Tensor recent_pool,
    torch::Tensor sink_v_pool,
    torch::Tensor middle_v_pool,
    torch::Tensor recent_v_pool,
    torch::Tensor merge_pool,
    torch::Tensor merge_v_pool,
    torch::Tensor sink_pos_pool,
    torch::Tensor middle_pos_pool,
    torch::Tensor recent_pos_pool,
    torch::Tensor merge_pos_pool,
    torch::Tensor k_flat_out,
    torch::Tensor v_flat_out,
    torch::Tensor pos_flat_out,
    torch::Tensor src_kind,
    torch::Tensor src_slot_global,
    torch::Tensor seg_lengths,
    torch::Tensor dst_token_offsets,
    torch::Tensor anchor_t_remap,
    torch::Tensor freqs_ft_flat,
    torch::Tensor freqs_fy_flat,
    torch::Tensor freqs_fx_flat,
    int64_t frame_seqlen,
    int64_t max_merge_tokens,
    int64_t head_dim
);

std::tuple<
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor
> _pyramidkv_plan_cuda(
    c10::intrusive_ptr<PyramidKVCacheManager> mgr,
    int64_t current_t,
    int64_t pass_kind
);

std::tuple<
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor
> _pyramidkv_plan_meta(
    c10::intrusive_ptr<PyramidKVCacheManager> mgr,
    int64_t current_t,
    int64_t pass_kind
);

// per-layer mega_plan op. See mega_plan.cc for body.
std::tuple<
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor
> _mega_plan_cuda(
    c10::intrusive_ptr<PyramidKVCacheManager> mgr,
    torch::Tensor states_bytes,
    int64_t layer_idx,
    int64_t current_t,
    int64_t pass_kind,
    int64_t sink_mode_code,
    int64_t sink_clamp_min,
    int64_t sink_clamp_max,
    int64_t decoupled_sink_lag,
    int64_t hist_mode_code,
    int64_t hist_relative_t_max,
    double  hist_soft_factor
);

// multi-chunk plan op. Same as _mega_plan_cuda but takes a
// current_t_list (int64 [num_chunks]) and emits chunk-concatenated
// descriptors with chunk-offset dst_token_offsets.
std::tuple<
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor
> _mega_plan_multi_cuda(
    c10::intrusive_ptr<PyramidKVCacheManager> mgr,
    torch::Tensor states_bytes,
    int64_t layer_idx,
    torch::Tensor current_t_list,
    int64_t pass_kind,
    int64_t sink_mode_code,
    int64_t sink_clamp_min,
    int64_t sink_clamp_max,
    int64_t decoupled_sink_lag,
    int64_t hist_mode_code,
    int64_t hist_relative_t_max,
    double  hist_soft_factor
);

std::tuple<
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor
> _mega_plan_multi_meta(
    c10::intrusive_ptr<PyramidKVCacheManager> mgr,
    torch::Tensor states_bytes,
    int64_t layer_idx,
    torch::Tensor current_t_list,
    int64_t pass_kind,
    int64_t sink_mode_code,
    int64_t sink_clamp_min,
    int64_t sink_clamp_max,
    int64_t decoupled_sink_lag,
    int64_t hist_mode_code,
    int64_t hist_relative_t_max,
    double  hist_soft_factor
);

std::tuple<
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor
> _mega_plan_meta(
    c10::intrusive_ptr<PyramidKVCacheManager> mgr,
    torch::Tensor states_bytes,
    int64_t layer_idx,
    int64_t current_t,
    int64_t pass_kind,
    int64_t sink_mode_code,
    int64_t sink_clamp_min,
    int64_t sink_clamp_max,
    int64_t decoupled_sink_lag,
    int64_t hist_mode_code,
    int64_t hist_relative_t_max,
    double  hist_soft_factor
);

void launch_pyramidkv_update(
    torch::Tensor new_k,
    torch::Tensor new_v,
    torch::Tensor new_pos,
    torch::Tensor sink_k_pool,
    torch::Tensor sink_v_pool,
    torch::Tensor middle_k_pool,
    torch::Tensor middle_v_pool,
    torch::Tensor recent_k_pool,
    torch::Tensor recent_v_pool,
    torch::Tensor sink_pos_pool,
    torch::Tensor middle_pos_pool,
    torch::Tensor recent_pos_pool,
    torch::Tensor dst_kind,
    torch::Tensor dst_slot_global,
    torch::Tensor src_frame_in_block,
    torch::Tensor src_head,
    int64_t H,
    int64_t frame_seqlen,
    int64_t head_dim
);

// strategy state mutation kernel (all 5 strategies, including
// merge accumulator with finalize + new_block signals). See
// anchor_store_update.cu for kernel definition.
void launch_mega_state_update(
    torch::Tensor states,
    torch::Tensor new_t_vals,
    torch::Tensor desc_dst_kind,
    torch::Tensor desc_dst_slot,
    torch::Tensor desc_src_frame,
    torch::Tensor desc_src_head,
    torch::Tensor desc_merge_accum_slot,
    torch::Tensor desc_merge_local_idx,
    torch::Tensor desc_merge_finalize_completed_idx,
    torch::Tensor desc_merge_is_new_block,
    int64_t num_heads,
    int64_t frames_in_block,
    int64_t pass_kind
);

// merge K/V scatter-add + divide-and-finalize.
// See mega_merge_accum.cu for kernel definition.
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
);
}  // namespace adahead

// Public op that operates on the manager directly. The descriptor tensors
// (src_kind / src_slot_global / seg_lengths / dst_token_offsets) come from
// the plan stage; for M1.4 they are produced by the test or by the upcoming
// pyramidkv_plan() helper. The kernel writes mgr.k_flat_out_ / v_flat_out_.
static void pyramidkv_pack(
    c10::intrusive_ptr<adahead::PyramidKVCacheManager> mgr,
    torch::Tensor src_kind,
    torch::Tensor src_slot_global,
    torch::Tensor seg_lengths,
    torch::Tensor dst_token_offsets,
    torch::Tensor anchor_t_remap,
    torch::Tensor freqs_ft_flat,
    torch::Tensor freqs_fy_flat,
    torch::Tensor freqs_fx_flat
) {
    TORCH_CHECK(mgr.get() != nullptr, "pyramidkv_pack: manager is null");
    adahead::launch_pyramidkv_pack(
        mgr->sink_k_pool(),
        mgr->middle_k_pool(),
        mgr->recent_k_pool(),
        mgr->sink_v_pool(),
        mgr->middle_v_pool(),
        mgr->recent_v_pool(),
        mgr->merge_k_pool(),
        mgr->merge_v_pool(),
        // pos pools threaded through for raw-K storage. pos_flat_out
        // is always allocated by the manager; pack writes into it whenever the
        // source pos pools have data.
        mgr->sink_pos_pool(),
        mgr->middle_pos_pool(),
        mgr->recent_pos_pool(),
        mgr->merge_pos_pool(),
        mgr->k_flat_out(),
        mgr->v_flat_out(),
        mgr->pos_flat_out(),
        src_kind, src_slot_global, seg_lengths, dst_token_offsets,
        // fused-RoPE descriptors. Pass empty tensors to fall back to
        // pure memcpy (used by tests / legacy callers).
        anchor_t_remap, freqs_ft_flat, freqs_fy_flat, freqs_fx_flat,
        mgr->frame_seqlen(),
        mgr->max_merge_tokens_per_anchor(),
        mgr->head_dim()
    );
}

static void pyramidkv_pack_meta(
    c10::intrusive_ptr<adahead::PyramidKVCacheManager> /*mgr*/,
    torch::Tensor /*src_kind*/,
    torch::Tensor /*src_slot_global*/,
    torch::Tensor /*seg_lengths*/,
    torch::Tensor /*dst_token_offsets*/,
    torch::Tensor /*anchor_t_remap*/,
    torch::Tensor /*freqs_ft_flat*/,
    torch::Tensor /*freqs_fy_flat*/,
    torch::Tensor /*freqs_fx_flat*/
) {}

// update op: writes incoming K/V frames into manager pool slots.
// dst_kind / dst_slot_global / src_frame_in_block / src_head describe each
// individual frame write; the kernel does pure data movement (no strategy
// decisions). Strategy logic stays in the Python plan path until M1.6.
//
// raw-K storage (raw-K storage): a new_pos param threads per-token (t, y, x)
// in alongside K/V. Pass empty for legacy paths that don't track pos.
static void pyramidkv_update(
    c10::intrusive_ptr<adahead::PyramidKVCacheManager> mgr,
    torch::Tensor new_k,           // [frames_in_block, H, frame_seqlen, head_dim]
    torch::Tensor new_v,
    torch::Tensor new_pos,         // [frames_in_block, H, frame_seqlen, 3] int64 OR empty
    torch::Tensor dst_kind,
    torch::Tensor dst_slot_global,
    torch::Tensor src_frame_in_block,
    torch::Tensor src_head
) {
    TORCH_CHECK(mgr.get() != nullptr, "pyramidkv_update: manager is null");
    adahead::launch_pyramidkv_update(
        new_k, new_v, new_pos,
        mgr->sink_k_pool(), mgr->sink_v_pool(),
        mgr->middle_k_pool(), mgr->middle_v_pool(),
        mgr->recent_k_pool(), mgr->recent_v_pool(),
        mgr->sink_pos_pool(), mgr->middle_pos_pool(), mgr->recent_pos_pool(),
        dst_kind, dst_slot_global, src_frame_in_block, src_head,
        mgr->num_heads(),
        mgr->frame_seqlen(),
        mgr->head_dim()
    );
}

static void pyramidkv_update_meta(
    c10::intrusive_ptr<adahead::PyramidKVCacheManager> /*mgr*/,
    torch::Tensor /*new_k*/,
    torch::Tensor /*new_v*/,
    torch::Tensor /*new_pos*/,
    torch::Tensor /*dst_kind*/,
    torch::Tensor /*dst_slot_global*/,
    torch::Tensor /*src_frame_in_block*/,
    torch::Tensor /*src_head*/
) {}

// mega_state_update wrapper.
// `states` is a raw uint8 byte buffer sized num_heads * sizeof(PerHeadState).
// The kernel reinterpret_cast's it to PerHeadState* and mutates in place.
//
// 8 descriptor outputs. The last two new signals (finalize_completed_idx,
// is_new_block) drive the merge_accum kernel: a new-block frame must zero
// the accumulator first; a finalize frame must divide-and-copy the
// accumulator into merge_*_pool[completed_idx].
static void mega_state_update(
    torch::Tensor states,                              // [num_heads * sizeof(PerHeadState)] uint8 (a!)
    torch::Tensor new_t_vals,                          // [frames_in_block] int64
    torch::Tensor desc_dst_kind,                       // [N] int32 (b!)
    torch::Tensor desc_dst_slot,                       // [N] int32 (c!)
    torch::Tensor desc_src_frame,                      // [N] int32 (d!)
    torch::Tensor desc_src_head,                       // [N] int32 (e!)
    torch::Tensor desc_merge_accum_slot,               // [N] int32 (f!)
    torch::Tensor desc_merge_local_idx,                // [N] int32 (g!)
    torch::Tensor desc_merge_finalize_completed_idx,   // [N] int32 (h!)
    torch::Tensor desc_merge_is_new_block,             // [N] int32 (i!)
    int64_t num_heads,
    int64_t frames_in_block,
    int64_t pass_kind
) {
    TORCH_CHECK(states.is_cuda(), "mega_state_update: states must be CUDA");
    TORCH_CHECK(states.dtype() == torch::kUInt8,
                "mega_state_update: states must be uint8 (raw byte buffer)");
    TORCH_CHECK(states.is_contiguous(), "mega_state_update: states must be contiguous");
    const int64_t expect_bytes =
        num_heads * static_cast<int64_t>(sizeof(adahead::PerHeadState));
    TORCH_CHECK(states.numel() >= expect_bytes,
                "mega_state_update: states buffer too small (",
                states.numel(), " < ", expect_bytes, ")");
    adahead::launch_mega_state_update(
        states, new_t_vals,
        desc_dst_kind, desc_dst_slot, desc_src_frame, desc_src_head,
        desc_merge_accum_slot, desc_merge_local_idx,
        desc_merge_finalize_completed_idx, desc_merge_is_new_block,
        num_heads, frames_in_block, pass_kind
    );
}

static void mega_state_update_meta(
    torch::Tensor /*states*/,
    torch::Tensor /*new_t_vals*/,
    torch::Tensor /*desc_dst_kind*/,
    torch::Tensor /*desc_dst_slot*/,
    torch::Tensor /*desc_src_frame*/,
    torch::Tensor /*desc_src_head*/,
    torch::Tensor /*desc_merge_accum_slot*/,
    torch::Tensor /*desc_merge_local_idx*/,
    torch::Tensor /*desc_merge_finalize_completed_idx*/,
    torch::Tensor /*desc_merge_is_new_block*/,
    int64_t /*num_heads*/,
    int64_t /*frames_in_block*/,
    int64_t /*pass_kind*/
) {}

// ABI probe — Python uses this to validate that its numpy structured-dtype
// layout matches the C++ struct exactly. Rebuilding the extension after a
// PerHeadState field change should always produce the matching size.
static int64_t mega_state_perhead_size() {
    return static_cast<int64_t>(sizeof(adahead::PerHeadState));
}

// mega_merge_accum wrapper. Caller pre-slices manager tensors
// by layer_idx; the kernel sees one layer's [H, ...] views of each pool.
static void mega_merge_accum(
    torch::Tensor states,                              // [H * sizeof(PerHeadState)] uint8
    torch::Tensor new_k,                               // [F, H, FSEQ, D] bf16 — raw K (raw-K storage)
    torch::Tensor new_v,                               // [F, H, FSEQ, D] bf16
    torch::Tensor new_pos,                             // [F, H, FSEQ, 3] int64
    torch::Tensor desc_dst_kind,                       // [H * F] int32
    torch::Tensor desc_merge_accum_slot,               // [H * F] int32
    torch::Tensor desc_merge_finalize_completed_idx,   // [H * F] int32
    torch::Tensor desc_merge_is_new_block,             // [H * F] int32
    torch::Tensor accum_sum_k,                         // [H, MMA, MTPA, D] fp32 (a!)
    torch::Tensor accum_sum_v,                         // [H, MMA, MTPA, D] fp32 (b!)
    torch::Tensor accum_pos,                           // [H, MMA, MTPA, 3] int64 (c!)
    torch::Tensor accum_group_ids,                     // [H, MMA, FSEQ] int32 (d!)
    torch::Tensor accum_tokens_per_group,              // [H, MMA, MTPA] fp32 (e!)
    torch::Tensor accum_num_groups,                    // [H, MMA] int32 (f!)
    torch::Tensor merge_k_pool,                        // [H, MMB, MTPA, D] bf16 (g!)
    torch::Tensor merge_v_pool,                        // [H, MMB, MTPA, D] bf16 (h!)
    torch::Tensor merge_pos_pool,                      // [H, MMB, MTPA, 3] int64 (i!)
    torch::Tensor merge_token_count                    // [H, MMB] int32 (j!)
) {
    adahead::launch_mega_merge_accum(
        states, new_k, new_v, new_pos,
        desc_dst_kind, desc_merge_accum_slot,
        desc_merge_finalize_completed_idx, desc_merge_is_new_block,
        accum_sum_k, accum_sum_v, accum_pos,
        accum_group_ids, accum_tokens_per_group, accum_num_groups,
        merge_k_pool, merge_v_pool, merge_pos_pool, merge_token_count
    );
}

static void mega_merge_accum_meta(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor
) {}

// Python-facing functions

void scatter_copy(
    torch::Tensor src_ptrs,     // [N] int64 device pointers
    torch::Tensor lengths,      // [N] int64 row counts
    torch::Tensor offsets,      // [N] int64 row offsets into dst
    torch::Tensor dst,          // [total_len, col_dim] output workspace
    int64_t col_dim             // number of columns
) {
    TORCH_CHECK(src_ptrs.is_cuda(), "src_ptrs must be on CUDA");
    TORCH_CHECK(dst.is_cuda(), "dst must be on CUDA");
    TORCH_CHECK(src_ptrs.dtype() == torch::kInt64, "src_ptrs must be int64");

    int N = src_ptrs.size(0);
    if (N == 0) return;

    launch_scatter_copy(src_ptrs, dst, offsets, lengths, col_dim, N);
}

void apply_pos_override(
    torch::Tensor pos,          // [total_len, 3] int64
    torch::Tensor starts,       // [M] int64
    torch::Tensor ends,         // [M] int64
    torch::Tensor vals          // [M] int64
) {
    TORCH_CHECK(pos.is_cuda(), "pos must be on CUDA");
    TORCH_CHECK(pos.size(1) == 3, "pos must have 3 columns");

    int M = starts.size(0);
    if (M == 0) return;

    launch_pos_override(pos, starts, ends, vals);
}

static int64_t mapping_mode_code(const std::string& mode) {
    if (mode == "none") return 0;
    if (mode == "relative_clamp") return 1;
    if (mode == "relative_softcap") return 2;
    return 0;
}

static void check_cuda_i64_1d(const torch::Tensor& t, const char* name, int64_t n) {
    TORCH_CHECK(t.is_cuda(), name, " must be on CUDA");
    TORCH_CHECK(t.dtype() == torch::kInt64, name, " must be int64");
    TORCH_CHECK(t.dim() == 1, name, " must be 1D");
    TORCH_CHECK(t.size(0) >= n, name, " is too small");
}

static void copy_ptrs_to_device(
    const std::vector<torch::Tensor>& tensors,
    torch::Tensor dst_ptrs,
    const char* name
) {
    const int64_t n = static_cast<int64_t>(tensors.size());
    std::vector<int64_t> host_ptrs;
    host_ptrs.reserve(n);
    c10::optional<c10::Device> device;
    for (int64_t i = 0; i < n; ++i) {
        const torch::Tensor& t = tensors[i];
        TORCH_CHECK(t.is_cuda(), name, "[", i, "] must be on CUDA");
        TORCH_CHECK(t.is_contiguous(), name, "[", i, "] must be contiguous");
        if (!device.has_value()) {
            device = t.device();
        } else {
            TORCH_CHECK(t.device() == device.value(), name, "[", i, "] is on a different CUDA device");
        }
        host_ptrs.push_back(reinterpret_cast<int64_t>(t.data_ptr()));
    }
    if (n == 0) return;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(dst_ptrs.device().index());
    cudaMemcpyAsync(
        dst_ptrs.data_ptr<int64_t>(),
        host_ptrs.data(),
        static_cast<size_t>(n) * sizeof(int64_t),
        cudaMemcpyHostToDevice,
        stream
    );
}

void refresh_readout_layout(
    std::vector<torch::Tensor> src_k,
    std::vector<torch::Tensor> src_v,
    std::vector<torch::Tensor> src_pos,
    torch::Tensor src_ptrs_k,
    torch::Tensor src_ptrs_v,
    torch::Tensor src_ptrs_pos,
    torch::Tensor offsets,
    torch::Tensor lengths,
    torch::Tensor flags,
    torch::Tensor dynamic_rope_t,
    torch::Tensor dst_k_raw,
    torch::Tensor dst_v,
    torch::Tensor dst_rope_pos,
    torch::Tensor dst_frame_ids,
    int64_t head_dim,
    int64_t current_t,
    std::string history_time_mapping_mode,
    int64_t history_relative_t_max,
    double history_time_soft_factor
) {
    const int N = static_cast<int>(src_k.size());
    TORCH_CHECK(src_v.size() == src_k.size(), "src_v length must match src_k");
    TORCH_CHECK(src_pos.size() == src_k.size(), "src_pos length must match src_k");
    if (N == 0) return;

    TORCH_CHECK(dst_k_raw.is_cuda(), "dst_k_raw must be on CUDA");
    TORCH_CHECK(dst_v.is_cuda(), "dst_v must be on CUDA");
    TORCH_CHECK(dst_rope_pos.is_cuda(), "dst_rope_pos must be on CUDA");
    TORCH_CHECK(dst_frame_ids.is_cuda(), "dst_frame_ids must be on CUDA");
    TORCH_CHECK(dst_rope_pos.dtype() == torch::kInt64, "dst_rope_pos must be int64");
    TORCH_CHECK(dst_frame_ids.dtype() == torch::kInt64, "dst_frame_ids must be int64");
    TORCH_CHECK(dst_k_raw.is_contiguous(), "dst_k_raw must be contiguous");
    TORCH_CHECK(dst_v.is_contiguous(), "dst_v must be contiguous");
    TORCH_CHECK(dst_rope_pos.is_contiguous(), "dst_rope_pos must be contiguous");
    TORCH_CHECK(dst_frame_ids.is_contiguous(), "dst_frame_ids must be contiguous");
    TORCH_CHECK(dst_k_raw.device() == dst_v.device(), "dst_k_raw/dst_v device mismatch");
    TORCH_CHECK(dst_k_raw.device() == dst_rope_pos.device(), "dst_k_raw/dst_rope_pos device mismatch");
    TORCH_CHECK(dst_k_raw.device() == dst_frame_ids.device(), "dst_k_raw/dst_frame_ids device mismatch");
    TORCH_CHECK(dst_k_raw.dtype() == dst_v.dtype(), "dst_k_raw/dst_v dtype mismatch");
    TORCH_CHECK(dst_k_raw.size(1) == head_dim, "dst_k_raw head_dim mismatch");
    TORCH_CHECK(dst_v.size(1) == head_dim, "dst_v head_dim mismatch");
    TORCH_CHECK(dst_rope_pos.size(1) == 3, "dst_rope_pos must have 3 columns");

    check_cuda_i64_1d(src_ptrs_k, "src_ptrs_k", N);
    check_cuda_i64_1d(src_ptrs_v, "src_ptrs_v", N);
    check_cuda_i64_1d(src_ptrs_pos, "src_ptrs_pos", N);
    check_cuda_i64_1d(offsets, "offsets", N);
    check_cuda_i64_1d(lengths, "lengths", N);
    check_cuda_i64_1d(flags, "flags", N);
    check_cuda_i64_1d(dynamic_rope_t, "dynamic_rope_t", N);
    TORCH_CHECK(src_ptrs_k.device() == dst_k_raw.device(), "descriptor device mismatch");
    TORCH_CHECK(src_ptrs_v.device() == dst_k_raw.device(), "descriptor device mismatch");
    TORCH_CHECK(src_ptrs_pos.device() == dst_k_raw.device(), "descriptor device mismatch");
    TORCH_CHECK(offsets.device() == dst_k_raw.device(), "descriptor device mismatch");
    TORCH_CHECK(lengths.device() == dst_k_raw.device(), "descriptor device mismatch");
    TORCH_CHECK(flags.device() == dst_k_raw.device(), "descriptor device mismatch");
    TORCH_CHECK(dynamic_rope_t.device() == dst_k_raw.device(), "descriptor device mismatch");

    for (int i = 0; i < N; ++i) {
        TORCH_CHECK(src_k[i].dtype() == dst_k_raw.dtype(), "src_k dtype mismatch");
        TORCH_CHECK(src_v[i].dtype() == dst_v.dtype(), "src_v dtype mismatch");
        TORCH_CHECK(src_pos[i].dtype() == torch::kInt64, "src_pos must be int64");
        TORCH_CHECK(src_k[i].size(1) == head_dim, "src_k head_dim mismatch");
        TORCH_CHECK(src_v[i].size(1) == head_dim, "src_v head_dim mismatch");
        TORCH_CHECK(src_pos[i].size(1) == 3, "src_pos must have 3 columns");
        TORCH_CHECK(src_k[i].device() == dst_k_raw.device(), "src_k device mismatch");
        TORCH_CHECK(src_v[i].device() == dst_v.device(), "src_v device mismatch");
        TORCH_CHECK(src_pos[i].device() == dst_rope_pos.device(), "src_pos device mismatch");
    }

    copy_ptrs_to_device(src_k, src_ptrs_k, "src_k");
    copy_ptrs_to_device(src_v, src_ptrs_v, "src_v");
    copy_ptrs_to_device(src_pos, src_ptrs_pos, "src_pos");

    launch_refresh_readout_layout(
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
        mapping_mode_code(history_time_mapping_mode),
        history_relative_t_max,
        history_time_soft_factor,
        N
    );
}

void anchor_store_write_frames(
    torch::Tensor k_seq,
    torch::Tensor v_seq,
    torch::Tensor pos_seq,
    torch::Tensor frame_desc,
    torch::Tensor store_k,
    torch::Tensor store_v,
    torch::Tensor store_pos
) {
    TORCH_CHECK(k_seq.is_cuda(), "k_seq must be on CUDA");
    TORCH_CHECK(v_seq.is_cuda(), "v_seq must be on CUDA");
    TORCH_CHECK(pos_seq.is_cuda(), "pos_seq must be on CUDA");
    TORCH_CHECK(frame_desc.is_cuda(), "frame_desc must be on CUDA");
    TORCH_CHECK(store_k.is_cuda(), "store_k must be on CUDA");
    TORCH_CHECK(store_v.is_cuda(), "store_v must be on CUDA");
    TORCH_CHECK(store_pos.is_cuda(), "store_pos must be on CUDA");
    TORCH_CHECK(k_seq.is_contiguous(), "k_seq must be contiguous");
    TORCH_CHECK(v_seq.is_contiguous(), "v_seq must be contiguous");
    TORCH_CHECK(pos_seq.is_contiguous(), "pos_seq must be contiguous");
    TORCH_CHECK(frame_desc.is_contiguous(), "frame_desc must be contiguous");
    TORCH_CHECK(store_k.is_contiguous(), "store_k must be contiguous");
    TORCH_CHECK(store_v.is_contiguous(), "store_v must be contiguous");
    TORCH_CHECK(store_pos.is_contiguous(), "store_pos must be contiguous");
    TORCH_CHECK(k_seq.dtype() == store_k.dtype(), "k_seq/store_k dtype mismatch");
    TORCH_CHECK(v_seq.dtype() == store_v.dtype(), "v_seq/store_v dtype mismatch");
    TORCH_CHECK(pos_seq.dtype() == torch::kInt64, "pos_seq must be int64");
    TORCH_CHECK(store_pos.dtype() == torch::kInt64, "store_pos must be int64");
    TORCH_CHECK(frame_desc.dtype() == torch::kInt64, "frame_desc must be int64");
    TORCH_CHECK(k_seq.dim() == 2, "k_seq must be [tokens, head_dim]");
    TORCH_CHECK(v_seq.dim() == 2, "v_seq must be [tokens, head_dim]");
    TORCH_CHECK(pos_seq.dim() == 2 && pos_seq.size(1) == 3, "pos_seq must be [tokens, 3]");
    TORCH_CHECK(frame_desc.dim() == 2 && frame_desc.size(1) == 2, "frame_desc must be [N, 2]");
    TORCH_CHECK(store_k.dim() == 3, "store_k must be [capacity, frame_seqlen, head_dim]");
    TORCH_CHECK(store_v.dim() == 3, "store_v must be [capacity, frame_seqlen, head_dim]");
    TORCH_CHECK(store_pos.dim() == 3 && store_pos.size(2) == 3, "store_pos must be [capacity, frame_seqlen, 3]");
    TORCH_CHECK(store_k.size(1) == store_v.size(1), "store_k/store_v frame_seqlen mismatch");
    TORCH_CHECK(store_k.size(2) == store_v.size(2), "store_k/store_v head_dim mismatch");
    TORCH_CHECK(store_pos.size(1) == store_k.size(1), "store_pos frame_seqlen mismatch");
    TORCH_CHECK(k_seq.size(1) == store_k.size(2), "head_dim mismatch");
    TORCH_CHECK(v_seq.size(1) == store_v.size(2), "head_dim mismatch");
    TORCH_CHECK(k_seq.device() == store_k.device(), "device mismatch");
    TORCH_CHECK(v_seq.device() == store_v.device(), "device mismatch");
    TORCH_CHECK(pos_seq.device() == store_pos.device(), "device mismatch");
    TORCH_CHECK(frame_desc.device() == store_k.device(), "descriptor device mismatch");

    int N = static_cast<int>(frame_desc.size(0));
    if (N == 0) return;
    launch_anchor_store_write_frames(
        k_seq,
        v_seq,
        pos_seq,
        frame_desc,
        store_k,
        store_v,
        store_pos,
        store_k.size(1),
        store_k.size(2),
        N
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("scatter_copy", &scatter_copy,
          "Scatter-copy multiple source tensors into workspace (CUDA)");
    m.def("apply_pos_override", &apply_pos_override,
          "Override pos[:, 0] for specified ranges (CUDA)");
    m.def("refresh_readout_layout", &refresh_readout_layout,
          "Refresh mutable PyramidKV readout layout segments (CUDA)");
    m.def("anchor_store_write_frames", &anchor_store_write_frames,
          "Write frame slices into contiguous PyramidKV anchor-store slots (CUDA)");
}

// ---------------------------------------------------------------------------
// TORCH_LIBRARY registration
// ---------------------------------------------------------------------------
// Exposes the same kernels as torch.ops.adahead.*. This is the path that will
// be used by all future C++ takeover work (M1.3+) and by CUDA Graph capture
// (M2). Keeping PYBIND11_MODULE above unchanged for backward compatibility
// during the migration; M1.6 removes the legacy path.
//
// Schema notes:
//   - Tensor(a!), Tensor(b!), ... mark mutating output buffers. Missing this
//     annotation lets functionalization passes silently elide the op.
//   - indptr-style int tensors must remain int32/int64 as written below;
//     FlashInfer-style ops we add later will use int32.
//   - Meta kernels are no-ops (the ops return void) but are required for
//     torch.compile / torch.export / functorch.vmap not to crash.

namespace adahead_meta {

void scatter_copy(
    torch::Tensor /*src_ptrs*/,
    torch::Tensor /*lengths*/,
    torch::Tensor /*offsets*/,
    torch::Tensor /*dst*/,
    int64_t /*col_dim*/
) {}

void apply_pos_override(
    torch::Tensor /*pos*/,
    torch::Tensor /*starts*/,
    torch::Tensor /*ends*/,
    torch::Tensor /*vals*/
) {}

void anchor_store_write_frames(
    torch::Tensor /*k_seq*/,
    torch::Tensor /*v_seq*/,
    torch::Tensor /*pos_seq*/,
    torch::Tensor /*frame_desc*/,
    torch::Tensor /*store_k*/,
    torch::Tensor /*store_v*/,
    torch::Tensor /*store_pos*/
) {}

void refresh_readout_layout(
    std::vector<torch::Tensor> /*src_k*/,
    std::vector<torch::Tensor> /*src_v*/,
    std::vector<torch::Tensor> /*src_pos*/,
    torch::Tensor /*src_ptrs_k*/,
    torch::Tensor /*src_ptrs_v*/,
    torch::Tensor /*src_ptrs_pos*/,
    torch::Tensor /*offsets*/,
    torch::Tensor /*lengths*/,
    torch::Tensor /*flags*/,
    torch::Tensor /*dynamic_rope_t*/,
    torch::Tensor /*dst_k_raw*/,
    torch::Tensor /*dst_v*/,
    torch::Tensor /*dst_rope_pos*/,
    torch::Tensor /*dst_frame_ids*/,
    int64_t /*head_dim*/,
    int64_t /*current_t*/,
    std::string /*history_time_mapping_mode*/,
    int64_t /*history_relative_t_max*/,
    double /*history_time_soft_factor*/
) {}

}  // namespace adahead_meta

TORCH_LIBRARY(adahead, m) {
    // PyramidKVCacheManager custom class.
    // Plan B — constructor takes 10 args now (added max_attend_chunks at end).
    m.class_<adahead::PyramidKVCacheManager>("PyramidKVCacheManager")
        .def(torch::init<
            int64_t, int64_t, int64_t, int64_t,
            int64_t, int64_t, int64_t,
            const std::string&, const std::string&,
            int64_t
        >())
        .def("reset", &adahead::PyramidKVCacheManager::reset)
        .def("set_strategy", &adahead::PyramidKVCacheManager::set_strategy)
        .def("sink_pool", &adahead::PyramidKVCacheManager::sink_pool)
        .def("middle_pool", &adahead::PyramidKVCacheManager::middle_pool)
        .def("recent_pool", &adahead::PyramidKVCacheManager::recent_pool)
        .def("sink_k_pool", &adahead::PyramidKVCacheManager::sink_k_pool)
        .def("sink_v_pool", &adahead::PyramidKVCacheManager::sink_v_pool)
        .def("middle_k_pool", &adahead::PyramidKVCacheManager::middle_k_pool)
        .def("middle_v_pool", &adahead::PyramidKVCacheManager::middle_v_pool)
        .def("recent_k_pool", &adahead::PyramidKVCacheManager::recent_k_pool)
        .def("recent_v_pool", &adahead::PyramidKVCacheManager::recent_v_pool)
        .def("sink_pos_pool", &adahead::PyramidKVCacheManager::sink_pos_pool)
        .def("middle_pos_pool", &adahead::PyramidKVCacheManager::middle_pos_pool)
        .def("recent_pos_pool", &adahead::PyramidKVCacheManager::recent_pos_pool)
        .def("head_t_table", &adahead::PyramidKVCacheManager::head_t_table)
        .def("write_idx", &adahead::PyramidKVCacheManager::write_idx)
        .def("valid_count", &adahead::PyramidKVCacheManager::valid_count)
        .def("strategy", &adahead::PyramidKVCacheManager::strategy)
        .def("k_flat_out", &adahead::PyramidKVCacheManager::k_flat_out)
        .def("v_flat_out", &adahead::PyramidKVCacheManager::v_flat_out)
        .def("cu_seqlens_k", &adahead::PyramidKVCacheManager::cu_seqlens_k)
        .def("pos_flat_out", &adahead::PyramidKVCacheManager::pos_flat_out)
        // merge pools
        .def("merge_k_pool", &adahead::PyramidKVCacheManager::merge_k_pool)
        .def("merge_v_pool", &adahead::PyramidKVCacheManager::merge_v_pool)
        .def("merge_pos_pool", &adahead::PyramidKVCacheManager::merge_pos_pool)
        .def("merge_token_count", &adahead::PyramidKVCacheManager::merge_token_count)
        .def("merge_accum_sum_k", &adahead::PyramidKVCacheManager::merge_accum_sum_k)
        .def("merge_accum_sum_v", &adahead::PyramidKVCacheManager::merge_accum_sum_v)
        .def("merge_accum_pos", &adahead::PyramidKVCacheManager::merge_accum_pos)
        .def("merge_accum_group_ids", &adahead::PyramidKVCacheManager::merge_accum_group_ids)
        .def("merge_accum_tokens_per_group",
             &adahead::PyramidKVCacheManager::merge_accum_tokens_per_group)
        .def("merge_accum_num_groups",
             &adahead::PyramidKVCacheManager::merge_accum_num_groups)
        .def("max_merge_blocks", &adahead::PyramidKVCacheManager::max_merge_blocks)
        .def("max_merge_active", &adahead::PyramidKVCacheManager::max_merge_active)
        .def("max_merge_tokens_per_anchor",
             &adahead::PyramidKVCacheManager::max_merge_tokens_per_anchor)
        .def("num_layers", &adahead::PyramidKVCacheManager::num_layers)
        .def("num_heads", &adahead::PyramidKVCacheManager::num_heads)
        .def("head_dim", &adahead::PyramidKVCacheManager::head_dim)
        .def("frame_seqlen", &adahead::PyramidKVCacheManager::frame_seqlen)
        .def("max_sink", &adahead::PyramidKVCacheManager::max_sink)
        .def("max_middle", &adahead::PyramidKVCacheManager::max_middle)
        .def("max_recent", &adahead::PyramidKVCacheManager::max_recent)
        .def("max_total", &adahead::PyramidKVCacheManager::max_total)
        .def("max_total_tokens", &adahead::PyramidKVCacheManager::max_total_tokens)
        // Plan B — new query: per-attend-call pack workspace capacity.
        .def("max_attend_chunks", &adahead::PyramidKVCacheManager::max_attend_chunks)
        .def("max_pack_tokens", &adahead::PyramidKVCacheManager::max_pack_tokens);

    m.def(
        "scatter_copy(Tensor src_ptrs, Tensor lengths, Tensor offsets, "
        "Tensor(a!) dst, int col_dim) -> ()"
    );
    m.def(
        "apply_pos_override(Tensor(a!) pos, Tensor starts, Tensor ends, "
        "Tensor vals) -> ()"
    );
    m.def(
        "anchor_store_write_frames(Tensor k_seq, Tensor v_seq, Tensor pos_seq, "
        "Tensor frame_desc, Tensor(a!) store_k, Tensor(b!) store_v, "
        "Tensor(c!) store_pos) -> ()"
    );
    m.def(
        "refresh_readout_layout(Tensor[] src_k, Tensor[] src_v, "
        "Tensor[] src_pos, "
        "Tensor(a!) src_ptrs_k, Tensor(b!) src_ptrs_v, Tensor(c!) src_ptrs_pos, "
        "Tensor offsets, Tensor lengths, Tensor flags, Tensor dynamic_rope_t, "
        "Tensor(d!) dst_k_raw, Tensor(e!) dst_v, Tensor(f!) dst_rope_pos, "
        "Tensor(g!) dst_frame_ids, int head_dim, int current_t, "
        "str history_time_mapping_mode, int history_relative_t_max, "
        "float history_time_soft_factor) -> ()"
    );

    // pack op — gathers sink/middle/recent slots into flat output.
    // Custom class args are passed by reference; mutability lives implicit
    // in the manager's internal tensors. No (a!) annotation on the class arg.
    //
    // fused RoPE: pack writes RoPE-rotated K directly when
    // anchor_t_remap + freqs_*_flat are provided. Pass empty tensors for
    // any of these args to keep the pure-memcpy behavior (back-compat for
    // mega_attention_step / unit tests).
    m.def(
        "pyramidkv_pack(__torch__.torch.classes.adahead.PyramidKVCacheManager mgr, "
        "Tensor src_kind, Tensor src_slot_global, "
        "Tensor seg_lengths, Tensor dst_token_offsets, "
        "Tensor anchor_t_remap, "
        "Tensor freqs_ft_flat, Tensor freqs_fy_flat, Tensor freqs_fx_flat) -> ()"
    );

    // plan op: returns (cu_seqlens_k, src_kind, src_slot_global,
    // seg_lengths, dst_token_offsets) device int32 tensors built on CPU and
    // H2D-copied non-blocking. Runs OUTSIDE any CUDA Graph capture.
    m.def(
        "pyramidkv_plan(__torch__.torch.classes.adahead.PyramidKVCacheManager mgr, "
        "int current_t, int pass_kind) "
        "-> (Tensor, Tensor, Tensor, Tensor, Tensor)"
    );

    // update op: writes incoming K/V frames into pool slots.
    // new_pos [F, H, FSEQ, 3] int64 threads per-token pos through
    // alongside K/V — pass empty tensor to skip the pos write.
    m.def(
        "pyramidkv_update(__torch__.torch.classes.adahead.PyramidKVCacheManager mgr, "
        "Tensor new_k, Tensor new_v, Tensor new_pos, "
        "Tensor dst_kind, Tensor dst_slot_global, "
        "Tensor src_frame_in_block, Tensor src_head) -> ()"
    );

    // mega_state_update: mutates per-head strategy state and
    // emits 8 descriptor outputs consumed by the existing data-movement
    // kernel and the upcoming merge_accum scatter-add kernel.
    // states: raw uint8 byte buffer reinterpret_cast'd to PerHeadState*.
    m.def(
        "mega_state_update(Tensor(a!) states, Tensor new_t_vals, "
        "Tensor(b!) desc_dst_kind, Tensor(c!) desc_dst_slot, "
        "Tensor(d!) desc_src_frame, Tensor(e!) desc_src_head, "
        "Tensor(f!) desc_merge_accum_slot, Tensor(g!) desc_merge_local_idx, "
        "Tensor(h!) desc_merge_finalize_completed_idx, "
        "Tensor(i!) desc_merge_is_new_block, "
        "int num_heads, int frames_in_block, int pass_kind) -> ()"
    );

    // per-layer mega_plan with dynamic RoPE remap.
    // Returns 7 tensors: cu_seqlens_k, src_kind, src_slot_global, seg_lengths,
    // dst_token_offsets, anchor_t_raw, anchor_t_remap.
    // sink_mode_code:  0=lag (default), 1=window_clamp
    // hist_mode_code:  0=none (identity), 1=relative_clamp, 2=relative_softcap
    m.def(
        "mega_plan(__torch__.torch.classes.adahead.PyramidKVCacheManager mgr, "
        "Tensor states_bytes, int layer_idx, int current_t, int pass_kind, "
        "int sink_mode_code, int sink_clamp_min, int sink_clamp_max, "
        "int decoupled_sink_lag, "
        "int hist_mode_code, int hist_relative_t_max, "
        "float hist_soft_factor) "
        "-> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)"
    );

    // multi-chunk plan: emits per-chunk descriptors back-to-back
    // with chunk-offset dst_token_offsets. current_t_list is a CPU
    // int64 [num_chunks] tensor; output shape scales with num_chunks.
    m.def(
        "mega_plan_multi(__torch__.torch.classes.adahead.PyramidKVCacheManager mgr, "
        "Tensor states_bytes, int layer_idx, Tensor current_t_list, int pass_kind, "
        "int sink_mode_code, int sink_clamp_min, int sink_clamp_max, "
        "int decoupled_sink_lag, "
        "int hist_mode_code, int hist_relative_t_max, "
        "float hist_soft_factor) "
        "-> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)"
    );

    m.def("mega_state_perhead_size() -> int");

    // merge K/V scatter-add + finalize. One layer per call.
    // raw-K storage raw-K storage: incoming K is raw (no RoPE applied), so the
    // kernel just sum-divides per group. Readout applies fresh 3D RoPE at
    // the merge anchor's (median_t, py, px). Phase A.1/A.2 inverse-rotate
    // params (ft/fy/fx_freqs) removed.
    m.def(
        "mega_merge_accum(Tensor states, "
        "Tensor new_k, Tensor new_v, Tensor new_pos, "
        "Tensor desc_dst_kind, Tensor desc_merge_accum_slot, "
        "Tensor desc_merge_finalize_completed_idx, "
        "Tensor desc_merge_is_new_block, "
        "Tensor(a!) accum_sum_k, Tensor(b!) accum_sum_v, "
        "Tensor(c!) accum_pos, Tensor(d!) accum_group_ids, "
        "Tensor(e!) accum_tokens_per_group, Tensor(f!) accum_num_groups, "
        "Tensor(g!) merge_k_pool, Tensor(h!) merge_v_pool, "
        "Tensor(i!) merge_pos_pool, Tensor(j!) merge_token_count) -> ()"
    );
}

TORCH_LIBRARY_IMPL(adahead, CUDA, m) {
    m.impl("scatter_copy", scatter_copy);
    m.impl("apply_pos_override", apply_pos_override);
    m.impl("anchor_store_write_frames", anchor_store_write_frames);
    m.impl("refresh_readout_layout", refresh_readout_layout);
    m.impl("pyramidkv_pack", pyramidkv_pack);
    m.impl("pyramidkv_update", pyramidkv_update);
    m.impl("mega_state_update", mega_state_update);
    m.impl("mega_merge_accum", mega_merge_accum);
}

// pyramidkv_plan has no Tensor args (only a custom class + ints), so the
// dispatcher cannot pick a backend from arg types. Register under
// CompositeExplicitAutograd so the same impl runs regardless of which
// backend the caller's tensors live on.
// mega_state_perhead_size is a pure constant query — same treatment.
TORCH_LIBRARY_IMPL(adahead, CompositeExplicitAutograd, m) {
    m.impl("pyramidkv_plan", adahead::_pyramidkv_plan_cuda);
    m.impl("mega_state_perhead_size", mega_state_perhead_size);
    m.impl("mega_plan", adahead::_mega_plan_cuda);
    m.impl("mega_plan_multi", adahead::_mega_plan_multi_cuda);
}

TORCH_LIBRARY_IMPL(adahead, Meta, m) {
    m.impl("scatter_copy", adahead_meta::scatter_copy);
    m.impl("apply_pos_override", adahead_meta::apply_pos_override);
    m.impl("anchor_store_write_frames", adahead_meta::anchor_store_write_frames);
    m.impl("refresh_readout_layout", adahead_meta::refresh_readout_layout);
    m.impl("pyramidkv_pack", pyramidkv_pack_meta);
    m.impl("pyramidkv_plan", adahead::_pyramidkv_plan_meta);
    m.impl("pyramidkv_update", pyramidkv_update_meta);
    m.impl("mega_state_update", mega_state_update_meta);
    m.impl("mega_merge_accum", mega_merge_accum_meta);
    m.impl("mega_plan", adahead::_mega_plan_meta);
    m.impl("mega_plan_multi", adahead::_mega_plan_multi_meta);
}

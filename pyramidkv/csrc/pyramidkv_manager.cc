// PyramidKVCacheManager implementation.
//
// Responsibilities are limited to allocation + ring-state lifecycle.
// All actual data movement (pack / update / strategy dispatch) lives in
// later milestones (M1.4 / M1.5) which take a manager reference.
//
// extends the constructor to allocate merge accumulator pools
// alongside the existing sink/middle/recent pools. Sizes derive from the
// compile-time bounds in anchor_store.cuh (kMaxMergeBlocks/Active/Tokens).
#include "pyramidkv_manager.h"

#include "anchor_store.cuh"

#include <ATen/ATen.h>
#include <c10/util/Optional.h>

#include <stdexcept>

namespace adahead {

namespace {

at::ScalarType parse_kv_dtype(const std::string& s) {
    if (s == "bfloat16" || s == "bf16") return at::kBFloat16;
    if (s == "float16" || s == "half" || s == "fp16") return at::kHalf;
    if (s == "float32" || s == "float" || s == "fp32") return at::kFloat;
    TORCH_CHECK(false, "PyramidKVCacheManager: unsupported kv dtype '", s, "'");
}

c10::Device parse_device(const std::string& s) {
    return c10::Device(s);  // "cuda:0" / "cpu" / etc.
}

}  // namespace

PyramidKVCacheManager::PyramidKVCacheManager(
    int64_t num_layers,
    int64_t num_heads,
    int64_t head_dim,
    int64_t frame_seqlen,
    int64_t max_sink_frames,
    int64_t max_middle_frames,
    int64_t max_recent_frames,
    const std::string& device_str,
    const std::string& kv_dtype_str,
    int64_t max_attend_chunks
)
    : num_layers_(num_layers),
      num_heads_(num_heads),
      head_dim_(head_dim),
      frame_seqlen_(frame_seqlen),
      max_sink_(max_sink_frames),
      max_middle_(max_middle_frames),
      max_recent_(max_recent_frames),
      max_attend_chunks_(max_attend_chunks) {

    TORCH_CHECK(num_layers > 0 && num_heads > 0 && head_dim > 0,
                "PyramidKVCacheManager: num_layers/num_heads/head_dim must be > 0");
    TORCH_CHECK(frame_seqlen > 0,
                "PyramidKVCacheManager: frame_seqlen must be > 0");
    TORCH_CHECK(max_sink_frames >= 0 && max_middle_frames >= 0 && max_recent_frames >= 0,
                "PyramidKVCacheManager: max_*_frames must be >= 0");
    TORCH_CHECK(max_attend_chunks > 0,
                "PyramidKVCacheManager: max_attend_chunks must be > 0 "
                "(usually = num_frame_per_block with safety margin)");

    const auto device = parse_device(device_str);
    const auto kv_dtype = parse_kv_dtype(kv_dtype_str);

    auto kv_opts = at::TensorOptions().dtype(kv_dtype).device(device);
    auto i64_opts = at::TensorOptions().dtype(at::kLong).device(device);
    auto i32_opts = at::TensorOptions().dtype(at::kInt).device(device);
    auto i8_opts = at::TensorOptions().dtype(at::kChar).device(device);

    const int64_t L = num_layers_;
    const int64_t H = num_heads_;
    const int64_t D = head_dim_;
    const int64_t F = frame_seqlen_;

    sink_k_pool_   = at::zeros({L, H, max_sink_, F, D}, kv_opts);
    sink_v_pool_   = at::zeros({L, H, max_sink_, F, D}, kv_opts);
    middle_k_pool_ = at::zeros({L, H, max_middle_, F, D}, kv_opts);
    middle_v_pool_ = at::zeros({L, H, max_middle_, F, D}, kv_opts);
    recent_k_pool_ = at::zeros({L, H, max_recent_, F, D}, kv_opts);
    recent_v_pool_ = at::zeros({L, H, max_recent_, F, D}, kv_opts);

    // per-kind pos pools, parallel to K/V. Each slot holds the
    // (t, y, x) tuple per token; readout packs these alongside K and feeds
    // them into a single fused Triton 3D RoPE.
    sink_pos_pool_   = at::zeros({L, H, max_sink_, F, 3}, i64_opts);
    middle_pos_pool_ = at::zeros({L, H, max_middle_, F, 3}, i64_opts);
    recent_pos_pool_ = at::zeros({L, H, max_recent_, F, 3}, i64_opts);

    const int64_t MT = max_total();
    // Plan B — pos_3d_pool_ (legacy pre-Option-C unified pos pool) dropped.
    // raw-K storage added per-kind sink/middle/recent_pos_pool_ that replaced it;
    // grep confirms no remaining readers. Saves L*H*MT*F*3*8 bytes
    // (~418 MB at L=30/H=12/MT=31/F=1560).
    head_t_table_ = at::full({L, H, MT}, -1, i64_opts);  // -1 = empty slot
    write_idx_    = at::zeros({L, H, 3}, i64_opts);
    valid_count_  = at::zeros({L, H, 3}, i64_opts);
    strategy_     = at::zeros({L, H, 8}, i8_opts);

    // Plan B — flat workspaces sized for ONE attend call's worst-case
    // pack output across all chunks, not the all-layer MTT. Old MTT was
    // L*H*MT*F ≈ 17M tokens at L=30/H=12/MT=31/F=1560 → ~9 GB across the
    // three flats. New cap is max_attend_chunks*H*(MT+MMB)*F which for
    // max_attend_chunks=8 sits at ~5.5M tokens → ~2.8 GB, saving ~6 GB.
    const int64_t pack_tokens = max_pack_tokens();
    k_flat_out_   = at::zeros({pack_tokens, 1, D}, kv_opts);
    v_flat_out_   = at::zeros({pack_tokens, 1, D}, kv_opts);
    cu_seqlens_k_ = at::zeros({max_attend_chunks_ * H + 1}, i32_opts);
    pos_flat_out_ = at::zeros({pack_tokens, 3}, i64_opts);

    // ---- M1 Day 5b — merge pools ----------------------------------------
    // Use compile-time bounds from anchor_store.cuh. Pyramid-forcing G6
    // doesn't actually use merge so these stay zero; allocations matter
    // for configs that do (~150 MB total at L=30 H=12).
    auto fp32_opts = at::TensorOptions().dtype(at::kFloat).device(device);

    const int64_t MMB  = static_cast<int64_t>(kMaxMergeBlocks);
    const int64_t MMA  = static_cast<int64_t>(kMaxMergeActive);
    const int64_t MTPA = static_cast<int64_t>(kMaxMergeTokens);

    merge_k_pool_       = at::zeros({L, H, MMB, MTPA, D}, kv_opts);
    merge_v_pool_       = at::zeros({L, H, MMB, MTPA, D}, kv_opts);
    merge_pos_pool_     = at::zeros({L, H, MMB, MTPA, 3}, i64_opts);
    merge_token_count_  = at::zeros({L, H, MMB}, i32_opts);

    merge_accum_sum_k_  = at::zeros({L, H, MMA, MTPA, D}, fp32_opts);
    merge_accum_sum_v_  = at::zeros({L, H, MMA, MTPA, D}, fp32_opts);
    merge_accum_pos_    = at::zeros({L, H, MMA, MTPA, 3}, i64_opts);
    merge_accum_group_ids_ =
        at::zeros({L, H, MMA, frame_seqlen_}, i32_opts);
    merge_accum_tokens_per_group_ =
        at::zeros({L, H, MMA, MTPA}, fp32_opts);
    merge_accum_num_groups_ = at::zeros({L, H, MMA}, i32_opts);
}

int64_t PyramidKVCacheManager::max_pack_tokens() const {
    // Worst-case packed tokens across one attend call's chunks. Each chunk
    // emits up to H × (max_total + max_merge_blocks) segments × F tokens,
    // and chunks are concatenated by mega_plan_multi into the same flats.
    return max_attend_chunks_
         * num_heads_
         * (max_total() + max_merge_blocks())
         * frame_seqlen_;
}

int64_t PyramidKVCacheManager::max_merge_blocks() const {
    return static_cast<int64_t>(kMaxMergeBlocks);
}
int64_t PyramidKVCacheManager::max_merge_active() const {
    return static_cast<int64_t>(kMaxMergeActive);
}
int64_t PyramidKVCacheManager::max_merge_tokens_per_anchor() const {
    return static_cast<int64_t>(kMaxMergeTokens);
}

void PyramidKVCacheManager::reset() {
    write_idx_.zero_();
    valid_count_.zero_();
    head_t_table_.fill_(-1);
    cu_seqlens_k_.zero_();

    // clear per-kind pos pools. Stale pos values stranded in
    // unused slots could leak into RoPE if the FIFO/state machine ever
    // emits them by mistake. K/V pools intentionally aren't zeroed here
    // because their valid range is gated by valid_count.
    sink_pos_pool_.zero_();
    middle_pos_pool_.zero_();
    recent_pos_pool_.zero_();

    // Merge: clear the two metadata tensors that gate "is anchor populated".
    // Data pools (merge_*_pool, accum buffers) get overwritten as new blocks
    // arrive — leaving them dirty is fine until the strategy says "use slot N".
    merge_token_count_.zero_();
    merge_accum_num_groups_.zero_();
}

void PyramidKVCacheManager::set_strategy(
    int64_t layer_idx,
    int64_t head_idx,
    int64_t kind,
    torch::Tensor params
) {
    TORCH_CHECK(layer_idx >= 0 && layer_idx < num_layers_,
                "PyramidKVCacheManager::set_strategy layer_idx out of range");
    TORCH_CHECK(head_idx >= 0 && head_idx < num_heads_,
                "PyramidKVCacheManager::set_strategy head_idx out of range");
    TORCH_CHECK(params.dtype() == at::kChar,
                "PyramidKVCacheManager::set_strategy params must be int8");
    TORCH_CHECK(params.dim() == 1, "params must be 1D");
    TORCH_CHECK(params.size(0) <= 7, "params longer than 7 (slot[0] is kind)");
    TORCH_CHECK(kind >= -128 && kind <= 127, "kind out of int8 range");

    // strategy_[L, H, 8] → slot[0] is kind, slot[1:1+len(params)] is params.
    auto cpu_params = params.to(at::kCPU).contiguous();
    auto target_dev = strategy_.device();
    const int64_t param_len = cpu_params.size(0);

    // Build a small int8[8] CPU buffer then copy once.
    auto buf = at::zeros({8}, at::TensorOptions().dtype(at::kChar));
    auto* buf_ptr = buf.data_ptr<int8_t>();
    buf_ptr[0] = static_cast<int8_t>(kind);
    if (param_len > 0) {
        std::memcpy(buf_ptr + 1, cpu_params.data_ptr<int8_t>(),
                    static_cast<size_t>(param_len));
    }

    // Copy slot to device.
    auto buf_dev = buf.to(target_dev, /*non_blocking=*/true);
    strategy_.index_put_({layer_idx, head_idx}, buf_dev);
}

}  // namespace adahead

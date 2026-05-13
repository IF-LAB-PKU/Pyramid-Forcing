// PyramidKVCacheManager: cross-layer buffer pool for PyramidKV cache state.
//
// Owns all long-lived device tensors used by the C++ pack/update kernels.
// Exposed to Python via TORCH_LIBRARY as
// torch.classes.adahead.PyramidKVCacheManager.
//
// Layout (single global instance, all 30 layers share the same allocator):
//   sink_pool   : [L, H, max_sink, frame_seqlen, D]   bf16
//   middle_pool : [L, H, max_middle, frame_seqlen, D] bf16
//   recent_pool : [L, H, max_recent, frame_seqlen, D] bf16
//   head_t_tbl  : [L, H, max_total]                   int64  (per-slot timestamp)
//   write_idx   : [L, H, 3]                           int64  (ring head per kind)
//   valid_count : [L, H, 3]                           int64  (used slots per kind)
//   strategy    : [L, H, 8]                           int8   (kind + params)
//
// Raw-K storage refactor — per-kind pos pools mirror K/V layout:
//   sink_pos_pool   : [L, H, max_sink, frame_seqlen, 3]   int64
//   middle_pos_pool : [L, H, max_middle, frame_seqlen, 3] int64
//   recent_pos_pool : [L, H, max_recent, frame_seqlen, 3] int64
//   (merge_pos_pool already exists below for the per-anchor merged pos.)
//
// Merge accumulator + finalized merge anchor pools:
//   merge_k_pool        : [L, H, MMB, MTPA, D]              bf16
//   merge_v_pool        : same                              bf16
//   merge_pos_pool      : [L, H, MMB, MTPA, 3]              int64
//   merge_token_count   : [L, H, MMB]                       int32
//                         (num_groups per finalized anchor; usually < MTPA)
//   merge_accum_sum_k   : [L, H, MMA, MTPA, D]              fp32
//                         (fp32 for stability — bf16 accumulator divergence)
//   merge_accum_sum_v   : same                              fp32
//   merge_accum_pos     : [L, H, MMA, MTPA, 3]              int64
//   merge_accum_group_ids        : [L, H, MMA, frame_seqlen]    int32
//                         (token → group mapping captured at first frame)
//   merge_accum_tokens_per_group : [L, H, MMA, MTPA]            fp32
//                         (count_per_group * block_frames; finalize divisor)
//   merge_accum_num_groups       : [L, H, MMA]                  int32
//   where MMB = kMaxMergeBlocks (6), MMA = kMaxMergeActive (2),
//         MTPA = kMaxMergeTokens (420).
//
// Output workspace (reused per step, preallocated):
//   k_flat_out  : [max_pack_tokens, 1, D]             bf16
//   v_flat_out  : [max_pack_tokens, 1, D]             bf16
//   cu_seqlens_k: [max_attend_chunks*H + 1]           int32  (FlashInfer convention)
//   pos_flat_out: [max_pack_tokens, 3]                int64
//
// max_total = max_sink + max_middle + max_recent.
// max_pack_tokens = max_attend_chunks * H * (max_total + max_merge_blocks) * F.
// (Plan B: workspace sized for ONE attend call's worst case across all chunks,
// not the all-layer-all-head MTT — saves ~7 GB at L=30, H=12, MT=31.)
//
// All Python-visible accessors return shallow views; the manager retains
// ownership and the buffers persist across forward passes.
#pragma once

#include <torch/extension.h>

#include <cstdint>
#include <string>

namespace adahead {

class PyramidKVCacheManager : public torch::CustomClassHolder {
public:
    PyramidKVCacheManager(
        int64_t num_layers,
        int64_t num_heads,
        int64_t head_dim,
        int64_t frame_seqlen,
        int64_t max_sink_frames,
        int64_t max_middle_frames,
        int64_t max_recent_frames,
        const std::string& device_str,
        const std::string& kv_dtype_str,
        // Plan B — upper bound on per-attend-call query chunks
        // (= num_frame_per_block with a safety margin). Sizes the flat
        // pack workspaces k_flat_out / v_flat_out / pos_flat_out, which
        // hold ONE layer's packed output across all chunks, NOT all
        // layers' tokens. Default 8 is plenty for num_frame_per_block ≤ 4.
        int64_t max_attend_chunks
    );

    // Wipe ring-buffer state (write_idx, valid_count, head_t_table). Pools
    // themselves are reused; update kernels overwrite fresh data on the next call.
    void reset();

    // Configure per-(layer, head) strategy. ``kind`` follows the encoding
    // documented in pyramidkv/cpp_strategy.py KIND_*. ``params`` is a small
    // int8 tensor of length up to 8 (period/interval/etc.).
    void set_strategy(
        int64_t layer_idx,
        int64_t head_idx,
        int64_t kind,
        torch::Tensor params  // CPU int8 [<=8], copied into strategy_ slot.
    );

    // ---- Pool accessors (views, not copies) -----------------------------
    // K and V are kept in parallel pools because flash_attn_varlen wants
    // separate K/V tensors and the pack kernel reads them together.
    torch::Tensor sink_pool() const { return sink_k_pool_; }      // alias for tests / back-compat
    torch::Tensor middle_pool() const { return middle_k_pool_; }
    torch::Tensor recent_pool() const { return recent_k_pool_; }
    torch::Tensor sink_k_pool() const { return sink_k_pool_; }
    torch::Tensor sink_v_pool() const { return sink_v_pool_; }
    torch::Tensor middle_k_pool() const { return middle_k_pool_; }
    torch::Tensor middle_v_pool() const { return middle_v_pool_; }
    torch::Tensor recent_k_pool() const { return recent_k_pool_; }
    torch::Tensor recent_v_pool() const { return recent_v_pool_; }
    // Raw-K storage — per-kind pos pools (parallel to K/V).
    torch::Tensor sink_pos_pool() const { return sink_pos_pool_; }
    torch::Tensor middle_pos_pool() const { return middle_pos_pool_; }
    torch::Tensor recent_pos_pool() const { return recent_pos_pool_; }
    torch::Tensor head_t_table() const { return head_t_table_; }
    torch::Tensor write_idx() const { return write_idx_; }
    torch::Tensor valid_count() const { return valid_count_; }
    torch::Tensor strategy() const { return strategy_; }

    // ---- Output workspace accessors -------------------------------------
    torch::Tensor k_flat_out() const { return k_flat_out_; }
    torch::Tensor v_flat_out() const { return v_flat_out_; }
    torch::Tensor cu_seqlens_k() const { return cu_seqlens_k_; }
    torch::Tensor pos_flat_out() const { return pos_flat_out_; }

    // ---- M1 Day 5b — merge pool accessors -------------------------------
    torch::Tensor merge_k_pool() const { return merge_k_pool_; }
    torch::Tensor merge_v_pool() const { return merge_v_pool_; }
    torch::Tensor merge_pos_pool() const { return merge_pos_pool_; }
    torch::Tensor merge_token_count() const { return merge_token_count_; }
    torch::Tensor merge_accum_sum_k() const { return merge_accum_sum_k_; }
    torch::Tensor merge_accum_sum_v() const { return merge_accum_sum_v_; }
    torch::Tensor merge_accum_pos() const { return merge_accum_pos_; }
    torch::Tensor merge_accum_group_ids() const { return merge_accum_group_ids_; }
    torch::Tensor merge_accum_tokens_per_group() const {
        return merge_accum_tokens_per_group_;
    }
    torch::Tensor merge_accum_num_groups() const { return merge_accum_num_groups_; }

    // ---- Dimension queries ----------------------------------------------
    int64_t num_layers() const { return num_layers_; }
    int64_t num_heads() const { return num_heads_; }
    int64_t head_dim() const { return head_dim_; }
    int64_t frame_seqlen() const { return frame_seqlen_; }
    int64_t max_sink() const { return max_sink_; }
    int64_t max_middle() const { return max_middle_; }
    int64_t max_recent() const { return max_recent_; }
    int64_t max_total() const { return max_sink_ + max_middle_ + max_recent_; }
    int64_t max_attend_chunks() const { return max_attend_chunks_; }
    // Plan B — flat workspaces are sized for one attend call's worst-case
    // pack output, not the all-layer-all-head MTT. Worst case per call:
    //   max_attend_chunks × H × (max_total + max_merge_blocks) × F
    int64_t max_pack_tokens() const;
    // Back-compat alias. Old callers used this as the buffer capacity;
    // since the buffers now sit at max_pack_tokens(), forward to it so
    // bounds checks keep working without name churn elsewhere.
    int64_t max_total_tokens() const { return max_pack_tokens(); }
    int64_t max_merge_blocks() const;        // = kMaxMergeBlocks
    int64_t max_merge_active() const;        // = kMaxMergeActive
    int64_t max_merge_tokens_per_anchor() const;  // = kMaxMergeTokens

private:
    int64_t num_layers_;
    int64_t num_heads_;
    int64_t head_dim_;
    int64_t frame_seqlen_;
    int64_t max_sink_;
    int64_t max_middle_;
    int64_t max_recent_;
    int64_t max_attend_chunks_;

    torch::Tensor sink_k_pool_;
    torch::Tensor sink_v_pool_;
    torch::Tensor middle_k_pool_;
    torch::Tensor middle_v_pool_;
    torch::Tensor recent_k_pool_;
    torch::Tensor recent_v_pool_;
    torch::Tensor sink_pos_pool_;    // Raw-K storage: per-kind pos pools
    torch::Tensor middle_pos_pool_;
    torch::Tensor recent_pos_pool_;
    torch::Tensor head_t_table_;
    torch::Tensor write_idx_;
    torch::Tensor valid_count_;
    torch::Tensor strategy_;

    torch::Tensor k_flat_out_;
    torch::Tensor v_flat_out_;
    torch::Tensor cu_seqlens_k_;
    torch::Tensor pos_flat_out_;

    // M1 Day 5b — merge accumulator + finalized merge anchor pools.
    torch::Tensor merge_k_pool_;
    torch::Tensor merge_v_pool_;
    torch::Tensor merge_pos_pool_;
    torch::Tensor merge_token_count_;
    torch::Tensor merge_accum_sum_k_;
    torch::Tensor merge_accum_sum_v_;
    torch::Tensor merge_accum_pos_;
    torch::Tensor merge_accum_group_ids_;
    torch::Tensor merge_accum_tokens_per_group_;
    torch::Tensor merge_accum_num_groups_;
};

}  // namespace adahead

// MegaPlanInfo: layout metadata for one block forward pass.
//
// Successor to PyramidKVPlanInfo (M1.4) with one additional output:
//   anchor_t_remap : per-frame remapped t for dynamic RoPE.
//
// The existing pack kernel (pyramidkv_pack) only uses src_kind / src_slot_global
// / seg_lengths / dst_token_offsets / cu_seqlens_k. Day 1 emits these
// identically to pyramidkv_plan so M3 can drop in mega_plan as a strict
// extension (no regression in pack behavior). The new anchor_t_remap is
// consumed by the upcoming K-RoPE kernel that will rotate pool K data using
// the remapped positions before flash_attn_varlen sees it.
//
// Layout (all device int32 unless noted; N = num_layers * num_heads):
//   cu_seqlens_k     : [N + 1]                      (FlashInfer convention)
//   src_kind         : [N * max_seg]               (-1 = inactive)
//                       0=sink, 1=middle, 2=recent
//   src_slot_global  : [N * max_seg]
//   seg_lengths      : [N * max_seg]               in TOKENS = frames * F
//   dst_token_offsets: [N * max_seg]
//   anchor_t_raw     : [N * max_total_frames] int64  raw t per anchor frame
//   anchor_t_remap   : [N * max_total_frames] int64  dynamic-RoPE-remapped t
//                                                    (Day 1 = identity)
//
// `max_total_frames` = max_sink + max_middle + max_recent (manager's
// max_total). Per-head padding entries use t = -1.
#pragma once

#include <torch/extension.h>

#include <cstdint>

namespace adahead {

struct MegaPlanInfo {
    int32_t num_layers;
    int32_t num_heads;
    int32_t head_dim;
    int32_t frame_seqlen;
    int32_t max_seg;          // = 3 (sink + middle + recent)
    int32_t max_total_frames; // = max_sink + max_middle + max_recent
    int32_t total_tokens;     // = cu_seqlens_k[N]

    int32_t current_t;
    int32_t pass_kind;        // 0 = noisy, 1 = clean

    // Device int32 outputs (5 existing).
    torch::Tensor cu_seqlens_k;
    torch::Tensor src_kind;
    torch::Tensor src_slot_global;
    torch::Tensor seg_lengths;
    torch::Tensor dst_token_offsets;

    // Device int64 outputs (NEW in M2).
    torch::Tensor anchor_t_raw;
    torch::Tensor anchor_t_remap;
};

}  // namespace adahead

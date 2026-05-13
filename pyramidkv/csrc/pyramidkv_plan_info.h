// M1.4 — PyramidKVPlanInfo: serializable layout metadata for one block forward.
//
// Inspired by FlashInfer's PrefillPlanInfo (csrc/batch_prefill.cu): the plan
// stage runs on CPU, packs all per-(layer, head) layout decisions into a
// flat int32 buffer, performs ONE H2D non-blocking copy, and the run stage
// (pyramidkv_pack_kernel) reads the resulting device-side buffers without
// further dispatch overhead.
//
// Layout in the device buffer (row-major), with N = num_layers * num_heads:
//
//   [ cu_seqlens_k (N+1, int32) | src_kind  (N*MAX_SEG, int32) ]
//   [ src_slot     (N*MAX_SEG, int32) | seg_lengths (N*MAX_SEG, int32) ]
//   [ dst_offsets  (N, int32)         | rope_t      (N*MAX_TOTAL_FRAMES, int32) ]
//
// All offsets are byte-free (token-row indices); the kernel multiplies by
// row_bytes when computing pointers.
//
// `src_kind` encodes which pool the segment lives in:
//     0 = sink_pool, 1 = middle_pool, 2 = recent_pool
// `src_slot` is the per-pool ring-buffer slot index for that segment.
//
// MAX_SEG / MAX_TOTAL_FRAMES are passed in by the caller; PlanInfo carries
// them as fields so the kernel can index correctly.
#pragma once

#include <torch/extension.h>

#include <cstdint>
#include <vector>

namespace adahead {

// Source pool encoding for plan_src_kind.
enum class SrcPool : int32_t {
    Sink = 0,
    Middle = 1,
    Recent = 2,
};

struct PyramidKVPlanInfo {
    int32_t num_layers;
    int32_t num_heads;
    int32_t head_dim;
    int32_t frame_seqlen;
    int32_t max_seg;          // per-(layer, head) upper bound on segment count
    int32_t max_total_frames; // per-(layer, head) upper bound on frames in flat output
    int32_t total_tokens;     // sum over heads of selected tokens (= cu_seqlens_k[N])

    // Block / pass identification (kept for future RoPE remap & debugging).
    int32_t current_t;
    int32_t pass_kind;        // 0 = noisy, 1 = clean

    // Device-side int32 buffers. All views into a single contiguous storage.
    // The kernel only reads these; the plan stage produces them.
    torch::Tensor cu_seqlens_k;   // [N+1]
    torch::Tensor src_kind;       // [N, max_seg]   (-1 = inactive)
    torch::Tensor src_slot;       // [N, max_seg]
    torch::Tensor seg_lengths;    // [N, max_seg]   (in tokens, multiples of frame_seqlen)
    torch::Tensor dst_offsets;    // [N]            (token offset into k_flat_out)
    torch::Tensor rope_t;         // [N, max_total_frames]  (per-frame remapped time, -1 = unused)

    // Bundle the int scalars into a returnable tensor for TORCH_LIBRARY ops.
    // Layout matches FromMetaVector below.
    std::vector<int64_t> ToMetaVector() const {
        return {
            static_cast<int64_t>(num_layers),
            static_cast<int64_t>(num_heads),
            static_cast<int64_t>(head_dim),
            static_cast<int64_t>(frame_seqlen),
            static_cast<int64_t>(max_seg),
            static_cast<int64_t>(max_total_frames),
            static_cast<int64_t>(total_tokens),
            static_cast<int64_t>(current_t),
            static_cast<int64_t>(pass_kind),
        };
    }

    static PyramidKVPlanInfo FromMetaVector(
        const std::vector<int64_t>& v,
        torch::Tensor cu_seqlens_k_,
        torch::Tensor src_kind_,
        torch::Tensor src_slot_,
        torch::Tensor seg_lengths_,
        torch::Tensor dst_offsets_,
        torch::Tensor rope_t_
    ) {
        TORCH_CHECK(v.size() == 9, "PyramidKVPlanInfo meta vector must have 9 ints");
        PyramidKVPlanInfo info;
        info.num_layers = static_cast<int32_t>(v[0]);
        info.num_heads = static_cast<int32_t>(v[1]);
        info.head_dim = static_cast<int32_t>(v[2]);
        info.frame_seqlen = static_cast<int32_t>(v[3]);
        info.max_seg = static_cast<int32_t>(v[4]);
        info.max_total_frames = static_cast<int32_t>(v[5]);
        info.total_tokens = static_cast<int32_t>(v[6]);
        info.current_t = static_cast<int32_t>(v[7]);
        info.pass_kind = static_cast<int32_t>(v[8]);
        info.cu_seqlens_k = std::move(cu_seqlens_k_);
        info.src_kind = std::move(src_kind_);
        info.src_slot = std::move(src_slot_);
        info.seg_lengths = std::move(seg_lengths_);
        info.dst_offsets = std::move(dst_offsets_);
        info.rope_t = std::move(rope_t_);
        return info;
    }
};

}  // namespace adahead

// mega_plan: per-layer CPU-side plan op.
//
// Successor to pyramidkv_plan (M1.4). Differences:
//   - Per-layer instead of all-layers (cu_seqlens_k is [H+1], not [N+1]).
//   - Emits one segment per OCCUPIED slot (not one per (kind, head)). This
//     handles cyclic's sparse phase-bucket layout correctly — cyclic writes
//     to slot = phase*bucket_cap + cursor which is non-contiguous; the M1.4
//     plan assumed slots 0..valid_count-1 which silently broke under
//     cyclic. mega_plan walks PerHeadState.cyclic_slot[] / tkey_slot[] to
//     find live slots.
//   - Adds anchor_t_raw / anchor_t_remap [N * max_total_frames] int64
//     outputs for the upcoming dynamic-RoPE K rotation kernel. Day 1 emits
//     identity remap; Day 2 ports map_dynamic_pos_time from pyramidkv/rope.py.
//
// Pack-kernel contract: each segment is one frame (seg_length = frame_seqlen
// in tokens). The existing pyramidkv_pack kernel iterates segments in parallel
// and does plain memcpy, so per-frame granularity is a strict generalization
// of the M1.4 per-kind-per-head segments.
//
// Inputs (per-layer):
//   mgr               : PyramidKVCacheManager (for max dims + valid_count[layer])
//   states_bytes      : uint8 [H * sizeof(PerHeadState)] — one layer's
//                        PerHeadState array. Provided by caller; M3 keeps
//                        these alive across forward calls.
//   layer_idx         : which layer slice this plan corresponds to
//   current_t         : current frame index in the absolute-t domain
//   pass_kind         : 0 = noisy (no state mutation, plan still describes
//                        what cache currently contains), 1 = clean
//
// Outputs (all device int32 unless noted; N_seg = H * max_total_frames):
//   cu_seqlens_k      [H+1]      int32
//   src_kind          [N_seg]    int32  -1 / 0=sink / 1=middle / 2=recent
//   src_slot_global   [N_seg]    int32  = h * max_slots[kind] + slot_in_pool
//   seg_lengths       [N_seg]    int32  = frame_seqlen (or 0 for inactive)
//   dst_token_offsets [N_seg]    int32
//   anchor_t_raw      [N_seg]    int64  raw t value of this anchor's frame
//   anchor_t_remap    [N_seg]    int64  dynamic-RoPE-remapped t (Day 1 = raw)
#include "anchor_store.cuh"
#include "pyramidkv_manager.h"
#include "mega_plan_info.h"

#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <cmath>
#include <limits>
#include <tuple>
#include <vector>

namespace adahead {

namespace {

inline int32_t pack_slot_global(int head, int slot_in_kind, int max_slots) {
    return head * max_slots + slot_in_kind;
}

// Mirrors pyramidkv/rope.py:map_sink_time. Sink frames sit in absolute-t domain
// but the model was trained over a sliding [sink_clamp_min, sink_clamp_max]
// window relative to current_t. When sync_t_raw exceeds the window, the
// remap shifts the position back to a representable training time.
inline int64_t map_sink_time(
    int64_t sync_t_raw,
    int sink_mode_code,   // 0 = lag, 1 = window_clamp
    int sink_clamp_min,
    int sink_clamp_max,
    int decoupled_sink_lag
) {
    if (sink_mode_code == 1) {  // window_clamp
        int64_t delta_t = sync_t_raw;
        if (delta_t < sink_clamp_min) delta_t = sink_clamp_min;
        if (delta_t > sink_clamp_max) delta_t = sink_clamp_max;
        int64_t out = sync_t_raw - delta_t;
        return out < 0 ? 0 : out;
    }
    // Default: classic fixed lag (mode "lag" in adaptive_cache).
    int64_t out = sync_t_raw - decoupled_sink_lag;
    return out < 0 ? 0 : out;
}

// Mirrors pyramidkv/rope.py:map_dynamic_pos_time. Applies to non-sink anchors
// (middle and recent). Relative distance to current_t is clamped (or softly
// compressed) into a training window, then converted back to an absolute t.
inline int64_t map_dynamic_pos_time(
    int64_t t,
    int64_t current_t,
    int hist_mode_code,            // 0 = none, 1 = relative_clamp, 2 = relative_softcap
    int hist_relative_t_max,
    double hist_soft_factor
) {
    if (hist_mode_code == 0 || hist_relative_t_max <= 0) return t;

    int64_t rel = current_t - t;
    if (rel < 0) rel = 0;

    if (hist_mode_code == 2) {  // relative_softcap
        int64_t over = rel - hist_relative_t_max;
        if (over < 0) over = 0;
        int64_t compressed_over =
            static_cast<int64_t>(std::llround(
                static_cast<double>(over) * hist_soft_factor));
        int64_t rel_mapped = (rel <= hist_relative_t_max)
            ? rel
            : (hist_relative_t_max + compressed_over);
        int64_t out = current_t - rel_mapped;
        return out < 0 ? 0 : out;
    }

    // mode 1 = relative_clamp.
    if (rel > hist_relative_t_max) rel = hist_relative_t_max;
    int64_t out = current_t - rel;
    return out < 0 ? 0 : out;
}

// Emit plan descriptors for ONE query chunk into pre-allocated CPU buffers.
// Used by both single-chunk mega_plan and multi-chunk mega_plan_multi.
//
// Output buffers are indexed by:
//   sk/sg/sl/dst/traw/tremap[chunk_seg_base + h * max_total_segments + frame_pos]
//   cu[chunk_cu_base + h + 1]
// where chunk_seg_base = chunk_idx * H * max_total_segments,
//       chunk_cu_base  = chunk_idx * H,
// allowing the multi-chunk wrapper to write num_chunks chunks' descriptors
// into one contiguous output buffer.
//
// running_offset_in: starting dst_token_offset for this chunk. Multi-chunk
// passes this in as the accumulated end of the previous chunk so all chunks'
// K writes target disjoint regions of mgr.k_flat_out.
//
// Returns the accumulated running_offset at the end of this chunk
// (= cu[chunk_cu_base + H]).
inline int32_t emit_chunk_plan(
    int chunk_idx,
    int H,
    int max_sink, int max_middle, int max_recent, int max_merge_blocks_eff,
    int max_total_segments,
    int F,
    int layer_idx,
    int64_t current_t,
    int64_t sync_t,
    const at::TensorAccessor<int64_t, 3>& vc_acc,
    const at::TensorAccessor<int32_t, 3>& mtc_acc,
    const PerHeadState* per_head,
    int32_t running_offset_in,
    int32_t* cu, int32_t* sk, int32_t* sg, int32_t* sl, int32_t* dst,
    int64_t* traw, int64_t* tremap,
    int hist_mode_code, int hist_relative_t_max, double hist_soft_factor
) {
    int32_t running_offset = running_offset_in;
    const int chunk_seg_base = chunk_idx * H * max_total_segments;
    const int chunk_cu_base = chunk_idx * H;

    for (int h = 0; h < H; ++h) {
        const int head_base = chunk_seg_base + h * max_total_segments;
        int frame_pos = 0;

        const PerHeadState& state = per_head[h];
        const int sink_dedup_thresh = static_cast<int>(state.sink_capacity);
        const int recent_min_t = std::numeric_limits<int>::max();

        // ---------- Sink frames ----------
        const int n_sink_raw = static_cast<int>(vc_acc[layer_idx][h][0]);
        const int n_sink = (sink_dedup_thresh > 0 && sink_dedup_thresh < n_sink_raw)
                         ? sink_dedup_thresh : n_sink_raw;
        for (int s = 0; s < n_sink && s < max_sink; ++s) {
            const int slot_idx = head_base + frame_pos;
            sk[slot_idx]  = 0;
            sg[slot_idx]  = pack_slot_global(h, s, max_sink);
            sl[slot_idx]  = F;
            dst[slot_idx] = running_offset;
            const int64_t t_raw = static_cast<int64_t>(s);
            traw[slot_idx]   = t_raw;
            tremap[slot_idx] = sync_t;
            running_offset += F;
            ++frame_pos;
        }

        // ---------- Recent frames ----------
        const int n_recent = static_cast<int>(vc_acc[layer_idx][h][2]);
        for (int r = 0; r < n_recent && r < max_recent; ++r) {
            const int slot_idx = head_base + frame_pos;
            sk[slot_idx]  = 2;
            sg[slot_idx]  = pack_slot_global(h, r, max_recent);
            sl[slot_idx]  = F;
            dst[slot_idx] = running_offset;
            traw[slot_idx]   = -1;
            tremap[slot_idx] = -1;
            running_offset += F;
            ++frame_pos;
        }

        // ---------- Middle frames ----------
        if (state.kind == SK_CYCLIC) {
            const int period = state.period > 0 ? state.period : 1;
            const int bucket_cap = state.bucket_cap > 0 ? state.bucket_cap : 1;
            const int phase_idx = static_cast<int>(
                ((static_cast<int64_t>(current_t) % period) + period) % period);
            const int phase_base = phase_idx * kMaxBucket;
            const int phase_end =
                phase_base + (bucket_cap < kMaxBucket ? bucket_cap : kMaxBucket);
            for (int i = phase_base; i < phase_end; ++i) {
                if (state.cyclic_slot[i] < 0) continue;
                if (state.cyclic_t[i] < sink_dedup_thresh) continue;
                if (state.cyclic_t[i] >= recent_min_t) continue;
                const int slot_idx = head_base + frame_pos;
                sk[slot_idx]  = 1;
                sg[slot_idx]  = pack_slot_global(h, i, max_middle);
                sl[slot_idx]  = F;
                dst[slot_idx] = running_offset;
                const int64_t t_val = static_cast<int64_t>(state.cyclic_t[i]);
                traw[slot_idx]   = t_val;
                tremap[slot_idx] = sync_t;
                running_offset += F;
                ++frame_pos;
                if (frame_pos >= max_total_segments) break;
            }
        } else if (state.kind == SK_STRIDE || state.kind == SK_LAG) {
            const int tc = static_cast<int>(state.tkey_count);
            for (int i = 0; i < tc && i < kMaxTKeyed; ++i) {
                if (state.tkey_t[i] < sink_dedup_thresh) continue;
                if (state.tkey_t[i] >= recent_min_t) continue;
                const int slot_idx = head_base + frame_pos;
                sk[slot_idx]  = 1;
                sg[slot_idx]  = pack_slot_global(h, i, max_middle);
                sl[slot_idx]  = F;
                dst[slot_idx] = running_offset;
                const int64_t t_val = static_cast<int64_t>(state.tkey_t[i]);
                traw[slot_idx]   = t_val;
                tremap[slot_idx] = (state.kind == SK_STRIDE) ? sync_t : -1;
                running_offset += F;
                ++frame_pos;
                if (frame_pos >= max_total_segments) break;
            }
        }
        if (state.kind == SK_MERGE) {
            const int mcc = static_cast<int>(state.merge_completed_count);
            for (int i = 0; i < mcc && i < kMaxMergeBlocks; ++i) {
                const int block_slot = state.merge_completed_slot[i];
                if (block_slot < 0 || block_slot >= max_merge_blocks_eff) continue;
                if (state.merge_completed_start_t[i] <= 0) continue;
                if (state.merge_completed_end_t[i] >= recent_min_t) continue;
                const int32_t tok_count = mtc_acc[layer_idx][h][block_slot];
                if (tok_count <= 0) continue;
                if (frame_pos >= max_total_segments) break;
                const int slot_idx = head_base + frame_pos;
                sk[slot_idx]  = 3;
                sg[slot_idx]  = pack_slot_global(h, block_slot, max_merge_blocks_eff);
                sl[slot_idx]  = tok_count;
                dst[slot_idx] = running_offset;
                const int64_t t_val =
                    static_cast<int64_t>(state.merge_completed_median_t[i]);
                traw[slot_idx]   = 0;
                tremap[slot_idx] = t_val;
                running_offset += tok_count;
                ++frame_pos;
            }
        }
        (void)hist_mode_code; (void)hist_relative_t_max; (void)hist_soft_factor;

        cu[chunk_cu_base + h + 1] = running_offset;
    }
    return running_offset;
}

}  // namespace

// Returns 7 device tensors. Caller materializes them into a MegaPlanInfo.
//
// RoPE-config args control how anchor_t_remap is computed from anchor_t_raw:
//   sink_mode_code  : 0 = "lag" (default), 1 = "window_clamp"
//   sink_clamp_min/max + decoupled_sink_lag : sink-mode params
//   hist_mode_code  : 0 = "none" (identity), 1 = "relative_clamp",
//                     2 = "relative_softcap"
//   hist_relative_t_max / hist_soft_factor : history-mode params
//
// Pyramid forcing 10 uses (sink=lag, lag=0, hist=none) → remap == raw t.
std::tuple<
    torch::Tensor,  // cu_seqlens_k       [H+1]  int32
    torch::Tensor,  // src_kind           [N]    int32
    torch::Tensor,  // src_slot_global    [N]    int32
    torch::Tensor,  // seg_lengths        [N]    int32
    torch::Tensor,  // dst_token_offsets  [N]    int32
    torch::Tensor,  // anchor_t_raw       [N]    int64
    torch::Tensor   // anchor_t_remap     [N]    int64
> mega_plan(
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
) {
    TORCH_CHECK(mgr.get() != nullptr, "mega_plan: manager is null");
    TORCH_CHECK(states_bytes.dtype() == torch::kUInt8,
                "mega_plan: states_bytes must be uint8");
    TORCH_CHECK(states_bytes.numel() >=
                static_cast<int64_t>(mgr->num_heads()) *
                static_cast<int64_t>(sizeof(PerHeadState)),
                "mega_plan: states_bytes too small for ", mgr->num_heads(), " heads");
    TORCH_CHECK(layer_idx >= 0 && layer_idx < mgr->num_layers(),
                "mega_plan: layer_idx out of range");
    (void)pass_kind;  // pass_kind reserved; plan output is the same for noisy/clean

    const int H = static_cast<int>(mgr->num_heads());
    const int F = static_cast<int>(mgr->frame_seqlen());
    const int max_sink = static_cast<int>(mgr->max_sink());
    const int max_middle = static_cast<int>(mgr->max_middle());
    const int max_recent = static_cast<int>(mgr->max_recent());
    const int max_merge_blocks_eff = static_cast<int>(mgr->max_merge_blocks());
    const int max_total_segments = max_sink + max_middle + max_recent
                                 + max_merge_blocks_eff;
    const int N = H * max_total_segments;

    // Pull manager valid_count[layer] to CPU for sink + recent counts.
    auto vc_cpu = mgr->valid_count().to(at::kCPU, /*non_blocking=*/false)
                                    .contiguous();
    auto vc_acc = vc_cpu.accessor<int64_t, 3>();  // [L, H, 3]

    // Pull merge_token_count[layer] to CPU so the planner can emit per-anchor
    // seg_lengths. Shape: [L, H, max_merge_blocks] int32 (matches manager).
    auto mtc_cpu = mgr->merge_token_count().to(at::kCPU, /*non_blocking=*/false)
                                          .contiguous();
    auto mtc_acc = mtc_cpu.accessor<int32_t, 3>();

    // PerHeadState array on CPU: copy the uint8 buffer to host once.
    auto states_cpu = states_bytes.to(at::kCPU, /*non_blocking=*/false)
                                  .contiguous();
    const auto* per_head =
        reinterpret_cast<const PerHeadState*>(states_cpu.data_ptr());

    // Build CPU descriptor buffers (pinned for fast H2D).
    auto i32_opts_cpu = at::TensorOptions()
        .dtype(at::kInt).device(at::kCPU).pinned_memory(true);
    auto i64_opts_cpu = at::TensorOptions()
        .dtype(at::kLong).device(at::kCPU).pinned_memory(true);

    auto cu_cpu       = at::zeros({H + 1}, i32_opts_cpu);
    auto sk_cpu       = at::full({N}, -1, i32_opts_cpu);
    auto sg_cpu       = at::zeros({N}, i32_opts_cpu);
    auto sl_cpu       = at::zeros({N}, i32_opts_cpu);
    auto dst_cpu      = at::zeros({N}, i32_opts_cpu);
    auto traw_cpu     = at::full({N}, -1, i64_opts_cpu);
    auto tremap_cpu   = at::full({N}, -1, i64_opts_cpu);

    auto cu     = cu_cpu.data_ptr<int32_t>();
    auto sk     = sk_cpu.data_ptr<int32_t>();
    auto sg     = sg_cpu.data_ptr<int32_t>();
    auto sl     = sl_cpu.data_ptr<int32_t>();
    auto dst    = dst_cpu.data_ptr<int32_t>();
    auto traw   = traw_cpu.data_ptr<int64_t>();
    auto tremap = tremap_cpu.data_ptr<int64_t>();

    cu[0] = 0;

    const int64_t sync_t = map_sink_time(
        static_cast<int64_t>(current_t),
        static_cast<int>(sink_mode_code),
        static_cast<int>(sink_clamp_min),
        static_cast<int>(sink_clamp_max),
        static_cast<int>(decoupled_sink_lag));

    emit_chunk_plan(
        /*chunk_idx=*/0, H,
        max_sink, max_middle, max_recent, max_merge_blocks_eff,
        max_total_segments, F, static_cast<int>(layer_idx),
        static_cast<int64_t>(current_t), sync_t,
        vc_acc, mtc_acc, per_head,
        /*running_offset_in=*/0,
        cu, sk, sg, sl, dst, traw, tremap,
        static_cast<int>(hist_mode_code),
        static_cast<int>(hist_relative_t_max),
        hist_soft_factor
    );


    // Allocate device tensors and non-blocking H2D copy.
    auto device = mgr->cu_seqlens_k().device();
    auto i32_opts_dev = at::TensorOptions().dtype(at::kInt).device(device);
    auto i64_opts_dev = at::TensorOptions().dtype(at::kLong).device(device);

    auto cu_dev     = at::empty({H + 1}, i32_opts_dev);
    auto sk_dev     = at::empty({N}, i32_opts_dev);
    auto sg_dev     = at::empty({N}, i32_opts_dev);
    auto sl_dev     = at::empty({N}, i32_opts_dev);
    auto dst_dev    = at::empty({N}, i32_opts_dev);
    auto traw_dev   = at::empty({N}, i64_opts_dev);
    auto tremap_dev = at::empty({N}, i64_opts_dev);

    cu_dev.copy_(cu_cpu,       /*non_blocking=*/true);
    sk_dev.copy_(sk_cpu,       /*non_blocking=*/true);
    sg_dev.copy_(sg_cpu,       /*non_blocking=*/true);
    sl_dev.copy_(sl_cpu,       /*non_blocking=*/true);
    dst_dev.copy_(dst_cpu,     /*non_blocking=*/true);
    traw_dev.copy_(traw_cpu,   /*non_blocking=*/true);
    tremap_dev.copy_(tremap_cpu, /*non_blocking=*/true);

    return std::make_tuple(
        cu_dev, sk_dev, sg_dev, sl_dev, dst_dev, traw_dev, tremap_dev
    );
}

static std::tuple<
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor
> mega_plan_meta(
    c10::intrusive_ptr<PyramidKVCacheManager> mgr,
    torch::Tensor /*states_bytes*/,
    int64_t /*layer_idx*/,
    int64_t /*current_t*/,
    int64_t /*pass_kind*/,
    int64_t /*sink_mode_code*/,
    int64_t /*sink_clamp_min*/,
    int64_t /*sink_clamp_max*/,
    int64_t /*decoupled_sink_lag*/,
    int64_t /*hist_mode_code*/,
    int64_t /*hist_relative_t_max*/,
    double  /*hist_soft_factor*/
) {
    const int H = static_cast<int>(mgr->num_heads());
    const int max_total_segments = static_cast<int>(mgr->max_total())
                                 + static_cast<int>(mgr->max_merge_blocks());
    const int N = H * max_total_segments;
    auto i32 = at::TensorOptions().dtype(at::kInt).device(at::kMeta);
    auto i64 = at::TensorOptions().dtype(at::kLong).device(at::kMeta);
    return std::make_tuple(
        at::empty({H + 1}, i32),
        at::empty({N}, i32),
        at::empty({N}, i32),
        at::empty({N}, i32),
        at::empty({N}, i32),
        at::empty({N}, i64),
        at::empty({N}, i64)
    );
}

// Bindings exposed by the bind translation unit.
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
) {
    return mega_plan(
        mgr, states_bytes, layer_idx, current_t, pass_kind,
        sink_mode_code, sink_clamp_min, sink_clamp_max, decoupled_sink_lag,
        hist_mode_code, hist_relative_t_max, hist_soft_factor
    );
}

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
) {
    return mega_plan_meta(
        mgr, states_bytes, layer_idx, current_t, pass_kind,
        sink_mode_code, sink_clamp_min, sink_clamp_max, decoupled_sink_lag,
        hist_mode_code, hist_relative_t_max, hist_soft_factor
    );
}

// multi-chunk plan: emit per-chunk plan descriptors back-to-back in one
// op call. dst_token_offsets accumulate globally across chunks so the pack
// kernel can write all chunks' K/V/pos into disjoint regions of
// mgr.k_flat_out / v_flat_out / pos_flat_out in a single launch.
//
// current_t_list: CPU int64 [num_chunks] — per-chunk query frame index.
// All other kwargs identical to mega_plan (sink/history-time-mapping config
// is shared across chunks).
//
// Outputs (size = num_chunks × original size):
//   cu_seqlens_k       [num_chunks * H + 1]      int32
//   src_kind           [num_chunks * N_per]      int32
//   src_slot_global    [num_chunks * N_per]      int32
//   seg_lengths        [num_chunks * N_per]      int32
//   dst_token_offsets  [num_chunks * N_per]      int32 (chunk-offset globally)
//   anchor_t_raw       [num_chunks * N_per]      int64
//   anchor_t_remap     [num_chunks * N_per]      int64 (per-chunk sync_t / -1)
// where N_per = H * (max_sink + max_middle + max_recent + max_merge_blocks).
static std::tuple<
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor
> mega_plan_multi(
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
) {
    TORCH_CHECK(mgr.get() != nullptr, "mega_plan_multi: manager is null");
    TORCH_CHECK(states_bytes.dtype() == torch::kUInt8,
                "mega_plan_multi: states_bytes must be uint8");
    TORCH_CHECK(states_bytes.numel() >=
                static_cast<int64_t>(mgr->num_heads()) *
                static_cast<int64_t>(sizeof(PerHeadState)),
                "mega_plan_multi: states_bytes too small for ", mgr->num_heads(), " heads");
    TORCH_CHECK(layer_idx >= 0 && layer_idx < mgr->num_layers(),
                "mega_plan_multi: layer_idx out of range");
    TORCH_CHECK(current_t_list.dtype() == at::kLong,
                "mega_plan_multi: current_t_list must be int64");
    TORCH_CHECK(current_t_list.dim() == 1,
                "mega_plan_multi: current_t_list must be 1D");
    (void)pass_kind;

    const int num_chunks = static_cast<int>(current_t_list.numel());
    TORCH_CHECK(num_chunks > 0, "mega_plan_multi: empty current_t_list");

    const int H = static_cast<int>(mgr->num_heads());
    const int F = static_cast<int>(mgr->frame_seqlen());
    const int max_sink = static_cast<int>(mgr->max_sink());
    const int max_middle = static_cast<int>(mgr->max_middle());
    const int max_recent = static_cast<int>(mgr->max_recent());
    const int max_merge_blocks_eff = static_cast<int>(mgr->max_merge_blocks());
    const int max_total_segments = max_sink + max_middle + max_recent
                                 + max_merge_blocks_eff;
    const int N_per = H * max_total_segments;
    const int N_total = num_chunks * N_per;
    const int CU_total = num_chunks * H + 1;

    auto vc_cpu = mgr->valid_count().to(at::kCPU, /*non_blocking=*/false)
                                    .contiguous();
    auto vc_acc = vc_cpu.accessor<int64_t, 3>();

    auto mtc_cpu = mgr->merge_token_count().to(at::kCPU, /*non_blocking=*/false)
                                          .contiguous();
    auto mtc_acc = mtc_cpu.accessor<int32_t, 3>();

    auto states_cpu = states_bytes.to(at::kCPU, /*non_blocking=*/false)
                                  .contiguous();
    const auto* per_head =
        reinterpret_cast<const PerHeadState*>(states_cpu.data_ptr());

    auto current_t_cpu = current_t_list.to(at::kCPU).contiguous();
    auto current_t_acc = current_t_cpu.accessor<int64_t, 1>();

    auto i32_opts_cpu = at::TensorOptions()
        .dtype(at::kInt).device(at::kCPU).pinned_memory(true);
    auto i64_opts_cpu = at::TensorOptions()
        .dtype(at::kLong).device(at::kCPU).pinned_memory(true);

    auto cu_cpu       = at::zeros({CU_total}, i32_opts_cpu);
    auto sk_cpu       = at::full({N_total}, -1, i32_opts_cpu);
    auto sg_cpu       = at::zeros({N_total}, i32_opts_cpu);
    auto sl_cpu       = at::zeros({N_total}, i32_opts_cpu);
    auto dst_cpu      = at::zeros({N_total}, i32_opts_cpu);
    auto traw_cpu     = at::full({N_total}, -1, i64_opts_cpu);
    auto tremap_cpu   = at::full({N_total}, -1, i64_opts_cpu);

    auto cu     = cu_cpu.data_ptr<int32_t>();
    auto sk     = sk_cpu.data_ptr<int32_t>();
    auto sg     = sg_cpu.data_ptr<int32_t>();
    auto sl     = sl_cpu.data_ptr<int32_t>();
    auto dst    = dst_cpu.data_ptr<int32_t>();
    auto traw   = traw_cpu.data_ptr<int64_t>();
    auto tremap = tremap_cpu.data_ptr<int64_t>();

    cu[0] = 0;
    int32_t running_offset = 0;
    for (int c = 0; c < num_chunks; ++c) {
        const int64_t current_t_c = current_t_acc[c];
        const int64_t sync_t_c = map_sink_time(
            current_t_c,
            static_cast<int>(sink_mode_code),
            static_cast<int>(sink_clamp_min),
            static_cast<int>(sink_clamp_max),
            static_cast<int>(decoupled_sink_lag));
        running_offset = emit_chunk_plan(
            c, H,
            max_sink, max_middle, max_recent, max_merge_blocks_eff,
            max_total_segments, F, static_cast<int>(layer_idx),
            current_t_c, sync_t_c,
            vc_acc, mtc_acc, per_head,
            running_offset,
            cu, sk, sg, sl, dst, traw, tremap,
            static_cast<int>(hist_mode_code),
            static_cast<int>(hist_relative_t_max),
            hist_soft_factor
        );
    }

    // Plan B — guard the pack workspace. running_offset is the total tokens
    // mega_plan_multi told pack to write into k_flat_out / v_flat_out /
    // pos_flat_out. If this exceeds the manager's buffer capacity (sized
    // for max_attend_chunks chunks), pack would overrun and corrupt memory.
    TORCH_CHECK(
        static_cast<int64_t>(running_offset) <= mgr->max_pack_tokens(),
        "mega_plan_multi: packed tokens (", running_offset,
        ") exceed pack-workspace capacity (", mgr->max_pack_tokens(),
        "). Either reduce chunks (", num_chunks,
        ") or rebuild the manager with a larger max_attend_chunks "
        "(currently ", mgr->max_attend_chunks(), ")."
    );

    auto device = mgr->cu_seqlens_k().device();
    auto i32_opts_dev = at::TensorOptions().dtype(at::kInt).device(device);
    auto i64_opts_dev = at::TensorOptions().dtype(at::kLong).device(device);

    auto cu_dev     = at::empty({CU_total}, i32_opts_dev);
    auto sk_dev     = at::empty({N_total}, i32_opts_dev);
    auto sg_dev     = at::empty({N_total}, i32_opts_dev);
    auto sl_dev     = at::empty({N_total}, i32_opts_dev);
    auto dst_dev    = at::empty({N_total}, i32_opts_dev);
    auto traw_dev   = at::empty({N_total}, i64_opts_dev);
    auto tremap_dev = at::empty({N_total}, i64_opts_dev);

    cu_dev.copy_(cu_cpu,       /*non_blocking=*/true);
    sk_dev.copy_(sk_cpu,       /*non_blocking=*/true);
    sg_dev.copy_(sg_cpu,       /*non_blocking=*/true);
    sl_dev.copy_(sl_cpu,       /*non_blocking=*/true);
    dst_dev.copy_(dst_cpu,     /*non_blocking=*/true);
    traw_dev.copy_(traw_cpu,   /*non_blocking=*/true);
    tremap_dev.copy_(tremap_cpu, /*non_blocking=*/true);

    return std::make_tuple(
        cu_dev, sk_dev, sg_dev, sl_dev, dst_dev, traw_dev, tremap_dev
    );
}

static std::tuple<
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor
> mega_plan_multi_meta(
    c10::intrusive_ptr<PyramidKVCacheManager> mgr,
    torch::Tensor /*states_bytes*/,
    int64_t /*layer_idx*/,
    torch::Tensor current_t_list,
    int64_t /*pass_kind*/,
    int64_t /*sink_mode_code*/,
    int64_t /*sink_clamp_min*/,
    int64_t /*sink_clamp_max*/,
    int64_t /*decoupled_sink_lag*/,
    int64_t /*hist_mode_code*/,
    int64_t /*hist_relative_t_max*/,
    double  /*hist_soft_factor*/
) {
    const int num_chunks = static_cast<int>(current_t_list.numel());
    const int H = static_cast<int>(mgr->num_heads());
    const int max_total_segments = static_cast<int>(mgr->max_total())
                                 + static_cast<int>(mgr->max_merge_blocks());
    const int N_per = H * max_total_segments;
    const int N_total = num_chunks * N_per;
    const int CU_total = num_chunks * H + 1;
    auto i32 = at::TensorOptions().dtype(at::kInt).device(at::kMeta);
    auto i64 = at::TensorOptions().dtype(at::kLong).device(at::kMeta);
    return std::make_tuple(
        at::empty({CU_total}, i32),
        at::empty({N_total}, i32),
        at::empty({N_total}, i32),
        at::empty({N_total}, i32),
        at::empty({N_total}, i32),
        at::empty({N_total}, i64),
        at::empty({N_total}, i64)
    );
}

// Public bindings — exposed via scatter_copy_bind.cpp.
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
) {
    return mega_plan_multi(
        mgr, states_bytes, layer_idx, current_t_list, pass_kind,
        sink_mode_code, sink_clamp_min, sink_clamp_max, decoupled_sink_lag,
        hist_mode_code, hist_relative_t_max, hist_soft_factor
    );
}

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
) {
    return mega_plan_multi_meta(
        mgr, states_bytes, layer_idx, current_t_list, pass_kind,
        sink_mode_code, sink_clamp_min, sink_clamp_max, decoupled_sink_lag,
        hist_mode_code, hist_relative_t_max, hist_soft_factor
    );
}

}  // namespace adahead

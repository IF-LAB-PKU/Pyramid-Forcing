// pyramidkv_pack_kernel: single mega-kernel that gathers K/V from
// sink/middle/recent pools into the manager's flat output workspace.
//
// Replaces the Python loop of 360 small launches + torch.cat in
// adaptive_cache.py:_materialize_readout_spec(). Runs after PlanInfo
// has been built by pyramidkv_plan() (CPU-side layout, one H2D copy of
// int32 descriptors).
//
// Each segment is a contiguous run of frames within one (layer, head)'s
// destination region. The kernel does a plain memcpy per segment;
// concurrency comes from issuing all segments in one launch.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace adahead {

namespace {

// Pool layout: [L, H, max_slots, frame_seqlen, head_dim] flattened to 1D.
// slot_global = ((layer * num_heads) + head) * max_slots + slot
//             = head_idx * max_slots + slot   (where head_idx = layer * num_heads + head)
// First-row offset within pool = slot_global * frame_seqlen * head_dim.

__global__ void pyramidkv_pack_kernel(
    const __nv_bfloat16* __restrict__ sink_k,
    const __nv_bfloat16* __restrict__ sink_v,
    const __nv_bfloat16* __restrict__ middle_k,
    const __nv_bfloat16* __restrict__ middle_v,
    const __nv_bfloat16* __restrict__ recent_k,
    const __nv_bfloat16* __restrict__ recent_v,
    const __nv_bfloat16* __restrict__ merge_k,    // kind=3 source (nullable when no merge heads)
    const __nv_bfloat16* __restrict__ merge_v,
    // pos pools (nullable, all 4 must be present together to enable
    // pos packing). Layouts mirror K/V pool layouts but with row_stride=3.
    const int64_t* __restrict__ sink_pos,
    const int64_t* __restrict__ middle_pos,
    const int64_t* __restrict__ recent_pos,
    const int64_t* __restrict__ merge_pos,
    __nv_bfloat16* __restrict__ k_flat_out,
    __nv_bfloat16* __restrict__ v_flat_out,
    int64_t* __restrict__ pos_flat_out,           // [N_tokens, 3] int64 OR nullptr
    const int32_t* __restrict__ src_kind,         // [N_segments]
    const int32_t* __restrict__ src_slot_global,  // [N_segments] = head_idx * max_slots[kind] + slot
    const int32_t* __restrict__ seg_lengths,      // [N_segments] in TOKENS
    const int32_t* __restrict__ dst_token_offsets,// [N_segments]
    // fused-RoPE inputs (all nullable; falls back to plain memcpy when null).
    // Pass anchor_t_remap = nullptr OR all three freq tables null to disable RoPE.
    const int64_t* __restrict__ anchor_t_remap,   // [N_segments] int64 (-1 sentinel)
    const float*   __restrict__ freqs_ft_flat,    // [ft_rows, ft_cols * 2] fp32
    const float*   __restrict__ freqs_fy_flat,    // [fy_rows, fy_cols * 2] fp32
    const float*   __restrict__ freqs_fx_flat,    // [fx_rows, fx_cols * 2] fp32
    int ft_rows, int ft_cols,
    int fy_rows, int fy_cols,
    int fx_rows, int fx_cols,
    int frame_seqlen,
    int max_merge_tokens,                          // stride per merge anchor (tokens)
    int head_dim
) {
    const int seg_id = blockIdx.x;
    const int kind = src_kind[seg_id];
    if (kind < 0) return;  // inactive segment (planner left padded slot)

    const int slot_global = src_slot_global[seg_id];
    const int seg_len = seg_lengths[seg_id];   // tokens
    if (seg_len <= 0) return;

    const int dst_token_off = dst_token_offsets[seg_id];
    const int row_stride = head_dim;

    // kind 0/1/2: stride per slot = frame_seqlen * head_dim
    // kind 3   : stride per slot = max_merge_tokens * head_dim
    const int pool_stride_tokens = (kind == 3) ? max_merge_tokens : frame_seqlen;
    const long src_base = static_cast<long>(slot_global) * pool_stride_tokens * head_dim;

    const __nv_bfloat16* src_k_ptr;
    const __nv_bfloat16* src_v_ptr;
    const int64_t* src_pos_ptr = nullptr;
    const long src_pos_base = static_cast<long>(slot_global) * pool_stride_tokens * 3;
    if (kind == 0) {
        src_k_ptr = sink_k + src_base;
        src_v_ptr = sink_v + src_base;
        if (sink_pos != nullptr) src_pos_ptr = sink_pos + src_pos_base;
    } else if (kind == 1) {
        src_k_ptr = middle_k + src_base;
        src_v_ptr = middle_v + src_base;
        if (middle_pos != nullptr) src_pos_ptr = middle_pos + src_pos_base;
    } else if (kind == 2) {
        src_k_ptr = recent_k + src_base;
        src_v_ptr = recent_v + src_base;
        if (recent_pos != nullptr) src_pos_ptr = recent_pos + src_pos_base;
    } else {
        // kind == 3: merge pool (per-anchor token segments)
        src_k_ptr = merge_k + src_base;
        src_v_ptr = merge_v + src_base;
        if (merge_pos != nullptr) src_pos_ptr = merge_pos + src_pos_base;
    }

    __nv_bfloat16* dst_k = k_flat_out + static_cast<long>(dst_token_off) * row_stride;
    __nv_bfloat16* dst_v = v_flat_out + static_cast<long>(dst_token_off) * row_stride;

    // fused RoPE on K. Per-segment effective_t override from
    // anchor_t_remap: when tremap >= 0, override pos.t; else use stored pos.t.
    const bool fuse_rope =
        (anchor_t_remap != nullptr) &&
        (freqs_ft_flat != nullptr || freqs_fy_flat != nullptr || freqs_fx_flat != nullptr) &&
        (src_pos_ptr != nullptr);
    const int64_t tremap = fuse_rope ? anchor_t_remap[seg_id] : -1;
    const int c_half = head_dim / 2;

    if (fuse_rope) {
        // Each thread handles one (token, pair) unit. K layout per token is
        // [c_half complex pairs]; pair p in [0, ft_cols) uses ft, in
        // [ft_cols, ft_cols+fy_cols) uses fy, rest uses fx.
        const int total_pairs = seg_len * c_half;
        const int ft_end = ft_cols;
        const int fy_end = ft_cols + fy_cols;
        for (int idx = threadIdx.x; idx < total_pairs; idx += blockDim.x) {
            const int tok = idx / c_half;
            const int p   = idx % c_half;

            int64_t pos_t = src_pos_ptr[tok * 3 + 0];
            int64_t pos_y = src_pos_ptr[tok * 3 + 1];
            int64_t pos_x = src_pos_ptr[tok * 3 + 2];
            if (tremap >= 0) pos_t = tremap;

            float fr = 1.0f, fi = 0.0f;
            if (p < ft_end) {
                int row = static_cast<int>(pos_t);
                if (row < 0) row = 0;
                if (row >= ft_rows) row = ft_rows - 1;
                const int col = p;
                fr = freqs_ft_flat[row * (ft_cols * 2) + col * 2];
                fi = freqs_ft_flat[row * (ft_cols * 2) + col * 2 + 1];
            } else if (p < fy_end) {
                int row = static_cast<int>(pos_y);
                if (row < 0) row = 0;
                if (row >= fy_rows) row = fy_rows - 1;
                const int col = p - ft_cols;
                fr = freqs_fy_flat[row * (fy_cols * 2) + col * 2];
                fi = freqs_fy_flat[row * (fy_cols * 2) + col * 2 + 1];
            } else {
                int row = static_cast<int>(pos_x);
                if (row < 0) row = 0;
                if (row >= fx_rows) row = fx_rows - 1;
                const int col = p - ft_cols - fy_cols;
                fr = freqs_fx_flat[row * (fx_cols * 2) + col * 2];
                fi = freqs_fx_flat[row * (fx_cols * 2) + col * 2 + 1];
            }

            const int k_off = tok * row_stride + p * 2;
            const float kr = __bfloat162float(src_k_ptr[k_off]);
            const float ki = __bfloat162float(src_k_ptr[k_off + 1]);
            const float new_kr = kr * fr - ki * fi;
            const float new_ki = kr * fi + ki * fr;
            dst_k[k_off]     = __float2bfloat16(new_kr);
            dst_k[k_off + 1] = __float2bfloat16(new_ki);
        }
        // V: plain memcpy (no RoPE).
        const int total_v_elems = seg_len * row_stride;
        for (int e = threadIdx.x; e < total_v_elems; e += blockDim.x) {
            dst_v[e] = src_v_ptr[e];
        }
    } else {
        // Legacy path: pure memcpy K and V.
        const int total_elems = seg_len * row_stride;
        for (int e = threadIdx.x; e < total_elems; e += blockDim.x) {
            dst_k[e] = src_k_ptr[e];
            dst_v[e] = src_v_ptr[e];
        }
    }

    // ---- raw-K storage: gather pos in lock-step with K/V ----
    // seg_len * 3 int64 elements; trivial cost vs K/V.
    if (pos_flat_out != nullptr && src_pos_ptr != nullptr) {
        int64_t* dst_pos = pos_flat_out + static_cast<long>(dst_token_off) * 3;
        const int pos_elems = seg_len * 3;
        for (int e = threadIdx.x; e < pos_elems; e += blockDim.x) {
            dst_pos[e] = src_pos_ptr[e];
        }
    }
}

}  // namespace

void launch_pyramidkv_pack(
    torch::Tensor sink_pool,
    torch::Tensor middle_pool,
    torch::Tensor recent_pool,
    torch::Tensor sink_v_pool,
    torch::Tensor middle_v_pool,
    torch::Tensor recent_v_pool,
    torch::Tensor merge_pool,                      // [L, H, max_merge_blocks, max_merge_tokens, D] or empty
    torch::Tensor merge_v_pool,
    // pos source pools (nullable when caller doesn't need pos packing).
    torch::Tensor sink_pos_pool,
    torch::Tensor middle_pos_pool,
    torch::Tensor recent_pos_pool,
    torch::Tensor merge_pos_pool,
    torch::Tensor k_flat_out,
    torch::Tensor v_flat_out,
    torch::Tensor pos_flat_out,                    // [max_total_tokens, 3] int64 or empty
    torch::Tensor src_kind,
    torch::Tensor src_slot_global,
    torch::Tensor seg_lengths,
    torch::Tensor dst_token_offsets,
    // fused-RoPE inputs (all optional; pass empty tensors to skip RoPE).
    torch::Tensor anchor_t_remap,                  // [N_seg] int64 or empty
    torch::Tensor freqs_ft_flat,                   // [ft_rows, ft_cols * 2] fp32 or empty
    torch::Tensor freqs_fy_flat,                   // [fy_rows, fy_cols * 2] fp32 or empty
    torch::Tensor freqs_fx_flat,                   // [fx_rows, fx_cols * 2] fp32 or empty
    int64_t frame_seqlen,
    int64_t max_merge_tokens,
    int64_t head_dim
) {
    TORCH_CHECK(sink_pool.is_cuda() && middle_pool.is_cuda() && recent_pool.is_cuda(),
                "pools must be on CUDA");
    TORCH_CHECK(k_flat_out.is_cuda() && v_flat_out.is_cuda(),
                "outputs must be on CUDA");
    TORCH_CHECK(sink_pool.scalar_type() == at::kBFloat16,
                "pyramidkv_pack currently only implements bf16 pools");
    TORCH_CHECK(src_kind.dtype() == at::kInt && src_slot_global.dtype() == at::kInt
                && seg_lengths.dtype() == at::kInt && dst_token_offsets.dtype() == at::kInt,
                "all PlanInfo descriptor tensors must be int32");

    const int N = static_cast<int>(src_kind.numel());
    if (N == 0) return;

    // Allow optional (zero-sized) merge pools when no merge heads are present.
    const __nv_bfloat16* merge_k_ptr = merge_pool.numel() > 0
        ? reinterpret_cast<const __nv_bfloat16*>(merge_pool.data_ptr())
        : nullptr;
    const __nv_bfloat16* merge_v_ptr = merge_v_pool.numel() > 0
        ? reinterpret_cast<const __nv_bfloat16*>(merge_v_pool.data_ptr())
        : nullptr;

    // pos pools and pos_flat_out are nullable; the kernel skips
    // the pos copy loop when nullptr.
    auto opt_i64_ptr = [](const torch::Tensor& t) -> const int64_t* {
        return (t.defined() && t.numel() > 0) ? t.data_ptr<int64_t>() : nullptr;
    };
    auto opt_f32_ptr = [](const torch::Tensor& t) -> const float* {
        return (t.defined() && t.numel() > 0) ? t.data_ptr<float>() : nullptr;
    };
    const int64_t* sink_pos_ptr   = opt_i64_ptr(sink_pos_pool);
    const int64_t* middle_pos_ptr = opt_i64_ptr(middle_pos_pool);
    const int64_t* recent_pos_ptr = opt_i64_ptr(recent_pos_pool);
    const int64_t* merge_pos_ptr  = opt_i64_ptr(merge_pos_pool);
    int64_t* pos_flat_ptr = (pos_flat_out.defined() && pos_flat_out.numel() > 0)
        ? pos_flat_out.data_ptr<int64_t>()
        : nullptr;

    // fused RoPE descriptors. Per-segment effective_t override comes from
    // anchor_t_remap; freqs flat tables are fp32 [rows, cols*2] (real, imag
    // interleaved). All optional — kernel falls back to plain memcpy.
    const int64_t* tremap_ptr = opt_i64_ptr(anchor_t_remap);
    const float* ft_ptr = opt_f32_ptr(freqs_ft_flat);
    const float* fy_ptr = opt_f32_ptr(freqs_fy_flat);
    const float* fx_ptr = opt_f32_ptr(freqs_fx_flat);
    int ft_rows = 0, ft_cols = 0;
    int fy_rows = 0, fy_cols = 0;
    int fx_rows = 0, fx_cols = 0;
    if (ft_ptr != nullptr) {
        TORCH_CHECK(freqs_ft_flat.dim() == 2 && freqs_ft_flat.dtype() == at::kFloat,
                    "freqs_ft_flat must be 2D fp32");
        ft_rows = static_cast<int>(freqs_ft_flat.size(0));
        ft_cols = static_cast<int>(freqs_ft_flat.size(1) / 2);
    }
    if (fy_ptr != nullptr) {
        TORCH_CHECK(freqs_fy_flat.dim() == 2 && freqs_fy_flat.dtype() == at::kFloat,
                    "freqs_fy_flat must be 2D fp32");
        fy_rows = static_cast<int>(freqs_fy_flat.size(0));
        fy_cols = static_cast<int>(freqs_fy_flat.size(1) / 2);
    }
    if (fx_ptr != nullptr) {
        TORCH_CHECK(freqs_fx_flat.dim() == 2 && freqs_fx_flat.dtype() == at::kFloat,
                    "freqs_fx_flat must be 2D fp32");
        fx_rows = static_cast<int>(freqs_fx_flat.size(0));
        fx_cols = static_cast<int>(freqs_fx_flat.size(1) / 2);
    }

    constexpr int kThreads = 256;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(k_flat_out.device().index());

    pyramidkv_pack_kernel<<<N, kThreads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(sink_pool.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(sink_v_pool.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(middle_pool.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(middle_v_pool.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(recent_pool.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(recent_v_pool.data_ptr()),
        merge_k_ptr,
        merge_v_ptr,
        sink_pos_ptr, middle_pos_ptr, recent_pos_ptr, merge_pos_ptr,
        reinterpret_cast<__nv_bfloat16*>(k_flat_out.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(v_flat_out.data_ptr()),
        pos_flat_ptr,
        src_kind.data_ptr<int32_t>(),
        src_slot_global.data_ptr<int32_t>(),
        seg_lengths.data_ptr<int32_t>(),
        dst_token_offsets.data_ptr<int32_t>(),
        tremap_ptr, ft_ptr, fy_ptr, fx_ptr,
        ft_rows, ft_cols, fy_rows, fy_cols, fx_rows, fx_cols,
        static_cast<int>(frame_seqlen),
        static_cast<int>(max_merge_tokens),
        static_cast<int>(head_dim)
    );
}

}  // namespace adahead

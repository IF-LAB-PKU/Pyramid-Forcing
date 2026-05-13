#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <math.h>

// Scatter-copy kernel: each thread block copies one segment from src to dst.
// src_ptrs[seg] is the raw device pointer to source data.
// dst is the workspace base pointer.
// offsets[seg] is the row offset in dst where this segment starts.
// lengths[seg] is the number of rows to copy.
// row_bytes is the number of bytes per row (cols * sizeof(element)).

__global__ void scatter_copy_kernel(
    const int64_t* __restrict__ src_ptrs,   // [N] source data pointers (as int64)
    char* __restrict__ dst,                  // workspace base
    const int64_t* __restrict__ offsets,     // [N] row offsets
    const int64_t* __restrict__ lengths,     // [N] row counts
    int64_t row_bytes,                       // bytes per row
    int N
) {
    const int seg = blockIdx.x;
    if (seg >= N) return;

    const int64_t len = lengths[seg];
    if (len <= 0) return;

    const char* src = reinterpret_cast<const char*>(src_ptrs[seg]);
    char* dst_seg = dst + offsets[seg] * row_bytes;
    const int64_t total_bytes = len * row_bytes;

    // Use 4-byte copies (safe for all alignments with PyTorch tensors)
    const int32_t* src4 = reinterpret_cast<const int32_t*>(src);
    int32_t* dst4 = reinterpret_cast<int32_t*>(dst_seg);
    const int64_t total_words = total_bytes / 4;

    for (int64_t i = static_cast<int64_t>(threadIdx.x); i < total_words;
         i += static_cast<int64_t>(blockDim.x)) {
        dst4[i] = src4[i];
    }
    // Handle remaining bytes (0-3 bytes)
    const int64_t tail_start = total_words * 4;
    if (threadIdx.x == 0) {
        for (int64_t j = tail_start; j < total_bytes; j++) {
            dst_seg[j] = src[j];
        }
    }
}

// Override kernel: set pos[:, 0] = val for specified ranges.
// pos is [total_len, 3] int64 tensor.
// Each thread block handles one override range.

__global__ void pos_override_kernel(
    int64_t* __restrict__ pos,              // [total_len, 3] flattened
    const int64_t* __restrict__ starts,     // [M] start indices
    const int64_t* __restrict__ ends,       // [M] end indices
    const int64_t* __restrict__ vals,       // [M] override values
    int M
) {
    const int range_idx = blockIdx.x;
    if (range_idx >= M) return;

    const int64_t start = starts[range_idx];
    const int64_t end = ends[range_idx];
    const int64_t val = vals[range_idx];

    // pos is [N, 3], stride=3. pos[i, 0] = pos[i*3]
    for (int64_t i = start + threadIdx.x; i < end; i += blockDim.x) {
        pos[i * 3] = val;
    }
}

// PyramidKV refresh descriptor flags.
// bit 0: apply dynamic history time mapping
// bit 1: frame ids should use physical/source time instead of mapped time
// bit 2: dynamic_rope_t contains an override for pos[:, 0]
__global__ void refresh_readout_layout_kernel(
    const int64_t* __restrict__ src_ptrs_k,
    const int64_t* __restrict__ src_ptrs_v,
    const int64_t* __restrict__ src_ptrs_pos,
    const int64_t* __restrict__ offsets,
    const int64_t* __restrict__ lengths,
    const int64_t* __restrict__ flags,
    const int64_t* __restrict__ dynamic_rope_t,
    char* __restrict__ dst_k_raw,
    char* __restrict__ dst_v,
    int64_t* __restrict__ dst_rope_pos,
    int64_t* __restrict__ dst_frame_ids,
    int64_t row_bytes,
    int64_t current_t,
    int64_t mapping_mode,
    int64_t history_relative_t_max,
    float history_time_soft_factor,
    int N
) {
    const int seg = blockIdx.x;
    if (seg >= N) return;

    const int64_t len = lengths[seg];
    if (len <= 0) return;

    const int64_t offset = offsets[seg];
    const int64_t flag = flags[seg];
    const char* src_k = reinterpret_cast<const char*>(src_ptrs_k[seg]);
    const char* src_v = reinterpret_cast<const char*>(src_ptrs_v[seg]);
    const int64_t* src_pos = reinterpret_cast<const int64_t*>(src_ptrs_pos[seg]);
    char* out_k = dst_k_raw + offset * row_bytes;
    char* out_v = dst_v + offset * row_bytes;
    int64_t* out_pos = dst_rope_pos + offset * 3;
    int64_t* out_frame = dst_frame_ids + offset;

    const int64_t total_bytes = len * row_bytes;
    const int32_t* src_k4 = reinterpret_cast<const int32_t*>(src_k);
    const int32_t* src_v4 = reinterpret_cast<const int32_t*>(src_v);
    int32_t* out_k4 = reinterpret_cast<int32_t*>(out_k);
    int32_t* out_v4 = reinterpret_cast<int32_t*>(out_v);
    const int64_t total_words = total_bytes / 4;

    for (int64_t i = static_cast<int64_t>(threadIdx.x); i < total_words;
         i += static_cast<int64_t>(blockDim.x)) {
        out_k4[i] = src_k4[i];
        out_v4[i] = src_v4[i];
    }

    const int64_t tail_start = total_words * 4;
    if (threadIdx.x == 0) {
        for (int64_t j = tail_start; j < total_bytes; j++) {
            out_k[j] = src_k[j];
            out_v[j] = src_v[j];
        }
    }

    const bool do_time_map = (flag & 1LL) != 0;
    const bool frame_physical = (flag & 2LL) != 0;
    const bool do_rope_override = (flag & 4LL) != 0;
    const int64_t rope_t = dynamic_rope_t[seg];

    for (int64_t row = static_cast<int64_t>(threadIdx.x); row < len;
         row += static_cast<int64_t>(blockDim.x)) {
        const int64_t src_t = src_pos[row * 3];
        int64_t mapped_t = src_t;
        if (do_rope_override) {
            mapped_t = rope_t;
        } else if (do_time_map && history_relative_t_max > 0) {
            int64_t rel = current_t - src_t;
            if (rel < 0) rel = 0;
            if (mapping_mode == 1) {
                if (rel > history_relative_t_max) rel = history_relative_t_max;
                mapped_t = current_t - rel;
            } else if (mapping_mode == 2) {
                if (rel > history_relative_t_max) {
                    const int64_t over = rel - history_relative_t_max;
                    const int64_t compressed_over = static_cast<int64_t>(
                        llrintf(static_cast<float>(over) * history_time_soft_factor)
                    );
                    rel = history_relative_t_max + compressed_over;
                }
                mapped_t = current_t - rel;
            }
            if (mapped_t < 0) mapped_t = 0;
        }

        out_pos[row * 3] = mapped_t;
        out_pos[row * 3 + 1] = src_pos[row * 3 + 1];
        out_pos[row * 3 + 2] = src_pos[row * 3 + 2];
        out_frame[row] = frame_physical ? src_t : mapped_t;
    }
}

__global__ void anchor_store_write_frames_kernel(
    const char* __restrict__ k_seq,
    const char* __restrict__ v_seq,
    const int64_t* __restrict__ pos_seq,
    const int64_t* __restrict__ frame_desc,
    char* __restrict__ store_k,
    char* __restrict__ store_v,
    int64_t* __restrict__ store_pos,
    int64_t frame_seqlen,
    int64_t head_dim,
    int64_t row_bytes,
    int N
) {
    const int frame = blockIdx.x;
    if (frame >= N) return;

    const int64_t src_frame = frame_desc[frame * 2];
    const int64_t slot = frame_desc[frame * 2 + 1];
    if (src_frame < 0 || slot < 0) return;

    const int64_t src_token = src_frame * frame_seqlen;
    const int64_t token_bytes = frame_seqlen * row_bytes;
    const char* src_k = k_seq + src_token * row_bytes;
    const char* src_v = v_seq + src_token * row_bytes;
    char* dst_k = store_k + slot * token_bytes;
    char* dst_v = store_v + slot * token_bytes;

    const int32_t* src_k4 = reinterpret_cast<const int32_t*>(src_k);
    const int32_t* src_v4 = reinterpret_cast<const int32_t*>(src_v);
    int32_t* dst_k4 = reinterpret_cast<int32_t*>(dst_k);
    int32_t* dst_v4 = reinterpret_cast<int32_t*>(dst_v);
    const int64_t total_words = token_bytes / 4;

    for (int64_t i = static_cast<int64_t>(threadIdx.x); i < total_words;
         i += static_cast<int64_t>(blockDim.x)) {
        dst_k4[i] = src_k4[i];
        dst_v4[i] = src_v4[i];
    }

    const int64_t tail_start = total_words * 4;
    if (threadIdx.x == 0) {
        for (int64_t j = tail_start; j < token_bytes; j++) {
            dst_k[j] = src_k[j];
            dst_v[j] = src_v[j];
        }
    }

    const int64_t* src_pos = pos_seq + src_token * 3;
    int64_t* dst_pos = store_pos + slot * frame_seqlen * 3;
    for (int64_t row = static_cast<int64_t>(threadIdx.x); row < frame_seqlen;
         row += static_cast<int64_t>(blockDim.x)) {
        dst_pos[row * 3] = src_pos[row * 3];
        dst_pos[row * 3 + 1] = src_pos[row * 3 + 1];
        dst_pos[row * 3 + 2] = src_pos[row * 3 + 2];
    }
}

// Launch wrappers called from scatter_copy_bind.cpp
#include <torch/extension.h>

void launch_scatter_copy(
    torch::Tensor src_ptrs,
    torch::Tensor dst,
    torch::Tensor offsets,
    torch::Tensor lengths,
    int64_t col_dim,
    int N
) {
    int64_t elem_size = dst.element_size();
    int64_t row_bytes = col_dim * elem_size;

    const int threads = 256;
    const int blocks = N;

    scatter_copy_kernel<<<blocks, threads>>>(
        src_ptrs.data_ptr<int64_t>(),
        reinterpret_cast<char*>(dst.data_ptr()),
        offsets.data_ptr<int64_t>(),
        lengths.data_ptr<int64_t>(),
        row_bytes,
        N
    );
}

void launch_pos_override(
    torch::Tensor pos,
    torch::Tensor starts,
    torch::Tensor ends,
    torch::Tensor vals
) {
    int M = starts.size(0);
    const int threads = 256;

    pos_override_kernel<<<M, threads>>>(
        pos.data_ptr<int64_t>(),
        starts.data_ptr<int64_t>(),
        ends.data_ptr<int64_t>(),
        vals.data_ptr<int64_t>(),
        M
    );
}

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
) {
    int64_t elem_size = dst_k_raw.element_size();
    int64_t row_bytes = head_dim * elem_size;
    const int threads = 256;
    const int blocks = N;

    refresh_readout_layout_kernel<<<blocks, threads>>>(
        src_ptrs_k.data_ptr<int64_t>(),
        src_ptrs_v.data_ptr<int64_t>(),
        src_ptrs_pos.data_ptr<int64_t>(),
        offsets.data_ptr<int64_t>(),
        lengths.data_ptr<int64_t>(),
        flags.data_ptr<int64_t>(),
        dynamic_rope_t.data_ptr<int64_t>(),
        reinterpret_cast<char*>(dst_k_raw.data_ptr()),
        reinterpret_cast<char*>(dst_v.data_ptr()),
        dst_rope_pos.data_ptr<int64_t>(),
        dst_frame_ids.data_ptr<int64_t>(),
        row_bytes,
        current_t,
        mapping_mode,
        history_relative_t_max,
        static_cast<float>(history_time_soft_factor),
        N
    );
}

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
) {
    int64_t elem_size = store_k.element_size();
    int64_t row_bytes = head_dim * elem_size;
    const int threads = 256;
    const int blocks = N;

    anchor_store_write_frames_kernel<<<blocks, threads>>>(
        reinterpret_cast<const char*>(k_seq.data_ptr()),
        reinterpret_cast<const char*>(v_seq.data_ptr()),
        pos_seq.data_ptr<int64_t>(),
        frame_desc.data_ptr<int64_t>(),
        reinterpret_cast<char*>(store_k.data_ptr()),
        reinterpret_cast<char*>(store_v.data_ptr()),
        store_pos.data_ptr<int64_t>(),
        frame_seqlen,
        head_dim,
        row_bytes,
        N
    );
}

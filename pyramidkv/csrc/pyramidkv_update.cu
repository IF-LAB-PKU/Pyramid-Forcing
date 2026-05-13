// pyramidkv_update_kernel: write incoming K/V frames into pool slots.
//
// Companion to pyramidkv_pack: pack READS pool slots into a flat workspace,
// update WRITES new K/V from the current denoising block into pool slots.
//
// Per-(layer, head, frame_in_block) target descriptor is built by the
// Python plan stage (or a future C++ plan extension) — this kernel does
// only the data movement, no strategy decision. Each block of the kernel
// grid handles one (frame, head) -> pool slot write.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace adahead {

namespace {

__global__ void pyramidkv_update_kernel(
    // Source: this block's new K/V, layout [frames_in_block, H, frame_seqlen, head_dim]
    // (we pass a flat pointer; element at frame f, head h, token t, dim d is at
    //  f * H * F * D + h * F * D + t * D + d).
    const __nv_bfloat16* __restrict__ new_k,
    const __nv_bfloat16* __restrict__ new_v,
    // Source: per-token pos_3d, layout [frames_in_block, H, frame_seqlen, 3] int64
    // (nullable — when null, pos pools aren't written and the legacy K/V-only
    // behavior runs).
    const int64_t* __restrict__ new_pos,
    // Destination pools, all [L, H, max_slots, F, D] flat.
    __nv_bfloat16* __restrict__ sink_k,
    __nv_bfloat16* __restrict__ sink_v,
    __nv_bfloat16* __restrict__ middle_k,
    __nv_bfloat16* __restrict__ middle_v,
    __nv_bfloat16* __restrict__ recent_k,
    __nv_bfloat16* __restrict__ recent_v,
    // per-kind pos pools [L, H, max_slots, F, 3] int64 (nullable).
    int64_t* __restrict__ sink_pos,
    int64_t* __restrict__ middle_pos,
    int64_t* __restrict__ recent_pos,
    // Write descriptor: one entry per write op.
    const int32_t* __restrict__ dst_kind,         // 0/1/2/-1
    const int32_t* __restrict__ dst_slot_global,  // packed (layer, head, slot) index into target pool
    const int32_t* __restrict__ src_frame_in_block,
    const int32_t* __restrict__ src_head,
    int H,
    int frame_seqlen,
    int head_dim
) {
    const int op_id = blockIdx.x;
    const int kind = dst_kind[op_id];
    if (kind < 0) return;  // skip slot

    const int slot_global = dst_slot_global[op_id];
    const int frame_in_block = src_frame_in_block[op_id];
    const int head = src_head[op_id];

    // Source offset (in elements) into new_k / new_v.
    // Layout: [frames_in_block, H, frame_seqlen, head_dim]
    const long src_base = (static_cast<long>(frame_in_block) * H + head) *
                          static_cast<long>(frame_seqlen) * head_dim;

    // Destination offset into the chosen pool. Each pool layout is
    // [L, H, max_slots, F, D] flattened; slot_global already encodes
    // (layer * H + head) * max_slots + slot, so we just multiply by F*D.
    const long dst_base = static_cast<long>(slot_global) *
                          static_cast<long>(frame_seqlen) * head_dim;

    __nv_bfloat16* dst_k_ptr;
    __nv_bfloat16* dst_v_ptr;
    int64_t* dst_pos_ptr = nullptr;
    if (kind == 0) {
        dst_k_ptr = sink_k + dst_base;
        dst_v_ptr = sink_v + dst_base;
        if (sink_pos != nullptr) dst_pos_ptr = sink_pos;
    } else if (kind == 1) {
        dst_k_ptr = middle_k + dst_base;
        dst_v_ptr = middle_v + dst_base;
        if (middle_pos != nullptr) dst_pos_ptr = middle_pos;
    } else {
        dst_k_ptr = recent_k + dst_base;
        dst_v_ptr = recent_v + dst_base;
        if (recent_pos != nullptr) dst_pos_ptr = recent_pos;
    }

    const __nv_bfloat16* src_k_ptr = new_k + src_base;
    const __nv_bfloat16* src_v_ptr = new_v + src_base;

    const int total_elems = frame_seqlen * head_dim;
    for (int e = threadIdx.x; e < total_elems; e += blockDim.x) {
        dst_k_ptr[e] = src_k_ptr[e];
        dst_v_ptr[e] = src_v_ptr[e];
    }

    // ---- raw-K storage: write per-token pos_3d alongside K/V ----
    // pos layout [L, H, max_slots, F, 3] flattened — slot_global * F * 3
    // gives the slot's pos base. F*3 elements total per op (= 4680 for
    // pyramid_forcing10, << K/V's F*D=200K), so loop cost is negligible.
    if (new_pos != nullptr && dst_pos_ptr != nullptr) {
        const long src_pos_base = (static_cast<long>(frame_in_block) * H + head)
                                  * static_cast<long>(frame_seqlen) * 3;
        const long dst_pos_base = static_cast<long>(slot_global)
                                  * static_cast<long>(frame_seqlen) * 3;
        const int pos_elems = frame_seqlen * 3;
        for (int e = threadIdx.x; e < pos_elems; e += blockDim.x) {
            dst_pos_ptr[dst_pos_base + e] = new_pos[src_pos_base + e];
        }
    }
}

}  // namespace

// External entry; called from scatter_copy_bind.cpp.
void launch_pyramidkv_update(
    torch::Tensor new_k,           // [frames_in_block, H, frame_seqlen, head_dim] bf16
    torch::Tensor new_v,
    torch::Tensor new_pos,         // [frames_in_block, H, frame_seqlen, 3] int64 OR empty
    torch::Tensor sink_k_pool,
    torch::Tensor sink_v_pool,
    torch::Tensor middle_k_pool,
    torch::Tensor middle_v_pool,
    torch::Tensor recent_k_pool,
    torch::Tensor recent_v_pool,
    torch::Tensor sink_pos_pool,   // [L, H, max_sink, F, 3] int64 OR empty
    torch::Tensor middle_pos_pool, // [L, H, max_middle, F, 3] int64 OR empty
    torch::Tensor recent_pos_pool, // [L, H, max_recent, F, 3] int64 OR empty
    torch::Tensor dst_kind,        // [N_writes] int32
    torch::Tensor dst_slot_global, // [N_writes] int32
    torch::Tensor src_frame_in_block, // [N_writes] int32
    torch::Tensor src_head,        // [N_writes] int32
    int64_t H,
    int64_t frame_seqlen,
    int64_t head_dim
) {
    TORCH_CHECK(new_k.is_cuda() && new_v.is_cuda(),
                "pyramidkv_update: new_k/new_v must be on CUDA");
    TORCH_CHECK(new_k.scalar_type() == at::kBFloat16,
                "pyramidkv_update currently only implements bf16");
    TORCH_CHECK(dst_kind.dtype() == at::kInt && dst_slot_global.dtype() == at::kInt
                && src_frame_in_block.dtype() == at::kInt && src_head.dtype() == at::kInt,
                "all descriptor tensors must be int32");

    const int N = static_cast<int>(dst_kind.numel());
    if (N == 0) return;

    // pos pools are optional. Pass empty tensors to disable.
    const int64_t* new_pos_ptr = nullptr;
    int64_t* sink_pos_ptr = nullptr;
    int64_t* middle_pos_ptr = nullptr;
    int64_t* recent_pos_ptr = nullptr;
    if (new_pos.defined() && new_pos.numel() > 0) {
        TORCH_CHECK(new_pos.dtype() == at::kLong,
                    "pyramidkv_update: new_pos must be int64");
        new_pos_ptr = new_pos.data_ptr<int64_t>();
    }
    if (sink_pos_pool.defined() && sink_pos_pool.numel() > 0) {
        sink_pos_ptr = sink_pos_pool.data_ptr<int64_t>();
    }
    if (middle_pos_pool.defined() && middle_pos_pool.numel() > 0) {
        middle_pos_ptr = middle_pos_pool.data_ptr<int64_t>();
    }
    if (recent_pos_pool.defined() && recent_pos_pool.numel() > 0) {
        recent_pos_ptr = recent_pos_pool.data_ptr<int64_t>();
    }

    constexpr int kThreads = 256;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(new_k.device().index());

    pyramidkv_update_kernel<<<N, kThreads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(new_k.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(new_v.data_ptr()),
        new_pos_ptr,
        reinterpret_cast<__nv_bfloat16*>(sink_k_pool.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(sink_v_pool.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(middle_k_pool.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(middle_v_pool.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(recent_k_pool.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(recent_v_pool.data_ptr()),
        sink_pos_ptr, middle_pos_ptr, recent_pos_ptr,
        dst_kind.data_ptr<int32_t>(),
        dst_slot_global.data_ptr<int32_t>(),
        src_frame_in_block.data_ptr<int32_t>(),
        src_head.data_ptr<int32_t>(),
        static_cast<int>(H),
        static_cast<int>(frame_seqlen),
        static_cast<int>(head_dim)
    );
}

}  // namespace adahead

// pyramidkv_plan: CPU-side layout planning.
//
// Reads manager state (valid_count, head_t_table) and produces the
// descriptor tensors consumed by pyramidkv_pack. One H2D copy at the end.
//
// Simplified vs. the full FlashInfer Plan/Run design:
//   - One segment per (layer, head, kind), assuming slots are filled
//     sequentially 0..valid_count[kind] (no ring-buffer wrap support).
//     seg_length = valid_count * frame_seqlen.
//   - dst layout is sink → middle → recent within each (layer, head),
//     concatenated head-major (layer-major, then head-major).
//   - cu_seqlens_k follows FlashInfer convention (int32, length N+1).
//
// All intermediate buffers are built on CPU; final outputs are H2D-copied
// to the manager's preallocated descriptor tensors with non_blocking=True.
// The plan stage runs ONCE per block, OUTSIDE any CUDA Graph capture.
#include "pyramidkv_manager.h"

#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <tuple>
#include <vector>

namespace adahead {

namespace {

// Pack (layer, head, slot_in_kind) → slot_global for the given pool.
// Pool layout: [L, H, max_slots_for_this_kind, F, D]; flatten = (l*H + h)*max_slots + slot.
inline int32_t pack_slot_global(int layer, int head, int slot_in_kind, int num_heads, int max_slots) {
    return ((layer * num_heads) + head) * max_slots + slot_in_kind;
}

}  // namespace

// Returns 5 device int32 tensors:
//   cu_seqlens_k       [N+1]   (per FlashInfer ragged convention)
//   src_kind           [3*N]
//   src_slot_global    [3*N]
//   seg_lengths        [3*N]
//   dst_token_offsets  [3*N]
// Inactive segments (kind=-1) are emitted explicitly so the kernel grid
// has a stable shape; the kernel early-returns on kind<0.
static std::tuple<
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor
> pyramidkv_plan(
    c10::intrusive_ptr<PyramidKVCacheManager> mgr,
    int64_t current_t,
    int64_t pass_kind
) {
    TORCH_CHECK(mgr.get() != nullptr, "pyramidkv_plan: manager is null");
    (void)current_t;
    (void)pass_kind;

    const int L = static_cast<int>(mgr->num_layers());
    const int H = static_cast<int>(mgr->num_heads());
    const int F = static_cast<int>(mgr->frame_seqlen());
    const int N = L * H;
    const int max_seg = 3;  // sink, middle, recent — extend later if needed

    // Pull ring-state to CPU. valid_count is small (L*H*3 int64 = a few KB).
    auto vc_cpu = mgr->valid_count().to(at::kCPU, /*non_blocking=*/false).contiguous();
    auto vc_acc = vc_cpu.accessor<int64_t, 3>();  // [L, H, 3]

    const int max_sink = static_cast<int>(mgr->max_sink());
    const int max_middle = static_cast<int>(mgr->max_middle());
    const int max_recent = static_cast<int>(mgr->max_recent());
    const int kind_max_slots[3] = {max_sink, max_middle, max_recent};

    // Build CPU buffers.
    auto i32_opts_cpu = at::TensorOptions().dtype(at::kInt).device(at::kCPU).pinned_memory(true);
    auto cu_seqlens_k_cpu = at::zeros({N + 1}, i32_opts_cpu);
    auto src_kind_cpu = at::full({N * max_seg}, -1, i32_opts_cpu);
    auto src_slot_global_cpu = at::zeros({N * max_seg}, i32_opts_cpu);
    auto seg_lengths_cpu = at::zeros({N * max_seg}, i32_opts_cpu);
    auto dst_token_offsets_cpu = at::zeros({N * max_seg}, i32_opts_cpu);

    auto cu = cu_seqlens_k_cpu.data_ptr<int32_t>();
    auto sk = src_kind_cpu.data_ptr<int32_t>();
    auto sg = src_slot_global_cpu.data_ptr<int32_t>();
    auto sl = seg_lengths_cpu.data_ptr<int32_t>();
    auto dst = dst_token_offsets_cpu.data_ptr<int32_t>();

    int32_t running_offset = 0;
    cu[0] = 0;
    // Output order per head matches Python pack: sink → recent (dynamic) → middle (anchors).
    // Pool kind enum stays 0=sink, 1=middle, 2=recent — only the iteration order changes.
    const int kind_order[3] = {0, 2, 1};
    for (int l = 0; l < L; ++l) {
        for (int h = 0; h < H; ++h) {
            const int head_idx = l * H + h;
            for (int kind_pos = 0; kind_pos < 3; ++kind_pos) {
                const int kind = kind_order[kind_pos];
                const int desc_pos = head_idx * max_seg + kind_pos;
                const int n_valid = static_cast<int>(vc_acc[l][h][kind]);
                if (n_valid <= 0) {
                    sk[desc_pos] = -1;  // inactive
                    sl[desc_pos] = 0;
                    continue;
                }
                sk[desc_pos] = kind;
                sg[desc_pos] = pack_slot_global(l, h, /*slot_in_kind=*/0, H, kind_max_slots[kind]);
                const int32_t seg_tokens = static_cast<int32_t>(n_valid) * F;
                sl[desc_pos] = seg_tokens;
                dst[desc_pos] = running_offset;
                running_offset += seg_tokens;
            }
            cu[head_idx + 1] = running_offset;
        }
    }

    // Single non-blocking H2D copy of the descriptor buffers into pre-allocated
    // device-side tensors. We allocate fresh device tensors per plan call
    // (they're small and the alloc is amortized by graph mempool reuse).
    auto device = mgr->cu_seqlens_k().device();
    auto i32_opts_dev = at::TensorOptions().dtype(at::kInt).device(device);

    auto cu_dev = at::empty({N + 1}, i32_opts_dev);
    auto sk_dev = at::empty({N * max_seg}, i32_opts_dev);
    auto sg_dev = at::empty({N * max_seg}, i32_opts_dev);
    auto sl_dev = at::empty({N * max_seg}, i32_opts_dev);
    auto dst_dev = at::empty({N * max_seg}, i32_opts_dev);

    cu_dev.copy_(cu_seqlens_k_cpu, /*non_blocking=*/true);
    sk_dev.copy_(src_kind_cpu, /*non_blocking=*/true);
    sg_dev.copy_(src_slot_global_cpu, /*non_blocking=*/true);
    sl_dev.copy_(seg_lengths_cpu, /*non_blocking=*/true);
    dst_dev.copy_(dst_token_offsets_cpu, /*non_blocking=*/true);

    return std::make_tuple(cu_dev, sk_dev, sg_dev, sl_dev, dst_dev);
}

static std::tuple<
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor
> pyramidkv_plan_meta(
    c10::intrusive_ptr<PyramidKVCacheManager> mgr,
    int64_t /*current_t*/,
    int64_t /*pass_kind*/
) {
    TORCH_CHECK(mgr.get() != nullptr, "pyramidkv_plan: manager is null");
    const int L = static_cast<int>(mgr->num_layers());
    const int H = static_cast<int>(mgr->num_heads());
    const int N = L * H;
    constexpr int max_seg = 3;
    auto opts = at::TensorOptions().dtype(at::kInt).device(at::kMeta);
    return std::make_tuple(
        at::empty({N + 1}, opts),
        at::empty({N * max_seg}, opts),
        at::empty({N * max_seg}, opts),
        at::empty({N * max_seg}, opts),
        at::empty({N * max_seg}, opts)
    );
}

// Forward declarations for the binding file.
std::tuple<
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor
> _pyramidkv_plan_cuda(
    c10::intrusive_ptr<PyramidKVCacheManager> mgr,
    int64_t current_t,
    int64_t pass_kind
) { return pyramidkv_plan(mgr, current_t, pass_kind); }

std::tuple<
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor
> _pyramidkv_plan_meta(
    c10::intrusive_ptr<PyramidKVCacheManager> mgr,
    int64_t current_t,
    int64_t pass_kind
) { return pyramidkv_plan_meta(mgr, current_t, pass_kind); }

}  // namespace adahead

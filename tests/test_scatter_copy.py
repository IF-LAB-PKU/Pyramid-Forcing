"""Tests for CUDA scatter-copy kernel.

These tests require a CUDA GPU. They verify that the custom scatter_copy
kernel produces identical results to torch.cat(out=workspace).
"""
import pytest
import torch

from headkv.rope import map_dynamic_pos_time


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


def _reference_scatter_copy(src_list, dst, col_dim):
    """Reference: torch.cat into dst."""
    if src_list:
        torch.cat(src_list, dim=0, out=dst[:sum(t.shape[0] for t in src_list)])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestScatterCopy:

    def test_basic_bf16(self, device):
        """Scatter-copy bf16 tensors matches torch.cat."""
        from headkv._scatter_ext import scatter_copy

        head_dim = 128
        lengths = [1560, 3120, 1560, 4680]
        src_list = [torch.randn(l, head_dim, device=device, dtype=torch.bfloat16) for l in lengths]
        total = sum(lengths)

        # Reference
        ref_out = torch.empty(total, head_dim, device=device, dtype=torch.bfloat16)
        torch.cat(src_list, dim=0, out=ref_out)

        # Custom kernel
        test_out = torch.empty(total, head_dim, device=device, dtype=torch.bfloat16)
        ptrs = torch.tensor([t.data_ptr() for t in src_list], dtype=torch.int64, device=device)
        lens = torch.tensor(lengths, dtype=torch.int64, device=device)
        offsets = torch.zeros(len(lengths), dtype=torch.int64, device=device)
        offsets[1:] = torch.cumsum(lens[:-1], 0)

        scatter_copy(ptrs, lens, offsets, test_out, head_dim)
        torch.cuda.synchronize()

        assert torch.equal(ref_out, test_out), f"Max diff: {(ref_out - test_out).abs().max()}"

    def test_int64_pos(self, device):
        """Scatter-copy int64 position tensors (3 columns)."""
        from headkv._scatter_ext import scatter_copy

        col_dim = 3
        lengths = [1560, 1560, 3120]
        src_list = [torch.randint(0, 100, (l, col_dim), device=device, dtype=torch.int64) for l in lengths]
        total = sum(lengths)

        ref_out = torch.empty(total, col_dim, device=device, dtype=torch.int64)
        torch.cat(src_list, dim=0, out=ref_out)

        test_out = torch.empty(total, col_dim, device=device, dtype=torch.int64)
        ptrs = torch.tensor([t.data_ptr() for t in src_list], dtype=torch.int64, device=device)
        lens = torch.tensor(lengths, dtype=torch.int64, device=device)
        offsets = torch.zeros(len(lengths), dtype=torch.int64, device=device)
        offsets[1:] = torch.cumsum(lens[:-1], 0)

        scatter_copy(ptrs, lens, offsets, test_out, col_dim)
        torch.cuda.synchronize()

        assert torch.equal(ref_out, test_out)

    def test_empty_segments(self, device):
        """Handle zero-length segments gracefully."""
        from headkv._scatter_ext import scatter_copy

        head_dim = 128
        src_list = [torch.randn(1560, head_dim, device=device, dtype=torch.bfloat16)]
        total = 1560

        test_out = torch.empty(total, head_dim, device=device, dtype=torch.bfloat16)
        ptrs = torch.tensor([src_list[0].data_ptr(), 0], dtype=torch.int64, device=device)
        lens = torch.tensor([1560, 0], dtype=torch.int64, device=device)
        offsets = torch.tensor([0, 1560], dtype=torch.int64, device=device)

        scatter_copy(ptrs, lens, offsets, test_out, head_dim)
        torch.cuda.synchronize()

        assert torch.equal(src_list[0], test_out)

    def test_many_segments(self, device):
        """36 segments (typical Phase B workload)."""
        from headkv._scatter_ext import scatter_copy

        head_dim = 128
        torch.manual_seed(42)
        n_seg = 36
        lengths = [torch.randint(100, 2000, (1,)).item() for _ in range(n_seg)]
        src_list = [torch.randn(l, head_dim, device=device, dtype=torch.bfloat16) for l in lengths]
        total = sum(lengths)

        ref_out = torch.empty(total, head_dim, device=device, dtype=torch.bfloat16)
        torch.cat(src_list, dim=0, out=ref_out)

        test_out = torch.empty(total, head_dim, device=device, dtype=torch.bfloat16)
        ptrs = torch.tensor([t.data_ptr() for t in src_list], dtype=torch.int64, device=device)
        lens = torch.tensor(lengths, dtype=torch.int64, device=device)
        offsets = torch.zeros(n_seg, dtype=torch.int64, device=device)
        offsets[1:] = torch.cumsum(lens[:-1], 0)

        scatter_copy(ptrs, lens, offsets, test_out, head_dim)
        torch.cuda.synchronize()

        assert torch.equal(ref_out, test_out)

    def test_override_pos_time(self, device):
        """Test sync_t override on pos[:, 0]."""
        from headkv._scatter_ext import apply_pos_override

        total = 4680
        pos = torch.zeros(total, 3, dtype=torch.int64, device=device)
        pos[:, 0] = torch.arange(total, device=device) // 1560

        # Override first 1560 tokens to sync_t=5
        override_starts = torch.tensor([0], dtype=torch.int64, device=device)
        override_ends = torch.tensor([1560], dtype=torch.int64, device=device)
        override_vals = torch.tensor([5], dtype=torch.int64, device=device)

        apply_pos_override(pos, override_starts, override_ends, override_vals)
        torch.cuda.synchronize()

        assert (pos[:1560, 0] == 5).all()
        assert (pos[1560:, 0] != 5).any()

    def test_anchor_store_write_frames(self, device):
        """Frame-copy API writes selected frames into contiguous store slots."""
        from headkv._scatter_ext import anchor_store_write_frames

        frame_seqlen = 4
        head_dim = 8
        k_seq = torch.arange(3 * frame_seqlen * head_dim, device=device, dtype=torch.float32).reshape(
            3 * frame_seqlen, head_dim
        )
        v_seq = k_seq + 1000
        pos_seq = torch.zeros(3 * frame_seqlen, 3, device=device, dtype=torch.long)
        pos_seq[:, 0] = torch.arange(3 * frame_seqlen, device=device, dtype=torch.long) // frame_seqlen
        pos_seq[:, 1] = torch.arange(3 * frame_seqlen, device=device, dtype=torch.long) % frame_seqlen
        frame_desc = torch.tensor([[2, 0], [0, 3]], device=device, dtype=torch.long)

        store_k = torch.full((4, frame_seqlen, head_dim), -1, device=device, dtype=torch.float32)
        store_v = torch.full_like(store_k, -2)
        store_pos = torch.full((4, frame_seqlen, 3), -3, device=device, dtype=torch.long)

        anchor_store_write_frames(k_seq, v_seq, pos_seq, frame_desc, store_k, store_v, store_pos)
        torch.cuda.synchronize()

        assert torch.equal(store_k[0], k_seq[2 * frame_seqlen:3 * frame_seqlen])
        assert torch.equal(store_v[0], v_seq[2 * frame_seqlen:3 * frame_seqlen])
        assert torch.equal(store_pos[0], pos_seq[2 * frame_seqlen:3 * frame_seqlen])
        assert torch.equal(store_k[3], k_seq[:frame_seqlen])
        assert torch.equal(store_v[3], v_seq[:frame_seqlen])
        assert torch.equal(store_pos[3], pos_seq[:frame_seqlen])
        assert (store_k[1] == -1).all()

    @pytest.mark.parametrize("frame_mode", ["physical", "mapped"])
    @pytest.mark.parametrize("mapping_mode", ["none", "relative_clamp", "relative_softcap"])
    def test_refresh_readout_layout(self, device, frame_mode, mapping_mode):
        """CUDA refresh matches Python K/V/pos/frame-id materialization."""
        from headkv._scatter_ext import refresh_readout_layout

        head_dim = 8
        current_t = 10
        rel_max = 3
        soft_factor = 0.5
        src_k = [
            torch.randn(3, head_dim, device=device, dtype=torch.bfloat16),
            torch.randn(2, head_dim, device=device, dtype=torch.bfloat16),
        ]
        src_v = [
            torch.randn(3, head_dim, device=device, dtype=torch.bfloat16),
            torch.randn(2, head_dim, device=device, dtype=torch.bfloat16),
        ]
        src_pos = [
            torch.tensor([[1, 0, 0], [6, 0, 1], [8, 1, 0]], device=device, dtype=torch.long),
            torch.tensor([[2, 2, 0], [9, 2, 1]], device=device, dtype=torch.long),
        ]
        offsets_cpu = [1, 5]
        lengths_cpu = [3, 2]
        flags_cpu = [1, 4]
        if frame_mode == "physical":
            flags_cpu = [flag | 2 for flag in flags_cpu]
        dynamic_rope_t_cpu = [0, 7]
        total = 7

        out_k = torch.full((total, head_dim), -9, device=device, dtype=torch.bfloat16)
        out_v = torch.full((total, head_dim), -8, device=device, dtype=torch.bfloat16)
        out_pos = torch.full((total, 3), -7, device=device, dtype=torch.long)
        out_frame_ids = torch.full((total,), -6, device=device, dtype=torch.long)

        n_seg = len(src_k)
        refresh_readout_layout(
            src_k,
            src_v,
            src_pos,
            torch.empty(n_seg, device=device, dtype=torch.int64),
            torch.empty(n_seg, device=device, dtype=torch.int64),
            torch.empty(n_seg, device=device, dtype=torch.int64),
            torch.tensor(offsets_cpu, device=device, dtype=torch.int64),
            torch.tensor(lengths_cpu, device=device, dtype=torch.int64),
            torch.tensor(flags_cpu, device=device, dtype=torch.int64),
            torch.tensor(dynamic_rope_t_cpu, device=device, dtype=torch.int64),
            out_k,
            out_v,
            out_pos,
            out_frame_ids,
            head_dim,
            current_t,
            mapping_mode,
            rel_max,
            soft_factor,
        )
        torch.cuda.synchronize()

        ref_k = torch.full_like(out_k, -9)
        ref_v = torch.full_like(out_v, -8)
        ref_pos = torch.full_like(out_pos, -7)
        ref_frame_ids = torch.full_like(out_frame_ids, -6)
        for i, (offset, length) in enumerate(zip(offsets_cpu, lengths_cpu)):
            ref_k[offset:offset + length] = src_k[i]
            ref_v[offset:offset + length] = src_v[i]
            pos = src_pos[i].clone()
            if flags_cpu[i] & 4:
                pos[:, 0] = dynamic_rope_t_cpu[i]
            elif flags_cpu[i] & 1:
                pos = map_dynamic_pos_time(
                    pos,
                    current_t=current_t,
                    history_time_mapping_mode=mapping_mode,
                    history_relative_t_max=rel_max,
                    history_time_soft_factor=soft_factor,
                    inplace=True,
                )
            ref_pos[offset:offset + length] = pos
            if frame_mode == "physical":
                ref_frame_ids[offset:offset + length] = src_pos[i][:, 0]
            else:
                ref_frame_ids[offset:offset + length] = pos[:, 0]

        assert torch.equal(ref_k, out_k)
        assert torch.equal(ref_v, out_v)
        assert torch.equal(ref_pos, out_pos)
        assert torch.equal(ref_frame_ids, out_frame_ids)

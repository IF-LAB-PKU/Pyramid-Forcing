"""Reproduce scatter_copy crash when _set_dynamic_store reallocates storage (R1).

This test simulates the block 3 crash scenario:
1. Create tensors simulating per-head dynamic KV stores
2. Take views (like src_k_list entries) and capture data_ptr()
3. Call _set_dynamic_store (reallocates underlying storage)
4. Call scatter_copy with the OLD data_ptr() values → stale pointer dereference

Expected: FAIL on current code (illegal memory access or wrong data).
After M2-impl fix (pin tensors): PASS.
"""
import pytest
import torch

pytestmark = pytest.mark.gpu


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


def _simulate_set_dynamic_store(store_list, idx, new_length, head_dim, device, dtype):
    """Simulate _set_dynamic_store: reallocate storage for a head."""
    new_tensor = torch.randn(new_length, head_dim, device=device, dtype=dtype)
    store_list[idx] = new_tensor
    return new_tensor


class TestScatterCopyReproBlock3:
    """Reproduce the block 3 illegal memory access caused by storage reallocation."""

    def test_stale_ptr_after_realloc(self, device):
        """Scatter with stale data_ptr after _set_dynamic_store reallocation.

        This is the core R1 scenario: between capturing data_ptr() and
        launching the scatter kernel, the underlying storage gets reallocated.
        """
        from pyramidkv._scatter_ext import scatter_copy, scatter_available

        if not scatter_available():
            pytest.skip("CUDA scatter extension not available")

        head_dim = 128
        dtype = torch.bfloat16
        n_heads = 12

        # Phase 1: Create initial dynamic stores (simulating block 1-2 state)
        dyn_stores = [
            torch.randn(1560, head_dim, device=device, dtype=dtype)
            for _ in range(n_heads)
        ]

        # Phase 2: Capture views and data_ptr (simulating src_k_list construction)
        src_views = [store[:1560] for store in dyn_stores]
        old_ptrs = [v.data_ptr() for v in src_views]

        # Phase 3: Simulate block 3 — _set_dynamic_store reallocates some heads
        # This is what happens when update() grows the dynamic store
        for i in range(0, n_heads, 3):
            _simulate_set_dynamic_store(
                dyn_stores, i, 3120, head_dim, device, dtype
            )

        # Phase 4: Build scatter descriptors with OLD pointers (the bug)
        lengths = [1560] * n_heads
        total = sum(lengths)
        ptrs = torch.tensor(old_ptrs, dtype=torch.int64, device=device)
        lens = torch.tensor(lengths, dtype=torch.int64, device=device)
        offsets = torch.zeros(n_heads, dtype=torch.int64, device=device)
        offsets[1:] = torch.cumsum(lens[:-1], 0)

        dst = torch.empty(total, head_dim, device=device, dtype=dtype)

        # This should crash or produce wrong results with stale pointers
        scatter_copy(ptrs, lens, offsets, dst, head_dim)
        torch.cuda.synchronize()

        # Verify correctness: compare against what the CURRENT views hold
        # After realloc, old_ptrs point to freed memory → data won't match
        ref = torch.cat(src_views, dim=0)
        assert torch.equal(dst, ref), (
            f"Scatter output doesn't match reference. "
            f"Max diff: {(dst - ref).abs().max().item():.6f}. "
            f"This indicates stale pointer dereference (R1 bug)."
        )

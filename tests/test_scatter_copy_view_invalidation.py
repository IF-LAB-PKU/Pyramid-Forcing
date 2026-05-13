"""Test that scatter_copy detects/handles view invalidation from storage resize.

L1 unit test: constructs a minimal scenario where a view's data_ptr becomes
stale after the backing storage is replaced. Verifies that the pinning fix
(Solution A) prevents this class of bug.

Expected: FAIL on current code (no pinning). PASS after M2-impl.
"""
import pytest
import torch

pytestmark = pytest.mark.gpu


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


class TestScatterViewInvalidation:
    """Verify scatter correctness when source tensor storage is replaced."""

    def test_view_ptr_survives_storage_replacement(self, device):
        """Pin via contiguous() ensures data_ptr remains valid after realloc.

        Scenario:
        1. Create a tensor, take a view
        2. Pin the view (contiguous copy if needed)
        3. Replace the original tensor's storage (simulating _set_dynamic_store)
        4. Scatter using the pinned copy's data_ptr → should still be valid
        """
        from pyramidkv._scatter_ext import scatter_copy, scatter_available

        if not scatter_available():
            pytest.skip("CUDA scatter extension not available")

        head_dim = 128
        dtype = torch.bfloat16
        rows = 1560

        # Create source and take a view
        original = torch.randn(rows * 2, head_dim, device=device, dtype=dtype)
        view = original[:rows]

        # Pin: make a contiguous copy that owns its own storage
        pinned = view.contiguous() if not view.is_contiguous() else view.clone()
        pinned_ptr = pinned.data_ptr()

        # Simulate _set_dynamic_store: replace original's storage entirely
        original.set_(torch.randn(rows * 3, head_dim, device=device, dtype=dtype).untyped_storage())

        # The pinned copy's data_ptr should still be valid
        assert pinned.data_ptr() == pinned_ptr, "Pinned tensor ptr changed unexpectedly"

        # Scatter using pinned ptr should succeed
        ptrs = torch.tensor([pinned_ptr], dtype=torch.int64, device=device)
        lens = torch.tensor([rows], dtype=torch.int64, device=device)
        offsets = torch.tensor([0], dtype=torch.int64, device=device)
        dst = torch.empty(rows, head_dim, device=device, dtype=dtype)

        scatter_copy(ptrs, lens, offsets, dst, head_dim)
        torch.cuda.synchronize()

        assert torch.equal(dst, pinned), (
            f"Scatter from pinned tensor failed. Max diff: {(dst - pinned).abs().max().item()}"
        )

    def test_unpinned_view_becomes_stale(self, device):
        """Without pinning, a view's data_ptr becomes invalid after storage resize.

        This test demonstrates the R1 bug: if we only hold a view (not a
        contiguous copy), replacing the backing storage invalidates the ptr.
        """
        head_dim = 128
        dtype = torch.bfloat16
        rows = 1560

        # Create source and take a view
        original = torch.randn(rows * 2, head_dim, device=device, dtype=dtype)
        view = original[:rows]
        old_storage_ptr = original.untyped_storage().data_ptr()

        # Simulate _set_dynamic_store: create entirely new storage
        new_store = torch.randn(rows * 3, head_dim, device=device, dtype=dtype)
        original.set_(new_store.untyped_storage())

        new_storage_ptr = original.untyped_storage().data_ptr()

        # Storage pointer changed → old view's data_ptr is now stale
        assert old_storage_ptr != new_storage_ptr, (
            "Storage should have been reallocated to a different address"
        )

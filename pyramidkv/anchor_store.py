"""Opt-in contiguous anchor storage for PyramidKV middle strategies."""
from __future__ import annotations

from dataclasses import dataclass
import os

import torch

from ._scatter_ext import anchor_store_write_frames


def contig_anchor_store_requested() -> bool:
    if os.environ.get("PYRAMIDKV_CPP_STRATEGY", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return False
    return os.environ.get("PYRAMIDKV_CONTIG_ANCHOR_STORE", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def contig_anchor_store_usable(k_seq: torch.Tensor, v_seq: torch.Tensor, pos_seq: torch.Tensor) -> bool:
    return (
        contig_anchor_store_requested()
        and k_seq.is_cuda
        and v_seq.is_cuda
        and pos_seq.is_cuda
        and k_seq.ndim == 2
        and v_seq.ndim == 2
        and pos_seq.ndim == 2
        and pos_seq.shape[1] == 3
    )


@dataclass(frozen=True)
class AnchorStoreRef:
    """Metadata reference to one frame stored in a contiguous per-sequence buffer."""

    t: int
    slot: int
    token_count: int


class ContiguousAnchorStore:
    """Per-strategy frame store backed by contiguous K/V/pos tensors.

    The store is intentionally small and strategy-local. Metadata stays on CPU,
    while frame payloads are written into fixed or growable CUDA buffers.
    """

    def __init__(self, capacity_frames: int):
        self.capacity_frames = int(capacity_frames)
        self._k: list[torch.Tensor | None] = []
        self._v: list[torch.Tensor | None] = []
        self._pos: list[torch.Tensor | None] = []
        self._next_slot: list[int] = []
        self._valid_count: list[int] = []

    def reset(self, num_seq: int) -> None:
        self._k = [None] * num_seq
        self._v = [None] * num_seq
        self._pos = [None] * num_seq
        self._next_slot = [0] * num_seq
        self._valid_count = [0] * num_seq

    def _capacity(self, idx: int) -> int:
        store = self._k[idx]
        return 0 if store is None else int(store.shape[0])

    def _ensure(
        self,
        idx: int,
        *,
        frame_seqlen: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        capacity = self._capacity(idx)
        if capacity > 0:
            if (
                self._k[idx].shape[1] == frame_seqlen
                and self._k[idx].shape[2] == head_dim
                and self._k[idx].device == device
                and self._k[idx].dtype == dtype
                and self._pos[idx].shape[1] == frame_seqlen
            ):
                return
            self._k[idx] = None
            self._v[idx] = None
            self._pos[idx] = None
            self._next_slot[idx] = 0
            self._valid_count[idx] = 0

        alloc = self.capacity_frames if self.capacity_frames > 0 else 16
        self._k[idx] = torch.empty((alloc, frame_seqlen, head_dim), device=device, dtype=dtype)
        self._v[idx] = torch.empty((alloc, frame_seqlen, head_dim), device=device, dtype=dtype)
        self._pos[idx] = torch.empty((alloc, frame_seqlen, 3), device=device, dtype=torch.long)

    def _grow(
        self,
        idx: int,
        *,
        frame_seqlen: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        old_cap = self._capacity(idx)
        new_cap = max(old_cap * 2, old_cap + 1, 16)
        new_k = torch.empty((new_cap, frame_seqlen, head_dim), device=device, dtype=dtype)
        new_v = torch.empty((new_cap, frame_seqlen, head_dim), device=device, dtype=dtype)
        new_pos = torch.empty((new_cap, frame_seqlen, 3), device=device, dtype=torch.long)
        if old_cap > 0:
            new_k[:old_cap] = self._k[idx]
            new_v[:old_cap] = self._v[idx]
            new_pos[:old_cap] = self._pos[idx]
        self._k[idx] = new_k
        self._v[idx] = new_v
        self._pos[idx] = new_pos

    def write_frame(
        self,
        idx: int,
        *,
        k_frame: torch.Tensor,
        v_frame: torch.Tensor,
        pos_frame: torch.Tensor,
        t: int,
        preferred_slot: int | None = None,
    ) -> AnchorStoreRef:
        frame_seqlen = int(k_frame.shape[0])
        head_dim = int(k_frame.shape[1])
        self._ensure(
            idx,
            frame_seqlen=frame_seqlen,
            head_dim=head_dim,
            device=k_frame.device,
            dtype=k_frame.dtype,
        )
        if preferred_slot is None:
            slot = self._next_slot[idx]
            if self.capacity_frames <= 0 and slot >= self._capacity(idx):
                self._grow(
                    idx,
                    frame_seqlen=frame_seqlen,
                    head_dim=head_dim,
                    device=k_frame.device,
                    dtype=k_frame.dtype,
                )
            self._next_slot[idx] = (slot + 1) % self._capacity(idx) if self.capacity_frames > 0 else slot + 1
        else:
            slot = int(preferred_slot)
            if slot >= self._capacity(idx):
                while slot >= self._capacity(idx):
                    self._grow(
                        idx,
                        frame_seqlen=frame_seqlen,
                        head_dim=head_dim,
                        device=k_frame.device,
                        dtype=k_frame.dtype,
                    )

        self._k[idx][slot].copy_(k_frame)
        self._v[idx][slot].copy_(v_frame)
        pos_src = pos_frame if pos_frame.dtype == torch.long else pos_frame.to(dtype=torch.long)
        self._pos[idx][slot].copy_(pos_src)
        self._valid_count[idx] = min(self._capacity(idx), self._valid_count[idx] + 1)
        return AnchorStoreRef(t=int(t), slot=slot, token_count=frame_seqlen)

    def write_frames(
        self,
        idx: int,
        *,
        k_seq: torch.Tensor,
        v_seq: torch.Tensor,
        pos_seq: torch.Tensor,
        frame_seqlen: int,
        frames: list[tuple[int, int, int]],
    ) -> list[AnchorStoreRef]:
        """Write multiple source frames into selected slots.

        Args:
            frames: tuples of (source_frame_idx, destination_slot, t).
        """
        if not frames:
            return []
        head_dim = int(k_seq.shape[1])
        self._ensure(
            idx,
            frame_seqlen=frame_seqlen,
            head_dim=head_dim,
            device=k_seq.device,
            dtype=k_seq.dtype,
        )
        max_slot = max(int(slot) for _, slot, _ in frames)
        while max_slot >= self._capacity(idx):
            self._grow(
                idx,
                frame_seqlen=frame_seqlen,
                head_dim=head_dim,
                device=k_seq.device,
                dtype=k_seq.dtype,
            )

        k_src = k_seq if k_seq.is_contiguous() else k_seq.contiguous()
        v_src = v_seq if v_seq.is_contiguous() else v_seq.contiguous()
        pos_src = pos_seq if pos_seq.dtype == torch.long else pos_seq.to(dtype=torch.long)
        pos_src = pos_src if pos_src.is_contiguous() else pos_src.contiguous()
        desc = torch.tensor(
            [(int(src_frame), int(slot)) for src_frame, slot, _ in frames],
            dtype=torch.long,
            device=k_src.device,
        )
        try:
            anchor_store_write_frames(
                k_src,
                v_src,
                pos_src,
                desc,
                self._k[idx],
                self._v[idx],
                self._pos[idx],
            )
        except Exception:
            for src_frame, slot, _ in frames:
                start = int(src_frame) * frame_seqlen
                end = start + frame_seqlen
                self._k[idx][slot].copy_(k_src[start:end])
                self._v[idx][slot].copy_(v_src[start:end])
                self._pos[idx][slot].copy_(pos_src[start:end])

        self._valid_count[idx] = min(self._capacity(idx), max(self._valid_count[idx], len(frames)))
        return [
            AnchorStoreRef(t=int(t), slot=int(slot), token_count=frame_seqlen)
            for _, slot, t in frames
        ]

    def view(self, idx: int, ref: AnchorStoreRef) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._k[idx][ref.slot], self._v[idx][ref.slot], self._pos[idx][ref.slot]

    def anchor_count(self, idx: int) -> int:
        return int(self._valid_count[idx]) if idx < len(self._valid_count) else 0

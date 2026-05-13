"""StrideStrategy — evenly-spaced anchor selection.

Stores all frame anchors and returns every `interval`-th frame
between sink and recent windows.
"""
from __future__ import annotations

from collections import OrderedDict

import torch

from .anchor_store import AnchorStoreRef, ContiguousAnchorStore, contig_anchor_store_requested, contig_anchor_store_usable
from .base import CollectedAnchor, FrameAnchor


class StrideStrategy:
    """Stride middle strategy.

    Args:
        interval: Select every `interval`-th frame (e.g. 6).
        capacity: Max number of stride anchors to keep per sequence.
            -1 means unlimited (store all aligned frames).
            When exceeded, the oldest (smallest t) anchor is evicted (FIFO).
        dynamic_rope: Whether to remap anchor time for RoPE.
    """

    def __init__(self, interval: int = 6, capacity: int = -1, dynamic_rope: bool = True):
        self.interval = max(1, int(interval))
        self.capacity = int(capacity)
        self.dynamic_rope = bool(dynamic_rope)
        # per (batch*head) -> OrderedDict[t -> FrameAnchor]
        self._anchors: list[OrderedDict[int, FrameAnchor | AnchorStoreRef]] = []
        self._anchor_store = ContiguousAnchorStore(self.capacity)
        self._anchor_store_stats = {
            "anchor_store_update_count": 0.0,
            "anchor_store_collect_count": 0.0,
            "anchor_store_fallback_count": 0.0,
        }

    def reset(self, num_seq: int) -> None:
        self._anchors = [OrderedDict() for _ in range(num_seq)]
        self._anchor_store.reset(num_seq)

    def pop_anchor_store_stats(self) -> dict[str, float]:
        stats = dict(self._anchor_store_stats)
        for key in self._anchor_store_stats:
            self._anchor_store_stats[key] = 0.0
        return stats

    def update(
        self,
        idx: int,
        k_seq: torch.Tensor,
        v_seq: torch.Tensor,
        pos_seq: torch.Tensor,
        frame_seqlen: int,
        current_t: int,
        t_vals: list[int] | None = None,
    ) -> None:
        if frame_seqlen <= 0 or k_seq.shape[0] < frame_seqlen:
            return
        if k_seq.shape[0] % frame_seqlen != 0:
            return

        store = self._anchors[idx]
        num_frames = k_seq.shape[0] // frame_seqlen
        if t_vals is None:
            t_start = int(current_t)
            t_vals = list(range(t_start, t_start + num_frames))
        use_store = contig_anchor_store_usable(k_seq, v_seq, pos_seq)
        if contig_anchor_store_requested() and not use_store:
            self._anchor_store_stats["anchor_store_fallback_count"] += 1.0

        store_frames: list[tuple[int, int, int]] = []
        for frame_idx in range(num_frames):
            t_val = t_vals[frame_idx]
            # Only store frames aligned to interval
            if t_val % self.interval != 0:
                continue
            preferred_slot = None
            if t_val in store:
                old_anchor = store[t_val]
                if isinstance(old_anchor, AnchorStoreRef):
                    preferred_slot = old_anchor.slot
                del store[t_val]
            elif use_store and self.capacity > 0 and len(store) + len(store_frames) >= self.capacity:
                if store:
                    _, old_anchor = store.popitem(last=False)
                    if isinstance(old_anchor, AnchorStoreRef):
                        preferred_slot = old_anchor.slot
                elif store_frames:
                    _, preferred_slot, _ = store_frames.pop(0)
            elif not use_store and self.capacity > 0 and len(store) >= self.capacity:
                _, old_anchor = store.popitem(last=False)
                if isinstance(old_anchor, AnchorStoreRef):
                    preferred_slot = old_anchor.slot

            if use_store:
                if preferred_slot is None:
                    preferred_slot = len(store) + len(store_frames)
                store_frames.append((frame_idx, preferred_slot, t_val))
            else:
                start = frame_idx * frame_seqlen
                end = start + frame_seqlen
                store[t_val] = FrameAnchor(
                    k=k_seq[start:end].clone(),
                    v=v_seq[start:end].clone(),
                    pos=pos_seq[start:end].clone(),
                    t=t_val,
                )
                # FIFO eviction: drop oldest anchor when over capacity
                if self.capacity > 0 and len(store) > self.capacity:
                    store.popitem(last=False)
        if use_store and store_frames:
            refs = self._anchor_store.write_frames(
                idx,
                k_seq=k_seq,
                v_seq=v_seq,
                pos_seq=pos_seq,
                frame_seqlen=frame_seqlen,
                frames=store_frames,
            )
            for ref in refs:
                store[ref.t] = ref
            self._anchor_store_stats["anchor_store_update_count"] += float(len(refs))

    def collect(
        self,
        idx: int,
        current_t: int,
        recent_min_t: int,
        sink_max_t: int,
    ) -> list[CollectedAnchor]:
        result: list[CollectedAnchor] = []
        store_hits = 0
        for t_val, anchor in sorted(self._anchors[idx].items()):
            if t_val <= sink_max_t:
                continue
            if t_val >= recent_min_t:
                continue
            if isinstance(anchor, AnchorStoreRef):
                anchor_k, anchor_v, anchor_pos = self._anchor_store.view(idx, anchor)
                source_kind = "anchor_store"
                store_hits += 1
            else:
                anchor_k, anchor_v, anchor_pos = anchor.k, anchor.v, anchor.pos
                source_kind = "tensor"
            result.append(CollectedAnchor(
                kind="frame",
                t=anchor.t,
                dynamic_rope=self.dynamic_rope,
                k=anchor_k,
                v=anchor_v,
                pos=anchor_pos,
                token_count=anchor_k.shape[0],
                source_kind=source_kind,
            ))
        if store_hits > 0:
            self._anchor_store_stats["anchor_store_collect_count"] += 1.0
        return result

    @staticmethod
    def select_frame_ids(num_frames: int, interval: int = 6) -> list[int]:
        """Return frame indices for stride selection (static utility)."""
        if num_frames <= 0:
            return []
        return sorted(set(f for f in range(0, num_frames, interval)))

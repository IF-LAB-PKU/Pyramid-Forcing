"""LagStrategy — fixed-offset anchor retrieval.

Maintains a sorted history of frame anchors. On collect, returns the
anchor at t = current_t - offset for each configured offset.
"""
from __future__ import annotations

from collections import OrderedDict

import torch

from .anchor_store import AnchorStoreRef, ContiguousAnchorStore, contig_anchor_store_requested, contig_anchor_store_usable
from .base import CollectedAnchor, FrameAnchor


class LagStrategy:
    """Lag middle strategy.

    Args:
        offsets: List of positive lag offsets (e.g. [6, 12]).
        history_frames: Max number of frame anchors to retain.
        dynamic_rope: Whether to remap anchor time for RoPE.
    """

    def __init__(
        self,
        offsets: list[int] | None = None,
        history_frames: int = 21,
        dynamic_rope: bool = False,
    ):
        raw_offsets = offsets if offsets is not None else [6]
        self.offsets = sorted(set(o for o in raw_offsets if o > 0))
        self.history_frames = max(1, int(history_frames))
        if self.offsets:
            self.history_frames = max(self.history_frames, max(self.offsets) + 1)
        self.dynamic_rope = bool(dynamic_rope)
        # per (batch*head) -> OrderedDict[t -> FrameAnchor]
        self._anchors: list[OrderedDict[int, FrameAnchor | AnchorStoreRef]] = []
        self._free_slots: list[list[int]] = []
        self._anchor_store = ContiguousAnchorStore(self.history_frames)
        self._anchor_store_stats = {
            "anchor_store_update_count": 0.0,
            "anchor_store_collect_count": 0.0,
            "anchor_store_fallback_count": 0.0,
        }

    def reset(self, num_seq: int) -> None:
        self._anchors = [OrderedDict() for _ in range(num_seq)]
        self._free_slots = [[] for _ in range(num_seq)]
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

        anchors = self._anchors[idx]
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
            preferred_slot = None
            if t_val in anchors:
                old_anchor = anchors[t_val]
                if isinstance(old_anchor, AnchorStoreRef):
                    preferred_slot = old_anchor.slot
                del anchors[t_val]
            elif use_store and len(anchors) + len(store_frames) >= self.history_frames:
                if anchors:
                    _, old_anchor = anchors.popitem(last=False)
                    if isinstance(old_anchor, AnchorStoreRef):
                        preferred_slot = old_anchor.slot
                elif store_frames:
                    _, preferred_slot, _ = store_frames.pop(0)
            elif not use_store and len(anchors) >= self.history_frames:
                _, old_anchor = anchors.popitem(last=False)
                if isinstance(old_anchor, AnchorStoreRef):
                    preferred_slot = old_anchor.slot
            elif use_store and self._free_slots[idx]:
                preferred_slot = self._free_slots[idx].pop()

            if use_store:
                if preferred_slot is None:
                    preferred_slot = len(anchors) + len(store_frames)
                store_frames.append((frame_idx, preferred_slot, t_val))
            else:
                start = frame_idx * frame_seqlen
                end = start + frame_seqlen
                anchors[t_val] = FrameAnchor(
                    k=k_seq[start:end].clone(),
                    v=v_seq[start:end].clone(),
                    pos=pos_seq[start:end].clone(),
                    t=t_val,
                )
                while len(anchors) > self.history_frames:
                    anchors.popitem(last=False)
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
                anchors[ref.t] = ref
            self._anchor_store_stats["anchor_store_update_count"] += float(len(refs))

    def collect(
        self,
        idx: int,
        current_t: int,
        recent_min_t: int,
        sink_max_t: int,
    ) -> list[CollectedAnchor]:
        result: list[CollectedAnchor] = []
        for lag in self.offsets:
            target_t = current_t - lag
            if target_t < 0:
                continue
            if target_t <= sink_max_t:
                continue
            if target_t >= recent_min_t:
                continue
            anchor = self._anchors[idx].get(target_t)
            if anchor is not None:
                if isinstance(anchor, AnchorStoreRef):
                    anchor_k, anchor_v, anchor_pos = self._anchor_store.view(idx, anchor)
                    source_kind = "anchor_store"
                    self._anchor_store_stats["anchor_store_collect_count"] += 1.0
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
        return result

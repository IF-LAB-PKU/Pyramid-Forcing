"""CyclicStrategy — phase-bucket anchor storage.

Stores frame anchors in buckets indexed by t mod period.
On collect, returns anchors from the current-phase bucket.
"""
from __future__ import annotations

from collections import deque

import torch

from .anchor_store import AnchorStoreRef, ContiguousAnchorStore, contig_anchor_store_requested, contig_anchor_store_usable
from .base import CollectedAnchor, FrameAnchor


class CyclicStrategy:
    """Cyclic (phase-bucket) middle strategy.

    Args:
        period: Phase period (e.g. 6 for 6-frame cycle).
        bucket_cap: Max anchors per phase bucket.
        dynamic_rope: Whether to remap anchor time to current sync_t.
    """

    def __init__(self, period: int = 6, bucket_cap: int = 1, dynamic_rope: bool = True):
        self.period = max(1, int(period))
        self.bucket_cap = max(1, int(bucket_cap))
        self.dynamic_rope = bool(dynamic_rope)
        # per (batch*head) -> per phase bucket -> deque of FrameAnchor
        self._buckets: list[list[deque[FrameAnchor | AnchorStoreRef]]] = []
        self._phase_cursors: list[list[int]] = []
        self._anchor_store = ContiguousAnchorStore(self.period * self.bucket_cap)
        self._anchor_store_stats = {
            "anchor_store_update_count": 0.0,
            "anchor_store_collect_count": 0.0,
            "anchor_store_fallback_count": 0.0,
        }

    def reset(self, num_seq: int) -> None:
        self._buckets = [
            [deque(maxlen=self.bucket_cap) for _ in range(self.period)] for _ in range(num_seq)
        ]
        self._phase_cursors = [[0 for _ in range(self.period)] for _ in range(num_seq)]
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

        num_frames = k_seq.shape[0] // frame_seqlen
        if t_vals is None:
            t_start = int(current_t)
            t_vals = list(range(t_start, t_start + num_frames))
        use_store = contig_anchor_store_usable(k_seq, v_seq, pos_seq)
        if contig_anchor_store_requested() and not use_store:
            self._anchor_store_stats["anchor_store_fallback_count"] += 1.0

        store_frames: list[tuple[int, int, int, int]] = []
        for frame_idx in range(num_frames):
            t_val = t_vals[frame_idx]
            phase = t_val % self.period
            bucket = self._buckets[idx][phase]
            if use_store:
                cursor = self._phase_cursors[idx][phase]
                slot = phase * self.bucket_cap + cursor
                self._phase_cursors[idx][phase] = (cursor + 1) % self.bucket_cap
                store_frames.append((frame_idx, slot, t_val, phase))
            else:
                start = frame_idx * frame_seqlen
                end = start + frame_seqlen
                bucket.append(FrameAnchor(
                    k=k_seq[start:end].clone(),
                    v=v_seq[start:end].clone(),
                    pos=pos_seq[start:end].clone(),
                    t=t_val,
                ))
        if use_store and store_frames:
            refs = self._anchor_store.write_frames(
                idx,
                k_seq=k_seq,
                v_seq=v_seq,
                pos_seq=pos_seq,
                frame_seqlen=frame_seqlen,
                frames=[(frame_idx, slot, t_val) for frame_idx, slot, t_val, _phase in store_frames],
            )
            for ref, (_frame_idx, _slot, _t_val, phase) in zip(refs, store_frames):
                self._buckets[idx][phase].append(ref)
            self._anchor_store_stats["anchor_store_update_count"] += float(len(refs))

    def collect(
        self,
        idx: int,
        current_t: int,
        recent_min_t: int,
        sink_max_t: int,
    ) -> list[CollectedAnchor]:
        phase_idx = current_t % self.period
        result: list[CollectedAnchor] = []
        store_hits = 0
        for anchor in self._buckets[idx][phase_idx]:
            t = anchor.t
            # Skip sink overlap (t=0 when sink exists)
            if t <= sink_max_t:
                continue
            # Skip recent overlap
            if t >= recent_min_t:
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

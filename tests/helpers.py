"""Shared test helpers for pyramidkv strategy and composition tests."""
from __future__ import annotations

import torch


def make_anchor_data(t: int, frame_seqlen: int = 4, head_dim: int = 8):
    """Create (k, v, pos) tensors simulating 1 frame at time t.

    pos has shape (frame_seqlen, 3) with pos[:, 0] = t (time dim).
    k, v have shape (frame_seqlen, head_dim).
    """
    k = torch.randn(frame_seqlen, head_dim)
    v = torch.randn(frame_seqlen, head_dim)
    pos = torch.zeros(frame_seqlen, 3)
    pos[:, 0] = t  # time dimension
    pos[:, 1] = torch.arange(frame_seqlen)  # y
    pos[:, 2] = 0  # x
    return k, v, pos


def make_multi_frame_input(t_values: list[int], frame_seqlen: int = 4, head_dim: int = 8):
    """Concatenate multiple frames into a single (k, v, pos) input."""
    ks, vs, ps = [], [], []
    for t in t_values:
        k, v, pos = make_anchor_data(t, frame_seqlen, head_dim)
        ks.append(k)
        vs.append(v)
        ps.append(pos)
    return torch.cat(ks, dim=0), torch.cat(vs, dim=0), torch.cat(ps, dim=0)

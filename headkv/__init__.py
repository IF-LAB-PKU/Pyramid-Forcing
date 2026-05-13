"""HeadKV frame selection strategies.

This package implements the [sink ... middle ... recent] architecture
for per-head KV cache management. Three middle strategies are available
and can be combined (union):

- CyclicStrategy: t mod T phase-bucket anchors
- LagStrategy: fixed-offset t-k anchors
- StrideStrategy: every k-th frame anchors
- MergeStrategy: spatiotemporal patch-block merging

HeadComposition ties sink_frames + middle strategies + recent_frames together
for each attention head. The factory module builds compositions from YAML config.
"""

from .base import FrameAnchor, HeadComposition, MiddleStrategy
from .cyclic import CyclicStrategy
from .lag import LagStrategy
from .stride import StrideStrategy
from .merge import MergeStrategy
from .recent import RecentStrategy
from .factory import (
    HEAD_LABEL_MAP,
    build_compositions,
    load_head_labels,
)
from .config import HeadKVConfig
from .cache import HeadKVCache
from .adaptive_cache import AdaptiveKVCache

__all__ = [
    "FrameAnchor",
    "HeadComposition",
    "MiddleStrategy",
    "CyclicStrategy",
    "LagStrategy",
    "StrideStrategy",
    "MergeStrategy",
    "RecentStrategy",
    "HEAD_LABEL_MAP",
    "build_compositions",
    "load_head_labels",
    "HeadKVConfig",
    "HeadKVCache",
    "AdaptiveKVCache",
]

import torch
from typing import List, Optional, Tuple

from .config import PyramidKVConfig


class PyramidKVCache:
    """Per-(layer, batch) sliding-window KV cache with per-head capacities.

    Each ``PyramidKVCache`` instance owns a flattened KV state for one transformer
    layer of one batch element. The cache is split into:

      * a **static** prefix that survives every step (used by the I2V context
        protection path to preserve first-frame KVs), and
      * a **dynamic** suffix that follows the active sliding window for each
        head (capacity comes from ``config.get_layer_capacities(layer_idx)``).

    The class focuses on storage management — appending new K/V, truncating to
    the per-head sliding window, and exposing a flat ragged ``(k_flat, v_flat,
    cu_seqlens_k)`` view that ``flash_attn_varlen_func`` consumes directly.
    For per-head adaptive strategies (sink + middle + recent composition,
    cyclic / stride / lag anchors, dynamic RoPE remapping) use the subclass
    :class:`AdaptiveKVCache`.

    Args:
        config: :class:`PyramidKVConfig` carrying per-head capacities and the
            full strategy/labels metadata for this run.
        batch_size: Number of independent sequences in this layer's cache
            (usually 1 for inference, > 1 for CFG / training).
        num_heads: Number of attention heads in this layer.
        head_dim: Per-head dimension (``D`` in ``[B, L, H, D]``).
        layer_idx: 0-based layer index used to look up per-layer capacities.
        is_i2v: If True, reserve a static prefix for the I2V context frames.
        context_len: Number of frames in the static prefix when ``is_i2v``.
        frame_seq_length: Tokens per frame. Falls back to
            ``config.frame_seq_length`` if not provided.
        prompt_value_cache_enabled: Reserved hook for caching cross-attention
            value projections; currently a no-op flag.
    """

    def __init__(
        self,
        config: PyramidKVConfig,
        batch_size: int,
        num_heads: int,
        head_dim: int,
        layer_idx: int,
        is_i2v: bool = False,
        context_len: int = 0,
        frame_seq_length: Optional[int] = None,
        prompt_value_cache_enabled: bool = False,
    ):
        self.layer_idx = layer_idx
        self.capacities = config.get_layer_capacities(layer_idx)
        self.num_heads = num_heads
        self.batch_size = batch_size
        self.head_dim = head_dim

        self.is_i2v = is_i2v
        self.context_len = context_len

        # 每帧的 token 数：优先使用显式参数，其次使用 config 中的值
        self.frame_seq_length = frame_seq_length if frame_seq_length is not None else getattr(config, "frame_seq_length", None)

        # Per-layer head compositions (or legacy policies). If absent, fallback to sliding-window.
        self.compositions_row = None
        if getattr(config, "compositions", None) is not None and 0 <= layer_idx < len(config.compositions):
            self.compositions_row = config.compositions[layer_idx]
        # Legacy alias for backward compatibility (used by AdaptiveKVCache._stable_strategy_kind)
        self.policies_row = self.compositions_row
        if hasattr(config, "get_layer_drop_mask"):
            self.drop_head_mask = torch.tensor(
                config.get_layer_drop_mask(layer_idx), dtype=torch.bool
            )
        else:
            self.drop_head_mask = torch.zeros(num_heads, dtype=torch.bool)
        self._any_drop: bool = bool(self.drop_head_mask.any())
        if hasattr(config, "get_layer_soft_ablate_mask"):
            self.soft_ablate_head_mask = torch.tensor(
                config.get_layer_soft_ablate_mask(layer_idx), dtype=torch.bool
            )
        else:
            self.soft_ablate_head_mask = torch.zeros(num_heads, dtype=torch.bool)
        if hasattr(config, "get_layer_af_groups"):
            self.af_group_row = config.get_layer_af_groups(layer_idx)
        else:
            self.af_group_row = [""] * num_heads
        # Soft ablation runtime knobs (can be overridden by pipeline args)
        self.soft_ablate_region = "none"
        self.soft_ablate_scale = 1.0
        self.prompt_value_cache_enabled = bool(prompt_value_cache_enabled)

        # Internal storage: List of B * H tensors
        self.static_k: List[Optional[torch.Tensor]] = [None] * (batch_size * num_heads)
        self.static_v: List[Optional[torch.Tensor]] = [None] * (batch_size * num_heads)

        self.dynamic_k: List[Optional[torch.Tensor]] = [None] * (batch_size * num_heads)
        self.dynamic_v: List[Optional[torch.Tensor]] = [None] * (batch_size * num_heads)
        self.global_end_index: List[int] = [0] * batch_size

    def update(
        self,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
        current_start: Optional[int] = None,
        **kwargs,
    ):
        """
        new_k, new_v: [B, L_new, H, D]
        """
        B, L_new, H, D = new_k.shape
        assert B == self.batch_size and H == self.num_heads and D == self.head_dim
        current_end = None
        if current_start is not None:
            current_end = current_start + L_new

        # [B, L, H, D] -> [B, H, L, D] -> [B*H, L, D]
        new_k_flat = new_k.transpose(1, 2).reshape(B * H, L_new, D)
        new_v_flat = new_v.transpose(1, 2).reshape(B * H, L_new, D)

        for i in range(B * H):
            batch_idx = i // H
            head_idx = i % H
            full_cap = self.capacities[head_idx]

            # I2V Context Initialization
            if self.is_i2v and self.static_k[i] is None:
                if L_new >= self.context_len:
                    # Capture context
                    self.static_k[i] = new_k_flat[i, :self.context_len].detach().clone()
                    self.static_v[i] = new_v_flat[i, :self.context_len].detach().clone()

                    # Remaining part to dynamic
                    if L_new > self.context_len:
                        self.dynamic_k[i] = new_k_flat[i, self.context_len:]
                        self.dynamic_v[i] = new_v_flat[i, self.context_len:]
                    else:
                        self.dynamic_k[i] = torch.empty(0, D, device=new_k.device, dtype=new_k.dtype)
                        self.dynamic_v[i] = torch.empty(0, D, device=new_v.device, dtype=new_v.dtype)
                    continue

            # Normal Dynamic Update
            curr_dyn_k = self.dynamic_k[i]
            curr_dyn_v = self.dynamic_v[i]
            overwrite = False
            if current_end is not None:
                overwrite = current_end <= self.global_end_index[batch_idx]
            # 动态容量：对 T2V 为 full_cap；对 I2V 需要扣除上下文长度
            dyn_cap = full_cap
            if self.is_i2v:
                dyn_cap = max(0, full_cap - self.context_len)

            head_impl = None
            if self.compositions_row is not None and 0 <= head_idx < len(self.compositions_row):
                head_impl = self.compositions_row[head_idx]

            # Non-adaptive path: simple sliding window for all head types
            if curr_dyn_k is None:
                k_merged = new_k_flat[i]
                v_merged = new_v_flat[i]
            elif overwrite:
                if curr_dyn_k.shape[0] >= L_new:
                    k_merged = torch.cat([curr_dyn_k[:-L_new], new_k_flat[i]], dim=0)
                    v_merged = torch.cat([curr_dyn_v[:-L_new], new_v_flat[i]], dim=0)
                else:
                    k_merged = new_k_flat[i]
                    v_merged = new_v_flat[i]
            else:
                k_merged = torch.cat([curr_dyn_k, new_k_flat[i]], dim=0)
                v_merged = torch.cat([curr_dyn_v, new_v_flat[i]], dim=0)

            if dyn_cap > 0 and k_merged.shape[0] > dyn_cap:
                k_merged = k_merged[-dyn_cap:]
                v_merged = v_merged[-dyn_cap:]

            self.dynamic_k[i] = k_merged
            self.dynamic_v[i] = v_merged
        if current_end is not None:
            for b in range(self.batch_size):
                self.global_end_index[b] = current_end

    def reset(self):
        self.static_k = [None] * (self.batch_size * self.num_heads)
        self.static_v = [None] * (self.batch_size * self.num_heads)
        self.dynamic_k = [None] * (self.batch_size * self.num_heads)
        self.dynamic_v = [None] * (self.batch_size * self.num_heads)
        self.global_end_index = [0] * self.batch_size

    def get_flat_kv(self, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        total_k_list = []
        total_v_list = []
        lengths = []

        for i in range(self.batch_size * self.num_heads):
            stat_k = self.static_k[i]
            dyn_k = self.dynamic_k[i]
            stat_v = self.static_v[i]
            dyn_v = self.dynamic_v[i]

            parts_k = []
            parts_v = []

            if stat_k is not None:
                parts_k.append(stat_k)
                parts_v.append(stat_v)
            if dyn_k is not None:
                parts_k.append(dyn_k)
                parts_v.append(dyn_v)

            if len(parts_k) == 0:
                # Fallback for empty cache (should rarely happen after first step)
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                t_k = torch.empty(0, self.head_dim, device=device)
                t_v = torch.empty(0, self.head_dim, device=device)
            elif len(parts_k) == 1:
                t_k = parts_k[0]
                t_v = parts_v[0]
            else:
                t_k = torch.cat(parts_k, dim=0)
                t_v = torch.cat(parts_v, dim=0)

            total_k_list.append(t_k)
            total_v_list.append(t_v)
            lengths.append(t_k.shape[0])

        k_flat = torch.cat(total_k_list, dim=0)
        v_flat = torch.cat(total_v_list, dim=0)

        cu_seqlens_k = torch.tensor([0] + lengths, dtype=torch.int32, device=k_flat.device).cumsum(0, dtype=torch.int32)
        max_seqlen_k = max(lengths) if lengths else 0

        return k_flat, v_flat, cu_seqlens_k, max_seqlen_k

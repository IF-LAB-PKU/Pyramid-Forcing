# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
import gc

try:
    import flash_attn_interface

    def is_hopper_gpu():
        if not torch.cuda.is_available():
            return False
        device_name = torch.cuda.get_device_name(0).lower()
        return any(name in device_name for name in ("h100", "h200", "hopper"))
    FLASH_ATTN_3_AVAILABLE = is_hopper_gpu()
except (ModuleNotFoundError, ImportError, OSError):
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except (ModuleNotFoundError, ImportError, OSError):
    FLASH_ATTN_2_AVAILABLE = False

# FLASH_ATTN_3_AVAILABLE = False

import warnings

from .capture import (
    FRAME_ATTENTION_CAPTURE,
    ATTENTION_WEIGHT_CAPTURE,
    compute_frame_attention_metrics_single_sequence,
)


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    def _build_region_mask(frame_ids: torch.Tensor, sync_t: int, region: str) -> torch.Tensor:
        """
        Build a key-frame selection mask for soft ablation.
        Supported tokens (can be combined by '+', e.g. 'sink1+recent4'):
          - sink1
          - recentN / recent3 / recent4
          - lagN  (exact lag frame: t-N)
        """
        region = str(region).strip().lower()
        if region in {"", "none", "off"}:
            return torch.zeros_like(frame_ids, dtype=torch.bool)

        def _token_mask(token: str) -> torch.Tensor:
            token = token.strip().lower()
            if token == "sink1":
                return frame_ids == 0
            if token.startswith("recent"):
                n_str = token[len("recent"):]
                n = int(n_str) if n_str.isdigit() else 0
                if n <= 0:
                    return torch.zeros_like(frame_ids, dtype=torch.bool)
                low = max(0, sync_t - (n - 1))
                return (frame_ids >= low) & (frame_ids <= sync_t)
            if token.startswith("lag"):
                lag_str = token[len("lag"):]
                lag = int(lag_str) if lag_str.isdigit() else -1
                if lag < 0:
                    return torch.zeros_like(frame_ids, dtype=torch.bool)
                target = max(0, sync_t - lag)
                return frame_ids == target
            return torch.zeros_like(frame_ids, dtype=torch.bool)

        out = torch.zeros_like(frame_ids, dtype=torch.bool)
        for tok in region.split("+"):
            if tok.strip():
                out |= _token_mask(tok)
        return out

    def _apply_soft_ablate_to_k_flat(
        k_flat_chunk: torch.Tensor,
        cu_seqlens_k_chunk: torch.Tensor,
        k_frame_ids_flat: torch.Tensor | None,
        chunk_start_token: int,
    ) -> torch.Tensor:
        """
        Soft ablation by scaling K vectors in selected frame regions for selected heads.
        Effectively scales QK^T logits before softmax for those regions.
        """
        if k_frame_ids_flat is None or frame_seqlen is None or frame_seqlen <= 0:
            return k_flat_chunk

        raw_mask = getattr(kv_cache, "soft_ablate_head_mask", None)
        if raw_mask is None:
            return k_flat_chunk
        soft_head_mask = torch.as_tensor(raw_mask, dtype=torch.bool, device=k_flat_chunk.device)
        if soft_head_mask.numel() != h or not torch.any(soft_head_mask):
            return k_flat_chunk

        region = str(getattr(kv_cache, "soft_ablate_region", "none"))
        if region.strip().lower() in {"", "none", "off"}:
            return k_flat_chunk

        scale = float(getattr(kv_cache, "soft_ablate_scale", 1.0))
        if scale >= 0.9999:
            return k_flat_chunk
        if scale < 0.0:
            scale = 0.0

        sync_t = int(chunk_start_token // frame_seqlen)
        for b_idx in range(b):
            for h_idx in range(h):
                if not bool(soft_head_mask[h_idx].item()):
                    continue
                seq_idx = b_idx * h + h_idx
                ks = int(cu_seqlens_k_chunk[seq_idx].item())
                ke = int(cu_seqlens_k_chunk[seq_idx + 1].item())
                if ke <= ks:
                    continue
                local_ids = k_frame_ids_flat[ks:ke].to(dtype=torch.long)
                select = _build_region_mask(local_ids, sync_t=sync_t, region=region)
                if not torch.any(select):
                    continue
                local_k = k_flat_chunk[ks:ke]
                local_k[select] = local_k[select] * scale
                k_flat_chunk[ks:ke] = local_k
        return k_flat_chunk

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
                device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    # apply attention
    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        # Note: dropout_p, window_size are not supported in FA3 now.
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic)[0].unflatten(0, (b, lq))
    else:
        assert FLASH_ATTN_2_AVAILABLE
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))

    # output
    return x.type(out_dtype)


def pyramidkv_attention(
    q,
    k,
    v,
    kv_cache,
    current_start=None,
    grid_sizes=None,
    freqs=None,
    start_frame=0,
    prompt_v=None,
    cache_update_mode="default",
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=True,
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
):
    """
    PyramidKV attention using FlashAttention varlen interface.

    Args:
        q, k, v: [B, L, H, D]
        kv_cache: PyramidKVCache
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256

    b, lq, h, d = q.shape
    out_dtype = q.dtype
    drop_head_mask = None
    _any_drop = getattr(kv_cache, "_any_drop", False)
    if _any_drop:
        raw_drop_mask = kv_cache.drop_head_mask
        drop_head_mask = torch.as_tensor(raw_drop_mask, dtype=torch.bool, device=q.device)
        if drop_head_mask.numel() != h:
            drop_head_mask = None
            _any_drop = False
    frame_seqlen = getattr(kv_cache, "_frame_seqlen", None) or getattr(kv_cache, "frame_seq_length", None)
    if frame_seqlen is None and grid_sizes is not None:
        frame_tokens = (grid_sizes[:, 1] * grid_sizes[:, 2]).to(torch.long)
        if torch.any(frame_tokens <= 0):
            raise ValueError(f"Invalid frame token sizes: {frame_tokens.tolist()}")
        if torch.unique(frame_tokens).numel() != 1:
            raise ValueError(f"Mixed frame token sizes in batch are not supported: {frame_tokens.tolist()}")
        frame_seqlen = int(frame_tokens[0].item())

    capture_obj = FRAME_ATTENTION_CAPTURE
    capture_enabled = capture_obj.enabled
    capture_this = capture_enabled and capture_obj.should_capture()
    capture_layer_idx = capture_obj.get_effective_layer_idx() if capture_this else None

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    def _build_region_mask(frame_ids: torch.Tensor, sync_t: int, region: str) -> torch.Tensor:
        """
        Build a key-frame selection mask for soft ablation.
        Supported tokens (can be combined by '+', e.g. 'sink1+recent4'):
          - sink1
          - recentN / recent3 / recent4
          - lagN  (exact lag frame: t-N)
        """
        region = str(region).strip().lower()
        if region in {"", "none", "off"}:
            return torch.zeros_like(frame_ids, dtype=torch.bool)

        def _token_mask(token: str) -> torch.Tensor:
            token = token.strip().lower()
            if token == "sink1":
                return frame_ids == 0
            if token.startswith("recent"):
                n_str = token[len("recent"):]
                n = int(n_str) if n_str.isdigit() else 0
                if n <= 0:
                    return torch.zeros_like(frame_ids, dtype=torch.bool)
                low = max(0, sync_t - (n - 1))
                return (frame_ids >= low) & (frame_ids <= sync_t)
            if token.startswith("lag"):
                lag_str = token[len("lag"):]
                lag = int(lag_str) if lag_str.isdigit() else -1
                if lag < 0:
                    return torch.zeros_like(frame_ids, dtype=torch.bool)
                target = max(0, sync_t - lag)
                return frame_ids == target
            return torch.zeros_like(frame_ids, dtype=torch.bool)

        out = torch.zeros_like(frame_ids, dtype=torch.bool)
        for tok in region.split("+"):
            if tok.strip():
                out |= _token_mask(tok)
        return out

    def _apply_soft_ablate_to_k_flat(
        k_flat_chunk: torch.Tensor,
        cu_seqlens_k_chunk: torch.Tensor,
        k_frame_ids_flat: torch.Tensor | None,
        chunk_start_token: int,
    ) -> torch.Tensor:
        """
        Soft ablation by scaling K vectors in selected frame regions for selected heads.
        Effectively scales QK^T logits before softmax for those regions.
        """
        if k_frame_ids_flat is None or frame_seqlen is None or frame_seqlen <= 0:
            return k_flat_chunk

        raw_mask = getattr(kv_cache, "soft_ablate_head_mask", None)
        if raw_mask is None:
            return k_flat_chunk
        soft_head_mask = torch.as_tensor(raw_mask, dtype=torch.bool, device=k_flat_chunk.device)
        if soft_head_mask.numel() != h or not torch.any(soft_head_mask):
            return k_flat_chunk

        region = str(getattr(kv_cache, "soft_ablate_region", "none"))
        if region.strip().lower() in {"", "none", "off"}:
            return k_flat_chunk

        scale = float(getattr(kv_cache, "soft_ablate_scale", 1.0))
        if scale >= 0.9999:
            return k_flat_chunk
        if scale < 0.0:
            scale = 0.0

        sync_t = int(chunk_start_token // frame_seqlen)
        for b_idx in range(b):
            for h_idx in range(h):
                if not bool(soft_head_mask[h_idx].item()):
                    continue
                seq_idx = b_idx * h + h_idx
                ks = int(cu_seqlens_k_chunk[seq_idx].item())
                ke = int(cu_seqlens_k_chunk[seq_idx + 1].item())
                if ke <= ks:
                    continue
                local_ids = k_frame_ids_flat[ks:ke].to(dtype=torch.long)
                select = _build_region_mask(local_ids, sync_t=sync_t, region=region)
                if not torch.any(select):
                    continue
                local_k = k_flat_chunk[ks:ke]
                local_k[select] = local_k[select] * scale
                k_flat_chunk[ks:ke] = local_k
        return k_flat_chunk

    def _capture_varlen_frame_attention(
        q_chunk: torch.Tensor,
        k_flat_chunk: torch.Tensor,
        cu_seqlens_k_chunk: torch.Tensor,
        chunk_start_token: int,
        k_frame_ids_flat: torch.Tensor | None,
    ) -> None:
        if not capture_this or capture_obj.on_frame_attention is None:
            return
        if frame_seqlen is None or frame_seqlen <= 0:
            return
        if k_frame_ids_flat is None:
            return

        capture_mode = capture_obj.capture_mode
        q_frame_ids_abs = (
            torch.arange(
                chunk_start_token,
                chunk_start_token + q_chunk.shape[1],
                device=q_chunk.device,
                dtype=torch.long,
            ) // frame_seqlen
        )
        q_unique = torch.unique(q_frame_ids_abs, sorted=True)
        if q_unique.numel() == 0:
            return
        k_max_frame = int(k_frame_ids_flat.max().item()) if k_frame_ids_flat.numel() > 0 else -1
        if k_max_frame < 0:
            return
        k_unique_global = torch.arange(k_max_frame + 1, device=q_chunk.device, dtype=torch.long)

        need_logits = capture_mode in {"logits_mean", "both"}
        need_prob = capture_mode in {"prob_mass", "both"}
        logits_global = (
            torch.zeros(h, q_unique.numel(), k_unique_global.numel(), dtype=torch.float32)
            if need_logits else None
        )
        prob_global = (
            torch.zeros(h, q_unique.numel(), k_unique_global.numel(), dtype=torch.float32)
            if need_prob else None
        )

        for b_idx in range(b):
            for h_idx in range(h):
                seq_idx = b_idx * h + h_idx
                ks = int(cu_seqlens_k_chunk[seq_idx].item())
                ke = int(cu_seqlens_k_chunk[seq_idx + 1].item())
                if ke <= ks:
                    continue
                q_seq = q_chunk[b_idx, :, h_idx, :]
                if q_scale is not None:
                    q_seq = q_seq * q_scale
                q_seq = q_seq.float()
                k_seq = k_flat_chunk[ks:ke].float()
                k_ids_seq = k_frame_ids_flat[ks:ke].to(torch.long)
                logits_local, prob_local, q_local_unique, k_local_unique = compute_frame_attention_metrics_single_sequence(
                    q_seq=q_seq,
                    k_seq=k_seq,
                    q_frame_ids=q_frame_ids_abs,
                    k_frame_ids=k_ids_seq,
                    softmax_scale=softmax_scale,
                    chunk_tokens=max(1, capture_obj.chunk_frames * frame_seqlen),
                    capture_mode=capture_mode,
                )
                q_index_map = {int(v.item()): i for i, v in enumerate(q_unique)}
                if need_logits and logits_local is not None:
                    for qi, qf in enumerate(q_local_unique):
                        qg = q_index_map[int(qf.item())]
                        for ki, kf in enumerate(k_local_unique):
                            logits_global[h_idx, qg, int(kf.item())] = logits_local[qi, ki].detach().cpu()
                if need_prob and prob_local is not None:
                    for qi, qf in enumerate(q_local_unique):
                        qg = q_index_map[int(qf.item())]
                        for ki, kf in enumerate(k_local_unique):
                            prob_global[h_idx, qg, int(kf.item())] = prob_local[qi, ki].detach().cpu()

        frame_attn = logits_global if capture_mode == "logits_mean" else prob_global
        capture_obj.on_frame_attention(
            layer_idx=capture_layer_idx,
            frame_attn=frame_attn,
            frame_attn_logits=logits_global,
            frame_attn_prob=prob_global,
            q_frames=int(q_unique.numel()),
            k_frames=int(k_unique_global.numel()),
            q_frame_indices=[int(v.item()) for v in q_unique],
            k_frame_indices=[int(v.item()) for v in k_unique_global],
            capture_mode=capture_mode,
            cache_update_mode=cache_update_mode,
        )

    kv_cache.update(
        k,
        v,
        current_start=current_start,
        grid_sizes=grid_sizes,
        freqs=freqs,
        start_frame=start_frame,
        prompt_v=prompt_v,
        cache_update_mode=cache_update_mode,
    )
    if fa_version is not None and fa_version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    def run_varlen(
        q_chunk: torch.Tensor,
        k_flat_chunk: torch.Tensor,
        v_flat_chunk: torch.Tensor,
        cu_seqlens_k_chunk: torch.Tensor,
        max_seqlen_k_chunk: int,
        cu_seqlens_q_override: torch.Tensor | None = None,
    ) -> torch.Tensor:
        lq_chunk = q_chunk.shape[1]
        q_flat_chunk = q_chunk.transpose(1, 2).reshape(b * h * lq_chunk, d)
        q_flat_chunk = half(q_flat_chunk).unsqueeze(1)
        k_flat_chunk = half(k_flat_chunk).unsqueeze(1)
        v_flat_chunk = half(v_flat_chunk).unsqueeze(1)

        if q_scale is not None:
            q_flat_chunk = q_flat_chunk * q_scale

        q_flat_chunk = q_flat_chunk.to(v_flat_chunk.dtype)
        k_flat_chunk = k_flat_chunk.to(v_flat_chunk.dtype)

        if cu_seqlens_q_override is not None:
            cu_seqlens_q_chunk = cu_seqlens_q_override
        else:
            cu_seqlens_q_chunk = torch.arange(
                0, (b * h + 1) * lq_chunk, step=lq_chunk, dtype=torch.int32, device=q.device
            )

        if (fa_version is None or fa_version == 3) and FLASH_ATTN_3_AVAILABLE:
            out_chunk = flash_attn_interface.flash_attn_varlen_func(
                q=q_flat_chunk,
                k=k_flat_chunk,
                v=v_flat_chunk,
                cu_seqlens_q=cu_seqlens_q_chunk,
                cu_seqlens_k=cu_seqlens_k_chunk,
                max_seqlen_q=lq_chunk,
                max_seqlen_k=max_seqlen_k_chunk,
                softmax_scale=softmax_scale,
                causal=causal,
                deterministic=deterministic
            )[0]
        else:
            assert FLASH_ATTN_2_AVAILABLE
            out_chunk = flash_attn.flash_attn_varlen_func(
                q=q_flat_chunk,
                k=k_flat_chunk,
                v=v_flat_chunk,
                cu_seqlens_q=cu_seqlens_q_chunk,
                cu_seqlens_k=cu_seqlens_k_chunk,
                max_seqlen_q=lq_chunk,
                max_seqlen_k=max_seqlen_k_chunk,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=(-1, -1),
                deterministic=deterministic
            )

        out = out_chunk.squeeze(1).reshape(b, h, lq_chunk, d).transpose(1, 2)
        if _any_drop:
            out[:, :, drop_head_mask, :] = 0
        return out

    try:
        use_decoupled = (
            getattr(kv_cache, "post_prune_rope", False)
            and getattr(kv_cache, "sink_grid_decoupling", False)
            and hasattr(kv_cache, "get_decoupled_flat_kv")
        )
        if use_decoupled:
            if freqs is None:
                raise ValueError("freqs is required when sink_grid_decoupling=True")
            if grid_sizes is None:
                raise ValueError("grid_sizes is required when sink_grid_decoupling=True")
            if frame_seqlen is None:
                raise ValueError("frame_seqlen is required when sink_grid_decoupling=True")
            if lq % frame_seqlen != 0:
                raise ValueError(f"q length {lq} must be divisible by frame_seqlen {frame_seqlen}.")

            out_buf = torch.empty(b, lq, h, d, device=q.device, dtype=out_dtype)
            base_start = int(current_start or 0)
            num_chunks = lq // frame_seqlen

            # Merged multi-chunk path: single FA call for all frame chunks
            _ablate_mask = getattr(kv_cache, "soft_ablate_head_mask", None)
            _has_ablate = _ablate_mask is not None and (
                isinstance(_ablate_mask, torch.Tensor) and _ablate_mask.any()
                if isinstance(_ablate_mask, torch.Tensor)
                else bool(_ablate_mask)
            )
            use_merged = (
                num_chunks > 1
                and hasattr(kv_cache, "get_decoupled_flat_kv_and_frames_multi")
                and not capture_this
                and not _has_ablate
            )
            if use_merged:
                current_starts = [base_start + c * frame_seqlen for c in range(num_chunks)]
                k_flat_m, v_flat_m, cu_seqlens_k_m, max_seqlen_k_m, _ = (
                    kv_cache.get_decoupled_flat_kv_and_frames_multi(
                        current_starts=current_starts,
                        grid_sizes=grid_sizes,
                        freqs=freqs,
                    )
                )
                num_seq = b * h
                # Q: [b, lq, h, d] → chunk-first layout for FA varlen
                # Target: [b*num_chunks*h*frame_seqlen, 1, d] ordered as
                #   chunk0_head0, chunk0_head1, ..., chunk1_head0, ...
                q_r = q.transpose(1, 2).reshape(b, h, num_chunks, frame_seqlen, d)
                q_flat_m = q_r.permute(0, 2, 1, 3, 4).contiguous().reshape(-1, d)
                q_flat_m = half(q_flat_m).unsqueeze(1)
                k_flat_m = half(k_flat_m).unsqueeze(1)
                v_flat_m = half(v_flat_m).unsqueeze(1)
                if q_scale is not None:
                    q_flat_m = q_flat_m * q_scale
                q_flat_m = q_flat_m.to(v_flat_m.dtype)
                k_flat_m = k_flat_m.to(v_flat_m.dtype)

                cu_seqlens_q_m = torch.arange(
                    0, (num_chunks * num_seq + 1) * frame_seqlen, step=frame_seqlen,
                    dtype=torch.int32, device=q.device,
                )

                if (fa_version is None or fa_version == 3) and FLASH_ATTN_3_AVAILABLE:
                    out_flat_m = flash_attn_interface.flash_attn_varlen_func(
                        q=q_flat_m, k=k_flat_m, v=v_flat_m,
                        cu_seqlens_q=cu_seqlens_q_m, cu_seqlens_k=cu_seqlens_k_m,
                        max_seqlen_q=frame_seqlen, max_seqlen_k=max_seqlen_k_m,
                        softmax_scale=softmax_scale, causal=causal, deterministic=deterministic,
                    )[0]
                else:
                    out_flat_m = flash_attn.flash_attn_varlen_func(
                        q=q_flat_m, k=k_flat_m, v=v_flat_m,
                        cu_seqlens_q=cu_seqlens_q_m, cu_seqlens_k=cu_seqlens_k_m,
                        max_seqlen_q=frame_seqlen, max_seqlen_k=max_seqlen_k_m,
                        dropout_p=dropout_p, softmax_scale=softmax_scale,
                        causal=causal, window_size=(-1, -1), deterministic=deterministic,
                    )

                # Output: [b*num_chunks*h*frame_seqlen, 1, d] → [b, lq, h, d]
                out_r = out_flat_m.squeeze(1).reshape(b, num_chunks, h, frame_seqlen, d)
                out_buf = out_r.permute(0, 2, 1, 3, 4).reshape(b, h, lq, d).transpose(1, 2).to(out_dtype)
                if _any_drop:
                    out_buf[:, :, drop_head_mask, :] = 0
                return out_buf

            # Fallback: per-chunk path (used when capture or soft_ablate is active)
            cu_seqlens_q_fixed = torch.arange(
                0, (b * h + 1) * frame_seqlen, step=frame_seqlen,
                dtype=torch.int32, device=q.device,
            )
            for offset in range(0, lq, frame_seqlen):
                q_chunk = q[:, offset:offset + frame_seqlen]
                if hasattr(kv_cache, "get_decoupled_flat_kv_and_frames"):
                    k_flat, v_flat, cu_seqlens_k, max_seqlen_k, k_frame_ids_flat = kv_cache.get_decoupled_flat_kv_and_frames(
                        current_start=base_start + offset,
                        grid_sizes=grid_sizes,
                        freqs=freqs,
                    )
                else:
                    k_flat, v_flat, cu_seqlens_k, max_seqlen_k = kv_cache.get_decoupled_flat_kv(
                        current_start=base_start + offset,
                        grid_sizes=grid_sizes,
                        freqs=freqs,
                    )
                    k_frame_ids_flat = None
                k_flat = _apply_soft_ablate_to_k_flat(
                    k_flat_chunk=k_flat,
                    cu_seqlens_k_chunk=cu_seqlens_k,
                    k_frame_ids_flat=k_frame_ids_flat,
                    chunk_start_token=base_start + offset,
                )
                _capture_varlen_frame_attention(
                    q_chunk=q_chunk,
                    k_flat_chunk=k_flat,
                    cu_seqlens_k_chunk=cu_seqlens_k,
                    chunk_start_token=base_start + offset,
                    k_frame_ids_flat=k_frame_ids_flat,
                )
                out_buf[:, offset:offset + frame_seqlen] = run_varlen(q_chunk, k_flat, v_flat, cu_seqlens_k, max_seqlen_k, cu_seqlens_q_override=cu_seqlens_q_fixed)
            return out_buf

        k_frame_ids_flat = None
        if getattr(kv_cache, "post_prune_rope", False):
            if hasattr(kv_cache, "get_flat_kv_and_pos"):
                k_flat, v_flat, cu_seqlens_k, max_seqlen_k, pos_ids = kv_cache.get_flat_kv_and_pos()
                if freqs is None:
                    raise ValueError("freqs is required when post_prune_rope=True")
                if hasattr(kv_cache, "apply_rope_to_flat_k"):
                    k_flat = kv_cache.apply_rope_to_flat_k(k_flat, pos_ids, freqs=freqs)
                    k_frame_ids_flat = pos_ids[:, 0].to(dtype=torch.long)
                else:
                    raise ValueError("kv_cache must provide apply_rope_to_flat_k for post-prune RoPE.")
            else:
                raise ValueError("kv_cache must provide get_flat_kv_and_pos or get_decoupled_flat_kv for post-prune RoPE.")
        else:
            k_flat, v_flat, cu_seqlens_k, max_seqlen_k = kv_cache.get_flat_kv()
            if frame_seqlen is not None and frame_seqlen > 0 and hasattr(kv_cache, "global_end_index"):
                k_frame_ids_flat = torch.empty((k_flat.shape[0],), dtype=torch.long, device=q.device)
                for b_idx in range(b):
                    global_end = int(kv_cache.global_end_index[b_idx])
                    for h_idx in range(h):
                        seq_idx = b_idx * h + h_idx
                        ks = int(cu_seqlens_k[seq_idx].item())
                        ke = int(cu_seqlens_k[seq_idx + 1].item())
                        if ke <= ks:
                            continue
                        seq_len = ke - ks
                        global_start = max(0, global_end - seq_len)
                        token_idx = torch.arange(global_start, global_end, device=q.device, dtype=torch.long)
                        k_frame_ids_flat[ks:ke] = token_idx // frame_seqlen

        k_flat = _apply_soft_ablate_to_k_flat(
            k_flat_chunk=k_flat,
            cu_seqlens_k_chunk=cu_seqlens_k,
            k_frame_ids_flat=k_frame_ids_flat,
            chunk_start_token=int(current_start or 0),
        )

        _capture_varlen_frame_attention(
            q_chunk=q,
            k_flat_chunk=k_flat,
            cu_seqlens_k_chunk=cu_seqlens_k,
            chunk_start_token=int(current_start or 0),
            k_frame_ids_flat=k_frame_ids_flat,
        )
        out = run_varlen(q, k_flat, v_flat, cu_seqlens_k, max_seqlen_k)
        return out.type(out_dtype)
    finally:
        if capture_enabled:
            capture_obj.current_layer_idx += 1


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
):
    # 优先检查流式帧级捕获（内存高效）
    if FRAME_ATTENTION_CAPTURE.enabled:
        if FRAME_ATTENTION_CAPTURE.should_capture():
            # 使用流式捕获：flash attention + 分块计算 frame-level attention
            out = FRAME_ATTENTION_CAPTURE.capture_and_forward(
                q=q, k=k, v=v,
                flash_attn_fn=flash_attention,
                q_lens=q_lens,
                k_lens=k_lens,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                q_scale=q_scale,
                causal=causal,
                window_size=window_size,
                deterministic=deterministic,
                dtype=dtype,
                version=fa_version,
            )
            return out
        else:
            # 不捕获，但仍需更新计数器
            FRAME_ATTENTION_CAPTURE.current_layer_idx += 1

    # 检查是否需要捕获完整注意力权重（旧方式，可能 OOM）
    if ATTENTION_WEIGHT_CAPTURE.enabled and ATTENTION_WEIGHT_CAPTURE.should_capture():
        out, attn_data = attention_with_weights(
            q=q, k=k, v=v,
            q_lens=q_lens, k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            dtype=dtype,
            return_logits=ATTENTION_WEIGHT_CAPTURE.capture_logits,  # 根据配置返回 logits 或 probs
        )
        # 存储注意力权重（移到 CPU 以节省 GPU 内存）
        ATTENTION_WEIGHT_CAPTURE.captured_weights.append({
            'layer_idx': ATTENTION_WEIGHT_CAPTURE.get_effective_layer_idx(),  # 使用模块化索引
            'attn_weights': attn_data.cpu(),
            'q_shape': q.shape,
            'k_shape': k.shape,
            'is_logits': ATTENTION_WEIGHT_CAPTURE.capture_logits,  # 标记是 logits 还是 probs
        })
        ATTENTION_WEIGHT_CAPTURE.current_layer_idx += 1
        return out

    ATTENTION_WEIGHT_CAPTURE.current_layer_idx += 1

    if FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE:
        return flash_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            version=fa_version,
        )
    else:
        if q_lens is not None or k_lens is not None:
            warnings.warn(
                'Padding mask is disabled when using scaled_dot_product_attention. It can have a significant impact on performance.'
            )
        attn_mask = None

        q = q.transpose(1, 2).to(dtype)
        k = k.transpose(1, 2).to(dtype)
        v = v.transpose(1, 2).to(dtype)

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=causal, dropout_p=dropout_p)

        out = out.transpose(1, 2).contiguous()
        return out


def attention_with_weights(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    dtype=torch.bfloat16,
    return_logits=True,
):
    """
    计算注意力并返回注意力权重。
    这比 flash attention 慢，但允许我们捕获注意力权重用于可视化。

    Args:
        q: Query 张量，形状 [B, Lq, Nq, C]
        k: Key 张量，形状 [B, Lk, Nk, C]
        v: Value 张量，形状 [B, Lk, Nk, C]
        return_logits: 如果 True，返回 pre-softmax logits（用于 Figure 4）；
                      否则返回 post-softmax 概率

    Returns:
        out: 输出张量，形状 [B, Lq, Nq, C]
        attn_data: 注意力数据，形状 [B, Nq, Lq, Lk]
                  如果 return_logits=True，这是 pre-softmax 分数（可以是负值）
                  如果 return_logits=False，这是 post-softmax 概率 [0,1]
    """
    out_dtype = q.dtype

    # q: [B, Lq, N, C] -> [B, N, Lq, C]
    # k: [B, Lk, N, C] -> [B, N, Lk, C]
    # v: [B, Lk, N, C] -> [B, N, Lk, C]
    q = q.transpose(1, 2).to(dtype)
    k = k.transpose(1, 2).to(dtype)
    v = v.transpose(1, 2).to(dtype)

    if q_scale is not None:
        q = q * q_scale

    # Support GQA/MQA: Q heads can be a multiple of K/V heads (Nq must be divisible by Nk).
    if q.shape[1] != k.shape[1]:
        n_q, n_k = q.shape[1], k.shape[1]
        if n_q % n_k != 0:
            raise ValueError(f"Nq must be divisible by Nk, got Nq={n_q}, Nk={n_k}")
        repeat_factor = n_q // n_k
        k = k.repeat_interleave(repeat_factor, dim=1)
        v = v.repeat_interleave(repeat_factor, dim=1)

    # 计算缩放因子
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5

    # 计算注意力分数（logits）: [B, N, Lq, Lk]
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) * softmax_scale

    bsz, _n, lq, lk = attn_scores.shape

    # Apply key padding mask if provided (k_lens is [B]).
    q_valid = None
    if k_lens is not None:
        key_idx = torch.arange(lk, device=attn_scores.device).view(1, 1, 1, lk)
        key_valid = key_idx < k_lens.view(bsz, 1, 1, 1)
        attn_scores = attn_scores.masked_fill(~key_valid, float('-inf'))

    # Track query validity to avoid NaNs when a padded query would be fully masked.
    if q_lens is not None:
        q_idx = torch.arange(lq, device=attn_scores.device).view(1, 1, lq, 1)
        q_valid = q_idx < q_lens.view(bsz, 1, 1, 1)
        attn_scores = attn_scores.masked_fill(~q_valid, 0.0)

    # 对齐非方阵 Q/K：query i 对应 key i + (lk - lq)。
    # 对于 varlen（k_lens/q_lens）场景，flash-attn 使用每个样本的有效长度来计算 offset，
    # 否则 window/causal 的对齐会与快路径不一致。
    if (k_lens is not None) or (q_lens is not None):
        lk_eff = k_lens if k_lens is not None else torch.full((bsz,), lk, device=attn_scores.device, dtype=torch.long)
        lq_eff = q_lens if q_lens is not None else torch.full((bsz,), lq, device=attn_scores.device, dtype=torch.long)
        offset = (lk_eff - lq_eff).view(bsz, 1, 1)  # [B,1,1]
    else:
        offset = lk - lq  # scalar

    q_pos = torch.arange(lq, device=attn_scores.device).view(1, lq, 1)  # [1,Lq,1]
    k_pos = torch.arange(lk, device=attn_scores.device).view(1, 1, lk)  # [1,1,Lk]
    center = q_pos + offset  # scalar or [B,1,1] -> [B,Lq,1]

    # 如果需要 causal mask
    if causal:
        # Mask positions where key is "in the future" relative to the aligned center.
        causal_mask = k_pos > center  # [B,Lq,Lk] or [1,Lq,Lk]
        attn_scores = attn_scores.masked_fill(causal_mask.unsqueeze(1), float('-inf'))

    # Sliding window local attention (if enabled).
    # Semantics follow the same "offset" convention as the causal mask for non-square Q/K:
    # query position i is aligned to key position i + (lk - lq) (varlen uses per-sample offset).
    if window_size != (-1, -1):
        left, right = window_size
        if left < 0:
            left = lk
        if right < 0:
            right = lk
        lower = center - left
        upper = center + right
        window_mask = (k_pos < lower) | (k_pos > upper)  # [B,Lq,Lk] or [1,Lq,Lk]
        attn_scores = attn_scores.masked_fill(window_mask.unsqueeze(1), float('-inf'))

    # 计算注意力权重（概率）
    # 当 key padding mask + window mask（或 causal mask）导致某些 query 行被完全屏蔽时，
    # softmax(-inf, -inf, ...) 会产生 NaN；flash-attn 在这种情况下会输出 0。
    row_has_any_valid = torch.isfinite(attn_scores).any(dim=-1, keepdim=True)
    attn_scores_for_softmax = attn_scores.masked_fill(~row_has_any_valid, 0.0)
    attn_weights = torch.softmax(attn_scores_for_softmax, dim=-1)
    attn_weights = attn_weights.masked_fill(~row_has_any_valid, 0.0)

    # 应用 dropout
    if dropout_p > 0.:
        attn_weights = torch.nn.functional.dropout(attn_weights, p=dropout_p)

    # 计算输出: [B, N, Lq, C]
    out = torch.matmul(attn_weights, v)

    if q_valid is not None:
        out = out.masked_fill(~q_valid, 0.0)
        attn_weights = attn_weights.masked_fill(~q_valid, 0.0)

    # 转置回来: [B, N, Lq, C] -> [B, Lq, N, C]
    out = out.transpose(1, 2).contiguous().to(out_dtype)

    # 根据配置返回 logits 或 probs
    if return_logits:
        return out, attn_scores  # 返回 pre-softmax logits
    else:
        return out, attn_weights  # 返回 post-softmax probs

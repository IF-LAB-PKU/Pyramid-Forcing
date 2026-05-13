"""Deprecated head selectors.

The IVC / semantic / trajectory selectors here were used by an earlier line
of experiments and are kept only so legacy YAMLs continue to import. New code
should use :mod:`pyramidkv.adaptive_cache` (per-head ``HeadComposition``)
instead.
"""

import warnings

import torch

warnings.warn(
    "pyramidkv.selectors is deprecated and will be removed in a future release; "
    "use pyramidkv.adaptive_cache.AdaptiveKVCache with HeadComposition strategies "
    "instead.",
    DeprecationWarning,
    stacklevel=2,
)


def _topk_mask(scores: torch.Tensor, k: int) -> torch.Tensor:
    mask = torch.zeros(scores.shape[0], dtype=torch.bool, device=scores.device)
    if k <= 0 or scores.numel() == 0:
        return mask
    k = min(k, scores.shape[0])
    idx = torch.topk(scores, k=k, largest=True, sorted=False).indices
    mask[idx] = True
    return mask


def _normalize_scores(scores: torch.Tensor) -> torch.Tensor:
    if scores.numel() == 0:
        return scores
    s_min = scores.min()
    s_max = scores.max()
    if torch.isclose(s_min, s_max):
        return torch.zeros_like(scores)
    return (scores - s_min) / (s_max - s_min)


class ThreeDIVCSelector:
    @staticmethod
    def _split_dims(d_model: int) -> list[int]:
        if d_model % 2 != 0:
            raise ValueError(f"d_model must be even, got {d_model}")
        c = d_model // 2
        return [c - 2 * (c // 3), c // 3, c // 3]

    @staticmethod
    def get_ivc_scores(pos_3d: torch.Tensor, d_model: int, freqs: torch.Tensor) -> torch.Tensor:
        if pos_3d.numel() == 0:
            return torch.empty(0, dtype=torch.float32, device=pos_3d.device)

        splits = ThreeDIVCSelector._split_dims(d_model)
        ft, fy, fx = freqs.split(splits, dim=1)

        pos_t = pos_3d[:, 0].clamp(min=0, max=max(0, ft.shape[0] - 1))
        pos_y = pos_3d[:, 1].clamp(min=0, max=max(0, fy.shape[0] - 1))
        pos_x = pos_3d[:, 2].clamp(min=0, max=max(0, fx.shape[0] - 1))

        cos_terms = []
        sin_terms = []

        if ft.shape[1] > 0:
            ft_sel = ft[pos_t]
            cos_terms.append(ft_sel.real)
            sin_terms.append(ft_sel.imag)
        if fy.shape[1] > 0:
            fy_sel = fy[pos_y]
            cos_terms.append(fy_sel.real)
            sin_terms.append(fy_sel.imag)
        if fx.shape[1] > 0:
            fx_sel = fx[pos_x]
            cos_terms.append(fx_sel.real)
            sin_terms.append(fx_sel.imag)

        if not cos_terms:
            return torch.zeros(pos_3d.shape[0], dtype=torch.float32, device=pos_3d.device)

        v_score = torch.cat(cos_terms, dim=1).sum(dim=1).abs()
        u_score = torch.cat(sin_terms, dim=1).sum(dim=1).abs()
        return torch.maximum(v_score, u_score).to(torch.float32)

    @staticmethod
    def get_ivc_mask(
        pos_3d: torch.Tensor,
        d_model: int,
        freqs: torch.Tensor,
        ratio: float = 0.1,
        min_keep: int = 1,
    ) -> torch.Tensor:
        if ratio <= 0:
            return torch.zeros(pos_3d.shape[0], dtype=torch.bool, device=pos_3d.device)
        scores = ThreeDIVCSelector.get_ivc_scores(pos_3d, d_model=d_model, freqs=freqs)
        k = max(min_keep, int(round(pos_3d.shape[0] * ratio)))
        return _topk_mask(scores, k=k)


class SemanticValueSelector:
    @staticmethod
    def _normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
        return x / (x.norm(dim=dim, keepdim=True) + eps)

    @staticmethod
    def prepare_prompt_values(prompt_v: torch.Tensor | None, num_heads: int, head_dim: int) -> torch.Tensor | None:
        if prompt_v is None:
            return None

        if prompt_v.ndim == 1:
            if prompt_v.shape[0] != head_dim:
                raise ValueError(f"prompt_v dim mismatch: expected {head_dim}, got {prompt_v.shape[0]}")
            return prompt_v.unsqueeze(0).expand(num_heads, -1)

        if prompt_v.ndim == 2:
            if prompt_v.shape == (num_heads, head_dim):
                return prompt_v
            if prompt_v.shape[1] == head_dim:
                return prompt_v.mean(dim=0, keepdim=True).expand(num_heads, -1)
            raise ValueError(
                f"prompt_v shape {tuple(prompt_v.shape)} is incompatible with num_heads={num_heads}, head_dim={head_dim}"
            )

        if prompt_v.ndim == 3:
            if prompt_v.shape[1] == num_heads and prompt_v.shape[2] == head_dim:
                return prompt_v.mean(dim=0)
            if prompt_v.shape[0] == num_heads and prompt_v.shape[2] == head_dim:
                return prompt_v.mean(dim=1)
            if prompt_v.shape[2] == head_dim:
                pooled = prompt_v.mean(dim=(0, 1), keepdim=False)
                return pooled.unsqueeze(0).expand(num_heads, -1)
            raise ValueError(
                f"prompt_v shape {tuple(prompt_v.shape)} is incompatible with num_heads={num_heads}, head_dim={head_dim}"
            )

        raise ValueError(f"Unsupported prompt_v ndim={prompt_v.ndim}")

    @staticmethod
    def get_semantic_scores(
        kv_v: torch.Tensor,
        prompt_v: torch.Tensor | None,
        seed_ratio: float = 0.01,
    ) -> torch.Tensor:
        single_head = (kv_v.ndim == 2)
        if single_head:
            kv_v = kv_v.unsqueeze(0)
        if kv_v.ndim != 3:
            raise ValueError(f"kv_v must be [L,D] or [H,L,D], got shape {tuple(kv_v.shape)}")

        h, l, d = kv_v.shape
        prompt_per_head = SemanticValueSelector.prepare_prompt_values(prompt_v, num_heads=h, head_dim=d)
        if prompt_per_head is None:
            scores = torch.zeros((h, l), dtype=torch.float32, device=kv_v.device)
            return scores[0] if single_head else scores

        kv_norm = SemanticValueSelector._normalize(kv_v.float(), dim=-1)
        prompt_norm = SemanticValueSelector._normalize(prompt_per_head.float(), dim=-1)

        all_scores = []
        for head_idx in range(h):
            sim = torch.matmul(kv_norm[head_idx], prompt_norm[head_idx])
            seed_k = max(1, int(round(l * max(0.0, seed_ratio))))
            seed_k = min(seed_k, l)
            seed_idx = torch.topk(sim, k=seed_k, largest=True, sorted=False).indices
            seed_vec = kv_v[head_idx][seed_idx].float().mean(dim=0)
            query = SemanticValueSelector._normalize(seed_vec + prompt_per_head[head_idx].float(), dim=0)
            refine = torch.matmul(kv_norm[head_idx], query)
            all_scores.append(refine.to(torch.float32))

        scores = torch.stack(all_scores, dim=0)
        return scores[0] if single_head else scores

    @staticmethod
    def get_semantic_mask(
        kv_v: torch.Tensor,
        prompt_v: torch.Tensor | None,
        ratio: float,
        seed_ratio: float = 0.01,
        min_keep: int = 1,
    ) -> torch.Tensor:
        if ratio <= 0:
            if kv_v.ndim == 2:
                return torch.zeros(kv_v.shape[0], dtype=torch.bool, device=kv_v.device)
            return torch.zeros(kv_v.shape[:2], dtype=torch.bool, device=kv_v.device)

        scores = SemanticValueSelector.get_semantic_scores(kv_v, prompt_v=prompt_v, seed_ratio=seed_ratio)
        if scores.ndim == 1:
            k = max(min_keep, int(round(scores.shape[0] * ratio)))
            return _topk_mask(scores, k=k)

        h, l = scores.shape
        masks = []
        for head_idx in range(h):
            k = max(min_keep, int(round(l * ratio)))
            masks.append(_topk_mask(scores[head_idx], k=k))
        return torch.stack(masks, dim=0)

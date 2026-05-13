# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
import gc


# ============================================================
# Frame-Level Attention Capture (Memory Efficient)
# ============================================================


def compute_frame_attention_metrics_single_sequence(
    q_seq: torch.Tensor,
    k_seq: torch.Tensor,
    q_frame_ids: torch.Tensor,
    k_frame_ids: torch.Tensor,
    softmax_scale: float | None = None,
    chunk_tokens: int = 4096,
    capture_mode: str = "prob_mass",
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor, torch.Tensor]:
    """
    计算单个序列（单个 batch-head）上的 frame-level attention 指标。

    Args:
        q_seq: [Lq, D]
        k_seq: [Lk, D]
        q_frame_ids: [Lq]，每个 query token 对应的绝对 frame id
        k_frame_ids: [Lk]，每个 key token 对应的绝对 frame id
        softmax_scale: QK^T 缩放
        chunk_tokens: 按 key token 维分块，避免大矩阵 OOM
        capture_mode: logits_mean / prob_mass / both

    Returns:
        logits_mean: [Q_frames, K_frames] 或 None
        prob_mass: [Q_frames, K_frames] 或 None
        q_unique: [Q_frames] 升序 frame id
        k_unique: [K_frames] 升序 frame id
    """
    if q_seq.ndim != 2 or k_seq.ndim != 2:
        raise ValueError(f"q_seq/k_seq must be 2D, got {q_seq.shape}/{k_seq.shape}")
    if q_frame_ids.ndim != 1 or k_frame_ids.ndim != 1:
        raise ValueError(f"q_frame_ids/k_frame_ids must be 1D, got {q_frame_ids.shape}/{k_frame_ids.shape}")
    if q_seq.shape[0] != q_frame_ids.shape[0] or k_seq.shape[0] != k_frame_ids.shape[0]:
        raise ValueError("Token length and frame-id length mismatch")

    if softmax_scale is None:
        softmax_scale = q_seq.shape[-1] ** -0.5

    if capture_mode not in {"logits_mean", "prob_mass", "both"}:
        raise ValueError(f"Unsupported capture_mode={capture_mode}")

    need_logits = capture_mode in {"logits_mean", "both"}
    need_prob = capture_mode in {"prob_mass", "both"}

    q_unique = torch.unique(q_frame_ids.to(torch.long), sorted=True)
    k_unique = torch.unique(k_frame_ids.to(torch.long), sorted=True)
    nqf = int(q_unique.numel())
    nkf = int(k_unique.numel())

    device = q_seq.device
    q_seq_f = q_seq.float()
    k_seq_f = k_seq.float()
    q_ids = q_frame_ids.to(device=device, dtype=torch.long)
    k_ids = k_frame_ids.to(device=device, dtype=torch.long)

    logits_mean = None
    prob_mass = None

    q_masks = [(q_ids == int(qf.item())) for qf in q_unique]

    if need_logits:
        logits_sum = torch.zeros(nqf, nkf, dtype=torch.float32, device=device)
        logits_cnt = torch.zeros(nqf, nkf, dtype=torch.float32, device=device)
        for k_start in range(0, k_seq_f.shape[0], chunk_tokens):
            k_end = min(k_start + chunk_tokens, k_seq_f.shape[0])
            k_chunk = k_seq_f[k_start:k_end]
            k_ids_chunk = k_ids[k_start:k_end]
            scores = torch.matmul(q_seq_f, k_chunk.transpose(0, 1)) * softmax_scale  # [Lq, Lk_chunk]

            for q_idx, q_mask in enumerate(q_masks):
                if not torch.any(q_mask):
                    continue
                q_scores = scores[q_mask]
                for k_idx, kf in enumerate(k_unique):
                    k_mask = (k_ids_chunk == int(kf.item()))
                    if not torch.any(k_mask):
                        continue
                    vals = q_scores[:, k_mask]
                    logits_sum[q_idx, k_idx] += vals.sum()
                    logits_cnt[q_idx, k_idx] += float(vals.numel())

        logits_mean = torch.where(logits_cnt > 0, logits_sum / logits_cnt.clamp_min(1.0), torch.zeros_like(logits_sum))

    if need_prob:
        lq = q_seq_f.shape[0]
        row_max = torch.full((lq,), float("-inf"), dtype=torch.float32, device=device)
        row_sumexp = torch.zeros((lq,), dtype=torch.float32, device=device)

        # pass-1: 统计 softmax 分母
        for k_start in range(0, k_seq_f.shape[0], chunk_tokens):
            k_end = min(k_start + chunk_tokens, k_seq_f.shape[0])
            k_chunk = k_seq_f[k_start:k_end]
            scores = torch.matmul(q_seq_f, k_chunk.transpose(0, 1)) * softmax_scale

            chunk_max = scores.max(dim=-1).values
            new_max = torch.maximum(row_max, chunk_max)
            exp_old = torch.exp(row_max - new_max) * row_sumexp
            exp_new = torch.exp(scores - new_max.unsqueeze(-1)).sum(dim=-1)
            row_sumexp = exp_old + exp_new
            row_max = new_max

        row_valid = row_sumexp > 0
        denom = row_sumexp.clamp_min(1e-12)

        prob_sum = torch.zeros(nqf, nkf, dtype=torch.float32, device=device)
        q_valid_cnt = torch.zeros(nqf, dtype=torch.float32, device=device)

        # pass-2: 计算概率质量并聚合到 frame-level
        for q_idx, q_mask in enumerate(q_masks):
            q_mask_valid = q_mask & row_valid
            q_valid_cnt[q_idx] = float(q_mask_valid.sum().item())

        for k_start in range(0, k_seq_f.shape[0], chunk_tokens):
            k_end = min(k_start + chunk_tokens, k_seq_f.shape[0])
            k_chunk = k_seq_f[k_start:k_end]
            k_ids_chunk = k_ids[k_start:k_end]
            scores = torch.matmul(q_seq_f, k_chunk.transpose(0, 1)) * softmax_scale
            probs = torch.exp(scores - row_max.unsqueeze(-1)) / denom.unsqueeze(-1)
            probs = torch.where(row_valid.unsqueeze(-1), probs, torch.zeros_like(probs))

            for q_idx, q_mask in enumerate(q_masks):
                q_mask_valid = q_mask & row_valid
                if not torch.any(q_mask_valid):
                    continue
                q_probs = probs[q_mask_valid]
                for k_idx, kf in enumerate(k_unique):
                    k_mask = (k_ids_chunk == int(kf.item()))
                    if not torch.any(k_mask):
                        continue
                    # 先对 key-frame 内 token 求和，再对 query token 求平均
                    mass_per_q = q_probs[:, k_mask].sum(dim=-1)
                    prob_sum[q_idx, k_idx] += mass_per_q.sum()

        denom_q = q_valid_cnt.clamp_min(1.0).unsqueeze(1)
        prob_mass = torch.where(q_valid_cnt.unsqueeze(1) > 0, prob_sum / denom_q, torch.zeros_like(prob_sum))

    return logits_mean, prob_mass, q_unique, k_unique

class FrameAttentionCapture:
    """
    流式帧级注意力捕获器（内存高效版本）。

    核心思想：
    1. 使用 flash attention 进行实际的前向传播（高效，无 OOM）
    2. 分块计算 Q @ K^T 得到 attention logits
    3. 立即聚合到 frame-level 并释放 token-level 数据

    内存使用：O(Q_tokens × chunk_K_tokens) 而非 O(Q_tokens × all_K_tokens)
    """

    def __init__(self):
        self.enabled = False
        self.layer_indices = None
        self.current_layer_idx = 0
        self.num_layers = 60  # 30 blocks × 2 (self + cross)

        # Frame-level 配置
        self.frame_seq_length = 1560
        self.num_heads = 12
        self.chunk_frames = 3  # 每次处理的 K frames 数量

        # 回调函数：接收 frame-level attention 数据
        self.on_frame_attention = None
        self.capture_mode = "prob_mass"

    def enable(
        self,
        layer_indices=None,
        num_layers=60,
        frame_seq_length=1560,
        num_heads=12,
        chunk_frames=3,
        on_frame_attention=None,
        capture_mode="prob_mass",
    ):
        """
        启用流式帧级注意力捕获。

        Args:
            layer_indices: 要捕获的层索引列表（self-attn 索引）
            num_layers: 总层数（用于取模）
            frame_seq_length: 每帧的 token 数量
            num_heads: 注意力头数
            chunk_frames: 每次处理的 K frames 数量（控制内存）
            on_frame_attention: 回调函数，签名 (layer_idx, frame_attn, q_frames, k_frames)
                frame_attn: [num_heads, q_frames, k_frames] 的 frame-level attention logits
        """
        self.enabled = True
        self.layer_indices = layer_indices
        self.num_layers = num_layers
        self.frame_seq_length = frame_seq_length
        self.num_heads = num_heads
        self.chunk_frames = chunk_frames
        self.on_frame_attention = on_frame_attention
        if capture_mode not in {"logits_mean", "prob_mass", "both"}:
            raise ValueError(f"Unsupported capture_mode={capture_mode}")
        self.capture_mode = capture_mode
        self.current_layer_idx = 0

    def disable(self):
        """禁用捕获。"""
        self.enabled = False
        self.current_layer_idx = 0
        self.on_frame_attention = None

    def should_capture(self):
        """检查是否应该捕获当前层。"""
        if not self.enabled:
            return False
        if self.layer_indices is None:
            return True
        effective_idx = self.current_layer_idx % self.num_layers
        return effective_idx in self.layer_indices

    def get_effective_layer_idx(self):
        """获取当前的有效层索引（模块化后）。"""
        return self.current_layer_idx % self.num_layers

    def compute_frame_attention_chunked(self, q, k, softmax_scale=None):
        """
        分块计算 frame-level attention logits。

        Args:
            q: Query 张量 [B, N, Lq, C]（已转置）
            k: Key 张量 [B, N, Lk, C]（已转置）
            softmax_scale: 缩放因子

        Returns:
            dict:
                logits_mean: [N, q_frames, k_frames] 或 None
                prob_mass: [N, q_frames, k_frames] 或 None
        """
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** -0.5

        b, n, lq, c = q.shape
        lk = k.shape[2]

        q_frames = lq // self.frame_seq_length
        k_frames = lk // self.frame_seq_length

        need_logits = self.capture_mode in {"logits_mean", "both"}
        need_prob = self.capture_mode in {"prob_mass", "both"}

        frame_attn_logits = (
            torch.zeros(n, q_frames, k_frames, dtype=torch.float32, device='cpu')
            if need_logits else None
        )
        frame_attn_prob = (
            torch.zeros(n, q_frames, k_frames, dtype=torch.float32, device='cpu')
            if need_prob else None
        )

        for qf in range(q_frames):
            q_token_start = qf * self.frame_seq_length
            q_token_end = (qf + 1) * self.frame_seq_length
            q_frame = q[:, :, q_token_start:q_token_end, :].float()  # [B, N, frame_tokens, C]

            if need_prob:
                row_max = torch.full(
                    (b, n, self.frame_seq_length),
                    float("-inf"),
                    dtype=torch.float32,
                    device=q_frame.device,
                )
                row_sumexp = torch.zeros_like(row_max)

            # pass-1 / logits：按 K chunk 扫描
            for kf_start in range(0, k_frames, self.chunk_frames):
                kf_end = min(kf_start + self.chunk_frames, k_frames)
                k_token_start = kf_start * self.frame_seq_length
                k_token_end = kf_end * self.frame_seq_length
                k_chunk = k[:, :, k_token_start:k_token_end, :].float()
                scores = torch.matmul(q_frame, k_chunk.transpose(-2, -1)) * softmax_scale

                if need_logits:
                    for kf_local in range(kf_end - kf_start):
                        kf = kf_start + kf_local
                        ks = kf_local * self.frame_seq_length
                        ke = (kf_local + 1) * self.frame_seq_length
                        frame_scores = scores[:, :, :, ks:ke]
                        frame_attn_logits[:, qf, kf] = frame_scores.mean(dim=(0, 2, 3)).cpu()

                if need_prob:
                    chunk_max = scores.max(dim=-1).values
                    new_max = torch.maximum(row_max, chunk_max)
                    exp_old = torch.exp(row_max - new_max) * row_sumexp
                    exp_new = torch.exp(scores - new_max.unsqueeze(-1)).sum(dim=-1)
                    row_sumexp = exp_old + exp_new
                    row_max = new_max

                del scores
                del k_chunk

            if need_prob:
                row_valid = row_sumexp > 0
                denom = row_sumexp.clamp_min(1e-12)
                for kf_start in range(0, k_frames, self.chunk_frames):
                    kf_end = min(kf_start + self.chunk_frames, k_frames)
                    k_token_start = kf_start * self.frame_seq_length
                    k_token_end = kf_end * self.frame_seq_length
                    k_chunk = k[:, :, k_token_start:k_token_end, :].float()
                    scores = torch.matmul(q_frame, k_chunk.transpose(-2, -1)) * softmax_scale
                    probs = torch.exp(scores - row_max.unsqueeze(-1)) / denom.unsqueeze(-1)
                    probs = torch.where(row_valid.unsqueeze(-1), probs, torch.zeros_like(probs))

                    for kf_local in range(kf_end - kf_start):
                        kf = kf_start + kf_local
                        ks = kf_local * self.frame_seq_length
                        ke = (kf_local + 1) * self.frame_seq_length
                        frame_probs = probs[:, :, :, ks:ke].sum(dim=-1)  # [B, N, frame_tokens]
                        frame_attn_prob[:, qf, kf] = frame_probs.mean(dim=(0, 2)).cpu()

                    del scores
                    del probs
                    del k_chunk

            torch.cuda.empty_cache()

        return {
            "logits_mean": frame_attn_logits,
            "prob_mass": frame_attn_prob,
        }

    def capture_and_forward(self, q, k, v, flash_attn_fn, **kwargs):
        """
        捕获 frame-level attention 并执行 flash attention 前向传播。

        Args:
            q, k, v: 输入张量 [B, L, N, C]
            flash_attn_fn: Flash attention 函数
            **kwargs: 传递给 flash_attn_fn 的其他参数

        Returns:
            out: Flash attention 的输出
        """
        # 1. 执行 flash attention 得到输出
        out = flash_attn_fn(q=q, k=k, v=v, **kwargs)

        # 2. 如果需要捕获，分块计算 frame-level attention
        if self.should_capture():
            # 转置为 [B, N, L, C] 格式
            q_t = q.transpose(1, 2)
            k_t = k.transpose(1, 2)

            # 处理 q_scale
            q_scale = kwargs.get('q_scale')
            if q_scale is not None:
                q_t = q_t * q_scale

            # 计算 frame-level attention
            softmax_scale = kwargs.get('softmax_scale')
            frame_metrics = self.compute_frame_attention_chunked(q_t, k_t, softmax_scale)
            frame_attn = (
                frame_metrics["logits_mean"]
                if self.capture_mode == "logits_mean"
                else frame_metrics["prob_mass"]
            )

            # 调用回调
            if self.on_frame_attention is not None:
                q_frames = q.shape[1] // self.frame_seq_length
                k_frames = k.shape[1] // self.frame_seq_length
                self.on_frame_attention(
                    layer_idx=self.get_effective_layer_idx(),
                    frame_attn=frame_attn,
                    frame_attn_logits=frame_metrics["logits_mean"],
                    frame_attn_prob=frame_metrics["prob_mass"],
                    q_frames=q_frames,
                    k_frames=k_frames,
                    q_frame_indices=list(range(q_frames)),
                    k_frame_indices=list(range(k_frames)),
                    capture_mode=self.capture_mode,
                )

            # 清理
            del q_t, k_t, frame_attn, frame_metrics
            gc.collect()
            torch.cuda.empty_cache()

        self.current_layer_idx += 1
        return out


# 全局实例
FRAME_ATTENTION_CAPTURE = FrameAttentionCapture()


# Global attention weight capture configuration
class AttentionWeightCapture:
    """
    全局注意力权重捕获配置。

    重要：捕获的是 pre-softmax logits（注意力分数），而非 softmax 后的概率！
    这对于复现论文 Figure 4 至关重要，因为论文中的 Y 轴范围是 [-4, 6]。
    """
    def __init__(self):
        self.enabled = False
        self.layer_indices = None  # None 表示捕获所有层，或者是层索引列表
        self.captured_weights = []  # 捕获的注意力权重列表
        self.current_layer_idx = 0  # 当前前向传播的层索引
        self.capture_logits = True  # 是否捕获 pre-softmax logits（默认 True）
        self.num_layers = 30  # Wan 模型的层数，用于取模

    def enable(self, layer_indices=None, capture_logits=True, num_layers=30):
        """
        启用注意力权重捕获。

        Args:
            layer_indices: 要捕获的层索引列表，None 表示全部
            capture_logits: 如果 True，捕获 pre-softmax logits；否则捕获 post-softmax probs
            num_layers: 模型的总层数，用于 current_layer_idx 取模
        """
        self.enabled = True
        self.layer_indices = layer_indices
        self.capture_logits = capture_logits
        self.num_layers = num_layers
        self.captured_weights = []
        self.current_layer_idx = 0

    def disable(self):
        """禁用注意力权重捕获。"""
        self.enabled = False
        self.captured_weights = []
        self.current_layer_idx = 0

    def reset(self):
        """重置捕获的权重，用于新的前向传播。"""
        self.captured_weights = []
        self.current_layer_idx = 0

    def should_capture(self):
        """检查是否应该捕获当前层（使用模块化索引）。"""
        if not self.enabled:
            return False
        if self.layer_indices is None:
            return True
        # 使用模块化索引，这样每个 denoising step 的相同层都会被检查
        effective_layer_idx = self.current_layer_idx % self.num_layers
        return effective_layer_idx in self.layer_indices

    def get_effective_layer_idx(self):
        """获取当前的有效层索引（模块化后）。"""
        return self.current_layer_idx % self.num_layers

    def save(self, path):
        """保存捕获的权重到磁盘。"""
        import torch
        torch.save({
            'attention_weights': self.captured_weights,
            'layer_indices': self.layer_indices,
            'capture_logits': self.capture_logits,
        }, path)
        print(f"Saved attention weights to {path}")


ATTENTION_WEIGHT_CAPTURE = AttentionWeightCapture()

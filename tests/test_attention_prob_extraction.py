import torch

from wan.modules.attention import compute_frame_attention_metrics_single_sequence


def _naive_metrics(q_seq, k_seq, q_ids, k_ids, softmax_scale):
    q_unique = torch.unique(q_ids, sorted=True)
    k_unique = torch.unique(k_ids, sorted=True)

    scores = torch.matmul(q_seq.float(), k_seq.float().transpose(0, 1)) * softmax_scale
    probs = torch.softmax(scores, dim=-1)

    logits = torch.zeros((q_unique.numel(), k_unique.numel()), dtype=torch.float32)
    mass = torch.zeros((q_unique.numel(), k_unique.numel()), dtype=torch.float32)

    for qi, qf in enumerate(q_unique):
        q_mask = q_ids == qf
        for ki, kf in enumerate(k_unique):
            k_mask = k_ids == kf
            logits[qi, ki] = scores[q_mask][:, k_mask].mean()
            mass[qi, ki] = probs[q_mask][:, k_mask].sum(dim=-1).mean()
    return logits, mass, q_unique, k_unique


def test_prob_mass_rows_sum_to_one():
    torch.manual_seed(0)
    q = torch.randn(8, 6)
    k = torch.randn(15, 6)
    q_ids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
    k_ids = torch.tensor([0] * 5 + [1] * 5 + [2] * 5, dtype=torch.long)

    _, mass, _, _ = compute_frame_attention_metrics_single_sequence(
        q_seq=q,
        k_seq=k,
        q_frame_ids=q_ids,
        k_frame_ids=k_ids,
        chunk_tokens=4,
        capture_mode="prob_mass",
    )
    assert mass is not None
    row_sum = mass.sum(dim=-1)
    assert torch.allclose(row_sum, torch.ones_like(row_sum), atol=1e-4, rtol=1e-4)


def test_chunk_consistency():
    torch.manual_seed(1)
    q = torch.randn(9, 8)
    k = torch.randn(18, 8)
    q_ids = torch.tensor([0] * 3 + [1] * 3 + [2] * 3, dtype=torch.long)
    k_ids = torch.tensor([0] * 6 + [1] * 6 + [2] * 6, dtype=torch.long)

    l1, p1, qf1, kf1 = compute_frame_attention_metrics_single_sequence(
        q_seq=q,
        k_seq=k,
        q_frame_ids=q_ids,
        k_frame_ids=k_ids,
        chunk_tokens=3,
        capture_mode="both",
    )
    l2, p2, qf2, kf2 = compute_frame_attention_metrics_single_sequence(
        q_seq=q,
        k_seq=k,
        q_frame_ids=q_ids,
        k_frame_ids=k_ids,
        chunk_tokens=7,
        capture_mode="both",
    )

    assert torch.equal(qf1, qf2)
    assert torch.equal(kf1, kf2)
    assert torch.allclose(l1, l2, atol=1e-5, rtol=1e-5)
    assert torch.allclose(p1, p2, atol=1e-5, rtol=1e-5)


def test_metrics_match_naive_softmax():
    torch.manual_seed(2)
    q = torch.randn(6, 5)
    k = torch.randn(12, 5)
    q_ids = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
    k_ids = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3], dtype=torch.long)
    scale = q.shape[-1] ** -0.5

    logits, mass, qf, kf = compute_frame_attention_metrics_single_sequence(
        q_seq=q,
        k_seq=k,
        q_frame_ids=q_ids,
        k_frame_ids=k_ids,
        softmax_scale=scale,
        chunk_tokens=4,
        capture_mode="both",
    )
    n_logits, n_mass, n_qf, n_kf = _naive_metrics(q, k, q_ids, k_ids, scale)

    assert torch.equal(qf, n_qf)
    assert torch.equal(kf, n_kf)
    assert torch.allclose(logits, n_logits, atol=1e-5, rtol=1e-5)
    assert torch.allclose(mass, n_mass, atol=1e-5, rtol=1e-5)

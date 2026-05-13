from pathlib import Path

import pytest

from utils.sampler import DistributedEvalSampler


BASE_DIR = Path(__file__).resolve().parents[1]


def _rank_indices(dataset_size: int, num_replicas: int):
    dataset = list(range(dataset_size))
    return [
        list(DistributedEvalSampler(dataset, num_replicas=num_replicas, rank=rank))
        for rank in range(num_replicas)
    ]


def test_distributed_eval_sampler_covers_uneven_prompt_count_without_dropping():
    rank_indices = _rank_indices(dataset_size=32, num_replicas=6)

    flattened = [idx for indices in rank_indices for idx in indices]
    assert sorted(flattened) == list(range(32))
    assert len(flattened) == 32
    assert len(set(flattened)) == 32


def test_distributed_eval_sampler_lengths_match_rank_indices():
    dataset = list(range(32))

    lengths = [
        len(DistributedEvalSampler(dataset, num_replicas=6, rank=rank))
        for rank in range(6)
    ]
    assert lengths == [6, 6, 5, 5, 5, 5]


def test_distributed_eval_sampler_rejects_invalid_rank():
    with pytest.raises(ValueError):
        DistributedEvalSampler(list(range(3)), num_replicas=2, rank=2)


def test_inference_uses_eval_sampler_without_drop_last():
    text = (BASE_DIR / "inference.py").read_text(encoding="utf-8")
    assert "from utils.sampler import DistributedEvalSampler" in text
    assert "sampler = DistributedEvalSampler(" in text
    assert "DistributedSampler" not in text
    assert "drop_last=True" not in text

from torch.utils.data import Sampler


class DistributedEvalSampler(Sampler):
    """Shard evaluation indices across ranks without dropping or padding."""

    def __init__(self, dataset, num_replicas: int, rank: int):
        if num_replicas <= 0:
            raise ValueError(f"num_replicas must be positive, got {num_replicas}")
        if rank < 0 or rank >= num_replicas:
            raise ValueError(f"rank must be in [0, {num_replicas}), got {rank}")

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.num_replicas))

    def __len__(self):
        if len(self.dataset) <= self.rank:
            return 0
        return ((len(self.dataset) - 1 - self.rank) // self.num_replicas) + 1

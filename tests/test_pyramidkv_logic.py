import os
import tempfile

import torch

from pyramidkv.config import PyramidKVConfig
from pyramidkv.cache import PyramidKVCache


def _build_config(num_layers, num_heads, capacities):
    entries = [(0, head_idx, cap) for head_idx, cap in enumerate(capacities)]
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".csv", delete=False) as f:
        for layer_idx, head_idx, cap in entries:
            f.write(f"{layer_idx},{head_idx},{cap}\n")
        path = f.name

    config = PyramidKVConfig(path, num_layers=num_layers, num_heads=num_heads)
    os.unlink(path)
    return config


def _make_tokens(start, length, num_heads, head_dim):
    vals = torch.arange(start, start + length, dtype=torch.float32)
    return vals.view(1, length, 1, 1).expand(1, length, num_heads, head_dim)


def test_t2v_heterogeneous_sliding_window():
    config = _build_config(num_layers=1, num_heads=2, capacities=[10, 20])
    cache = PyramidKVCache(
        config=config,
        batch_size=1,
        num_heads=2,
        head_dim=4,
        layer_idx=0,
        is_i2v=False,
        context_len=0,
    )

    k1 = _make_tokens(0, 5, num_heads=2, head_dim=4)
    v1 = k1.clone()
    cache.update(k1, v1)

    assert cache.dynamic_k[0].shape[0] == 5
    assert cache.dynamic_k[1].shape[0] == 5

    k2 = _make_tokens(5, 8, num_heads=2, head_dim=4)
    v2 = k2.clone()
    cache.update(k2, v2)

    head0_vals = cache.dynamic_k[0][:, 0].cpu().tolist()
    head1_vals = cache.dynamic_k[1][:, 0].cpu().tolist()

    assert head0_vals == list(range(3, 13))
    assert head1_vals == list(range(0, 13))


def test_i2v_context_protection():
    config = _build_config(num_layers=1, num_heads=1, capacities=[10])
    cache = PyramidKVCache(
        config=config,
        batch_size=1,
        num_heads=1,
        head_dim=4,
        layer_idx=0,
        is_i2v=True,
        context_len=4,
    )

    k1 = _make_tokens(0, 7, num_heads=1, head_dim=4)
    cache.update(k1, k1)

    assert cache.static_k[0][:, 0].cpu().tolist() == [0.0, 1.0, 2.0, 3.0]
    assert cache.dynamic_k[0][:, 0].cpu().tolist() == [4.0, 5.0, 6.0]

    k2 = _make_tokens(7, 10, num_heads=1, head_dim=4)
    cache.update(k2, k2)

    assert cache.static_k[0][:, 0].cpu().tolist() == [0.0, 1.0, 2.0, 3.0]
    assert cache.dynamic_k[0][:, 0].cpu().tolist() == [11.0, 12.0, 13.0, 14.0, 15.0, 16.0]

    k_flat, _, cu_seqlens_k, max_seqlen_k = cache.get_flat_kv()
    assert cu_seqlens_k.tolist() == [0, 10]
    assert max_seqlen_k == 10
    assert k_flat[:4, 0].cpu().tolist() == [0.0, 1.0, 2.0, 3.0]


def test_batch_isolation_and_cu_seqlens():
    config = _build_config(num_layers=1, num_heads=2, capacities=[5, 5])
    cache = PyramidKVCache(
        config=config,
        batch_size=2,
        num_heads=2,
        head_dim=4,
        layer_idx=0,
        is_i2v=False,
        context_len=0,
    )

    k = torch.zeros(2, 5, 2, 4, dtype=torch.float32)
    k[1].fill_(1.0)
    cache.update(k, k)

    k_flat, _, cu_seqlens_k, max_seqlen_k = cache.get_flat_kv()
    assert cu_seqlens_k.tolist() == [0, 5, 10, 15, 20]
    assert max_seqlen_k == 5
    assert torch.all(k_flat[:10] == 0)
    assert torch.all(k_flat[10:] == 1)


def test_prompt_value_cache_defaults_to_disabled():
    config = _build_config(num_layers=1, num_heads=1, capacities=[5])
    cache = PyramidKVCache(
        config=config,
        batch_size=1,
        num_heads=1,
        head_dim=4,
        layer_idx=0,
    )

    assert cache.prompt_value_cache_enabled is False

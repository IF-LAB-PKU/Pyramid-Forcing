import pytest
import torch

from utils.teacache import TeaCacheController, get_wan_teacache_coefficients


def test_teacache_reuses_residual_below_threshold():
    cache = TeaCacheController(
        rel_l1_thresh=10.0,
        coefficients=[0.0, 0.0, 0.0, 0.0, 0.01],
        max_skip_steps=3,
    )
    emb0 = torch.ones(1, 2, 6, 4)
    emb1 = emb0 * 1.001
    x0 = torch.ones(1, 8, 4)
    x1 = x0 + 2.0

    assert cache.should_skip(emb0) is False
    cache.update_residual(x0, x1)

    assert cache.should_skip(emb1) is True
    torch.testing.assert_close(cache.apply_cached_residual(x0), x1)
    assert cache.pop_stats() == {"full_calls": 1, "skipped_calls": 1}


def test_teacache_forces_full_calc_and_resets_skip_streak():
    cache = TeaCacheController(
        rel_l1_thresh=10.0,
        coefficients=[0.0, 0.0, 0.0, 0.0, 0.01],
        max_skip_steps=1,
    )
    emb = torch.ones(1, 1, 6, 4)
    x0 = torch.zeros(1, 4, 4)
    cache.should_skip(emb)
    cache.update_residual(x0, x0 + 1)

    assert cache.should_skip(emb * 1.001) is True
    assert cache.should_skip(emb * 1.002, force_calc=True) is False
    cache.update_residual(x0, x0 + 2)
    assert cache.should_skip(emb * 1.003) is True


def test_teacache_respects_max_skip_steps():
    cache = TeaCacheController(
        rel_l1_thresh=10.0,
        coefficients=[0.0, 0.0, 0.0, 0.0, 0.01],
        max_skip_steps=1,
    )
    emb = torch.ones(1, 1, 6, 4)
    x = torch.zeros(1, 4, 4)
    cache.should_skip(emb)
    cache.update_residual(x, x + 1)

    assert cache.should_skip(emb * 1.001) is True
    assert cache.should_skip(emb * 1.002) is False


def test_wan_teacache_coefficients_known_variants():
    coeffs = get_wan_teacache_coefficients("Wan2.1-T2V-1.3B")
    ret_coeffs = get_wan_teacache_coefficients("wan2.1_t2v_14b", use_ret_steps=True)

    assert len(coeffs) == 5
    assert len(ret_coeffs) == 5
    assert coeffs != ret_coeffs


def test_wan_teacache_coefficients_reject_unknown_variant():
    with pytest.raises(ValueError, match="Unknown TeaCache variant"):
        get_wan_teacache_coefficients("not-a-wan-model")


"""Shared pytest fixtures and CLI flags for the Pyramid Forcing test suite."""
from __future__ import annotations

import os
import random

import numpy as np
import pytest
import torch


def pytest_addoption(parser):
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run tests marked @pytest.mark.slow (full GPU integration paths).",
    )


def pytest_collection_modifyitems(config, items):
    """Skip @pytest.mark.slow tests unless --run-slow is passed."""
    if config.getoption("--run-slow"):
        return
    skip_slow = pytest.mark.skip(reason="needs --run-slow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture(scope="session")
def device() -> torch.device:
    """Preferred device for tests; CUDA when available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def dtype() -> torch.dtype:
    """Default dtype used by Pyramid Forcing's inference path."""
    return torch.bfloat16 if torch.cuda.is_available() else torch.float32


@pytest.fixture(autouse=True)
def deterministic_seeds():
    """Reseed RNGs before every test so failures reproduce locally."""
    seed = int(os.environ.get("ADAHEAD_TEST_SEED", "0"))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    yield

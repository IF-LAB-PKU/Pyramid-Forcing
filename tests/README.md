# Pyramid Forcing tests

| Subset                 | Command                                         | Notes                                                |
|------------------------|-------------------------------------------------|------------------------------------------------------|
| CPU-safe (default)     | `uv run --no-sync pytest tests/ -v -m "not gpu"`          | Skips GPU-required and `slow` tests.                 |
| GPU integration        | `uv run --no-sync pytest tests/ -v -m "gpu"`              | Needs CUDA + `flash-attn`. Verified on H200.         |
| Slow integration       | `uv run --no-sync pytest tests/ --run-slow -v`            | Full ~10 min integration suite (GPU required).       |

## Markers

The suite uses two pytest markers (registered in `pyproject.toml` under
`[tool.pytest.ini_options]` with `--strict-markers` enabled):

- `@pytest.mark.gpu` — requires a CUDA GPU and `flash-attn`.
- `@pytest.mark.slow` — multi-minute integration test. Skipped unless
  `--run-slow` is passed (logic in `tests/conftest.py`).

`tests/conftest.py` also exposes:

- `device` fixture (CUDA when available, otherwise CPU);
- `dtype` fixture (bf16 on GPU, fp32 on CPU);
- an `autouse` deterministic-seed fixture controlled by
  `ADAHEAD_TEST_SEED` (default `0`).

## CI

`.github/workflows/ci.yml` runs the CPU-only subset on every push and PR.
GPU tests are not run in CI — GitHub Actions does not offer free GPU
runners. The workflow also runs a privacy scan that fails on the in-house
path / hostname patterns scrubbed during release prep.

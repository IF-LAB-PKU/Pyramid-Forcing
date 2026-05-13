# Pyramid Forcing

**Pyramid Forcing: Head-Aware Pyramid KV Cache Policy for High-Quality Long Video Generation**

Project page: <https://if-lab-pku.github.io/Pyramid-Forcing/>

Training-free, head-aware pyramidal KVCache framework for autoregressive long
video generation. Built on top of
[Self Forcing](https://github.com/guandeh17/Self-Forcing) /
Causal Forcing with the [Wan2.1-T2V-1.3B](https://github.com/Wan-Video/Wan2.1)
base model.

## Requirements

- Python 3.10, CUDA 12.1, PyTorch 2.5.x, `flash-attn` 2.8.3
- GPU with 80 GB+ VRAM (developed on H200)

## Install

The project-page demo videos under `demo/static/videos/` are stored via
Git LFS. Install `git-lfs` (`apt install git-lfs` / `brew install
git-lfs`) and run `git lfs install` **before** cloning — otherwise the
mp4 files come down as ~130-byte pointer stubs.

```bash
git lfs install
git clone https://github.com/if-lab-pku/Pyramid-Forcing.git
cd Pyramid-Forcing
uv sync
```

If you only need the code (not the demo videos), you can skip LFS with
`GIT_LFS_SKIP_SMUDGE=1 git clone …` and the rest of the repo still
works.

If `flash-attn` fails to build, install the prebuilt wheel:

```bash
uv pip install flash-attn --no-build-isolation \
    --find-links https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

The wheel covers Linux x86_64 + Python 3.10 + CUDA 12.x + torch 2.5
(`cxx11abi=False`). Other platforms fall back to a ~30 min source build.
If the wheel URL ever 404s, pick the equivalent `2.8.3+cu12torch2.5...`
asset from the [flash-attn releases page](https://github.com/Dao-AILab/flash-attention/releases).
Numerical reproducibility of the paper assumes flash-attn 2.8.3 (not
`2.8.3.postN`).

## Download Weights

```bash
hf download Wan-AI/Wan2.1-T2V-1.3B --local-dir wan_models/Wan2.1-T2V-1.3B
hf download gdhe17/Self-Forcing checkpoints/self_forcing_dmd.pt --local-dir .
```

## Run

Single-GPU inference:

```bash
uv run --no-sync python inference.py \
    --config_path configs/pyramid-forcing.yaml \
    --checkpoint_path checkpoints/self_forcing_dmd.pt \
    --data_path prompts/MovieGenVideoBench_num32.txt \
    --output_folder videos/quick_test \
    --use_ema
```

Or use the wrapper (auto-handles single / multi-GPU and env vars):

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_pyramid_forcing.sh \
    --config configs/pyramid-forcing.yaml \
    --output-dir videos/Exp_release_120f \
    --num-frames 120
```

## Configs

| File | Use |
|---|---|
| [`configs/pyramid-forcing.yaml`](configs/pyramid-forcing.yaml) | Recommended Pyramid Forcing inference |
| [`configs/self-forcing.yaml`](configs/self-forcing.yaml) | Self Forcing baseline (no Pyramid Forcing) |

Per-head classification labels live under
[`configs/head_configs/`](configs/head_configs/). Two prompt sets are
shipped under [`prompts/`](prompts/) for quick smoke tests
(`MovieGenVideoBench_num32.txt`) and full VBench evaluation
(`MovieGenVideoBench_num128.txt`). The training pipeline (`train.py`) is
research-only and not required for inference.

## Citation

BibTeX coming soon.

## Acknowledgements

Built on [Wan2.1](https://github.com/Wan-Video/Wan2.1) (Apache-2.0) and
[Self-Forcing](https://github.com/guandeh17/Self-Forcing) (Apache-2.0). The
Self Forcing checkpoint is downloaded from HuggingFace
(`gdhe17/Self-Forcing`) and is not redistributed here.

## License

Apache-2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

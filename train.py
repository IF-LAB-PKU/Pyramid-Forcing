"""Self-Forcing training entry point (research-only).

The training pipeline depends on a preprocessed LMDB dataset of (prompt,
ODE-init-latent) pairs that is *not* included in the release. The public
release of Pyramid Forcing targets inference with the pretrained Self-Forcing
DMD checkpoint (see ``README.md`` → Quick Start); reproducing training from
scratch requires building the dataset yourself following the upstream
Self-Forcing data-prep instructions
(https://github.com/guandeh17/Self-Forcing).

If you don't already have the LMDB dataset, run inference instead:

    uv run --no-sync python inference.py \\
        --config_path configs/pyramid-forcing.yaml \\
        --checkpoint_path checkpoints/self_forcing_dmd.pt \\
        --data_path prompts/MovieGenVideoBench_num32.txt \\
        --output_folder videos/quick_test --use_ema
"""
import argparse
import os
from omegaconf import OmegaConf
import wandb

from trainer import DiffusionTrainer, GANTrainer, ODETrainer, ScoreDistillationTrainer


def _check_lmdb_dataset(config) -> None:
    """Fail fast with a clear message when the training LMDB is missing."""
    dataset_path = getattr(config, "data_path", None)
    if dataset_path and not os.path.exists(dataset_path):
        raise SystemExit(
            "Training dataset not found at:\n"
            f"    {dataset_path}\n\n"
            "train.py expects a preprocessed LMDB dataset of (prompt, "
            "ODE-init-latent) pairs. The release ships inference-only — "
            "follow the upstream Self-Forcing data-prep instructions "
            "(https://github.com/guandeh17/Self-Forcing), or run "
            "inference.py if you only need to generate videos."
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Self-Forcing training driver. Requires a preprocessed LMDB "
            "dataset built per the upstream Self-Forcing data-prep "
            "instructions."
        ),
    )
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--no_visualize", action="store_true")
    parser.add_argument("--logdir", type=str, default="", help="Path to the directory to save logs")
    parser.add_argument("--wandb-save-dir", type=str, default="", help="Path to the directory to save wandb logs")
    parser.add_argument("--disable-wandb", action="store_true")

    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)
    default_config = OmegaConf.load("configs/default_config.yaml")
    config = OmegaConf.merge(default_config, config)
    config.no_save = args.no_save
    config.no_visualize = args.no_visualize

    # get the filename of config_path
    config_name = os.path.basename(args.config_path).split(".")[0]
    config.config_name = config_name
    config.logdir = args.logdir
    config.wandb_save_dir = args.wandb_save_dir
    config.disable_wandb = args.disable_wandb

    _check_lmdb_dataset(config)

    if config.trainer == "diffusion":
        trainer = DiffusionTrainer(config)
    elif config.trainer == "gan":
        trainer = GANTrainer(config)
    elif config.trainer == "ode":
        trainer = ODETrainer(config)
    elif config.trainer == "score_distillation":
        trainer = ScoreDistillationTrainer(config)
    trainer.train()

    wandb.finish()


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from handwriting_ai.config import ExperimentConfig, load_config
from handwriting_ai.checkpoint import load_checkpoint
from handwriting_ai.data.dataset import dataset_summary
from handwriting_ai.training.autoencoder import train_autoencoder
from handwriting_ai.training.generator import train_generator
from handwriting_ai.training.recognizer import train_recognizer


PROFILES = {
    "gtx1660": ROOT_DIR / "configs" / "gtx1660.toml",
    "cpu_smoke": ROOT_DIR / "configs" / "cpu_smoke.toml",
}


def resolve_config_path(profile: str, config_path: str | None) -> Path:
    if config_path:
        return Path(config_path).expanduser().resolve()
    try:
        return PROFILES[profile]
    except KeyError as exc:
        choices = ", ".join(sorted(PROFILES))
        raise ValueError(f"Unknown profile {profile!r}. Available profiles: {choices}") from exc


def default_autoencoder_checkpoint(config: ExperimentConfig) -> Path:
    return config.run.out_dir / "autoencoder" / "best.pt"


def print_dataset_summary(config: ExperimentConfig) -> None:
    summary = dataset_summary(config.data)
    print("Dataset after preprocessing:")
    print(f"  raw examples:   {summary['raw_examples']}")
    print(f"  train examples: {summary['train_examples']}")
    print(f"  val examples:   {summary['val_examples']}")
    print(
        "  points:         "
        f"{summary['points_min']} min / {summary['points_median']:.1f} median / "
        f"{summary['points_max']} max"
    )
    print(
        "  text length:    "
        f"{summary['text_min']} min / {summary['text_median']:.1f} median / "
        f"{summary['text_max']} max"
    )


def require_checkpoint(path: Path, *, purpose: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"{purpose} checkpoint was not found: {path}\n"
            "Train the previous stage first or pass an explicit checkpoint path."
        )
    return path


def is_current_generator_checkpoint(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        payload = load_checkpoint(path, map_location="cpu")
    except Exception:
        return False
    return payload.get("model_type") == "latent_regressor" and "latent_normalization" in payload


def run_training(args: argparse.Namespace) -> None:
    config_path = resolve_config_path(args.profile, args.config)
    config = load_config(config_path)
    print(f"Profile: {args.profile}")
    print(f"Config:  {config_path}")
    print(f"Output:  {config.run.out_dir}")

    if args.stage in {"audit", "all", "autoencoder", "generator", "recognizer"}:
        print_dataset_summary(config)

    if args.stage == "audit":
        return

    autoencoder_checkpoint: Path | None = (
        Path(args.autoencoder_checkpoint).expanduser() if args.autoencoder_checkpoint else None
    )

    if args.stage in {"autoencoder", "all"}:
        if args.skip_existing and default_autoencoder_checkpoint(config).exists():
            autoencoder_checkpoint = default_autoencoder_checkpoint(config)
            print(f"Autoencoder already exists, skipping: {autoencoder_checkpoint}")
        else:
            autoencoder_checkpoint = train_autoencoder(config)

    if args.stage in {"generator", "all"}:
        if autoencoder_checkpoint is None:
            autoencoder_checkpoint = default_autoencoder_checkpoint(config)
        require_checkpoint(autoencoder_checkpoint, purpose="Autoencoder")
        generator_path = config.run.out_dir / "generator" / "best.pt"
        if args.skip_existing and is_current_generator_checkpoint(generator_path):
            print(f"Generator already exists, skipping: {generator_path}")
        else:
            if args.skip_existing and generator_path.exists():
                print(f"Generator checkpoint is legacy or incomplete, retraining: {generator_path}")
            train_generator(config, autoencoder_checkpoint)

    if args.stage in {"recognizer", "all"}:
        recognizer_path = config.run.out_dir / "recognizer" / "best.pt"
        if args.skip_existing and recognizer_path.exists():
            print(f"Recognizer already exists, skipping: {recognizer_path}")
        else:
            train_recognizer(config)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train the handwriting generation pipeline. "
            "Default stage is autoencoder because it is the first required model."
        )
    )
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILES),
        default="gtx1660",
        help="Convenient hardware/config profile.",
    )
    parser.add_argument(
        "--config",
        help="Explicit TOML config path. Overrides --profile config path.",
    )
    parser.add_argument(
        "--stage",
        choices=["audit", "autoencoder", "generator", "recognizer", "all"],
        default="autoencoder",
        help="Training stage to run.",
    )
    parser.add_argument(
        "--autoencoder-checkpoint",
        help="Autoencoder checkpoint for generator training. Defaults to runs/<profile>/autoencoder/best.pt.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Do not retrain a stage if its default best.pt checkpoint already exists.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run_training(args)
    except (FileNotFoundError, ValueError) as exc:
        parser.exit(2, f"{exc}\n")


if __name__ == "__main__":
    main()

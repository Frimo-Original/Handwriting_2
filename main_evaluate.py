from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch


ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from handwriting_ai.checkpoint import load_checkpoint
from handwriting_ai.config import ExperimentConfig, load_config
from handwriting_ai.data.codec import decode_tokens
from handwriting_ai.data.dataset import NormalizationStats, build_datasets
from handwriting_ai.data.rendering import render_points_to_image
from handwriting_ai.data.transforms import deltas_to_points
from handwriting_ai.generator_checkpoint import generator_checkpoint_problem, is_current_generator_payload
from handwriting_ai.inference import generate_points
from handwriting_ai.models import InkAutoencoder
from handwriting_ai.seed import resolve_device


PROFILES = {
    "gtx1660": ROOT_DIR / "configs" / "gtx1660.toml",
    "cpu_smoke": ROOT_DIR / "configs" / "cpu_smoke.toml",
}


def resolve_config_path(profile: str, config_path: str | None) -> Path:
    if config_path:
        return Path(config_path).expanduser().resolve()
    return PROFILES[profile]


def default_autoencoder_checkpoint(config: ExperimentConfig) -> Path:
    return config.run.out_dir / "autoencoder" / "best.pt"


def default_generator_checkpoint(config: ExperimentConfig, selection: str = "best") -> Path:
    filenames = {
        "best": "best.pt",
        "last": "last.pt",
        "train_best": "train_best.pt",
    }
    return config.run.out_dir / "generator" / filenames[selection]


def default_recognizer_checkpoint(config: ExperimentConfig) -> Path:
    return config.run.out_dir / "recognizer" / "best.pt"


def require_checkpoint(path: Path, *, purpose: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{purpose} checkpoint was not found: {path}")
    return path


def require_current_generator_checkpoint(path: Path) -> None:
    payload = load_checkpoint(path, map_location="cpu")
    if is_current_generator_payload(payload):
        return
    raise ValueError(
        f"Generator checkpoint is outdated or unsupported: {path}\n"
        f"Reason: {generator_checkpoint_problem(payload)}\n"
        "Retrain the generator with: python main_training.py --stage generator"
    )


def load_autoencoder(path: Path, device: torch.device) -> tuple[InkAutoencoder, NormalizationStats]:
    payload = load_checkpoint(path, map_location=device)
    model = InkAutoencoder(**payload["model_kwargs"]).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, NormalizationStats.from_dict(payload["normalization"])


def selected_items(config: ExperimentConfig, stats: NormalizationStats, split: str, count: int):
    train, val, _ = build_datasets(config.data, stats=stats)
    dataset = train if split == "train" else val
    return [dataset[i] for i in range(min(count, len(dataset)))]


def evaluate_autoencoder(args: argparse.Namespace, config: ExperimentConfig, device: torch.device) -> None:
    checkpoint = Path(args.autoencoder_checkpoint).expanduser() if args.autoencoder_checkpoint else default_autoencoder_checkpoint(config)
    require_checkpoint(checkpoint, purpose="Autoencoder")
    model, stats = load_autoencoder(checkpoint, device)
    out_dir = Path(args.out_dir) / args.split / "autoencoder"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Autoencoder checkpoint: {checkpoint}")

    for item in selected_items(config, stats, args.split, args.count):
        points = item["points"].unsqueeze(0).to(device)
        lengths = torch.tensor([points.shape[1]], dtype=torch.long, device=device)
        with torch.no_grad():
            reconstruction = model(points, lengths).reconstruction[0, : points.shape[1]].cpu().numpy()
        reconstruction[:, 2] = (1.0 / (1.0 + np.exp(-reconstruction[:, 2])) > args.pen_threshold).astype(np.float32)
        original_points = deltas_to_points(item["points"].numpy(), mean=stats.mean, std=stats.std)
        reconstructed_points = deltas_to_points(reconstruction, mean=stats.mean, std=stats.std)
        text = decode_tokens(item["text"].numpy().tolist())
        sample_prefix = f"sample_{int(item['sample_id']):04d}"
        render_points_to_image(original_points).save(out_dir / f"{sample_prefix}_original.png")
        render_points_to_image(reconstructed_points).save(out_dir / f"{sample_prefix}_reconstruction.png")
        print(f"{sample_prefix}: {text!r}")
    print(f"Saved autoencoder renders to {out_dir}")


def evaluate_generator(args: argparse.Namespace, config: ExperimentConfig, device: torch.device) -> None:
    autoencoder_checkpoint = Path(args.autoencoder_checkpoint).expanduser() if args.autoencoder_checkpoint else default_autoencoder_checkpoint(config)
    generator_checkpoint = (
        Path(args.generator_checkpoint).expanduser()
        if args.generator_checkpoint
        else default_generator_checkpoint(config, args.generator_selection)
    )
    require_checkpoint(generator_checkpoint, purpose="Generator")
    require_current_generator_checkpoint(generator_checkpoint)
    generator_payload = load_checkpoint(generator_checkpoint, map_location="cpu")
    generator_model_type = generator_payload.get("model_type", "latent_flow")
    if generator_model_type == "trajectory_generator":
        stats = NormalizationStats.from_dict(generator_payload["normalization"])
        downsample_factor = None
    else:
        require_checkpoint(autoencoder_checkpoint, purpose="Autoencoder")
        _, stats = load_autoencoder(autoencoder_checkpoint, device)
        autoencoder_payload = load_checkpoint(autoencoder_checkpoint, map_location="cpu")
        downsample_factor = int(autoencoder_payload["model_kwargs"]["downsample_factor"])
    out_dir = Path(args.out_dir) / args.split / "generator"
    out_dir.mkdir(parents=True, exist_ok=True)
    if generator_model_type == "trajectory_generator":
        print("Autoencoder checkpoint: not used for trajectory_generator inference")
    else:
        print(f"Autoencoder checkpoint: {autoencoder_checkpoint}")
    print(f"Generator checkpoint:   {generator_checkpoint}")

    for item in selected_items(config, stats, args.split, args.count):
        text = decode_tokens(item["text"].numpy().tolist())
        original_points = deltas_to_points(item["points"].numpy(), mean=stats.mean, std=stats.std)
        latent_length = None
        if not args.use_predicted_length:
            if generator_model_type == "trajectory_generator":
                latent_length = int(len(item["points"]))
            else:
                assert downsample_factor is not None
                latent_length = int(math.ceil(len(item["points"]) / downsample_factor))
        generated_points = generate_points(
            text=text,
            autoencoder_checkpoint=autoencoder_checkpoint,
            generator_checkpoint=generator_checkpoint,
            device=device,
            steps=args.steps if args.steps is not None else config.generator.flow_steps,
            temperature=args.temperature if args.temperature is not None else config.generator.temperature,
            latent_length=latent_length,
            max_latent_length=args.max_latent_length,
            pen_threshold=args.pen_threshold,
        )
        sample_prefix = f"sample_{int(item['sample_id']):04d}"
        render_points_to_image(original_points).save(out_dir / f"{sample_prefix}_original.png")
        render_points_to_image(generated_points).save(out_dir / f"{sample_prefix}_generated.png")
        print(f"{sample_prefix}: {text!r} generated_points={len(generated_points)}")
    print(f"Saved generator renders to {out_dir}")


def evaluate_recognizer(args: argparse.Namespace, config: ExperimentConfig) -> None:
    checkpoint = Path(args.recognizer_checkpoint).expanduser() if args.recognizer_checkpoint else default_recognizer_checkpoint(config)
    require_checkpoint(checkpoint, purpose="Recognizer")
    payload = load_checkpoint(checkpoint, map_location="cpu")
    print(f"Recognizer checkpoint: {checkpoint}")
    print(f"Epoch: {payload.get('epoch')}")
    print(f"Validation metrics: {payload.get('val_metrics')}")


def run(args: argparse.Namespace) -> None:
    config_path = resolve_config_path(args.profile, args.config)
    config = load_config(config_path)
    device = resolve_device(args.device or config.hardware.device)
    print(f"Profile: {args.profile}")
    print(f"Config:  {config_path}")
    print(f"Device:  {device}")

    if args.stage in {"autoencoder", "all"}:
        evaluate_autoencoder(args, config, device)
    if args.stage in {"generator", "all"}:
        evaluate_generator(args, config, device)
    if args.stage in {"recognizer", "all"}:
        evaluate_recognizer(args, config)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render diagnostics for trained handwriting models.")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="gtx1660")
    parser.add_argument("--config")
    parser.add_argument("--stage", choices=["autoencoder", "generator", "recognizer", "all"], default="all")
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--out-dir", default="outputs/eval")
    parser.add_argument("--device")
    parser.add_argument("--autoencoder-checkpoint")
    parser.add_argument("--generator-checkpoint")
    parser.add_argument(
        "--generator-selection",
        choices=["best", "last", "train_best"],
        default="best",
        help="Default generator checkpoint to use when --generator-checkpoint is not set.",
    )
    parser.add_argument("--recognizer-checkpoint")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument(
        "--max-latent-length",
        "--max-point-length",
        dest="max_latent_length",
        type=int,
        default=768,
        help="Maximum generated latent length.",
    )
    parser.add_argument("--pen-threshold", type=float, default=0.5)
    parser.add_argument(
        "--use-predicted-length",
        action="store_true",
        help="Use generator length prediction. By default target sample length is used for easier debugging.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run(args)
    except (FileNotFoundError, ValueError) as exc:
        parser.exit(2, f"{exc}\n")


if __name__ == "__main__":
    main()

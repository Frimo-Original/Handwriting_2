from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
import math
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from handwriting_ai.config import ExperimentConfig, load_config
from handwriting_ai.checkpoint import load_checkpoint
from handwriting_ai.generator_checkpoint import generator_checkpoint_problem, is_current_generator_payload
from handwriting_ai.inference import generate_points, save_points_json, save_points_png
from handwriting_ai.seed import resolve_device


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


def default_generator_checkpoint(config: ExperimentConfig, selection: str = "best") -> Path:
    filenames = {
        "best": "best.pt",
        "last": "last.pt",
        "train_best": "train_best.pt",
    }
    trajectory_path = config.run.out_dir / "trajectory_generator" / filenames[selection]
    if trajectory_path.exists():
        return trajectory_path
    return config.run.out_dir / "generator" / filenames[selection]


def make_safe_name(text: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", text.strip()).strip("_").lower()
    if not normalized:
        normalized = "sample"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{normalized[:32]}"


def resolve_output_paths(out_dir: Path, name: str | None, text: str) -> tuple[Path, Path]:
    sample_name = name or make_safe_name(text)
    return out_dir / f"{sample_name}.json", out_dir / f"{sample_name}.png"


def require_checkpoint(path: Path, *, purpose: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"{purpose} checkpoint was not found: {path}\n"
            "Train the required stages first or pass an explicit checkpoint path."
        )
    return path


def validate_generator_checkpoint(path: Path, *, allow_legacy_flow: bool) -> dict:
    payload = load_checkpoint(path, map_location="cpu")
    model_type = payload.get("model_type", "latent_flow")
    if is_current_generator_payload(payload):
        return payload
    if allow_legacy_flow and model_type == "latent_flow":
        print(f"Warning: using legacy flow generator checkpoint: {path}")
        return payload
    raise ValueError(
        f"Generator checkpoint is outdated or unsupported: {path}\n"
        f"Reason: {generator_checkpoint_problem(payload)}\n"
        "Retrain the generator with: python main_training.py --stage generator\n"
        "If you explicitly want to inspect the old flow checkpoint, pass --allow-legacy-flow."
    )


def run_generation(args: argparse.Namespace) -> None:
    text = args.text_flag or args.text or "Привет, это тестовая строка."
    config_path = resolve_config_path(args.profile, args.config)
    config = load_config(config_path)

    autoencoder_checkpoint = Path(args.autoencoder_checkpoint).expanduser() if args.autoencoder_checkpoint else default_autoencoder_checkpoint(config)
    generator_checkpoint = (
        Path(args.generator_checkpoint).expanduser()
        if args.generator_checkpoint
        else default_generator_checkpoint(config, args.generator_selection)
    )
    require_checkpoint(generator_checkpoint, purpose="Generator")
    generator_payload = validate_generator_checkpoint(generator_checkpoint, allow_legacy_flow=args.allow_legacy_flow)
    if generator_payload.get("model_type", "latent_flow") != "trajectory_generator":
        require_checkpoint(autoencoder_checkpoint, purpose="Autoencoder")
        autoencoder_payload = load_checkpoint(autoencoder_checkpoint, map_location="cpu")
        downsample_factor = int(autoencoder_payload["model_kwargs"]["downsample_factor"])
        max_latent_length = args.max_latent_length or math.ceil(
            (config.data.max_points or 4096) / downsample_factor
        )
    else:
        max_latent_length = args.max_latent_length or config.data.max_points or 4096

    out_dir = Path(args.out_dir).expanduser()
    out_json, out_png = resolve_output_paths(out_dir, args.name, text)
    device = resolve_device(args.device or config.hardware.device)
    points = generate_points(
        text=text,
        autoencoder_checkpoint=autoencoder_checkpoint,
        generator_checkpoint=generator_checkpoint,
        device=device,
        steps=args.steps if args.steps is not None else config.generator.flow_steps,
        temperature=args.temperature if args.temperature is not None else config.generator.temperature,
        latent_length=args.latent_length,
        max_latent_length=max_latent_length,
        pen_threshold=args.pen_threshold,
    )
    save_points_json(out_json, points)
    save_points_png(out_png, points)
    print(f"Text:  {text}")
    print(f"JSON:  {out_json}")
    print(f"PNG:   {out_png}")
    print(f"Points: {len(points)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate handwriting samples from trained checkpoints.")
    parser.add_argument("text", nargs="?", help="Text to write. You can also use --text.")
    parser.add_argument("--text", dest="text_flag", help="Text to write. Overrides positional text.")
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILES),
        default="gtx1660",
        help="Convenient hardware/config profile.",
    )
    parser.add_argument("--config", help="Explicit TOML config path. Overrides --profile config path.")
    parser.add_argument("--autoencoder-checkpoint", help="Defaults to runs/<profile>/autoencoder/best.pt.")
    parser.add_argument("--generator-checkpoint", help="Defaults to runs/<profile>/generator/best.pt.")
    parser.add_argument(
        "--generator-selection",
        choices=["best", "last", "train_best"],
        default="best",
        help="Default generator checkpoint to use when --generator-checkpoint is not set.",
    )
    parser.add_argument(
        "--allow-legacy-flow",
        action="store_true",
        help=(
            "Allow old LatentFlowTransformer checkpoints. "
            "New training uses content_aligned_latent_flow."
        ),
    )
    parser.add_argument("--out-dir", default="outputs/generated", help="Directory for JSON and PNG outputs.")
    parser.add_argument("--name", help="Output filename without extension.")
    parser.add_argument("--device", help="cpu, cuda, cuda:0, or auto. Defaults to config hardware.device.")
    parser.add_argument("--steps", type=int, help="Rectified-flow integration steps.")
    parser.add_argument("--temperature", type=float, help="Sampling temperature. Defaults to config generator.temperature.")
    parser.add_argument(
        "--latent-length",
        "--point-length",
        dest="latent_length",
        type=int,
        help="Force latent sequence length. Intended for diagnostics; normally durations are predicted.",
    )
    parser.add_argument(
        "--max-latent-length",
        "--max-point-length",
        dest="max_latent_length",
        type=int,
        default=None,
        help="Maximum generated latent length. Defaults to the data-profile limit.",
    )
    parser.add_argument("--pen-threshold", type=float, default=0.5)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run_generation(args)
    except (FileNotFoundError, ValueError) as exc:
        parser.exit(2, f"{exc}\n")


if __name__ == "__main__":
    main()

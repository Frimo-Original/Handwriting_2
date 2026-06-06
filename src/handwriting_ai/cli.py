from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from handwriting_ai.checkpoint import load_checkpoint
from handwriting_ai.config import load_config
from handwriting_ai.data.dataset import dataset_summary
from handwriting_ai.data.rendering import render_points_to_image
from handwriting_ai.data.transforms import validate_points
from handwriting_ai.generator_checkpoint import generator_checkpoint_problem, is_current_generator_payload
from handwriting_ai.inference import generate_points, save_points_json, save_points_png
from handwriting_ai.seed import resolve_device
from handwriting_ai.training.autoencoder import train_autoencoder
from handwriting_ai.training.generator import train_generator
from handwriting_ai.training.recognizer import train_recognizer


def _add_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="configs/gtx1660.toml", help="Path to TOML experiment config.")


def cmd_audit_data(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    summary = dataset_summary(config.data)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


def cmd_train_autoencoder(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    path = train_autoencoder(config, args.recognizer_checkpoint)
    print(f"Best autoencoder checkpoint: {path}")


def cmd_train_generator(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    path = train_generator(
        config,
        args.autoencoder_checkpoint,
        args.recognizer_checkpoint,
    )
    print(f"Best generator checkpoint: {path}")


def cmd_train_recognizer(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    path = train_recognizer(config)
    print(f"Best recognizer checkpoint: {path}")


def cmd_generate(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    payload = load_checkpoint(args.generator_checkpoint, map_location="cpu")
    if not is_current_generator_payload(payload):
        raise ValueError(
            f"Generator checkpoint is outdated or unsupported: {args.generator_checkpoint}\n"
            f"Reason: {generator_checkpoint_problem(payload)}\n"
            "Retrain the generator with: python main_training.py --stage generator"
        )
    if payload.get("model_type") != "trajectory_generator" and not args.autoencoder_checkpoint:
        raise ValueError("--autoencoder-checkpoint is required for latent generators")
    device = resolve_device(args.device or config.hardware.device)
    points = generate_points(
        text=args.text,
        autoencoder_checkpoint=args.autoencoder_checkpoint,
        generator_checkpoint=args.generator_checkpoint,
        device=device,
        steps=args.steps if args.steps is not None else config.generator.flow_steps,
        temperature=args.temperature if args.temperature is not None else config.generator.temperature,
        latent_length=args.latent_length,
        max_latent_length=args.max_latent_length,
        pen_threshold=args.pen_threshold,
    )
    save_points_json(args.out_json, points)
    if args.out_png:
        save_points_png(args.out_png, points)
    print(f"Saved {len(points)} points to {args.out_json}")


def cmd_render(args: argparse.Namespace) -> None:
    with Path(args.json_path).open("r", encoding="utf-8") as fh:
        points = validate_points(np.asarray(json.load(fh), dtype=np.float32))
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    render_points_to_image(points).save(output)
    print(f"Rendered {args.json_path} -> {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="handwriting-ai")
    sub = parser.add_subparsers(dest="command", required=True)

    audit = sub.add_parser("audit-data", help="Print dataset statistics after configured preprocessing.")
    _add_config(audit)
    audit.set_defaults(func=cmd_audit_data)

    ae = sub.add_parser(
        "train-autoencoder",
        help="Train the local content-preserving autoencoder.",
    )
    _add_config(ae)
    ae.add_argument("--recognizer-checkpoint", required=True)
    ae.set_defaults(func=cmd_train_autoencoder)

    gen = sub.add_parser(
        "train-generator",
        help="Train CTC-aligned duration-conditioned latent flow.",
    )
    _add_config(gen)
    gen.add_argument("--autoencoder-checkpoint", required=True)
    gen.add_argument("--recognizer-checkpoint", required=True)
    gen.set_defaults(func=cmd_train_generator)

    rec = sub.add_parser("train-recognizer", help="Train CTC trajectory recognizer.")
    _add_config(rec)
    rec.set_defaults(func=cmd_train_recognizer)

    generate = sub.add_parser("generate", help="Generate a plotter-ready trajectory from text.")
    _add_config(generate)
    generate.add_argument("--text", required=True)
    generate.add_argument(
        "--autoencoder-checkpoint",
        help="Required for content_aligned_latent_flow generation.",
    )
    generate.add_argument("--generator-checkpoint", required=True)
    generate.add_argument("--out-json", required=True)
    generate.add_argument("--out-png")
    generate.add_argument("--device")
    generate.add_argument("--steps", type=int)
    generate.add_argument("--temperature", type=float)
    generate.add_argument(
        "--latent-length",
        "--point-length",
        dest="latent_length",
        type=int,
        help="Force latent sequence length for diagnostics.",
    )
    generate.add_argument(
        "--max-latent-length",
        "--max-point-length",
        dest="max_latent_length",
        type=int,
        default=1024,
    )
    generate.add_argument("--pen-threshold", type=float, default=0.5)
    generate.set_defaults(func=cmd_generate)

    render = sub.add_parser("render", help="Render a JSON trajectory to PNG.")
    render.add_argument("--json-path", required=True)
    render.add_argument("--out", required=True)
    render.set_defaults(func=cmd_render)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)

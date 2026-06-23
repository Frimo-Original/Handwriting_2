from __future__ import annotations

from typing import Any


CURRENT_GENERATOR_TRAINING_VERSION = 7
CURRENT_TRAJECTORY_GENERATOR_TRAINING_VERSION = 1


def generator_training_version(payload: dict[str, Any]) -> int:
    value = payload.get("generator_training_version", 0)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def trajectory_generator_training_version(payload: dict[str, Any]) -> int:
    value = payload.get("trajectory_generator_training_version", 0)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def is_current_generator_payload(payload: dict[str, Any]) -> bool:
    model_type = payload.get("model_type")
    if model_type == "content_aligned_latent_flow":
        return generator_training_version(payload) >= CURRENT_GENERATOR_TRAINING_VERSION
    if model_type == "trajectory_generator":
        return (
            trajectory_generator_training_version(payload)
            >= CURRENT_TRAJECTORY_GENERATOR_TRAINING_VERSION
        )
    return False


def generator_checkpoint_problem(payload: dict[str, Any]) -> str:
    model_type = payload.get("model_type", "latent_flow")
    if model_type == "content_aligned_latent_flow":
        version = generator_training_version(payload)
        if version < CURRENT_GENERATOR_TRAINING_VERSION:
            return (
                f"generator_training_version={version}, "
                f"expected >= {CURRENT_GENERATOR_TRAINING_VERSION}"
            )
        return "unknown generator checkpoint problem"
    if model_type == "trajectory_generator":
        version = trajectory_generator_training_version(payload)
        if version < CURRENT_TRAJECTORY_GENERATOR_TRAINING_VERSION:
            return (
                f"trajectory_generator_training_version={version}, "
                f"expected >= {CURRENT_TRAJECTORY_GENERATOR_TRAINING_VERSION}"
            )
        return "unknown trajectory generator checkpoint problem"
    return (
        f"model_type={model_type!r}, "
        "expected 'content_aligned_latent_flow' or 'trajectory_generator'"
    )

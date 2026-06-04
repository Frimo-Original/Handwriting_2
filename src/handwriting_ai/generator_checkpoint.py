from __future__ import annotations

from typing import Any


CURRENT_GENERATOR_TRAINING_VERSION = 3


def generator_training_version(payload: dict[str, Any]) -> int:
    value = payload.get("generator_training_version", 0)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def is_current_generator_payload(payload: dict[str, Any]) -> bool:
    return (
        payload.get("model_type") == "latent_regressor"
        and "latent_normalization" in payload
        and generator_training_version(payload) >= CURRENT_GENERATOR_TRAINING_VERSION
    )


def generator_checkpoint_problem(payload: dict[str, Any]) -> str:
    model_type = payload.get("model_type", "latent_flow")
    if model_type != "latent_regressor":
        return f"model_type={model_type!r}, expected 'latent_regressor'"
    if "latent_normalization" not in payload:
        return "missing latent_normalization"
    version = generator_training_version(payload)
    if version < CURRENT_GENERATOR_TRAINING_VERSION:
        return (
            f"generator_training_version={version}, "
            f"expected >= {CURRENT_GENERATOR_TRAINING_VERSION}"
        )
    return "unknown generator checkpoint problem"

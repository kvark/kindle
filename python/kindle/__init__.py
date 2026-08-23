"""Pixel-first DreamerV3 agent with frozen DINOv3 perception."""

from ._native import (  # type: ignore[attr-defined]
    Agent,
    DINO_CHECKPOINT_REVISION,
    DINO_MODEL_ID,
    DREAMERV3_REVISION,
    default_config,
)

__all__ = [
    "Agent",
    "DINO_CHECKPOINT_REVISION",
    "DINO_MODEL_ID",
    "DREAMERV3_REVISION",
    "default_config",
]

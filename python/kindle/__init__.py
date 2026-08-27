"""Pixel-first DreamerV3 agent with frozen DINOv3 perception."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
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


def __getattr__(name: str) -> Any:
    """Load the GPU extension only when a native API symbol is requested.

    Pure-Python utilities such as Atari score aggregation can consequently run
    from a source checkout without first compiling the training extension.
    """
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    native = import_module("._native", __name__)
    for export in __all__:
        globals()[export] = getattr(native, export)
    return globals()[name]


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))

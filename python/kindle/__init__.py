"""kindle: a continually self-training RL agent.

This package re-exports the native pyo3 extension built via maturin.
"""

from .kindle import Agent, OBS_TOKEN_DIM  # type: ignore[attr-defined]

__all__ = ["Agent", "OBS_TOKEN_DIM"]

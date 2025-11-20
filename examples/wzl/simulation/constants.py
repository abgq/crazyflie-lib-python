"""Simulation-specific constants."""

from __future__ import annotations

from typing import Sequence, Tuple

CONTROL_RATE_HZ: int = 50
VISUAL_RATE_HZ: int = 60

INITIAL_DRONE_POS: Tuple[float, float, float] = (0.0, 0.0, 1.0)
INITIAL_DRONE_YAW: float = 0.0

ANCHORS: Sequence[Tuple[float, float, float]] = [
    (3.0, 0.0, 1.0),
]

DISTANCE_NOISE_STD: float = 0.1
VELOCITY_NOISE_STD: float = 0.02

SIM_ALGORITHM: str = "behavior"

__all__ = [
    "CONTROL_RATE_HZ",
    "VISUAL_RATE_HZ",
    "INITIAL_DRONE_POS",
    "INITIAL_DRONE_YAW",
    "ANCHORS",
    "DISTANCE_NOISE_STD",
    "VELOCITY_NOISE_STD",
    "SIM_ALGORITHM",
]

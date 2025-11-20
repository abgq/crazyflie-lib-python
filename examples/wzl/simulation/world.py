"""World state and kinematic integration for the simulator."""

from __future__ import annotations

import threading
from typing import Callable, Dict, Iterable, List, Sequence

import numpy as np

from . import constants


class World:
    """Simple kinematic world model."""

    def __init__(
        self,
        anchors: Sequence[Iterable[float]] | None = None,
        command_source: Callable[[], np.ndarray] | None = None,
    ) -> None:
        self._lock = threading.Lock()
        self._command_source = command_source or (lambda: np.zeros(3, dtype=float))
        self._fixed_z = constants.INITIAL_DRONE_POS[2]

        self.position = np.array(constants.INITIAL_DRONE_POS, dtype=float)
        self.velocity = np.zeros(3, dtype=float)
        self.yaw: float = constants.INITIAL_DRONE_YAW

        anchor_iter = anchors if anchors is not None else constants.ANCHORS
        self.anchors: List[np.ndarray] = [np.array(anchor, dtype=float) for anchor in anchor_iter]

    def attach_command_source(self, command_source: Callable[[], np.ndarray]) -> None:
        """Register a callable that returns the latest commanded velocity vector."""
        self._command_source = command_source

    def reset(self) -> None:
        """Reset the drone to its initial pose."""
        with self._lock:
            self.position = np.array(constants.INITIAL_DRONE_POS, dtype=float)
            self.velocity = np.zeros(3, dtype=float)
            self.yaw = constants.INITIAL_DRONE_YAW

    def set_altitude(self, z: float) -> None:
        """Directly set the world altitude (used by takeoff/land)."""
        with self._lock:
            self.position[2] = z

    def step(self, dt: float) -> None:
        """Advance the world state."""
        if dt <= 0.0:
            return

        command = np.array(self._command_source(), dtype=float)
        noise = np.random.normal(0.0, constants.VELOCITY_NOISE_STD, size=3)
        with self._lock:
            self.velocity = command + noise
            self.position = self.position + self.velocity * dt
            self.position[2] = self._fixed_z  # For now keep altitude constant

    def get_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Return copies of position and velocity."""
        with self._lock:
            return self.position.copy(), self.velocity.copy()

    def get_noisy_distances(self) -> Dict[int, float]:
        """Return noisy distances to each anchor."""
        with self._lock:
            pos = self.position.copy()

        distances: Dict[int, float] = {}
        for idx, anchor in enumerate(self.anchors):
            distance = np.linalg.norm(anchor - pos)
            noise = np.random.normal(0.0, constants.DISTANCE_NOISE_STD)
            distances[idx] = max(0.0, distance + noise)

        return distances


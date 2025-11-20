"""MotionCommander implementation for the simulator."""

from __future__ import annotations

import logging
import sys
import types
from typing import TYPE_CHECKING

import numpy as np

from . import constants

LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover - used for type checkers only
    from .sim_cf import SimCrazyflie


class SimMotionCommander:
    """Light-weight replacement for :class:`cflib.positioning.motion_commander.MotionCommander`."""

    VELOCITY = 0.2

    def __init__(self, cf: "SimCrazyflie", default_height: float = 0.3) -> None:
        self._cf = cf
        self._default_height = default_height
        self._is_flying = False

    # Public API -----------------------------------------------------------------

    def take_off(
        self,
        height: float | None = None,
        velocity: float | None = None,  # noqa: ARG002
    ) -> None:
        """Arm and move to the target hover height."""
        target = height if height is not None else self._default_height
        self._cf.platform.send_arming_request(True)
        self._cf.world.set_altitude(target)
        self._is_flying = True

    def land(self, height: float = 0.0, duration: float = 1.0) -> None:  # noqa: ARG002
        """Disarm and bring the drone down."""
        self.stop_linear_motion()
        self._cf.world.set_altitude(height)
        self._cf.platform.send_arming_request(False)
        self._is_flying = False

    def stop(self) -> None:
        """Alias for :meth:`stop_linear_motion`."""
        self.stop_linear_motion()

    def forward(self, velocity: float) -> None:
        self.start_linear_motion(velocity, 0.0, 0.0)

    def back(self, velocity: float) -> None:
        self.start_linear_motion(-velocity, 0.0, 0.0)

    def left(self, velocity: float) -> None:
        self.start_linear_motion(0.0, velocity, 0.0)

    def right(self, velocity: float) -> None:
        self.start_linear_motion(0.0, -velocity, 0.0)

    def up(self, velocity: float) -> None:
        self.start_linear_motion(0.0, 0.0, velocity)

    def down(self, velocity: float) -> None:
        self.start_linear_motion(0.0, 0.0, -velocity)

    def start_linear_motion(self, vx: float, vy: float, vz: float) -> None:
        vector = np.array([vx, vy, vz], dtype=float)
        self._cf.set_command_velocity(vector)

    def stop_linear_motion(self) -> None:
        self._cf.set_command_velocity(np.zeros(3, dtype=float))


def ensure_sim_motion_commander() -> None:
    """Inject :class:`SimMotionCommander` into the ``cflib`` namespace."""
    module_name = "cflib.positioning.motion_commander"
    sim_module = types.ModuleType(module_name)
    sim_module.MotionCommander = SimMotionCommander  # type: ignore[attr-defined]
    sys.modules[module_name] = sim_module
    LOGGER.debug("Patched %s with SimMotionCommander", module_name)


__all__ = ["SimMotionCommander", "ensure_sim_motion_commander"]

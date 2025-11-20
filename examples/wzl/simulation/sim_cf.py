"""Fake Crazyflie object that behaviors can interact with."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from .sim_commander import SimMotionCommander

LOGGER = logging.getLogger(__name__)


class _CommanderStub:
    def send_stop_setpoint(self) -> None:
        LOGGER.debug("SimCommander: send_stop_setpoint()")

    def send_notify_setpoint_stop(self) -> None:
        LOGGER.debug("SimCommander: send_notify_setpoint_stop()")


class _PlatformStub:
    def send_arming_request(self, armed: bool) -> None:
        LOGGER.debug("SimPlatform: send_arming_request(%s)", armed)


class SimCrazyflie:
    """Minimal Crazyflie facade expected by the behavior layer."""

    def __init__(self, world) -> None:
        self.world = world
        self.commander = _CommanderStub()
        self.platform = _PlatformStub()
        self._command_velocity = np.zeros(3, dtype=float)
        self.motion_commander = SimMotionCommander(self)

    # -- Velocity sharing -------------------------------------------------------

    def set_command_velocity(self, velocity: np.ndarray) -> None:
        self._command_velocity = np.array(velocity, dtype=float)

    def get_command_velocity(self) -> np.ndarray:
        return self._command_velocity.copy()

    def reset_velocity(self) -> None:
        self.set_command_velocity(np.zeros(3, dtype=float))


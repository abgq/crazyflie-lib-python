"""Abstract definitions for drone hardware interfaces."""

from __future__ import annotations

import abc
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import cflib.crazyflie

class DroneInterface(abc.ABC):
    """Abstract interface for drone control."""

    @abc.abstractmethod
    def arm(self, armed: bool) -> None:
        """Arm or disarm the drone motors."""

    @abc.abstractmethod
    def stop(self) -> None:
        """Send a stop setpoint to cut power immediately."""

    @abc.abstractmethod
    def cmd_hover(self, vx: float, vy: float, yaw_rate: float, z: float) -> None:
        """Send a hover setpoint command.

        Args:
            vx: Velocity in x-direction (m/s).
            vy: Velocity in y-direction (m/s).
            yaw_rate: Yaw rate (deg/s).
            z: Absolute height setpoint (m).
        """

    @abc.abstractmethod
    def sleep(self, duration: float) -> None:
        """Sleep for a specified duration (seconds)."""


class CrazyflieDrone(DroneInterface):
    """Concrete implementation of DroneInterface using cflib."""

    def __init__(self, cf: cflib.crazyflie.Crazyflie) -> None:
        self._cf = cf

    def arm(self, armed: bool) -> None:
        self._cf.platform.send_arming_request(armed)

    def stop(self) -> None:
        self._cf.commander.send_stop_setpoint()

    def cmd_hover(self, vx: float, vy: float, yaw_rate: float, z: float) -> None:
        self._cf.commander.send_hover_setpoint(vx, vy, yaw_rate, z)

    def sleep(self, duration: float) -> None:
        time.sleep(duration)

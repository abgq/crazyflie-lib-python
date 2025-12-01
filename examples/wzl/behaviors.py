"""Behavior strategy objects used by :mod:`examples.wzl.controller`."""

from __future__ import annotations

import logging
import math
import time
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Optional

from cflib.crazyflie import Crazyflie

from constants import (DW1K_ANTENNA_DELAY_RC, DW1K_RC_TO_SECONDS,
                       DW1K_TOF_SCALING, SPEED_OF_LIGHT)
from logger import SensorSample

LOGGER = logging.getLogger(__name__)


class BehaviorState(Enum):
    """Enumeration for the behavior state machine."""
    IDLE = auto()
    TAKEOFF = auto()
    ACTIVE = auto()
    LANDING = auto()
    FINISHED = auto()


def ranging_counter_to_distance(counter: int) -> float:
    """Convert DW1K ranging counter units to distance in meters."""
    effective = (counter - 4 * DW1K_ANTENNA_DELAY_RC) * DW1K_TOF_SCALING
    if effective < 0:
        effective = 0.0
    tof_seconds = effective * DW1K_RC_TO_SECONDS
    distance_m = tof_seconds * SPEED_OF_LIGHT
    return float(distance_m)


class Behavior(ABC):
    """
    Base class for Crazyflie control behaviors, implemented as a state machine.

    This class uses the Template Method design pattern. The `__step` method orchestrates
    the lifecycle (takeoff, active mission, landing) and should not be overridden.
    Subclasses must implement the `step_active` method to define their specific
    mission logic.
    """

    # --- State Machine Constants ---
    TAKEOFF_HEIGHT: float = 0.5  # Target height for takeoff [m]
    TAKEOFF_DURATION_S: float = 2.0  # Duration of the takeoff ramp [s]
    LANDING_HEIGHT: float = 0.05  # Target height for landing [m]
    LANDING_DURATION_S: float = 3.0  # Duration of the landing ramp [s]

    def __init__(self, cf: "Crazyflie") -> None:
        self._cf = cf
        self._log = logging.getLogger(self.__class__.__name__)

        # State machine attributes
        self._state: BehaviorState = BehaviorState.IDLE
        self._state_start_time: float = 0.0
        self._start_pos_z: float = 0.0  # Used to smoothly start landing ramp

    def on_start(self) -> None:
        """
        Hook executed before the control loop starts. Arms the drone and
        initiates the TAKEOFF sequence. This method is non-blocking.
        """
        self._log.info("Arming drone and starting behavior...")
        try:
            # The drone must be armed before we can send setpoints.
            if not self._cf.is_armed:
                self._cf.platform.send_arming_request(True)

            self._state = BehaviorState.TAKEOFF
            self._state_start_time = time.monotonic()
        except Exception:
            self._log.exception("Failed to arm or start takeoff sequence.")
            self._state = BehaviorState.FINISHED

    def on_stop(self) -> None:
        """
        Hook executed once when the controller stops. This is a final safety net
        to ensure the drone is disarmed.
        """
        self._log.info("Behavior stopping. Sending stop setpoint and disarming.")
        try:
            self._cf.commander.send_stop_setpoint()
            self._cf.platform.send_arming_request(False)
        except Exception:
            self._log.exception("Failed to send stop/disarm commands cleanly.")

    def step(self, sample: SensorSample) -> None:
        """Public entry point for the state machine."""
        self.__step(sample)

    def __step(self, sample: SensorSample) -> None:
        """
        Main state machine dispatcher. DO NOT OVERRIDE.

        This method is the core of the behavior lifecycle. It handles state
        transitions and calls the appropriate logic for each state.
        """
        now = time.monotonic()
        elapsed_in_state = now - self._state_start_time

        if self._state == BehaviorState.IDLE:
            # Do nothing, remain in this state until changed externally or by subclass
            pass

        elif self._state == BehaviorState.TAKEOFF:
            if elapsed_in_state < self.TAKEOFF_DURATION_S:
                ratio = elapsed_in_state / self.TAKEOFF_DURATION_S
                z = self.LANDING_HEIGHT + (self.TAKEOFF_HEIGHT - self.LANDING_HEIGHT) * ratio
                self._cf.commander.send_hover_setpoint(0, 0, 0, z)
            else:
                self._log.info("Takeoff complete. Transitioning to ACTIVE state.")
                self._state = BehaviorState.ACTIVE
                self._state_start_time = now

        elif self._state == BehaviorState.ACTIVE:
            self.step_active(sample)

        elif self._state == BehaviorState.LANDING:
            # On the first tick of the LANDING state, capture the current height.
            if self._start_pos_z == 0.0:
                current_z = sample.values.get("kalman.stateZ")
                if isinstance(current_z, float):
                    self._start_pos_z = current_z
                else:
                    # Fallback to the target height if Kalman state is not available.
                    self._start_pos_z = self.TAKEOFF_HEIGHT

            if elapsed_in_state < self.LANDING_DURATION_S:
                ratio = elapsed_in_state / self.LANDING_DURATION_S
                z = self._start_pos_z - (self._start_pos_z - self.LANDING_HEIGHT) * ratio
                z = max(z, self.LANDING_HEIGHT)  # Clamp to landing height
                self._cf.commander.send_hover_setpoint(0, 0, 0, z)
            else:
                self._log.info("Landing complete. Transitioning to FINISHED state.")
                self._state = BehaviorState.FINISHED
                self._state_start_time = now
                # Immediately send stop to cut motors
                self._cf.commander.send_stop_setpoint()

        elif self._state == BehaviorState.FINISHED:
            # The controller loop will see this state and terminate.
            # on_stop will be called as a final cleanup.
            pass

    @abstractmethod
    def step_active(self, sample: SensorSample) -> None:
        """
        Abstract method for mission-specific logic.

        This method is called on every tick while the behavior is in the
        `ACTIVE` state. Implement your flight control logic here.
        """
        raise NotImplementedError

    def trigger_landing(self) -> None:
        """
        Initiates the landing sequence. This can be called from `step_active`
        when the mission is complete or from an external safety check.
        """
        if self._state not in [BehaviorState.LANDING, BehaviorState.FINISHED]:
            self._log.info("Landing triggered. Transitioning to LANDING state.")
            self._state = BehaviorState.LANDING
            self._state_start_time = time.monotonic()
            # Reset the starting Z position to ensure it's re-captured on the next tick.
            self._start_pos_z = 0.0

    def is_finished(self) -> bool:
        """Returns True if the behavior has completed its lifecycle."""
        return self._state == BehaviorState.FINISHED


class IdleBehavior(Behavior):
    """Safe default behavior that performs no motion commands."""

    def __init__(self, cf: "Crazyflie") -> None:
        super().__init__(cf)
        self._last_log = 0.0

    def on_start(self) -> None:
        """Overrides base behavior to prevent takeoff and go straight to ACTIVE."""
        self._log.info("IdleBehavior started: transitioning to ACTIVE state.")
        self._state = BehaviorState.ACTIVE

    def on_stop(self) -> None:
        self._log.info("IdleBehavior stopped.")

    def step_active(self, sample: SensorSample) -> None:
        """Log battery and UWB counter at a low rate."""
        now = time.monotonic()
        if now - self._last_log >= 1.0:
            self._last_log = now
            raw = sample.values.get("dw1k.rangingCounter")
            vbattery = sample.values.get("pm.vbat")
            self._log.info("IdleBehavior: counter=%r, vbat=%r", raw, vbattery)


class RunAndTumbleBehavior(Behavior):
    """Reactive control strategy using Run & Tumble logic."""

    # --- Control Parameters for Run & Tumble ---
    SEARCH_VELOCITY_MPS: float = 0.4    # Forward speed when running
    TUMBLE_RATE_DEG_S: float = 75      # Yaw rate when tumbling (searching)
    GRADIENT_THRESHOLD_COUNTER: float = 5  # Sensitivity to distance change (counters)
    TARGET_COUNTER: float = 65500        # Distance to stop from anchor (counters)

    def __init__(self, cf: "Crazyflie") -> None:
        super().__init__(cf)
        self._prev_counter: Optional[float] = None
        self._last_vx: float = 0.0
        self._last_yaw_rate: float = 0.0

    def step_active(self, sample: SensorSample) -> None:
        counter = sample.values.get("dw1k.rangingCounter")
        if counter is None:
            self._log.warning("Missing rangingCounter in sample; holding position.")
            self._cf.commander.send_hover_setpoint(0, 0, 0, self.TAKEOFF_HEIGHT)
            return

        # Arrival Check
        if counter <= self.TARGET_COUNTER:
            self._log.info("Target counter %.1f reached; initiating landing.", counter)
            self.trigger_landing()
            return

        # Initialize previous counter
        if self._prev_counter is None:
            self._prev_counter = counter
            self._cf.commander.send_hover_setpoint(0, 0, 0, self.TAKEOFF_HEIGHT)
            return

        delta_r = counter - self._prev_counter
        self._prev_counter = counter

        # Original reactive logic with a dead zone.
        vx = self._last_vx
        yaw_rate = self._last_yaw_rate

        if delta_r < -self.GRADIENT_THRESHOLD_COUNTER:
            # Getting closer (Run)
            vx = self.SEARCH_VELOCITY_MPS
            yaw_rate = 0.0
        elif delta_r > self.GRADIENT_THRESHOLD_COUNTER:
            # Getting further (Tumble)
            vx = self.SEARCH_VELOCITY_MPS * 0.5
            yaw_rate = self.TUMBLE_RATE_DEG_S
        else:
            # In the dead zone, maintain the previous action.
            pass

        self._last_vx = vx
        self._last_yaw_rate = yaw_rate

        # Actuation
        self._cf.commander.send_hover_setpoint(vx, 0.0, yaw_rate, self.TAKEOFF_HEIGHT)


class SinusoidalBehavior(Behavior):
    """Gradient-seeking navigation using sinusoidal yaw modulation."""

    # Tuning Parameters
    VELOCITY_MPS = 0.3          # Forward flight speed
    DITHER_OMEGA = 1.5          # Frequency of sine wave (rad/s)
    DITHER_AMP = 0.5            # Amplitude of sine wave (rad/s)
    GAIN = 25.0                 # Learning rate for the bias (Gradient Gain)
    BIAS_LIMIT = 1.0            # Max yaw bias (rad/s) to prevent spinning
    TARGET_DIST_M = 0.5         # Stop distance

    def __init__(self, cf: "Crazyflie") -> None:
        super().__init__(cf)
        self._bias = 0.0
        self._prev_dist: Optional[float] = None

    def step_active(self, sample: SensorSample) -> None:
        raw_counter = sample.values.get("dw1k.rangingCounter")
        if raw_counter is None:
            self._log.warning("No ranging counter in sample; holding position.")
            self._cf.commander.send_hover_setpoint(0, 0, 0, self.TAKEOFF_HEIGHT)
            return

        dist = ranging_counter_to_distance(int(raw_counter))

        # Arrival Check
        if dist < self.TARGET_DIST_M:
            self._log.info("Target reached (%.2fm). Landing.", dist)
            self.trigger_landing()
            return

        # Initialize history
        if self._prev_dist is None:
            self._prev_dist = dist
            self._cf.commander.send_hover_setpoint(0, 0, 0, self.TAKEOFF_HEIGHT)
            return

        # Extremum Seeking Logic
        delta = dist - self._prev_dist
        now = time.monotonic()
        dither = self.DITHER_AMP * math.sin(self.DITHER_OMEGA * now)
        correction = -self.GAIN * delta * dither

        self._bias = max(-self.BIAS_LIMIT, min(self.BIAS_LIMIT, self._bias + correction))
        yaw_cmd = self._bias + dither

        # Actuate
        self._cf.commander.send_hover_setpoint(self.VELOCITY_MPS, 0.0, yaw_cmd, self.TAKEOFF_HEIGHT)
        self._prev_dist = dist


def get_behavior(mode: str, cf: "Crazyflie") -> Behavior:
    """Return a behavior instance for the requested mode."""
    normalized = (mode or "idle").strip().lower()
    mapping = {
        "idle": IdleBehavior,
        "run_tumble": RunAndTumbleBehavior,
        "sinusoidal": SinusoidalBehavior,
    }

    behavior_cls = mapping.get(normalized)
    if not behavior_cls:
        LOGGER.warning("Unknown controller mode '%s'; falling back to idle", mode)
        return IdleBehavior(cf)

    return behavior_cls(cf)


__all__ = [
    "Behavior",
    "IdleBehavior",
    "RunAndTumbleBehavior",
    "SinusoidalBehavior",
    "get_behavior",
]

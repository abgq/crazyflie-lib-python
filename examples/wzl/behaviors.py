"""Behavior strategy objects used by :mod:`examples.wzl.controller`."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
from cflib.positioning.motion_commander import MotionCommander


try:  # Local execution (python examples/wzl/main.py)
    from logger import SensorSample
except ImportError:  # Package execution (python -m examples.wzl.main)
    from .logger import SensorSample  # type: ignore[F401]

try:  # Local execution
    from constants import (
        SPEED_OF_LIGHT,
        DW1K_ANTENNA_DELAY_RC,
        DW1K_RC_TO_SECONDS,
        DW1K_TOF_SCALING,
    )
except ImportError:  # Package-style
    from .constants import (  # type: ignore[F401]
        SPEED_OF_LIGHT,
        DW1K_ANTENNA_DELAY_RC,
        DW1K_RC_TO_SECONDS,
        DW1K_TOF_SCALING,
    )

if TYPE_CHECKING:
    from cflib.crazyflie import Crazyflie


LOGGER = logging.getLogger(__name__)

def ranging_counter_to_distance(counter: int) -> float:
    """Convert DW1K ranging counter units to distance in meters.

    The conversion is:

        effective_count = (counter - 4 * DW1K_ANTENNA_DELAY_RC) * DW1K_TOF_SCALING
        tof_seconds     = effective_count * DW1K_RC_TO_SECONDS
        distance_m      = tof_seconds * SPEED_OF_LIGHT

    All calibration-specific constants live in constants.py.
    """
    # Subtract antenna delay in counter units
    effective = (counter - 4 * DW1K_ANTENNA_DELAY_RC) * DW1K_TOF_SCALING

    # If the result is negative, clamp to zero – this should not normally happen,
    # but avoids weird negative distances when things are misconfigured.
    if effective < 0:
        effective = 0.0

    tof_seconds = effective * DW1K_RC_TO_SECONDS

    distance_m = tof_seconds * SPEED_OF_LIGHT
    
    return float(distance_m)


class Behavior(ABC):
    """Base class for Crazyflie control behaviors."""

    def __init__(self, cf: "Crazyflie") -> None:
        self._cf = cf
        self._log = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def on_start(self) -> None:
        """Hook executed before the control loop thread starts."""

    @abstractmethod
    def step(self, sample: SensorSample) -> None:
        """Execute one control step based on the latest sample."""

    @abstractmethod
    def on_stop(self) -> None:
        """Hook executed once when the controller stops."""


class IdleBehavior(Behavior):
    """Safe default behavior that performs no motion commands."""

    def __init__(self, cf: "Crazyflie") -> None:
        super().__init__(cf)
        self._last_log = 0.0

    def on_start(self) -> None:
        self._log.info("IdleBehavior started: no motion commands will be sent")

    def on_stop(self) -> None:
        self._log.info("IdleBehavior stopped")

    def step(self, sample: SensorSample) -> None:
        # Log distance at 1 Hz
        raw = sample.values.get("dw1k.rangingCounter")
        vbattery = sample.values.get("pm.vbat")
        # distance = ranging_counter_to_distance(raw)
        now = time.monotonic()
        if now - self._last_log >= 1.0:
            self._log.info("UWB Counter: %.2f - Battery: %.2f V", raw, vbattery)
            self._last_log = now


class DemoMotionBehavior(Behavior):
    """Placeholder for future MotionCommander demos."""

    def step(self, sample: SensorSample) -> None:
        self._log.debug("DemoMotionBehavior tick with sample %s", sample.values)


class DemoHighLevelBehavior(Behavior):
    """Placeholder for future HighLevelCommander demos."""

    def step(self, sample: SensorSample) -> None:
        self._log.debug("DemoHighLevelBehavior tick with sample %s", sample.values)


class WzlBehavior(Behavior):
    """Move towards a UWB anchor using dw1k.rangingCounter-derived distance.

    Assumptions:
    - The anchor lies roughly along the drone's forward body axis at start.
    - We only have a scalar distance (no bearing), so we just move forward and
      monitor whether distance decreases.
    - Height is handled by MotionCommander; we stay at its default altitude.
    """

    # You can tweak these or later read them from constants.py if you prefer.
    _DISTANCE_THRESHOLD_COUNTER = 66100   # Stop when closer than this [m]
    _FORWARD_SPEED_MPS = 0.30      # Slow, safe forward speed [m/s]
    _MIN_IMPROVEMENT_COUNTER = 500     # If we get worse by more than this, stop

    def __init__(self, cf: "Crazyflie") -> None:
        super().__init__(cf)
        self._last_log = 0.0
        self._mc: MotionCommander | None = None
        self._last_counter: float | None = None
        self._in_air: bool = False
        self._moving: bool = False
        self._done: bool = False

    def on_start(self) -> None:
        """Create MotionCommander and hover at default height."""
        try:
            self._mc = MotionCommander(self._cf)

            self._cf.platform.send_arming_request(True)

            self._mc.take_off(height=0.5, velocity=0.3)
            self._in_air = True

            time.sleep(2.0)  # Allow some time to stabilize hover
            
            self._log.info(
                "WzlBehavior started: will move forward until rangingCounter < %d",
                self._DISTANCE_THRESHOLD_COUNTER,
            )
        except Exception:  # noqa: BLE001
            self._log.exception("Failed to create MotionCommander; behavior will be inert")
            self._mc = None

    def step(self, sample: SensorSample) -> None:
        # If we failed to create MotionCommander, do nothing.
        if self._mc is None or self._done:
            return

        raw = sample.values.get("dw1k.rangingCounter")
        vbattery = sample.values.get("pm.vbat")

        # distance = ranging_counter_to_distance(raw)
        if raw is None:
            self._log.warning("No rangingCounter in sample; ignoring")
            return
        
        counter = raw

        if counter <= 0:
            self._log.warning(
                "Invalid ranging counter %d, ignoring",
                counter
            )
            return

        # If we're within the target radius, stop and mark done.
        if counter <= self._DISTANCE_THRESHOLD_COUNTER:
            if self._moving:
                self._log.info(
                    "Reached counter threshold (%d <= %d); stopping and landing",
                    counter,
                    self._DISTANCE_THRESHOLD_COUNTER,
                )

                self._mc.stop()
                self._moving = False

                self._mc.land()
                self._in_air = False

                self._cf.platform.send_arming_request(False)
                
            self._done = True
            return

        # First valid distance: start moving forward slowly.
        if self._last_counter is None:
            self._log.info(
                "Initial counter %d; starting forward motion at %.2f m/s",
                counter,
                self._FORWARD_SPEED_MPS,
            )

            self._mc.start_linear_motion(self._FORWARD_SPEED_MPS, 0.0, 0.0)  
            self._moving = True

            self._last_counter = counter
            return

        # We already have a previous distance; check if we are improving.
        improvement = self._last_counter - counter

        if improvement >= self._MIN_IMPROVEMENT_COUNTER:
            # Getting closer in a meaningful way, keep going.
            self._last_counter = counter
            return

        if counter > self._last_counter + self._MIN_IMPROVEMENT_COUNTER:
            # Counter is clearly getting worse; likely facing wrong way.
            self._log.warning(
                "Counter increased (%d -> %d); stopping forward motion",
                self._last_counter,
                counter,
            )

            self._mc.stop()
            self._moving = False

            self._mc.land()
            self._in_air = False

            self._cf.platform.send_arming_request(False)

            self._done = True
            return
        
        # Log distance at 1 Hz
        now = time.monotonic()
        if now - self._last_log >= 1.0:
            self._log.info("UWB Counter: %d - Battery: %.2f V", counter, vbattery)
            self._last_log = now

        # Small change (noise-level): just keep going but update last_counter
        self._last_counter = counter

    def on_stop(self) -> None:
        """Ensure we stop any ongoing motion and land."""
        mc = self._mc
        self._mc = None

        try:
            if mc is None:
                return

            # Stop any forward motion
            if self._moving:
                try:
                    mc.stop()
                except Exception:
                    self._log.debug("MotionCommander.stop() failed during shutdown", exc_info=True)
                finally:
                    self._moving = False

            # Land if we think we are in the air
            if self._in_air:
                try:
                    mc.land(0.1)
                except Exception:
                    self._log.debug("MotionCommander.land() failed during shutdown", exc_info=True)
                finally:
                    self._in_air = False

            # Best‑effort: explicitly stop the internal _SetPointThread
            thread = getattr(mc, "_thread", None)
            if thread is not None and thread.is_alive():
                try:
                    # _SetPointThread.stop() enqueues TERMINATE_EVENT and joins
                    thread.stop()
                except Exception:
                    self._log.debug("Failed to stop MotionCommander thread cleanly", exc_info=True)
        finally:
            # Always try to disarm; never let this block shutdown
            try:
                self._cf.platform.send_arming_request(False)
            except Exception:
                self._log.debug("send_arming_request(False) failed during shutdown", exc_info=True)


class ProbeBehavior(Behavior):
    """Move towards a UWB anchor using only dw1k.rangingCounter-derived distance.

    Strategy:
    - Take off and start in a probe phase.
    - Probe several directions in the XY plane with short forward/back motions.
    - Pick the direction that yields the largest distance decrease.
    - Move steadily in that direction while distance keeps improving.
    - If distance stops improving or gets worse, go back to probing.
    - Land and disarm when within a distance threshold or when things clearly get worse.
    """

    # Tweak these as needed or move to constants.py later.
    _DISTANCE_THRESHOLD_COUNTER = 66200     # Stop when closer than this [m]

    _MOVE_SPEED_MPS = 0.25       # Speed during main move [m/s]
    _MOVE_DURATION_S = 4.0        # Duration of each main move segment [s]

    _MIN_IMPROVEMENT_COUNTER = 100        # Minimum improvement to consider "better" [m]

    _PROBE_SPEED_MPS = 0.20         # Speed during probing moves [m/s]
    _PROBE_DURATION_S = 3.0         # Duration of each probe leg (forward/back) [s]

    def __init__(self, cf: "Crazyflie") -> None:
        super().__init__(cf)
        self._last_log = 0.0
        self._mc: MotionCommander | None = None

        # State flags similar to WzlBehavior
        self._in_air = False
        self._moving = False
        self._done = False

        # Distance memory
        self._last_counter: float | None = None

        # High-level behavior state
        # probe | move
        self._state = "probe"

        # Probe-related state
        # Directions in body frame: +x, -x, +y, -y
        self._probe_dirs: list[tuple[float, float]] = [
            (1.0, 0.0),
            (-1.0, 0.0),
            (0.0, 1.0),
            (0.0, -1.0),
        ]
        self._probe_index = 0

        # Main move direction (body-frame vx, vy)
        self._move_dir: tuple[float, float] | None = None

        # Dynamic probe bookkeeping
        self._probe_phase = "idle"  # idle | forward | back
        self._probe_leg_start_time = 0.0
        self._probe_start_counter = None
        self._probe_leg_min_counter = None
        self._probe_best_dir = None
        self._probe_best_improvement = 0.0

    def on_start(self) -> None:
        """Create MotionCommander, arm, and take off to a safe height."""
        try:
            self._cf.platform.send_arming_request(True)

            self._mc = MotionCommander(self._cf)

            self._mc.take_off(height=0.5, velocity=0.3)
            self._in_air = True

            self._log.info(
                "ProbeBehavior started: probing for good direction, "
                "then moving until counter < %.2f",
                self._DISTANCE_THRESHOLD_COUNTER
            )
        except Exception:  # noqa: BLE001
            self._log.exception(
                "ProbeBehavior: failed to create MotionCommander; behavior will be inert"
            )
            self._mc = None

    def step(self, sample: SensorSample) -> None:
        # If we failed to create MotionCommander or we're done, do nothing.
        if self._mc is None or self._done:
            return

        counter = sample.values.get("dw1k.rangingCounter")
        vbattery = sample.values.get("pm.vbat")

        # Log distance at 0.2 Hz
        now = time.monotonic()
        if now - self._last_log >= 5.0:
            self._log.info(
                "UWB distance: %.2f m - Battery: %.2f V",
                counter,
                vbattery,
            )
            self._last_log = now

        if counter is None:
            self._log.warning("No rangingCounter in sample; ignoring")
            return
        
        # Initialize last_distance on first valid measurement
        if self._last_counter is None:
            self._last_counter = counter

        if counter <= 0.0:
            self._log.warning(
                "RangeOnlyBehavior: invalid ranging counter %s; ignoring",
                counter
            )
            return

        # Global stop condition: close enough -> stop & land
        if counter <= self._DISTANCE_THRESHOLD_COUNTER:
            if self._moving:
                self._log.info(
                    "RangeOnlyBehavior: reached distance threshold "
                    "(%.2f m <= %.2f m); stopping and landing",
                    counter,
                    self._DISTANCE_THRESHOLD_COUNTER,
                )
                self._mc.stop()
                self._moving = False

            if self._in_air:
                self._mc.land(0.2)
                self._in_air = False

            self._cf.platform.send_arming_request(False)
            self._done = True
            return

        # State machine
        mc = self._mc  # local alias (already checked for None above)
        if self._state == "probe":
            # Begin forward leg if not currently moving
            if self._probe_phase == "idle":
                direction = self._probe_dirs[self._probe_index]
                x, y = direction
                if x != 0.0 and y == 0.0:
                    if x > 0.0:
                        mc.forward(self._PROBE_SPEED_MPS)
                        self._moving = True
                    else:
                        mc.back(self._PROBE_SPEED_MPS)
                        self._moving = True
                elif x == 0.0 and y != 0.0:
                    if y > 0.0:
                        mc.right(self._PROBE_SPEED_MPS)
                        self._moving = True
                    else:
                        mc.left(self._PROBE_SPEED_MPS)
                        self._moving = True
                else:
                    self._log.error("Invalid probe direction: %s", direction)
                    return
                self._probe_leg_start_time = time.monotonic()
                self._probe_start_counter = counter
                self._probe_leg_min_counter = counter
                return

            # Update min counter seen during this leg
            if self._probe_leg_min_counter is None or counter < self._probe_leg_min_counter:
                self._probe_leg_min_counter = counter

            leg_elapsed = time.monotonic() - self._probe_leg_start_time
            if leg_elapsed >= self._PROBE_DURATION_S:
                direction = self._probe_dirs[self._probe_index]
                vx, vy = direction
                if self._probe_phase == "forward":
                    # Switch to back leg
                    mc.start_linear_motion(-vx * self._PROBE_SPEED_MPS, -vy * self._PROBE_SPEED_MPS, 0.0)
                    self._probe_phase = "back"
                    self._probe_leg_start_time = time.monotonic()
                    # Keep min counter from forward leg for improvement calc
                    return
                else:  # back leg finished
                    mc.stop()
                    self._moving = False
                    # Compute improvement for this direction
                    if self._probe_start_counter is not None and self._probe_leg_min_counter is not None:
                        improvement = self._probe_start_counter - self._probe_leg_min_counter
                        if improvement > self._probe_best_improvement:
                            self._probe_best_improvement = improvement
                            self._probe_best_dir = direction

                    # Advance to next direction
                    self._probe_index += 1
                    self._probe_phase = "idle"
                    self._probe_start_counter = None
                    self._probe_leg_min_counter = None

                    # If all directions probed, decide next action
                    if self._probe_index >= len(self._probe_dirs):
                        if self._probe_best_dir and self._probe_best_improvement >= self._MIN_IMPROVEMENT_COUNTER:
                            # Commit to move phase
                            mvx, mvy = self._probe_best_dir
                            mc.start_linear_motion(mvx * self._MOVE_SPEED_MPS, mvy * self._MOVE_SPEED_MPS, 0.0)
                            self._move_dir = self._probe_best_dir
                            self._state = "move"
                            self._moving = True
                            self._log.info(
                                "Probe complete: best dir=%s improvement=%.1f; entering move phase",
                                self._move_dir,
                                self._probe_best_improvement,
                            )
                        else:
                            # No strong improvement; retry probing cycle
                            self._log.info(
                                "Probe cycle yielded no improvement >= %.1f (best=%.1f); retrying",
                                self._MIN_IMPROVEMENT_COUNTER,
                                self._probe_best_improvement,
                            )
                        # Reset probe cycle bookkeeping regardless of outcome
                        self._probe_index = 0
                        self._probe_best_dir = None
                        self._probe_best_improvement = 0.0
                    return

        elif self._state == "move":
            # Evaluate progress; improvement is decrease in counter
            improvement = self._last_counter- counter
            if improvement >= self._MIN_IMPROVEMENT_COUNTER:
                # Significant progress; update baseline
                self._last_counter = counter
                return
            # Worsening beyond threshold -> re-enter probe phase
            if self._last_counter is not None and counter > self._last_counter + self._MIN_IMPROVEMENT_COUNTER:
                self._log.warning(
                    "Move phase worsening: counter rose from %.0f to %.0f (>%d); re-probing",
                    self._last_counter,
                    counter,
                    self._MIN_IMPROVEMENT_COUNTER,
                )
                try:
                    mc.stop()
                except Exception:  # noqa: BLE001
                    self._log.debug("Failed to stop during move->probe transition", exc_info=True)
                self._moving = False
                self._state = "probe"
                self._probe_phase = "idle"
                self._probe_index = 0
                self._probe_best_dir = None
                self._probe_best_improvement = 0.0
                # Keep last_counter so future improvement uses current value
                self._last_counter = counter
                return
            # Minor change (noise) -> keep moving; occasionally refresh baseline
            if improvement > 0:
                self._last_counter = counter

    def on_stop(self) -> None:
        """Ensure we stop any ongoing motion and land."""
        mc = self._mc
        self._mc = None

        try:
            if mc is None:
                return

            # Stop any motion
            if self._moving:
                try:
                    mc.stop()
                except Exception:
                    self._log.debug("MotionCommander.stop() failed during shutdown", exc_info=True)
                finally:
                    self._moving = False

            # Land if we think we are in the air
            if self._in_air:
                try:
                    mc.land(0.2)
                except Exception:
                    self._log.debug("MotionCommander.land() failed during shutdown", exc_info=True)
                finally:
                    self._in_air = False

            # Best‑effort: explicitly stop the internal _SetPointThread
            thread = getattr(mc, "_thread", None)
            if thread is not None and thread.is_alive():
                try:
                    # _SetPointThread.stop() enqueues TERMINATE_EVENT and joins
                    thread.stop()
                except Exception:
                    self._log.debug("Failed to stop MotionCommander thread cleanly", exc_info=True)
        finally:
            # Always try to disarm; never let this block shutdown
            try:
                self._cf.platform.send_arming_request(False)
            except Exception:
                self._log.debug("send_arming_request(False) failed during shutdown", exc_info=True)


def get_behavior(mode: str, cf: "Crazyflie") -> Behavior:
    """Return a behavior instance for the requested mode."""
    normalized = (mode or "idle").strip().lower()
    mapping = {
        "idle": IdleBehavior,
        "demo_motion": DemoMotionBehavior,
        "demo_highlevel": DemoHighLevelBehavior,
        "wzl": WzlBehavior,
        "probe": ProbeBehavior,
    }

    behavior_cls = mapping.get(normalized, IdleBehavior)
    if behavior_cls is IdleBehavior and normalized not in mapping:
        LOGGER.warning("Unknown controller mode '%s'; falling back to idle", mode)
    return behavior_cls(cf)


__all__ = [
    "Behavior",
    "IdleBehavior",
    "DemoMotionBehavior",
    "DemoHighLevelBehavior",
    "WzlBehavior",
    "ProbeBehavior",
    "get_behavior",
]

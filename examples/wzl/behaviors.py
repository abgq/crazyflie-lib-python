"""Behavior strategy objects used by :mod:`examples.wzl.controller`."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from enum import Enum, auto

from cflib.positioning.motion_commander import MotionCommander

from logger import SensorSample

from constants import (
    SPEED_OF_LIGHT,
    DW1K_ANTENNA_DELAY_RC,
    DW1K_RC_TO_SECONDS,
    DW1K_TOF_SCALING,
)

from cflib.crazyflie import Crazyflie


LOGGER = logging.getLogger(__name__)


class ProbeState(Enum):
    PROBE = auto()
    MOVE = auto()


class ProbePhase(Enum):
    IDLE = auto()
    FORWARD = auto()
    BACK = auto()


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
        """Log battery and UWB counter at a low rate without crashing on None."""
        raw = sample.values.get("dw1k.rangingCounter")
        vbattery = sample.values.get("pm.vbat")
        now = time.monotonic()
        if now - self._last_log >= 1.0:
            self._last_log = now
            if isinstance(raw, (int, float)) and isinstance(vbattery, (int, float)):
                self._log.info(
                    "IdleBehavior: UWB counter: %d - Battery: %.2f V",
                    int(raw),
                    float(vbattery),
                )
            else:
                self._log.info(
                    "IdleBehavior: missing/invalid data – counter=%r, vbat=%r",
                    raw,
                    vbattery,
                )


class DemoMotionBehavior(Behavior):
    """Placeholder for future MotionCommander demos."""

    def step(self, sample: SensorSample) -> None:
        self._log.debug("DemoMotionBehavior tick with sample %s", sample.values)


class DemoHighLevelBehavior(Behavior):
    """Placeholder for future HighLevelCommander demos."""

    def step(self, sample: SensorSample) -> None:
        self._log.debug("DemoHighLevelBehavior tick with sample %s", sample.values)


class WzlBehavior(Behavior):
    """Move towards a UWB anchor using raw dw1k.rangingCounter values (counter units).

    Assumptions:
    - The anchor lies roughly along the drone's forward body axis at start.
    - We only have a scalar ranging counter (no bearing), so we just move forward and
      monitor whether the counter decreases.
    - Height is handled by MotionCommander; we stay at its default altitude.
    """

    # You can tweak these or later read them from constants.py if you prefer.
    _DISTANCE_THRESHOLD_COUNTER = 66100  # Counter threshold (raw units) to stop
    _FORWARD_SPEED_MPS = 0.30  # Slow, safe forward speed [m/s]
    _MIN_IMPROVEMENT_COUNTER = 100  # If we get worse by more than this, stop
    _MAX_FORWARD_DURATION_S = 10.0  # Safety: max time to keep moving forward
    _NO_IMPROVEMENT_TIMEOUT_S = 5.0  # Safety: stop if no improvement for this long

    def __init__(self, cf: "Crazyflie") -> None:
        super().__init__(cf)
        self._last_log = 0.0
        self._mc: MotionCommander | None = None
        self._last_counter: float | None = None
        self._in_air: bool = False
        self._moving: bool = False
        self._done: bool = False
        self._move_start_time: float | None = None
        self._last_improvement_time: float | None = None

    def on_start(self) -> None:
        """Create MotionCommander and hover at default height."""
        try:
            self._cf.platform.send_arming_request(True)

            self._mc = MotionCommander(self._cf)

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
        now = time.monotonic()

        if raw is None:
            self._log.warning("WzlBehavior: no rangingCounter in sample; ignoring")
            return
        if not isinstance(raw, (int, float)):
            self._log.warning("WzlBehavior: invalid rangingCounter %r; ignoring", raw)
            return

        counter = float(raw)
        if counter <= 0:
            self._log.warning("WzlBehavior: non-positive ranging counter %r; ignoring", raw)
            return

        if now - self._last_log >= 1.0:
            self._last_log = now
            if isinstance(vbattery, (int, float)):
                self._log.info(
                    "WzlBehavior: UWB counter: %d - Battery: %.2f V",
                    int(counter),
                    float(vbattery),
                )
            else:
                self._log.info(
                    "WzlBehavior: missing/invalid data – counter=%r, vbat=%r",
                    counter,
                    vbattery,
                )

        # If we're within the target radius, stop and mark done.
        if counter <= self._DISTANCE_THRESHOLD_COUNTER:
            if self._moving:
                self._log.info(
                    "Reached counter threshold (%d <= %d); stopping and landing",
                    int(counter),
                    self._DISTANCE_THRESHOLD_COUNTER,
                )
                try:
                    self._mc.stop()
                except Exception:
                    self._log.debug("MotionCommander.stop() failed while stopping", exc_info=True)
                self._moving = False

            if self._in_air:
                try:
                    self._mc.land()
                except Exception:
                    self._log.debug("MotionCommander.land() failed while stopping", exc_info=True)
                self._in_air = False

            try:
                self._cf.platform.send_arming_request(False)
            except Exception:
                self._log.debug("send_arming_request(False) failed after threshold", exc_info=True)

            self._done = True
            self._move_start_time = None
            self._last_improvement_time = None
            return

        # First valid counter: start moving forward slowly.
        if self._last_counter is None:
            self._log.info(
                "Initial counter %d; starting forward motion at %.2f m/s",
                int(counter),
                self._FORWARD_SPEED_MPS,
            )

            self._mc.start_linear_motion(self._FORWARD_SPEED_MPS, 0.0, 0.0)
            self._moving = True
            self._move_start_time = now
            self._last_improvement_time = now
            self._last_counter = counter
            return

        # Safety: prevent endless forward motion
        if (
            self._moving
            and self._move_start_time is not None
            and now - self._move_start_time >= self._MAX_FORWARD_DURATION_S
        ):
            self._log.warning(
                "Forward motion timed out after %.1f s; stopping and landing",
                now - self._move_start_time,
            )
            try:
                self._mc.stop()
            except Exception:
                self._log.debug("MotionCommander.stop() failed on timeout", exc_info=True)
            self._moving = False
            if self._in_air:
                try:
                    self._mc.land()
                except Exception:
                    self._log.debug("MotionCommander.land() failed on timeout", exc_info=True)
                self._in_air = False
            try:
                self._cf.platform.send_arming_request(False)
            except Exception:
                self._log.debug("send_arming_request(False) failed on timeout", exc_info=True)
            self._done = True
            return

        if (
            self._moving
            and self._last_improvement_time is not None
            and now - self._last_improvement_time >= self._NO_IMPROVEMENT_TIMEOUT_S
        ):
            self._log.warning(
                "No significant counter improvement for %.1f s; stopping forward motion",
                self._NO_IMPROVEMENT_TIMEOUT_S,
            )
            try:
                self._mc.stop()
            except Exception:
                self._log.debug("MotionCommander.stop() failed on no-improvement", exc_info=True)
            self._moving = False
            if self._in_air:
                try:
                    self._mc.land()
                except Exception:
                    self._log.debug("MotionCommander.land() failed on no-improvement", exc_info=True)
                self._in_air = False
            try:
                self._cf.platform.send_arming_request(False)
            except Exception:
                self._log.debug("send_arming_request(False) failed on no-improvement", exc_info=True)
            self._done = True
            return

        # We already have a previous counter; check if we are improving.
        improvement = self._last_counter - counter

        if improvement >= self._MIN_IMPROVEMENT_COUNTER:
            # Getting closer in a meaningful way, keep going.
            self._last_counter = counter
            self._last_improvement_time = now
            return

        if counter > self._last_counter + self._MIN_IMPROVEMENT_COUNTER:
            # Counter is clearly getting worse; likely facing wrong way.
            self._log.warning(
                "Counter increased (%d -> %d); stopping forward motion",
                int(self._last_counter),
                int(counter),
            )

            try:
                self._mc.stop()
            except Exception:
                self._log.debug("MotionCommander.stop() failed while worsening", exc_info=True)
            self._moving = False

            if self._in_air:
                try:
                    self._mc.land()
                except Exception:
                    self._log.debug("MotionCommander.land() failed while worsening", exc_info=True)
                self._in_air = False

            try:
                self._cf.platform.send_arming_request(False)
            except Exception:
                self._log.debug("send_arming_request(False) failed while worsening", exc_info=True)

            self._done = True
            self._move_start_time = None
            self._last_improvement_time = None
            return
        
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
    """Move towards a UWB anchor using only dw1k.rangingCounter values (counter units).

    Strategy:
    - Take off and start in a probe phase.
    - Probe several directions in the XY plane with short forward/back motions.
    - Pick the direction that yields the largest counter decrease.
    - Move steadily in that direction while the counter keeps improving.
    - If the counter stops improving or gets worse, go back to probing.
    - Land and disarm when within a counter threshold or when things clearly get worse.
    """

    # Tweak these as needed or move to constants.py later.
    _DISTANCE_THRESHOLD_COUNTER = 66200  # Counter threshold (raw units) to stop

    _MOVE_SPEED_MPS = 0.40  # Speed during main move [m/s]
    _MOVE_DURATION_S = 5.0  # Max time to stay in move phase before re-probing [s]

    _MIN_IMPROVEMENT_COUNTER = 100  # Minimum counter improvement to consider "better"

    _PROBE_SPEED_MPS = 0.30  # Speed during probing moves [m/s]
    _PROBE_DURATION_S = 5.0  # Duration of each probe leg (forward/back) [s]

    _NO_IMPROVEMENT_TIMEOUT_S = 10.0  # Safety: re-probe if no progress for this long

    def halve_speeds(self) -> None:
        self._MOVE_SPEED_MPS *= 0.5
        self._PROBE_SPEED_MPS *= 0.5

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
        self._state: ProbeState = ProbeState.PROBE

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
        self._probe_phase: ProbePhase = ProbePhase.IDLE
        self._probe_leg_start_time = 0.0
        self._probe_start_counter = None
        self._probe_leg_min_counter = None
        self._probe_best_dir = None
        self._probe_best_improvement = 0.0
        self._probe_best_min_counter = None

        # Move-phase safety timers
        self._move_phase_start_time: float | None = None
        self._last_progress_time: float | None = None

    def _reset_to_probe(self, mc: MotionCommander, counter: float) -> None:
        """Stop motion and reset probe state for another cycle."""
        try:
            mc.stop()
        except Exception:
            self._log.debug("Failed to stop during move->probe transition", exc_info=True)
        self._moving = False
        self._state = ProbeState.PROBE
        self._probe_phase = ProbePhase.IDLE
        self._probe_index = 0
        self._probe_best_dir = None
        self._probe_best_improvement = 0.0
        self._probe_best_min_counter = None
        self._probe_start_counter = None
        self._probe_leg_min_counter = None
        self._move_phase_start_time = None
        self._last_progress_time = None
        self._last_counter = counter

    def on_start(self) -> None:
        """Create MotionCommander, arm, and take off to a safe height."""
        try:
            self._cf.platform.send_arming_request(True)

            self._mc = MotionCommander(self._cf)

            self._mc.take_off(height=0.4, velocity=0.3)
            self._in_air = True

            self._log.info(
                "ProbeBehavior started: probing for good direction, "
                "then moving until counter < %d (counter units)",
                self._DISTANCE_THRESHOLD_COUNTER,
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

        # Log counter at 1.0 Hz
        now = time.monotonic()
        if now - self._last_log >= 5.0:
            self._last_log = now
            if isinstance(counter, (int, float)) and isinstance(vbattery, (int, float)):
                self._log.info(
                    "UWB counter: %d - Battery: %.2f V",
                    int(counter),
                    float(vbattery),
                )
            else:
                self._log.info(
                    "Missing/invalid data - counter=%r, vbat=%r",
                    counter,
                    vbattery,
                )

        if counter is None:
            self._log.warning("No rangingCounter in sample; ignoring")
            return
        if not isinstance(counter, (int, float)):
            self._log.warning("Invalid ranging counter %r; ignoring", counter)
            return

        counter = int(counter)

        # Initialize last_counter on first valid measurement
        if self._last_counter is None:
            self._last_counter = counter

        if counter <= 0:
            self._log.warning("Non-positive ranging counter %r; ignoring", counter)
            return

        # Global stop condition: close enough -> stop & land
        if counter <= self._DISTANCE_THRESHOLD_COUNTER:
            if self._moving:
                self._log.info(
                    "Reached counter threshold (%d <= %d); stopping and landing",
                    int(counter),
                    self._DISTANCE_THRESHOLD_COUNTER,
                )
                try:
                    self._mc.stop()
                except Exception:
                    self._log.info("MotionCommander.stop() failed during threshold stop", exc_info=True)
                self._moving = False

            if self._in_air:
                try:
                    self._mc.land(0.2)
                except Exception:
                    self._log.info("MotionCommander.land() failed during threshold stop", exc_info=True)
                self._in_air = False

            try:
                self._cf.platform.send_arming_request(False)
            except Exception:
                self._log.info("send_arming_request(False) failed during threshold stop", exc_info=True)
            self._done = True
            return

        # State machine
        mc = self._mc  # local alias (already checked for None above)
        if self._state == ProbeState.PROBE:
            # Begin forward leg if not currently moving
            if self._probe_phase == ProbePhase.IDLE:
                vx, vy = self._probe_dirs[self._probe_index]
                # Use start_linear_motion so _PROBE_SPEED_MPS is a true velocity [m/s]
                mc.start_linear_motion(
                    vx * self._PROBE_SPEED_MPS,
                    vy * self._PROBE_SPEED_MPS,
                    0.0,
                )
                self._moving = True
                self._probe_phase = ProbePhase.FORWARD
                self._probe_leg_start_time = time.monotonic()
                self._probe_start_counter = counter
                self._probe_leg_min_counter = counter
                return

            # Update min counter seen during this leg
            if self._probe_leg_min_counter is None or counter < self._probe_leg_min_counter:
                self._probe_leg_min_counter = counter

            leg_elapsed = time.monotonic() - self._probe_leg_start_time
            if self._probe_phase == ProbePhase.FORWARD and leg_elapsed >= self._PROBE_DURATION_S:
                # Reverse direction for the back leg using the same speed
                mc.stop()
                self._moving = False
                vx, vy = self._probe_dirs[self._probe_index]
                mc.start_linear_motion(
                    -vx * self._PROBE_SPEED_MPS,
                    -vy * self._PROBE_SPEED_MPS,
                    0.0,
                )
                self._moving = True
                self._probe_phase = ProbePhase.BACK
                self._probe_leg_start_time = time.monotonic()
                return

            if self._probe_phase == ProbePhase.BACK and leg_elapsed >= self._PROBE_DURATION_S:
                mc.stop()
                self._moving = False
                # Compute improvement for this direction
                if self._probe_start_counter is not None and self._probe_leg_min_counter is not None:
                    improvement = self._probe_start_counter - self._probe_leg_min_counter
                    if improvement > self._probe_best_improvement:
                        self._probe_best_improvement = improvement
                        self._probe_best_dir = self._probe_dirs[self._probe_index]
                        self._probe_best_min_counter = self._probe_leg_min_counter

                # Advance to next direction
                self._probe_index += 1
                self._probe_phase = ProbePhase.IDLE
                self._probe_start_counter = None
                self._probe_leg_min_counter = None

                # If all directions probed, decide next action
                if self._probe_index >= len(self._probe_dirs):
                    if self._probe_best_dir and self._probe_best_improvement >= self._MIN_IMPROVEMENT_COUNTER:
                        # Commit to move phase
                        mvx, mvy = self._probe_best_dir
                        mc.start_linear_motion(
                            mvx * self._MOVE_SPEED_MPS,
                            mvy * self._MOVE_SPEED_MPS,
                            0.0,
                        )
                        self._moving = True
                        self._move_dir = self._probe_best_dir
                        self._state = ProbeState.MOVE
                        self._last_counter = counter
                        self._move_phase_start_time = time.monotonic()
                        self._last_progress_time = time.monotonic()
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
                    self._probe_best_min_counter = None
                return

        elif self._state == ProbeState.MOVE:
            # Evaluate progress; improvement is decrease in counter
            improvement = self._last_counter - counter if self._last_counter is not None else 0.0
            if improvement >= self._MIN_IMPROVEMENT_COUNTER:
                # Significant progress; update baseline
                self._last_counter = counter
                self._last_progress_time = time.monotonic()
                return
            if (
                self._move_phase_start_time is not None
                and time.monotonic() - self._move_phase_start_time >= self._MOVE_DURATION_S
            ):
                self._log.info(
                    "Move phase reached %.1f s; re-probing",
                    time.monotonic() - self._move_phase_start_time,
                )
                self.halve_speeds()
                self._reset_to_probe(mc, counter)
                return
            if (
                self._last_progress_time is not None
                and time.monotonic() - self._last_progress_time >= self._NO_IMPROVEMENT_TIMEOUT_S
            ):
                self._log.warning(
                    "No significant improvement for %.1f s; re-probing",
                    self._NO_IMPROVEMENT_TIMEOUT_S,
                )
                self._reset_to_probe(mc, counter)
                return
            # Worsening beyond threshold -> re-enter probe phase
            # if self._last_counter is not None and counter > self._last_counter + self._MIN_IMPROVEMENT_COUNTER:
            #     self._log.warning(
            #         "Move phase worsening: counter rose from %.0f to %.0f (>%d); re-probing",
            #         self._last_counter,
            #         counter,
            #         self._MIN_IMPROVEMENT_COUNTER,
            #     )
            #     self._reset_to_probe(mc, counter)
            #     return
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

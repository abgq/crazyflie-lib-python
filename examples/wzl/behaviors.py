"""Behavior strategy objects used by :mod:`examples.wzl.controller`."""

from __future__ import annotations

import logging
import math
import time
from abc import ABC, abstractmethod

from logger import SensorSample

from constants import (
    SPEED_OF_LIGHT,
    DW1K_ANTENNA_DELAY_RC,
    DW1K_RC_TO_SECONDS,
    DW1K_TOF_SCALING
)

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

def rad_to_deg(rad: float) -> float:
    """Convert radians to degrees."""
    return rad * (180.0 / math.pi)

class Behavior(ABC):
    """Base class for Crazyflie control behaviors."""

    # Default landing settings
    LANDING_HEIGHT = 0.05
    LANDING_STEPS = 20
    LANDING_SLEEP = 0.1
    STABILIZE_STEPS = 10
    FLIGHT_HEIGHT = 0.5  # Default flight height in meters

    # Quadrant Check Constants
    QC_SCAN_SPEED = 0.3     # m/s (Gentle speed)
    QC_SCAN_TIME = 3.0      # seconds (Duration of each leg)
    QC_TURN_SPEED = 45.0    # deg/s
    
    def __init__(self, cf: "Crazyflie") -> None:
        self._cf = cf
        self._log = logging.getLogger(self.__class__.__name__)
        self._last_altitude: float | None = None

        # Quadrant Detection State Variables
        self._qc_state = 0
        self._qc_timer = 0.0
        self._qc_min_counter = float('inf')
        self._qc_scores = [0.0, 0.0, 0.0, 0.0]  # [Fwd, Back, Left, Right]
        self._qc_target_yaw = 0.0

    @abstractmethod
    def on_start(self) -> None:
        """Hook executed before the control loop thread starts."""

    @abstractmethod
    def step(self, sample: SensorSample) -> None:
        """Execute one control step based on the latest sample."""

    @abstractmethod
    def on_stop(self) -> None:
        """Hook executed once when the controller stops."""

    def take_off(self, target_height: float, duration: float = 2.0) -> None:
        """Execute a blocking takeoff sequence."""
        self._cf.platform.send_arming_request(True)
        time.sleep(1.5)  # Allow time for arming
        self._log.info("Taking off to %.2f m over %.1f s", target_height, duration)
        steps = int(duration / self.LANDING_SLEEP)
        for i in range(steps):
            ratio = (i + 1) / steps
            current_height = target_height * ratio
            self._cf.commander.send_hover_setpoint(0.0, 0.0, 0.0, current_height)
            time.sleep(self.LANDING_SLEEP)

        # Stabilize at top
        for _ in range(self.STABILIZE_STEPS):
            self._cf.commander.send_hover_setpoint(0.0, 0.0, 0.0, target_height)
            time.sleep(self.LANDING_SLEEP)

    def land(self) -> None:
        """Execute a blocking landing sequence using the last known altitude."""
        start_h = self._last_altitude
        if start_h is None:
            self._log.warning("Landing without known altitude; defaulting to %f m", self.FLIGHT_HEIGHT)
            start_h = self.FLIGHT_HEIGHT

        self._log.info("Landing from %.2f m", start_h)

        # Stabilize
        for _ in range(self.STABILIZE_STEPS):
            self._cf.commander.send_hover_setpoint(0.0, 0.0, 0.0, start_h)
            time.sleep(self.LANDING_SLEEP)

        # Ramp down
        end_h = self.LANDING_HEIGHT
        steps = self.LANDING_STEPS

        # Ensure we don't try to ramp if we are already lower than landing height
        if start_h > end_h:
            for i in range(steps):
                ratio = (i + 1) / steps
                current_height = start_h - (start_h - end_h) * ratio
                self._cf.commander.send_hover_setpoint(0.0, 0.0, 0.0, current_height)
                time.sleep(self.LANDING_SLEEP)

        # Cut Power
        try:
            self._cf.commander.send_stop_setpoint()
        except Exception:
            self._log.warning("Failed to send stop setpoint", exc_info=True)

        # Disarm
        try:
            self._cf.platform.send_arming_request(False)
        except Exception:
            self._log.warning("Failed to disarm", exc_info=True)

    # --------------------------------------------------------------------------
    # Quadrant Detection Logic
    # --------------------------------------------------------------------------
    def init_quadrant_check(self) -> None:
        """Reset the state machine for the quadrant check."""
        self._qc_state = 0
        self._qc_min_counter = float('inf')
        self._qc_scores = [0.0, 0.0, 0.0, 0.0]
        self._qc_timer = time.monotonic()
        self._log.info("Quadrant Check: Initialized")

    def run_quadrant_check_step(self, sample: SensorSample) -> bool:
        """
        Execute one step of the "Plus Sign" maneuver.
        
        Returns:
            True if the process is complete (drone is aligned).
            False if the process is still running.
        """

        if self._qc_state == 10:
            return True
        
        # Extract data
        counter = sample.values.get("dw1k.rangingCounter")
        if counter is not None:
            # Track minimum distance seen during this specific leg
            if counter < self._qc_min_counter:
                self._qc_min_counter = counter
        else:
            self._log.warning("Quadrant Check: Missing rangingCounter in sample; ignoring")

        now = time.monotonic()
        dt = now - self._qc_timer
        
        # State Machine
        # 0: Start / Hover Stabilize
        # 1: FWD OUT (+X) | 2: FWD IN (-X)
        # 3: BCK OUT (-X) | 4: BCK IN (+X)
        # 5: LFT OUT (+Y) | 6: LFT IN (-Y)
        # 7: RGT OUT (-Y) | 8: RGT IN (+Y)
        # 9: CALC & TURN  | 10: DONE

        vx, vy, yaw_rate = 0.0, 0.0, 0.0

        # Helper to transition states
        def next_state(reset_min=True):
            self._qc_state += 1
            self._qc_timer = now
            if reset_min:
                self._qc_min_counter = float('inf')

        if self._qc_state == 0:
            # Hover for 1s to settle
            if dt > 1.0:
                self._log.info("Quadrant Check: Starting X-axis scan")
                next_state()

        # --- Forward Leg ---
        elif self._qc_state == 1: # Out
            vx = self.QC_SCAN_SPEED
            if dt > self.QC_SCAN_TIME:
                self._qc_scores[0] = self._qc_min_counter  # Save Score FWD
                next_state(reset_min=False) # Don't need min for return
        elif self._qc_state == 2: # Return
            vx = -self.QC_SCAN_SPEED
            if dt > self.QC_SCAN_TIME:
                next_state()

        # --- Backward Leg ---
        elif self._qc_state == 3: # Out
            vx = -self.QC_SCAN_SPEED
            if dt > self.QC_SCAN_TIME:
                self._qc_scores[1] = self._qc_min_counter  # Save Score BACK
                next_state(reset_min=False)
        elif self._qc_state == 4: # Return
            vx = self.QC_SCAN_SPEED
            if dt > self.QC_SCAN_TIME:
                self._log.info("Quadrant Check: X done. Scores (Fwd/Back): %s", 
                               [int(s) for s in self._qc_scores[:2]])
                next_state()

        # --- Left Leg (+Y) ---
        elif self._qc_state == 5: # Out
            vy = self.QC_SCAN_SPEED
            if dt > self.QC_SCAN_TIME:
                self._qc_scores[2] = self._qc_min_counter  # Save Score LEFT
                next_state(reset_min=False)
        elif self._qc_state == 6: # Return
            vy = -self.QC_SCAN_SPEED
            if dt > self.QC_SCAN_TIME:
                next_state()

        # --- Right Leg (-Y) ---
        elif self._qc_state == 7: # Out
            vy = -self.QC_SCAN_SPEED
            if dt > self.QC_SCAN_TIME:
                self._qc_scores[3] = self._qc_min_counter  # Save Score RIGHT
                next_state(reset_min=False)
        elif self._qc_state == 8: # Return
            vy = self.QC_SCAN_SPEED
            if dt > self.QC_SCAN_TIME:
                next_state()

        # --- Calculation & Alignment ---
        elif self._qc_state == 9:
            # Lower score is better (smaller distance counter)
            fwd, back, left, right = self._qc_scores
            
            # Determine vectors
            # X: +1 if Fwd < Back, else -1
            x_dir = 1.0 if fwd < back else -1.0
            
            # Y: +1 if Left < Right, else -1
            y_dir = 1.0 if left < right else -1.0

            # Calculate angle
            # atan2(y, x) gives angle from X-axis
            target_rad = math.atan2(y_dir, x_dir)
            target_deg = rad_to_deg(target_rad)

            self._log.info("Quadrant Check: Result X=%d, Y=%d -> Target Yaw %.1f deg", 
                           int(x_dir), int(y_dir), target_deg)
            
            # Simple Turn: P-controller or timed turn?
            # Timed turn is safer given we don't track current yaw well without logging it
            # Assuming we started at 0 yaw and haven't drifted much:
            turn_duration = abs(target_deg) / self.QC_TURN_SPEED
            yaw_rate = math.copysign(self.QC_TURN_SPEED, target_deg)
            
            if dt > turn_duration:
                self._log.info("Quadrant Check: Alignment complete")
                self._cf.commander.send_hover_setpoint(0.0, 0.0, 0.0, self.FLIGHT_HEIGHT)
                next_state()
            else:
                # We need to explicitly return here to apply the yaw_rate
                self._cf.commander.send_hover_setpoint(0.0, 0.0, yaw_rate, self.FLIGHT_HEIGHT)
                return False

        # Apply velocity command for the active scan states
        if self._qc_state < 9:
            self._cf.commander.send_hover_setpoint(vx, vy, 0.0, self.FLIGHT_HEIGHT)

        return False

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
        alt = sample.values.get("kalman.stateZ")

        # Update altitude even in Idle for consistency/debugging
        if isinstance(alt, (int, float)):
            self._last_altitude = float(alt)

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

class RunAndTumbleBehavior(Behavior):
    """Reactive control strategy using Run & Tumble logic."""
    
    # --- Control Parameters for Run & Tumble ---
    SEARCH_VELOCITY_MPS: float = 0.3        # Forward speed when running
    TUMBLE_RATE_DEG_S: float = 45           # Yaw rate when tumbling (searching)
    GRADIENT_THRESHOLD_COUNTER: float = 6   # Sensitivity to distance change (counters)
    TARGET_COUNTER: float = 66250           # Distance to stop from anchor (counters) 

    def __init__(self, cf: "Crazyflie") -> None:
        super().__init__(cf)
        self._active: bool = False
        self._prev_counter: float | None = None
        self._last_log: float = 0.0

        # State tracking for hysteresis
        self._last_vx: float = 0.0
        self._last_yaw_rate: float = 0.0

    def on_start(self) -> None:
        """Arm, takeoff, and prepare for reactive control."""
        try:
            self.take_off(self.FLIGHT_HEIGHT)
            self._active = True
            self._log.info("RunAndTumbleBehavior started")
            self.init_quadrant_check()

        except Exception:
            self._log.exception("RunAndTumbleBehavior failed to start")
            self._active = False

    def step(self, sample: SensorSample) -> None:
        if not self._active:
            return
        
        if not self.run_quadrant_check_step(sample):
            # Still running quadrant check
            return

        counter = sample.values.get("dw1k.rangingCounter")
        vbattery = sample.values.get("pm.vbat")
        alt = sample.values.get("kalman.stateZ")

        if counter is None or vbattery is None or alt is None:
            self._log.warning("Missing rangingCounter, vbat, or altitude in sample; ignoring")
            return

        if isinstance(alt, (int, float)):
            self._last_altitude = float(alt)

        if isinstance(counter, (int, float)):
            counter = int(counter)

        if isinstance(vbattery, (int, float)):
            vbattery = float(vbattery)

        # Log counter with 2 seconds interval
        now = time.monotonic()
        if now - self._last_log >= 2.0:
            self._last_log = now
            self._log.info(
                    "UWB counter: %d - Battery: %.2f V",
                    int(counter),
                    float(vbattery),
                )

        # Arrival Check
        if counter <= self.TARGET_COUNTER:
            self._log.info("Target counter %.1f reached; initiating landing sequence", counter)
            try:
                self.land()
            except Exception:
                self._log.exception("Error during manual landing sequence")
            finally:
                self._active = False
            return

        # Initialize previous counter if needed
        if self._prev_counter is None:
            self._prev_counter = counter
            # Default to slow search if no history
            self._cf.commander.send_hover_setpoint(self.SEARCH_VELOCITY_MPS * 0.5, 0.0, 0.0, self.FLIGHT_HEIGHT)
            self._last_vx = self.SEARCH_VELOCITY_MPS * 0.5
            self._last_yaw_rate = 0.0
            return

        delta_r = counter - self._prev_counter
        self._prev_counter = counter

        threshold = self.GRADIENT_THRESHOLD_COUNTER

        vx = self._last_vx
        yaw_rate = self._last_yaw_rate

        if delta_r < -threshold:
            # Getting closer (Run)
            self._log.info("Run: delta_r=%.2f < -%.2f", delta_r, threshold)
            vx = self.SEARCH_VELOCITY_MPS
            yaw_rate = 0.0
        elif delta_r > threshold:
            # Getting further (Tumble)
            self._log.info("Tumble: delta_r=%.2f > %.2f", delta_r, threshold)
            vx = self.SEARCH_VELOCITY_MPS * 0.5
            yaw_rate = self.TUMBLE_RATE_DEG_S
        else:
            # Noise/Deadband: Maintain previous
            pass

        self._last_vx = vx
        self._last_yaw_rate = yaw_rate

        # Actuation
        self._cf.commander.send_hover_setpoint(vx, 0.0, yaw_rate, self.FLIGHT_HEIGHT)

    def on_stop(self) -> None:
        """Stop, land, and disarm safely."""
        if not self._active:
            return
        try:
            self.land()
        except Exception:
            self._log.exception("Error during manual landing sequence")
        finally:
            self._active = False

class SinusoidalBehavior(Behavior):
    """Gradient-seeking navigation using sinusoidal yaw modulation."""

    # Tuning Parameters
    VELOCITY_MPS = 0.30             # Forward flight speed
    DITHER_OMEGA = 2.5              # Frequency of sine wave (rad/s)
    DITHER_AMP = 1.0                # Amplitude of sine wave
    GAIN = 20.0                     # Learning rate for the bias (Gradient Gain)
    BIAS_LIMIT = 1.0                # Max yaw bias (rad/s) to prevent spinning
    TARGET_DIST_COUNTER = 66200     # Stop distance

    # Landing/Safety Constants
    FLIGHT_HEIGHT = 0.5

    def __init__(self, cf: "Crazyflie") -> None:
        super().__init__(cf)
        self._bias = 0.0
        self._prev_counter: int | None = None
        self._active = False
        self._last_log: float = 0.0

    def on_start(self) -> None:
        """Arm, takeoff, and prepare for sinusoidal control."""
        try:
            self._cf.platform.send_arming_request(True)
            time.sleep(1.0)  # Allow time for arming
            self.take_off(self.FLIGHT_HEIGHT)
            time.sleep(1.0)  # Allow time for stabilization
            self._active = True
            self.init_quadrant_check()
            self._log.info("SinusoidalBehavior started")
        except Exception:
            self._log.exception("SinusoidalBehavior failed to start")
            self._active = False

    def step(self, sample: SensorSample) -> None:
        if not self._active:
            return
        
        if not self.run_quadrant_check_step(sample):
            # Still running quadrant check
            return
        
        # Data
        vbattery = sample.values.get("pm.vbat")
        counter = sample.values.get("dw1k.rangingCounter")
        alt = sample.values.get("kalman.stateZ")

        if counter is None or alt is None or vbattery is None:
            self._log.warning("Missing rangingCounter, altitude, or battery voltage in sample; ignoring")
            return

        self._last_altitude = float(alt)
        counter = int(counter)
        vbattery = float(vbattery)

        # Logging
        now = time.monotonic()

        if now - self._last_log >= 1.0:
            self._last_log = now
            self._log.info("Altitude: %.2f, Ranging Counter: %d, Battery Voltage: %.2f", self._last_altitude, counter, vbattery)

        # Convert to meters
        # try:
        #     dist = ranging_counter_to_distance(int(raw_counter))
        # except (ValueError, TypeError):
        #     return

        # Arrival Check
        if counter < self.TARGET_DIST_COUNTER:
            self._log.info("Target reached (%d < %d). Landing.", counter, self.TARGET_DIST_COUNTER)
            self.land()
            self._active = False
            return

        # Algorithm (Extremum Seeking)
        # 1. Check prev_dist
        if self._prev_counter is None:
            self._prev_counter = counter
            # Hover in place while initializing history
            self._cf.commander.send_hover_setpoint(0.0, 0.0, 0.0, self.FLIGHT_HEIGHT)
            return

        # 2. Calculate delta
        delta = counter - self._prev_counter

        # 3. Calculate sinusoidal perturbation
        now = time.monotonic()
        dither = self.DITHER_AMP * math.sin(self.DITHER_OMEGA * now)

        # 4. Gradient Update
        # Logic: If moving Closer (delta < 0) while Turning Left (dither > 0), correction is positive.
        correction = -self.GAIN * delta * dither

        self._log.info("Delta: %d, Dither: %.3f, Correction: %.3f, New Bias: %.3f", 
                       delta, dither, correction, self._bias + correction)

        # 5. Update Bias
        self._bias += correction

        # 6. Clamp Bias
        self._bias = max(-self.BIAS_LIMIT, min(self.BIAS_LIMIT, self._bias))

        # 7. Calculate Output
        yaw_cmd = rad_to_deg(self._bias + dither)

        # 8. Actuate
        self._cf.commander.send_hover_setpoint(self.VELOCITY_MPS, 0.0, yaw_cmd, self.FLIGHT_HEIGHT)

        # 9. Update prev_dist
        self._prev_counter = counter

    def on_stop(self) -> None:
        """Stop hook: Execute safe landing if not already done."""
        if self._active:
            self.land()
            self._active = False

def get_behavior(mode: str, cf: "Crazyflie") -> Behavior:
    """Return a behavior instance for the requested mode."""
    normalized = (mode or "idle").strip().lower()
    mapping = {
        "idle": IdleBehavior,
        "run_tumble": RunAndTumbleBehavior,
        "sinusoidal": SinusoidalBehavior,
        "shark": SinusoidalBehavior,
    }

    behavior_cls = mapping.get(normalized, IdleBehavior)
    if behavior_cls is IdleBehavior and normalized not in mapping:
        LOGGER.warning("Unknown controller mode '%s'; falling back to idle", mode)
    return behavior_cls(cf)

__all__ = [
    "Behavior",
    "IdleBehavior",
    "RunAndTumbleBehavior",
    "SinusoidalBehavior",
    "get_behavior",
]

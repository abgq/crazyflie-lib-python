### **Future Reference: Upgrading ESC Gradient Estimation**

**The Issue:**
The current `SinusoidalBehavior` calculates the gradient using naive subtraction (`delta = current - prev`). This amplifies high-frequency noise (jitter) from the UWB sensor, causing the drone's heading correction to be twitchy and unreliable.

**The Solution:**
Replace the naive derivative with a **High-Pass Washout Filter** (HPF) as recommended in Section 5.2 of the algorithm report. This smooths the signal and extracts the dither correlation more cleanly.

**Implementation Steps:**

**1. Update `__init__` in `wzl/behaviors.py**`
Add the filter state variable and calculate the smoothing coefficient `alpha` based on the loop timing.

```python
# In wzl/behaviors.py

from constants import CONTROL_PERIOD_MS  # Ensure this is imported

class SinusoidalBehavior(Behavior):
    def __init__(self, drone: DroneInterface) -> None:
        super().__init__(drone)
        # ... existing init code ...
        
        # --- NEW: HPF State & Tuning ---
        self._hpf_val: float = 0.0
        
        # Calculate Alpha (Filter Coefficient)
        # Formula: alpha = RC / (RC + dt) where RC = 1 / (omega_0 / 2)
        dt = CONTROL_PERIOD_MS / 1000.0
        omega_h = self.DITHER_OMEGA / 2.0
        rc = 1.0 / omega_h
        self._alpha = rc / (rc + dt)
        
        self._log.info("ESC: Alpha calculated as %.3f", self._alpha)

```

**2. Update `step` in `wzl/behaviors.py**`
Replace the simple `delta` calculation with the filter logic.

```python
    def step(self, sample: SensorSample) -> None:
        # ... (start of method, checks for None) ...

        # Cast to float
        current_val = float(counter)

        # 1. Initialize History (Handle first run)
        if self._prev_counter is None:
            self._prev_counter = current_val
            self.drone.cmd_hover(0.0, 0.0, 0.0, self.FLIGHT_HEIGHT)
            return

        # --- THE FIX ---
        
        # Calculate the raw difference
        diff = current_val - self._prev_counter
        
        # Apply High-Pass Filter (Washout)
        # xi[k] = alpha * xi[k-1] + alpha * (current - prev)
        self._hpf_val = self._alpha * (self._hpf_val + diff)
        
        # Use filtered signal for correlation
        gradient_signal = self._hpf_val 

        # ---------------------------

        now = time.monotonic()
        dither = self.DITHER_AMP * math.sin(self.DITHER_OMEGA * now)

        # Update Bias Logic
        if self._update_bias_counter == self.UPDATE_BIAS_EVERY_N:
            # CHANGE: Use 'gradient_signal' instead of 'delta'
            correction = -self.GAIN * gradient_signal * dither 
            
            self._bias += correction
            self._bias = max(-self.BIAS_LIMIT, min(self.BIAS_LIMIT, self._bias))
            self._update_bias_counter = 0
        else:
            self._update_bias_counter += 1

        # ... (rest of method) ...

```

**Tuning Note:**
Switching to the HPF changes the magnitude of the signal feeding into the bias update. If the drone starts oscillating (swinging its nose wildly), **reduce** `self.GAIN`. Start by lowering it from `0.2` to `0.1` or `0.05`.

### **Future Reference: Upgrading Quadrant Check to Closed-Loop Turning**

**The Issue:**
The current "Quadrant Check" uses **Open-Loop Timing** to align the drone (e.g., "spin for 2 seconds"). This is inaccurate because rotation speed varies with battery voltage and air resistance. The drone often finishes the turn pointing the wrong way, forcing the main algorithm to compensate for a bad initial heading.

**The Solution:**
Switch to **Closed-Loop Control**. Instead of guessing the duration, read the drone's actual heading (`kalman.stateYaw`) and command the motors to turn until the error is zero.

**Implementation Steps:**

**1. Update `wzl/constants.py**`
Add the yaw state to the logging configuration so the behavior can "see" where it is pointing.

```python
# In wzl/constants.py -> LOG_CONFIGS

    LogBlockConfig(
        name="Position",
        period_ms=1000,
        variables=[
            LogVariableConfig(name="kalman.stateZ"),
            LogVariableConfig(name="kalman.stateYaw"),  # <--- NEW: Add this line
        ],
    ),

```

**2. Add Helper to `wzl/behaviors.py**`
Add this function at the top of the file (or inside the class) to handle angle wrapping (e.g., ensuring the transition from 359° to 1° is treated as 2°, not -358°).

```python
def normalize_angle(angle_deg: float) -> float:
    """Normalize angle to [-180, 180] range."""
    while angle_deg > 180:
        angle_deg -= 360
    while angle_deg < -180:
        angle_deg += 360
    return angle_deg

```

**3. Update logic in `wzl/behaviors.py**`
Replace the "State 17" block inside `run_quadrant_check_step` with this P-controller logic.

```python
        # ... inside run_quadrant_check_step ...

        # --- Calculation & Alignment (State 17) ---
        elif self._qc_state == 17:
            # 1. Get current heading
            current_yaw = sample.values.get("kalman.stateYaw")
            
            # Wait for valid data
            if current_yaw is None:
                return False

            # 2. Calculate Target (Execute Once)
            if self._qc_target_yaw == 0.0 and self._qc_scores[0] != 0.0:
                 fwd, back, left, right = self._qc_scores
                 x_dir = 1.0 if fwd < back else -1.0
                 y_dir = 1.0 if left < right else -1.0
                 
                 # Calculate relative turn needed
                 target_rad = math.atan2(y_dir, x_dir)
                 target_rel_deg = rad_to_deg(target_rad)
                 
                 # Set absolute target
                 self._qc_target_yaw = normalize_angle(current_yaw + target_rel_deg)
                 self._log.info("Target Yaw calculated: %.1f deg", self._qc_target_yaw)

            # 3. Closed-Loop P-Controller
            error = normalize_angle(self._qc_target_yaw - current_yaw)
            
            # Completion Check (Deadband of 5 degrees)
            if abs(error) < 5.0:
                self._log.info("Aligned (Error=%.1f). Finishing.", error)
                self.drone.cmd_hover(0.0, 0.0, 0.0, self.FLIGHT_HEIGHT)
                self._qc_state = 20 # Done
                self.sleep_timer = now
                return True

            # Calculate Command (Proportional Gain = 1.5)
            yaw_cmd = 1.5 * error
            
            # Safety Clamp (Max 60 deg/s)
            yaw_cmd = max(-60.0, min(60.0, yaw_cmd))
            
            self.drone.cmd_hover(0.0, 0.0, yaw_cmd, self.FLIGHT_HEIGHT)
            return False

```

**Tuning Note:**
The Gain `1.5` determines how aggressive the turn is.

* If it oscillates (wags left/right at the target), lower it to `1.0`.
* If it takes too long to finish the last few degrees, raise it to `2.0`.

### **Future Reference: Centralizing Calibration & Tuning**

**The Issue:**

1. **Uncalibrated Hardware:** The `DW1K_ANTENNA_DELAY_RC` constant determines the "zero point" of your distance measurement. If this is wrong, the drone will have a constant offset (e.g., thinking it is 1.0m away when it is actually 0.5m away), leading to failed arrival checks or collisions.
2. **Hardcoded Logic:** Tuning parameters like `GAIN`, `DITHER_AMP`, and `VELOCITY` are buried inside the logic classes in `behaviors.py`. Changing them during field tests is error-prone and slow.

**The Solution:**
Centralize all physics and tuning constants in `wzl/constants.py` and perform a static physical calibration to set the antenna delay.

**Implementation Steps:**

**1. Calibration Procedure (Hardware)**
Before flying, verify the antenna delay.

* **Setup:** Place the Crazyflie exactly **1.00 meter** (tape measured) from the anchor.
* **Test:** Run the `IdleBehavior` and observe the logged `dw1k.rangingCounter` or calculated distance.
* **Tune:** Adjust `DW1K_ANTENNA_DELAY_RC` in `wzl/constants.py` until the reported distance matches 1.00m.
* *If reading > 1.00m:* Increase delay.
* *If reading < 1.00m:* Decrease delay.



**2. Update `wzl/constants.py**`
Move the ESC tuning parameters from the behavior file to the constants file.

```python
# In wzl/constants.py

# ... existing constants ...

# --- ESC Tuning Constants ---
ESC_VELOCITY: float = 0.25       # Forward flight speed (m/s)
ESC_DITHER_OMEGA: float = 3.5    # Dither frequency (rad/s) -> approx 0.5 Hz
ESC_DITHER_AMP: float = 0.75     # Amplitude of yaw sine wave
ESC_GAIN: float = 0.1            # Learning rate (Gradient Gain)
ESC_BIAS_LIMIT: float = 0.4      # Max yaw bias (rad/s)
ESC_TARGET_COUNTER: int = 66200  # Stop distance (approx 0.5m - depends on calibration!)

```

**3. Update `wzl/behaviors.py**`
Modify `SinusoidalBehavior` to import and use these constants instead of local variables.

```python
# In wzl/behaviors.py

# 1. Add Imports
from constants import (
    ESC_VELOCITY, ESC_DITHER_OMEGA, ESC_DITHER_AMP, 
    ESC_GAIN, ESC_BIAS_LIMIT, ESC_TARGET_COUNTER
)

class SinusoidalBehavior(Behavior):
    # DELETE the hardcoded class attributes (VELOCITY_MPS, DITHER_OMEGA, etc.)
    
    def __init__(self, drone: DroneInterface) -> None:
        super().__init__(drone)
        # ...
        
    def step(self, sample: SensorSample) -> None:
        if not self._active:
            return
            
        # ... (data extraction) ...

        # 2. Use Constants in Logic
        now = time.monotonic()
        dither = ESC_DITHER_AMP * math.sin(ESC_DITHER_OMEGA * now)

        # ... (gradient calculation) ...

        if self._update_bias_counter == self.UPDATE_BIAS_EVERY_N:
            # Use centralized GAIN
            correction = -ESC_GAIN * gradient_signal * dither
            
            self._bias += correction
            # Use centralized LIMIT
            self._bias = max(-ESC_BIAS_LIMIT, min(ESC_BIAS_LIMIT, self._bias))
            
            self._update_bias_counter = 0
        else:
            self._update_bias_counter += 1

        # Use centralized VELOCITY
        yaw_cmd = rad_to_deg(self._bias + dither)
        self.drone.cmd_hover(ESC_VELOCITY, 0.0, yaw_cmd, self.FLIGHT_HEIGHT)
        
        # 3. Use Centralized Arrival Check
        if counter < ESC_TARGET_COUNTER:
             self._log.info("Target reached. Landing.")
             self.land()
             self._active = False
             return
```

### **Future Reference: Adding Lost Signal Watchdog**

**The Issue:**
The current controller lacks a timeout mechanism. If the radio link fails (e.g., PC crash, interference), the queue empties, and the controller enters "Zero-Order Hold" mode. It recycles the last known sensor data indefinitely. The drone sees "good battery" and "last known position," so it keeps flying blindly forever.

**The Solution:**
Implement a **Watchdog Timer** in the safety check. We compare the timestamp of the sensor data against the current system time. If the data is older than **500ms** (0.5 seconds), we assume the link is dead and trigger an immediate stop.

**Implementation Steps:**

**1. Modify `wzl/controller.py**`
Locate the `_check_safety` method and insert the age check before looking at the battery voltage.

```python
    def _check_safety(self, sample: SensorSample) -> bool:
        """Return True if safety triggered and controller should stop."""
        
        # --- NEW: Watchdog Logic ---
        # Calculate how old this sample is (monotonic time comparison)
        data_age = time.monotonic() - sample.timestamp
        
        # Threshold: 0.5 seconds (5 missed control cycles)
        if data_age > 0.5:
            LOGGER.error("Watchdog: Link lost! Data age: %.3fs. Stopping.", data_age)
            self._stop_event.set() # Triggers clean shutdown and landing
            return True
        # ---------------------------

        # Existing Battery Check
        vbat = sample.values.get("pm.vbat")
        if isinstance(vbat, numbers.Real) and float(vbat) < VBAT_MIN:
            LOGGER.warning("Battery low (%.2f V); stopping.", float(vbat))
            self._stop_event.set()
            return True
            
        return False

```

**Why 0.5 seconds?**
The control loop runs at 10Hz (100ms).

* 100-200ms lag is normal for radio retries.
* 500ms means you've missed 5 full control cycles. That's not lag; that's a disconnect.

### **Future Reference: Refactoring State Machine with Enums**

**The Issue:**
The "Quadrant Check" logic in `wzl/behaviors.py` relies on "magic numbers" (integer states like 0, 17, 20) to control the flow. This makes the code difficult to read ("What does state 17 do?") and fragile to modify (inserting a step requires renumbering everything).

**The Solution:**
Replace the raw integers with a Python `IntEnum`. This assigns meaningful names to the steps (e.g., `ALIGNING`, `DONE`), making the logic self-documenting.

**Implementation Steps:**

**1. Add Import in `wzl/behaviors.py**`
Add the enum tools to the imports at the top of the file.

```python
from enum import IntEnum, auto

```

**2. Define the Enum Class**
Add this class definition before the `Behavior` class.

```python
class QCState(IntEnum):
    INIT = 0
    # The scan legs currently rely on math (1-16), so we define the boundaries
    SCAN_START = 1
    SCAN_END = 16
    ALIGNING = 17
    DONE = 20

```

**3. Update `run_quadrant_check_step` in `wzl/behaviors.py**`
Replace the hardcoded numbers with the new Enum constants.

```python
    def run_quadrant_check_step(self, sample: SensorSample) -> bool:
        # Done state
        if self._qc_state == QCState.DONE:
            return True

        # ... (counter extraction) ...

        # Helper to transition
        def next_state():
            self._qc_state += 1
            self.sleep_timer = now

        # --- State: Start ---
        if self._qc_state == QCState.INIT:
             # ... existing logic ...

        # --- State: Scanning (Legs) ---
        # We keep the range check because of the specific math used for legs
        elif QCState.SCAN_START <= self._qc_state <= QCState.SCAN_END:
             # ... existing logic ...

        # --- State: Alignment ---
        elif self._qc_state == QCState.ALIGNING:
             # ... (Your logic, preferably the Closed-Loop one from Issue #3) ...
             
             # When finished:
             self._qc_state = QCState.DONE
             self.sleep_timer = now
             return True

        return False

```

**Benefit:**
Six months from now, you won't have to guess what `if state == 17` means. You'll see `if state == QCState.ALIGNING` and know exactly what's happening.
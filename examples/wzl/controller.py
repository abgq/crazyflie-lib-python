"""Crazyflie control loop utilities."""

from __future__ import annotations

import logging
import numbers
import threading
import time
from queue import Empty, Queue
from typing import Any, Dict, Iterable, Optional

from behaviors import Behavior, get_behavior
from constants import (
    CONTROLLER_MODE,
    CONTROL_PERIOD_MS,
    LOG_CONFIGS,
    VBAT_MIN,
)
from logger import SensorSample

LOGGER = logging.getLogger(__name__)

class CrazyflieController:
    """Consume SensorSample objects and delegate actions to behaviors.

    Per-mode logic lives in :mod:`examples.wzl.behaviors`, keeping this class
    focused on plumbing, safety, and lifecycle management.
    """

    def __init__(
        self,
        cf: "cflib.crazyflie.Crazyflie",
        sample_queue: Queue[SensorSample],
        mode: str = CONTROLLER_MODE,
        log_configs: Iterable[Dict[str, Any]] | None = None,
    ) -> None:
        """
        Args:
            cf: Connected Crazyflie instance.
            sample_queue: Queue populated by :class:`CrazyflieLogger`.
            mode: Controller mode string (e.g. ``idle`` or a future ``demo_motion``).
            log_configs: Logging configuration in sync with :class:`CrazyflieLogger`.
        """
        self._cf = cf
        self._queue = sample_queue
        self._mode = mode
        self._log_configs = list(log_configs or LOG_CONFIGS)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._last_sample: Optional[SensorSample] = None
        self._behavior: Behavior = get_behavior(mode, cf)
        self._behavior_stopped = False

    def start(self) -> None:
        """Start the background control loop."""
        if self._thread and self._thread.is_alive():
            LOGGER.debug("CrazyflieController already running")
            return

        self._behavior_stopped = False
        self._stop_event.clear()
        try:
            self._behavior.on_start()
        except Exception:  # noqa: BLE001
            LOGGER.exception("behavior.on_start() raised an exception; aborting start")
            return

        self._thread = threading.Thread(target=self._run_loop, name="cf-controller", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Request the control loop to stop and wait for it."""
        self._stop_event.set()
        if self._thread:
            try:
                self._thread.join(timeout=2.0)
            except Exception:
                LOGGER.exception("self._thread.join(timeout=2.0) raised an exception")
        if not self._behavior_stopped:
            try:
                self._behavior.on_stop()
            except Exception:  # noqa: BLE001
                LOGGER.exception("Behavior.on_stop() raised an exception")
            finally:
                self._behavior_stopped = True

    def is_running(self) -> bool:
        """Return ``True`` while the control loop thread is alive."""
        return bool(self._thread and self._thread.is_alive())

    def _run_loop(self) -> None:
        """Main control loop running at ``CONTROL_PERIOD_MS``."""
        period_s = CONTROL_PERIOD_MS / 1000.0
        LOGGER.info("Controller loop started in '%s' mode", self._mode)

        next_control_time = time.monotonic() + period_s

        try:
            while not self._stop_event.is_set():
                now = time.monotonic()
                sleep_time = next_control_time - now
                
                # Wait for the next control cycle (timer-based)
                self._stop_event.wait(max(0, sleep_time))

                # 1. DRAIN AND MERGE
                # Collect all samples that arrived during the sleep
                pending_sample = None
                while True:
                    try:
                        new_sample = self._queue.get_nowait()

                        LOGGER.info("Controller received sample: %s", self._format_sample(new_sample))

                        if pending_sample is None:
                            pending_sample = new_sample
                        else:
                            # Generic Merge: Update only what is fresh
                            for k, v in new_sample.values.items():
                                if v is not None:
                                    pending_sample.values[k] = v
                            pending_sample.timestamp = new_sample.timestamp

                    except Empty:
                        break

                # 2. UPDATE STATE (if we got fresh data)
                if pending_sample is not None:
                    self._last_sample = pending_sample

                # 3. CONTROL STEP
                # Even if we didn't get new data, we step with the old data (Zero-Order Hold)
                if self._last_sample is not None:
                    if self._check_safety(self._last_sample):
                        break
                    self._step(self._last_sample)

                next_control_time += period_s

                # Handle scheduling drift
                if next_control_time < time.monotonic():
                    next_control_time = time.monotonic() + period_s

        except Exception:  # noqa: BLE001
            LOGGER.exception("Controller loop crashed; requesting stop")
            self._stop_event.set()
        finally:
            LOGGER.info("Controller loop stopped")

    def _check_safety(self, sample: SensorSample) -> bool:
        """Return True if safety triggered and controller should stop."""
        vbat = sample.values.get("pm.vbat")
        if isinstance(vbat, numbers.Real) and float(vbat) < VBAT_MIN:
            LOGGER.warning("Battery low (%.2f V < %.2f V); stopping controller", float(vbat), VBAT_MIN)
            self._stop_event.set()
            return True
        return False

    def _format_sample(self, sample: SensorSample) -> str:
        """Format sample values with aligned columns for easy reading."""
        if not sample.values:
            return "{}"
        
        # Format each key-value pair with aligned columns on a single line
        parts = []
        for key, value in sorted(sample.values.items()):
            if value is None:
                formatted_value = "        None"
            elif isinstance(value, float):
                formatted_value = f"{value:12.6f}"
            else:
                formatted_value = f"{value:>12}"
            parts.append(f"{key}: {formatted_value}")
        
        return "  |  ".join(parts)

    def _step(self, sample: SensorSample) -> None:
        """Execute a single logical step by delegating to the active behavior."""
        try:
            self._behavior.step(sample)
        except Exception:  # noqa: BLE001
            LOGGER.exception("Behavior.step() raised an exception")
            self._stop_event.set()

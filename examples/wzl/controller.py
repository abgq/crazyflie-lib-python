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
from filters import FilterBank
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

        # New FilterBank initialization
        self._filter = FilterBank(self._log_configs)

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
            LOGGER.exception("behavior.on_start() raised an exception")

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

        try:
            while not self._stop_event.is_set():
                loop_start = time.time()
                
                # Fetch the latest sample (processing all intermediate ones in the filter)
                latest_filtered = self._drain_latest_sample()

                if latest_filtered is not None:
                    self._last_sample = latest_filtered

                # Use _last_sample directly (it is already filtered)
                if self._last_sample is not None:
                    if self._check_safety(self._last_sample):
                        break
                    self._step(self._last_sample)

                elapsed = time.time() - loop_start
                sleep_time = max(0.0, period_s - elapsed)
                if sleep_time:
                    self._stop_event.wait(timeout=sleep_time)
        except Exception:  # noqa: BLE001
            LOGGER.exception("Controller loop crashed; requesting stop")
            self._stop_event.set()
        finally:
            LOGGER.info("Controller loop stopped")

    def _drain_latest_sample(self) -> Optional[SensorSample]:
        """
        Drain the queue, apply filters to ALL samples to update state,
        and return the most recent filtered sample.
        """
        latest_filtered: Optional[SensorSample] = None
        while True:
            try:
                raw_sample = self._queue.get_nowait()
                # Apply filter immediately to every sample to keep history correct
                latest_filtered = self._apply_filters(raw_sample)
            except Empty:
                break
        return latest_filtered

    def _apply_filters(self, sample: SensorSample) -> SensorSample:
        """Apply per-variable moving averages when enabled."""
        if not self._filter.is_enabled():
            return sample

        filtered_values = dict(sample.values)
        for name, value in sample.values.items():
            if isinstance(value, numbers.Real):
                try:
                    filtered_values[name] = self._filter.update(name, float(value))
                except ValueError:
                    # FilterBank raises ValueError for unknown variables.
                    filtered_values[name] = self._filter.update(name, float(value))

        return SensorSample(timestamp=sample.timestamp, values=filtered_values)

    def _check_safety(self, sample: SensorSample) -> bool:
        """Return True if safety triggered and controller should stop."""
        vbat = sample.values.get("pm.vbat")
        if isinstance(vbat, numbers.Real) and float(vbat) < VBAT_MIN:
            LOGGER.warning("Battery low (%.2f V < %.2f V); stopping controller", float(vbat), VBAT_MIN)
            self._stop_event.set()
            return True
        return False

    def _step(self, sample: SensorSample) -> None:
        """Execute a single logical step by delegating to the active behavior."""
        try:
            self._behavior.step(sample)
        except Exception:  # noqa: BLE001
            LOGGER.exception("Behavior.step() raised an exception")
            self._stop_event.set()

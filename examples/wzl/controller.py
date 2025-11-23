"""Crazyflie control loop utilities."""

from __future__ import annotations

import logging
import numbers
import threading
import time
from collections import deque
from queue import Empty, Queue
from typing import Any, Deque, Dict, Iterable, Optional

from behaviors import Behavior, get_behavior
from constants import (
    CONTROLLER_MODE,
    CONTROL_PERIOD_MS,
    LOG_CONFIGS,
    MAX_FLIGHT_TIME_S,
    VBAT_MIN,
)
from logger import SensorSample

LOGGER = logging.getLogger(__name__)

class MovingAverageFilter:
    """Independent moving-average filters per variable."""

    def __init__(self, window_sizes: Dict[str, int]) -> None:
        """
        Args:
            window_sizes: Mapping between variable name and window length.
        """
        self.window_sizes = {name: int(size) for name, size in window_sizes.items()}
        self.buffers: Dict[str, Deque[float]] = {}
        self._enabled = any(size > 0 for size in self.window_sizes.values())

    def is_enabled(self) -> bool:
        """Return ``True`` if any variable requests filtering."""
        return self._enabled

    def update(self, name: str, value: float) -> float:
        """Return the filtered value for ``name`` given a new ``value`` sample."""
        window_size = self.window_sizes.get(name, 0)
        if window_size <= 0:
            return value

        buf = self.buffers.setdefault(name, deque(maxlen=window_size))
        buf.append(value)
        return sum(buf) / len(buf)


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
        self._filter = MovingAverageFilter(self._extract_filter_windows(self._log_configs))
        self._last_sample: Optional[SensorSample] = None
        self._behavior: Behavior = get_behavior(mode, cf)
        self._behavior_stopped = False
        self._start_time: Optional[float] = None

    def start(self) -> None:
        """Start the background control loop."""
        if self._thread and self._thread.is_alive():
            LOGGER.debug("CrazyflieController already running")
            return

        self._behavior_stopped = False
        self._stop_event.clear()
        self._start_time = time.monotonic()
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
                latest = self._drain_latest_sample()

                if latest is not None:
                    self._last_sample = latest

                if self._last_sample is not None:
                    filtered_sample = self._apply_filters(self._last_sample)
                    if self._check_safety(filtered_sample):
                        break
                    self._step(filtered_sample)

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
        """Return the most recent sample from the queue, draining older entries."""
        latest: Optional[SensorSample] = None
        while True:
            try:
                latest = self._queue.get_nowait()
            except Empty:
                break
        return latest

    def _apply_filters(self, sample: SensorSample) -> SensorSample:
        """Apply per-variable moving averages when enabled."""
        if not self._filter.is_enabled():
            return sample

        filtered_values = dict(sample.values)
        for name, value in sample.values.items():
            if isinstance(value, numbers.Real):
                filtered_values[name] = self._filter.update(name, float(value))
        return SensorSample(timestamp=sample.timestamp, values=filtered_values)

    def _check_safety(self, sample: SensorSample) -> bool:
        """Return True if safety triggered and controller should stop."""
        vbat = sample.values.get("pm.vbat")
        if isinstance(vbat, numbers.Real) and float(vbat) < VBAT_MIN:
            LOGGER.warning("Battery low (%.2f V < %.2f V); stopping controller", float(vbat), VBAT_MIN)
            self._stop_event.set()
            return True
        if MAX_FLIGHT_TIME_S is not None and self._start_time is not None:
            if time.monotonic() - self._start_time >= MAX_FLIGHT_TIME_S:
                LOGGER.warning(
                    "Max flight time %.1f s exceeded; stopping controller",
                    MAX_FLIGHT_TIME_S,
                )
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

    @staticmethod
    def _extract_filter_windows(log_configs: Iterable[Dict[str, Any]]) -> Dict[str, int]:
        """Build a mapping of variable name -> filter window size."""
        windows: Dict[str, int] = {}
        for cfg in log_configs:
            cfg_name = cfg.get("name", "unnamed")
            for entry in cfg.get("variables", []):
                name, window = CrazyflieController._parse_variable_entry(entry, cfg_name)
                if name:
                    windows[name] = max(0, window)
        return windows

    @staticmethod
    def _parse_variable_entry(entry: Any, log_name: str) -> tuple[Optional[str], int]:
        """Normalize a log variable entry."""
        if isinstance(entry, str):
            return entry, 0
        if isinstance(entry, dict):
            name = entry.get("name")
            if not name:
                LOGGER.warning("Variable entry without name in log config '%s'", log_name)
                return None, 0
            raw_window = entry.get("filter_window", 0)
            try:
                window = int(raw_window)
            except (TypeError, ValueError):
                LOGGER.warning(
                    "Invalid filter_window '%s' for variable '%s' in log config '%s'",
                    raw_window,
                    name,
                    log_name,
                )
                window = 0
            return str(name), window
        LOGGER.warning("Unsupported variable entry %r in log config '%s'", entry, log_name)
        return None, 0

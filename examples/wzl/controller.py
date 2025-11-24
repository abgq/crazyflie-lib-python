"""Crazyflie control loop utilities."""

from __future__ import annotations

import logging
import numbers
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from queue import Empty, Queue
from typing import Any, Deque, Dict, Iterable, Optional

from behaviors import Behavior, get_behavior
from constants import (
    CONTROLLER_MODE,
    CONTROL_PERIOD_MS,
    LOG_CONFIGS,
    VBAT_MIN,
)
from logger import SensorSample

LOGGER = logging.getLogger(__name__)


class SignalFilter(ABC):
    """Abstract base class for signal filters."""

    @abstractmethod
    def update(self, value: float) -> float:
        """Process a new sample and return the filtered value."""


class NoFilter(SignalFilter):
    """Returns the value immediately without modification."""

    def update(self, value: float) -> float:
        return value


class MovingAverageFilter(SignalFilter):
    """Simple Moving Average (SMA) filter."""

    def __init__(self, window_size: int) -> None:
        self.window_size = int(window_size)
        self.buffer: Deque[float] = deque(maxlen=self.window_size)

    def update(self, value: float) -> float:
        if self.window_size <= 0:
            return value
        self.buffer.append(value)
        return sum(self.buffer) / len(self.buffer)


class ExponentialFilter(SignalFilter):
    """Exponential Moving Average (EMA) filter."""

    def __init__(self, alpha: float) -> None:
        self.alpha = float(alpha)
        self.last_value: Optional[float] = None

    def update(self, value: float) -> float:
        if self.last_value is None:
            self.last_value = value
            return value
        self.last_value = self.alpha * value + (1 - self.alpha) * self.last_value
        return self.last_value


class FilterBank:
    """Manages a collection of filters mapped to variable names."""

    def __init__(self, log_configs: Iterable[Dict[str, Any]]) -> None:
        """
        Args:
            log_configs: List of log configuration dictionaries.
        """
        self.filters: Dict[str, SignalFilter] = {}
        self._enabled = False

        for cfg in log_configs:
            cfg_name = cfg.get("name", "unnamed")
            for entry in cfg.get("variables", []):
                name, filter_instance = self._create_filter_for_entry(entry, cfg_name)
                if name:
                    self.filters[name] = filter_instance
                    if not isinstance(filter_instance, NoFilter):
                        self._enabled = True

    def is_enabled(self) -> bool:
        """Return ``True`` if any variable requests filtering."""
        return self._enabled

    def update(self, name: str, value: float) -> float:
        """Return the filtered value for ``name``.

        Raises:
            ValueError: If ``name`` is not a known variable.
        """
        if name not in self.filters:
            raise ValueError(f"Unknown variable '{name}' in FilterBank update")
        return self.filters[name].update(value)

    def _create_filter_for_entry(self, entry: Any, log_name: str) -> tuple[Optional[str], SignalFilter]:
        """Parse a config entry and return (variable_name, FilterInstance)."""
        if isinstance(entry, str):
            return entry, NoFilter()

        if isinstance(entry, dict):
            name = entry.get("name")
            if not name:
                LOGGER.warning("Variable entry without name in log config '%s'", log_name)
                return None, NoFilter()

            filter_cfg = entry.get("filter")

            # Support legacy "filter_window" key if present and no "filter" dict
            # This maintains backward compatibility if needed, but the prompt says
            # "Parsing Logic: ... Look for a 'filter' key".
            # However, the prompt specifically discusses refactoring.
            # I will stick to the new spec: Look for "filter" key.
            # If "filter_window" was there, it would be ignored unless I added logic.
            # Given the strict task, I will stick to "filter" dict logic.
            # But wait, the prompt says "Iterate through the variables... Look for a 'filter' key".
            # It does not explicitly forbid legacy handling, but usually refactoring implies moving forward.
            # I'll stick to the "filter" key logic as requested.

            if not filter_cfg or not isinstance(filter_cfg, dict):
                # Fallback to check for legacy filter_window if we want to be nice?
                # The prompt implies we are rewriting the config, so maybe not needed.
                # Actually, I'll stick to the requested logic:
                # "Otherwise (or if the key is missing), create NoFilter."
                return str(name), NoFilter()

            ftype = filter_cfg.get("type")

            if ftype == "SMA":
                window = filter_cfg.get("window", 0)
                try:
                    w_int = int(window)
                    if w_int > 0:
                        return str(name), MovingAverageFilter(w_int)
                except (ValueError, TypeError):
                    LOGGER.warning("Invalid SMA window '%s' for '%s'", window, name)
                # Fallback
                return str(name), NoFilter()

            elif ftype == "EMA":
                alpha = filter_cfg.get("alpha", 1.0)
                try:
                    a_float = float(alpha)
                    return str(name), ExponentialFilter(a_float)
                except (ValueError, TypeError):
                    LOGGER.warning("Invalid EMA alpha '%s' for '%s'", alpha, name)
                return str(name), NoFilter()

            # Type "None" or unknown
            return str(name), NoFilter()

        LOGGER.warning("Unsupported variable entry %r in log config '%s'", entry, log_name)
        return None, NoFilter()


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
                    # We should just keep the raw value in that case.
                    # However, strictly speaking, the FilterBank is initialized with the same log_configs
                    # as the Logger, so all variables logged *should* be in the FilterBank.
                    # If an unknown variable appears, it might be from a different source or dynamic.
                    # Given the instruction "This should cause an error", I should probably let it crash or log it?
                    # The instruction "1. Actually this should cause an error" referred to `FilterBank.update`.
                    # So `FilterBank.update` raises the error.
                    # If I catch it here and suppress it, I'm hiding the error.
                    # If I don't catch it, the loop might crash (caught in _run_loop).
                    # I will NOT catch it here, allowing the error to propagate as requested.
                    # Re-reading: "1. Actually this should cause an error. Good catch. Fix this as well."
                    # This implies I should let it be an error.
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

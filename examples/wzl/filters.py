"""Signal filtering utilities."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Deque, Dict, Iterable, List, Optional
import numbers

# Prevent circular imports if we needed to import SensorSample,
# but we can just use Any/Duck Typing for the sample object in FilterBank.
# If we need type hinting we can use TYPE_CHECKING
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from logger import SensorSample

LOGGER = logging.getLogger(__name__)


class SignalFilter(ABC):
    """Abstract base class for signal filters."""

    @abstractmethod
    def update(self, value: float, context: Dict[str, Any] | None = None) -> Optional[float]:
        """Process a new sample and return the filtered value, or None to reject."""


class NoFilter(SignalFilter):
    """Returns the value immediately without modification."""

    def update(self, value: float, context: Dict[str, Any] | None = None) -> float:
        return value


class StepLimitFilter(SignalFilter):
    """Rejects values that jump more than ``threshold`` from the last valid value."""

    def __init__(self, threshold: float) -> None:
        self.threshold = float(threshold)
        self.last_valid_value: Optional[float] = None

    def update(self, value: float, context: Dict[str, Any] | None = None) -> Optional[float]:
        if self.last_valid_value is None:
            self.last_valid_value = value
            return value

        delta = abs(value - self.last_valid_value)
        if delta > self.threshold:
            return None

        self.last_valid_value = value
        return value


class FreshnessFilter(SignalFilter):
    """Rejects values if the trigger variable has not changed."""

    def __init__(self, trigger_var: str) -> None:
        self.trigger_var = trigger_var
        self.last_trigger_val: Any = None

    def update(self, value: float, context: Dict[str, Any] | None = None) -> Optional[float]:
        if context is None:
            # Safer to reject if no context provided when one is required
            return None

        if self.trigger_var not in context:
            # Trigger variable missing from context -> Reject
            return None

        current_trigger_val = context[self.trigger_var]

        # Initial case: always fresh
        if self.last_trigger_val is None:
            self.last_trigger_val = current_trigger_val
            return value

        # Check for change
        if current_trigger_val == self.last_trigger_val:
            # Duplicate / Stale -> Reject
            return None

        # Value is fresh
        self.last_trigger_val = current_trigger_val
        return value


class ChainFilter(SignalFilter):
    """Runs a sequence of filters; stops and returns None if any filter returns None."""

    def __init__(self, filters: List[SignalFilter]) -> None:
        self.filters = list(filters)

    def update(self, value: float, context: Dict[str, Any] | None = None) -> Optional[float]:
        current_val = value
        for f in self.filters:
            out = f.update(current_val, context=context)
            if out is None:
                return None
            current_val = out
        return current_val


class MovingAverageFilter(SignalFilter):
    """Simple Moving Average (SMA) filter."""

    def __init__(self, window_size: int) -> None:
        self.window_size = int(window_size)
        self.buffer: Deque[float] = deque(maxlen=self.window_size)

    def update(self, value: float, context: Dict[str, Any] | None = None) -> float:
        if self.window_size <= 0:
            return value
        self.buffer.append(value)
        return sum(self.buffer) / len(self.buffer)


class ExponentialFilter(SignalFilter):
    """Exponential Moving Average (EMA) filter."""

    def __init__(self, alpha: float) -> None:
        self.alpha = float(alpha)
        self.last_value: Optional[float] = None

    def update(self, value: float, context: Dict[str, Any] | None = None) -> float:
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

    def process_sample(self, sample: SensorSample) -> None:
        """Apply filters to the sample in-place.

        Args:
            sample: The SensorSample to process. values modified in-place.
        """
        if not self.is_enabled():
            return

        # Iterate through configured filters to be efficient
        for name, filter_instance in self.filters.items():
            if name in sample.values:
                val = sample.values[name]
                # Only filter numeric values
                if isinstance(val, numbers.Real):
                    try:
                        filtered_val = filter_instance.update(float(val), context=sample.values)
                        # Explicitly set to filtered_val (which might be None)
                        sample.values[name] = filtered_val
                    except Exception:
                         # Catch any filter errors to prevent crashing the loop
                         LOGGER.exception("Error filtering variable '%s'", name)
                         # Safe fallback? Leave as is or set None?
                         # User instruction implies "Fail Safe", but exception here is unexpected code error.
                         # Leaving as-is is safer than deleting data on bug, but 'None' is safer for control.
                         # I will leave as is to preserve raw data for debugging unless instructed otherwise.
                         pass

    def _create_single_filter(self, filter_cfg: Dict[str, Any], var_name: str) -> SignalFilter:
        """Instantiate a single filter from a dictionary config."""
        ftype = filter_cfg.get("type")

        if ftype == "SMA":
            window = filter_cfg.get("window", 0)
            try:
                w_int = int(window)
                if w_int > 0:
                    return MovingAverageFilter(w_int)
            except (ValueError, TypeError):
                LOGGER.warning("Invalid SMA window '%s' for '%s'", window, var_name)
            return NoFilter()

        elif ftype == "EMA":
            alpha = filter_cfg.get("alpha", 1.0)
            try:
                a_float = float(alpha)
                return ExponentialFilter(a_float)
            except (ValueError, TypeError):
                LOGGER.warning("Invalid EMA alpha '%s' for '%s'", alpha, var_name)
            return NoFilter()

        elif ftype == "StepLimit":
            threshold = filter_cfg.get("threshold", 0.0)
            try:
                t_float = float(threshold)
                return StepLimitFilter(t_float)
            except (ValueError, TypeError):
                LOGGER.warning("Invalid StepLimit threshold '%s' for '%s'", threshold, var_name)
            return NoFilter()

        elif ftype == "Freshness":
            trigger = filter_cfg.get("trigger")
            if trigger:
                return FreshnessFilter(str(trigger))
            else:
                 LOGGER.warning("Freshness filter for '%s' missing 'trigger'", var_name)

        # Type "None" or unknown
        return NoFilter()

    def _create_filter_for_entry(self, entry: Any, log_name: str) -> tuple[Optional[str], SignalFilter]:
        """Parse a config entry and return (variable_name, FilterInstance)."""
        if isinstance(entry, str):
            return entry, NoFilter()

        if isinstance(entry, dict):
            name = entry.get("name")
            if not name:
                LOGGER.warning("Variable entry without name in log config '%s'", log_name)
                return None, NoFilter()
            var_name = str(name)

            filter_cfg = entry.get("filter")
            if not filter_cfg:
                return var_name, NoFilter()

            # Case A: List of filters -> ChainFilter
            if isinstance(filter_cfg, list):
                chain_list = []
                for item in filter_cfg:
                    if isinstance(item, dict):
                        chain_list.append(self._create_single_filter(item, var_name))
                if not chain_list:
                    return var_name, NoFilter()
                return var_name, ChainFilter(chain_list)

            # Case B: Single dict
            if isinstance(filter_cfg, dict):
                return var_name, self._create_single_filter(filter_cfg, var_name)

            # Unknown type
            return var_name, NoFilter()

        LOGGER.warning("Unsupported variable entry %r in log config '%s'", entry, log_name)
        return None, NoFilter()

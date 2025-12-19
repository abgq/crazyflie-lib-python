"""Signal filtering utilities."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import deque
from typing import Deque, Dict, List, Optional, Type, Any
import numbers

from models import FilterConfig, LogBlockConfig, LogVariableConfig, SensorSample, FilterContext

LOGGER = logging.getLogger(__name__)


class SignalFilter(ABC):
    """Abstract base class for signal filters."""

    @abstractmethod
    def update(self, value: float, context: FilterContext | None = None) -> Optional[float]:
        """Process a new sample and return the filtered value, or None to reject."""


class NoFilter(SignalFilter):
    """Returns the value immediately without modification."""

    def update(self, value: float, context: FilterContext | None = None) -> float:
        return value


class StepLimitFilter(SignalFilter):
    """Rejects values that jump more than ``threshold`` from the last valid value."""

    def __init__(self, config: FilterConfig) -> None:
        self.threshold = float(config.threshold) if config.threshold is not None else 0.0
        self.last_valid_value: Optional[float] = None

    def update(self, value: float, context: FilterContext | None = None) -> Optional[float]:
        if self.last_valid_value is None:
            self.last_valid_value = value
            return value

        delta = abs(value - self.last_valid_value)
        if delta > self.threshold:
            LOGGER.debug("StepLimitFilter rejected value %f (delta %f > threshold %f)", value, delta, self.threshold)
            return None

        self.last_valid_value = value
        return value


class FreshnessFilter(SignalFilter):
    """Rejects values if the trigger variable has not changed."""

    def __init__(self, config: FilterConfig) -> None:
        self.trigger_var = str(config.trigger) if config.trigger else ""
        self.last_trigger_val: Any = None

    def update(self, value: float, context: FilterContext | None = None) -> Optional[float]:
        if context is None:
            # Safer to reject if no context provided when one is required
            return None

        if not self.trigger_var or self.trigger_var not in context:
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

    def update(self, value: float, context: FilterContext | None = None) -> Optional[float]:
        current_val = value
        for f in self.filters:
            out = f.update(current_val, context=context)
            if out is None:
                return None
            current_val = out
        return current_val


class MovingAverageFilter(SignalFilter):
    """Simple Moving Average (SMA) filter."""

    def __init__(self, config: FilterConfig) -> None:
        self.window_size = int(config.window) if config.window is not None else 0
        self.buffer: Deque[float] = deque(maxlen=self.window_size)

    def update(self, value: float, context: FilterContext | None = None) -> float:
        if self.window_size <= 0:
            return value
        self.buffer.append(value)
        return sum(self.buffer) / len(self.buffer)


class ExponentialFilter(SignalFilter):
    """Exponential Moving Average (EMA) filter."""

    def __init__(self, config: FilterConfig) -> None:
        self.alpha = float(config.alpha) if config.alpha is not None else 1.0
        self.last_value: Optional[float] = None

    def update(self, value: float, context: FilterContext | None = None) -> float:
        if self.last_value is None:
            self.last_value = value
            return value
        self.last_value = self.alpha * value + (1 - self.alpha) * self.last_value
        return self.last_value


class FilterBank:
    """Manages a collection of filters mapped to variable names."""

    # Factory mapping
    FILTER_MAP: Dict[str, Type[SignalFilter]] = {
        "SMA": MovingAverageFilter,
        "EMA": ExponentialFilter,
        "StepLimit": StepLimitFilter,
        "Freshness": FreshnessFilter,
    }

    def __init__(self, log_configs: List[LogBlockConfig]) -> None:
        """
        Args:
            log_configs: List of log configuration objects.
        """
        self.filters: Dict[str, SignalFilter] = {}
        self._enabled = False

        for cfg in log_configs:
            for variable in cfg.variables:
                name, filter_instance = self._create_filter_for_entry(variable)
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
                         pass

    def _create_single_filter(self, config: FilterConfig, var_name: str) -> SignalFilter:
        """Instantiate a single filter from a config object using the factory map."""
        filter_cls = self.FILTER_MAP.get(config.type)
        if filter_cls:
            try:
                return filter_cls(config)
            except Exception:
                LOGGER.warning("Failed to initialize filter '%s' for '%s'", config.type, var_name, exc_info=True)
        else:
            LOGGER.warning("Unknown filter type '%s' for '%s'", config.type, var_name)

        return NoFilter()

    def _create_filter_for_entry(self, variable: LogVariableConfig) -> tuple[Optional[str], SignalFilter]:
        """Parse a variable config and return (variable_name, FilterInstance)."""
        var_name = variable.name

        if not variable.filters:
            return var_name, NoFilter()

        # Case A: List of filters -> ChainFilter
        # Since we normalized constants, this is the only case effectively,
        # but LogVariableConfig.filters is a list.
        if len(variable.filters) > 1:
            chain_list = []
            for f_cfg in variable.filters:
                 chain_list.append(self._create_single_filter(f_cfg, var_name))
            return var_name, ChainFilter(chain_list)

        # Case B: Single filter (optimization)
        if len(variable.filters) == 1:
            return var_name, self._create_single_filter(variable.filters[0], var_name)

        return var_name, NoFilter()

"""Signal filtering utilities."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Deque, Dict, Iterable, Optional

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

            if not filter_cfg or not isinstance(filter_cfg, dict):
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

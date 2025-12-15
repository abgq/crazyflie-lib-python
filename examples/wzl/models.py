"""Shared data models and configuration schemas."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass(slots=True)
class SensorSample:
    """Snapshot of the latest Crazyflie log values."""
    timestamp: float
    values: Dict[str, Any]

@dataclass(slots=True, frozen=True)
class FilterConfig:
    """Configuration for a single signal filter.

    This class handles the polymorphic nature of filter definitions.
    Fields are optional depending on the 'type'.
    """
    type: str
    window: Optional[int] = None
    alpha: Optional[float] = None
    threshold: Optional[float] = None
    trigger: Optional[str] = None

@dataclass(slots=True, frozen=True)
class LogVariableConfig:
    """Configuration for a single variable within a log block."""
    name: str
    fetch_as: Optional[str] = None
    filters: List[FilterConfig] = field(default_factory=list)

@dataclass(slots=True, frozen=True)
class LogBlockConfig:
    """Configuration for a Crazyflie log block."""
    name: str
    period_ms: int
    variables: List[LogVariableConfig]

# Type alias for the context passed to filters (essentially the full sample values)
FilterContext = Dict[str, Any]

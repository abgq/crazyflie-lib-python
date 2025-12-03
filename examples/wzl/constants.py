"""Configuration constants for the wzl Crazyflie example.

Extend the :data:`LOG_CONFIGS` list with additional dictionaries whenever you
need to stream more variables or change their log periods. Each variable can
carry its own ``filter_window`` value (set to ``0`` to disable filtering).
"""

from __future__ import annotations

from typing import Any, Dict, List

CF_URI: str = "radio://0/80/2M/E7E7E7E7E7"
"""Default Crazyflie URI. Update this to match the actual link configuration."""

LOG_CONFIGS: List[Dict[str, Any]] = [
    {
        "name": "power",
        "period_ms": 1000,  # 1 Hz logging
        "variables": [
            {
                "name": "pm.vbat",
                # No filter configuration (defaulting to NoFilter)
            },
        ],
    },
    {
        "name": "UWB",
        "period_ms": 20,  # 50 Hz logging
        "variables": [
            {
                "name": "dw1k.rangingCounter",
                "filter": [
                    {
                        "type": "StepLimit",
                        "threshold": 1000
                    },
                    {
                        "type": "SMA",
                        "window": 5
                    }
                ]
            },
        ],
    },
    {
        "name": "state",
        "period_ms": 100,  # 10 Hz logging
        "variables": [
            {
                "name": "kalman.stateZ",
            },
        ],
    },
]
"""List of log block configurations consumed by :class:`CrazyflieLogger`."""

CONTROL_PERIOD_MS: int = 100
"""Controller loop period in milliseconds."""

VBAT_MIN: float = 3.3
"""Minimum safe battery voltage, in volts."""

QUEUE_MAX_SIZE: int = 1
"""Maximum number of :class:`SensorSample` objects retained in the queue."""

CONTROLLER_MODE: str = "run_tumble"
"""
Default controller mode. Options: 'idle', 'run_tumble', and 'sinusoidal'. Default is 'idle'.
"""

# --- DW1000 / UWB calibration constants ---

# Speed of light in vacuum [m/s]. Feel free to tweak if you want to be pedantic.
SPEED_OF_LIGHT: float = 299_792_458.0

# Antenna delay in "ranging counter" units. Set this from your calibration;
# inaccurate values lead to incorrect distances. ranging_counter_to_distance()
# assumes this is already tuned for your hardware.
DW1K_ANTENNA_DELAY_RC: int = 17280  # TODO: set from your calibration procedure

# Conversion from one ranging-counter unit to seconds. Must be derived from
# your DW1000 setup or measured experimentally; keep as-is until calibrated.
DW1K_RC_TO_SECONDS: float = (1.0 / 499.2e6 / 128.0)  # TODO: replace with calibrated value

# Factor depending on ranging scheme (1.0 for one-way TOF, 0.5 for symmetric
# TWR). ranging_counter_to_distance() relies on this being set correctly.
DW1K_TOF_SCALING: float = 0.5

__all__ = [
    "CF_URI",
    "LOG_CONFIGS",
    "CONTROL_PERIOD_MS",
    "VBAT_MIN",
    "QUEUE_MAX_SIZE",
    "CONTROLLER_MODE",
    "SPEED_OF_LIGHT",
    "DW1K_ANTENNA_DELAY_RC",
    "DW1K_RC_TO_SECONDS",
    "DW1K_TOF_SCALING",
]

"""Simulation logger that mimics Crazyflie log streaming."""

from __future__ import annotations

import logging
import time
from queue import Empty, Full, Queue
from threading import Event, Thread
from typing import Dict

from examples.wzl import constants as real_constants
from examples.wzl.logger import SensorSample

from . import constants
from .world import World

LOGGER = logging.getLogger(__name__)


def distance_to_counter(distance_m: float) -> float:
    """Convert meters to a synthetic ``dw1k.rangingCounter`` value."""
    tof_seconds = distance_m / real_constants.SPEED_OF_LIGHT
    effective = tof_seconds / real_constants.DW1K_RC_TO_SECONDS
    raw = effective / real_constants.DW1K_TOF_SCALING + 4 * real_constants.DW1K_ANTENNA_DELAY_RC
    return max(0.0, raw)


class SimULogger:
    """Produce SensorSample objects from the simulated world."""

    def __init__(
        self,
        world: World,
        sample_queue: "Queue[SensorSample]",
        stop_event: Event,
        pause_event: Event,
    ) -> None:
        self._world = world
        self._queue = sample_queue
        self._stop_event = stop_event
        self._pause_event = pause_event
        self._thread = Thread(target=self._run, name="sim-logger", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=1.0)

    def _run(self) -> None:
        period = 1.0 / constants.CONTROL_RATE_HZ
        while not self._stop_event.is_set():
            if self._pause_event.is_set():
                time.sleep(0.05)
                continue

            values = self._collect_sample()
            sample = SensorSample(timestamp=time.time(), values=values)
            self._push(sample)
            time.sleep(period)

    def _collect_sample(self) -> Dict[str, float]:
        distances = self._world.get_noisy_distances()
        values: Dict[str, float] = {"pm.vbat": 3.9}
        if distances:
            primary = distances[min(distances.keys())]
            values["dw1k.rangingCounter"] = distance_to_counter(primary)
            for idx, dist in distances.items():
                values[f"uwb.distance.{idx}"] = dist
        return values

    def _push(self, sample: SensorSample) -> None:
        while not self._stop_event.is_set():
            try:
                self._queue.put_nowait(sample)
                return
            except Full:
                try:
                    self._queue.get_nowait()
                except Empty:
                    pass


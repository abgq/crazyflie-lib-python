"""Simulation controller that reuses the real behavior layer."""

from __future__ import annotations

import logging
import threading
import time
from queue import Empty, Queue
from typing import Optional

from examples.wzl.logger import SensorSample

from . import constants
from .sim_commander import ensure_sim_motion_commander
from .sim_cf import SimCrazyflie

LOGGER = logging.getLogger(__name__)

# Ensure behaviors import the simulated MotionCommander.
ensure_sim_motion_commander()
from examples.wzl import behaviors  # noqa: E402  (import after patching)


class SimController:
    """Runs Crazyflie behaviors in the simulated environment."""

    def __init__(
        self,
        cf: SimCrazyflie,
        sample_queue: "Queue[SensorSample]",
        mode: str,
        stop_event: threading.Event,
        pause_event: threading.Event,
        algorithm: str = "behavior",
    ) -> None:
        self._cf = cf
        self._queue = sample_queue
        self._mode = mode
        self._stop_event = stop_event
        self._pause_event = pause_event
        self._thread: Optional[threading.Thread] = None
        self._algorithm = algorithm
        self._behavior = behaviors.get_behavior(mode, cf) if algorithm == "behavior" else None
        self._custom = None
        if algorithm != "behavior":
            from . import algorithms

            self._custom = algorithms.get_algorithm(cf)
        self._behavior_started = False
        self._behavior_stopped = False

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        if self._algorithm == "behavior" and self._behavior:
            try:
                self._behavior.on_start()
                self._behavior_started = True
            except Exception:  # noqa: BLE001
                LOGGER.exception("Behavior.on_start() failed")
        elif self._custom:
            self._custom.on_start()

        self._thread = threading.Thread(target=self._run, name="sim-controller", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        if self._algorithm == "behavior":
            self._stop_behavior()
        elif self._custom:
            self._custom.on_stop()
        self._cf.reset_velocity()

    def _stop_behavior(self) -> None:
        if self._behavior_started and not self._behavior_stopped and self._behavior:
            try:
                self._behavior.on_stop()
            except Exception:  # noqa: BLE001
                LOGGER.exception("Behavior.on_stop() failed")
            finally:
                self._behavior_stopped = True

    def _run(self) -> None:
        period = 1.0 / constants.CONTROL_RATE_HZ
        last_sample: Optional[SensorSample] = None

        while not self._stop_event.is_set():
            if self._pause_event.is_set():
                time.sleep(0.05)
                continue

            sample = self._drain_latest_sample()
            if sample is not None:
                last_sample = sample

            if last_sample is not None:
                try:
                    if self._algorithm == "behavior" and self._behavior:
                        self._behavior.step(last_sample)
                    elif self._custom:
                        self._custom.step(last_sample)
                except Exception:  # noqa: BLE001
                    LOGGER.exception("Controller loop crashed; stopping simulation")
                    self._stop_event.set()
                    break

            time.sleep(period)

    def _drain_latest_sample(self) -> Optional[SensorSample]:
        latest: Optional[SensorSample] = None
        while True:
            try:
                latest = self._queue.get_nowait()
            except Empty:
                break
        return latest

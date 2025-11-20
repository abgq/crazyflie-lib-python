"""Entry point for the wzl simulation."""

from __future__ import annotations

import logging
import threading
import time
from queue import Queue

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.wzl import constants as real_constants
from examples.wzl.logger import SensorSample

try:
    from . import constants
    from .sim_cf import SimCrazyflie
    from .sim_controller import SimController
    from .sim_logger import SimULogger
    from .vis import SimulationViewer
    from .world import World
except ImportError:  # pragma: no cover - script execution fallback
    from examples.wzl.simulation import constants  # type: ignore
    from examples.wzl.simulation.sim_cf import SimCrazyflie  # type: ignore
    from examples.wzl.simulation.sim_controller import SimController  # type: ignore
    from examples.wzl.simulation.sim_logger import SimULogger  # type: ignore
    from examples.wzl.simulation.vis import SimulationViewer  # type: ignore
    from examples.wzl.simulation.world import World  # type: ignore


LOGGER = logging.getLogger(__name__)


class SimulationControl:
    """Shared state for pause/stop coordination."""

    def __init__(self) -> None:
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()

    def request_stop(self) -> None:
        self.stop_event.set()

    def toggle_pause(self) -> None:
        if self.pause_event.is_set():
            self.pause_event.clear()
            LOGGER.info("Simulation resumed")
        else:
            self.pause_event.set()
            LOGGER.info("Simulation paused")


def _world_loop(world: World, control: SimulationControl) -> None:
    dt = 1.0 / constants.CONTROL_RATE_HZ
    while not control.stop_event.is_set():
        if control.pause_event.is_set():
            time.sleep(0.05)
            continue
        world.step(dt)
        time.sleep(dt)


def run() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    LOGGER.info("Starting Crazyflie simulation")

    control = SimulationControl()
    world = World()
    sim_cf = SimCrazyflie(world)
    world.attach_command_source(sim_cf.get_command_velocity)

    sample_queue: Queue[SensorSample] = Queue(maxsize=256)

    sim_logger = SimULogger(world, sample_queue, control.stop_event, control.pause_event)
    controller = SimController(
        sim_cf,
        sample_queue,
        mode=real_constants.CONTROLLER_MODE,
        stop_event=control.stop_event,
        pause_event=control.pause_event,
        algorithm=constants.SIM_ALGORITHM,
    )

    world_thread = threading.Thread(target=_world_loop, args=(world, control), name="world", daemon=True)
    world_thread.start()
    sim_logger.start()
    controller.start()

    viewer = SimulationViewer(
        world=world,
        stop_event=control.stop_event,
        stop_callback=control.request_stop,
        pause_toggle=control.toggle_pause,
        reset_callback=world.reset,
    )

    try:
        viewer.run()
    except KeyboardInterrupt:
        LOGGER.info("KeyboardInterrupt – stopping simulation")
        control.request_stop()
    finally:
        controller.stop()
        sim_logger.stop()
        control.request_stop()
        world_thread.join(timeout=1.0)

    LOGGER.info("Simulation stopped")


if __name__ == "__main__":
    run()

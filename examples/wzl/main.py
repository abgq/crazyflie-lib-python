"""Entry point for the wzl Crazyflie example."""

from __future__ import annotations

import logging
import multiprocessing
import queue
import time
from typing import Optional
import threading

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

from constants import CF_URI, CONTROLLER_MODE, LOG_CONFIGS, QUEUE_MAX_SIZE
from controller import CrazyflieController
from logger import CrazyflieLogger, SensorSample
from plotter import run_plotter


LOGGER = logging.getLogger(__name__)

def main() -> None:
    """Connect to a Crazyflie, start logging and run the controller."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    LOGGER.info("Initializing Crazyflie drivers")
    cflib.crtp.init_drivers()

    sample_queue: queue.Queue[SensorSample] = queue.Queue(maxsize=QUEUE_MAX_SIZE)
    plotter_queue: multiprocessing.Queue = multiprocessing.Queue(maxsize=100)

    controller: Optional[CrazyflieController] = None
    cf_logger: Optional[CrazyflieLogger] = None
    plotter_process: Optional[multiprocessing.Process] = None

    # Start Plotter Process
    plotter_process = multiprocessing.Process(
        target=run_plotter,
        args=(plotter_queue,),
        name="cf-plotter"
    )
    plotter_process.start()

    with SyncCrazyflie(CF_URI, cf=Crazyflie(rw_cache="./cache")) as scf:
        try:
            LOGGER.info("Connected to Crazyflie at %s", CF_URI)
            cf = scf.cf
            cf_logger = CrazyflieLogger(cf, sample_queue, LOG_CONFIGS)
            controller = CrazyflieController(cf, sample_queue, CONTROLLER_MODE, LOG_CONFIGS, plotter_queue=plotter_queue)

            cf_logger.start()
            time.sleep(2.0)  # Allow some time to fill the queue
            controller.start()

            # Future GUI components can observe the same queue without
            # modifying the controller or logger.
            while controller.is_running():
                time.sleep(0.01)

                # Check if plotter is still alive (optional)
                if not plotter_process.is_alive():
                     LOGGER.warning("Plotter process died unexpectedly.")
                     # We can choose to respawn or just log it.
                     # For now, we continue flying.

        except KeyboardInterrupt:
            LOGGER.info("KeyboardInterrupt received; stopping controller")
        except Exception:  # noqa: BLE001
            LOGGER.exception("Unexpected error in main loop")
        finally:
            if controller:
                LOGGER.info("Stopping controller...")
                controller.stop()
                LOGGER.info("Controller stopped")
            if cf_logger:
                LOGGER.info("Stopping logger...")
                cf_logger.stop()
                LOGGER.info("Logger stopped")

            # Stop Plotter
            if plotter_process and plotter_process.is_alive():
                LOGGER.info("Stopping plotter process...")
                try:
                    plotter_queue.put(None) # Sentinel
                    plotter_process.join(timeout=3.0)
                except Exception:
                    LOGGER.exception("Error joining plotter process")

                if plotter_process.is_alive():
                    LOGGER.warning("Plotter process did not exit cleanly; terminating.")
                    plotter_process.terminate()

    LOGGER.info("Crazyflie session closed")
    LOGGER.info("Threads still alive: %s", threading.enumerate())



if __name__ == "__main__":
    main()

"""Entry point for the wzl Crazyflie example."""

from __future__ import annotations

import logging
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
from interfaces import CrazyflieDrone



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
    cf_controller: Optional[CrazyflieController] = None
    cf_logger: Optional[CrazyflieLogger] = None

    with SyncCrazyflie(CF_URI, cf=Crazyflie(rw_cache="./cache")) as scf:
        try:
            LOGGER.info("Connected to Crazyflie at %s", CF_URI)
            cf = scf.cf

            # Instantiate the interface wrapper
            cf_interface = CrazyflieDrone(cf)

            cf_logger = CrazyflieLogger(cf, sample_queue, LOG_CONFIGS)
            cf_controller = CrazyflieController(cf_interface, sample_queue, CONTROLLER_MODE, LOG_CONFIGS)

            cf_logger.start()
            time.sleep(2.0)  # Allow some time to fill the queue
            cf_controller.start()

            # Future GUI components can observe the same queue without
            # modifying the controller or logger.
            while cf_controller.is_running():
                time.sleep(0.01)
        except KeyboardInterrupt:
            LOGGER.info("KeyboardInterrupt received; stopping controller")
        except Exception:  # noqa: BLE001
            LOGGER.exception("Unexpected error in main loop")
        finally:
            if cf_controller:
                LOGGER.info("Stopping controller...")
                cf_controller.stop()
                LOGGER.info("Controller stopped")
            if cf_logger:
                LOGGER.info("Stopping logger...")
                cf_logger.stop()
                LOGGER.info("Logger stopped")
    LOGGER.info("Crazyflie session closed")
    LOGGER.info("Threads still alive: %s", threading.enumerate())



if __name__ == "__main__":
    main()

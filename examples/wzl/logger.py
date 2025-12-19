"""Crazyflie logging utilities."""

from __future__ import annotations

import logging
import threading
import time
from queue import Empty, Full, Queue
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from cflib.crazyflie.log import LogConfig

if TYPE_CHECKING:
    from recorder import DataRecorder

from constants import LOG_CONFIGS
from filters import FilterBank
from models import LogBlockConfig, LogVariableConfig, SensorSample


LOGGER = logging.getLogger(__name__)


class CrazyflieLogger:
    """Attach Crazyflie log blocks and forward samples into a queue."""

    def __init__(
        self,
        cf: "cflib.crazyflie.Crazyflie",
        sample_queue: Queue[SensorSample],
        log_configs: List[LogBlockConfig] | None = None,
        recorder: DataRecorder | None = None,
    ) -> None:
        """
        Prepare the logger.

        Args:
            cf: Connected Crazyflie instance.
            sample_queue: Queue that will receive :class:`SensorSample` objects.
            log_configs: Optional overrides for :data:`constants.LOG_CONFIGS`.
            recorder: Optional instance of DataRecorder to archive all samples.
        """
        self._cf = cf
        self._queue = sample_queue
        self._log_configs = list(log_configs or LOG_CONFIGS)
        self._recorder = recorder
        self._lock = threading.Lock()
        self._latest_values: Dict[str, Any] = {}
        self._logconfs: List[LogConfig] = []
        self._running = False
        self._filter = FilterBank(self._log_configs)

    def start(self) -> None:
        """Create log configurations and start streaming data."""
        if self._running:
            LOGGER.debug("CrazyflieLogger already running")
            return

        self._running = True
        for cfg in self._log_configs:
            name = cfg.name
            period_ms = cfg.period_ms
            variables = cfg.variables
            if not variables:
                LOGGER.warning("Skipping log config '%s' without variables", name)
                continue

            logconf = LogConfig(name=name, period_in_ms=period_ms)
            for entry in variables:
                var_name = entry.name
                fetch_as = entry.fetch_as
                try:
                    logconf.add_variable(var_name, fetch_as)
                except KeyError:
                    LOGGER.warning("Variable '%s' not found for log config '%s'", var_name, name)

            logconf.data_received_cb.add_callback(self._log_callback)
            logconf.error_cb.add_callback(self._log_error_callback)

            try:
                self._cf.log.add_config(logconf)
                logconf.start()
                self._logconfs.append(logconf)
                LOGGER.info("Started log config '%s' (%d ms)", name, period_ms)
            except KeyError:
                LOGGER.exception("Failed to add log config '%s'", name)

    def stop(self) -> None:
        """Stop all log configurations and detach callbacks."""
        if not self._running:
            return

        self._running = False
        for logconf in self._logconfs:
            try:
                logconf.stop()
            except Exception:  # noqa: BLE001 - best-effort cleanup
                LOGGER.exception("Failed to stop log config '%s'", logconf.name)
        self._logconfs.clear()

    def _log_callback(self, timestamp: float, data: Dict[str, Any], logconf: LogConfig) -> None:
        """Handle new log data coming from cflib."""
        if not self._running:
            return

        now = time.monotonic()

        LOGGER.debug("Received log data from '%s' at %f: %s", logconf.name, now, data)
        
        # 1. Update State with RAW data (The "Zero-Order Hold")
        with self._lock:
            self._latest_values.update(data)
            snapshot = dict(self._latest_values)

        # 2. Create Sample
        sample = SensorSample(timestamp=now, values=snapshot)

        # Preserve Raw Data for Plotting
        raw_snapshot = snapshot.copy()
        for k, v in raw_snapshot.items():
            sample.values[f"{k}_raw"] = v

        # 3. Apply Filtering (Sanitize the Output)
        # This will set values to None if they are stale or outliers
        if self._filter.is_enabled():
            self._filter.process_sample(sample)

        if self._recorder:
            self._recorder.record(sample)

        # 4. Push to Controller
        self._push_sample(sample)

    def _log_error_callback(self, logconf: LogConfig, msg: str) -> None:
        """Receive log errors from cflib."""
        LOGGER.error("Log config '%s' reported error: %s", logconf.name, msg)

    def _push_sample(self, sample: SensorSample) -> None:
        """Push a sample into the queue, dropping the oldest if full."""
        try:
            self._queue.put_nowait(sample)
        except Full:
            LOGGER.warning("Sample queue full; dropping oldest sample to insert new one")
            try:
                self._queue.get_nowait()
            except Empty:
                pass
            try:
                self._queue.put_nowait(sample)
            except Full:
                LOGGER.error("Failed to push sample into full queue")

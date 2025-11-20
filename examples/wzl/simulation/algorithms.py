"""Custom algorithm hooks for the simulator."""

from __future__ import annotations

from examples.wzl.logger import SensorSample


class NoOpAlgorithm:
    """Fallback algorithm that does nothing."""

    def __init__(self, cf) -> None:  # noqa: ANN001
        self._cf = cf

    def on_start(self) -> None:
        self._cf.reset_velocity()

    def step(self, sample: SensorSample) -> None:  # noqa: ARG002
        # Intentionally left blank; extend this class for custom control.
        return

    def on_stop(self) -> None:
        self._cf.reset_velocity()


def get_algorithm(cf) -> NoOpAlgorithm:  # noqa: ANN001
    """Return the algorithm implementation to use."""
    return NoOpAlgorithm(cf)


__all__ = ["get_algorithm", "NoOpAlgorithm"]

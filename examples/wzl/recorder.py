"""Data recording utility for post-hoc analysis."""

from __future__ import annotations

from typing import List, Dict, Any
from models import SensorSample

class DataRecorder:
    """Simple in-memory storage for sensor samples."""

    def __init__(self) -> None:
        self._history: List[SensorSample] = []

    def record(self, sample: SensorSample) -> None:
        """Append a sample to the history.

        Note: We store the reference. If the sample object is modified
        later (which shouldn't happen), the history will reflect that.
        """
        self._history.append(sample)

    def get_data(self) -> List[Dict[str, Any]]:
        """Flatten the history into a list of dictionaries for easy plotting.

        Returns:
            List of dicts, e.g. [{'time': 1.2, 'var_a': 10, 'var_a_raw': 12}, ...]
        """
        export_data = []
        if not self._history:
            return export_data

        # Normalize start time to 0.0 seconds
        start_time = self._history[0].timestamp

        for s in self._history:
            # Create a flat dict merging timestamp and all values
            row = {"time": s.timestamp - start_time}
            row.update(s.values)
            export_data.append(row)

        return export_data

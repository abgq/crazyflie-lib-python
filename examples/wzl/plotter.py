"""Standalone PyQt6 application for real-time data visualization."""

from __future__ import annotations

import collections
import logging
import multiprocessing
import queue
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication, QMainWindow

LOGGER = logging.getLogger(__name__)

class CrazyfliePlotter(QMainWindow):
    """
    Main window for the Crazyflie Observer dashboard.

    Reads data from a multiprocessing.Queue and updates plots dynamically
    based on a provided configuration dictionary.
    """

    def __init__(
        self,
        plotter_queue: multiprocessing.Queue,
        config: Dict[str, Any]
    ) -> None:
        super().__init__()
        self._queue = plotter_queue
        self._config = config
        self._buffers: Dict[str, collections.deque] = {}
        self._curves: Dict[str, pg.PlotDataItem] = {}
        self._text_items: Dict[str, pg.TextItem] = {}

        # Helper to map fields to their TextItems for updates
        # Structure: { source_key: [ (TextItem, format_str), ... ] }
        self._text_field_map: Dict[str, List[tuple[pg.TextItem, str]]] = collections.defaultdict(list)

        self.setWindowTitle(config.get("window_title", "Crazyflie Observer"))
        self.resize(1000, 600)

        # Main Layout Container
        self._layout_widget = pg.GraphicsLayoutWidget()
        self.setCentralWidget(self._layout_widget)
        self._build_layout()

        # Update Timer
        self._timer = QTimer()
        self._timer.timeout.connect(self._update)
        self._timer.start(config.get("refresh_rate_ms", 50))

    def _build_layout(self) -> None:
        """Parse configuration and build the dashboard layout."""
        buffer_size = self._config.get("buffer_size", 200)

        layout_rows = self._config.get("layout", [])
        for row in layout_rows:
            for item_config in row:
                item_type = item_config.get("type")
                title = item_config.get("title", "")

                if item_type == "plot":
                    plot_item = self._layout_widget.addPlot(title=title)
                    plot_item.setLabel("left", item_config.get("ylabel", ""))
                    plot_item.showGrid(x=True, y=True)
                    plot_item.addLegend()

                    for trace in item_config.get("traces", []):
                        source = trace["source"]
                        # Initialize buffer if not exists
                        if source not in self._buffers:
                            self._buffers[source] = collections.deque(maxlen=buffer_size)

                        # Create curve
                        pen = pg.mkPen(
                            color=trace.get("color", "w"),
                            width=trace.get("width", 1)
                        )
                        curve = plot_item.plot(name=trace.get("name", source), pen=pen)
                        self._curves[source] = curve

                elif item_type == "text":
                    # For text items, we use a ViewBox or just a Layout item?
                    # GraphicsLayoutWidget cells expect GraphicsItems.
                    # We can use a LayoutItem (GLViewWidget) or add a ViewBox and put TextItem in it.
                    # Simpler: Create a Layout, add labels to it.
                    # pyqtgraph's Layout capability inside GraphicsLayoutWidget is grid-based.
                    # We can add a ViewBox and disable mouse interaction to hold text.
                    view = self._layout_widget.addViewBox()
                    view.setAspectLocked()
                    view.disableAutoRange()
                    view.setMouseEnabled(x=False, y=False)

                    # We will stack text items vertically
                    # Since we can't easily do HTML layout in ViewBox, we'll just position them manually
                    # or use the title of the viewbox.
                    # Actually, let's use a specialized approach: a formatted string in a single TextItem
                    # or multiple TextItems.
                    # Let's try multiple TextItems arranged vertically.

                    # Title
                    # Note: addViewBox doesn't support title directly like addPlot
                    # So we might want to add a LabelItem above it?
                    # For simplicity, we assume the user is okay with basic text rendering.

                    fields = item_config.get("fields", [])
                    count = len(fields)

                    for i, field in enumerate(fields):
                        source = field["source"]
                        label = field.get("label", source)
                        fmt = field.get("fmt", "{}")
                        color = field.get("color", "#FFFFFF")

                        text_item = pg.TextItem(
                            text=f"{label}: --",
                            color=color,
                            anchor=(0, 0)
                        )
                        # Spread them out vertically in the viewbox [0, 1] x [0, 1] coordinate system
                        # 0 is bottom, 1 is top.
                        y_pos = 1.0 - (i + 1) * (1.0 / (count + 1))
                        text_item.setPos(0.1, y_pos)
                        view.addItem(text_item)

                        self._text_field_map[source].append((text_item, f"{label}: {fmt}"))

            self._layout_widget.nextRow()

    def _update(self) -> None:
        """Process all pending packets and update the GUI."""
        updates_needed: set[str] = set()
        latest_values: Dict[str, Any] = {}

        # 1. Bulk Drain
        while True:
            try:
                packet = self._queue.get_nowait()
            except queue.Empty:
                break

            # Sentinel Check
            if packet is None:
                LOGGER.info("Plotter received shutdown sentinel.")
                self.close()
                return

            # Update Latest Cache for Text Fields
            latest_values.update(packet)

            # Update Buffers for Plots
            # Note: We need to update buffers for EVERY packet to maintain time-series integrity
            # effectively (assuming we plot against index, not timestamp).
            # If we want to support non-uniform sampling, we'd need X and Y buffers.
            # Here we assume simple scrolling Y-buffers.

            # Iterate over all registered buffer sources
            for source, buffer in self._buffers.items():
                val = packet.get(source)
                if val is None:
                    buffer.append(float('nan'))
                else:
                    # Ensure it's numeric
                    try:
                        buffer.append(float(val))
                    except (ValueError, TypeError):
                        buffer.append(float('nan'))

            # Mark that we have new data
            if packet:
                 # In a high-speed loop, we just need to know we processed at least one packet
                 # effectively, but since we updated buffers inside the loop, we are good.
                 pass

        # 2. Redraw Plots
        # We only redraw the curves that exist
        for source, curve in self._curves.items():
            buf = self._buffers.get(source)
            if buf:
                # Convert deque to array for faster plotting
                curve.setData(np.array(buf))

        # 3. Update Text Fields (Only need latest value)
        for source, items in self._text_field_map.items():
            if source in latest_values:
                val = latest_values[source]
                for (text_item, fmt_str) in items:
                    try:
                        formatted = fmt_str.format(val)
                    except Exception:
                        formatted = f"Error: {val}"
                    text_item.setText(formatted)

def run_plotter(plotter_queue: multiprocessing.Queue) -> None:
    """Entry point for the plotter process."""
    # Re-import constants here to avoid pickling issues if passed directly,
    # or just to ensure we have the config.
    from constants import DASHBOARD_CONFIG

    app = QApplication(sys.argv)

    # Enable High DPI display
    if hasattr(pg, 'setConfigOptions'):
        pg.setConfigOptions(antialias=True)

    window = CrazyfliePlotter(plotter_queue, DASHBOARD_CONFIG)
    window.show()

    LOGGER.info("Plotter process started")
    sys.exit(app.exec())

"""Visualization of the simulated Crazyflie."""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
from vispy import app, scene
from vispy.scene import visuals

from . import constants
from .world import World

LOGGER = logging.getLogger(__name__)

_DRONE_X_RADIUS = 0.12
_DRONE_AXIS_LEN = 0.35
_ANCHOR_RADIUS = 0.08


class SimulationViewer:
    """VisPy-based viewer with simple keyboard controls."""

    def __init__(
        self,
        world: World,
        stop_event,
        stop_callback: Callable[[], None],
        pause_toggle: Callable[[], None],
        reset_callback: Callable[[], None],
    ) -> None:
        self._world = world
        self._stop_event = stop_event
        self._stop_callback = stop_callback
        self._pause_toggle = pause_toggle
        self._reset_callback = reset_callback

        self._canvas = scene.SceneCanvas(keys="interactive", size=(800, 600), show=True, title="Crazyflie Simulation", bgcolor="white")
        self._view = self._canvas.central_widget.add_view()
        self._view.camera = "turntable"
        self._view.camera.distance = 8
        self._view.bgcolor = "white"

        grid = visuals.XYZAxis(parent=self._view.scene)
        grid.transform = scene.transforms.STTransform(scale=(1.0, 1.0, 1.0))

        self._anchors = np.array([anchor for anchor in world.anchors], dtype=float)
        self._anchor_spheres = []
        self._create_anchor_visuals()

        self._drone_x_template, self._drone_x_connect = self._build_drone_x_template()
        self._drone_axes_template, self._drone_axes_connect, self._drone_axes_colors = self._build_axis_template()
        self._drone_x = visuals.Line(parent=self._view.scene, color="#333333", width=3)
        self._drone_axes = visuals.Line(parent=self._view.scene, width=3)

        self._timer = app.Timer(interval=1.0 / constants.VISUAL_RATE_HZ, connect=self._on_timer, start=True)
        self._canvas.events.key_press.connect(self._on_key_press)
        self._canvas.events.close.connect(lambda event: self._request_stop())

    def run(self) -> None:
        """Start the VisPy application loop."""
        app.run()

    # ------------------------------------------------------------------ visuals --

    def _create_anchor_visuals(self) -> None:
        for anchor in self._anchors:
            sphere = visuals.Sphere(
                parent=self._view.scene,
                radius=_ANCHOR_RADIUS,
                color="royalblue",
                method="latitude",
            )
            sphere.transform = scene.transforms.STTransform(translate=anchor)
            self._anchor_spheres.append(sphere)

    @staticmethod
    def _build_drone_x_template() -> tuple[np.ndarray, np.ndarray]:
        r = _DRONE_X_RADIUS
        points = np.array(
            [
                [-r, -r, 0.0],
                [r, r, 0.0],
                [-r, r, 0.0],
                [r, -r, 0.0],
            ],
            dtype=float,
        )
        connect = np.array([[0, 1], [2, 3]], dtype=int)
        return points, connect

    @staticmethod
    def _build_axis_template() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        l = _DRONE_AXIS_LEN
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [l, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, l, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, l],
            ],
            dtype=float,
        )
        connect = np.array([[0, 1], [2, 3], [4, 5]], dtype=int)
        colors = np.array(
            [
                [1.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
            ],
            dtype=float,
        )
        return points, connect, colors

    def _update_drone_visuals(self, position: np.ndarray) -> None:
        pos = position.reshape(1, 3)
        x_points = self._drone_x_template + pos
        axes_points = self._drone_axes_template + pos
        self._drone_x.set_data(pos=x_points, connect=self._drone_x_connect, color="#333333")
        self._drone_axes.set_data(pos=axes_points, connect=self._drone_axes_connect, color=self._drone_axes_colors)

    # --------------------------------------------------------------------- events

    def _on_timer(self, event) -> None:  # noqa: ANN001
        if self._stop_event.is_set():
            app.quit()
            return

        position, _velocity = self._world.get_state()
        self._update_drone_visuals(position)

    def _on_key_press(self, event) -> None:  # noqa: ANN001
        name = event.key.name if event.key else ""
        key = name.lower()
        if key == " " or key == "space":
            self._pause_toggle()
        elif key == "r":
            self._reset_callback()
        elif key in ("q", "escape"):
            self._request_stop()

    def _request_stop(self) -> None:
        self._stop_callback()
        app.quit()

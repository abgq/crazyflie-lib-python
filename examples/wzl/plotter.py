"""Generic plotting utility."""

from __future__ import annotations

from typing import List, Dict, Any, Optional
import matplotlib.pyplot as plt

class DataPlotter:
    """Generates figures based on recorded data and a configuration recipe."""

    def __init__(self, data: List[Dict[str, Any]]) -> None:
        """
        Args:
            data: List of flat dictionaries (output from DataRecorder.get_data())
        """
        self.data = data
        self.timestamps = [row.get("time", 0.0) for row in data]

    def plot(self, config: List[Dict[str, Any]]) -> None:
        """
        Generate subplots based on the provided config.

        Config Format:
        [
            {
                "title": "UWB Debugging",
                "ylabel": "Distance (counts)",
                "vars": ["dw1k.rangingCounter", "dw1k.rangingCounter_raw"]
            },
            ...
        ]
        """
        if not self.data:
            print("[Plotter] No data to plot.")
            return

        num_plots = len(config)
        if num_plots == 0:
            return

        # Create a figure with stacked subplots
        fig, axes = plt.subplots(num_plots, 1, sharex=True, figsize=(10, 3 * num_plots))

        # Ensure axes is iterable even if there is only 1 plot
        if num_plots == 1:
            axes = [axes]

        for i, panel in enumerate(config):
            ax = axes[i]
            title = panel.get("title", f"Plot {i}")
            ylabel = panel.get("ylabel", "Value")
            variables = panel.get("vars", [])

            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.grid(True, linestyle='--', alpha=0.6)

            for var_name in variables:
                # Extract data for this variable, handling missing keys (None) gracefully
                values = [row.get(var_name) for row in self.data]

                # Filter out Nones for plotting continuity (optional, or let matplotlib handle gaps)
                # Here we just plot them; matplotlib usually skips Nones.

                # Style logic: If it ends in "_raw", make it faint and dashed
                if var_name.endswith("_raw"):
                    ax.plot(self.timestamps, values, label=var_name, linestyle='--', alpha=0.5, linewidth=1)
                else:
                    ax.plot(self.timestamps, values, label=var_name, linewidth=1.5)

            ax.legend(loc="upper right", fontsize='small')

        axes[-1].set_xlabel("Time (s)")
        plt.tight_layout()
        plt.show()

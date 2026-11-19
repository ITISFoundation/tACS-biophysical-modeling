"""Amplitude plots use the same marker areas and Figure 4b line weight."""
import unittest
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import numpy as np
import pandas as pd

from src.plotting import marker_sizes, plot_metric_vs_amplitude, plot_rose_grid, polar_scatter


class AmplitudeMarkerStyleTests(unittest.TestCase):
    def test_rose_grid_keeps_translucent_histograms_and_opaque_vectors(self):
        phases = pd.DataFrame({"phase_deg": [0, 60, 120, 180, 240, 300]})
        for style, alpha in (("bar", 0.65), ("line", 0.9)):
            with self.subTest(style=style):
                fig = plot_rose_grid([[phases]], colors="blue", style=style)
                try:
                    fig.canvas.draw()
                    ax = fig.axes[0]
                    if style == "bar":
                        self.assertTrue(ax.patches)
                        for patch in ax.patches:
                            self.assertEqual(patch.get_facecolor()[3], alpha)
                    else:
                        self.assertEqual(ax.lines[0].get_alpha(), alpha)
                    vector = ax.lines[-1]
                    self.assertEqual(
                        to_rgba(vector.get_color(), vector.get_alpha())[3], 1
                    )
                finally:
                    plt.close(fig)

    def test_notebook_palette_is_fully_opaque(self):
        notebook = json.loads(
            (Path(__file__).resolve().parents[1] / "analysis.ipynb").read_text()
        )
        palette_source = next(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if "FREQ_COLORS =" in "".join(cell.get("source", []))
        )
        namespace = {"matplotlib": matplotlib, "pd": pd}
        exec(compile(palette_source, "notebook palette", "exec"), namespace)
        for palette_name in ("FREQ_COLORS", "CELL_COLORS"):
            for colors in namespace[palette_name].values():
                for color in colors.values():
                    self.assertEqual(to_rgba(color)[3], 1)

    def test_markers_lines_and_error_bars_are_opaque(self):
        rows = pd.DataFrame({
            "stim_amp": [0.5, 1.0], "plv": [0.1, 0.2],
            "plv_std": [0.01, 0.02],
        })
        for polar in (False, True):
            for color in ("#cf5656d3", "#9367bdb0", "#56B4E9"):
                with self.subTest(polar=polar, color=color):
                    fig, ax = plt.subplots(
                        subplot_kw={"projection": "polar"} if polar else {}
                    )
                    try:
                        if polar:
                            polar_scatter(
                                ax, rows, 0.05, color, z_col="plv",
                                means=True, line=True,
                            )
                        else:
                            plot_metric_vs_amplitude(ax, rows, 0.05, color)
                        fig.canvas.draw()
                        points = ax.collections[0]
                        expected = [to_rgba(color, alpha=1)]
                        np.testing.assert_allclose(points.get_facecolors(), expected)
                        np.testing.assert_allclose(points.get_edgecolors(), expected)
                        for line in ax.lines:
                            self.assertEqual(line.get_alpha(), 1)
                        for collection in ax.collections[1:]:
                            np.testing.assert_allclose(
                                collection.get_edgecolors()[:, 3], 1
                            )
                    finally:
                        plt.close(fig)

    def test_marker_sizes_match_figure_4b(self):
        amplitudes = np.array([0, 0.25, 0.5, 1, 2, 4])
        figure_4b_areas = np.array([10, 14, 18, 25, 35, 50])
        expected = 0.25 * 10 * (figure_4b_areas / 10) ** 2
        np.testing.assert_allclose(marker_sizes(amplitudes), expected)

    def test_cartesian_and_polar_plots_share_size_and_line_weight(self):
        rows = pd.DataFrame({"stim_amp": [0.5, 1.0], "plv": [0.1, 0.2]})
        expected_sizes = marker_sizes(np.array([0, 0.5, 1.0]))

        for polar in (False, True):
            with self.subTest(polar=polar):
                fig, ax = plt.subplots(
                    subplot_kw={"projection": "polar"} if polar else {}
                )
                try:
                    if polar:
                        polar_scatter(
                            ax, rows, 0.05, "blue", z_col="plv",
                            means=True, line=True, edgecolors="face",
                        )
                    else:
                        plot_metric_vs_amplitude(ax, rows, 0.05, "blue")

                    fig.canvas.draw()
                    np.testing.assert_allclose(
                        ax.collections[0].get_sizes(), expected_sizes
                    )
                    self.assertTrue(ax.lines)
                    self.assertAlmostEqual(ax.lines[-1].get_linewidth(), 0.5)
                finally:
                    plt.close(fig)


if __name__ == "__main__":
    unittest.main()

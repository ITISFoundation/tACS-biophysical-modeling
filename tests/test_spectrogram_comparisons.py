"""Numerical and selection checks for the simplified Morlet workflow."""
import csv
import json
from pathlib import Path
from contextlib import contextmanager
import shutil
import unittest
import uuid

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from src.spectrograms import morlet_kernel
from src.spectrogram_comparisons import morlet_power, load_runs, plot_grid


@contextmanager
def fixture_directory():
    # Python 3.13's Windows TemporaryDirectory ACL can exclude the sandbox
    # identity. A workspace directory with inherited permissions is sufficient.
    parent = Path(__file__).resolve().parents[1] / "output"
    parent.mkdir(exist_ok=True)
    root = parent / f"spectrogram-test-{uuid.uuid4().hex}"
    root.mkdir()
    try:
        yield root
    finally:
        if root.resolve().parent != parent.resolve():
            raise RuntimeError("Fixture cleanup escaped its output directory")
        shutil.rmtree(root)


class MorletTests(unittest.TestCase):
    def test_tone_matches_direct_convolution_and_peaks_at_ten_hz(self):
        t = (np.arange(2000) + 0.5) * 0.005
        rate = np.cos(2 * np.pi * 10 * t)
        frequencies = np.arange(1, 20.01, 0.125)
        power = morlet_power(rate, frequencies)
        i = np.flatnonzero(frequencies == 10)[0]
        wavelet = morlet_kernel(10)
        half = len(wavelet) // 2
        expected = np.convolve(np.pad(rate, (half, half), mode="reflect"), wavelet.conj()[::-1], mode="valid")
        np.testing.assert_allclose(power[i], abs(expected)**2, atol=1e-12)
        peak = frequencies[np.argmax(power[:, (t > 3) & (t < 7)].mean(axis=1))]
        self.assertAlmostEqual(peak, 10, delta=0.75)

    def test_three_period_windows_shrink_and_dc_is_rejected(self):
        frequencies = np.arange(1, 20.01, 0.125)
        for periods in (3, 4, 6):
            duration = np.array([(len(morlet_kernel(f, window_periods=periods))-1)*0.005 for f in frequencies])
            self.assertTrue(np.all(duration * frequencies >= periods - 1e-12))
            self.assertTrue(np.all(np.diff(duration) <= 0))
        self.assertLess(morlet_power(np.ones(500), frequencies).max(), 1e-25)
        with self.assertRaises(ValueError):
            morlet_power(np.zeros(500), frequencies, periods=2)
        with self.assertRaises(ValueError):
            morlet_power(np.zeros(500), [10, 100])

    def test_plot_shows_every_coefficient_even_for_short_recordings(self):
        time = (np.arange(500) + 0.5)*0.005
        power = morlet_power(np.sin(2*np.pi*10*time), [1, 5, 10, 20])
        run = dict(time=time, frequency=np.array([1, 5, 10, 20]), power_db=10*np.log10(np.maximum(power, 1e-20)),
                   name="example", state="low", seed=843, amplitude_vm=1, delay_s=1.5,
                   stop_s=2.5, observed_stop=2.5, stimuli=[])
        fig = plot_grid([[run, None]], (-40, 10), "test")
        mesh = fig.axes[0].collections[0]
        self.assertFalse(np.ma.getmaskarray(mesh.get_array()).any())
        np.testing.assert_allclose(mesh.get_array(), run["power_db"])
        self.assertTrue(any("No recording" in t.get_text() for t in fig.axes[1].texts))
        fig.canvas.draw()
        plt.close(fig)

    def test_excludes_circuit_identities_but_keeps_hex0_node_set(self):
        with fixture_directory() as directory:
            root = Path(directory)
            (root / "data").mkdir()
            nodes = root / "nodes.json"
            nodes.write_text(json.dumps({"hex0": {"population": "S1nonbarrel_neurons", "node_id": [0, 1, 2, 3]}}))
            with (root / "data.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["sim_id", "source_file"])
                writer.writeheader()
                writer.writerow(dict(sim_id="O1_10Hz", source_file="data/alias_run/config.json"))
            for name in ("sim10Hz1Vm_seed2", "hex0_10Hz", "o1_10Hz", "alias_run"):
                folder = root / "data" / name
                (folder / "reporting").mkdir(parents=True)
                config = dict(run=dict(tstop=2500, random_seed=84), node_set="hex0",
                              output=dict(output_dir="reporting", spikes_file="spikes.h5"),
                              inputs={"stim": dict(input_type="extracellular_stimulation", delay=1100,
                                                   duration=1400, fields=[dict(Ex=1, frequency=10, phase=1.570796)])})
                (folder / "config.json").write_text(json.dumps(config))
                with h5py.File(folder / "reporting" / "spikes.h5", "w") as file:
                    group = file.create_group("spikes/S1nonbarrel_neurons")
                    group.create_dataset("timestamps", data=[100, 2400]).attrs["units"] = "ms"
                    group.create_dataset("node_ids", data=[0, 0])
            records, audit = load_runs(root, nodes, [1, 5, 10, 20])
            self.assertEqual([r["name"] for r in records], ["sim10Hz1Vm_seed2"])
            self.assertEqual(sum(r["status"] == "excluded" for r in audit), 3)
            self.assertFalse(records[0]["standard"])
            self.assertEqual(records[0]["seed"], 84)
            self.assertAlmostEqual(records[0]["rate"].sum() * 0.005, 2 / 4)


if __name__ == "__main__":
    unittest.main()

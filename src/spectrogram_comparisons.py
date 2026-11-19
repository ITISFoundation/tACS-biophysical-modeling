"""Simple Morlet power and comparison grids used by analysis.ipynb."""
import csv
import json
import re
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import fftconvolve

from .spectrograms import load_record, morlet_kernel


def morlet_power(rate, frequencies, dt=0.005, periods=3.0):
    """Convolve with one Morlet per frequency; return |coefficient|**2.

    Full support >= periods/f seconds; retain every sample, reflect outer edges.
    periods counts the full support, not the conventional Morlet n_cycles.
    """
    rate, frequencies = np.asarray(rate, float), np.asarray(frequencies, float)
    if rate.ndim != 1 or rate.size < 2 or not np.isfinite(rate).all():
        raise ValueError("rate must be a finite 1-D array with at least two samples")
    if not np.isfinite(periods) or periods < 3:
        raise ValueError("periods must be at least 3")
    if (frequencies.ndim != 1 or frequencies.size < 2
            or not np.isfinite(frequencies).all() or np.any(np.diff(frequencies) <= 0)):
        raise ValueError("frequencies must be finite and strictly increasing")
    power = []
    for frequency in frequencies:
        wavelet = morlet_kernel(frequency, dt, periods)
        half = len(wavelet) // 2
        padded = np.pad(rate, (half, half), mode="reflect")
        coefficient = fftconvolve(padded, wavelet.conj()[::-1], mode="valid")
        power.append(np.abs(coefficient) ** 2)
    return np.asarray(power)


def load_runs(root, node_sets_path, frequencies, periods=3.0):
    """Load every eligible local run once; return records and a selection audit."""
    root = Path(root)
    node_sets = json.loads(Path(node_sets_path).read_text())
    aliases = {}
    with (root / "data.csv").open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            folder = row["source_file"].replace("\\", "/").split("/")[-2]
            aliases.setdefault(folder, []).append(row["sim_id"])
    runs, audit = [], []
    for folder in sorted((root / "data").iterdir()):
        if not folder.is_dir():
            continue
        identity = " ".join([folder.name, *aliases.get(folder.name, [])])
        if re.search(r"(?:^|[_\W])(?:hex0|O1)(?:[_\W]|$)", identity, re.I):
            audit.append(dict(name=folder.name, status="excluded", reason="hex0/O1 comparison run"))
            continue
        try:
            configs = list(folder.glob("*.json"))
            if len(configs) != 1:
                raise ValueError(f"Expected one simulation config; found {len(configs)}")
            config = json.loads(configs[0].read_text())
            fields = [(s, f) for s in config.get("inputs", {}).values()
                      if s.get("input_type") == "extracellular_stimulation"
                      for f in s.get("fields", [])
                      if any(float(f.get(k, 0)) != 0 for k in ("Ex", "Ey", "Ez"))]
            if len(fields) > 1:
                raise ValueError("Multiple active fields: select a protocol explicitly")
            run = load_record(configs[0], node_sets, window_periods=periods, compute_spectrum=False)
            run.pop("overlap", None)  # The dense transform retains every time sample.
            run.update(state="high" if "highcorrelation" in identity.lower() else "low",
                       seed=config["run"].get("random_seed"), stimulus_hz=0.0,
                       amplitude_vm=0.0, delay_s=0.0, duration_s=0.0, standard=False)
            if fields:
                stim, field = fields[0]
                run.update(stimulus_hz=float(field["frequency"]),
                           amplitude_vm=round(float(np.linalg.norm([field.get(k, 0) for k in ("Ex", "Ey", "Ez")])), 5),
                           delay_s=float(stim["delay"]) / 1000, duration_s=float(stim["duration"]) / 1000)
                run["standard"] = bool(
                    not re.search("replay|disconnect|delay|long|unconverged|invert", identity, re.I)
                    and np.isclose(run["delay_s"], 1.5) and np.isclose(run["duration_s"], 1)
                    and np.isclose(run["stop_s"], 2.5)
                    and np.isclose(field.get("phase", 0), np.pi / 2, atol=1e-5))
            run["frequency"] = np.asarray(frequencies)
            run["time"] = run["rate_time"]
            run["power"] = morlet_power(run["rate"], frequencies, run["bin_s"], periods)
            run["power_db"] = 10 * np.log10(np.maximum(run["power"], 1e-20))
            runs.append(run)
            audit.append(dict(name=folder.name, status="loaded", reason="; ".join(run["warnings"])))
        except (OSError, KeyError, ValueError) as exc:
            audit.append(dict(name=folder.name, status="skipped", reason=str(exc)))
    if not runs:
        raise ValueError(f"No eligible recordings loaded: {audit}")
    return runs, audit


def plot_grid(rows, limits, title, *, row_labels=None, column_labels=None, align_onset=True):
    """Plot a rectangular list of records (None = unavailable) on shared axes."""
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("rows must be a nonempty rectangular list")
    present = [r for row in rows for r in row if r is not None]
    if not present:
        raise ValueError("No recordings match this selection")
    offsets = [r["delay_s"] if align_onset else 0 for r in present]
    xmin = min(-offset for offset in offsets)
    xmax = max(r["stop_s"] - offset for r, offset in zip(present, offsets))
    fig, axes = plt.subplots(len(rows), len(rows[0]), squeeze=False,
                             figsize=(4.8 * len(rows[0]), 3.2 * len(rows)), layout="constrained")
    for i, row in enumerate(rows):
        for j, run in enumerate(row):
            ax = axes[i, j]
            heading = column_labels[j] if column_labels else ""
            if run is None:
                ax.set_axis_off()
                ax.text(0.5, 0.5, heading + "\nNo recording", ha="center", va="center", transform=ax.transAxes)
                continue
            offset = run["delay_s"] if align_onset else 0
            mesh = ax.pcolormesh(run["time"] - offset, run["frequency"], run["power_db"],
                                 shading="gouraud", cmap="magma", vmin=limits[0], vmax=limits[1], rasterized=True)
            for stim in run["stimuli"]:
                ax.axvline(stim["start"] - offset, color="white", lw=0.7, ls="--")
                ax.axvline(stim["end"] - offset, color="white", lw=0.7, ls="--")
                ax.hlines(stim["frequencies"], stim["start"] - offset, stim["end"] - offset,
                          color="cyan", lw=0.7, ls=":")
            if run["observed_stop"] < run["stop_s"]:
                ax.axvspan(run["observed_stop"] - offset, run["stop_s"] - offset, color="lightgray")
            label = (row_labels[i] + "\n") if row_labels else ""
            ax.set(xlim=(xmin, xmax), ylim=(run["frequency"][0], run["frequency"][-1]),
                   xlabel="Time from onset (s)" if align_onset else "Simulation time (s)",
                   ylabel=label + "Analyzed frequency (Hz)")
            name = textwrap.fill(run["name"], 38)
            ax.set_title(f"{heading}\n{name}\n{run['state']} state | {run['amplitude_vm']:g} V/m | seed {run['seed']}", fontsize=9)
    fig.suptitle(title, fontsize=14)
    fig.colorbar(mesh, ax=axes.ravel().tolist(), label="Morlet power (dB re 1 (spikes/s/neuron)²)", shrink=0.85, extend="both")
    fig.supxlabel("Morlet transform with reflected recording edges. Dashed: stimulation boundaries; dotted: applied frequency.", fontsize=9)
    return fig

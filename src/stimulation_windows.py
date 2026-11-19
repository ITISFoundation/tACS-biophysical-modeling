"""Compare population PSD and spike PLV in two halves and the full stimulus."""
from pathlib import Path
import csv
import json

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import welch

from .spectrograms import POPULATION, resolve_ids


WINDOWS = ("First half", "Second half", "Full period")
COLORS = ("#2478a5", "#db7933", "#393d47")
PAIRS = ((1, 0), (0, 2), (1, 2))


def compare_windows(spike_times_ms, n_neurons, start_ms, stop_ms, frequency_hz=10.0):
    """Use disjoint [start, midpoint), [midpoint, stop) and their exact union.

    PSD: 5 ms bins, independent 10 ms Gaussian smoothing within each window,
    periodic Hann, constant detrending, density scaling, 50% Welch overlap.
    A shared half-duration segment controls spectral smoothing; a second estimate
    uses each entire window. Zero padding provides a common 0.2 Hz display grid
    for the recording, without increasing spectral resolution.
    PLV: magnitude of the mean spike-phase vector, pooled over all reported cells.
    The phase reference is stimulation onset in every window; constant stimulus
    phase/orientation offsets do not affect the reported magnitudes.
    """
    spikes = np.asarray(spike_times_ms, dtype=float)
    if (spikes.ndim != 1 or not np.isfinite(spikes).all()
            or not np.isfinite([start_ms, stop_ms, frequency_hz, n_neurons]).all()
            or stop_ms <= start_ms or n_neurons <= 0 or int(n_neurons) != n_neurons
            or frequency_hz <= 0):
        raise ValueError("Need finite spike times, positive cell count/frequency and ordered bounds")
    bin_ms = 5.0
    n_bins = round((stop_ms - start_ms) / bin_ms)
    if n_bins < 8 or n_bins % 2 or not np.isclose(n_bins * bin_ms, stop_ms - start_ms):
        raise ValueError("Duration must contain an even number of 5 ms bins (at least 8)")
    half = n_bins // 2
    nfft = max(1000, n_bins)
    midpoint = (start_ms + stop_ms) / 2
    bounds = ((start_ms, midpoint), (midpoint, stop_ms), (start_ms, stop_ms))
    records, common_psds, native_psds, rates = [], [], [], []
    for name, (lo, hi) in zip(WINDOWS, bounds):
        selected = spikes[(spikes >= lo) & (spikes < hi)]
        count = len(selected)
        edges = lo + np.arange(round((hi - lo) / bin_ms) + 1) * bin_ms
        rate = np.histogram(selected, bins=edges)[0] / (bin_ms * 1e-3 * n_neurons)
        smooth = gaussian_filter1d(rate, sigma=2, mode="reflect")
        options = dict(fs=200, window="hann", nfft=nfft, detrend="constant", scaling="density")
        f, common = welch(smooth, nperseg=half, noverlap=half // 2, **options)
        _, native = welch(smooth, nperseg=len(smooth), noverlap=0, **options)
        phases = 2 * np.pi * frequency_hz * (selected - start_ms) * 1e-3
        z = np.mean(np.exp(1j * phases)) if count else complex(np.nan, np.nan)
        records.append(dict(window=name, start_ms=lo, stop_ms=hi, duration_ms=hi-lo,
                            n_spikes=count, mean_rate_hz=count / ((hi-lo)*1e-3*n_neurons),
                            plv=float(abs(z)), plv_real=float(z.real), plv_imag=float(z.imag),
                            welch_segments=1 + (len(rate)-half)//(half-half//2),
                            native_bin_spacing_hz=200/len(rate)))
        common_psds.append(common)
        native_psds.append(native)
        rates.append(rate)
    return dict(records=records, frequency_hz=f, common_psd=np.array(common_psds),
                native_psd=np.array(native_psds), rates=rates, plv_frequency_hz=frequency_hz,
                bin_ms=bin_ms, nperseg=half, nfft=nfft, n_neurons=int(n_neurons))


def find_long_config(root):
    """Resolve only the requested long run; never substitute the short recording."""
    root = Path(root)
    with (root / "data.csv").open(newline="", encoding="utf-8-sig") as handle:
        rows = [r for r in csv.DictReader(handle) if r["sim_id"] == "10Hz1Vm_long"]
    if len(rows) > 1:
        raise ValueError("Duplicate 10Hz1Vm_long rows in data.csv")
    if rows:
        path = root / rows[0]["source_file"]
        if path.is_file():
            return path
    candidates = [root / "data" / folder / filename
                  for folder in ("10Hz1Vm_long", "sim10Hz1Vm_long")
                  for filename in ("sim_with_stim.json", "simulation_config.json")]
    found = [p for p in candidates if p.is_file()]
    if len(found) == 1:
        return found[0]
    raise FileNotFoundError(
        "10Hz1Vm_long is unavailable or ambiguous. Set LONG_CONFIG to its simulation JSON; "
        "the existing 10Hz1Vm recording is not a substitute.")


def load_long_run(config_path, node_sets_path):
    """Read the report directly, using configured timing and all simulated cells."""
    config_path = Path(config_path)
    config = json.loads(config_path.read_text())
    stimuli = [s for s in config.get("inputs", {}).values()
               if s.get("input_type") == "extracellular_stimulation"
               and any(any(float(f.get(k, 0)) != 0 for k in ("Ex", "Ey", "Ez"))
                       for f in s.get("fields", []))]
    if len(stimuli) != 1:
        raise ValueError("Expected exactly one active extracellular stimulus")
    stim = stimuli[0]
    fields = [f for f in stim["fields"] if any(float(f.get(k, 0)) != 0 for k in ("Ex", "Ey", "Ez"))]
    if any(not np.isclose(float(f["frequency"]), 10) for f in fields):
        raise ValueError("Expected 10 Hz stimulation")
    start, duration = float(stim["delay"]), float(stim["duration"])
    stop = start + duration
    if float(config["run"]["tstop"]) < stop:
        raise ValueError("Need the complete configured stimulation period")
    ids = resolve_ids(json.loads(Path(node_sets_path).read_text()), config["node_set"])
    spike_file = config_path.parent / config["output"]["output_dir"] / config["output"]["spikes_file"]
    with h5py.File(spike_file, "r") as handle:
        group = handle["spikes"][POPULATION]
        units = group["timestamps"].attrs.get("units", "ms")
        if isinstance(units, bytes):
            units = units.decode()
        if units not in ("ms", "s"):
            raise ValueError(f"Unsupported timestamp units: {units}")
        times = group["timestamps"][:] * (1000 if units == "s" else 1)
        if times.size != group["node_ids"].size or not np.isfinite(times).all():
            raise ValueError("Invalid spike report")
    if times.size == 0 or times.max() < stop - 250:
        raise ValueError("Spike report does not verify the last 250 ms of stimulation; check completeness")
    if np.any((times < 0) | (times > float(config["run"]["tstop"]))):
        raise ValueError("Spikes extend outside the configured recording")
    result = compare_windows(times, len(ids), start, stop)
    result["source"] = dict(config=str(config_path.resolve()), spike_file=str(spike_file.resolve()),
                            node_sets=str(Path(node_sets_path).resolve()))
    return result


def plot_comparison(result):
    """Return PSD and PLV figures, including all three signed pairwise differences."""
    import matplotlib.pyplot as plt

    f = result["frequency_hz"]
    half_ms, full_ms = result["records"][0]["duration_ms"], result["records"][2]["duration_ms"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
    for i, (label, color) in enumerate(zip(WINDOWS, COLORS)):
        axes[0].plot(f, result["common_psd"][i], color=color, label=label)
        axes[1].plot(f, result["native_psd"][i], color=color, label=label)
    for (a, b), color in zip(PAIRS, COLORS):
        axes[2].plot(f, result["common_psd"][a] - result["common_psd"][b],
                     color=color, label=f"{WINDOWS[a]} − {WINDOWS[b]}")
    for ax in axes:
        ax.set(xlim=(1, 20), xlabel="Frequency (Hz)")
        ax.axvline(2, color="0.65", ls=":", lw=1)
        ax.axvline(10, color="0.65", ls=":", lw=1)
        ax.legend(frameon=False, fontsize=8)
        ax.grid(alpha=0.15)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    axes[0].set(title=f"Matched Welch: {half_ms/1000:g} s segments", ylabel="PSD [(spikes/s/neuron)²/Hz]", ylim=(0, None))
    axes[1].set(title="Each entire window: Hann periodogram", ylabel="PSD [(spikes/s/neuron)²/Hz]", ylim=(0, None))
    axes[2].set(title="Differences: matched Welch", ylabel="ΔPSD [(spikes/s/neuron)²/Hz]")
    axes[2].axhline(0, color="0.4", lw=0.8)
    fig.suptitle(f"10Hz1Vm_long · first {half_ms:g} ms, second {half_ms:g} ms, full {full_ms:g} ms")

    plv_fig, ax = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    values = np.array([r["plv"] for r in result["records"]])
    ax[0].bar(WINDOWS, values, color=COLORS)
    for i, value in enumerate(values):
        if np.isfinite(value):
            ax[0].annotate(f"{value:.5f}", (i, value), xytext=(0, 4), textcoords="offset points", ha="center")
    ax[0].set(ylabel="Population spike PLV", title=f"PLV at {result['plv_frequency_hz']:g} Hz")
    ax[0].set_ylim(0, min(1.05, max(0.05, np.nanmax(values)*1.25)) if np.isfinite(values).any() else 1)
    differences = [values[a]-values[b] for a, b in PAIRS]
    ax[1].bar([f"{WINDOWS[a]}\n− {WINDOWS[b]}" for a, b in PAIRS], differences, color=COLORS)
    for i, value in enumerate(differences):
        if np.isfinite(value):
            ax[1].annotate(f"{value:+.5f}", (i, value), xytext=(0, 4 if value >= 0 else -4),
                           textcoords="offset points", ha="center", va="bottom" if value >= 0 else "top")
    ax[1].axhline(0, color="0.4", lw=0.8)
    ax[1].set(ylabel="ΔPLV (absolute difference)", title="Signed differences")
    ax[1].margins(y=0.25)
    return fig, plv_fig


def export_comparison(result, output_dir):
    """Save figures and numeric results; return window and difference tables."""
    import pandas as pd
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(result["records"])
    for freq in (2, 10):
        index = np.flatnonzero(np.isclose(result["frequency_hz"], freq))
        if len(index) != 1:
            raise ValueError(f"PSD grid must contain {freq} Hz")
        summary[f"matched_psd_{freq}hz"] = result["common_psd"][:, index[0]]
    columns = ["plv", "matched_psd_2hz", "matched_psd_10hz"]
    differences = pd.DataFrame([
        dict(comparison=f"{WINDOWS[a]} minus {WINDOWS[b]}",
             **{f"delta_{c}": summary.iloc[a][c]-summary.iloc[b][c] for c in columns})
        for a, b in PAIRS])
    summary.to_csv(output_dir / "window_summary.csv", index=False)
    differences.to_csv(output_dir / "window_differences.csv", index=False)
    np.savez_compressed(output_dir / "spectra.npz", frequency_hz=result["frequency_hz"],
                        common_psd=result["common_psd"], native_psd=result["native_psd"],
                        window_names=np.array(WINDOWS), full_rate=result["rates"][2])
    metadata = {k: result[k] for k in ("bin_ms", "nperseg", "nfft", "n_neurons", "plv_frequency_hz")}
    metadata.update(source=result.get("source"), sigma_ms=10, smoothing_mode="reflect",
                    psd_window="hann", detrend="constant", scaling="density", noverlap=result["nperseg"]//2,
                    bounds="[start, stop)", records=result["records"])
    (output_dir / "manifest.json").write_text(json.dumps(metadata, indent=2))
    figures = plot_comparison(result)
    for name, fig in zip(("psd_windows", "plv_windows"), figures):
        for extension in ("png", "svg"):
            fig.savefig(output_dir / f"{name}.{extension}", dpi=180, bbox_inches="tight")
    return summary, differences, figures

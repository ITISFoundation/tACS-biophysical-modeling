"""Full-recording Morlet spectrograms of population spike rate.

5 ms spike bins; Morlet windows spanning at least 3 periods; at most 10% overlap between sampled
wavelet supports. Wavelet duration and time spacing decrease with frequency.
The transform runs continuously through stimulation onset and offset.
Run with ``python -m src.spectrograms --node-sets PATH``.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import re
import textwrap
from pathlib import Path

import h5py
import matplotlib
if __name__ == "__main__":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

POPULATION = "S1nonbarrel_neurons"


def resolve_ids(node_sets, name):
    """Resolve explicit SONATA node IDs and unions (including silent neurons)."""
    spec = node_sets[name]
    if isinstance(spec, list):
        return np.unique(np.concatenate([resolve_ids(node_sets, n) for n in spec]))
    if spec.get("population") != POPULATION or "node_id" not in spec:
        raise ValueError(f"Node set {name!r} needs explicit IDs for {POPULATION}")
    return np.unique(np.asarray(spec["node_id"], dtype=np.int64))


def morlet_kernel(frequency, bin_s=0.005, window_periods=3.0):
    """Unit-energy, zero-mean complex Morlet, truncated at +/-3 Gaussian sigma.

    The full support spans at least window_periods / frequency seconds.
    sigma_t = window_periods / (6*frequency) places the requested window at
    +/-3 sigma. Round each half up to a whole sample so the support is never
    shorter than requested. This is full support, not the Gaussian FWHM or
    the conventional Morlet n_cycles parameter (equivalent to pi*periods/3).
    The small zero-mean correction also accounts for truncation.
    """
    if (not np.isfinite([frequency, bin_s, window_periods]).all()
            or bin_s <= 0 or not 0 < frequency < 0.5 / bin_s or window_periods < 1):
        raise ValueError("Need positive bin size, frequency below Nyquist and window_periods >= 1")
    sigma = window_periods / (6 * frequency)
    half = int(np.ceil(window_periods / (2 * frequency * bin_s)))
    time = np.arange(-half, half + 1) * bin_s
    envelope = np.exp(-0.5 * (time / sigma) ** 2)
    carrier = np.exp(2j * np.pi * frequency * time)
    wavelet = (carrier - np.sum(carrier * envelope) / envelope.sum()) * envelope
    return wavelet / np.linalg.norm(wavelet)


def rate_spectrogram(rate, bin_s=0.005, window_periods=3.0, overlap=0.4, frequencies=None):
    """Morlet power across one continuous recording, with sparse time sampling.

    Return frequencies and one time/power array per frequency. Each frequency
    uses a hop of ceil((1-overlap)*kernel_length) samples, so adjacent sampled
    supports overlap by at most ``overlap`` (10% by default). Higher frequencies
    have shorter supports and finer time sampling. No stimulation boundary is
    used in the calculation. Only the recording's outer edges are reflected.

    Power is abs(coefficient)**2 for discrete unit-energy wavelets, not a PSD.
    There is no averaging or interpolation between coefficients.
    """
    rate = np.asarray(rate, dtype=float)
    frequencies = np.asarray(np.arange(1, 20.01, 0.25) if frequencies is None else frequencies,
                             dtype=float)
    if rate.ndim != 1 or rate.size < 2 or not np.isfinite(rate).all():
        raise ValueError("Need a finite 1-D recording with at least two bins")
    if not np.isfinite(overlap) or not 0 <= overlap < 1:
        raise ValueError("overlap must be between 0 (inclusive) and 1 (exclusive)")
    if (frequencies.ndim != 1 or frequencies.size < 2
            or not np.all(np.diff(frequencies) > 0)):
        raise ValueError("Need at least two strictly increasing frequencies")
    times, powers = [], []
    for frequency in frequencies:
        wavelet = morlet_kernel(frequency, bin_s, window_periods)
        half = wavelet.size // 2
        hop = int(np.ceil((1 - overlap) * wavelet.size))
        centers = np.arange(0, rate.size, hop)
        padded = np.pad(rate, (half, half), mode="reflect")
        coefficients = signal.fftconvolve(padded, wavelet.conj()[::-1], mode="valid")
        times.append((centers + 0.5) * bin_s)
        powers.append(np.abs(coefficients[centers]) ** 2)
    return frequencies, times, powers


def load_record(config_path, node_sets, bin_s=0.005, window_periods=3.0, overlap=0.1,
                compute_spectrum=True):
    """Load the population rate; optionally leave spectral analysis to the caller."""
    config = json.loads(config_path.read_text())
    output = config["output"]
    spike_path = config_path.parent / output["output_dir"] / output["spikes_file"]
    ids = resolve_ids(node_sets, config["node_set"])
    if not ids.size:
        raise ValueError("Empty simulated node set")
    with h5py.File(spike_path, "r") as report:
        population = report["spikes"][POPULATION]
        units = population["timestamps"].attrs.get("units", "ms")
        if isinstance(units, bytes):
            units = units.decode()
        if units not in ("ms", "s"):
            raise ValueError(f"Unsupported timestamp units: {units}")
        timestamps = population["timestamps"][:] * (0.001 if units == "ms" else 1)
        node_ids = population["node_ids"][:]
    if not np.isfinite(timestamps).all() or timestamps.size != node_ids.size:
        raise ValueError("Invalid spike timestamps or node IDs")
    # Reports already contain the simulated population. Some exports renumber
    # IDs locally (0..N-1); filtering by original circuit IDs loses real spikes.
    outside_reference = int((~np.isin(node_ids, ids)).sum())
    stop_s = float(config["run"]["tstop"]) / 1000
    if np.any((timestamps < 0) | (timestamps > stop_s)):
        raise ValueError("Spike timestamps fall outside the configured recording")
    warnings = []
    last_spike = float(timestamps.max()) if timestamps.size else None
    # Spike-only files contain no recording-end marker. A long final gap is
    # ambiguous: mask it conservatively and make the uncertainty explicit.
    observed_stop = stop_s
    if last_spike is not None and stop_s - last_spike > 0.25:
        observed_stop = np.ceil(last_spike / bin_s) * bin_s
        warnings.append(f"End-time mismatch: config {stop_s:g} s, last spike {last_spike:.4f} s; tail unverified")
    n_bins = round(observed_stop / bin_s)
    if not np.isclose(n_bins * bin_s, observed_stop):
        raise ValueError("Recording duration must be divisible by bin size")
    edges = np.arange(n_bins + 1) * bin_s
    counts, _ = np.histogram(timestamps, bins=edges)
    rate = counts / (bin_s * ids.size)
    frequencies, times, power = (rate_spectrogram(rate, bin_s, window_periods, overlap)
                                 if compute_spectrum else (None, None, None))
    stimuli = []
    for name, stim in config.get("inputs", {}).items():
        if stim.get("input_type") != "extracellular_stimulation":
            continue
        fields = [f for f in stim.get("fields", [])
                  if any(float(f.get(k, 0)) != 0 for k in ("Ex", "Ey", "Ez"))]
        if not fields:
            continue
        start = float(stim["delay"]) / 1000
        end = min(stop_s, start + float(stim["duration"]) / 1000)
        if start < end:
            stimuli.append(dict(name=name, start=max(0, start), end=end,
                                frequencies=[float(f["frequency"]) for f in fields]))
    return dict(name=config_path.parent.name, config=str(config_path),
                spike_file=str(spike_path), stop_s=stop_s, observed_stop=observed_stop,
                last_spike_s=last_spike, n_neurons=int(ids.size), n_spikes=int(counts.sum()),
                spikes_outside_reference_node_ids=outside_reference,
                stimuli=stimuli, warnings=warnings,
                frequency=frequencies, time=times, power=power,
                bin_s=bin_s, window_periods=window_periods, overlap=overlap,
                rate=rate, rate_time=(edges[:-1] + edges[1:]) / 2)


def draw_spectrogram(ax, record, limits):
    # Each frequency has its own time grid. Draw one row at a time, without
    # interpolating sparse coefficients onto a falsely dense shared time grid.
    frequencies = record["frequency"]
    frequency_edges = np.r_[frequencies[0] - (frequencies[1] - frequencies[0]) / 2,
                            (frequencies[:-1] + frequencies[1:]) / 2,
                            frequencies[-1] + (frequencies[-1] - frequencies[-2]) / 2]
    stop, observed = record["stop_s"], record["observed_stop"]
    for i, (times, power) in enumerate(zip(record["time"], record["power"])):
        time_edges = np.r_[0, (times[:-1] + times[1:]) / 2, observed]
        mesh = ax.pcolormesh(time_edges, frequency_edges[i:i + 2],
                             10 * np.log10(np.maximum(power[None, :], 1e-20)),
                             shading="flat", cmap="magma", vmin=limits[0], vmax=limits[1],
                             rasterized=True)
    # Only the actual recording edges need padding; never pad at stimulation.
    half_widths = np.array([morlet_kernel(f, record["bin_s"], record["window_periods"]).size // 2
                            * record["bin_s"] for f in frequencies])
    for left, right in [(np.zeros_like(half_widths), np.minimum(half_widths, observed)),
                        (np.maximum(0, observed - half_widths), np.full_like(half_widths, observed))]:
        ax.fill_betweenx(frequencies, left, right, facecolor="none",
                         edgecolor=(1, 1, 1, 0.35), hatch="///", linewidth=0)
    if observed < stop:
        ax.axvspan(observed, stop, color="#dedede")
        ax.text((observed + stop) / 2, 12.5, "Unverified\ntail", ha="center", fontsize=9)
    for stim in record["stimuli"]:
        for freq in stim["frequencies"]:
            if frequencies[0] <= freq <= frequencies[-1]:
                ax.hlines(freq, stim["start"], stim["end"], color="#56d4e6", ls=":", lw=0.9)
    # A separate phase strip above the axes leaves spectral colors unobscured.
    ax.plot([0, stop], [1.035, 1.035], transform=ax.get_xaxis_transform(),
            color="#98a4b1", lw=6, clip_on=False, solid_capstyle="butt")
    for stim in record["stimuli"]:
        ax.plot([stim["start"], stim["end"]], [1.035, 1.035],
                transform=ax.get_xaxis_transform(), color="#21a6b7", lw=6,
                clip_on=False, solid_capstyle="butt")
    ax.set(xlim=(0, stop), ylim=(frequencies[0], frequencies[-1]), xlabel="Time from simulation start (s)",
           ylabel="Frequency (Hz)", yticks=[1, 5, 10, 15, 20])
    return mesh


def grouped_overviews(records, output_dir, limits):
    """Plot each stimulation frequency/state with its baseline first."""
    baselines = {"low": "resting", "high": "highCorrelation_convergence_stopStart"}
    by_name = {record["name"]: record for record in records}
    groups = {}
    for record in records:
        match = re.match(r"^(highCorrelation_)?sim(\d+(?:\.\d+)?)Hz(\d+(?:\.\d+)?)Vm", record["name"])
        if match is None:
            continue
        state = "high" if match[1] else "low"
        if state == "low" and "highCorrelation" in record["name"]:
            continue
        frequency = float(match[2])
        amplitude = float(match[3])
        groups.setdefault((state, frequency), []).append((amplitude, record))
    overviews = []
    for state, frequency in sorted(groups, key=lambda key: (key[0] == "high", key[1])):
        baseline = by_name.get(baselines[state])
        if baseline is None:
            raise ValueError(f"Missing {state}-correlation baseline: {baselines[state]}")
        stimulated = sorted(groups[state, frequency], key=lambda item: (item[0], item[1]["name"]))
        panels = [baseline, *[record for amplitude, record in stimulated]]
        ncols = min(3, len(panels))
        nrows = (len(panels) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 3.6 * nrows),
                                 squeeze=False, layout="constrained")
        fig.suptitle(f"{frequency:g} Hz stimulation | {state.capitalize()}-correlation", fontsize=18)
        for position, (ax, record) in enumerate(zip(axes.flat, panels)):
            mesh = draw_spectrogram(ax, record, limits)
            title = ("Unstimulated baseline: " if position == 0 else "") + record["name"].replace("_", " ")
            ax.set_title(textwrap.fill(title, 38) + (" *" if record["warnings"] else ""),
                         fontsize=10, pad=15)
            ax.tick_params(labelsize=9)
        for ax in axes.flat[len(panels):]:
            ax.set_visible(False)
        fig.colorbar(mesh, ax=list(axes.flat), label="Morlet power (dB re 1 (spikes/s/neuron)²)",
                     extend="both", shrink=0.85)
        fig.supxlabel("Baseline first; stimulated runs ordered by amplitude and seed.\n"
                      "Shared Morlet power scale; full recordings from time zero. * Timing warning.", fontsize=9)
        filename = f"grouped_{state}Correlation_{frequency:g}Hz.png"
        fig.savefig(Path(output_dir) / filename, dpi=150)
        plt.close(fig)
        overviews.append(dict(file=filename, state=state, frequency_hz=frequency,
                              simulations=[record["name"] for record in panels]))
    return overviews


def generate(data_dir, output_dir, node_sets_path, window_periods=3.0, overlap=0.1):
    """Generate every configured simulation, individual PNGs and overview sheets."""
    node_sets_path = Path(node_sets_path).expanduser()
    if not node_sets_path.is_file() and node_sets_path == Path.home() / "Downloads/O1_data/node_sets.json":
        mounted_path = Path(data_dir).resolve().parent.parent.parent / "Downloads/O1_data/node_sets.json"
        if mounted_path.is_file():
            node_sets_path = mounted_path
    if not node_sets_path.is_file():
        raise FileNotFoundError(f"Circuit node sets not found: {node_sets_path}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    node_sets = json.loads(node_sets_path.read_text())
    records, skipped = [], []
    for folder in sorted(Path(data_dir).iterdir()):
        if not folder.is_dir():
            continue
        configs = sorted(folder.glob("*.json"))
        if len(configs) != 1:
            skipped.append(dict(name=folder.name, reason=f"Expected one config, found {len(configs)}"))
            continue
        try:
            records.append(load_record(configs[0], node_sets, window_periods=window_periods, overlap=overlap))
        except (OSError, KeyError, ValueError) as exc:
            skipped.append(dict(name=folder.name, reason=str(exc)))
    if not records:
        raise ValueError(f"No simulations loaded: {skipped}")
    # Shared display limits; unclipped Morlet powers remain in NPZ files.
    values = np.concatenate([10 * np.log10(np.maximum(np.concatenate(r["power"]), 1e-20)) for r in records])
    limits = np.percentile(values, [2, 99])
    if limits[1] <= limits[0]:
        limits = np.array([limits[0] - 1, limits[1] + 1])
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    caption = (f"5 ms bins | Morlet, ≥{window_periods:g} periods/window | at most {100 * overlap:g}% support overlap | "
               f"shared power scale {limits[0]:.1f} to {limits[1]:.1f} dB\n"
               "Gray strip: rest / no field; teal: stimulation; dotted line: applied frequency; "
               "hatched: reflected recording edges")
    color_label = "Morlet power (dB re 1 (spikes/s/neuron)²)"
    metadata = []
    for r in records:
        fig, axes = plt.subplots(2, 1, figsize=(11, 6.8), height_ratios=[1, 2.6], layout="constrained")
        fig.suptitle(r["name"] + ("  [timing warning]" if r["warnings"] else ""), fontsize=14)
        axes[0].plot(r["rate_time"], r["rate"], color="#3e6683", lw=0.65)
        axes[0].set(xlim=(0, r["stop_s"]), ylabel="Spike rate\n(spikes/s/neuron)")
        for stim in r["stimuli"]:
            axes[0].axvspan(stim["start"], stim["end"], color="#21a6b7", alpha=0.12)
        axes[0].set_title(f"{r['n_neurons']:,} simulated neurons | {r['n_spikes']:,} spikes", fontsize=10)
        mesh = draw_spectrogram(axes[1], r, limits)
        fig.colorbar(mesh, ax=axes, label=color_label, extend="both", shrink=0.85)
        fig.supxlabel(caption + ("\n" + "; ".join(r["warnings"]) if r["warnings"] else ""), fontsize=8)
        fig.savefig(output_dir / f"{r['name']}.png", dpi=160)
        plt.close(fig)
        # Flatten the frequency-specific time grids without pickled/ragged arrays.
        # Row i occupies row_offsets[i]:row_offsets[i+1] in time_s and wavelet_power.
        offsets = np.r_[0, np.cumsum([len(t) for t in r["time"]])]
        np.savez_compressed(output_dir / f"{r['name']}.npz", frequency_hz=r["frequency"],
                            time_s=np.concatenate(r["time"]), wavelet_power=np.concatenate(r["power"]),
                            row_offsets=offsets, rate_time_s=r["rate_time"],
                            window_duration_s=np.array([(morlet_kernel(f, r["bin_s"], window_periods).size - 1)
                                                        * r["bin_s"] for f in r["frequency"]]),
                            firing_rate_hz_per_neuron=r["rate"],
                            window_periods=window_periods, overlap=overlap)
        metadata.append({k: v for k, v in r.items()
                         if k not in {"frequency", "time", "power", "rate", "rate_time"}})
    overview_names = []
    for page, start in enumerate(range(0, len(records), 9), 1):
        fig, axes = plt.subplots(3, 3, figsize=(16, 11), layout="constrained")
        fig.suptitle(f"Full-recording Morlet spectrograms | 1–20 Hz | {page}", fontsize=18)
        for ax, r in zip(axes.flat, records[start:start + 9]):
            mesh = draw_spectrogram(ax, r, limits)
            title = r["name"].replace("_", " ")
            ax.set_title(textwrap.fill(title, 38) + (" *" if r["warnings"] else ""), fontsize=10, pad=15)
            ax.tick_params(labelsize=9)
        for ax in axes.flat[len(records[start:start + 9]):]:
            ax.set_visible(False)
        fig.colorbar(mesh, ax=list(axes.flat), label=color_label, extend="both", shrink=0.85)
        fig.supxlabel(caption + "\n* See individual plot for timing warning.", fontsize=9)
        name = f"overview_{page:02d}.png"
        fig.savefig(output_dir / name, dpi=150)
        plt.close(fig)
        overview_names.append(name)
    grouped = grouped_overviews(records, output_dir, limits)
    manifest = dict(parameters=dict(bin_ms=5, window_periods=window_periods, overlap=overlap,
                                    estimator="complex Morlet", support_sigma=3,
                                    normalization="discrete unit energy, zero mean",
                                    window_duration="at least window_periods/frequency seconds; rounded up to samples",
                                    time_sampling="frequency-dependent; hop=ceil((1-overlap)*kernel_samples)",
                                    frequency_range_hz=[1, 20], color_limits_db=limits.tolist(),
                                    node_sets=str(node_sets_path)), simulations=metadata, skipped=skipped,
                    grouped_overviews=grouped)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        columns = ["name", "n_neurons", "n_spikes", "stop_s", "last_spike_s", "warnings"]
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(metadata)
    links = "".join(f'<a href="{n}"><img src="{n}" alt="Overview {i}"></a>'
                    for i, n in enumerate(overview_names, 1))
    grouped_links = "".join(
        f'<h3>{group["frequency_hz"]:g} Hz, {group["state"]}-correlation</h3>'
        f'<a href="{group["file"]}"><img src="{group["file"]}" alt="Baseline and stimulated spectrograms"></a>'
        for group in grouped
    )
    cards = "".join(f'<details><summary>{html.escape(r["name"])}</summary>'
                    f'<a href="{r["name"]}.png"><img loading="lazy" src="{r["name"]}.png"></a>'
                    f'<a href="{r["name"]}.npz">Numerical data (NPZ)</a></details>' for r in records)
    missing = "".join(f'<li>{html.escape(s["name"])}: {html.escape(s["reason"])}</li>' for s in skipped)
    document = f'''<!doctype html><html lang="en"><meta charset="utf-8">
<title>Morlet spectrograms — 1–20 Hz</title>
<style>body{{font:16px system-ui;max-width:1300px;margin:36px auto;padding:0 20px;color:#172938}}
img{{width:100%;height:auto}}summary{{cursor:pointer;padding:12px;background:#edf3f6;margin-top:8px}}
p{{line-height:1.6}}</style><h1>Morlet spectrograms · 1–20 Hz</h1>
<p>{len(records)} simulations, from t = 0 through the configured end. All simulated neurons
(including silent neurons) enter the firing-rate denominator. No smoothing or bandpass filter.
Gray phase strip: no electric field; teal: electric-field stimulation.</p>
<p>Morlet transform of each complete recording, with windows spanning at least {window_periods:g} periods and at most
{100 * overlap:g}% overlap between adjacent sampled wavelet supports. The Gaussian width is
sigma = periods / (6 frequency); full supports span at least periods / frequency seconds. Higher frequencies therefore
use shorter wavelets and finer time sampling. No split, padding or restart at stimulation
onset or offset. Wavelets near a boundary naturally include activity from both sides.</p>
<p>Colours show squared magnitudes of unit-energy, zero-mean Morlet coefficients (not PSD).
Shared limits: {limits[0]:.2f} to {limits[1]:.2f} dB (pooled 2nd–99th percentiles).
Hatched recording edges use reflected padding. Gray unverified tails flag a gap exceeding
250 ms between the last spike and configured stop. Numerical data retain unclipped power:
row_offsets indexes each frequency's time_s and wavelet_power arrays.</p>
<h2>By frequency and network state (baseline first)</h2>{grouped_links}
<h2>All recordings</h2>{links}<h2>Individual recordings</h2>{cards}
<h2>Unavailable simulations ({len(skipped)})</h2><ul>{missing}</ul></html>'''
    (output_dir / "index.html").write_text(document, encoding="utf-8")
    print(f"Generated {len(records)} spectrograms and {len(overview_names)} overviews in {output_dir}")
    print(f"Skipped: {skipped}")
    for r in records:
        if r["warnings"]:
            print(r["name"], r["warnings"])
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/spectrograms_1_20Hz"))
    parser.add_argument("--node-sets", type=Path, required=True)
    parser.add_argument("--window-periods", type=float, default=3.0,
                        help="minimum periods per full wavelet support: duration = periods / frequency")
    parser.add_argument("--overlap", type=float, default=0.1,
                        help="maximum overlap fraction of adjacent sampled wavelet supports (0 to <1)")
    args = parser.parse_args()
    generate(args.data_dir, args.output_dir, args.node_sets, args.window_periods, args.overlap)

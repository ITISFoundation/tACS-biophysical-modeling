"""Frequency-wise baseline PSD ratios and reproducible plots from saved rates."""
from pathlib import Path
import argparse
import csv
import json
import re
import warnings

import numpy as np


def baseline_psd_ratio(spectrum, baseline, *, db=False, min_baseline=0.0):
    """Divide linear PSDs on identical grids; invalid denominators become NaN.

    No total-power normalization or epsilon addition is applied. ``min_baseline``
    is an optional absolute PSD threshold in the input PSD's units.
    """
    f, power = (np.asarray(x, dtype=float) for x in spectrum)
    bf, base = (np.asarray(x, dtype=float) for x in baseline)
    if (f.ndim != 1 or not f.size or f.shape != power.shape or bf.shape != base.shape
            or f.shape != bf.shape or not np.isfinite(f).all()
            or not np.allclose(f, bf) or np.any(np.diff(f) <= 0)):
        raise ValueError("PSD arrays must have the same increasing, finite frequency grid")
    if not np.isfinite(min_baseline) or min_baseline < 0:
        raise ValueError("min_baseline must be finite and nonnegative")
    valid = np.isfinite(base) & (base > min_baseline) & np.isfinite(power) & (power >= 0)
    ratio = np.full(power.shape, np.nan)
    np.divide(power, base, out=ratio, where=valid)
    if not valid.all():
        warnings.warn(f"Masked {(~valid).sum()} invalid/low-baseline PSD bins", stacklevel=2)
    if db:
        with np.errstate(divide="ignore"):
            ratio = 10 * np.log10(ratio)
    return f.copy(), ratio


def shared_ylim(curves, xlim=(5, 20), step=5.0):
    """One y-range covering every ``(frequencies, values)`` curve inside ``xlim``.

    Only the plotted frequencies count (autoscaling would also fit the hidden 0-5 and >20 Hz
    bins). NaN/inf values (masked baseline bins) are ignored. Limits are padded by 2% and
    rounded outward to multiples of ``step``, so every plot of a set gets the same ticks.
    """
    values = np.concatenate([np.asarray(y, dtype=float)[(np.asarray(f) >= xlim[0]) & (np.asarray(f) <= xlim[1])]
                             for f, y in curves])
    values = values[np.isfinite(values)]
    if not values.size:
        raise ValueError("No finite values inside xlim")
    pad = 0.02 * (values.max() - values.min())
    low, high = step * np.floor((values.min() - pad) / step), step * np.ceil((values.max() + pad) / step)
    return (float(low), float(high)) if high > low else (float(low - step), float(high + step))


def generate_baseline_plots(root=Path(".")):
    """Use the notebook's state pairing, selections, 100-sample Welch segments.

    Input: full-population unsmoothed firing-rate NPZs from src.spectrograms.
    Smoothing is applied after selecting the analysis interval, as in get_psd_data.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import welch

    root = Path(root)
    input_dir = root / "output/spectrograms_1_20Hz"
    output = root / "output/psd_baseline_ratios"
    output.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader((root / "data.csv").open()))
    baseline_ids = {"low": "resting", "high": "baseline_highCorrelation_stopStart"}
    selections = {"low": r"highCorrelation|replay|hex0|O1|obi|baseline|resting",
                  "high": r"baseline|replay"}
    manifest = dict(nperseg=100, bin_ms=5, smoothing_sigma_ms=10,
                    baseline_window_ms=[1500, 5000], groups=[], warnings=[])

    def spectrum(row, baseline=False):
        name = Path(row["source_file"]).parent.name
        with np.load(input_dir / f"{name}.npz") as data:
            t, rate = data["rate_time_s"], data["firing_rate_hz_per_neuron"]
            start = 1.5 if baseline else float(row["delay"]) / 1000
            stop = float(row["tstop"]) / 1000
            if stop > t[-1] + .0025 + 1e-9:
                raise ValueError(f"Requested window exceeds available data: {name}")
            rate = rate[(t >= start) & (t < stop)]
        if rate.size < 100:
            raise ValueError(f"Less than one Welch segment: {name}")
        return welch(gaussian_filter1d(rate, sigma=2), fs=200, window="hann",
                     nperseg=100, noverlap=50, detrend="constant", scaling="density")

    figures = {}
    for state, baseline_id in baseline_ids.items():
        baseline_row = next(r for r in rows if r["sim_id"] == baseline_id)
        base = spectrum(baseline_row, baseline=True)
        for freq in ((7, 10, 15) if state == "low" else (7, 10)):
            members = [r for r in rows if float(r["stim_freq"]) == freq
                       and not re.search(selections[state], r["sim_id"], flags=re.IGNORECASE)
                       and (state == "low" or "highCorrelation" in r["sim_id"])]
            curves = []
            group = dict(state=state, frequency_hz=freq, baseline=baseline_id, amplitudes={})
            for amp in sorted({float(r["stim_amp"]) for r in members}):
                runs = [r for r in members if float(r["stim_amp"]) == amp]
                powers = [spectrum(r) for r in runs]
                if any(not np.array_equal(p[0], base[0]) for p in powers):
                    raise ValueError("Welch frequency grids differ")
                mean = np.mean([p[1] for p in powers], axis=0)
                f, ratio = baseline_psd_ratio((base[0], mean), base)
                curves.append((amp, len(runs), f, ratio))
                group["amplitudes"][str(amp)] = [r["sim_id"] for r in runs]
                np.savez_compressed(output / f"{state}_{freq}Hz_{amp:g}Vm.npz",
                                    frequency_hz=f, stimulated_psd=mean, baseline_psd=base[1], ratio=ratio)
            manifest["groups"].append(group)
            figures[state, freq] = curves
    # Every dB plot gets the same y-axis so the five state/frequency panels can be compared.
    db_ylim = shared_ylim([(f, 10 * np.log10(ratio)) for curves in figures.values()
                           for _, _, f, ratio in curves])
    manifest["db_ylim"] = db_ylim
    for (state, freq), curves in figures.items():
        for db in (False, True):
            fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
            draw(ax, curves, state, freq, baseline_ids[state], db, ylim=db_ylim if db else None)
            fig.supxlabel("500 ms Hann segments · 50% overlap · 2 Hz frequency spacing\n"
                          "Mean stimulated PSD across seeds / state-matched baseline PSD", fontsize=9)
            stem = f"psd_ratio_{state}_{freq}Hz" + ("_db" if db else "")
            for ext in ("png", "svg"):
                fig.savefig(output / f"{stem}.{ext}", dpi=160)
            plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), layout="constrained")
    for ax, ((state, freq), curves) in zip(axes.flat, figures.items()):
        draw(ax, curves, state, freq, baseline_ids[state], False)
    axes.flat[-1].axis("off")
    axes.flat[-1].text(0, .9, "Ratio = PSD stimulation / PSD baseline\n\n"
                     "1: unchanged power\n2: twice baseline\n0.5: half baseline\n\n"
                     "5–20 Hz; 2 Hz frequency spacing\n"
                     "Frequency-wise division; no total-power scaling\n"
                     "Descriptive estimates, not significance tests", va="top", fontsize=12)
    fig.suptitle("Power relative to the corresponding network-state baseline", fontsize=17)
    fig.savefig(output / "overview.png", dpi=160)
    plt.close(fig)
    manifest["warnings"].append("sim10Hz0.5Vm_seed2 uses CSV window 1.1–2.5 s; config ends at 3.5 s. Timing requires verification.")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Saved 5 ratio plots, 5 dB plots, numerical arrays and overview to {output}")


def draw(ax, curves, state, freq, baseline_id, db, ylim=None):
    import matplotlib.pyplot as plt
    for (amp, n, f, ratio), color in zip(curves, plt.get_cmap("viridis")(np.linspace(.1, .9, len(curves)))):
        ax.plot(f, 10 * np.log10(ratio) if db else ratio, label=f"{amp:g} V/m (n={n})", color=color)
    ax.axhline(0 if db else 1, color="black", ls="--", lw=1, label="Baseline")
    ax.axvline(freq, color="0.5", ls=":", lw=1)
    ax.set(xlim=(5, 20), xlabel="Frequency (Hz)",
           ylabel="10 log10(PSD / baseline) [dB]" if db else "PSD / baseline PSD (fold)",
           title=f"{state.capitalize()} correlation · {freq} Hz\nBaseline: {baseline_id}")
    if ylim is not None:
        ax.set_ylim(ylim)
    elif not db:
        ax.set_ylim(bottom=0)
    ax.grid(alpha=.2)
    ax.legend(fontsize=8, frameon=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    generate_baseline_plots(parser.parse_args().root)

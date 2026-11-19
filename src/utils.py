"""Data loading and analysis helpers for the coupling-project notebooks."""

from __future__ import annotations

import os
import math
import warnings
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
from scipy import signal, stats
from scipy.ndimage import gaussian_filter1d


POPULATION = "S1nonbarrel_neurons"
CELL_TYPES = ("PYR", "INT_INH", "INT_EXC")
BASELINE_START_MS = 1500.0
CORRECTION_METHOD = "holm"

@contextmanager
def silence_fdstderr():
    """Silence low-level stderr output from HDF5 and BluePySnap calls."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved_stderr = os.dup(2)
    os.dup2(devnull, 2)
    try:
        yield
    finally:
        os.dup2(saved_stderr, 2)
        os.close(saved_stderr)
        os.close(devnull)


def load_data(sim_config_path: str | Path, with_soma: bool = True):
    """Load simulation, soma report, node table, and spikes for the population."""
    import bluepysnap as bp

    with silence_fdstderr():
        try:
            simulation = bp.Simulation(str(sim_config_path))
        except Exception:
            return None, None, None, None
        all_data = None
        try:
            if with_soma:
                all_data = simulation.reports["soma"][POPULATION].get()
        except Exception:
            all_data = None
        try:
            spike_data = simulation.spikes[POPULATION].get()
        except Exception:
            spike_data = None
        try:
            cell_data = simulation.circuit.nodes[POPULATION].get(
                group=simulation.config["node_set"]
            )
        except Exception:
            cell_data = None
    return simulation, all_data, cell_data, spike_data


@lru_cache(maxsize=64)
def load_simulation(sim_config_path: str | Path):
    """Cached ``load_data`` without the soma report, plus depth ranks.

    Returns a namespace with ``simulation``, ``cells``, ``spikes`` and ``cell_to_rank``.
    Every analysis helper below loads through this function, so each simulation is read
    from disk only once per session. Treat the returned objects as read-only.
    """
    simulation, _all_data, cells, spikes = load_data(sim_config_path, with_soma=False)
    if simulation is None or cells is None or spikes is None:
        raise FileNotFoundError(f"Could not load {sim_config_path}. Did you extract data.zip?")
    return SimpleNamespace(simulation=simulation, cells=cells, spikes=spikes,
                           cell_to_rank=geom_rank_cell_nodes(cells, simulation))


# --------------------------------------------------------------------------- #
# Simulation table (data.csv)
# --------------------------------------------------------------------------- #
def load_simulation_table(path="data.csv"):
    """Read ``data.csv``: numeric metadata, complex ``complex``/``plv`` features, ``abs_plv`` magnitudes."""
    table = remove_unnamed_columns(pd.read_csv(path))
    for column in ("delay", "tstop", "stim_freq", "stim_amp", "stim_phase", "stim_orientation"):
        table[column] = pd.to_numeric(table[column], errors="raise")
    # "complex" and "plv" features are complex numbers, stored as strings such as "(1+2j)".
    # Parse them so merge_and_average keeps them, and add the magnitude of every plv column as
    # "abs_plv...", which is what the amplitude plots and the linear fits use.
    for column in table.columns:
        if column.startswith(("complex", "plv")):
            table[column] = table[column].astype(complex)
    for column in [c for c in table.columns if c.startswith("plv")]:
        table[f"abs_{column}"] = table[column].abs()
    return table


def check_data_available(simulations):
    """Warn about ``source_file`` paths that do not exist on disk."""
    missing = simulations.loc[~simulations["source_file"].map(lambda p: Path(p).is_file()), "sim_id"]
    if len(missing) == len(simulations):
        warnings.warn("No simulation files found. Extract data.zip into the project root first.")
    elif len(missing):
        warnings.warn(f"{len(missing)} simulation(s) have no source file: {', '.join(missing)}")
    else:
        print(f"All {len(simulations)} simulations found.")


def sim_row(simulations, sim_id):
    """Return the single-row DataFrame for ``sim_id`` (raises if absent or duplicated)."""
    rows = simulations[simulations["sim_id"] == sim_id]
    if len(rows) != 1:
        raise KeyError(f"Expected exactly one simulation named {sim_id!r} in data.csv, found {len(rows)}.")
    return rows


def select_simulations(simulations, include=None, exclude=None, freqs=None, amps=None,
                       stim_phase=None, stim_orientation=None):
    """Filter IDs (regexes), frequencies, amplitudes and optional angles in radians.

    Angles use ``np.isclose`` to tolerate floating-point rounding.
    Results are sorted by frequency, amplitude and ID.
    """
    ids = simulations["sim_id"].fillna("").astype(str)
    mask = pd.Series(True, index=simulations.index)
    if include is not None:
        mask &= ids.str.contains(include, regex=True)
    if exclude is not None:
        mask &= ~ids.str.contains(exclude, case=False, regex=True)
    if freqs is not None:
        mask &= simulations["stim_freq"].isin(freqs)
    if amps is not None:
        mask &= simulations["stim_amp"].isin(amps)
    if stim_phase is not None:
        mask &= np.isclose(simulations["stim_phase"], stim_phase)
    if stim_orientation is not None:
        mask &= np.isclose(simulations["stim_orientation"], stim_orientation)
    return simulations[mask].sort_values(["stim_freq", "stim_amp", "sim_id"]).reset_index(drop=True)


def rows_at(rows, freq=None, amp=None):
    """Rows at a stimulation frequency and/or amplitude."""
    mask = pd.Series(True, index=rows.index)
    if freq is not None:
        mask &= np.isclose(rows["stim_freq"], freq)
    if amp is not None:
        mask &= np.isclose(rows["stim_amp"], amp)
    return rows[mask]


def define_central_axis(soma_pos):
    """Return the PCA-derived central axis of the cortical column."""
    from sklearn.decomposition import PCA

    center = np.mean(soma_pos, axis=0).values
    pca = PCA(n_components=3)
    pca.fit(soma_pos)
    main_axis = pca.components_[0]
    elevation = np.arctan2(main_axis[2], np.sqrt(main_axis[0] ** 2 + main_axis[1] ** 2))
    azimuth = np.arctan2(main_axis[1], main_axis[0])
    return center, azimuth, elevation, main_axis


def rotation_matrix_from_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the rotation matrix that rotates unit vector ``a`` onto ``b``."""
    v = np.cross(a, b)
    c = np.dot(a, b)
    v_norm = np.linalg.norm(v)
    if v_norm < 1e-12:
        if c > 0:
            return np.eye(3)
        orthogonal = np.array([1.0, 0.0, 0.0])
        if abs(a[0]) > 0.9:
            orthogonal = np.array([0.0, 1.0, 0.0])
        axis = orthogonal - np.dot(orthogonal, a) * a
        axis = axis / np.linalg.norm(axis)
        return np.eye(3) - 2.0 * np.outer(axis, axis)
    k = v / v_norm
    k_matrix = np.array(
        [[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]]
    )
    angle = np.arccos(np.clip(c, -1.0, 1.0))
    return np.eye(3) + np.sin(angle) * k_matrix + (1 - np.cos(angle)) * (k_matrix @ k_matrix)


def get_soma_pos(simulation_data):
    """Return soma coordinates for the project population."""
    return simulation_data.circuit.nodes[POPULATION].get(properties=["x", "y", "z"])


def geom_rank_cell_nodes(cell_data, simulation_data):
    """Rank cells by depth after rotating the cortical-column axis onto z."""
    soma_position = get_soma_pos(simulation_data)
    _center, _azimuth, _elevation, pca_axis = define_central_axis(soma_position)
    pca_axis = pca_axis / np.linalg.norm(pca_axis)
    rotation = rotation_matrix_from_vectors(pca_axis, np.array([0.0, 0.0, 1.0]))
    rotated_position = soma_position @ rotation.T
    rotated_position.columns = ["x", "y", "z"]
    ranked_cells = cell_data.copy()
    ranked_cells.update(rotated_position[["x", "y", "z"]])
    ranked_cells.sort_values(by="z", ascending=False, inplace=True)
    ranked_cells["rank"] = ranked_cells["z"].rank(method="first", ascending=True).astype(int)
    return ranked_cells["rank"]


def get_estim_data(simulation_data):
    """Return electric-field configuration values, or ``None`` when absent."""
    try:
        estim = simulation_data.config["inputs"]["ex_efields"]
        fields = estim["fields"]
        delay = estim["delay"]
        duration = estim["duration"]
        ramp_up = estim["ramp_up_time"]
        ramp_down = estim["ramp_down_time"]
    except Exception:
        return None
    return fields, delay, duration, ramp_up, ramp_down


def get_firingrate_hist(
    simulation_data, spike_data, bin_width, n_neurons, delay=None, time_stop=None,
    center=True, smoothed=False, sigma_ms=10,
):
    """Return population firing rate and bin centres for a simulation window."""
    stimulation_info = get_estim_data(simulation_data)
    if delay is None:
        delay = 200 if stimulation_info is None else stimulation_info[1]
    if time_stop is None:
        time_stop = simulation_data.time_stop
    if n_neurons <= 0 or time_stop <= delay:
        return np.array([], dtype=float), np.array([], dtype=float)
    bins = np.arange(delay, time_stop + bin_width, bin_width)
    hist_counts, _ = np.histogram(spike_data.index.tolist(), bins=bins)
    firing_rate = hist_counts / (bin_width * 1e-3) / n_neurons
    if center:
        bins = bins[1:] - bin_width / 2
    if smoothed:
        firing_rate = gaussian_filter1d(firing_rate, sigma=sigma_ms / bin_width)
    return firing_rate, bins


def cell_type_ids(cell_data, cell_type=None, layers=None):
    """Node ids of a cell type ('PYR', 'INT_INH', 'INT_EXC'; None/'ALL' = all), optionally per layer."""
    if layers is not None:
        cell_data = cell_data[cell_data["layer"].astype(int).isin(layers)]
    if cell_type in (None, "ALL"):
        return cell_data.index
    morph_class, _, synapse_class = cell_type.partition("_")
    mask = cell_data["morph_class"] == morph_class
    if synapse_class:
        mask &= cell_data["synapse_class"] == synapse_class
    return cell_data.index[mask]


def extract_main_cell_types(data, cell_data, cell_to_rank):
    """Split spike data and depth ranks into the three displayed cell groups."""
    data_split = {}
    for cell_type in CELL_TYPES:
        node_ids = cell_type_ids(cell_data, cell_type)
        data_split[cell_type] = [
            data[data.isin(node_ids)],
            cell_to_rank.loc[cell_to_rank.index.isin(node_ids)],
        ]
    return data_split


def windowed_spike_phases(spike_data, t_stop, freq, phase0=0.0, stim_start=0.0) -> np.ndarray:
    """Return wrapped spike phases in the requested stimulation window."""
    spike_times_ms = np.array(spike_data.index.tolist())
    spike_times_ms = spike_times_ms[spike_times_ms > stim_start]
    spike_times_ms = spike_times_ms[spike_times_ms < t_stop]
    return (2 * np.pi * freq * (spike_times_ms - stim_start) * 1e-3 + phase0) % (2 * np.pi)


def get_ppc(spike_data, t_stop, freq, phase0=0.0, stim_start=0.0):
    """Return pairwise phase consistency for spikes in an analysis window."""
    theta = windowed_spike_phases(spike_data, t_stop, freq, phase0, stim_start)
    n_spikes = len(theta)
    if n_spikes < 2:
        return np.nan
    phase_sum = np.sum(np.exp(1j * theta))
    return (np.abs(phase_sum) ** 2 - n_spikes) / (n_spikes * (n_spikes - 1))


def get_plv(spike_data, t_stop, freq, phase0=0.0, stim_start=0.0):
    """Return phase-locking value for spikes in an analysis window."""
    theta = windowed_spike_phases(spike_data, t_stop, freq, phase0, stim_start)
    if len(theta) == 0:
        return np.nan
    return np.abs(np.mean(np.exp(1j * theta)))


def get_data(config_folders):
    """Load all data required for each configured simulation folder."""
    data_dict = {}
    for config_folder in config_folders:
        simulation_data, all_data, cell_data, spike_data = load_data(
            f"{config_folder}/sim_with_stim.json"
        )
        cell_to_rank = geom_rank_cell_nodes(cell_data, simulation_data)
        data_dict[config_folder] = [simulation_data, all_data, cell_data, spike_data, cell_to_rank]
    return data_dict


def compute_fourier(hist, sample_period_s):
    """Return one-sided FFT frequencies, amplitude, phase, and coefficients."""
    yf = fft(hist)
    n_samples = hist.shape[0]
    frequencies = fftfreq(n_samples, sample_period_s)
    positive = slice(0, n_samples // 2)
    xf_pos = frequencies[positive]
    amplitude = 2.0 / n_samples * np.abs(yf[positive])
    phase = np.angle(yf[positive])
    return xf_pos, amplitude, phase, yf


def get_spiking(data, cell_type=None):
    """Spikes and depth ranks of a simulation row (optionally one cell type). Cached load."""
    sim = load_simulation(data["source_file"].values[0])
    spike_data, cell_to_rank = sim.spikes, sim.cell_to_rank
    if cell_type not in (None, "ALL"):
        spike_data, cell_to_rank = extract_main_cell_types(spike_data, sim.cells, cell_to_rank)[cell_type]
    return spike_data, cell_to_rank


def goertzel_component(hist: np.ndarray, sample_period_s: float, freq: float) -> tuple[complex, float]:
    """Return the exact-frequency complex Fourier coefficient and amplitude."""
    n = np.arange(hist.shape[0])
    time = n * sample_period_s
    angular_frequency = 2 * np.pi * freq
    coeff = np.sum(hist * np.exp(-1j * angular_frequency * time))
    amplitude = np.abs(coeff) / hist.shape[0]
    return coeff, amplitude


def _baseline_window(data, start, time_stop):
    """Default baseline window: [BASELINE_START_MS, the row's tstop]."""
    start = BASELINE_START_MS if start is None else start
    time_stop = float(data["tstop"].values[0]) if time_stop is None else time_stop
    return start, time_stop

def get_psd_data(data, cell_type=None, start=None, time_stop=None, bin_width=5, nperseg=256):
    """Return the Welch power spectral density of the smoothed population rate for one row.

    Returns ``(frequencies, psd)``. ``nperseg`` samples per segment fixes the frequency grid, so
    spectra that are to be compared or overlaid must use the same value (and it cannot exceed the
    number of bins in the window: 1 s at 5 ms bins is 200). Welch detrends each segment, so bin 0
    is not the DC level of the firing rate.
    """
    start, time_stop = _baseline_window(data, start, time_stop)
    spike_data, cell_to_rank = get_spiking(data, cell_type)
    simulation_data = load_simulation(data["source_file"].values[0]).simulation
    hist, _ = get_firingrate_hist(
        simulation_data, spike_data, bin_width=bin_width, n_neurons=cell_to_rank.shape[0],
        delay=start, time_stop=time_stop, smoothed=True,
    )
    return signal.welch(hist, fs=1 / (bin_width * 1e-3), nperseg=nperseg)

def get_baseline_data(data, cell_type=None, start=None, time_stop=None, bin_width=5):
    """Return the FFT of the smoothed population rate for one row and cell group.

    Returns ``(frequencies, amplitude, phase, coefficients)`` (see ``compute_fourier``). This is the
    spectrum the stored features are defined on (``get_baseline_complex`` / ``_power`` /
    ``_amplitude_norm`` use it). For plotting a power spectral density use ``get_psd_data``, which
    applies Welch averaging and returns power rather than amplitude.
    """
    start, time_stop = _baseline_window(data, start, time_stop)
    spike_data, cell_to_rank = get_spiking(data, cell_type)
    simulation_data = load_simulation(data["source_file"].values[0]).simulation
    hist, _ = get_firingrate_hist(
        simulation_data, spike_data, bin_width=bin_width, n_neurons=cell_to_rank.shape[0],
        delay=start, time_stop=time_stop, smoothed=True,
    )
    return compute_fourier(hist, bin_width * 1e-3)


def psd_by_amplitude(runs, baseline, freq, amps, cell_type=None, nperseg=200, bin_width=5, start=None, time_stop=None):
    """Welch PSD of the population firing rate per stimulation amplitude, plus the baseline.

    Returns ``{0.0: (frequencies, psd), amp: (frequencies, psd), ...}``: the baseline first (over
    its own analysis window), then every amplitude of ``runs`` at stimulation frequency ``freq``
    (over that run's [delay, tstop] window). Seeds of one condition are averaged. Every spectrum
    uses the same ``nperseg``, so all of them share one frequency grid and can be overlaid;
    amplitudes without a simulation are skipped.
    """
    frequencies, baseline_psd = get_psd_data(baseline, cell_type=cell_type, nperseg=nperseg,
                                             bin_width=bin_width)
    spectra = {0.0: (frequencies, baseline_psd)}
    for amp in amps:
        rows = rows_at(runs, freq, amp)
        powers = []
        for position in range(len(rows)):
            row = rows.iloc[[position]]
            run_start = float(row["delay"].iloc[0]) if start is None else start
            run_stop = float(row["tstop"].iloc[0]) if time_stop is None else time_stop
            run_frequencies, run_psd = get_psd_data(
                row, cell_type=cell_type, start=run_start,
                time_stop=run_stop, nperseg=nperseg, bin_width=bin_width,
            )
            if run_frequencies.shape != frequencies.shape or not np.allclose(run_frequencies, frequencies):
                raise ValueError(f"{row['sim_id'].iloc[0]} has a different frequency grid than the baseline.")
            powers.append(run_psd)
        if powers:
            spectra[float(amp)] = (frequencies, np.mean(powers, axis=0))
    return spectra


def get_baseline_complex(data, target_freq, cell_type=None, start=None, time_stop=None):
    """Return the baseline FFT coefficient nearest the target frequency."""
    xf_pos, _amplitude, _phase, yf = get_baseline_data(data, cell_type, start, time_stop)
    phase0 = data["stim_phase"].values[0]
    orientation = data["stim_orientation"].values[0]
    # reference the complex value by the stimulation phase and the field orientation
    yf = yf * np.exp(-1j * (phase0 + orientation))
    return yf[np.argmin(np.abs(xf_pos - target_freq))]


def get_baseline_power(data, target_freq, cell_type=None, start=None, time_stop=None):
    """Return the total-power-normalised baseline value nearest target frequency."""
    xf_pos, amplitude, _phase, _yf = get_baseline_data(data, cell_type, start, time_stop)
    power = amplitude ** 2
    power_norm = power / np.sum(power)
    return power_norm[np.argmin(np.abs(xf_pos - target_freq))]


def get_baseline_amplitude_norm(data, target_freq, cell_type=None, start=None, time_stop=None):
    """Return the DC-normalised baseline amplitude nearest target frequency.

    Same definition as the stored ``amplitude_norm`` feature (the FFT scaling cancels in the ratio).
    """
    xf_pos, amplitude, _phase, _yf = get_baseline_data(data, cell_type, start, time_stop)
    return amplitude[np.argmin(np.abs(xf_pos - target_freq))] / amplitude[0]


def _baseline_spike_phases(spike_data, t_stop, freq, phase0=0.0, phase_orientation=0.0, delay=0.0):
    spike_times_s = np.array(spike_data.index.tolist()) * 1e-3
    spike_times_s = spike_times_s[spike_times_s > delay]
    spike_times_s = spike_times_s[spike_times_s < t_stop]
    return 2 * np.pi * freq * (spike_times_s - delay) + phase0 + phase_orientation


def get_baseline_plv(data, target_freq, cell_type=None, start=None, time_stop=None):
    """Return baseline PLV for one simulation row and optional cell group."""
    start, time_stop = _baseline_window(data, start, time_stop)
    spike_data, _cell_to_rank = get_spiking(data, cell_type)
    theta = _baseline_spike_phases(spike_data, time_stop * 1e-3, target_freq, delay=start * 1e-3)
    if len(theta) == 0:
        return np.nan
    return np.mean(np.exp(1j * theta))


def get_baseline_plv_magnitude(data, target_freq, cell_type=None, start=None, time_stop=None):
    """Return the length of the baseline PLV vector (what the amplitude plots show)."""
    return np.abs(get_baseline_plv(data, target_freq, cell_type, start, time_stop))


def get_baseline_ppc(data, target_freq, cell_type=None, start=None, time_stop=None):
    """Return baseline pairwise phase consistency for one simulation row and cell group."""
    start, time_stop = _baseline_window(data, start, time_stop)
    spike_data, _cell_to_rank = get_spiking(data, cell_type)
    theta = _baseline_spike_phases(spike_data, time_stop * 1e-3, target_freq, delay=start * 1e-3)
    n_spikes = len(theta)
    if n_spikes < 2:
        return np.nan
    return float((np.abs(np.sum(np.exp(1j * theta))) ** 2 - n_spikes) / (n_spikes * (n_spikes - 1)))


BASELINE_FUNCTIONS = {
    "complex": get_baseline_complex,
    "plv": get_baseline_plv,
    "abs_plv": get_baseline_plv_magnitude,
    "ppc": get_baseline_ppc,
    "power_norm": get_baseline_power,
    "amplitude_norm": get_baseline_amplitude_norm,
}


def get_baseline(data, metric, target_freq, cell_type=None, start=None, time_stop=None):
    """Baseline value of a stored feature.

    ``'plv'`` is the complex PLV vector (for polar plots) and ``'abs_plv'`` its magnitude (for
    amplitude plots); also ``'complex'``, ``'ppc'``, ``'power_norm'`` and ``'amplitude_norm'``.
    """
    return BASELINE_FUNCTIONS[metric](data, target_freq, cell_type, start, time_stop)


def get_phase(
    spikes,
    FREQ_HZ=10.0,
    T0_MS=1500.0,
    phase0=0.0,
    phase_orientation=0.0,
    T_STOP_MS=np.inf,
    ANALYSIS_START_MS=None,
):
    """Return spike phases using the same reference convention as stored features."""
    if ANALYSIS_START_MS is None:
        ANALYSIS_START_MS = T0_MS
    selected = spikes[
        (spikes.index > ANALYSIS_START_MS) & (spikes.index < T_STOP_MS)
    ]
    phase_rad = _feature_spike_phases(
        spikes,
        float(T_STOP_MS) * 1e-3,
        FREQ_HZ,
        phase0,
        phase_orientation,
        float(T0_MS) * 1e-3,
        float(ANALYSIS_START_MS) * 1e-3,
    ) % (2 * np.pi)
    return pd.DataFrame(
        {"cell_id": selected.to_numpy(), "phase_rad": phase_rad, "phase_deg": np.degrees(phase_rad)},
        index=selected.index,
    )


def row_phases(row, freq=None, cell_type=None, baseline=False, layers=None):
    """Spike phases of one data.csv row, using the same convention as the stored features.

    Stimulated rows use the window [delay, tstop] and the row's stim phase/orientation.
    ``baseline=True`` uses [BASELINE_START_MS, tstop] with zero phase (for unstimulated rows).
    ``freq`` defaults to the row's stimulation frequency.
    """
    row = row.iloc[0] if isinstance(row, pd.DataFrame) else row
    freq = row["stim_freq"] if freq is None else freq
    sim = load_simulation(row["source_file"])
    spikes = sim.spikes
    if cell_type not in (None, "ALL") or layers is not None:
        spikes = spikes[spikes.isin(cell_type_ids(sim.cells, cell_type, layers))]
    if baseline:
        start, phase0, orientation = BASELINE_START_MS, 0.0, 0.0
    else:
        start, phase0, orientation = row["delay"], row["stim_phase"], row["stim_orientation"]
    return get_phase(spikes, freq, start, phase0, orientation, row["tstop"])


def pooled_phases(rows, freq=None, cell_type=None, baseline=False, layers=None, tag_replicates=False):
    """Concatenate ``row_phases`` over several rows (e.g. seeds of one condition).

    ``tag_replicates=True`` makes ``cell_id`` a (replicate, node) tuple.
    """
    frames = []
    for replicate, (_, row) in enumerate(rows.iterrows()):
        phases = row_phases(row, freq, cell_type, baseline, layers)
        if tag_replicates:
            phases["node_id"] = phases["cell_id"]
            phases["cell_id"] = [(replicate, node) for node in phases["node_id"]]
            phases["sim_id"] = row["sim_id"]
        frames.append(phases)
    if not frames:
        return pd.DataFrame(columns=["cell_id", "phase_rad", "phase_deg"])
    return pd.concat(frames)


def phase_grid(runs, baseline, freqs, amps, cell_type=None):
    """Pooled phases for a frequency x amplitude grid, with the baseline as the first column.

    Returns ``grid[i][j]`` for ``freqs[i]`` and ``(0,) + amps`` column ``j``; missing conditions
    are empty DataFrames. Pass the result to ``plotting.plot_rose_grid``.
    """
    return [
        [pooled_phases(baseline, freq, cell_type, baseline=True)]
        + [pooled_phases(rows_at(runs, freq, amp), freq, cell_type) for amp in amps]
        for freq in freqs
    ]


def get_phases_from_h5(simulations, sim_name):
    """Load all spike phases for the simulation identified by ``sim_name``."""
    return row_phases(sim_row(simulations, sim_name))


def _kuiper_pvalue(lam, n_terms=100):
    """Return the asymptotic Q_K(lambda) for the Kuiper statistic."""
    j = np.arange(1, n_terms + 1)
    p_value = 2 * np.sum((4 * j ** 2 * lam ** 2 - 1) * np.exp(-2 * j ** 2 * lam ** 2))
    return float(np.clip(p_value, 0.0, 1.0))


def _prepare_spike_phases(phases):
    """Return finite spike phases wrapped to [0, 2*pi)."""
    phases = np.asarray(phases, dtype=float).ravel()
    phases = phases[np.isfinite(phases)]
    if phases.size < 2:
        raise ValueError("Kuiper's test requires at least two finite spike phases.")
    return np.mod(phases, 2 * np.pi)


def kuiper_one_sample(phases, n_terms=100):
    """Test spike phases against a circular-uniform distribution.

    The phases must be in radians. They are wrapped to [0, 2*pi), normalized
    to [0, 1), sorted, and compared with the uniform CDF. This is the
    one-sample Kuiper test (sometimes informally called the one-sided test).
    """
    phases = _prepare_spike_phases(phases)
    x = np.sort(phases / (2 * np.pi))
    n = x.size
    i = np.arange(1, n + 1)
    d_plus = np.max(i / n - x)
    d_minus = np.max(x - (i - 1) / n)
    statistic = d_plus + d_minus
    lam = statistic * (np.sqrt(n) + 0.155 + 0.24 / np.sqrt(n))
    return {
        "V": statistic,
        "D_plus": d_plus,
        "D_minus": d_minus,
        "p": _kuiper_pvalue(lam, n_terms),
        "n": n,
        "test": "one-sample uniformity",
        "method": "Stephens asymptotic approximation",
    }


def kuiper_two_sample(phases_a, phases_b, n_terms=100):
    """Test whether two samples of spike phases have the same distribution.

    Both phase arrays must be in radians. The test compares their empirical
    CDFs around the full circle and is invariant to the choice of phase origin.
    """
    a = np.sort(_prepare_spike_phases(phases_a) / (2 * np.pi))
    b = np.sort(_prepare_spike_phases(phases_b) / (2 * np.pi))
    grid = np.sort(np.unique(np.concatenate([a, b])))
    cdf_a = np.searchsorted(a, grid, side="right") / a.size
    cdf_b = np.searchsorted(b, grid, side="right") / b.size
    difference = cdf_a - cdf_b
    d_plus = np.max(difference)
    d_minus = np.max(-difference)
    statistic = d_plus + d_minus
    n_eff = a.size * b.size / (a.size + b.size)
    lam = statistic * (np.sqrt(n_eff) + 0.155 + 0.24 / np.sqrt(n_eff))
    return {
        "V": statistic,
        "D_plus": d_plus,
        "D_minus": d_minus,
        "p": _kuiper_pvalue(lam, n_terms),
        "n1": a.size,
        "n2": b.size,
        "test": "two-sample",
        "method": "Stephens asymptotic approximation",
    }


# --------------------------------------------------------------------------- #
# Cell-level phase preferences
# --------------------------------------------------------------------------- #
CELL_PHASE_COLUMNS = ["cell_id", "phase_rad", "phase_deg", "plv", "n_spikes"]


def _windowed_cell_spike_phases(row, freq, cell_type=None, baseline=False, layers=None,
                               use_analysis_window=False):
    """Spike phases of one row in (start, tstop), keeping the id of the cell that fired.

    Phase convention, as everywhere else in the project:
    ``2*pi*freq*(t - start) + stim_phase + stim_orientation``. Stimulated rows use
    start = ``delay``; ``baseline=True`` uses start = BASELINE_START_MS and zero phase/orientation.
    """
    sim = load_simulation(row["source_file"])
    spikes = sim.spikes
    if cell_type not in (None, "ALL") or layers is not None:
        spikes = spikes[spikes.isin(cell_type_ids(sim.cells, cell_type, layers))]
    if baseline:
        start, phase0, orientation = BASELINE_START_MS, 0.0, 0.0
    else:
        start, phase0, orientation = float(row["delay"]), float(row["stim_phase"]), float(row["stim_orientation"])
    times_ms = spikes.index.to_numpy(dtype=float)
    window_start = float(row["analysis_start"]) if use_analysis_window and not baseline else start
    window_stop = float(row["analysis_stop"]) if use_analysis_window and not baseline else float(row["tstop"])
    in_window = (times_ms > window_start) & (times_ms < window_stop)
    phases = 2 * np.pi * freq * (times_ms[in_window] - start) * 1e-3 + phase0 + orientation
    return pd.DataFrame({"cell_id": spikes.to_numpy()[in_window], "phase_rad": phases})


def cell_phases(rows, freq=None, cell_type=None, baseline=False, layers=None, min_spikes=1,
                use_analysis_window=False):
    """One preferred phase per model cell: the angle of the mean spike-phase vector.

    Cells with fewer than ``min_spikes`` spikes in the analysis window (in particular cells that
    do not spike at all) are omitted, so a histogram of ``phase_rad`` counts active cells.
    Several rows (e.g. seeds of one condition) are combined by cell id before averaging, so every
    model cell still contributes at most once. ``plv`` is each cell's own phase-locking value.
    ``use_analysis_window=True`` selects analysis_start < t < analysis_stop for
    stimulated runs, retaining delay as the phase reference (as in cross-frequency PLV).
    """
    rows = rows.to_frame().T if isinstance(rows, pd.Series) else rows
    frames = [
        _windowed_cell_spike_phases(row, row["stim_freq"] if freq is None else freq,
                                   cell_type, baseline, layers, use_analysis_window)
        for _, row in rows.iterrows()
    ]
    if not frames:
        return pd.DataFrame(columns=CELL_PHASE_COLUMNS)
    spikes = pd.concat(frames, ignore_index=True)
    spikes["cos"], spikes["sin"] = np.cos(spikes["phase_rad"]), np.sin(spikes["phase_rad"])
    cells = spikes.groupby("cell_id").agg(n_spikes=("phase_rad", "size"), cos=("cos", "mean"), sin=("sin", "mean"))
    cells = cells[cells["n_spikes"] >= min_spikes].reset_index()
    cells["phase_rad"] = np.mod(np.arctan2(cells["sin"], cells["cos"]), 2 * np.pi)
    cells["phase_deg"] = np.degrees(cells["phase_rad"])
    cells["plv"] = np.hypot(cells["cos"], cells["sin"])
    return cells[CELL_PHASE_COLUMNS]


def cell_phase_grid(runs, baseline, freqs, amps, cell_type=None, min_spikes=1):
    """``phase_grid`` with one preferred phase per active cell instead of one entry per spike."""
    return [
        [cell_phases(baseline, freq, cell_type, baseline=True, min_spikes=min_spikes)]
        + [cell_phases(rows_at(runs, freq, amp), freq, cell_type, min_spikes=min_spikes) for amp in amps]
        for freq in freqs
    ]


def paired_cell_phases(cells_a, cells_b):
    """Inner-join two ``cell_phases`` tables on ``cell_id`` (cells active in both conditions)."""
    return cells_a.merge(cells_b, on="cell_id", suffixes=("_a", "_b"))


def _moore_resultant(ranked_cos, ranked_sin, signs=None):
    """Moore's R' for rank-weighted unit vectors; ``signs`` (k x n) gives randomized statistics."""
    n = ranked_cos.size
    if signs is None:
        return np.hypot(ranked_cos.sum(), ranked_sin.sum()) / n ** 1.5
    return np.hypot(signs @ ranked_cos, signs @ ranked_sin) / n ** 1.5


def moore_paired_test(cells_a, cells_b, n_permutations=0, seed=None, batch_size=256):
    """Moore's paired test for dependent circular samples (Moore 1980; Zar, Biostatistical Analysis).

    Cells are matched on ``cell_id``. For each cell the change vector
    ``(cos b - cos a, sin b - sin a)`` is computed, its length is ranked and its direction is
    kept; ``R' = |sum(rank_i * u_i)| / n**1.5``. Under H0 (no systematic within-cell change)
    the change directions are symmetric, so

    * ``p_asymptotic`` uses ``12 * n**3 * R'**2 / (n (n+1) (2n+1)) ~ chi2(2)``;
    * ``p`` (if ``n_permutations > 0``) is a randomization test that swaps the two labels within
      random cells (flips change vectors), which is exact for any n.

    Cells whose preferred phase is identical in both conditions carry no direction and are dropped.
    """
    pairs = paired_cell_phases(cells_a, cells_b)
    dx = np.cos(pairs["phase_rad_b"]) - np.cos(pairs["phase_rad_a"])
    dy = np.sin(pairs["phase_rad_b"]) - np.sin(pairs["phase_rad_a"])
    length = np.hypot(dx, dy).to_numpy()
    keep = length > 1e-12
    n = int(keep.sum())
    result = {"n_pairs": len(pairs), "n_moore": n}
    if n < 2:
        return dict(result, R=np.nan, p=np.nan, p_asymptotic=np.nan, mean_shift_deg=np.nan)
    from scipy.stats import rankdata

    direction = np.arctan2(dy.to_numpy()[keep], dx.to_numpy()[keep])
    ranks = rankdata(length[keep])
    ranked_cos, ranked_sin = ranks * np.cos(direction), ranks * np.sin(direction)
    statistic = _moore_resultant(ranked_cos, ranked_sin)
    chi2 = 12 * n ** 3 * statistic ** 2 / (n * (n + 1) * (2 * n + 1))
    shift = np.exp(1j * (pairs["phase_rad_b"] - pairs["phase_rad_a"]).to_numpy()).mean()
    result.update(R=statistic, p_asymptotic=float(np.exp(-chi2 / 2)),
                  mean_shift_deg=float(np.degrees(np.angle(shift))))
    if n_permutations:
        rng = np.random.default_rng(seed)
        exceed, done = 0, 0
        while done < n_permutations:
            k = min(batch_size, n_permutations - done)
            signs = rng.choice((-1.0, 1.0), size=(k, n))
            exceed += int(np.sum(_moore_resultant(ranked_cos, ranked_sin, signs) >= statistic))
            done += k
        result["p"] = (exceed + 1) / (n_permutations + 1)
        result["method"] = f"randomization ({n_permutations} within-cell label swaps)"
    else:
        result["p"] = result["p_asymptotic"]
        result["method"] = "asymptotic chi2(2)"
    return result


def compare_cell_phases(cells_a, cells_b, paired=True, n_permutations=0, seed=None):
    """Per-cell Kuiper two-sample test, plus Moore's paired test on the cells active in both.

    Returns a flat dict (``kuiper_*`` and ``moore_*`` columns) for building result tables.
    The Kuiper test treats the two sets of preferred phases as independent samples; when the
    same cells appear in both conditions, Moore's paired test is the one that respects pairing.
    """
    result = {"n_cells_a": len(cells_a), "n_cells_b": len(cells_b)}
    if len(cells_a) > 1 and len(cells_b) > 1:
        kuiper = kuiper_two_sample(cells_a["phase_rad"], cells_b["phase_rad"])
        result.update(kuiper_V=kuiper["V"], kuiper_p=kuiper["p"])
    else:
        result.update(kuiper_V=np.nan, kuiper_p=np.nan)
    if paired:
        moore = moore_paired_test(cells_a, cells_b, n_permutations=n_permutations, seed=seed)
        result.update({f"moore_{key}": value for key, value in moore.items()})
    return result


def cross_frequency_moore_tests(phase_tables, pairs=((7, 10), (10, 15), (7, 15)),
                                alpha=0.05, n_permutations=19999, seed=20260921):
    """Compare evaluation frequencies at fixed state, stimulus frequency and amplitude.

    ``phase_tables[(state, stim_freq, amp, cell_type)]`` maps evaluation frequency
    to a ``cell_phases`` table. Moore pairs cells by ID. Holm correction is over
    the planned frequency pairs separately for each key, not across amplitudes.
    Missing/undefined tests retain their place in the family (p=1 internally),
    but are reported as untested with NaN adjusted p, never as nonsignificant data.
    These are circular preferred-phase tests, not tests of scalar PLV magnitude.
    """
    pairs = tuple(pairs)
    if not pairs or any(a == b for a, b in pairs):
        raise ValueError("Provide distinct evaluation-frequency pairs")
    if len({frozenset(pair) for pair in pairs}) != len(pairs):
        raise ValueError("Duplicate frequency pairs")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    records = []
    for (state, stim_freq, amp, cell_type), spectra in phase_tables.items():
        family = []
        for freq_a, freq_b in pairs:
            cells_a, cells_b = spectra.get(freq_a), spectra.get(freq_b)
            row = dict(cell_type=cell_type, stim_freq=stim_freq,
                       freq_a=freq_a, freq_b=freq_b, amp_a=amp, amp_b=amp,
                       connectivity_a=True, connectivity_b=True,
                       correlation_a=state, correlation_b=state,
                       n_cells_a=0 if cells_a is None else len(cells_a),
                       n_cells_b=0 if cells_b is None else len(cells_b),
                       moore_p=np.nan, moore_p_asymptotic=np.nan, moore_R=np.nan,
                       moore_n_pairs=0, moore_n_moore=0, moore_mean_shift_deg=np.nan,
                       tested=False, status="missing frequency/condition")
            if cells_a is not None and cells_b is not None:
                for cells in (cells_a, cells_b):
                    if cells["cell_id"].duplicated().any():
                        raise ValueError("Each cell must occur at most once per condition")
                    if not np.isfinite(cells["phase_rad"].to_numpy(float)).all():
                        raise ValueError("Preferred phases must be finite")
                result = moore_paired_test(cells_a, cells_b,
                                           n_permutations=n_permutations, seed=seed)
                row.update({f"moore_{key}": value for key, value in result.items()})
                row["tested"] = bool(np.isfinite(row["moore_p"]))
                row["status"] = "tested" if row["tested"] else "fewer than two nonzero paired changes"
            family.append(row)
        adjusted = adjust_pvalues([r["moore_p"] if r["tested"] else 1 for r in family], method="holm")
        for row, p_adjusted in zip(family, adjusted):
            row.update(moore_p_holm=float(p_adjusted) if row["tested"] else np.nan,
                       moore_p_n_tests=len(pairs),
                       moore_p_significant=bool(row["tested"] and p_adjusted <= alpha))
        records.extend(family)
    return pd.DataFrame.from_records(records)


def compare_phase_changes(base_a, stim_a, base_b, stim_b):
    """Do two disjoint cell groups (e.g. pyramidal cells and interneurons) change their preferred
    phases differently under stimulation?

    The groups are different cells, so they cannot be paired with each other. Within each group,
    every cell active in both its baseline and its stimulated ``cell_phases`` table contributes its
    change vector ``exp(i*phase_stim) - exp(i*phase_base)`` (the vector Moore's paired test uses).
    The mean change vector of a group is how far stimulation moved the group's mean phase vector.
    The two mean change vectors are compared with a two-sample Wald test with unequal covariances,
    ``D' (S_a/n_a + S_b/n_b)^-1 D ~ chi2(2)``, accurate for groups of hundreds of cells or more. A
    difference in the size and a difference in the direction of the shift both count.

    Returns, per group (``_a``, ``_b``): ``n_pairs``, the mean resultant length of the preferred
    phases of those cells at baseline and during stimulation (``R_base``, ``R_stim``), and the length
    and direction of the mean change vector (``change``, ``change_deg``); then ``change_difference``
    (``change_a - change_b``), ``wald_chi2`` and ``wald_p``.
    """
    result, means, covariances = {}, [], []
    for name, base, stim in (("a", base_a, stim_a), ("b", base_b, stim_b)):
        pairs = paired_cell_phases(base, stim)
        u_base = np.exp(1j * pairs["phase_rad_a"].to_numpy(dtype=float))
        u_stim = np.exp(1j * pairs["phase_rad_b"].to_numpy(dtype=float))
        change = u_stim - u_base
        n = change.size
        result[f"n_pairs_{name}"] = n
        result[f"R_base_{name}"] = float(np.abs(u_base.mean())) if n else np.nan
        result[f"R_stim_{name}"] = float(np.abs(u_stim.mean())) if n else np.nan
        result[f"change_{name}"] = float(np.abs(change.mean())) if n else np.nan
        result[f"change_deg_{name}"] = float(np.degrees(np.angle(change.mean()))) if n else np.nan
        vectors = np.column_stack([change.real, change.imag])
        means.append(vectors.mean(axis=0) if n else np.full(2, np.nan))
        covariances.append(np.cov(vectors, rowvar=False) / n if n > 2 else np.full((2, 2), np.nan))
    result["change_difference"] = result["change_a"] - result["change_b"]
    if min(result["n_pairs_a"], result["n_pairs_b"]) < 3:
        return dict(result, wald_chi2=np.nan, wald_p=np.nan)
    difference = means[0] - means[1]
    chi2 = float(difference @ np.linalg.solve(covariances[0] + covariances[1], difference))
    return dict(result, wald_chi2=chi2, wald_p=float(np.exp(-chi2 / 2)))


def sim_label(rows):
    """Identifier of the simulation(s) behind one condition: sorted ``sim_id`` values joined by ';'."""
    rows = rows.to_frame().T if isinstance(rows, pd.Series) else rows
    return ";".join(sorted(rows["sim_id"].astype(str)))


def cell_phase_contrasts(side_a, side_b, baseline_a=None, baseline_b=None, labels=None, **test_kwargs):
    """Compare two conditions and each of them with its own baseline (Kuiper + paired Moore).

    Each side is ``(condition, cells)``: ``condition`` is a dict describing the condition with the
    same keys on every side, e.g. ``{"freq": 10, "amp": 2, "connectivity": True, "correlation": "low"}``
    (``amp=0`` marks a baseline, ``freq`` is the phase-reference frequency), and ``cells`` is a
    ``cell_phases`` table.

    Returns up to three rows with identical columns: A vs B, ``baseline_a`` vs A and ``baseline_b``
    vs B (baseline rows only if given). The baseline is always the first sample, so
    ``moore_mean_shift_deg`` is the phase change caused by stimulation. Columns: ``labels`` (e.g.
    ``{"cell_type": "PYR"}``), then ``<key>_a``, ``<key>_b`` for each condition key in order
    (``freq_a, freq_b, amp_a, amp_b, ...``), then the ``compare_cell_phases`` results.
    """
    def row(first, second):
        (condition_1, cells_1), (condition_2, cells_2) = first, second
        metadata = {}
        for key in condition_1:
            metadata[f"{key}_a"], metadata[f"{key}_b"] = condition_1[key], condition_2[key]
        return {**(labels or {}), **metadata, **compare_cell_phases(cells_1, cells_2, **test_kwargs)}

    rows = [row(side_a, side_b)]
    if baseline_a is not None:
        rows.append(row(baseline_a, side_a))
    if baseline_b is not None:
        rows.append(row(baseline_b, side_b))
    return rows


def adjust_pvalues(p_values, method="bonferroni"):
    """Family-wise error correction of a vector of p-values ('bonferroni' or 'holm')."""
    p_values = np.asarray(p_values, dtype=float)
    m = p_values.size
    if m == 0:
        return p_values
    if method == "bonferroni":
        return np.minimum(p_values * m, 1.0)
    if method == "holm":
        order = np.argsort(p_values)
        stepped = np.maximum.accumulate((m - np.arange(m)) * p_values[order])
        adjusted = np.empty(m)
        adjusted[order] = np.minimum(stepped, 1.0)
        return adjusted
    raise ValueError(f"Unknown correction method {method!r}")


def correct_pvalues(tests, p_columns, *, key_cols, method="holm", alpha=0.05):
    method = method.lower()
    if method not in {"bonferroni", "holm"}:
        raise ValueError("method must be 'bonferroni' or 'holm'")

    key_cols = list(key_cols)
    p_columns = list(p_columns)

    # Count each test once, even if it appears in multiple tables.
    unique = tests.drop_duplicates(subset=key_cols).copy()

    for p_col in p_columns:
        # Repeated test keys must have the same p-value.
        inconsistent = (
            tests.groupby(key_cols, dropna=False)[p_col]
            .nunique(dropna=False)
            .gt(1)
            .any()
        )
        if inconsistent:
            raise ValueError(f"Duplicate test keys have different {p_col} values")

        raw = unique[p_col].to_numpy(dtype=float)
        if not np.isfinite(raw).all() or ((raw < 0) | (raw > 1)).any():
            raise ValueError(f"{p_col} must contain finite p-values in [0, 1]")

        n_tests = raw.size
        if method == "bonferroni":
            adjusted = np.minimum(raw * n_tests, 1.0)
        else:  # Holm–Bonferroni
            order = np.argsort(raw, kind="mergesort")
            ranked_adjusted = np.maximum.accumulate(
                raw[order] * np.arange(n_tests, 0, -1)
            )
            adjusted = np.empty(n_tests, dtype=float)
            adjusted[order] = np.minimum(ranked_adjusted, 1.0)

        corrected_col = f"{p_col}_{method}"
        unique[corrected_col] = adjusted
        unique[f"{p_col}_n_tests"] = n_tests
        unique[f"{p_col}_significant"] = unique[corrected_col].le(alpha)

    correction_cols = [
        col
        for p_col in p_columns
        for col in (
            f"{p_col}_{method}",
            f"{p_col}_n_tests",
            f"{p_col}_significant",
        )
    ]
    return tests.merge(
        unique[key_cols + correction_cols],
        on=key_cols,
        how="left",
        sort=False,
        validate="many_to_one",
    )


def fixed_sequence_pvalues(tests, p_columns, *, chain_cols, order_col, family_cols=(), descending=True,
                           alpha=0.05, gate=None):
    """Fixed-sequence (hierarchical step-down) testing, alpha split over the chains of each family.

    ``family_cols`` splits the tests into separate families (e.g. one per cell type), each with its
    own familywise error rate ``alpha``. Inside a family, every combination of ``chain_cols`` is one
    chain (e.g. one dose-response curve) tested in a pre-specified order of ``order_col`` (highest
    level first when ``descending``): the first step at ``alpha / n_chains``, every later step only
    if all earlier steps of its chain were rejected, so the chain stops at the first non-significant
    test. Within a chain no alpha is split (closed testing); across the chains of a family it is
    split by Bonferroni.

    ``gate`` optionally names a boolean column: a step whose gate is False is not tested and its
    chain stops there. This can only remove rejections of the ungated procedure, so the familywise
    error rate stays controlled whatever the gate is computed from.

    For every ``p_col`` adds ``<p_col>_n_chains`` (chains in the family), ``<p_col>_tested`` (the
    step was reached and passed the gate), ``<p_col>_fixed_seq`` (adjusted p,
    ``min(1, n_chains * max(p of this and all earlier steps))``, NaN for steps that were not tested)
    and ``<p_col>_significant`` (tested and adjusted p <= alpha). Missing p-values count as 1.
    Rows keep their original order.
    """
    family_cols, chain_cols, p_columns = list(family_cols), list(chain_cols), list(p_columns)
    if not chain_cols:
        raise ValueError("chain_cols must name at least one column")
    if not tests.index.is_unique:
        raise ValueError("tests must have a unique index")
    if tests.duplicated(subset=family_cols + chain_cols + [order_col]).any():
        raise ValueError(f"Each chain must have at most one test per {order_col!r} level")
    passes = (tests[gate].astype("boolean").fillna(False).astype(bool) if gate is not None
              else pd.Series(True, index=tests.index))
    families = ([group for _, group in tests.groupby(family_cols, dropna=False, sort=False)]
                if family_cols else [tests])

    result = tests.copy()
    for p_col in p_columns:
        n_chains = pd.Series(0, index=tests.index)
        tested = pd.Series(False, index=tests.index)
        adjusted = pd.Series(np.nan, index=tests.index)
        for family in families:
            chains = family.groupby(chain_cols, dropna=False, sort=False)
            for _, chain in chains:
                running_max, reached = 0.0, True
                for index, p in (chain.sort_values(order_col, ascending=not descending, kind="mergesort")[p_col]
                                 .astype(float).fillna(1.0).items()):
                    running_max = max(running_max, p)
                    n_chains[index] = chains.ngroups
                    tested[index] = reached and passes[index]
                    if tested[index]:
                        adjusted[index] = min(1.0, chains.ngroups * running_max)
                    reached = tested[index] and adjusted[index] <= alpha
        result[f"{p_col}_n_chains"] = n_chains
        result[f"{p_col}_tested"] = tested
        result[f"{p_col}_fixed_seq"] = adjusted
        result[f"{p_col}_significant"] = tested & adjusted.le(alpha)
    return result


def minimum_effective_levels(tests, significant_col, *, group_cols, order_col):
    """Lowest ``order_col`` level declared significant in each group (NaN if none).

    With ``fixed_sequence_pvalues`` run from the highest level down, the significant levels of a
    chain are contiguous from the top, so this is the minimum effective dose / amplitude.
    """
    group_cols = list(group_cols)
    significant = tests.loc[tests[significant_col].astype(bool)]
    lowest = significant.groupby(group_cols, dropna=False)[order_col].min()
    groups = tests[group_cols].drop_duplicates()
    return groups.merge(lowest.rename(f"min_effective_{order_col}").reset_index(),
                        on=group_cols, how="left").reset_index(drop=True)


PAPER_P_THRESHOLDS = (0.05, 0.005, 0.0005, 0.00005, 0.000005)


def paper_p_value(p, thresholds=PAPER_P_THRESHOLDS, ns_label="n.s."):
    """Format a (corrected) p-value for a paper: the smallest threshold it is below.

    ``0.02 -> "< 0.05"``, ``0.003 -> "< 0.005"``, ``1e-12 -> "< 0.000005"`` (the last threshold is
    the floor). Values at or above the largest threshold give ``ns_label``; missing values give "".
    """
    if p is None or pd.isna(p):
        return ""
    thresholds = sorted(thresholds, reverse=True)
    if p >= thresholds[0]:
        return ns_label
    below = [threshold for threshold in thresholds if p < threshold]
    return f"< {below[-1]:.10f}".rstrip("0")


def circ_stats(phases):
    """Return mean circular phase and PLV for a vector of phases."""
    mean_vector = np.exp(1j * phases).mean()
    return pd.Series({"mean_phase": np.angle(mean_vector) % (2 * np.pi), "plv": np.abs(mean_vector)})


def average_metrics_by_amp(rows, stim_amps=(0.25, 1, 4)):
    """Average stored PLV magnitudes and PPC over simulation-table rows.

    Pass rows from ``load_simulation_table`` for one stimulation frequency.
    Uses each row's saved analysis window; no simulation or circuit is opened.
    PLV is the mean of per-run magnitudes, not the magnitude of a mean vector.
    Requested amplitudes without runs retain NaN metrics and a zero seed count.
    """
    if not isinstance(rows, pd.DataFrame):
        raise TypeError("Pass simulation-table rows, not source_file paths.")
    stim_amps = list(stim_amps)
    selected = rows[rows["stim_amp"].isin(stim_amps)]
    if selected["stim_freq"].nunique() > 1:
        raise ValueError("Select one stimulation frequency before averaging amplitudes.")
    result = {"stim_amp": stim_amps, "plv": [], "ppc": [], "n_seeds": []}
    for amp in stim_amps:
        runs = selected[selected["stim_amp"] == amp]
        result["plv"].append(runs["plv"].astype(complex).abs().mean())
        result["ppc"].append(runs["ppc"].mean())
        result["n_seeds"].append(len(runs))
    return result


def remove_unnamed_columns(df):
    """Drop CSV index columns introduced by previous exports."""
    return df.loc[:, ~df.columns.str.contains("^Unnamed")]


STD_COLUMNS = tuple(
    f"{feature}{suffix}"
    for feature in ("abs_plv", "ppc")
    for suffix in ("", "_PYR", "_INT_INH", "_INT_EXC")
)


def merge_and_average(data, group_cols=("stim_amp", "stim_freq"), std_cols=STD_COLUMNS):
    """Merge seed replicates while preserving identifying metadata.

    Numeric feature columns are averaged over the seeds of each group. For every column in
    ``std_cols`` that is present and numeric, a ``<col>_std`` column holds the sample standard
    deviation across seeds (NaN for a single seed), and ``n_seeds`` counts the merged runs.
    These are what the error bars in ``plotting.plot_metric_vs_amplitude`` use; by default they
    cover the PLV magnitude and the PPC, for all cells and per cell type.
    """
    metadata_cols = list(data.columns[:data.columns.get_loc("source_file") + 1])
    metadata_cols += ["deployment", "neurodamus"]
    metadata_cols = list(dict.fromkeys(col for col in metadata_cols if col in data.columns))
    mean_cols = [
        col for col in data.select_dtypes(include="number").columns
        if col not in metadata_cols and col not in group_cols
    ]
    count_col = "sim_id" if "sim_id" in data.columns else mean_cols[0]
    return data.groupby(list(group_cols), as_index=False).agg(
        **{"n_seeds": (count_col, "size")},
        **{col: (col, "first") for col in metadata_cols if col not in group_cols},
        **{col: (col, "mean") for col in mean_cols},
        **{f"{col}_std": (col, lambda values: values.std(ddof=1))
           for col in std_cols if col in mean_cols},
    )

def compute_mean_firing_rate(spikeData, tstop, tstart):
    """Return {node_id: firing rate (Hz) in [tstart, tstop]} for every node that ever spiked.

    ``spikeData`` is a Series indexed by spike time (ms) whose values are node ids.
    """
    window = spikeData[(spikeData.index >= tstart) & (spikeData.index <= tstop)]
    counts = window.value_counts().reindex(np.unique(spikeData.values), fill_value=0)
    if tstop <= tstart:
        return {node_id: 0 for node_id in counts.index}
    return (counts / ((tstop - tstart) * 1e-3)).to_dict()


def population_firing_rates(rows, start_ms=None, bin_width=5):
    """Mean smoothed population firing rate (Hz/neuron) per row, plus per-cell rates.

    The window is [start_ms, tstop]; ``start_ms`` defaults to each row's ``delay``.
    """
    records = []
    for _, row in rows.iterrows():
        sim = load_simulation(row["source_file"])
        start = row["delay"] if start_ms is None else start_ms
        stop = row["tstop"]
        window = sim.spikes[(sim.spikes.index >= start) & (sim.spikes.index <= stop)]
        rate, _bins = get_firingrate_hist(sim.simulation, window, bin_width, sim.cells.shape[0],
                                          delay=start, time_stop=stop, smoothed=True)
        records.append(dict(
            sim_id=row["sim_id"], stim_freq=row["stim_freq"], stim_amp=row["stim_amp"],
            firing_rate_hz=float(np.mean(rate)),
            mean_firing_rates=compute_mean_firing_rate(sim.spikes, stop, start),
        ))
    return pd.DataFrame(records)


def per_cell_firing_rates(rows, cell_type=None, start_ms=None, layers=None):
    """Firing rate (Hz) of every simulated cell, one column per row of ``rows``.

    The window is [start_ms, tstop] and ``start_ms`` defaults to each row's ``delay``. Cells that do
    not spike in the window are kept with a rate of 0, so the columns are aligned cell by cell
    across simulations of the same circuit (the index is the node id).
    """
    columns = {}
    for _, row in rows.iterrows():
        sim = load_simulation(row["source_file"])
        cells = sim.cells if cell_type in (None, "ALL") and layers is None else \
            sim.cells.loc[cell_type_ids(sim.cells, cell_type, layers)]
        start = float(row["delay"] if start_ms is None else start_ms)
        stop = float(row["tstop"])
        spikes = sim.spikes[(sim.spikes.index >= start) & (sim.spikes.index <= stop)]
        counts = spikes[spikes.isin(cells.index)].value_counts()
        columns[row["sim_id"]] = counts.reindex(cells.index, fill_value=0) / ((stop - start) * 1e-3)
    return pd.DataFrame(columns)


def _greenhouse_geisser(matrix):
    """Greenhouse-Geisser epsilon for a cells x conditions matrix (sphericity correction)."""
    k = matrix.shape[1]
    covariance = np.cov(matrix, rowvar=False)
    diagonal_mean = np.mean(np.diag(covariance))
    row_means = covariance.mean(axis=1)
    grand_mean = covariance.mean()
    numerator = k ** 2 * (diagonal_mean - grand_mean) ** 2
    denominator = (k - 1) * ((covariance ** 2).sum() - 2 * k * (row_means ** 2).sum() + k ** 2 * grand_mean ** 2)
    if denominator <= 0:
        return 1.0
    return float(np.clip(numerator / denominator, 1 / (k - 1), 1.0))


def repeated_measures_anova(matrix, levels):
    """One-way repeated-measures ANOVA and its linear trend, for a cells x conditions matrix.

    ``matrix`` holds one row per cell (the matched unit) and one column per condition; ``levels``
    are the numeric values of those conditions (e.g. stimulation amplitudes), used for the linear
    contrast. Returns the F test over conditions with the Greenhouse-Geisser sphericity correction,
    its effect size, and the linear trend across ``levels`` (slope per unit of ``levels``).
    """
    matrix = np.asarray(matrix, dtype=float)
    matrix = matrix[np.isfinite(matrix).all(axis=1)]
    n, k = matrix.shape
    levels = np.asarray(levels, dtype=float)
    if n < 2 or k < 2:
        return {"n_cells": n, "n_conditions": k}

    grand_mean = matrix.mean()
    ss_conditions = n * ((matrix.mean(axis=0) - grand_mean) ** 2).sum()
    ss_cells = k * ((matrix.mean(axis=1) - grand_mean) ** 2).sum()
    ss_error = ((matrix - grand_mean) ** 2).sum() - ss_conditions - ss_cells
    df_conditions, df_error = k - 1, (k - 1) * (n - 1)
    f_statistic = (ss_conditions / df_conditions) / (ss_error / df_error)
    epsilon = _greenhouse_geisser(matrix)

    # Linear contrast across the condition levels: one value per cell, tested against zero.
    contrast = levels - levels.mean()
    per_cell_slope = matrix @ contrast / (contrast ** 2).sum()
    slope_test = stats.ttest_1samp(per_cell_slope, 0.0)

    return {
        "n_cells": n,
        "n_conditions": k,
        "anova_F": float(f_statistic),
        "anova_df1": float(df_conditions * epsilon),
        "anova_df2": float(df_error * epsilon),
        "anova_p": float(stats.f.sf(f_statistic, df_conditions * epsilon, df_error * epsilon)),
        "anova_partial_eta_sq": float(ss_conditions / (ss_conditions + ss_error)),
        "greenhouse_geisser_epsilon": epsilon,
        "trend_slope": float(per_cell_slope.mean()),
        "trend_slope_median": float(np.median(per_cell_slope)),
        "trend_t": float(slope_test.statistic),
        "trend_p": float(slope_test.pvalue),
        "trend_cohens_d": float(per_cell_slope.mean() / per_cell_slope.std(ddof=1)) if per_cell_slope.std(ddof=1) else np.nan,
    }


def firing_rate_amplitude_trend(runs, freq, amps, baseline=None, cell_type=None, start_ms=None,
                                include_baseline=True):
    """Does the firing rate change with stimulation amplitude at one frequency?

    Cells are the matched units: every cell contributes its firing rate at each amplitude (seeds of
    one amplitude are averaged), so the design is repeated measures. Reports

    * ``anova_*``: one-way repeated-measures ANOVA over the amplitude levels, Greenhouse-Geisser
      corrected, with partial eta squared;
    * ``trend_*``: the linear contrast across amplitudes, i.e. the per-cell slope in Hz per V/m
      (``trend_slope``), tested against zero, with Cohen's d;
    * ``friedman_*`` / ``kendall_w`` / ``page_*``: nonparametric equivalents, Page's L testing
      specifically for a monotone increase with amplitude;
    * ``run_slope`` / ``run_p``: the same slope with whole simulations as replicates, which is the
      only inference that does not rely on cells being matched units of one network;
    * ``mean_rate_<amp>``: the mean firing rate at each level, and ``baseline_rate_hz``.

    ``include_baseline=True`` adds the unstimulated ``baseline`` run as amplitude 0.
    """
    levels, matrices, run_rates, run_amps = [], [], [], []
    if include_baseline and baseline is not None:
        levels.append(0.0)
        matrices.append(per_cell_firing_rates(baseline, cell_type, start_ms=BASELINE_START_MS).mean(axis=1))
    for amp in amps:
        rows = rows_at(runs, freq, amp)
        if rows.empty:
            continue
        rates = per_cell_firing_rates(rows, cell_type, start_ms=start_ms)
        levels.append(float(amp))
        matrices.append(rates.mean(axis=1))
        run_rates.extend(rates.mean(axis=0).tolist())
        run_amps.extend([float(amp)] * rates.shape[1])
    if len(levels) < 2:
        return {"n_conditions": len(levels)}

    matrix = pd.concat(matrices, axis=1, join="inner")
    matrix.columns = levels
    values = matrix.to_numpy()

    result = {"cell_type": cell_type or "ALL", **repeated_measures_anova(values, levels)}
    friedman = stats.friedmanchisquare(*values.T)
    result["friedman_chi2"], result["friedman_p"] = float(friedman.statistic), float(friedman.pvalue)
    result["kendall_w"] = float(friedman.statistic / (values.shape[0] * (len(levels) - 1)))
    page = stats.page_trend_test(values, ranked=False)
    result["page_L"], result["page_p"] = float(page.statistic), float(page.pvalue)

    if len(set(run_amps)) > 1:
        run_fit = stats.linregress(run_amps, run_rates)
        result.update(n_runs=len(run_rates), run_slope=float(run_fit.slope), run_p=float(run_fit.pvalue),
                      run_r_squared=float(run_fit.rvalue ** 2))
    else:
        result.update(n_runs=len(run_rates), run_slope=np.nan, run_p=np.nan, run_r_squared=np.nan)

    result.update({f"mean_rate_{level:g}": float(column.mean()) for level, column in matrix.items()})
    if baseline is not None:
        baseline_rate = per_cell_firing_rates(baseline, cell_type, start_ms=BASELINE_START_MS).mean(axis=1).mean()
        result["baseline_rate_hz"] = float(baseline_rate)
        result["trend_percent_per_Vm"] = float(100 * result["trend_slope"] / baseline_rate) if baseline_rate else np.nan
    return result


def synaptic_delay_tables(circuit_config, base_target="hex0",
                          connectome="S1nonbarrel_neurons__S1nonbarrel_neurons__chemical"):
    """Mean synaptic delay by pre/post layer and synapse class (requires conntility).

    Returns a namespace with ``table`` (grouped mean/std/count) and ``pivot`` (pre x post mean delay).
    """
    import bluepysnap as bp
    from conntility import ConnectivityMatrix

    load_cfg = {"loading": {"properties": ["x", "y", "z", "mtype", "etype", "layer", "synapse_class"],
                            "base_target": base_target}}
    matrix = ConnectivityMatrix.from_bluepy(bp.Circuit(circuit_config), load_cfg, connectome=connectome,
                                            edge_property="delay", agg_func="mean")
    edges = matrix.matrix.tocoo()  # rows: presynaptic, cols: postsynaptic, data: mean delay
    synapse_class = matrix.vertices["synapse_class"].values
    layer = matrix.vertices["layer"].values.astype(str)
    edge_table = pd.DataFrame({
        "pre_layer": layer[edges.row], "pre_sc": synapse_class[edges.row],
        "post_layer": layer[edges.col], "post_sc": synapse_class[edges.col],
        "delay": edges.data,
    })
    table = edge_table.groupby(["pre_layer", "pre_sc", "post_layer", "post_sc"])["delay"].agg(["mean", "std", "count"])
    edge_table["pre"] = edge_table["pre_layer"] + "_" + edge_table["pre_sc"]
    edge_table["post"] = edge_table["post_layer"] + "_" + edge_table["post_sc"]
    pivot = edge_table.pivot_table(index="pre", columns="post", values="delay", aggfunc="mean")
    return SimpleNamespace(table=table, pivot=pivot)


FEATURE_BIN_MS = 5.0
FEATURE_SMOOTH_SIGMA_MS = 10.0
FEATURE_BASELINE_MS = 1000.0
FEATURE_MAIN_AXIS = (0.63254108, -0.2055676, 0.74674878)
FEATURE_NAMES = (
    "plv",
    "ppc",
    "deltafr",
    "complex",
    "amplitude_norm",
    "power_norm",
    "total_power",
)
FEATURE_GROUPS = ("ALL", "PYR", "INT_INH", "INT_EXC")
FEATURE_COLUMNS = tuple(
    feature if group == "ALL" else f"{feature}_{group}"
    for group in FEATURE_GROUPS
    for feature in FEATURE_NAMES
)


def feature_column_name(feature: str, group: str) -> str:
    """Return the CSV column name for a feature and cell group."""
    return feature if group == "ALL" else f"{feature}_{group}"


def empty_feature_values() -> dict[str, Any]:
    """Return a feature record with every project feature set to ``None``."""
    return {
        feature_column_name(feature, group): None
        for group in FEATURE_GROUPS
        for feature in FEATURE_NAMES
    }


def field_phase_orientation(field) -> float:
    """Phase offset of an electric field: ``field["phase_orientation"]`` if present, else the angle
    between the field vector and the cortical main axis (pi when the field opposes the axis)."""
    if "phase_orientation" in field:
        return float(field["phase_orientation"])
    vector = (field.get("Ex", 0), field.get("Ey", 0), field.get("Ez", 0))
    return angle_between_3d(FEATURE_MAIN_AXIS, vector)


def angle_between_3d(a, b) -> float:
    """Return the angle between two 3D vectors in radians."""
    dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    magnitude_a = math.sqrt(a[0] ** 2 + a[1] ** 2 + a[2] ** 2)
    magnitude_b = math.sqrt(b[0] ** 2 + b[1] ** 2 + b[2] ** 2)
    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0
    cosine = max(-1.0, min(1.0, dot / (magnitude_a * magnitude_b)))
    return math.acos(cosine)


def _finite_or_none(value: Any) -> Any:
    """Convert NumPy scalars and non-finite feature values for CSV output."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, complex):
        if np.isnan(value.real) or np.isnan(value.imag):
            return None
        return value
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0 or np.isnan(denominator):
        return np.nan
    return numerator / denominator


def _feature_spike_phases(
    spike_data,
    t_stop_s: float,
    freq: float,
    phase0: float = 0.0,
    phase_orientation: float = 0.0,
    stim_start_s: float = 0.0,
    delay_s: float = 0.0,
) -> np.ndarray:
    """Return spike phases in the project convention.

    ``phi = 2*pi*freq*(t - stim_start) + phase0 + phase_orientation``, with spikes taken from
    ``delay_s < t < t_stop_s``. ``phase_orientation`` is the angle between the field vector and the
    cortical axis, so phase 0 is the peak of the field component that drives the column (the same
    waveform ``plotting.define_stimulation_function`` draws). Every phase in the project uses this
    convention, so the stored features, ``get_phase`` and ``cell_phases`` all give the same angle.
    """
    spike_times_s = np.asarray(spike_data.index.tolist(), dtype=float) * 1e-3
    spike_times_s = spike_times_s[spike_times_s > delay_s]
    spike_times_s = spike_times_s[spike_times_s < t_stop_s]
    return 2 * np.pi * freq * (spike_times_s - stim_start_s) + phase0 + phase_orientation


def _feature_ppc(
    spike_data,
    t_stop_s: float,
    freq: float,
    phase0: float = 0.0,
    phase_orientation: float = 0.0,
    stim_start_s: float = 0.0,
    delay_s: float = 0.0,
) -> float:
    phases = _feature_spike_phases(
        spike_data,
        t_stop_s,
        freq,
        phase0,
        phase_orientation,
        stim_start_s,
        delay_s,
    )
    n_spikes = len(phases)
    if n_spikes < 2:
        return np.nan
    phase_sum = np.sum(np.exp(1j * phases))
    return float((np.abs(phase_sum) ** 2 - n_spikes) / (n_spikes * (n_spikes - 1)))


def _feature_plv(
    spike_data,
    t_stop_s: float,
    freq: float,
    phase0: float = 0.0,
    phase_orientation: float = 0.0,
    stim_start_s: float = 0.0,
    delay_s: float = 0.0,
) -> float:
    """PLV phase 0° = positive peak of the applied field, 180° = negative peak. Confirmed numerically: spikes placed at the field maxima give a PLV angle of 0.00°, spikes at the minima give 180.00°.
90° is a quarter cycle after the peak, i.e. where the field is falling through zero."""
    phases = _feature_spike_phases(
        spike_data,
        t_stop_s,
        freq,
        phase0,
        phase_orientation,
        stim_start_s,
        delay_s,
    )
    if len(phases) == 0:
        return np.nan
    z = np.exp(1j * phases).mean()
    print(float(np.angle(z)))
    return z


def _smoothed_firing_rate(
    simulation_data,
    spike_data,
    n_neurons: int,
    delay: float,
    time_stop: float,
    bin_width: float,
) -> np.ndarray:
    """Return the smoothed population firing rate used for CSV features."""
    firing_rate, _bins = get_firingrate_hist(
        simulation_data,
        spike_data,
        bin_width=bin_width,
        n_neurons=n_neurons,
        delay=delay,
        time_stop=time_stop,
    )
    firing_rate = np.asarray(firing_rate, dtype=float)
    if firing_rate.size == 0:
        return firing_rate
    return gaussian_filter1d(firing_rate, sigma=FEATURE_SMOOTH_SIGMA_MS / bin_width)


def _feature_fourier(hist: np.ndarray, sample_period_s: float):
    """Return the historical Fourier scaling used by the feature CSV."""
    coefficients = np.fft.fft(hist)
    frequencies = np.fft.fftfreq(hist.shape[0], sample_period_s)
    positive = slice(0, hist.shape[0] // 2)
    amplitude = np.abs(coefficients[positive]) / hist.shape[0]
    return frequencies[positive], amplitude, coefficients[positive]


def _group_feature_values(
    simulation_data,
    spike_data,
    n_neurons: int,
    analysis_start: float,
    analysis_stop: float,
    stim_freq: float,
    stim_phase: float,
    stim_orientation: float,
    bin_width: float,
    true_delay: float | None,
) -> dict[str, Any]:
    """Calculate the seven stored features for one cell group."""
    if n_neurons <= 0:
        return {feature: None for feature in FEATURE_NAMES}

    if not np.isfinite(stim_freq) or stim_freq < 0:
        raise ValueError("stim_freq must be finite and nonnegative")

    start_s = float(analysis_start) * 1e-3
    stop_s = float(analysis_stop) * 1e-3
    duration_s = stop_s - start_s
    if not np.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("Analysis duration must be finite and positive")

    phase_start_s = (
        start_s if true_delay is None else float(true_delay) * 1e-3
    )

    # Reuse the existing helper so spike selection and phase referencing are
    # identical to the PLV calculation:
    #
    # phi_k = 2*pi*f*(t_k - phase_start_s) + stim_phase + stim_orientation
    #
    # The helper selects analysis_start < spike_time < analysis_stop.
    phases = _feature_spike_phases(
        spike_data,
        t_stop_s=stop_s,
        freq=stim_freq,
        phase0=stim_phase,
        phase_orientation=stim_orientation,
        stim_start_s=phase_start_s,
        delay_s=start_s,
    )
    phases = np.asarray(phases, dtype=float)
    if not np.all(np.isfinite(phases)):
        raise ValueError("Spike phases must be finite")

    n_spikes = phases.size
    phase_sum = np.exp(1j * phases).sum()

    # For r_spike(t) = sum_k delta(t-t_k) / n_neurons:
    #
    # C_plus = (1/T) integral r_spike(t) * exp(+i*phi(t)) dt
    #        = sum_k exp(+i*phi_k) / (n_neurons*T)
    #        = mean_rate * Z_plus.
    #
    # Hence angle(C_plus) equals the rose-vector angle whenever nonzero.
    # The conventional negative-exponent coefficient is conj(C_plus).
    # No additional onset correction is needed: phases already contain it.
    mean_rate = n_spikes / (n_neurons * duration_s)
    spike_coefficient = complex(
        phase_sum / (n_neurons * duration_s)
    )

    baseline_anchor = analysis_start if true_delay is None else true_delay
    firing_rate = _smoothed_firing_rate(
        simulation_data,
        spike_data,
        n_neurons,
        analysis_start,
        analysis_stop,
        bin_width,
    )
    baseline_rate = _smoothed_firing_rate(
        simulation_data,
        spike_data,
        n_neurons,
        baseline_anchor - FEATURE_BASELINE_MS,
        baseline_anchor,
        bin_width,
    )
    if firing_rate.size == 0:
        return {feature: None for feature in FEATURE_NAMES}

    frequencies, amplitude, coefficients = _feature_fourier(
        firing_rate, bin_width * 1e-3
    )
    if frequencies.size == 0:
        return {feature: None for feature in FEATURE_NAMES}

    stim_index = int(np.argmin(np.abs(frequencies - stim_freq)))
    #adjust for difference between analysis start and true delay
    referenced_coefficients = coefficients * np.exp(-1j * (stim_phase + stim_orientation))
    referenced_coefficients *= np.exp(-1j * 2 * np.pi * stim_freq * (analysis_start - true_delay) * 1e-3) if true_delay is not None else 1

    power = amplitude**2
    total_power = float(np.sum(power))
    power_norm = _safe_ratio(float(power[stim_index]), total_power)
    amplitude_norm = _safe_ratio(float(amplitude[stim_index]), float(amplitude[0]))

    phase_start_s = true_delay * 1e-3 if true_delay is not None else analysis_start * 1e-3
    delay_s = analysis_start * 1e-3
    plv = _feature_plv(
        spike_data,
        analysis_stop * 1e-3,
        stim_freq,
        stim_phase,
        stim_orientation,
        phase_start_s,
        delay_s,
    )
    ppc = _feature_ppc(
        spike_data,
        analysis_stop * 1e-3,
        stim_freq,
        stim_phase,
        stim_orientation,
        phase_start_s,
        delay_s,
    )
    deltafr = np.nan
    if baseline_rate.size > 0:
        deltafr = _safe_ratio(float(np.mean(firing_rate)), float(np.mean(baseline_rate)))

    return {
        "plv": _finite_or_none(plv),
        "ppc": _finite_or_none(ppc),
        "deltafr": _finite_or_none(deltafr),
        "complex": _finite_or_none(referenced_coefficients[stim_index]),
        "complex_freq": _finite_or_none(spike_coefficient),
        "amplitude_norm": _finite_or_none(amplitude_norm),
        "power_norm": _finite_or_none(power_norm),
        "total_power": _finite_or_none(total_power),
    }


def _normalise_analysis_window(start: float, stop: float, simulation_stop: float) -> tuple[float, float]:
    start = max(0.0, float(start))
    stop = max(0.0, float(stop))
    if simulation_stop > 0:
        start = min(start, simulation_stop)
        stop = min(stop, simulation_stop)
    if stop < start:
        start, stop = stop, start
    if stop <= start and simulation_stop > start:
        stop = min(simulation_stop, start + FEATURE_BIN_MS)
    return start, stop


def calculate_simulation_features(
    config_path: str | Path,
    analysis_start: float | None = None,
    analysis_stop: float | None = None,
    true_delay: float | None = None,
    bin_width: float = FEATURE_BIN_MS,
) -> dict[str, Any]:
    """Recalculate every feature column for the simulation at ``config_path``.

    The calculations intentionally retain the original importer conventions so
    their output reproduces the feature values stored in ``data.csv``.
    """
    simulation_data, _all_data, cell_data, spike_data = load_data(config_path, with_soma=False)
    if simulation_data is None or cell_data is None or spike_data is None:
        raise ValueError(f"Could not load simulation, cells, or spikes: {config_path}")

    stimulation_info = get_estim_data(simulation_data)
    if stimulation_info is None:
        default_start = min(1000.0, float(simulation_data.time_stop))
        default_stop = float(simulation_data.time_stop)
        if true_delay is None:
            true_delay = default_start
        stim_freq = 0.0
        stim_phase = 0.0
        stim_orientation = 0.0
    else:
        fields, stim_delay, duration, _ramp_up, _ramp_down = stimulation_info
        if true_delay is None:
            true_delay = float(stim_delay)
        default_start = float(stim_delay)
        default_stop = (
            float(stim_delay) + float(duration)
            if duration > 0
            else float(simulation_data.time_stop)
        )
        field = fields[0] if fields else {}
        stim_freq = float(field.get("frequency", 0))
        stim_phase = float(field.get("phase", 0.0))
        stim_orientation = field_phase_orientation(field)

    feature_start, feature_stop = _normalise_analysis_window(
        default_start if analysis_start is None else analysis_start,
        default_stop if analysis_stop is None else analysis_stop,
        float(simulation_data.time_stop),
    )
    cell_to_rank = geom_rank_cell_nodes(cell_data.copy(), simulation_data)
    group_data = {"ALL": (spike_data, cell_to_rank)}
    group_data.update(extract_main_cell_types(spike_data, cell_data, cell_to_rank))

    features = empty_feature_values()
    for group, (group_spikes, group_ranks) in group_data.items():
        if group not in FEATURE_GROUPS:
            continue
        values = _group_feature_values(
            simulation_data,
            group_spikes,
            n_neurons=group_ranks.shape[0],
            analysis_start=feature_start,
            analysis_stop=feature_stop,
            stim_freq=stim_freq,
            stim_phase=stim_phase,
            stim_orientation=stim_orientation,
            bin_width=bin_width,
            true_delay=true_delay,
        )
        for feature, value in values.items():
            features[feature_column_name(feature, group)] = value
    return features


# Backwards-compatible names used by prior notebooks and scripts.
get_somaPos = get_soma_pos
getSpiking = get_spiking
getBaselineData = get_baseline_data
getBaselineComplex = get_baseline_complex
getBaselinePower = get_baseline_power
getBaselinePLV = get_baseline_plv

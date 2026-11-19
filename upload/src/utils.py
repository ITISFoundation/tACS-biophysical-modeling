from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

import bluepysnap as bp
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA

POPULATION = "S1nonbarrel_neurons"


@contextmanager
def silence_fdstderr():
    """Silence low-level stderr output from HDF5/BluePySnap calls."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved_stderr = os.dup(2)
    os.dup2(devnull, 2)

    try:
        yield
    finally:
        os.dup2(saved_stderr, 2)
        os.close(saved_stderr)
        os.close(devnull)


def load_data(sim_config_path: str | Path):
    """Load simulation, soma report, node table, and spikes for the project population."""
    with silence_fdstderr():
        try:
            simulation = bp.Simulation(str(sim_config_path))
        except Exception:
            return None, None, None, None

        try:
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


def define_central_axis(soma_pos):
    center = np.mean(soma_pos, axis=0).values

    pca = PCA(n_components=3)
    pca.fit(soma_pos)
    main_axis = pca.components_[0]

    elevation = np.arctan2(main_axis[2], np.sqrt(main_axis[0] ** 2 + main_axis[1] ** 2))
    azimuth = np.arctan2(main_axis[1], main_axis[0])

    return center, azimuth, elevation, main_axis


def rotation_matrix_from_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the rotation matrix that rotates unit vector a onto unit vector b."""
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
        [
            [0.0, -k[2], k[1]],
            [k[2], 0.0, -k[0]],
            [-k[1], k[0], 0.0],
        ]
    )
    angle = np.arccos(np.clip(c, -1.0, 1.0))
    return np.eye(3) + np.sin(angle) * k_matrix + (1 - np.cos(angle)) * (k_matrix @ k_matrix)


def get_somaPos(simulation_data):
    return simulation_data.circuit.nodes[POPULATION].get(properties=["x", "y", "z"])


def geom_rank_cell_nodes(cell_data, simulation_data):
    """Rank cells by depth after rotating the cortical column axis onto z."""
    soma_position = get_somaPos(simulation_data)
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
    simulation_data,
    spike_data,
    bin_width,
    n_neurons,
    delay=None,
    time_stop=None,
    center=True,
    smoothed=False,
    sigma_ms=10,
):
    stimulation_info = get_estim_data(simulation_data)
    if delay is None:
        if stimulation_info is None:
            delay = 200
        else:
            _fields, delay, _duration, _ramp_up, _ramp_down = stimulation_info
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


def extract_main_cell_types(data, cell_data, cell_to_rank):
    data_split = {}
    for cell_type in ("PYR", "INT"):
        if cell_type == "INT":
            for synapse_type in ("INH", "EXC"):
                cells = cell_data[cell_data["morph_class"] == cell_type]
                node_ids = np.array(cells[cells["synapse_class"] == synapse_type].index)
                data_split[f"{cell_type}_{synapse_type}"] = [
                    data[data.isin(node_ids)],
                    cell_to_rank.loc[cell_to_rank.index.isin(node_ids)],
                ]
        else:
            node_ids = np.array(cell_data[cell_data["morph_class"] == cell_type].index)
            data_split[cell_type] = [
                data[data.isin(node_ids)],
                cell_to_rank.loc[cell_to_rank.index.isin(node_ids)],
            ]
    return data_split


def _windowed_spike_phases(
    spike_data,
    t_stop,
    freq,
    phase0=0.0,
    phase_orientation=0.0,
    stim_start=0.0,
    delay=0.0,
) -> np.ndarray:
    spike_times_s = np.array(spike_data.index.tolist()) * 1e-3
    spike_times_s = spike_times_s[spike_times_s > delay]
    spike_times_s = spike_times_s[spike_times_s < t_stop]
    return 2 * np.pi * freq * (spike_times_s - stim_start) + phase0 + phase_orientation


def get_ppc(
    spike_data,
    t_stop,
    freq,
    phase0=0.0,
    phase_orientation=0.0,
    stim_start=0.0,
    delay=0.0,
):
    """Pairwise Phase Consistency for spikes in a seconds-based analysis window."""
    theta = _windowed_spike_phases(
        spike_data,
        t_stop,
        freq,
        phase0=phase0,
        phase_orientation=phase_orientation,
        stim_start=stim_start,
        delay=delay,
    )
    n_spikes = len(theta)
    if n_spikes < 2:
        return np.nan

    phase_sum = np.sum(np.exp(1j * theta))
    return (np.abs(phase_sum) ** 2 - n_spikes) / (n_spikes * (n_spikes - 1))


def get_plv(
    spike_data,
    t_stop,
    freq,
    phase0=0.0,
    phase_orientation=0.0,
    stim_start=0.0,
    delay=0.0,
):
    """Phase Locking Value for spikes in a seconds-based analysis window."""
    theta = _windowed_spike_phases(
        spike_data,
        t_stop,
        freq,
        phase0=phase0,
        phase_orientation=phase_orientation,
        stim_start=stim_start,
        delay=delay,
    )
    if len(theta) == 0:
        return np.nan
    return np.abs(np.mean(np.exp(1j * theta)))


def get_data(config_folders):
    data_dict = {}
    for config_folder in config_folders:
        simulation_data, all_data, cell_data, spike_data = load_data(
            f"{config_folder}/sim_with_stim.json"
        )
        cell_to_rank = geom_rank_cell_nodes(cell_data, simulation_data)
        data_dict[config_folder] = [
            simulation_data,
            all_data,
            cell_data,
            spike_data,
            cell_to_rank,
        ]
    return data_dict


def compute_fourier(
    hist: np.ndarray,
    sample_period_s: float,
    referenced: bool = True,
    sim_id: str | None = None,
    csv_path: str | Path = "data.csv",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    yf = np.fft.fft(hist)
    frequencies = np.fft.fftfreq(hist.shape[0], sample_period_s)
    positive = slice(0, hist.shape[0] // 2)
    amplitude = np.abs(yf[positive]) / hist.shape[0]

    if referenced and sim_id is not None:
        data = pd.read_csv(csv_path)
        data_log = data[data["sim_id"] == sim_id]
        if not data_log.empty:
            phase0 = data_log["stim_phase"].values[0]
            phase_orientation = data_log["stim_orientation"].values[0]
            freq = data_log["stim_freq"].values[0]
            delay = data_log["analysis_start"].values[0]
            stim_start = data_log["delay"].values[0]
            t0 = delay - stim_start if stim_start is not None else 0.0
            yf = yf * np.exp(
                -1j * (phase0 + phase_orientation + 2 * np.pi * freq * t0 * 1e-3)
            )

    return frequencies[positive], amplitude, yf[positive]

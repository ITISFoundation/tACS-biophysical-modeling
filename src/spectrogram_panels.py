"""Onset-aligned rate averages and Morlet amplitude panels."""
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .spectrograms import load_record, morlet_kernel
from .spectrogram_comparisons import morlet_power


def average_rates(records, onsets, bin_s=0.005):
    """Equal-weight average on the intersection of onset-relative sample grids.

    No interpolation, zero filling, or changing the number of seeds over time.
    Returns bin centres, averaged rate, and the aligned individual rates.
    """
    if not records or len(records) != len(onsets):
        raise ValueError("Need one onset per recording")
    grids = []
    for record, onset in zip(records, onsets):
        positions = (np.asarray(record['rate_time']) - onset) / bin_s - 0.5
        indices = np.rint(positions).astype(int)
        if not np.allclose(positions, indices, rtol=0, atol=1e-7):
            raise ValueError("Stimulation onsets must align with the firing-rate bins")
        if not np.all(np.diff(indices) == 1):
            raise ValueError("Need continuous, equally spaced rate samples")
        grids.append(indices)
    first, last = max(g[0] for g in grids), min(g[-1] for g in grids)
    if last <= first:
        raise ValueError("Seed recordings have no common interval")
    rates = np.stack([r['rate'][first-g[0]:last-g[0]+1] for r, g in zip(records, grids)])
    return (np.arange(first, last+1) + 0.5) * bin_s, rates.mean(axis=0), rates


def load_amplitude_averages(root, rows_by_state, node_sets_path, frequencies, *,
                            bin_s=0.005, periods=3.0, stimulus_hz=10.0):
    """Load explicitly selected rows; average rates before calculating power."""
    root = Path(root)
    nodes = json.loads(Path(node_sets_path).read_text())
    panels, audit = {}, []
    for state, rows in rows_by_state.items():
        panels[state] = {}
        for amplitude, group in rows.groupby('stim_amp', sort=True):
            records, onsets, offsets, seeds, names, phases = [], [], [], [], [], []
            seen_paths = set()
            for row in group.to_dict('records'):
                config_path = (root / row['source_file']).resolve()
                if re.search('hex0|O1', row['sim_id'] + ' ' + config_path.parent.name, re.I):
                    raise ValueError(f"Excluded circuit-comparison run selected: {row['sim_id']}")
                if config_path in seen_paths:
                    raise ValueError(f"Duplicate report selected: {config_path}")
                seen_paths.add(config_path)
                config = json.loads(config_path.read_text())
                fields = [(s, f) for s in config.get('inputs', {}).values()
                          if s.get('input_type') == 'extracellular_stimulation'
                          for f in s.get('fields', [])
                          if any(float(f.get(k, 0)) != 0 for k in ('Ex', 'Ey', 'Ez'))]
                if len(fields) != 1:
                    raise ValueError(f"Expected one active field: {config_path}")
                stimulus, field = fields[0]
                vector = np.array([field.get(k, 0) for k in ('Ex', 'Ey', 'Ez')], float)
                if not np.isclose(field['frequency'], stimulus_hz) or not np.isclose(np.linalg.norm(vector), amplitude):
                    raise ValueError(f"Table and configured stimulation disagree: {config_path}")
                seed = config['run']['random_seed']
                if seed in seeds:
                    raise ValueError(f"Repeated seed {seed} at {state}, {amplitude:g} V/m")
                record = load_record(config_path, nodes, bin_s=bin_s, compute_spectrum=False)
                onset = float(stimulus['delay']) / 1000
                duration = float(stimulus['duration']) / 1000
                records.append(record)
                onsets.append(onset)
                offsets.append(duration)
                seeds.append(seed)
                names.append(row['sim_id'])
                phases.append(np.exp(1j * float(field.get('phase', 0))) * vector / np.linalg.norm(vector))
                audit.append(dict(state=state, amplitude_vm=float(amplitude), sim_id=row['sim_id'],
                                  seed=seed, source_file=str(config_path), n_neurons=record['n_neurons'],
                                  onset_s=onset, duration_s=duration, configured_stop_s=record['stop_s'],
                                  observed_stop_s=record['observed_stop'], warnings='; '.join(record['warnings'])))
            if not np.allclose(phases, phases[0]):
                raise ValueError("Seed field phases/orientations differ; onset alignment alone is insufficient")
            time, mean_rate, seed_rates = average_rates(records, onsets, bin_s)
            power = morlet_power(mean_rate, frequencies, dt=bin_s, periods=periods)
            panels[state][float(amplitude)] = dict(
                time=time, rate=mean_rate, seed_rates=seed_rates, power=power,
                power_db=10*np.log10(np.maximum(power, np.finfo(float).tiny)),
                frequency=np.asarray(frequencies), n_seeds=len(seeds), seeds=seeds,
                names=names, offsets=offsets, bin_s=bin_s, periods=periods,
                start_s=float(time[0]-bin_s/2), stop_s=float(time[-1]+bin_s/2),
            )
            for entry in audit[-len(records):]:
                entry.update(common_start_from_onset_s=float(time[0]-bin_s/2),
                             common_stop_from_onset_s=float(time[-1]+bin_s/2), n_seeds=len(seeds))
    if not any(panels.values()):
        raise ValueError("No selected simulations")
    return panels, audit


def plot_amplitude_panel(runs, amplitudes, limits, title, xlim):
    """One panel per amplitude, with shared power limits and marked padded edges."""
    ncols = 3
    nrows = int(np.ceil((len(amplitudes)+1)/ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 3.3*nrows),
                             squeeze=False, layout='constrained')
    mesh = None
    for ax, amplitude in zip(axes.flat, amplitudes):
        run = runs.get(float(amplitude))
        ax.set(xlim=xlim, ylim=(2, 30), xlabel='Time from stimulation onset (s)',
               ylabel='Frequency (Hz)', yticks=[2, 10, 20, 30])
        if run is None:
            ax.set_title(f'{amplitude:g} V/m | unavailable')
            ax.set_facecolor('0.95')
            ax.text(0.5, 0.5, 'No eligible simulation', ha='center', va='center', transform=ax.transAxes)
            continue
        time, frequency = run['time'], run['frequency']
        mesh = ax.pcolormesh(time, frequency, run['power_db'], shading='nearest',
                             cmap='magma', vmin=limits[0], vmax=limits[1], rasterized=True)
        half = np.array([len(morlet_kernel(f, run['bin_s'], run['periods']))//2*run['bin_s']
                         for f in frequency])
        for left, right in [
            (np.full_like(frequency, run['start_s']), np.minimum(time[0]+half, run['stop_s'])),
            (np.maximum(time[-1]-half, run['start_s']), np.full_like(frequency, run['stop_s'])),
        ]:
            ax.fill_betweenx(frequency, left, right, facecolor='none', edgecolor=(1, 1, 1, 0.4),
                             hatch='///', linewidth=0)
        for left, right in [(xlim[0], run['start_s']), (run['stop_s'], xlim[1])]:
            if right-left > 1e-8:
                ax.axvspan(left, right, color='0.9')
                ax.text((left+right)/2, 16, 'Outside common\nrecording', ha='center', va='center',
                        rotation=90, fontsize=8, color='0.4')
        ax.axvline(0, color='white', ls='--', lw=1)
        for offset in sorted(set(run['offsets'])):
            if xlim[0] <= offset <= xlim[1]:
                ax.axvline(offset, color='white', ls='--', lw=1)
        ax.hlines(10, max(0, run['start_s']), min(min(run['offsets']), run['stop_s']),
                  color='cyan', ls=':', lw=1)
        ax.set_title(f"{amplitude:g} V/m | {run['n_seeds']} seed{'s' if run['n_seeds'] != 1 else ''}")
    notes_ax = axes.flat[len(amplitudes)]
    notes_ax.set_axis_off()
    notes_ax.text(0.04, 0.92,
                  '10 Hz stimulation\n\nEqual-weight mean firing rate across seeds,\n'
                  'then one Morlet transform per amplitude.\n\n'
                  '2–30 Hz; at least 3 periods per frequency.\n'
                  'Hatching: reflected edges of common interval.\n'
                  'White dashed: stimulation boundaries.\nCyan dotted: 10 Hz.\n\n'
                  'One colour scale across both brain states.',
                  transform=notes_ax.transAxes, va='top', fontsize=10, linespacing=1.4)
    for ax in list(axes.flat)[len(amplitudes)+1:]:
        ax.set_axis_off()
    if mesh is not None:
        fig.colorbar(mesh, ax=list(axes.flat)[:len(amplitudes)],
                     label='Morlet power (dB re 1 (spikes/s/neuron)²)', shrink=0.9)
    fig.suptitle(title, fontsize=15)
    return fig

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from src.utils import field_phase_orientation, get_estim_data, get_firingrate_hist

def plot_amplitude_phase(xf_pos, amplitude, phase, freq, idx_base=None,figsize=(10,4),multiple_sims=False,normalized=True,colorList=None):
    """Two-panel plot: amplitude and phase spectra."""

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    if multiple_sims:
        amplitudes = amplitude
        phases = phase
    else:
        amplitudes = [amplitude]
        phases = [phase]
        if not isinstance(normalized,bool):
            normalized = [normalized]

    for idx, amplitude in enumerate(amplitudes):

        if colorList is not None:
            color = colorList[idx]
        else:
            color = 'C'+str(idx)
            
        phase = phases[idx]

        phase = phase % (2 * np.pi)

        if isinstance(normalized,bool):
            if normalized:
                amplitude = amplitude / amplitude[0]
                axes[0].set(title='DC-Normalized Amplitude Spectrum', xlabel='Frequency (Hz)', ylabel='Amplitude')
            else:
                axes[0].set(title='Raw Amplitude Spectrum', xlabel='Frequency (Hz)', ylabel='Amplitude')
        else:
            amplitude /= normalized[idx] # Normalizes ampltiude by baseline amplitudes
            
            axes[0].set_ylim([0,np.max(amplitude[xf_pos<20])])
            axes[0].set(title='Baseline-Normalized Amplitude Spectrum', xlabel='Frequency (Hz)', ylabel='Amplitude')

        axes[0].set_xlim([0,20])

        # Amplitude
        axes[0].stem(xf_pos[1:], amplitude[1:], basefmt=" ", linefmt=color)
        axes[0].grid(True)

        # Phase
        axes[1].stem(xf_pos[1:], phase[1:], basefmt=" ", linefmt=color)
        axes[1].set(title='Phase Spectrum', xlabel='Frequency (Hz)', ylabel='Phase (radians)')
        axes[1].set_yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
        axes[1].set_yticklabels(['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
        axes[1].grid(True)

            # Highlight target frequency
        if freq > 0:
            idx_pos = np.argmin(np.abs(xf_pos - freq))
            axes[0].stem(xf_pos[idx_pos], amplitude[idx_pos], markerfmt='ro', basefmt=" ", linefmt=color)
            axes[1].stem(xf_pos[idx_pos], phase[idx_pos], markerfmt='ro', basefmt=" ", linefmt=color)
        if idx_base is not None:
            axes[0].stem(xf_pos[idx_base], amplitude[idx_base], markerfmt='go', basefmt=" ", linefmt=color)
            axes[1].stem(xf_pos[idx_base], phase[idx_base], markerfmt='go', basefmt=" ", linefmt=color)

    # if idx_base is None:
    #     power = amplitude ** 2
    #     idx_base = np.argmax(power / np.sum(power))



    _add_legend(fig, freq, xf_pos, idx_base)
    return fig

def plot_power_spectrum(
    xf_pos, amplitude, freq, idx_base=None, figsize=(6, 4),
    multiple_sims=False, normalized=True, colorList=None,
    log_scale=False, labels=None,
):
    """Plot one or more power spectra on the same frequency grid."""

    amplitudes = amplitude if multiple_sims else [amplitude]
    amplitudes = [np.asarray(a, dtype=float).copy() for a in amplitudes]
    n_sims = len(amplitudes)

    # Accept either one shared grid or one grid per simulation.
    xf_pos = np.asarray(xf_pos, dtype=float)
    if xf_pos.ndim == 2:
        if xf_pos.shape[0] != n_sims or not np.allclose(xf_pos, xf_pos[0]):
            raise ValueError("All simulations must use identical frequency bins.")
        xf_pos = xf_pos[0]
    elif xf_pos.ndim != 1:
        raise ValueError("xf_pos must be a 1-D frequency grid.")

    if xf_pos.size < 2:
        raise ValueError("xf_pos must contain at least two frequency bins.")

    if any(a.ndim != 1 or a.shape != xf_pos.shape for a in amplitudes):
        raise ValueError(
            "Every amplitude array must have the same length as xf_pos."
        )

    if idx_base is not None and not 0 <= idx_base < len(xf_pos):
        raise IndexError("idx_base is outside the frequency grid.")

    colors = (
        [f"C{i % 10}" for i in range(n_sims)]
        if colorList is None
        else list(colorList)
    )
    if len(colors) < n_sims:
        raise ValueError("colorList needs one color per simulation.")

    labels = (
        [f"Simulation {i + 1}" for i in range(n_sims)]
        if labels is None
        else labels
    )

    is_bool_normalized = isinstance(normalized, (bool, np.bool_))
    if not is_bool_normalized:
        normalizers = np.asarray(normalized, dtype=float).reshape(-1)
        if normalizers.size != n_sims or np.any(np.isclose(normalizers, 0)):
            raise ValueError("normalized must contain one non-zero value per simulation.")

    fig, ax = plt.subplots(figsize=figsize)

    for idx, amp in enumerate(amplitudes):
        if is_bool_normalized and normalized:
            if np.isclose(amp[0], 0):
                raise ValueError("Cannot normalize by amplitude[0] when it is zero.")
            power = (amp / amp[0]) ** 2
            power_norm = power / power.sum()
            ax.set(title="Normalized Power Spectrum", ylabel="Power / total")

        elif is_bool_normalized:
            power_norm = amp ** 2
            ax.set(title="Raw Power Spectrum", ylabel="Power")

        else:
            power = (amp / normalizers[idx]) ** 2
            power_norm = power / power.sum()
            ax.set(title="Baseline-normalized Power Spectrum", ylabel="Power / total")

        plot_fn = ax.semilogy if log_scale else ax.plot
        plot_fn(
            xf_pos[1:],
            power_norm[1:],
            color=colors[idx],
            label=labels[idx],
        )

        if freq > 0:
            idx_pos = np.argmin(np.abs(xf_pos - freq))
            ax.plot(xf_pos[idx_pos], power_norm[idx_pos], "ro", zorder=3)

        if idx_base is not None:
            ax.plot(xf_pos[idx_base], power_norm[idx_base], "go", zorder=3)

    ax.set(xlabel="Frequency (Hz)", xlim=(0, 20))
    ax.grid(True)

    if multiple_sims:
        ax.legend(title="Simulation")

    _add_legend(fig, freq, xf_pos, idx_base)
    return fig


def _add_legend(fig, freq, xf_pos, idx_base):
    """Shared legend helper for stimulation and baseline frequency markers."""
    from matplotlib.lines import Line2D
    legend_handles = []
    if freq > 0:
        legend_handles.append(Line2D([], [], color='red', marker='o', linestyle='None',
                                     label=f'Stimulation freq ({freq} Hz)'))
    if idx_base is not None:
        legend_handles.append(Line2D([], [], color='green', marker='o', linestyle='None',
                                    label=f'Max baseline power ({xf_pos[idx_base]:.1f} Hz)'))
    fig.legend(handles=legend_handles, loc='lower center', ncol=2, frameon=True, fontsize=9)
    fig.tight_layout(rect=[0, 0.04, 1, 1])


def plot_fourier_analysis(xf_pos, amplitude, phase, freq, save_path, idx_base=None,figsize=[(10,4),(6,4)],multiple_sims=False,normalized=True,colorList=None):
    """Wrapper: saves both figures with suffixed filenames."""
    base, ext = save_path.rsplit('.', 1)

    if not isinstance(normalized,bool):
        if multiple_sims:
            for idx, normalizationFactor in enumerate(normalized):
                if normalizationFactor.shape != amplitude[idx].shape:
                    raise ValueError("Normalization factor must have same shape as amplitude")
        else:
            if normalized.shape != amplitude.shape:
                    raise ValueError("Normalization factor must have same shape as amplitude")

    fig_ap = plot_amplitude_phase(xf_pos, amplitude, phase, freq, idx_base,figsize=figsize[0],multiple_sims=multiple_sims,normalized=normalized,colorList=colorList)
    # fig_ap.suptitle(f"{title} — Amplitude & Phase")
    fig_ap.savefig(f"{base}_amp_phase.{ext}")
    plt.close(fig_ap)

    fig_pw = plot_power_spectrum(xf_pos, amplitude, freq, idx_base,figsize=figsize[1],multiple_sims=multiple_sims,normalized=normalized,colorList=colorList)
    # fig_pw.suptitle(f"{title} — Power Spectrum")
    fig_pw.savefig(f"{base}_power.{ext}",dpi=600)
    plt.close(fig_pw)

def define_psth_plot(simulation_time, title, n_neurons,figsize=(16,10)):
    fig, (ax_raster, ax_stimulation) = plt.subplots(
    2, 1,
    figsize=figsize,
    sharex=True,
    gridspec_kw={'height_ratios': [2, 1]}
    )
  
    ax_raster.set_xlabel('Time (ms)')
    ax_raster.set_ylabel('Neurons')
    # ax_raster.set_title(f'{title}')
    ax_raster.grid(False)
    ax_raster.set_ylim(0, n_neurons)

    ax_hist = ax_raster.twinx()

    ax_hist.set_ylabel('Populaion Firing rate (Hz)')
    ax_hist.set_ylim(0,1)

    ax_stimulation.set_xlabel('Time (ms)')
    ax_stimulation.set_ylabel('Amplitude (V/m)')
    ax_stimulation.set_title('Electric field stimulation')
    ax_stimulation.set_ylim(-4.2,4.2)

    ax_hist2 = ax_stimulation.twinx()

    ax_hist2.set_ylabel('Populaion Firing rate (Hz)')
    ax_hist2.set_ylim(0.1,0.4)

    ax_stimulation.set_xlabel('Time (ms)')
    ax_stimulation.set_ylabel('Amplitude (V/m)')
    ax_stimulation.set_title('Electric field stimulation')
    ax_stimulation.set_ylim(-4.2,4.2)

    ax_stimulation.grid()


    for ax in (ax_raster, ax_stimulation):
        ax.set_xlim(0, simulation_time)

    ax_raster.tick_params(axis='x', labelbottom=True, bottom=True)
    return fig, (ax_raster, ax_stimulation, ax_hist, ax_hist2)

def define_stimulation_function(simulationData, dt):

    stimulation_info = get_estim_data(simulationData)
  
    simulation_time = simulationData.time_stop
    time = np.arange(0, simulation_time, dt)
    stimulation_func = np.zeros((time.shape[0],))
    if stimulation_info is not None:

        Efields,delay,duration,rampup,rampdown = stimulation_info
        if simulation_time-delay < duration:
            duration = simulation_time-delay
        delay_idx = int(round(delay / dt))
        dur_samples = int(round(duration / dt))
        end_idx = min(delay_idx + dur_samples, len(time))
        tseg = time[delay_idx:end_idx] - time[delay_idx]
        for Efield in Efields:
            a = np.round(np.sqrt(Efield['Ex']**2 + Efield['Ey']**2 + Efield['Ez']**2),2)
            # cos(2*pi*f*(t - delay) + phase + phase_orientation): the same reference the PLV uses,
            # so phase 0 of the PLV is the peak of this curve.
            orientation = field_phase_orientation(Efield)
            stimulation_func_part = a*np.sin(
                2*np.pi*Efield['frequency']*1e-3*tseg + (Efield["phase"] + orientation + np.pi / 2)
            )

            stimulation_func[delay_idx:end_idx] += stimulation_func_part

        #code from https://github.com/openbraininstitute/neurodamus/blob/weji/stimulus_electrodes/neurodamus/core/stimuli.py
        if rampup != 0:
            ramp_up_number = int(rampup / dt)
            if ramp_up_number > 0:
                ramp_up = np.linspace(0, 1, ramp_up_number)
                stimulation_func[delay_idx:delay_idx + ramp_up_number] *= ramp_up

        if rampdown != 0:
            ramp_down_number = int(rampdown / dt)
            if ramp_down_number > 0:
                ramp_down = np.linspace(1, 0, ramp_down_number)
                stimulation_func[end_idx - ramp_down_number:end_idx] *= ramp_down

    return stimulation_func

def multiple_psth_plot(spikeData, title, cell_to_rank, simulationData,pop_names, bin_width=1, save_histogram=None, smoothed=False, sigma_ms=10,figsize=(16,10),colorList=None, normalized=False):
    #dt = simulationData.dt in ms-1 == make sure this is clear
    #simulation_time = simulationData.tstop
    #spikes have different sampling frequency from stimulation

    fig, ax_hist = plt.subplots(1,1,figsize=figsize)

    for i in range(len(spikeData)):

        if colorList is not None:
            color = colorList[i]
        else:
            color = 'C'+str(i)
    
        simulation_time = simulationData[i].time_stop
        dt = simulationData[i].dt

        spikeData_ranked = spikeData[i].map(cell_to_rank[i])

        #plot raster plot
        time = np.arange(0, simulation_time, dt)

        #create histogram of spike train
        firing_rate, bins = get_firingrate_hist(simulationData[i], spikeData_ranked, bin_width, cell_to_rank[i].shape[0], delay=0, time_stop=simulation_time)
        # bins, firing_rate = define_histogram(spikeData_ranked, simulation_time, bin_width, cell_to_rank.shape[0])

        sigma_samples = sigma_ms / bin_width  
        smoothed_fr = gaussian_filter1d(firing_rate, sigma=sigma_samples)

        if normalized:
            smoothed_fr /= np.max(smoothed_fr)
        ax_hist.plot(
        bins,
        smoothed_fr,
        color=color,
        lw=2,
        label='Firing rate '+pop_names[i]
    )

    ax_hist.set_title(title)
    ax_hist.set_xlabel('Time (ms)')
    if normalized:
        ax_hist.set_ylabel('Normalized firing rate')
    else:
        ax_hist.set_ylabel('Population\nfiring rate (Hz)')
    ax_hist.legend()

    plt.tight_layout()
    if save_histogram is not None:
        plt.savefig(save_histogram,dpi=600)
    return fig


def psth_grid_plot(
    spike_data,
    cell_to_rank,
    simulation_data,
    labels,
    *,
    bin_width=5,
    sigma_ms=10,
    ncols=4,
    waveform_step_ms=1.0,
    figsize=None,
    title="Population firing rates for all simulations",
):
    """Plot one compact population PSTH panel per simulation in a shared figure.

    Each panel contains the raw histogram, a Gaussian-smoothed trace and the applied stimulation
    waveform on a secondary axis. Firing-rate and stimulation axes use common scales across all
    panels so both quantities can be compared directly between simulations.
    """
    from matplotlib.lines import Line2D

    lengths = {len(spike_data), len(cell_to_rank), len(simulation_data), len(labels)}
    if len(lengths) != 1:
        raise ValueError("spike_data, cell_to_rank, simulation_data and labels must have equal lengths")
    n_panels = len(labels)
    if n_panels == 0:
        raise ValueError("At least one simulation is required")
    if ncols < 1:
        raise ValueError("ncols must be at least 1")
    if bin_width <= 0 or sigma_ms < 0 or waveform_step_ms <= 0:
        raise ValueError("bin_width and waveform_step_ms must be positive; sigma_ms cannot be negative")

    ncols = min(int(ncols), n_panels)
    nrows = int(np.ceil(n_panels / ncols))
    if figsize is None:
        figsize = (3.4 * ncols, 2.25 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharey=True, squeeze=False)

    stimulation_waveforms = []
    maximum_stimulation = 0.0
    for simulation in simulation_data:
        dt = float(simulation.dt)
        if dt <= 0:
            raise ValueError("Every simulation must have a positive dt")
        waveform = define_stimulation_function(simulation, dt)
        stimulation_waveforms.append((dt, waveform))
        if waveform.size:
            maximum_stimulation = max(maximum_stimulation, float(np.max(np.abs(waveform))))

    # A common secondary scale makes field amplitudes visually comparable across panels.
    stimulation_limit = max(1.0, 1.05 * maximum_stimulation)
    stimulation_axes = []

    for panel_index, (ax, spikes, ranks, simulation, label, stimulation_data) in enumerate(zip(
        axes.ravel(), spike_data, cell_to_rank, simulation_data, labels, stimulation_waveforms
    )):
        dt, stimulation = stimulation_data
        stimulation_time = np.arange(stimulation.size, dtype=float) * dt
        waveform_stride = max(1, int(round(waveform_step_ms / dt)))

        stimulation_ax = ax.twinx()
        stimulation_axes.append(stimulation_ax)
        stimulation_ax.plot(
            stimulation_time[::waveform_stride],
            stimulation[::waveform_stride],
            color="C2",
            lw=0.8,
            alpha=0.75,
            zorder=1,
        )
        stimulation_ax.set_ylim(-stimulation_limit, stimulation_limit)
        stimulation_ax.grid(False)
        stimulation_ax.tick_params(axis="y", colors="C2", labelsize=6, length=2, pad=1)
        if panel_index % ncols == ncols - 1:
            stimulation_ax.set_yticks([-stimulation_limit, 0, stimulation_limit])
        else:
            stimulation_ax.set_yticks([])

        ranked_spikes = spikes.map(ranks)
        firing_rate, bins = get_firingrate_hist(
            simulation,
            ranked_spikes,
            bin_width,
            ranks.shape[0],
            delay=0,
            time_stop=simulation.time_stop,
        )
        firing_rate = np.asarray(firing_rate, dtype=float)
        bins = np.asarray(bins, dtype=float)
        smoothed_rate = (
            gaussian_filter1d(firing_rate, sigma=sigma_ms / bin_width)
            if sigma_ms > 0
            else firing_rate.copy()
        )

        ax.step(bins, firing_rate, where="mid", color="C0", lw=0.7, alpha=0.3, zorder=2)
        ax.plot(bins, smoothed_rate, color="C1", lw=1.4, zorder=3)

        ax.set_title(str(label), fontsize=8, wrap=True)
        ax.set_xlim(0, float(simulation.time_stop))
        ax.set_xticks([0, float(simulation.time_stop) / 2, float(simulation.time_stop)])
        ax.tick_params(labelsize=7)
        ax.grid(axis="y", alpha=0.15, lw=0.5)

    for ax in axes.ravel()[n_panels:]:
        ax.set_visible(False)

    fig.suptitle(title, y=0.998)
    fig.supxlabel("Time (ms)")
    fig.supylabel("Population firing rate (Hz)")
    fig.text(0.995, 0.5, "Stimulation amplitude (V/m)", va="center", ha="right", rotation=-90)
    fig.legend(
        handles=[
            Line2D([], [], color="C0", lw=1, alpha=0.4, label="Raw PSTH"),
            Line2D([], [], color="C1", lw=1.5, label=f"Smoothed PSTH (σ={sigma_ms:g} ms)"),
            Line2D([], [], color="C2", lw=1, alpha=0.75, label="Stimulation waveform"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.982),
        ncol=3,
        frameon=False,
    )
    fig.tight_layout(rect=(0.02, 0.02, 0.985, 0.95))
    fig.stimulation_axes = stimulation_axes
    return fig, axes


def psth_plot(spikeData, title, cell_to_rank, simulationData, pop_size=None, bin_width=1, save_histogram=None, smoothed=False, sigma_ms=10,figsize=(16,10)):
    
    #dt = simulationData.dt in ms-1 == make sure this is clear
    #simulation_time = simulationData.tstop
    #spikes have different sampling frequency from stimulation
    simulation_time = simulationData.time_stop
    dt = simulationData.dt
    if pop_size is None: 
        pop_size = cell_to_rank.shape[0]
 
    fig, (ax_raster, ax_stimulation, ax_hist, ax_hist2) = define_psth_plot(simulation_time, title, pop_size,figsize=figsize)

    pointSize = figsize[1]/20

    spikeData_ranked = spikeData.map(cell_to_rank)
    spikes = spikeData_ranked.reset_index().to_numpy()
    cell_ids_rank = spikes[:,1]
    spike = spikes[:,0]
    #plot raster plot
    time = np.arange(0, simulation_time, dt)
    ax_raster.scatter(spike, cell_ids_rank, s=pointSize, c='black')
    #create histogram of spike train
    firing_rate, bins = get_firingrate_hist(simulationData, spikeData_ranked, bin_width, cell_to_rank.shape[0], delay=0, time_stop=simulation_time)
    # bins, firing_rate = define_histogram(spikeData_ranked, simulation_time, bin_width, cell_to_rank.shape[0])
    ax_hist.step(bins, firing_rate, where='mid', color='C0', label='Population firing rate')

    if smoothed:
        sigma_samples = sigma_ms / bin_width
        smoothed_fr = gaussian_filter1d(firing_rate, sigma=sigma_samples)
    #     ax_hist.plot(
    #     bins,
    #     smoothed_fr,
    #     color='C1',
    #     lw=2,
    #     label=f'Smoothed firing rate (σ={sigma_ms} ms)'
    # )


    stimulation_plot = define_stimulation_function(simulationData, dt)
    ax_stimulation.plot(
        time,
        stimulation_plot,
        color='slategrey',
        label='Applied stimulation'
    )

    if smoothed:
        ax_hist2.plot(
            bins,
            smoothed_fr,
            color='#c50000',
            lw=2,
            label=f'Smoothed firing rate (σ={sigma_ms} ms)'
        )

    plt.tight_layout()
    if save_histogram is not None:
        plt.savefig(save_histogram,dpi=600)
    return fig, (ax_raster, ax_stimulation, ax_hist)


def plot_condition_means(df, baseline, ax, color, id_col='stim_amp', z_col='complex', max_amp=None):
    """Plot group-mean complex values on a polar axis."""
    z = df[z_col].astype(complex)
    if max_amp is not None:
        z = z[df['stim_amp'] <= max_amp]
    means = z.groupby(df[id_col]).mean().sort_index()
    zs = np.insert(means.values, 0, baseline)
    ax.plot(np.angle(zs), np.abs(zs), '-', c=color, lw=2, ms=5)
    return means


def plot_linear_fit(x, y, ax, color='k', label=None):
    """Fit and plot a least-squares line, printing its R-squared value."""
    from scipy import stats

    slope, intercept, r_value, _p_value, _std_err = stats.linregress(x, y)
    r_squared = r_value ** 2
    ax.plot(
        np.array([x[0], x[-1]]),
        np.array([slope * x[0] + intercept, x[-1] * slope + intercept]),
        c=color,
        label=label,
    )
    print(f"R^2 = {r_squared:.4f}")


def plot_condition_means_linear(df, baseline, ax, color, id_col='stim_amp', z_col='plv', label=None,
                                linewidth=0.5):
    """Plot group means plus the baseline point on a Cartesian axis."""
    means = df.groupby(id_col)[z_col].mean().sort_index()
    xs = np.insert(means.index.values.astype(float), 0, 0.0)
    ys = np.insert(means.values, 0, baseline)
    ax.plot(xs, ys, '-', c=color, lw=linewidth, ms=5, label=label, alpha=1)
    return means


def annotate_significance(ax, x1, x2, label, y=None, pad=0.06, tick=0.02, color="0.25",
                          fontsize=7, linewidth=1.0):
    """Draw a significance bracket between two x positions and write ``label`` above it.

    ``y`` is the height of the bracket; by default it sits ``pad`` (a fraction of the current
    y-range) above everything drawn so far. ``tick`` is the length of the downward end ticks, also
    as a fraction of the y-range. The y-limit is extended when needed, and the top of the
    annotation is returned so that brackets can be stacked.
    """
    low, high = ax.get_ylim()
    span = high - low
    if y is None:
        y = max(ax.dataLim.y1, low) + pad * span
    ax.plot([x1, x1, x2, x2], [y - tick * span, y, y, y - tick * span],
            color=color, linewidth=linewidth, clip_on=False, zorder=4)
    ax.text((x1 + x2) / 2, y + 0.01 * span, label, ha="center", va="bottom",
            fontsize=fontsize, color=color, clip_on=False, zorder=4)
    top = y + 0.11 * span          # leaves room for the label itself
    if top > high:
        ax.set_ylim(low, top)
    return top


def plot_metric_vs_amplitude(ax, df, baseline, color, z_col='plv', label=None, mean_line=True,
                             marker='o', yerr_col="auto", ylim=None,
                             amp_scale=1.0, size_power=1.0, linewidth=0.5, xlim=None):
    """Plot condition means of ``z_col`` against amplitude, with baseline at 0 V/m.

    ``yerr_col="auto"`` uses ``<z_col>_std`` when present, otherwise it calculates the sample
    standard deviation across rows at each amplitude. Raw replicate rows are represented by one
    mean marker per amplitude. Pass a column name to use another stored error column, or ``None``
    for no error bars.

    ``amp_scale`` multiplies the marker areas and ``size_power > 1`` widens the gap between the
    smallest and largest markers (see ``marker_sizes``). ``linewidth`` is the width of the line
    joining the means (``mean_line=True``).
    """
    grouped = df.groupby("stim_amp", sort=True)
    condition_means = grouped[z_col].mean()
    amps = np.insert(condition_means.index.to_numpy(float), 0, 0.0)
    values = np.insert(condition_means.to_numpy(float), 0, baseline)
    ax.scatter(
        amps,
        values,
        color=color,
        alpha=1,
        marker=marker,
        s=marker_sizes(amps, power=size_power) * amp_scale,
        label=None if mean_line else label,
    )
    if yerr_col == "auto":
        std_col = f"{z_col}_std"
        errors = grouped[z_col].std(ddof=1)
        if std_col in df.columns:
            errors = grouped[std_col].first().combine_first(errors)
    elif yerr_col is not None and yerr_col in df.columns:
        errors = df.groupby("stim_amp", sort=True)[yerr_col].first()
    else:
        errors = None
    if errors is not None:
        error_amps = np.insert(errors.index.to_numpy(float), 0, 0.0)
        error_values = np.insert(condition_means.reindex(errors.index).to_numpy(float), 0, baseline)
        error_values_array = np.nan_to_num(errors.to_numpy(float), nan=0.0)
        error_values_array = np.insert(error_values_array, 0, 0.0)
        ax.errorbar(error_amps, error_values, yerr=error_values_array, fmt='none', ecolor=color,
                    alpha=1, elinewidth=0.5, capsize=3, capthick=0.5, zorder=2)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if mean_line:
        plot_condition_means_linear(df, baseline, ax, color, z_col=z_col, label=label,
                                    linewidth=linewidth)
    if xlim is not None:
        ax.set_xlim(*xlim)


AMPLITUDE_MARKER_AREAS = {
    0.0: 2.5,
    0.25: 4.9,
    0.5: 8.1,
    1.0: 15.625,
    2.0: 30.625,
    4.0: 62.5,
}


def marker_sizes(amps, power=1.0):
    """Map stimulation amplitudes to Figure 4b's shared marker areas.

    The returned values are Matplotlib scatter areas in points squared. Known experiment
    amplitudes have fixed sizes; any intermediate amplitude is linearly interpolated.

    ``power`` stretches the size range nonlinearly around its smallest area.
    """
    amplitudes = np.asarray(amps, dtype=float)
    anchors = np.asarray(tuple(AMPLITUDE_MARKER_AREAS), dtype=float)
    table = np.asarray(tuple(AMPLITUDE_MARKER_AREAS.values()), dtype=float)
    if power != 1.0:
        table = table.min() * (table / table.min()) ** power
    return np.interp(amplitudes, anchors, table)


def polar_scatter(ax, df, baseline_complex=None, color='k', label=None, z_col='complex', max_amp=None, marker='o',
                  linestyle='-', line=False, means=False, edgecolors='face',
                  rlim=None, rticks=None, rtick_labels=None, rlabel_position=135, rtick_size=None,
                  rtick_label_every=2, amp_scale=1.0, size_power=1.0,
                  linewidth=0.5, radial_linewidth=0.5):
    """Plot complex values as polar points, marker area growing with amplitude.

    ``baseline_complex`` is drawn at 0 V/m. ``line=True`` adds a radial line per point and
    ``means=True`` plots one mean complex value per amplitude and connects those mean points;
    otherwise every row is plotted individually.

    Radial axis: ``rlim=(low, high)`` sets the range, ``rticks`` the radii of the circles,
    ``rtick_labels`` their text (``[]`` keeps the circles but hides the numbers),
    ``rlabel_position`` the angle in degrees at which the labels sit, and ``rtick_size`` their font
    size. ``rtick_label_every=2`` labels alternate circles while retaining every radial grid line;
    use 1 to label every circle. Each setting is applied only when given, so several calls can
    share one axis; without a fixed range the radial range is autoscaled by matplotlib.

    ``amp_scale`` multiplies all marker areas; ``size_power > 1`` additionally widens the gap
    between the smallest and largest markers (see ``marker_sizes``). ``linewidth`` is the width of
    the line joining the means (``means=True``) and ``radial_linewidth`` that of the radial lines
    (``line=True``).
    """
    data = df
    if max_amp is not None:
        data = data[data['stim_amp'] <= max_amp]
    if means:
        condition_means = data.groupby('stim_amp', sort=True)[z_col].apply(
            lambda values: values.astype(complex).mean()
        )
        z = condition_means.to_numpy(dtype=complex)
        amps = condition_means.index.to_numpy(dtype=float)
    else:
        z = data[z_col].astype(complex).to_numpy()
        amps = data['stim_amp'].to_numpy(dtype=float)
    if baseline_complex is not None:
        amps = np.insert(amps, 0, 0.0)
        z = np.insert(z, 0, baseline_complex)
    theta = np.angle(z)
    r = np.abs(z)
    if line:
        for angle, radius in zip(theta, r):
            ax.plot(
                [angle, angle],
                [0, radius],
                linestyle=linestyle,
                c=color,
                lw=radial_linewidth,
                alpha=1,
                zorder=2,
            )
    ax.scatter(
        theta,
        r,
        marker=marker,
        facecolors=color,
        alpha=1,
        label=label,
        s=marker_sizes(amps, power=size_power)*amp_scale,
        edgecolors=edgecolors,
        zorder=3,
    )
    if means:
        ax.plot(theta, r, '-', c=color, lw=linewidth, zorder=2, alpha=1)
    if rlim is not None:
        ax.set_rlim(*rlim)
    if rticks is not None:
        ax.set_rticks(rticks)
    if rtick_label_every is not None:
        if int(rtick_label_every) != rtick_label_every or rtick_label_every < 1:
            raise ValueError("rtick_label_every must be a positive integer or None")
        ticks = ax.get_yticks()
        if rtick_labels is None:
            labels = [f"{tick:g}" for tick in ticks]
        else:
            if len(rtick_labels) == 0:
                labels = [""] * len(ticks)
            elif len(rtick_labels) != len(ticks):
                raise ValueError("rtick_labels must have the same length as the radial tick locations")
            else:
                labels = list(rtick_labels)
        every = int(rtick_label_every)
        ax.set_yticks(ticks, [label if index % every == 0 else "" for index, label in enumerate(labels)])
    elif rtick_labels is not None:
        ax.set_yticklabels(rtick_labels)
    if rlabel_position is not None:
        ax.set_rlabel_position(rlabel_position)
    if rtick_size is not None:
        ax.tick_params(axis='y', labelsize=rtick_size)



def _darken(color, amount=0.6):
    """Return ``color`` scaled toward black."""
    import colorsys
    from matplotlib.colors import to_rgb

    r, g, b = to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = max(0.0, min(1.0, l * amount))
    return colorsys.hls_to_rgb(h, l, s)


def rose_rmax(datasets, N_BINS=36, normalize=False):
    """Return the tallest rose-plot bin across the supplied data sets."""
    edges = np.linspace(0, 2 * np.pi, N_BINS + 1)
    sequence = datasets.values() if isinstance(datasets, dict) else datasets
    tops = []
    for result in sequence:
        phase = np.radians(np.asarray(result['phase_deg'], dtype=float))
        counts, _ = np.histogram(phase, bins=edges)
        heights = counts / counts.sum() if normalize and counts.sum() else counts.astype(float)
        tops.append(heights.max() if heights.size else 0.0)
    return max(tops, default=0.0)


def _population_plv(result):
    """Length of the mean resultant vector of ``result['phase_deg']``."""
    phase = np.radians(np.asarray(result['phase_deg'], dtype=float))
    return float(np.abs(np.exp(1j * phase).mean())) if phase.size else 0.0


def plot_psd_overlay(psds, labels=None, colors=None, stim_freq=None, normalize=False,
                     log_scale=True, xlim=(0, 20), figsize=(6, 4), state=None, ax=None,
                     legend_title=None, linewidth=1.5, title=None, baseline_psd=None,
                     baseline_db=False, min_baseline=0.0, ylim=None):
    """Overlay power spectral densities that share one frequency grid.

    ``psds`` is a sequence of ``(frequencies, psd)`` pairs as returned by ``utils.get_psd_data``
    (Welch), or a dict mapping label -> pair. The values are plotted as they are, because Welch
    already returns power; ``normalize=True`` divides each trace by its own total power, which
    compares spectral shape rather than absolute power. ``stim_freq`` marks the stimulation
    frequency with a dashed vertical line. ``baseline_psd=(f, Pbase)`` divides each
    PSD by Pbase frequency by frequency, without total-power normalization.
    ``baseline_db=True`` displays 10*log10 of that ratio (use log_scale=False).
    Invalid denominators are masked; min_baseline is an optional absolute cutoff.
    ``ylim`` fixes the y-axis, e.g. ``psd.shared_ylim(...)`` to give a set of plots one scale.
    Returns the figure.
    """
    if isinstance(psds, dict):
        labels = list(psds) if labels is None else labels
        psds = list(psds.values())
    psds = [(np.asarray(f, dtype=float), np.asarray(p, dtype=float)) for f, p in psds]
    if not psds:
        raise ValueError("No spectra to plot.")
    if baseline_psd is not None and normalize:
        raise ValueError("Choose baseline division or total-power normalization, not both")
    if baseline_db and (baseline_psd is None or log_scale):
        raise ValueError("baseline_db requires baseline_psd and log_scale=False")
    grid = psds[0][0]
    for frequencies, power in psds:
        if frequencies.shape != grid.shape or not np.allclose(frequencies, grid):
            raise ValueError("All spectra must share one frequency grid (use the same nperseg).")
        if power.shape != grid.shape:
            raise ValueError("Each PSD must have the same length as its frequency grid.")
    if baseline_psd is not None:
        from src.psd import baseline_psd_ratio
        psds = [baseline_psd_ratio(p, baseline_psd, db=baseline_db,
                                  min_baseline=min_baseline) for p in psds]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    if colors is None:
        colors = [f"C{i % 10}" for i in range(len(psds))]
    if labels is None:
        labels = [f"Spectrum {i + 1}" for i in range(len(psds))]

    for (frequencies, power), color, label in zip(psds, colors, labels):
        if normalize:
            total = power.sum()
            if total <= 0:
                raise ValueError("Cannot normalize a spectrum with zero total power.")
            power = power / total
        plot_fn = ax.semilogy if log_scale else ax.plot
        plot_fn(frequencies, power, color=color, label=label, lw=linewidth)

    if stim_freq:
        ax.axvline(stim_freq, color="0.4", linestyle="--", lw=1, zorder=0)
    if baseline_psd is not None:
        ax.axhline(0 if baseline_db else 1, color="0.4", linestyle="--", lw=1)
    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title(f"{state}, {stim_freq} Hz")
        if state == "Low-correlation" and baseline_psd is None:
            ax.set_ylim(0, 0.0018)
        elif state == "High-correlation" and baseline_psd is None:
            ax.set_ylim(0, 0.03)
    ylabel = "Power / total" if normalize else "PSD (Hz$^2$/Hz)"
    if baseline_psd is not None:
        ylabel = "10 log10(PSD / baseline PSD) [dB]" if baseline_db else "PSD / baseline PSD (fold)"
    ax.set(xlabel="Frequency (Hz)", ylabel=ylabel, xlim=xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.grid(True, alpha=0.3)
    ax.legend(title=legend_title, frameon=False, fontsize=8)
    return fig


def plot_rose_grid(phases, row_labels=None, col_labels=None, colors=None, normalize=True,
                   shared_scale=False, rticks=None, cell=1.4, show_n=False, n_label="n",
                   count_ticks=False, plv_scale=None, **rose_kwargs):
    """Grid of rose plots from ``phases[row][col]`` (DataFrames with ``phase_deg``).

    Works for spike-level phases (one row per spike) and for ``utils.cell_phases`` tables
    (one row per active cell, so ``normalize=False`` bars are numbers of cells).

    * Empty or ``None`` panels are removed. ``colors`` is one colour, one per row, or a
      nested list with one per panel.
    * ``shared_scale=True`` puts every panel on the same radial scale (use with
      ``normalize=False`` to compare counts across panels).
    * ``show_n=True`` writes ``f"{n_label}={n}"`` (number of spikes or cells) in each panel.
    * ``count_ticks`` labels the radial rings with bar heights (counts, or fractions if
      ``normalize=True``): ``True`` on every panel, ``"first"`` only on the first panel
      (enough with ``shared_scale=True``). ``rticks`` (in plot units) applies otherwise.
    * ``plv_scale=None`` uses 0.5, or the largest panel PLV if that is bigger, so PLV lines are
      never clipped; pass a number to fix it.
    * Other keyword arguments (e.g. ``N_BINS``, ``hist_fill``, ``style``) go to ``plot_rose``.

    Returns the figure.
    """
    n_rows, n_cols = len(phases), max(len(row) for row in phases)
    fig, ax = plt.subplots(n_rows, n_cols, subplot_kw={"projection": "polar"},
                           figsize=(n_cols * cell, n_rows * cell), squeeze=False)
    # leave a strip on the left and top for the labels; panels stay flush otherwise
    fig.subplots_adjust(left=0.05, right=1, bottom=0, top=0.94, wspace=0, hspace=0)
    pos = [[ax[r][c].get_position() for c in range(n_cols)] for r in range(n_rows)]
    row_colors = [colors] * n_rows if colors is None or isinstance(colors, str) else list(colors)
    filled = [p for row in phases for p in row if p is not None and len(p)]
    rmax = rose_rmax(filled, N_BINS=rose_kwargs.get("N_BINS", 36), normalize=normalize) if shared_scale else None
    if plv_scale is None:
        plv_scale = max([0.5] + [_population_plv(p) for p in filled])
    first_panel = True
    for r, row in enumerate(phases):
        for c in range(n_cols):
            panel = row[c] if c < len(row) else None
            if panel is None or len(panel) == 0:
                ax[r][c].remove()
                continue
            color = row_colors[r][c] if isinstance(row_colors[r], (list, tuple)) else row_colors[r]
            label_counts = count_ticks is True or (count_ticks == "first" and first_panel)
            plot_rose(panel, ax=ax[r][c], color=color, normalize=normalize, rmax=rmax,
                      plv_scale=plv_scale, count_ticks=label_counts, **rose_kwargs)
            if rticks is not None and not count_ticks:
                ax[r][c].set_yticks(rticks)
            first_panel = False
            if show_n:
                ax[r][c].text(0.0, 0.0, f"{n_label}={len(panel)}", transform=ax[r][c].transAxes, fontsize=7)
    for c, label in enumerate(col_labels or []):
        fig.text(0.5 * (pos[0][c].x0 + pos[0][c].x1), 0.955, label, ha="center", va="center", fontsize=9)
    for r, label in enumerate(row_labels or []):
        fig.text(0.015, 0.5 * (pos[r][0].y0 + pos[r][0].y1), label, ha="center", va="center",
                 fontsize=9, rotation=90,
                 color=row_colors[r] if isinstance(row_colors[r], str) else "k")
    return fig


def plot_rose(results, N_BINS=36, labels=None, normalize=True, style='bar', color=None,
              plv_scale=0.5, hist_fill=0.9, title=None, ax=None, legend=True,
              figsize=(2.75, 2.75), rmax=None, count_ticks=False):
    """Draw the notebook's rose plot, including phase-locking vectors.

    Bars are scaled so the tallest bin reaches ``hist_fill * plv_scale``. Pass a shared
    ``rmax`` (see ``rose_rmax``) to put several panels on one radial scale.
    Histograms remain translucent (bars: 0.65, outlines: 0.9); PLV vectors are opaque.
    ``count_ticks=True`` labels the radial rings with bar heights (counts when
    ``normalize=False``, e.g. numbers of cells for ``utils.cell_phases`` tables).
    """
    from matplotlib.colors import is_color_like

    if isinstance(results, dict):
        items = list(results.items())
    elif isinstance(results, (list, tuple)):
        items = [(f"set {i}", result) for i, result in enumerate(results, 1)]
    else:
        items = [('', results)]
    if labels is not None:
        items = [(label, result) for label, (_, result) in zip(labels, items)]

    edges = np.linspace(0, 2 * np.pi, N_BINS + 1)
    width = np.diff(edges)
    centers = edges[:-1] + width / 2
    if ax is None:
        _, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=figsize)

    default = plt.get_cmap('tab10')(np.arange(len(items)) % 10)
    if color is None:
        colors = default
    elif is_color_like(color):
        colors = [color] * len(items)
    elif isinstance(color, dict):
        colors = [color.get(label, default_color) for (label, _), default_color in zip(items, default)]
    else:
        colors = list(color)
        if len(colors) < len(items):
            colors = [colors[i % len(colors)] for i in range(len(items))]

    hists, zs, ns = [], [], []
    for _, result in items:
        phase = np.radians(np.asarray(result['phase_deg'], dtype=float))
        counts, _ = np.histogram(phase, bins=edges)
        heights = counts / counts.sum() if normalize and counts.sum() else counts.astype(float)
        hists.append(heights)
        zs.append(np.exp(1j * phase).mean() if phase.size else 0j)
        ns.append(phase.size)

    if rmax is None:
        rmax = max((heights.max() for heights in hists if heights.size), default=0.0)
    hists_plot = [heights / rmax * hist_fill * plv_scale for heights in hists] if rmax > 0 else hists

    for (label, _), color_i, heights, z, n in zip(items, colors, hists_plot, zs, ns):
        legend_label = f"{label} (n={n})" if label else f"n={n}"
        if style == 'bar':
            ax.bar(centers, heights, width=width, bottom=0, color=color_i, edgecolor=color_i,
                   linewidth=0.6, alpha=0.65, label=legend_label)
        else:
            ax.plot(np.append(centers, centers[0]), np.append(heights, heights[0]),
                    color=color_i, lw=1.8, alpha=0.9, label=legend_label)

        angle = float(np.angle(z))
        angle_deg = np.degrees(angle)
        print(f"angle={angle:.3f} ({angle_deg:.1f}°)")
        plv = float(np.abs(z))
        print(f"plv={plv:.3f}")
        if plv > plv_scale:
            import warnings
            warnings.warn(f"PLV={plv:.3f} exceeds plv_scale={plv_scale}; line will be clipped")
        ax.plot([angle, angle], [0, plv], color=_darken(color_i, 0.6), lw=2,
                solid_capstyle='round', zorder=3)

    ax.set_ylim(0, plv_scale)
    ax.set_theta_zero_location('E')
    ax.set_theta_direction(1)
    ax.set_xticklabels([])
    if count_ticks and rmax > 0:
        from matplotlib.ticker import MaxNLocator

        ticks = MaxNLocator(nbins=2, integer=not normalize).tick_values(0, rmax)
        ticks = ticks[(ticks > 0) & (ticks <= rmax)]
        ax.set_yticks(ticks / rmax * hist_fill * plv_scale)
        ax.set_yticklabels([f"{tick:g}" for tick in ticks], fontsize=6)
    else:
        ax.set_yticklabels([])
    if title:
        ax.set_title(title, pad=12)
    if len(items) > 1 and legend:
        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1.05), frameon=False)
    return ax

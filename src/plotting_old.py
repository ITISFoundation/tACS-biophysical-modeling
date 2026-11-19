import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from src.utils import get_firingrate_hist

def plot_amplitude_phase(xf_pos, amplitude, phase, freq, idx_base=None):
    """Two-panel plot: amplitude and phase spectra."""
    phase = phase % (2 * np.pi)
    amplitude = amplitude / amplitude[0]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Amplitude
    axes[0].stem(xf_pos[1:], amplitude[1:], basefmt=" ")
    axes[0].set(title='Normalized Amplitude Spectrum', xlabel='Frequency (Hz)', ylabel='Amplitude')
    axes[0].grid(True)

    # Phase
    axes[1].stem(xf_pos[1:], phase[1:], basefmt=" ")
    axes[1].set(title='Phase Spectrum', xlabel='Frequency (Hz)', ylabel='Phase (radians)')
    axes[1].set_yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    axes[1].set_yticklabels(['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
    axes[1].grid(True)

    # if idx_base is None:
    #     power = amplitude ** 2
    #     idx_base = np.argmax(power / np.sum(power))

    # Highlight target frequency
    if freq > 0:
        idx_pos = np.argmin(np.abs(xf_pos - freq))
        axes[0].stem(xf_pos[idx_pos], amplitude[idx_pos], markerfmt='ro', basefmt=" ")
        axes[1].stem(xf_pos[idx_pos], phase[idx_pos], markerfmt='ro', basefmt=" ")
    if idx_base is not None:
        axes[0].stem(xf_pos[idx_base], amplitude[idx_base], markerfmt='go', basefmt=" ")
        axes[1].stem(xf_pos[idx_base], phase[idx_base], markerfmt='go', basefmt=" ")

    _add_legend(fig, freq, xf_pos, idx_base)
    return fig


def plot_power_spectrum(xf_pos, amplitude, freq, idx_base=None):
    """Single-panel plot: total-power-normalised power spectrum."""
    amplitude = amplitude / amplitude[0]
    power = amplitude ** 2
    power_norm = power / np.sum(power)

    # if idx_base is None:
    #     idx_base = np.argmax(power_norm)

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.stem(xf_pos[1:], power_norm[1:], basefmt=" ")
    ax.set(title='Normalized Power Spectrum', xlabel='Frequency (Hz)', ylabel='Power / total')
    ax.grid(True)

    if freq > 0:
        idx_pos = np.argmin(np.abs(xf_pos - freq))
        ax.stem(xf_pos[idx_pos], power_norm[idx_pos], markerfmt='ro', basefmt=" ")

    if idx_base is not None:
        ax.stem(xf_pos[idx_base], power_norm[idx_base], markerfmt='go', basefmt=" ")

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


def plot_fourier_analysis(xf_pos, amplitude, phase, freq, save_path, idx_base=None):
    """Wrapper: saves both figures with suffixed filenames."""
    base, ext = save_path.rsplit('.', 1)

    fig_ap = plot_amplitude_phase(xf_pos, amplitude, phase, freq, idx_base)
    # fig_ap.suptitle(f"{title} — Amplitude & Phase")
    fig_ap.savefig(f"{base}_amp_phase.{ext}")
    plt.close(fig_ap)

    fig_pw = plot_power_spectrum(xf_pos, amplitude, freq, idx_base)
    # fig_pw.suptitle(f"{title} — Power Spectrum")
    fig_pw.savefig(f"{base}_power.{ext}")
    plt.close(fig_pw)

def define_psth_plot(simulation_time, title, n_neurons):
    fig, (ax_raster, ax_stimulation) = plt.subplots(
    2, 1,
    figsize=(14, 12),
    sharex=True,
    gridspec_kw={'height_ratios': [3, 1]}
    )
  
    ax_raster.set_xlabel('Time (ms)', fontsize=20)
    ax_raster.set_ylabel('Neuron IDs', fontsize=20)

    ax_raster.grid(False)
    ax_raster.set_ylim(0, n_neurons)

    ax_hist = ax_raster.twinx()

    ax_hist.set_ylabel('Population firing rate (Hz)', fontsize=20)
    ax_hist.set_ylim(0,1)

    ax_stimulation.set_xlabel('Time (ms)', fontsize=20)
    ax_stimulation.set_ylabel('Amplitude (V/m)', fontsize=20)
    ax_stimulation.set_title('Electric field stimulation', fontsize=20)
    ax_stimulation.set_ylim(-2.2,2.2)

    ax_stimulation.grid()


    for ax in (ax_raster, ax_stimulation):
        ax.set_xlim(0, simulation_time)
        ax.tick_params(axis='both', labelsize=20)

    ax_hist.tick_params(axis='y', labelsize=20)

    ax_raster.tick_params(axis='x', labelbottom=True, bottom=True)
    return fig, (ax_raster, ax_stimulation, ax_hist)

def get_estim_data(simulationData):
    try:
        eStim = simulationData.config["inputs"]['ex_efields']
        Efields = eStim['fields']
        delay = eStim['delay']
        duration = eStim['duration']
        rampup =eStim['ramp_up_time']
        rampdown = eStim['ramp_down_time']
         #in ms the sampling frequency - 10_000 Hz
    

    except Exception as e:
      
        return None

    return Efields,delay,duration,rampup,rampdown

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
            stimulation_func_part = a*np.sin(2*np.pi*Efield['frequency']*1e-3*tseg + (Efield["phase"] + np.pi / 2))

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

def psth_plot_old(spikeData, title, cell_to_rank, simulationData, pop_size=None, bin_width=1, save_histogram=None, smoothed=False, sigma_ms=10):
    #dt = simulationData.dt in ms-1 == make sure this is clear
    #simulation_time = simulationData.tstop
    #spikes have different sampling frequency from stimulation
    simulation_time = simulationData.time_stop
    dt = simulationData.dt
    if pop_size is None: 
        pop_size = cell_to_rank.shape[0]
 
    fig, (ax_raster, ax_stimulation, ax_hist) = define_psth_plot(simulation_time, title, pop_size)

    
    spikeData_ranked = spikeData.map(cell_to_rank)
    spikes = spikeData_ranked.reset_index().to_numpy()
    cell_ids_rank = spikes[:,1]
    spike = spikes[:,0]
    #plot raster plot
    time = np.arange(0, simulation_time, dt)
    ax_raster.scatter(spike, cell_ids_rank, s=0.5, c='black')
    #create histogram of spike train
    firing_rate, bins = get_firingrate_hist(simulationData, spikeData_ranked, bin_width, cell_to_rank.shape[0], delay=0, time_stop=simulation_time)
    # bins, firing_rate = define_histogram(spikeData_ranked, simulation_time, bin_width, cell_to_rank.shape[0])
    ax_hist.step(bins, firing_rate, where='mid', color='C0', label='Population firing rate')

    if smoothed:
        sigma_ms = 10.0                # adjusted for frequency resolution up to 15Hz
        sigma_samples = sigma_ms / bin_width  
        smoothed_fr = gaussian_filter1d(firing_rate, sigma=sigma_samples)
        ax_hist.plot(
        bins,
        smoothed_fr,
        color='C1',
        lw=2,
        label=f'Smoothed firing rate (σ={sigma_ms} ms)'
    )


    stimulation_plot = define_stimulation_function(simulationData, dt)
    ax_stimulation.plot(
        time,
        stimulation_plot,
        color='C2',
        label='Applied stimulation'
    )
    
    plt.tight_layout()
    if save_histogram is not None:
        plt.savefig(save_histogram)
    return fig, (ax_raster, ax_stimulation, ax_hist)
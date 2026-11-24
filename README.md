# Mechanistic insights into state-dependent synchronization of neural activity by weak tACS fields

This repo contains the data and code required to replicate the figures in the paper [Mechanistic insights into state-dependent synchronization of neural activity by weak tACS fields](https://www.biorxiv.org/content/10.64898/2026.11.20.768020) by Habek, Tharayil, et al.

## Overview

This repo provides tools to upload, process, and analyze results of *in silico* simulations of tACS in biophysically-detailed neural circuits. Simulations are configured with `json` files using the [SONATA format](https://github.com/openbraininstitute/sonata-extension) and are run with [Neurodamus](https://github.com/openbraininstitute/neurodamus). The `json` files for each simulation reported in the paper are available in the `data/` folder of this repo, but this repo alone is insufficient to rerun the simulations reported in the paper. In order to rerun the simulations, follow the instructions in the [here](https://github.com/openbraininstitute/neurodamus) to install Neurodamus. Then, follow the instructions in the [Neurodamus ReadME](https://github.com/openbraininstitute/neurodamus) to run each of the simulations specified by the `simulation_config.json` files in this repo.

As running the simulations is computationally expensive, simulation output files are also available in the `data/` folder. Following the instructions below, this repo allows you to import the simulation outputs into the provided `data.csv` file. After a simulation is imported, several electrophysiological features are automatically calculated and added to the database (Note that for the simulations used in the paper, these features are already present in the `data.csv` file, so there is no need to re-import them). The calculated features include phase-locking metrics, firing-rate changes, Fourier-based stimulation-frequency measures, and total power estimates.


## Instructions

Please create a conda environment from the requirements.txt file.

Activate the environment.

### Expected Data Folder Structure

```text
tACS
├── sim10Hz1Vm/
│    ├── reporting/
│    │   └── spikes.h5
│    └── simulation_config.json
│    ...
```

Make sure that each simulation folder contains:

- a `reporting/` folder
- a `spikes.h5` file inside the `reporting/` folder
- a `simulation_config.json` file

The path to each `simulation_config.json` file should match the paths listed in the `source_file` column of the `data.csv` database.

### Pointing the Simulations at Your Circuit

Every simulation config file in `data/` stores the path to the circuit config file on which the simulation was
run (see the [SONATA](https://github.com/openbraininstitute/sonata-extension) documentation for details). In the simulation config files provided in this repo, those paths refer to someone else's machine. Before loading
simulation results in the analysis notebook or re-running any simulation, download your own copy of the circuit from [Zenodo](https://doi.org/10.5281/zenodo.7930276), then
retarget all of the simulation config files to your own copy of the circuit:

```bash
python -m src.set_network_path /path/to/O1_data/circuit_config.json
```

This rewrites the `network` entry in every `data/*/sim_with_stim.json` and
`data/*/sim_with_stim_new.json`, leaving the rest of each file untouched. It
refuses to change anything unless the circuit configuration you name actually
exists, and it is safe to re-run. An absolute path is recommended: a relative one
is resolved from each simulation folder, not from where you launched the command.

Include `simulation_config.json` to retarget the alternate
long-run configuration as well:

```bash
python -m src.set_network_path "$HOME/Downloads/O1_data/circuit_config.json" \
  --filenames sim_with_stim.json sim_with_stim_new.json simulation_config.json
```

Note: The notebook's `node_sets_path` (see below) is a separate setting; changing it alone does
not update the circuit paths stored in the simulation configurations. After
retargeting, restart the notebook kernel and run Setup again to clear cached
simulation objects.

### Uploading a New Simulation

To upload the simulation configuration of a new simulation, run the following command in your terminal:

```bash
python3 run.py
```

Follow the instructions shown in the terminal.

At the end of the upload process, you will be asked to provide:

- the machine used for deployment of the simulation
- the Neurodamus version used
- a unique simulation name

Use a unique simulation name so that the simulation can be found easily, especially if there are multiple simulations with the same signature.

Before uploading there will be a check if there already is a simulation with the written id_name, if there is, It will ask you to choose another name, or if you want to replace this row with this new row.


#### Analysis time window

During the upload process, you will be able to choose the time window over which neural activity metrics are calculated. During this step, you will see a raster plot preview of the chosen simulation and two lines marking the beginning and end of the analysis window.
These two lines will initially be set to the start and end of the stimulation period, but you can choose to define your custom period of analysis by moving the start and end line. Once you are happy with the defined time period, close the window, and the upload process will continue.

#### Automatically Calculated Features

Once a simulation is imported, the following features are automatically calculated and added:

- Phase Locking Value (PLV)<sup>[1](#ref-1)</sup>
- Pairwise Phase Consistency (PPC)<sup>[2](#ref-2)</sup>
- &Delta; firing rate
- Normalized power at the stimulated frequency, divided by total power
- Normalized amplitude at the stimulated frequency, divided by the DC component
- Complex value of the Fourier coefficient at the stimulated frequency
- Total power

These features are calculated for the following cell groups:

- all cells
- pyramidal cells
- inhibitory interneurons
- excitatory interneurons

### Reproducing figures from the paper

All of the figures in the paper can be reproduced by running the notebook `analysis.ipynb`. Note that paths to the downloaded circuit configuration file must be set manually in this notebook; cells where these need to be updated are clearly marked.

The `analysis.ipynb` notebook begins with a **Setup** Section; except as noted below, this section must be run before generating any of the figures. Each figure in the paper is created in a different section of the notebook; each section can be run independently.

In the following subsections, we include some notes on particular choices made during the analysis:

#### Simple Morlet spectrograms

The **Morlet spectrograms: frequency and brain state** section in `analysis.ipynb`
does not produce figures that are included in the paper. The same analysis is also
available in `notebooks/spectrograms.ipynb`; run either independently of Setup.
Set `morlet.node_sets` to the matching circuit node-set file.

Population spikes become a firing-rate signal in 5 ms bins, normalized by the
complete neuron count. For each frequency from 1 to 20 Hz, we convolve the rate with
a zero-mean, unit-energy complex Morlet wavelet and plot its squared magnitude
in dB. Full wavelet support is at least `periods / frequency` seconds (`periods`
must be at least 3). Every time sample is retained. Recording edges are reflected
and displayed without masking; low-frequency edge estimates depend on reflection.
The notebook explains the wavelet, convolution, normalization and time/frequency
tradeoff step by step. The 0.125 Hz plotting grid is not the resolving power.

The main grid compares brain states and stimulation frequencies at the selected
`morlet.amplitude` and `morlet.seed`. The second compares `morlet.amplitudes` and
states at `morlet.stimulus_hz`. Main panels match the standard stimulation timing;
all other eligible runs remain in overview pages. Every plot shares one pooled 2nd–99th percentile colour scale. There are no
baseline ratios, edge-mask controls, or separate export framework.

Figures and two CSV tables (inventory and selection audit) are saved under
`output/morlet_spectrograms/`. Linear power arrays remain available in
`morlet.runs`. The helper module `src/spectrogram_comparisons.py` contains just
three functions: `morlet_power`, `load_runs`, and `plot_grid`. Run its numerical
checks with `python -m unittest discover -s tests -p 'test_spectrogram*.py'`.

#### Power spectral densities

The **Long stimulation: first half, second half and full-period PSD / PLV** section
analyses the simulation entitled `10Hz1Vm_long`; this section can be run indepently of the **Setup** section.
Unlike other simulations reported in this paper, tACS is applied for 2000 ms in the simulation `10Hz1Vm_long`. 
The **Long simulation** section of the notebook compares neural activity in the 1100–2300 ms, 2300–3500 ms and the complete 1100–3500 ms stimulation intervals, with matched
1200 ms Welch segments, an additional whole-window periodogram, and population
spike PLV at 10 Hz. Both PSD and PLV include all three signed pairwise differences.
The section reads timing from the simulation configuration and uses
`LONG_NODE_SETS` for the complete population size. Set `LONG_CONFIG` if the
simulation is stored outside `data/` and the simulation table.
Figures, CSV tables, NPZ spectra and a parameter manifest are saved under
`output/10Hz1Vm_long_windows/`. The full period contains 4.8 cycles at 2 Hz;
each half contains only 2.4, so the low-frequency comparison is descriptive.

The PSD section also exports frequency-wise baseline ratios and dB ratios with
`plot_psd_overlay(..., baseline_psd=(frequencies, baseline_power), log_scale=False)`.
Set `baseline_db=True` for `10*log10(PSD / baseline PSD)`. Ratios use the same-state unstimulated baseline;
1 means unchanged power. Small baseline values can amplify noise, so interpret
ratios alongside absolute PSDs. Zero/invalid denominators are masked.
All dB ratio plots share one y-axis (`ylim=src.psd.shared_ylim(...)`). It spans
every plotted curve between 5 and 20 Hz, rounded outward to 5 dB, so states and
stimulation frequencies can be compared directly.

To regenerate the five state/frequency comparisons from the full-recording
rate arrays saved by the spectrogram command above, run `python -m src.psd`.
PNG, SVG and numerical NPZ outputs go to `output/psd_baseline_ratios/`.
This uses the current notebook's 100-sample Welch setting (2 Hz spacing), 10 ms
Gaussian smoothing, and averages linear PSDs over seeds before division.

#### Notes on Visualization

The analysis notebook uses solid, fully opaque series colors for both network
states, including amplitude markers, connecting lines, and error bars. Phase
distributions in `plot_rose_grid` retain translucent histograms and opaque PLV
vectors. Amplitude markers share Figure 4b's size mapping. Rerun Setup
and the plotting cells after changing the palette to refresh the figures.

#### References

<a id="ref-1"></a>
1. Tran, H., Shirinpour, S., & Opitz, A. (2022). Effects of transcranial alternating current stimulation on spiking activity in computational models of single neocortical neurons. *NeuroImage, 250*, 118953. https://doi.org/10.1016/j.neuroimage.2022.118953
<a id="ref-2"></a>
2. Parasuram, H., Nanda, S., Dura-Bernal, S., Rahimian, A., Kozloski, J., & Lytton, W. W. (2016). Computational modeling of single neuron extracellular electric potentials and network local field potentials using LFPsim. *Frontiers in Computational Neuroscience, 10*, 65. https://doi.org/10.3389/fncom.2016.00065

## Citation

If you use this code, please cite the following paper: 

Habek, E., Tharayil, J., et al. (2026). Mechanistic insights into state-dependent synchronization of neural activity by weak tACS fields. *bioRxiv 2026.11.20.768020*; https://doi.org/10.64898/2026.11.20.768020

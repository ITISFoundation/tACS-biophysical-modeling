# Mechanistic insights into state-dependent synchronization of neural activity by weak tACS fields

This repo contains the data and code required to replicate the figures in the paper **Mechanistic insights into state-dependent synchronization of neural activity by weak tACS fields** by Habek, Tharayil, et al.

## Overview

This project provides tools to upload, process, and analyze simulation results. After a simulation is imported, several electrophysiological features are automatically calculated and added to the database.

The calculated features include phase-locking metrics, firing-rate changes, Fourier-based stimulation-frequency measures, and total power estimates.

This repo alone is insufficient to rerun the simulations reported in the paper. In order to rerun the simulations, follow the instructions in the [here](https://github.com/openbraininstitute/neurodamus) to install Neurodamus. Then, follow the instructions in the [Neurodamus ReadME](https://github.com/openbraininstitute/neurodamus) to run each of the simulations specified by the `simulation_config.json` files in this repo.

## Instructions

Please create a conda environment from the requirements.txt file.

Activate the environment.

### Pointing the Simulations at Your Circuit

Every simulation configuration in `data/` stores the path to the circuit it was
run against, so those paths refer to someone else's machine. Before loading
simulation results in the analysis notebook or re-running any simulation,
retarget them all at your own copy of the circuit:

```bash
python -m src.set_network_path /path/to/O1_data/circuit_config.json
```

This rewrites the `network` entry in every `data/*/sim_with_stim.json` and
`data/*/sim_with_stim_new.json`, leaving the rest of each file untouched. It
refuses to change anything unless the circuit configuration you name actually
exists, and it is safe to re-run. An absolute path is recommended: a relative one
is resolved from each simulation folder, not from where you launched the command.

For this WSL setup, `C:\Users\habek\Downloads\O1_data` is available through
`~/Downloads/O1_data`. Include `simulation_config.json` to retarget the alternate
long-run configuration as well:

```bash
python -m src.set_network_path "$HOME/Downloads/O1_data/circuit_config.json" \
  --filenames sim_with_stim.json sim_with_stim_new.json simulation_config.json
```

The notebook's `node_sets_path` is a separate setting; changing it alone does
not update the circuit paths stored in the simulation configurations. After
retargeting, restart the notebook kernel and run Setup again to clear cached
simulation objects.

### Uploading a New Simulation

To upload the simulation configuration from a new simulation, run the following command in your terminal:

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

Cool example of use:
Maybe you want the same simulation but features from different time frames, just import a simulation 2 times with different id names and defined time frame of feature extraction...

### Automatically Calculated Features

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

### Time frame of analysis

Before doing the feature extraction there will be a raster plot preview of the chosen simulation and two lines.
These two lines will initially be set to the start and end of the stimulation period, but you can choose to define your custom period of analysis by moving the start and end line. Once you are happy with the defined time period. Close.

### Visualization

The analysis notebook uses solid, fully opaque series colors for both network
states, including amplitude markers, connecting lines, and error bars. Phase
distributions in `plot_rose_grid` retain translucent histograms and opaque PLV
vectors. Amplitude markers share Figure 4b's size mapping. Rerun Setup
and the plotting cells after changing the palette to refresh the figures.

### Simple Morlet spectrograms

The **Morlet spectrograms: frequency and brain state** section in `analysis.ipynb`
replaces the previous spectrogram sections. The same five-cell analysis is also
available in `notebooks/spectrograms.ipynb`; run either independently of Setup.
Set `morlet.node_sets` to the matching circuit node-set file.

Population spikes become a firing-rate signal in 5 ms bins, normalized by the
complete neuron count. For each frequency from 1 to 20 Hz, convolve the rate with
a zero-mean, unit-energy complex Morlet wavelet and plot its squared magnitude
in dB. Full wavelet support is at least `periods / frequency` seconds (`periods`
must be at least 3). Every time sample is retained. Recording edges are reflected
and displayed without masking; low-frequency edge estimates depend on reflection.
The notebook explains the wavelet, convolution, normalization and time/frequency
tradeoff step by step. The 0.125 Hz plotting grid is not the resolving power.

The main grid compares brain states and stimulation frequencies at the selected
`morlet.amplitude` and `morlet.seed`. The second compares `morlet.amplitudes` and
states at `morlet.stimulus_hz`. Main panels match the standard stimulation timing;
all other eligible runs remain in overview pages. Named hex0/O1 comparison runs
are excluded, while the `hex0` node set used by other experiments is retained.
Every plot shares one pooled 2nd–99th percentile colour scale. There are no
baseline ratios, edge-mask controls, or separate export framework.

Figures and two CSV tables (inventory and selection audit) are saved under
`output/morlet_spectrograms/`. Linear power arrays remain available in
`morlet.runs`. The helper module `src/spectrogram_comparisons.py` contains just
three functions: `morlet_power`, `load_runs`, and `plot_grid`. Run its numerical
checks with `python -m unittest discover -s tests -p 'test_spectrogram*.py'`.

The older command `python -m src.spectrograms --node-sets PATH` remains available
for legacy rate/NPZ exports consumed by `src.psd`; it is not part of the new
notebook analysis. Its output directory is `output/spectrograms_1_20Hz/`.

### Sandbox for analysis of simulations - analysis.ipynb notebook

The **Long stimulation: first half, second half and full-period PSD / PLV** section
analyses `10Hz1Vm_long` independently of Setup. It compares 1100–2300 ms,
2300–3500 ms and the complete 1100–3500 ms stimulation interval, with matched
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
Set `baseline_db=True` for `10*log10(PSD / baseline PSD)`. The original absolute
PSD figures remain available. Ratios use the same-state unstimulated baseline;
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

The analysis.ipynb notebook showcases how to get the visualizations of the raster plot with the population firing rate, the Fourier spectrum for the defined stimulation period, as well as the Power Spectrum Density plot.

    View this notebook also as a sandbox where you can play with the visualizations and data of the simulations for more custom plots.

### Analysis of multiple simulation, comparison, etc. - GUI interface analysis.py

In case you want to compare multiple simulations logged in the data.csv here you have pre-defined visualizations for comparisons.

....

### Expected Data Folder Structure

```text
tACS
├── sim10Hz1Vm/
│    ├── reporting/
│    │   └── spikes.h5
│    └── simulation_config.json
│    ...
```
### Notes

Make sure that each simulation folder contains:

- a `reporting/` folder
- a `spikes.h5` file inside the `reporting/` folder
- a `simulation_config.json` file

The path to each `simulation_config.json` file should match the paths listed in the `source_file` column of the `data.csv` database.

### References

<a id="ref-1"></a>
1. Tran, H., Shirinpour, S., & Opitz, A. (2022). Effects of transcranial alternating current stimulation on spiking activity in computational models of single neocortical neurons. *NeuroImage, 250*, 118953. https://doi.org/10.1016/j.neuroimage.2022.118953
<a id="ref-2"></a>
2. Parasuram, H., Nanda, S., Dura-Bernal, S., Rahimian, A., Kozloski, J., & Lytton, W. W. (2016). Computational modeling of single neuron extracellular electric potentials and network local field potentials using LFPsim. *Frontiers in Computational Neuroscience, 10*, 65. https://doi.org/10.3389/fncom.2016.00065

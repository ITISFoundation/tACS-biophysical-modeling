# Mechanistic insights into state-dependent synchronization of neural activity by weak tACS fields

This repo contains the data and code required to replicate the figures in the paper [Mechanistic insights into state-dependent synchronization of neural activity by weak tACS fields](https://www.biorxiv.org/content/10.64898/2026.11.20.768020) by Habek, Tharayil, et al.

## Overview

This repo provides tools to import, process, and analyze the results of *in silico* simulations of tACS in biophysically detailed neural circuits. It contains:

- the configuration and spike output of every simulation reported in the paper (`data/`);
- `data.csv`, a table with one row per simulation that lists its stimulation parameters and precomputed electrophysiological features: phase-locking metrics, firing-rate changes, Fourier-based measures at the stimulation frequency, and total power;
- the notebook `analysis.ipynb`, which reproduces every figure in the paper, and the Python code it uses (`src/`);
- a tool that adds new simulations to `data.csv` and calculates their features (`upload/run.py`).

Simulations are configured with `json` files in the [SONATA format](https://github.com/openbraininstitute/sonata-extension) and run with [Neurodamus](https://github.com/openbraininstitute/neurodamus). Running them is computationally expensive, so their outputs are included and you don't need to rerun anything to reproduce the figures. You do need a local copy of the circuit the simulations were run on (see [Pointing the simulations at your circuit](#pointing-the-simulations-at-your-circuit)). To rerun the simulations, see [Re-running the simulations](#re-running-the-simulations-optional).

## Repository layout

| Path | Contents |
|---|---|
| `data/<simulation>/` | One folder per simulation: `sim_with_stim.json` (the SONATA simulation config) and `reporting/spikes.h5` (the spike output) |
| `data.csv` | One row per simulation: stimulation parameters, analysis window, and calculated features |
| `analysis.ipynb` | Reproduces all figures in the paper |
| `notebooks/spectrograms.ipynb` | Morlet spectrograms (not included in the paper) |
| `src/` | Analysis code and command-line tools |
| `upload/run.py` | Adds a simulation to `data.csv` and calculates its features |
| `run_sim.sh` | Runs a list of simulations with Neurodamus, one after another |
| `tests/` | Unit tests |

Each simulation folder has this structure:

```text
data/
├── sim7Hz1Vm/
│   ├── reporting/
│   │   └── spikes.h5
│   └── sim_with_stim.json
...
```

The `source_file` column of `data.csv` gives the path of each simulation's config relative to the repository root (for example, `data/sim7Hz1Vm/sim_with_stim.json`). If you move or add simulations, keep these paths in sync.

## Installation

`requirements.txt` was generated with `pip freeze` from the authors' conda environment. The analysis packages (numpy, scipy, pandas, matplotlib, bluepysnap, and others) are pinned to exact versions. The remaining entries, mainly Jupyter and its dependencies, have the form `package @ file:///…`. They point to conda build folders on the authors' machine that don't exist anywhere else, so `pip install -r requirements.txt` fails as is.

To install, create and activate a conda environment, then install from `requirements.txt` with the local paths removed:

```bash
conda create -n tacs "python>=3.11"
conda activate tacs
pip install -r <(sed -E 's/ @ file:.*$//' requirements.txt)
```

The `sed` command strips the `@ file:///…` part of each local entry. The pinned packages are installed at their exact versions, and the others at the newest versions compatible with them. The `<(…)` syntax works in bash and zsh. The pinned numpy and contourpy versions require Python 3.11 or newer.

## Pointing the simulations at your circuit

Every simulation config in `data/` stores the path to the circuit config file that the simulation was run on: its `network` entry (see the [SONATA](https://github.com/openbraininstitute/sonata-extension) documentation for details). In this repo, those paths point to the authors' machines. Before you load simulation results in the notebooks or rerun a simulation:

1. Download your own copy of the circuit from [Zenodo](https://doi.org/10.5281/zenodo.7930276).

2. Retarget all simulation configs to your copy of the circuit:

   ```bash
   python -m src.set_network_path /path/to/O1_data/circuit_config.json
   ```

   This rewrites the `network` entry in every `data/*/sim_with_stim.json` and `data/*/sim_with_stim_new.json` and leaves the rest of each file untouched. If the circuit config you name doesn't exist, nothing is changed, and the command is safe to re-run. Use an absolute path: a relative path is resolved from each simulation folder, not from the directory you run the command in. To rewrite config files with other names, pass the full list with `--filenames`. To use a folder other than `data/`, pass `--data-dir`.

3. Set the node-set paths in the notebooks to the `node_sets.json` file in the same circuit download. These are separate settings: changing them doesn't update the circuit paths stored in the simulation configs, and step 2 doesn't change them.

   | Variable | Where to set it |
   |---|---|
   | `node_sets_path` | `analysis.ipynb`, in the **Setup** cell marked **MUST BE CONFIGURED MANUALLY FOR YOUR MACHINE** |
   | `LONG_NODE_SETS` | `analysis.ipynb`, in the **Long stimulation** section |
   | `morlet.node_sets` | `notebooks/spectrograms.ipynb`, in the first code cell |

4. If a notebook was already open, restart its kernel and run **Setup** again to clear cached simulation objects.

## Reproducing the figures from the paper

All of the figures in the paper are produced by `analysis.ipynb`. The notebook begins with a **Setup** section, which you need to run first (and again after every kernel restart). Each figure has its own section, and after Setup each section can be run independently. The **Long stimulation** section doesn't need Setup at all.

The notes below explain particular choices made in some sections.

### Long stimulation: first half, second half and full period

The **Long stimulation: first half, second half and full-period PSD / PLV** section analyzes only the simulation `10Hz1Vm_long` (`data/sim10Hz1Vm_long/`). In this simulation tACS is applied for 2400 ms (1100–3500 ms), compared with 1000 ms in the other simulations reported in the paper.

The section compares neural activity in three windows: the first half (1100–2300 ms), the second half (2300–3500 ms), and the complete stimulation period (1100–3500 ms). For each window, it computes:

- a power spectral density (PSD) from matched 1200 ms Welch segments;
- a periodogram over the whole window;
- the population spike PLV at 10 Hz.

For both PSD and PLV, all three signed pairwise differences between windows are reported.

The stimulation timing is read from the simulation config, and the complete population size comes from `LONG_NODE_SETS`. The simulation is located through `data.csv` or the `data/` folder; set `LONG_CONFIG` only if it is stored somewhere else. Figures, CSV tables, NPZ spectra, and a parameter manifest are saved under `output/10Hz1Vm_long_windows/`.

The full period contains 4.8 cycles at 2 Hz, but each half contains only 2.4, so the low-frequency comparison between halves is descriptive only.

### PSD relative to baseline

The **PSD relative to the corresponding baseline** subsection reuses the PSDs computed in the cell just before it. Run the **Power spectra: stimulation amplitudes overlaid per frequency** section from its first cell before this one.

Each stimulated PSD is divided, frequency by frequency, by the unstimulated baseline PSD of the same network state. A ratio of 1 means power is unchanged. Small baseline values can amplify noise, so interpret the ratios alongside the absolute PSDs. Zero or invalid denominators are masked.

The ratios, and the same ratios in dB, are exported with `plot_psd_overlay(..., baseline_psd=(frequencies, baseline_power), log_scale=False)`. Set `baseline_db=True` to plot `10*log10(PSD / baseline PSD)`. All dB plots share one y-axis (`ylim=src.psd.shared_ylim(...)`). It spans every plotted curve between 5 and 20 Hz, rounded outward to the nearest 5 dB, so different states and stimulation frequencies can be compared directly.

To regenerate the five state/frequency comparisons outside the notebook, first save the full-recording firing-rate arrays, then compute the ratios:

```bash
python -m src.spectrograms --node-sets /path/to/O1_data/node_sets.json   # writes output/spectrograms_1_20Hz/
python -m src.psd                                                        # writes output/psd_baseline_ratios/
```

`src.psd` saves PNG, SVG, and numerical NPZ outputs. It uses the notebook's settings: 100-sample Welch segments (2 Hz frequency spacing), 10 ms Gaussian smoothing, and linear PSDs averaged over seeds before division.

### Visualization notes

Both network states are drawn in solid, fully opaque colors, including amplitude markers, connecting lines, and error bars. Phase distributions drawn with `plot_rose_grid` keep translucent histograms and opaque PLV vectors. Amplitude markers use the same size mapping as Figure 4b. After you change the palette, rerun Setup and the plotting cells to refresh the figures.

## Morlet spectrograms (not included in the paper)

`notebooks/spectrograms.ipynb` computes time-frequency spectrograms of population activity. It doesn't depend on `analysis.ipynb`. Before running it, set `morlet.node_sets` to the node-set file of your circuit.

**Method.** The notebook explains each step, including the wavelet, convolution, normalization, and time/frequency tradeoff, in detail:

1. Population spikes are binned into a firing-rate signal with 5 ms bins and normalized by the total number of neurons.
2. For each frequency from 1 to 20 Hz, the rate is convolved with a zero-mean, unit-energy complex Morlet wavelet. The spectrogram shows the squared magnitude of the result in dB.
3. The full wavelet support lasts at least `periods / frequency` seconds, and `periods` must be at least 3.
4. Every time sample is kept. Recording edges are reflected and displayed without masking, so low-frequency estimates near the edges depend on the reflection.

Frequencies are plotted on a 0.125 Hz grid. That is only the plotting grid, not the frequency resolution. No baseline ratios or edge masking are applied.

**Figures.**

- The main grid compares brain states and stimulation frequencies at the amplitude `morlet.amplitude` and seed `morlet.seed`.
- The second grid compares the amplitudes in `morlet.amplitudes` and brain states at the stimulation frequency `morlet.stimulus_hz`.
- The main panels show only runs with the standard stimulation timing. All other eligible runs appear in the overview pages.
- Every plot shares one color scale: the 2nd–99th percentile of the pooled power.

**Outputs.** Figures and two CSV tables (an inventory of runs and a selection audit) are saved under `output/morlet_spectrograms/`. The linear power arrays are kept in `morlet.runs`. The helper module `src/spectrogram_comparisons.py` contains three functions: `morlet_power`, `load_runs`, and `plot_grid`.

## Adding a new simulation to `data.csv`

The simulations used in the paper are already in `data.csv` with all features calculated, so you need this step only for new simulations.

Put the new simulation in a folder with the same structure as those in `data/` (see [Repository layout](#repository-layout)). Then run:

```bash
python upload/run.py
```

This opens a window that walks you through the import:

1. **Choose the simulation config** (`sim_with_stim.json`). Rows are added to `data.csv` by default, and you can choose a different CSV file in the same window.
2. **Choose the analysis window.** A raster plot of the simulation opens, with two lines marking the start and end of the window over which features are calculated. The lines start at the beginning and end of the stimulation period (or at 500 ms and the end of the simulation, if the config defines no `ex_efields` stimulus). Drag them to set a custom window, then close the plot to continue.
3. **Add the row.** The features are calculated and shown in a preview. When you add the row, you are asked for:
   - a unique simulation ID. If the ID is already in the CSV, you can either replace the existing row or choose another ID. A descriptive ID makes the simulation easy to find later, especially when several simulations have the same stimulation parameters.
   - the machine the simulation was run on;
   - the Neurodamus version used.

To import from the command line instead of the window:

```bash
python upload/run.py --import data/<simulation>/sim_with_stim.json \
  --sim_id <unique_id> --deployment <machine> --neurodamus <version>
```

Add `--select-time-frame` to choose the analysis window in the raster plot, or `--analysis-start` and `--analysis-stop` (in ms) to set it directly. Otherwise the default window described above is used. Add `--replace-existing` to overwrite an existing row with the same ID. Run `python upload/run.py --help` to see all options.

### Calculated features

When a simulation is imported, the following features are calculated and added to its row:

| Feature | Column |
|---|---|
| Phase Locking Value (PLV)<sup>[1](#ref-1)</sup> | `plv` |
| Pairwise Phase Consistency (PPC)<sup>[2](#ref-2)</sup> | `ppc` |
| &Delta; firing rate | `deltafr` |
| Power at the stimulation frequency, divided by total power | `power_norm` |
| Amplitude at the stimulation frequency, divided by the DC component | `amplitude_norm` |
| Complex value of the Fourier coefficient at the stimulation frequency | `complex` |
| Total power | `total_power` |

Each feature is calculated for four cell groups. A suffix on the column name identifies the group:

| Cell group | Column suffix | Example |
|---|---|---|
| All cells | *(none)* | `plv` |
| Pyramidal cells | `_PYR` | `plv_PYR` |
| Inhibitory interneurons | `_INT_INH` | `plv_INT_INH` |
| Excitatory interneurons | `_INT_EXC` | `plv_INT_EXC` |

## Re-running the simulations (optional)

This repo alone isn't enough to rerun the simulations. You also need:

1. Neurodamus, installed by following the instructions in the [Neurodamus repository](https://github.com/openbraininstitute/neurodamus).
2. A local copy of the circuit, with the simulation configs retargeted to it (see [Pointing the simulations at your circuit](#pointing-the-simulations-at-your-circuit)).

Then run each simulation's `sim_with_stim.json` as described in the [Neurodamus README](https://github.com/openbraininstitute/neurodamus).

`run_sim.sh` runs several simulations one after another:

1. Edit the `FOLDERS` list at the top of the script. Each folder must contain a `sim_with_stim.json`.
2. Set `NEURODAMUS_PYTHON` to the Neurodamus directory that contains `init.py`.
3. Run the script from the directory that contains the compiled `x86_64/special`, or set `WORKDIR` to that directory:

   ```bash
   NEURODAMUS_PYTHON=/path/to/neurodamus/python bash run_sim.sh
   ```

Optional environment variables: `NP` (number of MPI ranks, default 120), `MPI_EXTRA_ARGS` (extra `mpirun` flags), `STOP_ON_ERROR=1` (stop at the first failed run), and `LOG_DIR` (default `$WORKDIR/sim_logs`).

## Running the tests

```bash
python -m unittest discover -s tests
```

To run only the numerical checks for the spectrogram code:

```bash
python -m unittest discover -s tests -p 'test_spectrogram*.py'
```

## References

<a id="ref-1"></a>
1. Tran, H., Shirinpour, S., & Opitz, A. (2022). Effects of transcranial alternating current stimulation on spiking activity in computational models of single neocortical neurons. *NeuroImage, 250*, 118953. https://doi.org/10.1016/j.neuroimage.2022.118953
<a id="ref-2"></a>
2. Parasuram, H., Nanda, S., Dura-Bernal, S., Rahimian, A., Kozloski, J., & Lytton, W. W. (2016). Computational modeling of single neuron extracellular electric potentials and network local field potentials using LFPsim. *Frontiers in Computational Neuroscience, 10*, 65. https://doi.org/10.3389/fncom.2016.00065

## Citation

If you use this code, please cite the following paper: 

Habek, E., Tharayil, J., et al. (2026). Mechanistic insights into state-dependent synchronization of neural activity by weak tACS fields. *bioRxiv 2026.11.20.768020*; https://doi.org/10.64898/2026.11.20.768020

## License

This code is released under the PolyForm Noncommercial License 1.0.0. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE) for the full terms and for commercial licensing.

#!/usr/bin/env python3
"""
Import a Neurodamus simulation into the simulation table (data.csv).

Running this file without arguments opens a small GUI where you can choose a
`sim_with_stim.json` file. The selected config is parsed, features are
calculated from its spikes over the chosen analysis window, and the result is
appended as a row in a CSV file (data.csv by default).

For headless use and testing, from the repository root:
    python upload/run.py --import data/sim10Hz1Vm_disconnect/sim_with_stim.json
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

SCRIPT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CSV_PATH = SCRIPT_DIR / "data.csv"

# Make ``src`` the project's analysis package (repo-root src/), so imported features are computed
# by the same code as the notebooks and recalculate_data.ipynb.
sys.path.insert(0, str(SCRIPT_DIR))

LOGO_RED = "#e60024"
LOGO_RED_ACTIVE = "#ff1238"
LOGO_RED_PRESSED = "#a8001a"
LOGO_WHITE = "#ffffff"
LOGO_BACKGROUND = "#fff5f7"
LOGO_PANEL = "#ffffff"
LOGO_SOFT_WHITE = "#f9f9f9"
LOGO_TEXT = "#2f2f2f"
LOGO_MUTED_TEXT = "#5f5f5f"
LOGO_DISABLED = "#8c8c8c"
APP_FONT = ("TkDefaultFont", 12)
APP_FONT_BOLD = ("TkDefaultFont", 12, "bold")
BUTTON_FONT = ("TkDefaultFont", 12, "bold")
PREVIEW_FONT = ("TkFixedFont", 12)

DEFAULT_EX_EFIELDS = {
    "delay": 0,
    "duration": 0,
    "node_set": 0,
    "fields": [
        {
            "Ex": 0,
            "Ey": 0,
            "Ez": 0,
            "frequency": 0,
            "phase": 0,
        }
    ],
    "ramp_up_time": 0,
    "ramp_down_time": 0,
}

FEATURE_BIN_MS = 5.0
FEATURE_SMOOTH_SIGMA_MS = 10.0
DEFAULT_ANALYSIS_START_MS = 500.0
FEATURE_MAIN_AXIS = (0.63254108, -0.2055676, 0.74674878)
AnalysisWindow = tuple[float, float]


class ConfigImportError(ValueError):
    """Raised when a simulation config cannot be imported."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_json(path: Path) -> dict[str, Any]:
    import json

    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except FileNotFoundError as exc:
        raise ConfigImportError(f"File not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigImportError(f"Invalid JSON in {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ConfigImportError("Simulation config must be a JSON object.")
    return data


def get_ex_efields(config: dict[str, Any]) -> dict[str, Any]:
    inputs = config.get("inputs")
    if not isinstance(inputs, dict):
        return copy.deepcopy(DEFAULT_EX_EFIELDS)

    ex_efields = inputs.get("ex_efields")
    if not isinstance(ex_efields, dict):
        return copy.deepcopy(DEFAULT_EX_EFIELDS)

    fields = ex_efields.get("fields")
    if fields is not None and not isinstance(fields, list):
        ex_efields = copy.deepcopy(ex_efields)
        ex_efields["fields"] = copy.deepcopy(DEFAULT_EX_EFIELDS["fields"])

    return ex_efields


def path_for_json(path: Path, base: Path | None = None) -> str:
    resolved = path.resolve()
    if base is not None:
        return os.path.relpath(resolved, base.resolve())
    return str(resolved)


def angle_between_3d(a, b):
    dot = a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

    mag_a = math.sqrt(a[0]**2 + a[1]**2 + a[2]**2)
    mag_b = math.sqrt(b[0]**2 + b[1]**2 + b[2]**2)

    if mag_a == 0 or mag_b == 0:
        return 0

    cos_theta = dot / (mag_a * mag_b)
    cos_theta = max(-1, min(1, cos_theta))

    return math.acos(cos_theta)


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def config_time_stop(config: dict[str, Any]) -> float:
    return to_float(config.get("run", {}).get("tstop", 0), 0.0)


def default_analysis_window(
    config: dict[str, Any],
    simulation_time: float | None = None,
) -> AnalysisWindow:
    time_stop = to_float(simulation_time, 0.0) or config_time_stop(config)
    inputs = config.get("inputs")
    ex_efields = inputs.get("ex_efields") if isinstance(inputs, dict) else None

    if isinstance(ex_efields, dict):
        start = to_float(ex_efields.get("delay"), 0.0)
        duration = to_float(ex_efields.get("duration"), 0.0)
        stop = start + duration if duration > 0 else time_stop
    else:
        start = min(DEFAULT_ANALYSIS_START_MS, time_stop) if time_stop > 0 else 0.0
        stop = time_stop

    return normalize_analysis_window(start, stop, time_stop)


def normalize_analysis_window(
    start: float,
    stop: float,
    simulation_time: float | None = None,
) -> AnalysisWindow:
    start = max(0.0, to_float(start, 0.0))
    stop = max(0.0, to_float(stop, 0.0))
    time_stop = to_float(simulation_time, 0.0)

    if time_stop > 0:
        start = min(start, time_stop)
        stop = min(stop, time_stop)
    if stop < start:
        start, stop = stop, start
    if stop <= start and time_stop > start:
        stop = min(time_stop, start + FEATURE_BIN_MS)

    return start, stop


def resolve_analysis_window(
    config: dict[str, Any],
    analysis_window: AnalysisWindow | None = None,
    simulation_time: float | None = None,
) -> AnalysisWindow:
    if analysis_window is None:
        return default_analysis_window(config, simulation_time)
    start, stop = analysis_window
    return normalize_analysis_window(start, stop, simulation_time)


def select_analysis_window(
    config_path: Path,
    config: dict[str, Any],
    parent: tk.Tk | None = None,
) -> AnalysisWindow:
    fallback_window = default_analysis_window(config)

    import matplotlib.pyplot as plt
    from src.plotting import psth_plot
    from src.utils import geom_rank_cell_nodes, load_data
    simulation_data, _all_data, cell_data, spike_data = load_data(config_path)
    if simulation_data is None or cell_data is None or spike_data is None:
        if parent is not None:
            raise ConfigImportError("Could not load simulation data. Something is wrong.")

    try:
        cell_to_rank = geom_rank_cell_nodes(cell_data.copy(), simulation_data)
        start, stop = resolve_analysis_window(config, None, simulation_data.time_stop)
        fig, (ax_raster, ax_stimulation, _ax_hist) = psth_plot(
            spike_data,
            "Choose start/end lines, then close.",
            cell_to_rank,
            simulation_data,
            bin_width=FEATURE_BIN_MS,
            smoothed=True,
        )

    except Exception:
        if parent is not None:
            messagebox.showwarning(
                "Raster preview unavailable",
                "Could not load simulation spike data. Using the default analysis window.",
                parent=parent,
            )
        return fallback_window

    state = {"start": start, "stop": stop, "active": None}
    axes = (ax_raster, ax_stimulation)
    line_groups = {
        "start": [ax.axvline(start, color="green", linewidth=2) for ax in axes],
        "stop": [ax.axvline(stop, color="red", linewidth=2) for ax in axes],
    }
    threshold = max(to_float(simulation_data.time_stop, 0.0) * 0.015, FEATURE_BIN_MS)
    # fig.suptitle("Choose start/end lines, then close.")

    def set_line_x(name: str, x_value: float) -> None:
        state[name] = x_value
        for line in line_groups[name]:
            line.set_xdata([x_value, x_value])

    def on_press(event: Any) -> None:
        if event.xdata is None:
            return
        distances = {
            "start": abs(event.xdata - state["start"]),
            "stop": abs(event.xdata - state["stop"]),
        }
        nearest = min(distances, key=distances.get)
        if distances[nearest] <= threshold:
            state["active"] = nearest

    def on_motion(event: Any) -> None:
        active = state["active"]
        if active is None or event.xdata is None:
            return
        x_value = max(0.0, min(float(event.xdata), float(simulation_data.time_stop)))
        if active == "start":
            x_value = min(x_value, state["stop"] - FEATURE_BIN_MS)
        else:
            x_value = max(x_value, state["start"] + FEATURE_BIN_MS)
        set_line_x(active, x_value)
        fig.canvas.draw_idle()

    def on_release(_event: Any) -> None:
        state["active"] = None

    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("button_release_event", on_release)
    plt.show()

    selected_window = normalize_analysis_window(
        state["start"],
        state["stop"],
        simulation_data.time_stop,
    )
    plt.close(fig)
    return selected_window


def get_smoothed_firing_rate(
    simulation_data: Any,
    spike_data: Any,
    n_neurons: int,
    delay: float,
    time_stop: float,
    bin_width: float,
) -> np.ndarray:
    if n_neurons <= 0 or time_stop <= delay:
        return np.array([], dtype=float)

    from scipy.ndimage import gaussian_filter1d
    from src.utils import get_firingrate_hist

    hist, _ = get_firingrate_hist(
        simulation_data,
        spike_data,
        bin_width=bin_width,
        n_neurons=n_neurons,
        delay=delay,
        time_stop=time_stop,
    )
    hist = np.asarray(hist, dtype=float)
    if hist.size == 0:
        return hist

    sigma_samples = FEATURE_SMOOTH_SIGMA_MS / bin_width
    return gaussian_filter1d(hist, sigma=sigma_samples)


def get_features(
    config_path: Path | None = None,
    analysis_start: float | None = None,
    analysis_stop: float | None = None,
    true_delay: float | None = None,
    bin_width: float = FEATURE_BIN_MS,
) -> dict[str, Any]:
    """Feature columns for one simulation, from ``src.utils.calculate_simulation_features``."""
    from src.utils import FEATURE_COLUMNS, calculate_simulation_features, empty_feature_values

    if config_path is None:
        return empty_feature_values()
    try:
        features = calculate_simulation_features(
            config_path, analysis_start, analysis_stop, true_delay, bin_width
        )
    except ValueError as exc:
        raise ConfigImportError(f"Could not calculate features for {config_path}: {exc}") from exc
    # keep only the columns data.csv stores
    return {column: features.get(column) for column in FEATURE_COLUMNS}


def build_simulation_record(
    config_path: Path,
    config: dict[str, Any],
    output_path: Path = DEFAULT_CSV_PATH,
    analysis_window: AnalysisWindow | None = None,
) -> dict[str, Any]:
    ex_efields = copy.deepcopy(get_ex_efields(config))
    fields = ex_efields.get("fields", [])
    first_field = fields[0] if fields and isinstance(fields[0], dict) else {}

    stim_efs = (first_field.get("Ex", 0), first_field.get("Ey", 0), first_field.get("Ez", 0))
    stim_amp = np.round(float(np.sqrt(stim_efs[0]**2 + stim_efs[1]**2 + stim_efs[2]**2)), 2)    
    stim_orientation = np.round(angle_between_3d(stim_efs, FEATURE_MAIN_AXIS), 4)
    stim_phase = first_field.get("phase", 0)
    stim_freq = first_field.get("frequency", 0) 
    stim_rampup = ex_efields.get("ramp_up_time", 0)
    stim_rampdown = ex_efields.get("ramp_down_time", 0)
    delay =  ex_efields.get("delay", 0)
    duration =  ex_efields.get("duration", 0)
    node_set = ex_efields.get("node_set", 0)
    seed = config.get("run", {}).get("random_seed", 0)
    tstop = config.get("run", {}).get("tstop", 0)
    dt = config.get("run", {}).get("dt", 0)
    node_set = config.get("node_set", 0)
    simulator = config.get("target_simulator", "unknown")
    import_date = datetime.now(timezone.utc).date().isoformat()

    analysis_start, analysis_stop = resolve_analysis_window(config, analysis_window)
    sim_features = get_features(config_path, analysis_start, analysis_stop, delay)
    sim_log ={
        "time_import": import_date,
        "dt": dt,
        "tstop": tstop,
        "seed": seed,
        "simulator": simulator,
        "node_set": node_set,
        "delay": delay,
        "duration": duration,
        "stim_freq": stim_freq,
        "stim_phase": stim_phase,
        "stim_amp": stim_amp,
        "stim_orientation": stim_orientation,
        "stim_rampup": stim_rampup,
        "stim_rampdown": stim_rampdown,
        "analysis_start": analysis_start,
        "analysis_stop": analysis_stop,
        "source_file": path_for_json(config_path, output_path.parent)
    }
    sim_record = sim_log | sim_features

    return sim_record


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, complex):
        if math.isnan(value.real) or math.isnan(value.imag):
            return None
        return f"{value.real}{value.imag:+}j"
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def load_dataframe(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return pd.DataFrame()

    try:
        return pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    except pd.errors.ParserError as exc:
        raise ConfigImportError(f"Could not parse CSV file {csv_path}: {exc}") from exc


def save_dataframe(csv_path: Path, dataframe: pd.DataFrame) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(csv_path, index=False)


def normalized_sim_id(value: Any) -> str:
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    if value is None:
        return ""
    return str(value).strip()


def find_existing_sim_id(dataframe: pd.DataFrame, sim_id: str) -> int | None:
    sim_id = normalized_sim_id(sim_id)
    if dataframe.empty or "sim_id" not in dataframe.columns or not sim_id:
        return None

    matches = dataframe["sim_id"].map(normalized_sim_id) == sim_id
    if not matches.any():
        return None
    return int(matches[matches].index[0])


def existing_sim_id_row(csv_path: Path, sim_id: str) -> int | None:
    return find_existing_sim_id(load_dataframe(csv_path), sim_id)


def align_row_to_dataframe(
    dataframe: pd.DataFrame,
    row: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    for column in row.columns:
        if column not in dataframe.columns:
            dataframe[column] = pd.NA
    for column in dataframe.columns:
        if column not in row.columns:
            row[column] = pd.NA
    if len(dataframe.columns) > 0:
        row = row[list(dataframe.columns)]
    return dataframe, row


def with_deployment_metadata(
    record: dict[str, Any],
    sim_id: str,
    deployment: str,
    neurodamus: str,
) -> dict[str, Any]:
    record_with_metadata = {
        "sim_id": sim_id.strip(),
        **copy.deepcopy(record),
    }
    record_with_metadata["deployment"] = deployment.strip()
    record_with_metadata["neurodamus"] = neurodamus.strip()

    return record_with_metadata


def append_record(
    csv_path: Path,
    record: dict[str, Any],
    replace_existing: bool = False,
) -> tuple[pd.DataFrame, int]:
    dataframe = load_dataframe(csv_path)
    row = pd.DataFrame.from_records([json_ready(record)])
    if dataframe.empty and len(dataframe.columns) == 0:
        save_dataframe(csv_path, row)
        return row, 1

    dataframe, row = align_row_to_dataframe(dataframe, row)
    if replace_existing:
        row_sim_id = normalized_sim_id(row.iloc[0].get("sim_id"))
        existing_index = find_existing_sim_id(dataframe, row_sim_id)
        if existing_index is not None:
            dataframe.loc[existing_index, dataframe.columns] = row.iloc[0]
            save_dataframe(csv_path, dataframe)
            return dataframe, existing_index + 1

    dataframe = pd.concat([dataframe, row], ignore_index=True)
    save_dataframe(csv_path, dataframe)
    return dataframe, len(dataframe)


def import_config(
    config_path: Path,
    csv_path: Path,
    sim_id: str = "",
    deployment: str = "",
    neurodamus: str = "",
    analysis_window: AnalysisWindow | None = None,
    replace_existing: bool = False,
) -> tuple[dict[str, Any], int]:
    if not normalized_sim_id(sim_id):
        raise ConfigImportError("A unique simulation ID is required.")
    existing_index = existing_sim_id_row(csv_path, sim_id)
    if existing_index is not None and not replace_existing:
        row_number = existing_index + 1
        raise ConfigImportError(
            f"Simulation ID '{sim_id}' already exists in row {row_number}. "
            "Choose another ID or replace the existing row."
        )

    config = load_json(config_path)
    record = build_simulation_record(config_path, config, csv_path, analysis_window)
    record = with_deployment_metadata(record, sim_id, deployment, neurodamus)
    _, row_number = append_record(csv_path, record, replace_existing=replace_existing)
    return record, row_number


def format_preview(record: dict[str, Any]) -> str:
    import json

    return json.dumps(json_ready(record), indent=2)


class SimulationImporterApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("OBP neurodamus stimulation config importer for tACS")
        self.root.minsize(820, 600)
        self._configure_theme()

        self.config_path = tk.StringVar()
        self.database_path = tk.StringVar(value=str(DEFAULT_CSV_PATH))
        self.status = tk.StringVar(value="Choose a simulation config CSV file.")
        self.pending_record: dict[str, Any] | None = None
        self.pending_analysis_window: AnalysisWindow | None = None

        self._build_widgets()

    def _configure_theme(self) -> None:
        self.root.configure(background=LOGO_BACKGROUND)
        self.root.option_add("*selectBackground", LOGO_RED)
        self.root.option_add("*selectForeground", LOGO_WHITE)

        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure(
            ".",
            background=LOGO_BACKGROUND,
            foreground=LOGO_TEXT,
            fieldbackground=LOGO_PANEL,
            bordercolor=LOGO_RED,
            lightcolor=LOGO_RED,
            darkcolor=LOGO_RED,
            troughcolor=LOGO_SOFT_WHITE,
            font=APP_FONT,
        )
        style.configure(
            "Logo.TLabelframe",
            background=LOGO_BACKGROUND,
            bordercolor=LOGO_RED,
            relief="solid",
        )
        style.configure(
            "Logo.TLabelframe.Label",
            background=LOGO_BACKGROUND,
            foreground=LOGO_RED,
            font=APP_FONT_BOLD,
        )
        style.configure("Logo.TFrame", background=LOGO_BACKGROUND)
        style.configure(
            "Logo.TLabel",
            background=LOGO_BACKGROUND,
            foreground=LOGO_MUTED_TEXT,
            font=APP_FONT,
        )
        style.configure(
            "Logo.TEntry",
            bordercolor=LOGO_RED,
            fieldbackground=LOGO_PANEL,
            foreground=LOGO_TEXT,
            insertcolor=LOGO_RED,
            lightcolor=LOGO_RED,
            darkcolor=LOGO_RED,
            padding=6,
            font=APP_FONT,
        )
        style.map(
            "Logo.TEntry",
            bordercolor=[("focus", LOGO_RED_ACTIVE), ("!focus", LOGO_RED)],
            fieldbackground=[("readonly", LOGO_SOFT_WHITE), ("!disabled", LOGO_PANEL)],
            foreground=[("disabled", LOGO_DISABLED), ("!disabled", LOGO_TEXT)],
        )
        style.configure(
            "Logo.TButton",
            background=LOGO_RED,
            bordercolor=LOGO_RED,
            focusthickness=2,
            focuscolor=LOGO_RED_ACTIVE,
            foreground=LOGO_WHITE,
            padding=(14, 8),
            font=BUTTON_FONT,
        )
        style.map(
            "Logo.TButton",
            background=[
                ("pressed", LOGO_RED_PRESSED),
                ("active", LOGO_RED_ACTIVE),
                ("disabled", LOGO_DISABLED),
                ("!disabled", LOGO_RED),
            ],
            bordercolor=[
                ("pressed", LOGO_RED_PRESSED),
                ("active", LOGO_WHITE),
                ("!disabled", LOGO_RED),
            ],
            foreground=[("disabled", "#cfcfcf"), ("!disabled", LOGO_WHITE)],
        )
        for scrollbar_style in (
            "Logo.Vertical.TScrollbar",
            "Logo.Horizontal.TScrollbar",
        ):
            style.configure(
                scrollbar_style,
                arrowcolor=LOGO_WHITE,
                background=LOGO_RED,
                bordercolor=LOGO_PANEL,
                darkcolor=LOGO_RED,
                lightcolor=LOGO_RED,
                troughcolor=LOGO_SOFT_WHITE,
            )
            style.map(
                scrollbar_style,
                background=[("active", LOGO_RED_ACTIVE), ("!disabled", LOGO_RED)],
            )

    def _build_widgets(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(2, weight=1)

        file_frame = ttk.LabelFrame(
            self.root,
            text="Simulation config",
            style="Logo.TLabelframe",
        )
        file_frame.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 6))
        file_frame.columnconfigure(0, weight=1)

        config_entry = ttk.Entry(
            file_frame,
            textvariable=self.config_path,
            style="Logo.TEntry",
        )
        config_entry.grid(row=0, column=0, sticky="ew", padx=(8, 6), pady=8)

        browse_button = ttk.Button(
            file_frame,
            text="Choose...",
            command=self.choose_config_file,
            style="Logo.TButton",
        )
        browse_button.grid(row=0, column=1, padx=(0, 8), pady=8)

        database_frame = ttk.LabelFrame(
            self.root,
            text="CSV data file",
            style="Logo.TLabelframe",
        )
        database_frame.grid(row=1, column=0, sticky="ew", padx=12, pady=6)
        database_frame.columnconfigure(0, weight=1)

        database_entry = ttk.Entry(
            database_frame,
            textvariable=self.database_path,
            style="Logo.TEntry",
        )
        database_entry.grid(row=0, column=0, sticky="ew", padx=(8, 6), pady=8)

        database_button = ttk.Button(
            database_frame,
            text="Change...",
            command=self.choose_database_file,
            style="Logo.TButton",
        )
        database_button.grid(row=0, column=1, padx=(0, 8), pady=8)

        preview_frame = ttk.LabelFrame(
            self.root,
            text="Extracted data preview",
            style="Logo.TLabelframe",
        )
        preview_frame.grid(row=2, column=0, sticky="nsew", padx=12, pady=6)
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(0, weight=1)

        self.preview = tk.Text(
            preview_frame,
            wrap="none",
            height=18,
            state="disabled",
            font=PREVIEW_FONT,
            background=LOGO_PANEL,
            foreground=LOGO_TEXT,
            insertbackground=LOGO_RED,
            selectbackground=LOGO_RED,
            selectforeground=LOGO_WHITE,
            highlightbackground=LOGO_RED,
            highlightcolor=LOGO_RED_ACTIVE,
            highlightthickness=1,
            borderwidth=0,
            padx=12,
            pady=12,
        )
        self.preview.grid(row=0, column=0, sticky="nsew")

        y_scroll = ttk.Scrollbar(
            preview_frame,
            orient="vertical",
            command=self.preview.yview,
            style="Logo.Vertical.TScrollbar",
        )
        y_scroll.grid(row=0, column=1, sticky="ns")
        self.preview.configure(yscrollcommand=y_scroll.set)

        x_scroll = ttk.Scrollbar(
            preview_frame,
            orient="horizontal",
            command=self.preview.xview,
            style="Logo.Horizontal.TScrollbar",
        )
        x_scroll.grid(row=1, column=0, sticky="ew")
        self.preview.configure(xscrollcommand=x_scroll.set)

        footer = ttk.Frame(self.root, style="Logo.TFrame")
        footer.grid(row=3, column=0, sticky="ew", padx=12, pady=(6, 12))
        footer.columnconfigure(0, weight=1)

        status_label = ttk.Label(
            footer,
            textvariable=self.status,
            style="Logo.TLabel",
        )
        status_label.grid(row=0, column=0, sticky="w")

        self.import_button = ttk.Button(
            footer,
            text="Add Row",
            command=self.import_selected_config,
            state="disabled",
            style="Logo.TButton",
        )
        self.import_button.grid(row=0, column=1, padx=(8, 0))

        close_button = ttk.Button(
            footer,
            text="Close",
            command=self.root.destroy,
            style="Logo.TButton",
        )
        close_button.grid(row=0, column=2, padx=(8, 0))

    def choose_config_file(self) -> None:
        selected = filedialog.askopenfilename(
            title="Choose simulation config",
            initialdir=str(Path.cwd()),
            filetypes=[
                ("JSON files", "*.json"),
                ("All files", "*"),
            ],
        )
        if not selected:
            return
        self.config_path.set(selected)
        self.pending_analysis_window = None
        self.load_selected_config()

    def choose_database_file(self) -> None:
        selected = filedialog.asksaveasfilename(
            title="Choose CSV data file",
            initialdir=str(Path(self.database_path.get()).resolve().parent),
            initialfile=Path(self.database_path.get()).name,
            defaultextension=".csv",
            filetypes=[
                ("CSV files", "*.csv"),
                ("All files", "*"),
            ],
        )
        if not selected:
            return
        self.database_path.set(selected)
        if self.config_path.get():
            self.load_selected_config()

    def load_selected_config(self) -> None:
        config_path = Path(self.config_path.get()).expanduser()
        csv_path = Path(self.database_path.get()).expanduser()

        try:
            config = load_json(config_path)
            if self.pending_analysis_window is None:
                self.status.set("Select the analysis time frame, then close the raster plot.")
                self.root.update_idletasks()
                self.pending_analysis_window = select_analysis_window(
                    config_path,
                    config,
                    self.root,
                )
            self.status.set("Calculating features for the selected time frame...")
            self.root.update_idletasks()
            self.pending_record = build_simulation_record(
                config_path,
                config,
                csv_path,
                self.pending_analysis_window,
            )
        except ConfigImportError as exc:
            self.pending_record = None
            self.import_button.configure(state="disabled")
            self.set_preview("")
            self.status.set(str(exc))
            messagebox.showerror("Could not load config", str(exc))
            return

        self.set_preview(format_preview(self.pending_record))
        self.import_button.configure(state="normal")
        self.status.set("Ready to add this simulation as a CSV row.")

    def import_selected_config(self) -> None:
        if self.pending_record is None:
            self.status.set("Choose a valid simulation config first.")
            return

        csv_path = Path(self.database_path.get()).expanduser()
        sim_id, replace_existing = self.ask_unique_sim_id(csv_path)
        if sim_id is None:
            self.status.set("Add row cancelled.")
            return

        deployment = self.ask_required_text(
            title="Deployment machine",
            prompt="Name of the deployment machine used for this simulation",
        )
        if deployment is None:
            self.status.set("Add row cancelled.")
            return

        neurodamus = self.ask_required_text(
            title="Neurodamus version",
            prompt="Neurodamus package version used for this simulation",
        )
        if neurodamus is None:
            self.status.set("Add row cancelled.")
            return

        record = with_deployment_metadata(
            self.pending_record,
            sim_id,
            deployment,
            neurodamus,
        )
        try:
            _, row_number = append_record(
                csv_path,
                record,
                replace_existing=replace_existing,
            )
        except ConfigImportError as exc:
            self.status.set(str(exc))
            messagebox.showerror("Could not add row", str(exc))
            return
        except OSError as exc:
            self.status.set(str(exc))
            messagebox.showerror("Could not write CSV", str(exc))
            return

        action = "Replaced" if replace_existing else "Added"
        message = f"{action} CSV row {sim_id} in {csv_path}"
        self.status.set(message)
        self.set_preview(format_preview(record))
        messagebox.showinfo("Row saved", message)

    def ask_unique_sim_id(self, csv_path: Path) -> tuple[str | None, bool]:
        while True:
            sim_id = self.ask_required_text(
                title="Simulation ID",
                prompt="Give your simulation a unique identifier (example: 10Hz1Vm-disconnect)",
            )
            if sim_id is None:
                return None, False

            try:
                existing_index = existing_sim_id_row(csv_path, sim_id)
            except ConfigImportError as exc:
                messagebox.showerror("Could not check simulation ID", str(exc))
                return None, False

            if existing_index is None:
                return sim_id, False

            row_number = existing_index + 1
            replace_existing = messagebox.askyesnocancel(
                "Simulation ID already exists",
                (
                    f"'{sim_id}' already exists in row {row_number}.\n\n"
                    "Yes: replace that row\n"
                    "No: choose another name\n"
                    "Cancel: stop"
                ),
                parent=self.root,
            )
            if replace_existing is None:
                return None, False
            if replace_existing:
                return sim_id, True

    def ask_required_text(self, title: str, prompt: str) -> str | None:
        while True:
            value = simpledialog.askstring(title, prompt, parent=self.root)
            if value is None:
                return None
            value = value.strip()
            if value:
                return value
            messagebox.showerror(title, "Please enter a value before adding the row.")

    def set_preview(self, value: str) -> None:
        self.preview.configure(state="normal")
        self.preview.delete("1.0", tk.END)
        self.preview.insert("1.0", value)
        self.preview.configure(state="disabled")


def run_gui() -> int:
    root = tk.Tk()
    SimulationImporterApp(root)
    root.mainloop()
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Open a GUI to import simulation config stimulation data, or import "
            "one config from the command line."
        )
    )
    parser.add_argument(
        "--import",
        dest="import_path",
        type=Path,
        help="simulation config JSON file to import without opening the GUI",
    )
    parser.add_argument(
        "--csv",
        "--database",
        dest="csv_path",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help=f"CSV data file path (default: {DEFAULT_CSV_PATH})",
    )

    parser.add_argument(
        "--sim_id",
        default=None,
        help="Unique identifier for this simulation",
    )

    parser.add_argument(
        "--deployment",
        default=None,
        help="deployment machine name to store in the CSV row",
    )
    parser.add_argument(
        "--neurodamus",
        default=None,
        help="Neurodamus package version to store in the CSV row",
    )
    parser.add_argument(
        "--analysis-start",
        type=float,
        default=None,
        help="analysis window start time in ms",
    )
    parser.add_argument(
        "--analysis-stop",
        type=float,
        default=None,
        help="analysis window stop time in ms",
    )
    parser.add_argument(
        "--select-time-frame",
        action="store_true",
        help="open the raster preview to choose the analysis window",
    )
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="replace the existing row when --sim_id already exists",
    )
    parser.add_argument(
        "--print-record",
        action="store_true",
        help="print the extracted record after importing",
    )
    return parser.parse_args(argv)


def resolve_cli_sim_id(
    csv_path: Path,
    sim_id: str | None,
    replace_existing: bool,
) -> tuple[str, bool]:
    while True:
        if not normalized_sim_id(sim_id):
            if not sys.stdin.isatty():
                raise ConfigImportError("A unique simulation ID is required.")
            sim_id = input("Simulation ID: ").strip()
            continue

        existing_index = existing_sim_id_row(csv_path, sim_id)
        if existing_index is None:
            return sim_id, False
        if replace_existing:
            return sim_id, True

        row_number = existing_index + 1
        if not sys.stdin.isatty():
            raise ConfigImportError(
                f"Simulation ID '{sim_id}' already exists in row {row_number}. "
                "Use --replace-existing or choose another ID."
            )

        answer = input(
            f"Simulation ID '{sim_id}' already exists in row {row_number}. "
            "Replace it? [y/N/new/cancel]: "
        ).strip().lower()
        if answer in {"y", "yes", "replace", "r"}:
            return sim_id, True
        if answer in {"new", "n", "no", ""}:
            sim_id = input("New simulation ID: ").strip()
            continue
        raise ConfigImportError("Import cancelled.")


def cli_analysis_window(args: argparse.Namespace) -> AnalysisWindow | None:
    if args.analysis_start is not None or args.analysis_stop is not None:
        if args.analysis_start is None or args.analysis_stop is None:
            raise ConfigImportError(
                "Use both --analysis-start and --analysis-stop, or neither."
            )
        return normalize_analysis_window(args.analysis_start, args.analysis_stop)

    if args.select_time_frame:
        config = load_json(args.import_path)
        return select_analysis_window(args.import_path, config)

    return None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])

    if args.import_path is None:
        return run_gui()

    sim_id = args.sim_id
    deployment = args.deployment
    neurodamus = args.neurodamus
    if deployment is None and sys.stdin.isatty():
        deployment = input("Deployment machine: ").strip()
    if neurodamus is None and sys.stdin.isatty():
        neurodamus = input("Neurodamus package version: ").strip()

    try:
        sim_id, replace_existing = resolve_cli_sim_id(
            args.csv_path,
            sim_id,
            args.replace_existing,
        )
        analysis_window = cli_analysis_window(args)
        record, row_number = import_config(
            args.import_path,
            args.csv_path,
            sim_id,
            deployment or "",
            neurodamus or "",
            analysis_window=analysis_window,
            replace_existing=replace_existing,
        )
    except ConfigImportError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"Error writing CSV: {exc}", file=sys.stderr)
        return 1

    action = "Replaced" if replace_existing else "Added"
    print(f"{action} CSV row {row_number} in {args.csv_path}")
    if args.print_record:
        print(format_preview(record))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
Runtime comparison of the HBM variants and MPM (paper: Fig. 14).

Strip plot (log scale) of the wall time of every push, per model, with the median as a black bar.
HBM times come from TIMINGS_PUSH_LONG.csv (written by model_eval.py --push). MPM times come from
the timed MPM re-runs (<task>/timed_run/raw_telemetry.csv) and count only the pushing motion
(found from the spatula trajectory in <task>/data_from_mpm/).

Input:  one or several push-evaluation folders.
Output: <out-dir>/Comparison/PDF_Plots/<view>_strip.pdf and <out-dir>/Comparison/CSV_Data/<view>_data.csv,
        one view per input folder and task, plus all 1-step and all 2-step pushes pooled.

Usage:
    python plotting/plot_push_times.py --inputs <folder> [<folder> ...] --tags <tag> [<tag> ...] --out-dir <folder>
"""

import os
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================================================================
# CONFIGURATION
# =========================================================================

FIG_SIZE = (3.5, 1.8) # Figure size (width, height) in inches

TIME_AXIS_RANGE = (1e-4, 1e2)  # Wall-time axis limits [s] (log scale)

# Model names in the CSVs -> labels in the plot
MODEL_MAP = {
    "MPM-Timed": "MPM",
}

# Models to plot, top to bottom (use the plot labels, i.e. the right side of MODEL_MAP)
ACTIVE_MODELS = ['PS', 'PW', 'PSE', 'PWE', 'MPM']
# reversed: the first model ends up at the top of the horizontal plot
ACTIVE_MODELS = ACTIVE_MODELS[::-1]

# Colour per model
MODEL_COLORS = {
    'PS':   '#0072B2',    # blue
    'PW':   '#E69F00',    # amber
    'PSE':  '#009E73',    # green
    'PWE':  '#D55E00',    # vermillion
    'MPM':  '#CC79A7',    # purple
}

# Task names (HBM CSV and MPM folders) -> labels
TASK_MAP = {
    "TaskA_Sweep1": "Start-to-Mid (1-step)",
    "taskA":        "Start-to-Mid (1-step)",
    "TaskB_Sweep2": "Mid-to-End (1-step)",
    "taskB":        "Mid-to-End (1-step)",
    "TaskC_Chained": "Start-to-End (2-step)",
    "taskC":        "Start-to-End (2-step)"
}

# Output subfolders
OUTPUT_CONFIG = {
    'comparison_dir': 'Comparison',
    'csv_subdir': 'CSV_Data',
    'pdf_subdir': 'PDF_Plots',
    'dpi': 150
}

# Matplotlib styling
PLOT_RC_PARAMS = {
    'font.size': 8,
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'axes.titlesize': 3,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'legend.fontsize': 8,
    'axes.linewidth': 0.1,
    'font.family': 'serif',
    'axes.titlepad': 10,
    'pdf.fonttype': 42,  # Force TrueType fonts in PDFs (Fixes Type 3 errors)
    'ps.fonttype': 42    # Force TrueType fonts in PostScript
    }

MARKER_SIZE = 15       # Size of the data points

# =========================================================================
# DATA EXTRACTION FUNCTIONS
# =========================================================================

def get_kinematic_phases(meta_path, traj_path):
    """Parses the trajectory to find the start and end steps of active pushing."""
    phases = []
    if os.path.exists(meta_path) and os.path.exists(traj_path):
        with open(meta_path, 'r') as f: meta_mpm = json.load(f)
        with open(traj_path, 'r') as f: traj_data = json.load(f)

        safe_xy = np.array(meta_mpm.get("spatula_safe_pose", [0.0, 0.0])[:2])
        traj = {int(k): v for k, v in traj_data.items()}

        in_field, start_step = False, 0
        for step in sorted(traj.keys()):
            dist = np.linalg.norm(np.array(traj[step])[:2] - safe_xy)
            if dist > 0.001 and not in_field:
                in_field, start_step = True, step
            elif dist <= 0.001 and in_field:
                in_field = False
                phases.append((start_step, step))
        if in_field:
            phases.append((start_step, sorted(traj.keys())[-1]))
    return phases

def extract_phase_durations(df, phases):
    """Sums up wall time, sim time, and steps for all active pushing phases."""
    wall_duration = sim_duration = 0.0
    step_duration = 0

    if df.empty or not phases:
        return np.nan, np.nan, np.nan

    for start_step, end_step in phases:
        idx_start = (np.abs(df['step'] - start_step)).argmin()
        idx_end = (np.abs(df['step'] - end_step)).argmin()

        wall_duration += (df.iloc[idx_end]['timestamp'] - df.iloc[idx_start]['timestamp'])
        sim_duration += (df.iloc[idx_end]['sim_time'] - df.iloc[idx_start]['sim_time'])
        step_duration += (df.iloc[idx_end]['step'] - df.iloc[idx_start]['step'])

    return wall_duration, sim_duration, step_duration

# =========================================================================
# PLOTTING
# =========================================================================

def plot_comparison_strip(df, view_name, out_dir, model_order):
    """Plots individual data points (strip plot) with a distinct median marker."""
    if df.empty: return

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.set_xscale('log')

    scenarios = df['Scenario'].unique()
    indices_base = np.arange(len(model_order))

    n_scenarios = len(scenarios)
    # Use subtle offsets if multiple scenarios are plotted together
    offsets = [0.0] if n_scenarios == 1 else ([-0.1, 0.1] if n_scenarios == 2 else np.linspace(-0.15, 0.15, n_scenarios))

    for i, scenario in enumerate(scenarios):
        scenario_df = df[df['Scenario'] == scenario]

        for idx, model in enumerate(model_order):
            model_data = scenario_df[scenario_df['Model'] == model]['Total_Time'].dropna()
            if model_data.empty: continue

            model_color = MODEL_COLORS.get(model, '#7F7F7F')

            # 1. Compute the robust median
            median_val = model_data.median()

            # 2. Add slight random jitter to prevent points from stacking invisibly
            jitter = np.random.uniform(-0.04, 0.04, size=len(model_data))
            positions = indices_base[idx] + offsets[i] + jitter

            # Individual data points
            ax.scatter(model_data, positions,
                       alpha=0.6, color=model_color, edgecolors='none',
                       s=MARKER_SIZE * 3, zorder=2)

            # Median as a black bar
            ax.plot([median_val, median_val], [indices_base[idx] + offsets[i] - 0.08, indices_base[idx] + offsets[i] + 0.08],
                    color='black', linewidth=1.5, zorder=3)

    ax.set_yticks(indices_base)
    ax.set_yticklabels(model_order)
    ax.set_xlim(TIME_AXIS_RANGE)
    ax.set_xlabel("Wall Time [s]")
    ax.set_ylabel("Model")
    ax.invert_yaxis()

    ax.grid(True, which="major", axis="x", ls="-", alpha=0.4, linewidth=0.4)
    ax.grid(True, which="minor", axis="x", ls="--", alpha=0.2, linewidth=0.2)

    # Time axis at the top
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")

    ax.set_title(f"Performance Comparison\n{view_name.replace('_', ' ')}")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{view_name}_strip.pdf"), dpi=OUTPUT_CONFIG['dpi'])
    plt.close()

# =========================================================================
# MAIN EXECUTION
# =========================================================================

def main():
    # Apply standard text styling
    plt.rcParams.update(PLOT_RC_PARAMS)

    parser = argparse.ArgumentParser(description="Plot the HBM and MPM runtimes per push.")
    parser.add_argument("--inputs", nargs='+', required=True, help="Push-evaluation folders")
    parser.add_argument("--tags", nargs='+', help="Optional label per input folder (default: Exp_1, Exp_2, ...)")
    parser.add_argument("--out-dir", required=True, help="Output folder")
    args = parser.parse_args()

    # Labels: given tags, or "Exp_1", "Exp_2", ...
    tags = args.tags if args.tags else [f"Exp_{i+1}" for i in range(len(args.inputs))]

    # ---------------------------------------------------------
    # Output folders
    # ---------------------------------------------------------
    base_dir = args.out_dir
    dir_comp = os.path.join(base_dir, OUTPUT_CONFIG['comparison_dir'])

    dir_comp_csv = os.path.join(dir_comp, OUTPUT_CONFIG['csv_subdir'])
    dir_comp_pdf = os.path.join(dir_comp, OUTPUT_CONFIG['pdf_subdir'])

    os.makedirs(dir_comp, exist_ok=True)
    os.makedirs(dir_comp_csv, exist_ok=True)
    os.makedirs(dir_comp_pdf, exist_ok=True)

    comp_records = []

    # ---------------------------------------------------------
    # Collect the runtimes
    # ---------------------------------------------------------
    for folder, tag in zip(args.inputs, tags):

        # A. HBM
        hbm_csv = os.path.join(folder, "TIMINGS_PUSH_LONG.csv")
        if os.path.exists(hbm_csv):
            df_hbm = pd.read_csv(hbm_csv)
            for _, row in df_hbm.iterrows():
                task_mapped = TASK_MAP.get(str(row['Task']).strip(), row['Task'])
                model_mapped = MODEL_MAP.get(row['Model'], row['Model'])
                comp_records.append({
                    'Scenario': tag, 'Task': task_mapped, 'Run': row['Run'],
                    'Model': model_mapped,
                    'Total_Time': row.get('Push_Time_sec', 0.0) + row.get('Erosion_Time_sec', 0.0)
                })

        # B. MPM (timed re-runs)
        mpm_staging = os.path.join(folder, "mpm_push_staging")
        if not os.path.exists(mpm_staging): continue

        for run_name in os.listdir(mpm_staging):
            run_path = os.path.join(mpm_staging, run_name)
            if not os.path.isdir(run_path): continue

            for task_name in os.listdir(run_path):
                task_path = os.path.join(run_path, task_name)
                if not os.path.isdir(task_path): continue

                dir_untimed = os.path.join(task_path, "data_from_mpm")
                dir_timed = os.path.join(task_path, "timed_run")
                if not (os.path.exists(dir_untimed) and os.path.exists(dir_timed)): continue

                task_mapped = TASK_MAP.get(task_name, task_name)
                meta_path = os.path.join(dir_untimed, "raw_telemetry_meta.json")
                traj_path = os.path.join(dir_untimed, "raw_telemetry_trajectory.json")

                # Only the active pushing phases count
                phases = get_kinematic_phases(meta_path, traj_path)

                df_timed = pd.read_csv(os.path.join(dir_timed, "raw_telemetry.csv"))
                w_t, _, _ = extract_phase_durations(df_timed, phases)

                model_timed = MODEL_MAP.get('MPM-Timed', 'MPM-Timed')
                comp_records.append({'Scenario': tag, 'Task': task_mapped, 'Run': run_name, 'Model': model_timed,
                                     'Total_Time': w_t})

    # ---------------------------------------------------------
    # Plots and CSVs
    # ---------------------------------------------------------
    if comp_records:
        df_comp = pd.DataFrame(comp_records)
        df_comp['Step_Type'] = df_comp['Task'].apply(lambda x: '2-step' if '2-step' in str(x) else '1-step')

        # Only the models in ACTIVE_MODELS
        df_comp = df_comp[df_comp['Model'].isin(ACTIVE_MODELS)]

        def trigger_plot(df_subset, view_name):
            if df_subset.empty: return

            # Plotted values as CSV
            df_subset[ ['Scenario', 'Task', 'Run', 'Model', 'Total_Time'] ].to_csv(
                os.path.join(dir_comp_csv, f"{view_name}_data.csv"), index=False)

            plot_comparison_strip(df_subset, view_name, dir_comp_pdf, ACTIVE_MODELS)

        # View 1: one plot per input folder and task
        for (scenario, task), group_df in df_comp.groupby(['Scenario', 'Task']):
            clean_task = str(task).replace(" ", "_").replace("(", "").replace(")", "")
            trigger_plot(group_df, f"{scenario}_{clean_task}")

        # View 2: all 1-step and all 2-step pushes pooled
        df_1step = df_comp[df_comp['Step_Type'] == '1-step'].copy()
        df_2step = df_comp[df_comp['Step_Type'] == '2-step'].copy()
        df_1step['Scenario'] = 'All 1-Steps'
        df_2step['Scenario'] = 'All 2-Steps'
        trigger_plot(df_1step, "Global_Binned_1-Steps")
        trigger_plot(df_2step, "Global_Binned_2-Steps")

    print(f"Data generation complete. Output saved to: '{dir_comp}'.")

if __name__ == "__main__":
    main()

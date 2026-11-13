"""
Clearance score: residual volume in the clearance region over the pushes (paper: Fig. 16).

One line per erosion group (w/o erosion: PS, PW; w/ erosion: PSE, PWE): the median over all runs
of that group, with a shaded band from the smallest to the largest run. Symlog y-axis, so both the
large drop at the start and the last few cm³ are readable.

Input:  per run, clearance_scores_recomputed.csv (from scripts/clearance_scores.py),
        given as the CSV path or as the run's experiment folder (the CSV is then found inside:
        run_*/planner_archive/*planning_session/physical_scan_001/clearance_scores_recomputed.csv).
        Tags are <material>/<model>, one per input, same order. Bad or missing inputs are skipped
        with a WARNING.
Output: <out-dir>/clearance_score_envelope.pdf and <out-dir>/clearance_data.csv (every plotted
        point and its source CSV).

Usage:
    python plotting/plot_clearance_score.py --inputs <csv or folder> [...] \\
        --tags <material>/<model> [...] --out-dir <folder>
"""

import argparse
import glob
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

# =========================================================================
# CONFIGURATION
# =========================================================================

# Models: erosion group per model (looked up here, never parsed from names)
MODEL_ORDER = ['PS', 'PW', 'PSE', 'PWE']
MODEL_EROSION = {'PS': 'no', 'PW': 'no', 'PSE': 'yes', 'PWE': 'yes'}

# Materials (order in the printed summary)
MATERIAL_ORDER = ['grapes', 'coffee_beans', 'oats']

# Erosion groups
EROSION_COLORS = {'no': '#0072B2', 'yes': '#009E73'}
EROSION_LABELS = {'no': 'w/o erosion (PS, PW)', 'yes': 'w/ erosion (PSE, PWE)'}

PLOT_RC_PARAMS = {
    'font.family': 'serif',
    'font.size': 14,
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'axes.linewidth': 1.5,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'pdf.fonttype': 42,  # Force TrueType fonts in PDFs (Fixes Type 3 errors)
    'ps.fonttype': 42
}
GEOMETRY = {'figsize': (7, 4), 'dpi': 300}
AXIS_CONFIG = {
    'xlabel': 'Steps',
    'ylabel': 'Residual volume [cm³]',
    'grid_major': {'linestyle': '-', 'alpha': 0.6, 'linewidth': 1.2},
}
LEGEND_CONFIG = {
    'loc': 'upper right',
    'frameon': True,
    'facecolor': 'white',
    'framealpha': 0.9,
    'edgecolor': 'black'
}
DATA_CONFIG = {
    'volume_conversion_factor': 1_000_000,  # m^3 to cm^3
    'target_column': 'actual_volume',
}
ENVELOPE_CONFIG = {
    'linthresh': 1.0,                      # [cm^3] below this the y-axis is linear (so 0 can be shown)
    'yticks': [0, 1, 10, 100, 1000],
    'band_alpha': 0.25,
    'median_linewidth': 4,
}


# =========================================================================
# LOADING (lenient: warn and skip, never crash on bad input)
# =========================================================================

def warn(msg):
    print(f"WARNING: {msg}")


RECOMPUTED_CSV_PATTERN = "run_*/planner_archive/*planning_session/physical_scan_001/clearance_scores_recomputed.csv"


def find_csv(input_path):
    """Experiment folder -> its clearance_scores_recomputed.csv. A .csv path is returned as is.
    Returns None (with a WARNING) if nothing or more than one CSV is found."""
    input_path = os.path.expanduser(input_path)
    if os.path.isfile(input_path):
        return input_path
    if not os.path.isdir(input_path):
        warn(f"Not found, skipping: {input_path}")
        return None
    matches = sorted(glob.glob(os.path.join(input_path, RECOMPUTED_CSV_PATTERN)))
    if len(matches) == 0:
        warn(f"No clearance_scores_recomputed.csv inside {input_path}. Skipping.")
        return None
    if len(matches) > 1:
        warn(f"{len(matches)} clearance_scores_recomputed.csv inside {input_path}, expected 1. Skipping:\n    "
             + "\n    ".join(matches))
        return None
    return matches[0]


def load_table(inputs, tags):
    """Returns a long table: material, model, erosion, step, volume_cm3, source_csv."""
    if len(inputs) != len(tags):
        warn(f"{len(inputs)} inputs but {len(tags)} tags. Using the first "
             f"{min(len(inputs), len(tags))} pairs.")

    rows, seen = [], set()
    for input_path, tag in zip(inputs, tags):
        path = find_csv(input_path)
        if path is None:
            continue
        print(f"  {tag:18s} <- {path}")
        if tag.count('/') != 1:
            warn(f"Tag '{tag}' is not <material>/<model>. Skipping {path}")
            continue
        material, model = tag.split('/')
        if model not in MODEL_EROSION:
            warn(f"Unknown model '{model}' in '{tag}' (known: {MODEL_ORDER}). Skipping {path}")
            continue
        if material not in MATERIAL_ORDER:
            warn(f"Unknown material '{material}' in '{tag}'. Using it anyway.")
        if (material, model) in seen:
            warn(f"Tag '{tag}' given twice. Keeping the first, skipping {path}")
            continue
        try:
            df = pd.read_csv(path)
        except Exception as e:
            warn(f"Could not read {path} ({e}). Skipping.")
            continue
        if DATA_CONFIG['target_column'] not in df.columns:
            warn(f"Column '{DATA_CONFIG['target_column']}' missing in {path}. Skipping.")
            continue

        seen.add((material, model))
        vol = df[DATA_CONFIG['target_column']].to_numpy(dtype=float) * DATA_CONFIG['volume_conversion_factor']
        for step, v in enumerate(vol):
            rows.append({'material': material, 'model': model, 'erosion': MODEL_EROSION[model],
                         'step': step, 'volume_cm3': v, 'source_csv': os.path.abspath(path)})

    if not rows:
        print("ERROR: nothing could be loaded. See warnings above.")
        sys.exit(1)

    table = pd.DataFrame(rows)
    materials, models = order_of('material', table), order_of('model', table)
    missing = [f"{ma}/{mo}" for ma in materials for mo in models if (ma, mo) not in seen]
    if missing:
        warn(f"Not plotted (no data): {missing}")
    return table


def order_of(column, table):
    """Known keys in their configured order, then unknown ones alphabetically."""
    order = {'material': MATERIAL_ORDER, 'model': MODEL_ORDER}[column]
    present = list(table[column].unique())
    return [k for k in order if k in present] + sorted(k for k in present if k not in order)


# =========================================================================
# PLOT
# =========================================================================

def plot_envelope(table, out_dir):
    """One bold median line per erosion group + a band from the smallest to the largest run."""
    fig, ax = plt.subplots(figsize=GEOMETRY['figsize'])

    for erosion in ['no', 'yes']:
        group = table[table['erosion'] == erosion]
        if group.empty:
            continue
        # One column per run (material, model), one row per step.
        runs_per_step = group.pivot_table(index='step', columns=['material', 'model'], values='volume_cm3')
        n_runs = runs_per_step.shape[1]

        ax.fill_between(runs_per_step.index,
                        runs_per_step.min(axis=1), runs_per_step.max(axis=1),
                        color=EROSION_COLORS[erosion], alpha=ENVELOPE_CONFIG['band_alpha'], linewidth=0)
        ax.plot(runs_per_step.index, runs_per_step.median(axis=1),
                color=EROSION_COLORS[erosion], linewidth=ENVELOPE_CONFIG['median_linewidth'],
                label=EROSION_LABELS[erosion])
        print(f"  envelope: {EROSION_LABELS[erosion]} uses {n_runs} runs")

    max_step = int(table['step'].max())
    ax.set_yscale('symlog', linthresh=ENVELOPE_CONFIG['linthresh'])
    ax.set_yticks(ENVELOPE_CONFIG['yticks'])
    ax.set_yticklabels([str(t) for t in ENVELOPE_CONFIG['yticks']])
    ax.set_ylim(0, ENVELOPE_CONFIG['yticks'][-1] * 1.5)
    ax.set_xlim(0, max_step)
    ax.set_xticks(range(0, max_step + 1))
    ax.set_xlabel(AXIS_CONFIG['xlabel'])
    ax.set_ylabel(AXIS_CONFIG['ylabel'])
    ax.grid(True, which='major', zorder=0, **AXIS_CONFIG['grid_major'])
    ax.legend(**LEGEND_CONFIG)

    plt.tight_layout()
    out_path = os.path.join(out_dir, 'clearance_score_envelope.pdf')
    fig.savefig(out_path, bbox_inches='tight', dpi=GEOMETRY['dpi'])
    plt.close(fig)
    print(f"  wrote {out_path}")


# =========================================================================
# MAIN
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="Plot the clearance score (residual volume over the pushes).")
    parser.add_argument('--inputs', nargs='+', required=True,
                        help="clearance_scores_recomputed.csv files (or experiment folders), one per tag.")
    parser.add_argument('--tags', nargs='+', required=True, help="<material>/<model>, one per input, same order.")
    parser.add_argument('--out-dir', required=True, help="Output folder")
    args = parser.parse_args()

    plt.rcParams.update(PLOT_RC_PARAMS)
    os.makedirs(args.out_dir, exist_ok=True)

    table = load_table(args.inputs, args.tags)

    last = table.loc[table.groupby(['material', 'model'])['step'].idxmax()]
    pivot = last.pivot(index='material', columns='model', values='volume_cm3')
    pivot = pivot.reindex(index=order_of('material', table), columns=order_of('model', table))
    print(f"\nLoaded {len(last)} runs. Final residual volume [cm³] at each run's last step:")
    print(pivot.round(3).to_string(), "\n")

    csv_path = os.path.join(args.out_dir, 'clearance_data.csv')
    table.to_csv(csv_path, index=False)
    print(f"  wrote {csv_path}")

    plot_envelope(table, args.out_dir)
    print("Done.")


if __name__ == '__main__':
    main()

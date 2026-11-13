"""
Model fit to the bucket-lift data (paper: Fig. 10).

Bar plot of the IoU between prediction and measurement, for HBM and MPM, in the two bucket-lift
scenarios: settling (pile left after lifting the bucket) and retention (material kept in the bucket).

Input:  one LIFT_SUMMARY_LEADERBOARD.csv (used as is), or one or several LIFT_SUMMARY_LONG_FORMAT.csv
        (e.g. one per material; all runs are pooled and mean/std recomputed over all of them).
Output: <out-dir>/model_fit_plot.pdf and <out-dir>/README.txt (lists the input files).

Usage:
    python plotting/plot_bucketlift_model_fit.py --inputs <csv> [<csv> ...] --out-dir <folder>
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

# Global styling for tight, high-readability plots
plt.rcParams.update({
    'font.size': 13,
    'axes.labelsize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 12,
    'font.weight': 'bold',
    'pdf.fonttype': 42,  # Force TrueType fonts in PDFs (Fixes Type 3 errors)
    'ps.fonttype': 42    # Force TrueType fonts in PostScript
})


def load_summary(csv_paths):
    """One LIFT_SUMMARY_LEADERBOARD.csv -> used as is.
    One or more LIFT_SUMMARY_LONG_FORMAT.csv (one per run, e.g. one file per material)
    -> all runs are pooled and mean/std are recomputed over ALL of them."""
    if len(csv_paths) == 1 and 'LONG_FORMAT' not in os.path.basename(csv_paths[0]):
        return pd.read_csv(csv_paths[0])
    for p in csv_paths:
        if 'LONG_FORMAT' not in os.path.basename(p):
            raise SystemExit(f"When passing several CSVs, all must be LIFT_SUMMARY_LONG_FORMAT.csv. Got: {p}")
    df_long = pd.concat([pd.read_csv(p) for p in csv_paths], ignore_index=True)
    pooled = df_long.groupby('Label')['IoU'].agg(IoU_mean='mean', IoU_std='std', n_runs='count').reset_index()
    print("Pooled over", len(csv_paths), "file(s):")
    print(pooled.to_string(index=False))
    return pooled


def get_metrics(df):
    """Mean and std of the IoU per scenario (settling, retention) and model (HBM, MPM)."""
    mapping_main = {
        'Settling_Real_vs_HBM': ('Settling', 'HBM'),
        'Retention_Real_vs_HBM': ('Retention', 'HBM'),
        'Settling_Real_vs_MPM': ('Settling', 'MPM'),
        'Retention_Real_vs_MPM': ('Retention', 'MPM'),
    }
    results = {
        'Settling': {'HBM': {'mean': 0, 'std': 0}, 'MPM': {'mean': 0, 'std': 0}},
        'Retention': {'HBM': {'mean': 0, 'std': 0}, 'MPM': {'mean': 0, 'std': 0}}
    }
    for label, (cat, model) in mapping_main.items():
        subset = df[df['Label'] == label]
        if not subset.empty:
            results[cat][model]['mean'] = subset['IoU_mean'].values[0]
            results[cat][model]['std'] = subset['IoU_std'].values[0]
    return results


def plot_engine(ax, data, x_label, show_legend=True, y_min=0.7, text_y_offset=0.005):
    """Grouped bars (HBM, MPM) with std whiskers and the mean written inside each bar."""
    scenarios = ['Settling', 'Retention']
    x = np.arange(len(scenarios))
    width = 0.42

    c_hbm = '#4C72B0'
    c_mpm = '#DD8452'

    hbm_means = [data[s]['HBM']['mean'] for s in scenarios]
    hbm_stds = [data[s]['HBM']['std'] for s in scenarios]

    mpm_means = [data[s]['MPM']['mean'] for s in scenarios]
    mpm_stds = [data[s]['MPM']['std'] for s in scenarios]

    # Text placement based on y_min
    text_y_pos = y_min + text_y_offset

    # Bars with standard deviation whiskers
    ax.bar(x - width/2, hbm_means, width, label='HBM', color=c_hbm, zorder=3,
           edgecolor='black', linewidth=0.8, yerr=hbm_stds, capsize=4, error_kw={'elinewidth': 1.5})

    ax.bar(x + width/2, mpm_means, width, label='MPM', color=c_mpm, zorder=3,
           edgecolor='black', linewidth=0.8, yerr=mpm_stds, capsize=4, error_kw={'elinewidth': 1.5})

    for i in range(len(scenarios)):
        # Mean inside the bar (horizontal, bottom-aligned)
        ax.text(x[i] - width/2, text_y_pos, f"{hbm_means[i]:.3f}",
                ha='center', va='bottom', color='white', fontweight='bold', fontsize=11)

        ax.text(x[i] + width/2, text_y_pos, f"{mpm_means[i]:.3f}",
                ha='center', va='bottom', color='white', fontweight='bold', fontsize=11)

    ax.set_xlabel(x_label, labelpad=5)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.set_ylim(y_min, 1.0)
    ax.yaxis.grid(True, linestyle=':', alpha=0.4, zorder=0)

    if show_legend:
        ax.legend(loc='upper left', frameon=True, fontsize=10, ncol=2, handletextpad=0.4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot the HBM and MPM fit to the bucket-lift data (Fig. 10).")
    parser.add_argument('--inputs', type=str, nargs='+', required=True,
                        help="One LIFT_SUMMARY_LEADERBOARD.csv, or one or several LIFT_SUMMARY_LONG_FORMAT.csv (pooled)")
    parser.add_argument('--out-dir', type=str, required=True, help="Output folder")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    data = get_metrics(load_summary(args.inputs))

    fig, ax = plt.subplots(figsize=(7, 2.5))
    plot_engine(ax, data, "Model fit to bucket-lift data")
    ax.set_ylabel('IoU [-]')
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, 'model_fit_plot.pdf'), bbox_inches='tight', pad_inches=0.01)
    plt.close(fig)

    # README.txt in the output folder lists the input files
    readme_path = os.path.join(args.out_dir, 'README.txt')
    with open(readme_path, 'w') as f:
        f.write("Plot Generation Log\n")
        f.write("===================\n")
        for p in args.inputs:
            f.write(f"Input CSV: {p}\n")

    print(f"model_fit_plot.pdf and README.txt written to '{args.out_dir}/'.")

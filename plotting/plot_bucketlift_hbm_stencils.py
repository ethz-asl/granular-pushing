"""
HBM erosion-stencil sweep on the bucket-lift data (paper: Fig. 8).

IoU of the HBM prediction over stencil size, one line per stencil type (LRUD, DIAG, BLOC, DISC).
Left panel: settling (pile left after lifting the bucket), right panel: retention (material kept
in the bucket).

Input:  aggregated_results.csv of the stencil sweep (hbm_erosion_stencil_sweep.py aggregate).
Output: <out-dir>/stencil_sweep_plot.pdf, plus the plotted IoU values as tables
        (iou_mean_settling.txt, iou_mean_retention.txt).

Usage:
    python plotting/plot_bucketlift_hbm_stencils.py --input <aggregated_results.csv> --out-dir <folder>
"""
import pandas as pd
import argparse
import os
import sys
import matplotlib.pyplot as plt

# =========================================================================
# FIGURE PLOTTING CONFIGURATIONS
# =========================================================================
# Global typography
PLOT_RC_PARAMS = {
    'font.size': 8,
    'axes.labelsize': 8,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'font.weight': 'bold',
    'pdf.fonttype': 42,  # Force TrueType fonts in PDFs (Fixes Type 3 errors)
    'ps.fonttype': 42    # Force TrueType fonts in PostScript
}

# Figure size (width, height) in inches
FIGSIZE = (3.5, 1.25)

# Plot limits, ticks, and labels
AXIS_CONFIG = {
    'ymin': 0.75,
    'ymax': 1.0,
    'xmin': 3,
    'xmax': 19,
    'yticks': [0.8, 0.9, 1.0],
    'xticks': [3, 5, 7, 9, 11, 13, 15, 17, 19],
    'ylabel': 'IoU [-]',
    'xlabel': 'Stencil Size [cells]',
    'labelpad': 5,
    'line_width': 1,
    'marker_size': 3
}

# Data styling per stencil type
STENCIL_STYLES = {
    'LRUD': {'color': 'blue', 'marker': 'o'},
    'DIAG': {'color': 'red', 'marker': '^'},
    'BLOC': {'color': 'darkgreen', 'marker': 's'},
    'DISC': {'color': 'darkorange', 'marker': 'D'}
}

# Panel labels (e.g., "(a) Settling")
PANEL_LABEL_PROPS = {
    'fontsize': 8,
    'ha': 'center',
    'va': 'bottom',
    'bbox': dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9, edgecolor='lightgray')
}

# Legend settings
LEGEND_CONFIG = {
    'loc': 'upper center',
    'frameon': True,
    'fontsize': 6,
    'ncol': 2,
    'handletextpad': 0.4,
    'order_idx': [0, 3, 1, 2] # Map LRUD(0), DISC(3), DIAG(1), BLOC(2)
}

# Panel labels per scenario
METADATA = {
    'settling': {'label': '(a) Settling'},
    'retention': {'label': '(b) Retention'},
}

OUTPUT_FILENAME = 'stencil_sweep_plot.pdf'


def parse_stencil(val):
    """
    Parses stencil string (e.g., 'lrud-1', 'bloc-3') to extract Type and Size.
    """
    try:
        parts = val.split('-')
        sType = parts[0].upper()
        num = int(parts[1])

        if sType == 'BLOC':
            size = num
        else:
            size = 2 * num + 1

        return sType, size
    except Exception:
        return None, None


def main():
    # 1. Command line
    parser = argparse.ArgumentParser(description="Plot the HBM erosion-stencil sweep (Fig. 8).")
    parser.add_argument('--input', required=True, help="aggregated_results.csv of the stencil sweep")
    parser.add_argument('--out-dir', required=True, help="Output folder")
    args = parser.parse_args()

    # 2. Validate input
    input_path = args.input
    if not os.path.exists(input_path):
        print(f"Error: File not found at {input_path}")
        sys.exit(1)

    # 3. Output folder
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Reading data from: {input_path}")
    print(f"Output folder: {args.out_dir}")

    # 4. Load data
    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)

    # 5. Stencil type and size from the stencil name
    parsed_data = df['stencil'].apply(parse_stencil)
    df['Type'], df['Size'] = zip(*parsed_data)
    df = df.dropna(subset=['Size'])
    df['Size'] = df['Size'].astype(int)

    target_cols = ['LRUD', 'DIAG', 'BLOC', 'DISC']

    # ---------------------------------------------------------
    # IoU per stencil type and size (also saved as tables)
    # ---------------------------------------------------------
    c_pivot = None
    p_pivot = None

    if 'Settling_Real_vs_HBM_IoU_mean' in df.columns:
        c_pivot = df.pivot(index='Size', columns='Type', values='Settling_Real_vs_HBM_IoU_mean')
        c_pivot = c_pivot[[c for c in target_cols if c in c_pivot.columns]]
        c_pivot.index.name = 'stencil_size'
        path_settling = os.path.join(args.out_dir, 'iou_mean_settling.txt')
        c_pivot.to_csv(path_settling, sep=' ')
        print(f"Generated: {os.path.basename(path_settling)}")

    if 'Retention_Real_vs_HBM_IoU_mean' in df.columns:
        p_pivot = df.pivot(index='Size', columns='Type', values='Retention_Real_vs_HBM_IoU_mean')
        p_pivot = p_pivot[[c for c in target_cols if c in p_pivot.columns]]
        p_pivot.index.name = 'stencil_size'
        path_retention = os.path.join(args.out_dir, 'iou_mean_retention.txt')
        p_pivot.to_csv(path_retention, sep=' ')
        print(f"Generated: {os.path.basename(path_retention)}")

    # ---------------------------------------------------------
    # Figure: settling (left) and retention (right)
    # ---------------------------------------------------------
    if c_pivot is None or p_pivot is None:
        print("Error: the input needs both Settling_... and Retention_... IoU columns.")
        sys.exit(1)

    plt.rcParams.update(PLOT_RC_PARAMS)

    def plot_data(ax, df_settling, df_retention):
        """One line per stencil type: settling solid, retention dashed."""
        if df_settling is not None:
            for col in target_cols:
                if col in df_settling.columns:
                    ax.plot(df_settling.index, df_settling[col], label=f"{col}",
                        linestyle='-', color=STENCIL_STYLES[col]['color'],
                        marker=STENCIL_STYLES[col]['marker'],
                        markersize=AXIS_CONFIG['marker_size'],
                        linewidth=AXIS_CONFIG['line_width'])
        if df_retention is not None:
            for col in target_cols:
                if col in df_retention.columns:
                    ax.plot(df_retention.index, df_retention[col], label=f"{col}",
                        linestyle='--', color=STENCIL_STYLES[col]['color'],
                        marker=STENCIL_STYLES[col]['marker'],
                        markersize=AXIS_CONFIG['marker_size'],
                        linewidth=AXIS_CONFIG['line_width'])

        ax.set_ylim(bottom=AXIS_CONFIG['ymin'], top=AXIS_CONFIG['ymax'])
        ax.set_xlim(left=AXIS_CONFIG['xmin'], right=AXIS_CONFIG['xmax'])
        ax.grid(True, which='major', linestyle='-', alpha=0.5)

    fig_lr, axes_lr = plt.subplots(1, 2, figsize=FIGSIZE, sharey=True)

    plot_data(axes_lr[0], c_pivot, None)
    plot_data(axes_lr[1], None, p_pivot)

    for ax in axes_lr:
        ax.set_yticks(AXIS_CONFIG['yticks'])
        ax.set_xticks(AXIS_CONFIG['xticks'])
        ax.set_xticklabels([str(x) for x in AXIS_CONFIG['xticks']])
        ax.set_xlabel(AXIS_CONFIG['xlabel'], labelpad=AXIS_CONFIG['labelpad'])

    # Left-most axis gets the shared Y-label
    axes_lr[0].set_ylabel(AXIS_CONFIG['ylabel'])

    # Panel labels
    axes_lr[0].text(0.5, 0.08, METADATA['settling']['label'], transform=axes_lr[0].transAxes,
                    fontsize=PANEL_LABEL_PROPS['fontsize'], ha=PANEL_LABEL_PROPS['ha'], va=PANEL_LABEL_PROPS['va'], bbox=PANEL_LABEL_PROPS['bbox'])
    axes_lr[1].text(0.5, 0.08, METADATA['retention']['label'], transform=axes_lr[1].transAxes,
                    fontsize=PANEL_LABEL_PROPS['fontsize'], ha=PANEL_LABEL_PROPS['ha'], va=PANEL_LABEL_PROPS['va'], bbox=PANEL_LABEL_PROPS['bbox'])

    # Legend in the first panel, reordered so the rows read LRUD, DIAG / DISC, BLOC
    h_lr, l_lr = axes_lr[0].get_legend_handles_labels()
    if len(h_lr) >= 4:
        idx = LEGEND_CONFIG['order_idx']
        h_lr = [h_lr[i] for i in idx]
        l_lr = [l_lr[i] for i in idx]

    axes_lr[0].legend(h_lr, l_lr, loc=LEGEND_CONFIG['loc'], frameon=LEGEND_CONFIG['frameon'],
                      fontsize=LEGEND_CONFIG['fontsize'], ncol=LEGEND_CONFIG['ncol'], handletextpad=LEGEND_CONFIG['handletextpad'])

    fig_lr.subplots_adjust(wspace=0.08)
    fig_lr.savefig(os.path.join(args.out_dir, OUTPUT_FILENAME), bbox_inches='tight', format='pdf')
    plt.close(fig_lr)

    print(f"Generated: {OUTPUT_FILENAME}")


if __name__ == "__main__":
    main()

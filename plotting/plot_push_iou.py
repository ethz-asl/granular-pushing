"""
Push prediction fidelity (paper: Fig. 13).

Grouped bar plot of the IoU between predicted and measured height map, per model, for 1-step
predictions (one push) and 2-step predictions (two pushes chained).

Input:  one or several push-evaluation folders (each with SUMMARY_LONG_FORMAT.csv from model_eval.py --push).
Output: <out-dir>/agg2_<tag>_steps_bar.pdf (one per input folder) and
        <out-dir>/agg3_all_combined_steps_bar.pdf (all inputs together).

Usage:
    python plotting/plot_push_iou.py --inputs <folder> [<folder> ...] --tags <tag> [<tag> ...] \
        --models PS PW PSE PWE MPM --out-dir <folder>
"""
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# =========================================================================
# FIGURE PLOTTING CONFIGURATIONS
# =========================================================================

PLOT_RC_PARAMS = {
    'font.size': 20,
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'axes.titlesize': 22,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 17,
    'axes.linewidth': 1.2,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'pdf.fonttype': 42,  # Force TrueType fonts in PDFs (Fixes Type 3 errors)
    'ps.fonttype': 42    # Force TrueType fonts in PostScript
}

GEOMETRY = {
    'figsize': (7, 5),
    'bar_width': 0.13,
    'first_gap': 0.2,  # Tiny gap multiplier (20% of a bar's width) between first and the rest
}

AXIS_CONFIG = {
    'ymin': 0.0,
    'ymax': 1.0,
    'ylabel': 'IoU [-]',
    'title_pad': 15,
    'grid': {
        'axis': 'y',
        'linestyle': '--',
        'alpha': 0.5,
        'linewidth': 1.0
    }
}

MODEL_CONFIG = {
    'mapping': {'NOPUSH': 'NONE'},
    'order': ['NONE', 'PS', 'PW', 'PSE', 'PWE', 'MPM'],
    'colors': {
        'NONE': '#7F8C8D',    # grey
        'PS':   '#0072B2',    # blue
        'PW':   '#E69F00',    # amber
        'PSE':  '#009E73',    # green
        'PWE':  '#D55E00',    # vermillion
        'MPM':  '#CC79A7',    # purple
        }
}

BAR_STYLES = {
    'edgecolor': 'black',
    'linewidth': 0.8,
    'alpha': 0.9,
    'capsize': 4,
    'error_kw': {'elinewidth': 1.5, 'markeredgewidth': 1.5}
}

ANNOTATION_PROPS = {
    'y_offset': 0.01,
    'color': 'white',
    'fontsize': 16,
    'fontweight': 'bold',
    'rotation': 'vertical',
    'zorder': 10,
    'format_str': '{:.3f}'
}

LEGEND_CONFIG = {
    'loc': 'upper right',
    'frameon': True,
    'fontsize': 14,
    'ncol': 3,
    'columnspacing': 0.8,
    'handletextpad': 0.3,
    'order_idx': [0, 3, 1, 4, 2, 5]
}

OUTPUT_CONFIG = {
    'base_dir': None,
    'bar_subdir': None,
    'dpi': 300,
    'bbox_inches': 'tight',
    'grouped_prefix': 'agg2_{}_steps',
    'combined_prefix': 'agg3_all_combined_steps'
}

def parse_args():
    parser = argparse.ArgumentParser(description='Aggregate and plot experiment results.')
    parser.add_argument('--inputs', type=str, nargs='+', required=True, 
                        help='Push-evaluation folders (each with SUMMARY_LONG_FORMAT.csv).')
    parser.add_argument('--tags', type=str, nargs='+', 
                        help='Optional label per input folder (default: folder name).')

    parser.add_argument('--out-dir', type=str, required=True, 
                        help='Output folder.')

    # What to plot (default: everything)
    parser.add_argument('--models', type=str, nargs='+', default=MODEL_CONFIG['order'],
                        choices=MODEL_CONFIG['order'],
                        help='Models to plot (drawn in the fixed order NONE PS PW PSE PWE MPM). '
                             'E.g. --models PS PW PSE PWE MPM to drop NONE.')
    parser.add_argument('--steps', type=str, nargs='+', default=['1-step', '2-step'],
                        choices=['1-step', '2-step'],
                        help='Which step groups to plot on the x-axis.')
    
    return parser.parse_args()

def load_and_preprocess_data(experiments, tags, models):
    # if human readable tags are provided, use them; otherwise, default to folder names
    if tags and len(tags) != len(experiments):
        raise ValueError("Number of tags must match the number of experiments.")
    if not tags:
        tags = [Path(exp).name for exp in experiments]

    # crawl through each provided experiment folder, read the SUMMARY_LONG_FORMAT.csv, and aggregate into a single DataFrame
    all_data = []
    for exp_path, tag in zip(experiments, tags):
        csv_path = Path(exp_path) / 'SUMMARY_LONG_FORMAT.csv'
        if not csv_path.exists():
            print(f"Warning: {csv_path} not found. Skipping.")
            continue
            
        # Read the CSV, add a 'Tag' column for labeling, and map model names to more readable formats
        df = pd.read_csv(csv_path)
        df['Tag'] = tag
        df['Model'] = df['Model'].map(lambda x: MODEL_CONFIG['mapping'].get(x, x)) # Map to readable names, defaulting to original if not found
        
        def get_step(task):
            if pd.isna(task): return 'Other'
            if 'TaskA' in task or 'TaskB' in task: return '1-step'
            elif 'TaskC' in task: return '2-step'
            return 'Other'
            
        df['Step'] = df['Task'].apply(get_step)
        all_data.append(df)
        
    if not all_data:
        raise ValueError("No valid data found.")
        
    combined_df = pd.concat(all_data, ignore_index=True)
    # Keep only the requested models, in the fixed order from MODEL_CONFIG (not the CLI order).
    model_order = [m for m in MODEL_CONFIG['order'] if m in models]
    combined_df = combined_df[combined_df['Model'].isin(model_order)]
    combined_df = combined_df[combined_df['Step'] != 'Other']
    return combined_df, model_order

def annotate_rects_consistent(ax, rects, y_min):
    """
    Vertical white text at the base of the bars.
    """
    text_y_pos = y_min + 0.01 
    for rect in rects:
        height = rect.get_height()
        if np.isnan(height) or height == 0: continue

        ax.text(rect.get_x() + rect.get_width() / 2., text_y_pos, 
            ANNOTATION_PROPS['format_str'].format(height),
            ha='center', va='bottom', 
            color=ANNOTATION_PROPS['color'], 
            fontsize=ANNOTATION_PROPS['fontsize'], 
            fontweight=ANNOTATION_PROPS['fontweight'], 
            rotation=ANNOTATION_PROPS['rotation'], 
            zorder=ANNOTATION_PROPS['zorder'])
        
def main():
    # Apply global styling
    plt.rcParams.update(PLOT_RC_PARAMS)

    args = parse_args()

    OUTPUT_CONFIG['base_dir'] = args.out_dir

    print(OUTPUT_CONFIG['base_dir'])
    print(args.inputs)

    # Setup directories
    bar_dir = Path(OUTPUT_CONFIG['base_dir'])
    bar_dir.mkdir(parents=True, exist_ok=True)
    
    df, model_order = load_and_preprocess_data(args.inputs, args.tags, args.models)
    print(f"Models: {model_order} | Steps: {args.steps}")
    
    # Extract explicit colors mapped to the model order
    colors = [MODEL_CONFIG['colors'][m] for m in model_order]

    def create_grouped_plots(plot_df, title, filename):
        steps = args.steps
        x = np.arange(len(steps))
        width = GEOMETRY['bar_width']
        total_width = width * len(model_order) 
        
        fig, ax = plt.subplots(figsize=GEOMETRY['figsize'])
        ax.grid(True, **AXIS_CONFIG['grid'])

        for i, (model_name, color) in enumerate(zip(model_order, colors)):
            means = [plot_df[(plot_df['Step'] == s) & (plot_df['Model'] == model_name)]['IoU'].mean() for s in steps]
            stds = [plot_df[(plot_df['Step'] == s) & (plot_df['Model'] == model_name)]['IoU'].std() for s in steps]
            means = [m if not pd.isna(m) else 0 for m in means]
            stds = [s if not pd.isna(s) else 0 for s in stds]
            offset = x - (total_width / 2) + (i * width) + (width / 2)
            
            # Add a tiny gap between NONE (the baseline) and the rest, only if NONE is plotted
            tiny_gap = GEOMETRY['first_gap'] * width 
            if model_order[0] == 'NONE' and i > 0:
                offset += tiny_gap

            rects = ax.bar(offset, means, width, label=model_name, yerr=stds, color=color, **BAR_STYLES)
            annotate_rects_consistent(ax, rects, AXIS_CONFIG['ymin'])
            
        ax.set_xticks(x)
        ax.set_xticklabels(steps)
        ax.set_ylim(AXIS_CONFIG['ymin'], AXIS_CONFIG['ymax'])
        
        ax.set_ylabel(AXIS_CONFIG['ylabel'])
        ax.set_title(title, pad=AXIS_CONFIG['title_pad'])
        
        handles, labels = ax.get_legend_handles_labels()
        
        # Pull ordering from config. order_idx is written for all 6 models;
        # with a subset of models, keep the plotting order instead.
        if len(handles) == len(LEGEND_CONFIG['order_idx']):
            ordered_handles = [handles[idx] for idx in LEGEND_CONFIG['order_idx']]
            ordered_labels = [labels[idx] for idx in LEGEND_CONFIG['order_idx']]
        else:
            ordered_handles, ordered_labels = handles, labels
        
        # Isolate matplotlib-compliant kwargs from our config dict
        legend_kwargs = {k: v for k, v in LEGEND_CONFIG.items() if k != 'order_idx'}
        legend_kwargs['ncol'] = min(LEGEND_CONFIG['ncol'], len(handles))
        ax.legend(ordered_handles, ordered_labels, **legend_kwargs)

        fig.tight_layout()
        fig.savefig(bar_dir / f'{filename}_bar.pdf', dpi=OUTPUT_CONFIG['dpi'], bbox_inches=OUTPUT_CONFIG['bbox_inches'])
        plt.close(fig)

    print("Generating Grouped Aggregations...")
    for tag in df['Tag'].unique(): 
        # Format the grouped filename dynamically
        file_prefix = OUTPUT_CONFIG['grouped_prefix'].format(tag.replace("/", "_"))
        create_grouped_plots(df[df['Tag'] == tag], f'[{tag}] IoU by Step', file_prefix)
    
    create_grouped_plots(df, 'Combined Model Performance', OUTPUT_CONFIG['combined_prefix'])
        
if __name__ == '__main__':
    main()
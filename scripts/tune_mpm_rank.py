"""
Rank the evaluations of an MPM tuning run by score (step 2 of the MPM calibration).

Input:  optimization_history.csv, written by tune_mpm.py (one row per evaluated parameter set).
Output: <log name>_analysis/ranked_results.csv next to the input, best first. The top row holds
        the calibrated parameters (ground friction, friction coefficient, critical fraction, voxel size).

Usage:
    python scripts/tune_mpm_rank.py --csv_log <tuning_folder>/optimization_history.csv
"""
import argparse
import os

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description='Rank the evaluations of an MPM tuning run by score.')
    parser.add_argument('--csv_log', type=str, required=True, help='Path to optimization_history.csv')
    args = parser.parse_args()

    # Output folder next to the log: <log name>_analysis/
    csv_path = os.path.abspath(args.csv_log)
    root_dir = os.path.dirname(csv_path)
    csv_filename = os.path.basename(csv_path).replace('.csv', '')
    output_folder = os.path.join(root_dir, f"{csv_filename}_analysis")
    os.makedirs(output_folder, exist_ok=True)

    df = pd.read_csv(csv_path, sep=None, engine='python')
    df.columns = df.columns.str.strip()

    ranked = df.sort_values(by='Score', ascending=False)
    out_path = os.path.join(output_folder, 'ranked_results.csv')
    ranked.to_csv(out_path, index=False)

    print(f"Ranked results saved to: {out_path}")
    print("Best evaluation:")
    print(ranked.head(1).to_string(index=False))


if __name__ == "__main__":
    main()

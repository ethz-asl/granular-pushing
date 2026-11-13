"""
Stencil sweep: run model_eval.py (--lift) once per HBM erosion stencil and collect the results.

  run        one model_eval.py run per stencil of the grid below (lrud, diag, disc: 1-13; bloc: 3-27),
             reusing the MPM data of --MPM-DUMMY-RUN (only HBM is compared)
  aggregate  collect the HBM IoUs (Settling and Retention) and runtimes of all runs into
             aggregated_results.csv

Plot with plotting/plot_bucketlift_hbm_stencils.py. See docs/REPRODUCE.md.

Usage:
    python scripts/hbm_erosion_stencil_sweep.py run --EXP-DATA-DIR <bucketlift_folder> \
        --MPM-DUMMY-RUN <model_eval output with MPM data> --sweep-name <name> --aor <deg> --output-dir <folder>
    python scripts/hbm_erosion_stencil_sweep.py aggregate --EXP-DATA-DIR <bucketlift_folder> \
        --MPM-DUMMY-RUN <same> --sweep-name <name> --aor <deg> --path <output folder of run>
"""
import os
import argparse
import subprocess
import itertools
import pandas as pd
import json
import re
from datetime import datetime

# ==========================================
# 1. CONFIGURATION
# ==========================================

# [EVAL SCRIPT TO RUN]
SCRIPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_eval.py")

# [INPUT / OUTPUT] Set from the command line in __main__
EXP_DATA_DIR = ""
MPM_DUMMY_RUN = ""
MPM_DIR = ""
SWEEP_NAME = ""
timestamp = pd.Timestamp.now().strftime("%Y-%m-%d_%H%M")
SWEEP_FOLDER = ""
OUTPUT_BASE_DIR = ""   # the runs are written into OUTPUT_BASE_DIR/<run folders>

# [STENCILS] lrud = cross, diag = diagonals, bloc = full square, disc = disk (see parse_erosion_stencil)
erosion_stencils_lrud = ['lrud-1', 'lrud-2', 'lrud-3', 'lrud-4', 'lrud-5', 'lrud-6', 'lrud-7', 'lrud-8', 'lrud-9', 'lrud-10', 'lrud-11', 'lrud-12', 'lrud-13']
# from diag-1 to diag-13
erosion_stencils_diag = ['diag-1', 'diag-2', 'diag-3', 'diag-4', 'diag-5', 'diag-6', 'diag-7', 'diag-8', 'diag-9', 'diag-10', 'diag-11', 'diag-12', 'diag-13']
# from bloc-3 to bloc-27
erosion_stencils_bloc = ['bloc-3', 'bloc-5', 'bloc-7', 'bloc-9', 'bloc-11', 'bloc-13', 'bloc-15', 'bloc-17', 'bloc-19', 'bloc-21', 'bloc-23', 'bloc-25', 'bloc-27']
# from disc-1 to disc-13
erosion_stencils_disc = ['disc-1', 'disc-2', 'disc-3', 'disc-4', 'disc-5', 'disc-6', 'disc-7', 'disc-8', 'disc-9', 'disc-10', 'disc-11', 'disc-12', 'disc-13']
# Combine all stencil types
erosion_stencils = erosion_stencils_lrud + erosion_stencils_diag + erosion_stencils_bloc + erosion_stencils_disc

# [GRID] parameter -> (values, short name used in the folder names)
SWEEP_CONFIG = {
    "erosion-stencil":      (erosion_stencils,       "stencil"),
}

# [FIXED ARGS] Same for every run; set in __main__
FIXED_ARGS = {}

# [AGGREGATION] HBM IoU for Settling and Retention, plus runtimes
TARGET_CSV = "LIFT_SUMMARY_LEADERBOARD.csv"
METRICS_TO_EXTRACT = ["IoU_mean", "IoU_std"]
TARGET_ROW_ID = [
    "Settling_Real_vs_HBM",
    "Retention_Real_vs_HBM"
]
TIMINGS_CSV = "TIMINGS_LIFT.csv"
TIMINGS_ROW_ID = "Mean"
TIMINGS_METRICS = [
    "Settling_Time_sec", 
    "Retention_Time_sec", 
    "Warmup_sec", 
    "Warmup_Verify_sec",
    "Settling_Erosion_Steps", 
    "Retention_Erosion_Steps"
]

BATCH_NAME = ""

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def resolve_sweep_path():
    """Folder of this sweep (OUTPUT_BASE_DIR, as BATCH_NAME is empty)."""
    """Determines the final path where the batch config and runs will reside."""
    if BATCH_NAME is None:
        folder_name = f"sweep_{get_timestamp()}"
        return os.path.join(OUTPUT_BASE_DIR, folder_name)
    elif BATCH_NAME == "":
        return OUTPUT_BASE_DIR
    else:
        return os.path.join(OUTPUT_BASE_DIR, BATCH_NAME)

def parse_sweep_config(config_dict):
    keys = []
    value_lists = []
    aliases = {}
    
    for key, (vals, alias) in config_dict.items():
        keys.append(key)
        value_lists.append(vals)
        aliases[key] = alias
        
    return keys, value_lists, aliases

def generate_jobs(config_dict, sweep_name=None):
    """One job per combination of the grid values, each with a unique analysis name."""
    keys, value_lists, aliases = parse_sweep_config(config_dict)
    jobs = []
    
    for combination in itertools.product(*value_lists):
        job_params = dict(zip(keys, combination))
        
        # Create Unique Analysis Name
        name_parts = []
        if sweep_name is not None:
            name_parts.append(sweep_name)
        for k, v in job_params.items():
            alias = aliases.get(k, k) 
            name_parts.append(f"{alias}{v}")
        
        run_name = "_".join(name_parts)
        
        job = job_params.copy()
        job['analysis-name'] = run_name
        jobs.append(job)
        
    return jobs

def save_configs(sweep_path, jobs_count):
    """Save the sweep settings (experiment_state.json, sweep_config.yaml) into the sweep folder."""
    # 1. Machine Readable
    machine_state = {
        "timestamp": get_timestamp(),
        "sweep_config": SWEEP_CONFIG,
        "fixed_args": FIXED_ARGS,
        "metrics": METRICS_TO_EXTRACT,
        "target_csv": TARGET_CSV,
        "target_row": TARGET_ROW_ID
    }
    
    json_path = os.path.join(sweep_path, "experiment_state.json")
    with open(json_path, 'w') as f:
        json.dump(machine_state, f, indent=4)

    # 2. Human Readable
    yaml_path = os.path.join(sweep_path, "sweep_config.yaml")
    with open(yaml_path, 'w') as f:
        f.write(f"# Sweep Configuration\n")
        f.write(f"# Generated: {get_timestamp()}\n")
        f.write(f"Total_Jobs: {jobs_count}\n\n")

        f.write("Paths:\n")
        f.write(f"  Script:      {SCRIPT_PATH}\n")
        f.write(f"  Input_Root:  {EXP_DATA_DIR}\n")
        f.write(f"  Output_Dir:  {sweep_path}\n\n")

        f.write("Fixed_Arguments:\n")
        for k, v in FIXED_ARGS.items():
            f.write(f"  {k}: {v}\n")
        f.write("\n")

        f.write("Sweep_Configuration:\n")
        for key, (vals, alias) in SWEEP_CONFIG.items():
            f.write(f"  {key}:\n")
            f.write(f"    alias: {alias}\n")
            f.write(f"    values: {vals}\n")

    print(f"--- Configs saved to: {sweep_path} ---")

def build_cmd(job_params, sweep_path):
    """model_eval.py command line for one job."""
    # cmd = ["uv", "run", SCRIPT_PATH]
    cmd = ["python", SCRIPT_PATH]
    
    # Paths
    cmd.extend(["--root", EXP_DATA_DIR])
    cmd.extend(["--output-dir", sweep_path]) 
    
    # Fixed Args
    for k, v in FIXED_ARGS.items():
        if v is True: cmd.append(f"--{k}")
        elif v is False: continue
        else: cmd.extend([f"--{k}", str(v)])
            
    # Sweep Args (includes 'analysis-name')
    for k, v in job_params.items():
        cmd.extend([f"--{k}", str(v)])
             
    return cmd

# ==========================================
# 3. CORE LOGIC
# ==========================================

def run_experiments():
    """Run all jobs one after the other; a failed run is reported and skipped."""
    if not os.path.exists(SCRIPT_PATH):
        print(f"Error: Script not found at {SCRIPT_PATH}")
        return

    sweep_path = resolve_sweep_path()
    os.makedirs(sweep_path, exist_ok=True)
    print(f"--- Using Sweep Directory: {sweep_path} ---")

    jobs = generate_jobs(SWEEP_CONFIG, sweep_name=SWEEP_NAME)
    
    # Every run needs its own folder
    unique_names = set(j['analysis-name'] for j in jobs)
    if len(unique_names) < len(jobs):
        print("CRITICAL ERROR: Duplicate analysis names generated.")
        return

    save_configs(sweep_path, len(jobs))

    print(f"--- Starting Grid Search: {len(jobs)} runs ---")
    print(f"[PREVIEW] First run name: {jobs[0]['analysis-name']}")

    for i, job in enumerate(jobs):
        cmd = build_cmd(job, sweep_path)
        print(f"\n[{i+1}/{len(jobs)}] Running: {job['analysis-name']}")
        
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"   FAILED with exit code {e.returncode}")
        except KeyboardInterrupt:
            print("\n   Interrupted by user.")
            exit(1)
            
    print(f"\n--- Batch Complete. Results in {sweep_path} ---")
    print(f"To aggregate, run:\n   python scripts/hbm_erosion_stencil_sweep.py aggregate --path {sweep_path} (plus the same arguments as for 'run')")


def aggregate_results(target_path, output_csv_name):
    """Collect IoU and runtimes of every run in `target_path` into one CSV (one row per stencil)."""
    if target_path is None:
        target_path = resolve_sweep_path()
        
    if not os.path.exists(target_path):
        print(f"Error: Path does not exist: {target_path}")
        return

    print(f"--- Aggregating using CURRENT SCRIPT settings (Wide Format) ---")
    
    # 1. Targets as a list
    targets = TARGET_ROW_ID if isinstance(TARGET_ROW_ID, list) else [TARGET_ROW_ID]
    
    print(f"SWEEP_CONFIG: {SWEEP_CONFIG}")

    jobs = generate_jobs(SWEEP_CONFIG)
    keys, _, aliases = parse_sweep_config(SWEEP_CONFIG)
    results = []

    for job in jobs:
        base_name = job['analysis-name']
        
        # --- Find the run folder by its parameters, e.g. 'stencillrud-1' ---
        stencil_id_parts = []
        for k, v in job.items():
            if k != 'analysis-name':
                header_name = aliases.get(k, k)
                stencil_id_parts.append(f"{header_name}{v}")
        
        core_id = "_".join(stencil_id_parts) 
        escaped_id = re.escape(core_id)

        # (?![0-9]) so that 'lrud-1' does not match 'lrud-11'
        pattern = re.compile(rf"{escaped_id}(?![0-9])")

        all_dirs = [d for d in os.listdir(target_path) if os.path.isdir(os.path.join(target_path, d))]

        matches = [d for d in all_dirs if pattern.search(d)]

        if len(matches) == 0:
            print(f"Warning: No matching folder found containing '{core_id}'. Skipping.")
            folder_path = None
            folder_name = base_name 
        elif len(matches) > 1:
            print(f"Ambiguity! Multiple folders match '{core_id}': {matches}")
            print(f"Picking the most recent one (last in list)...")
            matches.sort() # with timestamped names, the newest is last
            folder_name = matches[-1]
            folder_path = os.path.join(target_path, folder_name)
        else:
            folder_path = os.path.join(target_path, matches[0])
            folder_name = matches[0]

        # --- One row: run name, parameters, metrics, runtimes ---
        entry = {"RUN": folder_name}
        for full_param, value in job.items():
            if full_param == 'analysis-name': continue
            header_name = aliases.get(full_param, full_param)
            entry[header_name] = value

        file_path = os.path.join(folder_path, TARGET_CSV) if folder_path else None
        df = None
        if file_path and os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
            except Exception:
                df = None

        # One column per target and metric, e.g. Settling_Real_vs_HBM_IoU_mean
        for target_id in targets:
            if df is not None:
                row_match = df[df.iloc[:, 0] == target_id]
            else:
                row_match = pd.DataFrame()

            for metric in METRICS_TO_EXTRACT:
                col_name = f"{target_id}_{metric}"
                
                if not row_match.empty and metric in df.columns:
                    entry[col_name] = row_match.iloc[0][metric]
                else:
                    entry[col_name] = None

            # Runtimes: the 'Mean' row of TIMINGS_LIFT.csv
            timings_path = os.path.join(folder_path, TIMINGS_CSV) if folder_path else None
            df_timings = None
            if timings_path and os.path.exists(timings_path):
                try:
                    df_timings = pd.read_csv(timings_path)
                except Exception:
                    pass
            
            for metric in TIMINGS_METRICS:
                col_name = f"Timing_{metric}"
                entry[col_name] = None
                if df_timings is not None and not df_timings.empty:
                    row_match = df_timings[df_timings.iloc[:, 0] == TIMINGS_ROW_ID]
                    if not row_match.empty and metric in df_timings.columns:
                        entry[col_name] = row_match.iloc[0][metric]

        results.append(entry)

    final_df = pd.DataFrame(results)
    
    # Column order: RUN, metrics, runtimes, parameters
    param_headers = [aliases.get(k, k) for k in keys]
    
    metric_headers = []
    for t in targets:
        for m in METRICS_TO_EXTRACT:
            metric_headers.append(f"{t}_{m}")

    for m in TIMINGS_METRICS:
        metric_headers.append(f"Timing_{m}")

    cols = ["RUN"] + metric_headers + param_headers
    final_df = final_df[cols]

    save_path = os.path.join(target_path, output_csv_name)
    final_df.to_csv(save_path, index=False)
    
    print(f"\nAggregation Complete. Saved to: {save_path}")
    print(final_df.to_markdown(index=False))

# ==========================================
# 4. MAIN ENTRY POINT
# ==========================================

if __name__ == "__main__":
    # Arguments shared by 'run' and 'aggregate'
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("--EXP-DATA-DIR", required=True, help="The complete path to the experiment data.")
    parent_parser.add_argument("--MPM-DUMMY-RUN", required=True, default=None, help="The name of the folder with the MPM dummy data.")
    parent_parser.add_argument("--sweep-name", required=True, help="Name of the sweep (used in the run folder names).")
    parent_parser.add_argument("--aor", default=25, required=True, help="Angle of Repose for HBM.")
    parent_parser.add_argument("--output-dir", default=None, help="Output folder of the sweep (default: <EXP-DATA-DIR>/<sweep-name>_<timestamp>).")


    parser = argparse.ArgumentParser(description="Experiment Manager")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", parents=[parent_parser], help="Execute the grid search")

    agg_parser = subparsers.add_parser("aggregate", parents=[parent_parser], help="Aggregate results")
    agg_parser.add_argument("--path", required=False, default=None, help="The sweep folder.")
    agg_parser.add_argument("--out", default="aggregated_results.csv", help="Output filename")

    args = parser.parse_args()

    # Module-level settings from the command line
    EXP_DATA_DIR = args.EXP_DATA_DIR
    MPM_DUMMY_RUN = args.MPM_DUMMY_RUN
    MPM_DIR = os.path.join(
        EXP_DATA_DIR, 
        MPM_DUMMY_RUN,
        "mpm_lift_staging")

    SWEEP_NAME = args.sweep_name
    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d_%H%M")
    SWEEP_FOLDER = f"{SWEEP_NAME}_{timestamp}"
    # OUTPUT_BASE_DIR = os.path.join(EXP_DATA_DIR, SWEEP_FOLDER)
    OUTPUT_BASE_DIR = args.output_dir if args.output_dir else os.path.join(EXP_DATA_DIR, SWEEP_FOLDER)

    # [FIXED ARGS]
    FIXED_ARGS = {
        "headless": True,
        "lift": True,
        "aor": args.aor,
    }

    if MPM_DIR is not None:
        FIXED_ARGS["mpm-root"] = MPM_DIR


    if args.command == "run":
        run_experiments()
    elif args.command == "aggregate":
        aggregate_results(target_path=args.path, output_csv_name=args.out)
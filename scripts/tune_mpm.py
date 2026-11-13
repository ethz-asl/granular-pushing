"""
MPM calibration: tune four MPM parameters so that MPM matches the bucket-lift data.

Parameters (PARAM_CONFIG below, with start values and bounds): material friction, critical
(volume) fraction, ground friction and voxel size. Optimizer: Nelder-Mead (scipy fmin).
Each evaluation runs model_eval.py --lift --tuning-mpm and reads the Settling IoU of MPM
(row Settling_Real_vs_MPM, column IoU_mean of LIFT_SUMMARY_LEADERBOARD.csv).

Log: <output-dir>/optimization_history.csv, one row per evaluation. Rank it with
tune_mpm_rank.py. See docs/REPRODUCE.md.

Usage:
    python scripts/tune_mpm.py --DATA-ROOT <bucketlift_folder> --AOR <deg> --EROSION-STENCIL <stencil> \
        --DENSITY <kg/m3> --TUNING-NAME <name> --output-dir <folder>
"""
import subprocess
import pandas as pd
import numpy as np
import os
import datetime
import argparse
import sys
import glob
from scipy.optimize import fmin

# ==========================================
# 1. CONFIGURATION
# ==========================================

# Set from the command line in __main__
DATA_ROOT = None
AOR = None
EROSION_STENCIL = None
TIMESTAMP = None
TUNING_NAME = None
OPTIMIZATION_DIR = None
LOG_FILE = None
DENSITY = None

# --- PATHS ---
SCRIPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_eval.py")

# --- TARGET: maximize the Settling IoU of MPM ---
TARGET_ROW = "Settling_Real_vs_MPM"  
TARGET_COL = "IoU_mean"
MAXIMIZE = True 

# --- OPTIMIZATION PARAMETERS ---
# active: tuned, or kept fixed at 'initial' (it is passed to model_eval.py either way)
PARAM_CONFIG = {
    "friction": {
        "active": True, "arg": "--mpm-friction-coeff", "initial": 0.5, "bounds": (0.01, 1.0)
    },
    "critical_fraction": {
        "active": True, "arg": "--mpm-critical-fraction", "initial": 0.5, "bounds": (0.01, 1.0)
    },
    "ground_friction": {
        "active": True, "arg": "--mpm-ground-friction", "initial": 0.5, "bounds": (0.01, 1.0)
    },
    "voxel_size": {
        "active": True, "arg": "--mpm-voxel-size", "initial": 0.01, "bounds": (0.005, 0.05)
    },
}

# --- GLOBAL STATE ---
ITERATION_COUNT = 0

# ==========================================
# 2. HELPER: WIDE SIMPLEX GENERATION
# ==========================================

def generate_bounded_simplex(x0, bounds, step_ratio=0.2):
    """
    Creates an initial simplex (N+1 points) that spans 'step_ratio' (e.g. 20%)
    of the allowed range for each parameter. This forces the optimizer to 
    explore broadly rather than getting stuck in local noise.
    """
    n = len(x0)
    # Initialize simplex array: (N+1) rows, N columns
    simplex = np.zeros((n + 1, n))
    
    # Point 0 is always the user-defined initial guess
    simplex[0] = x0
    
    print(f"\n>>> GENERATING INITIAL SIMPLEX (Spread: {step_ratio*100:.1f}%)")
    print(f"    Center Point: {x0}")
    
    for i in range(n):
        # Start with a copy of the initial guess
        point = np.array(x0, copy=True)
        
        # Get bounds for this parameter
        low, high = bounds[i]
        param_range = high - low
        
        # Calculate step size based on the total range available
        step_size = param_range * step_ratio
        
        # Directional Logic:
        # If adding the step keeps us in bounds, go UP.
        # If adding goes out of bounds, subtract (go DOWN).
        if point[i] + step_size <= high:
            point[i] += step_size
        elif point[i] - step_size >= low:
            point[i] -= step_size
        else:
            # Fallback: If the point is stuck near a boundary and the step 
            # is huge, just move to the center of the range.
            point[i] = (low + high) / 2.0
            
        simplex[i + 1] = point
        print(f"    Simplex Point {i+1}: {point}")
        
    return simplex

# ==========================================
# 3. SHARED READING LOGIC
# ==========================================

def read_score_from_folder(folder_path):
    """
    Tries to read the target metric from the CSV in the given folder.
    Returns: (score, cost, success_bool)
    """
    csv_path = os.path.join(folder_path, "LIFT_SUMMARY_LEADERBOARD.csv")
    
    print(f"      [READ] Checking file: {csv_path}", flush=True)
    
    if not os.path.exists(csv_path):
        print(f"      [READ] ERROR: File does not exist.", flush=True)
        return np.nan, 1e9, False
        
    try:
        df = pd.read_csv(csv_path)
        row_mask = df.iloc[:, 0] == TARGET_ROW
        
        if not row_mask.any():
            print(f"      [READ] ERROR: Target Row '{TARGET_ROW}' not found in CSV.", flush=True)
            print(f"      [READ] Available Rows: {df.iloc[:, 0].tolist()}", flush=True)
            return np.nan, 1e9, False
            
        score = df.loc[row_mask, TARGET_COL].values[0]
        
        if pd.isna(score):
            print(f"      [READ] ERROR: Score at column '{TARGET_COL}' is NaN.", flush=True)
            return np.nan, 1e9, False
            
        # Success
        cost = -score if MAXIMIZE else score
        print(f"      [READ] SUCCESS: Raw Score = {score} | Optimization Cost = {cost}", flush=True)
        return score, cost, True
        
    except Exception as e:
        print(f"      [READ] CRITICAL EXCEPTION: {e}", flush=True)
        return np.nan, 1e9, False

# ==========================================
# 4. CHECK: READ THE SCORES OF AN EARLIER RUN (--verify-read)
# ==========================================

def run_read_verification(target_folder):
    """Read the score of every step_* folder of an earlier tuning run and exit (checks the CSV reading)."""
    print(f"--- DEBUG: Verifying Read Logic in {target_folder} ---")
    print(f"Target Row: '{TARGET_ROW}' | Target Col: '{TARGET_COL}'")
    
    step_folders = sorted(glob.glob(os.path.join(target_folder, "step_*")))
    
    if not step_folders:
        print("No 'step_XXX' folders found.")
        return

    print(f"{'FOLDER':<40} | {'SCORE':<10} | {'STATUS'}")
    print("-" * 70)
    
    success_count = 0
    for folder in step_folders:
        folder_name = os.path.basename(folder)
        score, cost, success = read_score_from_folder(folder)
        
        status = "OK" if success else "FAIL"
        score_str = f"{score:.5f}" if success else "NaN"
        
        print(f"{folder_name:<40} | {score_str:<10} | {status}")
        if success: success_count += 1
            
    print("-" * 70)
    print(f"Verified {len(step_folders)} folders. Success: {success_count}/{len(step_folders)}")
    sys.exit(0)

# ==========================================
# 5. OPTIMIZATION LOGIC
# ==========================================

def load_best_from_previous_run(run_folder):
    """--resume: start from the best parameters of an earlier tuning run."""
    print(f"\n>>> LOADING PREVIOUS HISTORY from: {run_folder}", flush=True)
    history_path = os.path.join(run_folder, "optimization_history.csv")
    if not os.path.exists(history_path):
        print(f"Error: No history file in {run_folder}")
        sys.exit(1)
        
    df = pd.read_csv(history_path)
    best_row = df.sort_values(by="Cost", ascending=True).iloc[0]
    
    print(f"    Best previous run: Iteration {int(best_row['Iteration'])}", flush=True)
    print(f"    Score: {best_row['Score']:.4f} (Cost: {best_row['Cost']:.4f})", flush=True)

    updated_cnt = 0
    for key, config in PARAM_CONFIG.items():
        if key in df.columns:
            config['initial'] = best_row[key]
            if config['active']: updated_cnt += 1
            
    if updated_cnt == 0:
        print("Warning: No matching parameters found to update.", flush=True)
    else:
        print(f"    Updated {updated_cnt} parameters to best known values.", flush=True)

def setup_logging():
    """Create the output folder and the header of optimization_history.csv."""
    os.makedirs(OPTIMIZATION_DIR, exist_ok=True)
    active_keys = [k for k,v in PARAM_CONFIG.items() if v['active']]
    headers = ["Iteration", "Cost", "Score"] + active_keys
    with open(LOG_FILE, "w") as f:
        f.write(",".join(headers) + "\n")
    print(f"\n>>> OUTPUT DIRECTORY: {OPTIMIZATION_DIR}", flush=True)
    print(f">>> LOG FILE: {LOG_FILE}\n", flush=True)


def objective_function(x):
    """One evaluation: run model_eval.py with the parameters `x`, return the cost (-IoU); 1e9 on failure or out of bounds."""
    global ITERATION_COUNT
    ITERATION_COUNT += 1
    if np.isscalar(x): x = [x]
    
    print("="*60, flush=True)
    print(f"ITERATION #{ITERATION_COUNT} START", flush=True)
    print("="*60, flush=True)

    # 1. BOUNDS CHECK
    active_keys = [k for k,v in PARAM_CONFIG.items() if v['active']]
    for i, key in enumerate(active_keys):
        min_v, max_v = PARAM_CONFIG[key]['bounds']
        if x[i] < min_v or x[i] > max_v: 
            print(f"   [BOUNDS] VIOLATION: {key}={x[i]:.4f} outside ({min_v}, {max_v})", flush=True)
            return 1e9

    # 2. BUILD COMMAND
    step_prefix = f"step_{ITERATION_COUNT:03d}"
    
    cmd_args = []
    active_idx = 0
    folder_param_suffix = ""

    for key, cfg in PARAM_CONFIG.items():
        val = x[active_idx] if cfg['active'] else cfg['initial']
        if cfg['active']: 
            active_idx += 1
            folder_param_suffix += f"_{key[:4]}{val:.3f}"
        cmd_args.append(cfg['arg'])
        cmd_args.append(str(val))

    analysis_name = step_prefix + folder_param_suffix

    cmd = [
        # "uv", "run", SCRIPT_PATH,
        "python", SCRIPT_PATH,
        "--root", DATA_ROOT,
        "--output-dir", OPTIMIZATION_DIR,
        "--analysis-name", analysis_name,
        "--headless", 
        "--lift",
        "--aor", str(AOR) ,
        "--erosion-stencil", EROSION_STENCIL,
        "--tuning-mpm",
        "--mpm-density", DENSITY,
    ] + cmd_args

    print(f"   [EXEC] Running subprocess for analysis: {analysis_name}", flush=True)
    print(f"   [EXEC] Command: {' '.join(cmd)}", flush=True)

    # 3. RUN SIMULATION
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        print(f"   [EXEC] Finished successfully.", flush=True)
    except subprocess.CalledProcessError as e:
        print(f"   [EXEC] Run failed. Exit code: {e.returncode}", flush=True)
        return 1e9

    # 4. FIND THE FOLDER
    search_pattern = os.path.join(OPTIMIZATION_DIR, analysis_name + "*")
    candidates = glob.glob(search_pattern)
    candidates = [f for f in candidates if os.path.isdir(f)]

    if not candidates:
        print(f"   [ERROR] No folder found matching pattern: {search_pattern}", flush=True)
        return 1e9
    
    actual_folder = max(candidates, key=os.path.getmtime)
    print(f"   [FOUND] Located output folder: {os.path.basename(actual_folder)}", flush=True)

    # 5. READ RESULT
    score, cost, success = read_score_from_folder(actual_folder)
    
    if not success:
        print("   [RESULT] Failed to extract score. Penalizing.", flush=True)
        cost = 1e9

    # 6. LOG
    with open(LOG_FILE, "a") as f:
        vals_str = ",".join([f"{v:.6f}" for v in x])
        f.write(f"{ITERATION_COUNT},{cost:.6f},{score if success else 0.0:.6f},{vals_str}\n")
    
    return cost

# ==========================================
# 6. MAIN
# ==========================================

if __name__ == "__main__":
    print(">>> SCRIPT STARTING...", flush=True)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, help="Path to previous run folder")
    parser.add_argument("--verify-read", type=str, help="Path to previous run folder to debug CSV reading")
    parser.add_argument("--init-spread", type=float, default=0.2, help="Size of initial search area relative to bounds (0.0 to 1.0). Default 0.2 (20%%).")    

    parser.add_argument("--DATA-ROOT", required=True, type=str, default=None, help="Path where the experiment data lies.")
    parser.add_argument("--AOR", required=True, type=float, default=None, help="Angle of Repose (deg)")    
    parser.add_argument("--EROSION-STENCIL", required=True, type=str, default=None, help="Stencil for HBM.")
    parser.add_argument("--TUNING-NAME", required=True, type=str, default=None, help="Tag for this tuning session.")
    parser.add_argument("--DENSITY", required=True, type=str, default=None, help="Density of material.")
    parser.add_argument("--output-dir", type=str, default=None, help="Output folder of the tuning run (default: <DATA-ROOT>/<TUNING-NAME>_<timestamp>).")


    args = parser.parse_args()


    DATA_ROOT = args.DATA_ROOT
    AOR = args.AOR
    # Needed to run model_eval.py; HBM is not evaluated here
    EROSION_STENCIL = args.EROSION_STENCIL
    TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    TUNING_NAME = args.TUNING_NAME
    # OPTIMIZATION_DIR = os.path.join(DATA_ROOT, f"{TUNING_NAME}_{TIMESTAMP}")
    OPTIMIZATION_DIR = args.output_dir if args.output_dir else os.path.join(DATA_ROOT, f"{TUNING_NAME}_{TIMESTAMP}")
    LOG_FILE = os.path.join(OPTIMIZATION_DIR, "optimization_history.csv")
    DENSITY = args.DENSITY

    if args.verify_read:
        run_read_verification(args.verify_read)

    if args.resume:
        load_best_from_previous_run(args.resume)

    setup_logging()
    
    # 1. Setup Active Params
    active_keys = [k for k,v in PARAM_CONFIG.items() if v['active']]
    initial_guess = np.array([PARAM_CONFIG[k]['initial'] for k in active_keys])
    bounds_list = [PARAM_CONFIG[k]['bounds'] for k in active_keys]
    
    print(f">>> CONFIGURATION:", flush=True)
    print(f"    Active Params: {active_keys}", flush=True)
    print(f"    Initial Guess: {initial_guess}", flush=True)
    print(f"    Init Spread:   {args.init_spread} ({args.init_spread*100:.1f}% of range)", flush=True)
    
    # 2. Generate Custom Simplex
    custom_simplex = generate_bounded_simplex(
        initial_guess, 
        bounds_list, 
        step_ratio=args.init_spread
    )
    
    print(f"    Max Iter: 100 | ftol: 0.001 | xtol: 0.001", flush=True)
    print("\n>>> STARTING FMIN OPTIMIZER NOW...\n", flush=True)
    
    # 3. Run Optimization
    # Stops after 100 iterations or when parameters and cost change by less than 0.001
    best_x = fmin(
        objective_function, 
        initial_guess, 
        initial_simplex=custom_simplex,
        xtol=0.001, 
        ftol=0.001, 
        maxiter=100, 
        disp=True
    )

    if np.isscalar(best_x): best_x = [best_x]
    
    print("\n" + "="*60, flush=True)
    print("OPTIMIZATION FINISHED", flush=True)
    print("="*60, flush=True)
    
    summary_path = os.path.join(OPTIMIZATION_DIR, "final_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Best Result:\n")
        print("Final Best Parameters:", flush=True)
        for i, k in enumerate(active_keys):
            line = f"{k}: {best_x[i]:.6f}"
            print(f"  {line}", flush=True)
            f.write(f"{line}\n")
            
    print(f"\nFull results saved to: {OPTIMIZATION_DIR}", flush=True)
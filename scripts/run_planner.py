"""
Clearance planner: plans pushes that clear granular material out of a circular region (paper: clearance task).

From one scan of the pile, the planner predicts every candidate push with the HBM, picks the one
that leaves the least material in the region, and repeats on the predicted map (system.n_steps
pushes, open-loop). Settings: one YAML file per run in granular_pushing/planner/configs/.

Two modes:
  --offline-input  Plan on a recorded scan (.ply file or folder), no robot. Used to re-run the experiments.
  --data-root      Live on the robot: watch <data-root>/<system.ply_location> for new scans. A scan with
                   PRESWEEP in its name triggers planning; the pushes are written to
                   <active-root>/actions_log.csv for the robot. Scans with POSTSWEEP are scored and
                   shown next to the prediction.

Output: one <timestamp>_planning_session/ folder, in --archive-root (offline) or in
<data-root>/planner_archive/ (live), with actions_log.csv (the planned pushes; the first rows,
INIT_TEST, are boundary test pushes), the config used, clearance_scores.csv per scan and images.
Opens a live window.

Usage:
    python scripts/run_planner.py --offline-input <scan.ply> --config <config.yaml> --archive-root <folder>
    python scripts/run_planner.py --data-root <run_folder> --config <config.yaml> --active-root <shared_folder>
"""
import argparse
import time
import logging
import shutil
from pathlib import Path
from datetime import datetime

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from ruamel.yaml import YAML

from planner_policies import get_planner

from model_eval_utils import (
    compute_height_map,
    wait_for_stability,
    visualize_height_map,
    fig_to_array
)
from hbm_model import elevationMap2D, parse_erosion_stencil, compute_clearance_volume

# --- YAML Configuration ---
ryaml = YAML()
ryaml.preserve_quotes = True
ryaml.indent(mapping=2, sequence=4, offset=2)
ryaml.representer.add_representer(type(None), lambda self, data: self.represent_scalar('tag:yaml.org,2002:null', 'null'))

def safe_float_representer(self, data):
    return self.represent_float(data)

ryaml.representer.add_representer(float, safe_float_representer)
ryaml.representer.add_representer(np.float64, safe_float_representer)

# --- Configuration Constants ---
POLL_INTERVAL = 0.5
FILE_STABILITY_WAIT = 0.5
CSV_HEADER = "global_step,source_filename,rollout_step,start_x,start_y,end_x,end_y\n"
VIZ_ROTATE_DEG = 270  # rotation of the live-window images (display only)

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("Planner")


# ==============================================================================
# 1. CORE HELPERS & ITERATORS
# ==============================================================================

def update_nested_config_strict(config_dict, key_path, value):
    keys = key_path.split('.')
    curr = config_dict
    for i, k in enumerate(keys):
        if k not in curr:
            raise KeyError(f"Path '{'.'.join(keys[:i+1])}' not found in YAML.")
        if i == len(keys) - 1:
            orig_val = curr[k]
            if isinstance(orig_val, bool):
                curr[k] = str(value).lower() in ['true', '1', 't', 'y', 'yes']
            elif isinstance(orig_val, (float, int)) and not isinstance(orig_val, bool):
                curr[k] = float(value) if '.' in str(value) else int(value)
            elif orig_val is None:
                curr[k] = value
            else:
                curr[k] = type(orig_val)(value)
        else:
            curr = curr[k]

def get_offline_iterator(input_path: Path):
    logger.info(f"Monitoring offline source: {input_path}")
    if input_path.is_file():
        yield ("ANY", input_path)
    elif input_path.is_dir():
        for file_path in sorted(list(input_path.glob("*.ply"))):
            yield ("ANY", file_path)

def get_live_iterator(source_dir: Path, dest_dir: Path, pre_filter: str, post_filter: str):
    processed = {f.name for f in source_dir.glob("*.ply")} if source_dir.exists() else set()
    logger.info(f"Monitoring LIVE source: {source_dir} -> Copying to {dest_dir}")
    
    while True:
        if not source_dir.exists():
            time.sleep(1)
            continue

        for file_path in sorted(list(source_dir.glob("*.ply"))):
            if file_path.name not in processed:
                wait_for_stability(file_path, FILE_STABILITY_WAIT)
                dest_path = dest_dir / file_path.name
                shutil.copy2(file_path, dest_path)
                processed.add(file_path.name)
                
                if pre_filter and pre_filter in file_path.name:
                    yield ("PRE", dest_path)
                elif post_filter and post_filter in file_path.name:
                    yield ("POST", dest_path)
                elif not pre_filter and not post_filter:
                    yield ("ANY", dest_path)
        
        time.sleep(POLL_INTERVAL)


# ==============================================================================
# 2. STORAGE & LOGGING MANAGER
# ==============================================================================

class SessionStorage:
    """Handles all directory creation, clear file naming, and CSV logging."""
    def __init__(self, root_dir: Path, active_dir: Path = None):
        self.root = root_dir
        self.active_dir = active_dir
        self.actions_csv = self.root / "actions_log.csv"
        self.active_actions_csv = self.active_dir / "actions_log.csv" if self.active_dir else None
        
        # Setup base directories
        self.root.mkdir(parents=True, exist_ok=True)
        if self.active_dir:
            self.active_dir.mkdir(parents=True, exist_ok=True)

        with open(self.actions_csv, 'w') as f: 
            f.write(CSV_HEADER)
            
        # The robot reads the planned pushes from the active actions_log
        if self.active_actions_csv:
            with open(self.active_actions_csv, 'w') as f:
                f.write(CSV_HEADER)

    def _write_action_row(self, row: str):
        """Helper to cleanly write to both archive and active robot CSVs."""
        with open(self.actions_csv, 'a') as f: 
            f.write(row)
        if self.active_actions_csv:
            with open(self.active_actions_csv, 'a') as f: 
                f.write(row)

    def log_test_sweeps(self, test_sweeps):
        """Logs initial boundary sweeps to both logs."""
        for sx, sy, ex, ey in test_sweeps:
            self._write_action_row(f"-1,INIT_TEST,-1,{sx},{sy},{ex},{ey}\n")

    def create_scan_dir(self, scan_idx: int) -> Path:
        scan_dir = self.root / f"physical_scan_{scan_idx:03d}"
        scan_dir.mkdir(parents=True, exist_ok=True)
        return scan_dir


    def log_score(self, scan_dir: Path, filename: str, actual_score: float, predicted_score: float = None, is_initial: bool = False):
        csv_path = scan_dir / "clearance_scores.csv"
        mode = 'w' if is_initial else 'a'
        with open(csv_path, mode) as f:
            if is_initial: 
                f.write("filename,actual_volume,predicted_volume\n")
            
            # If there's no prediction (e.g., the very first input scan), log "N/A"
            pred_str = f"{predicted_score:.5f}" if predicted_score is not None else "N/A"
            f.write(f"{filename},{actual_score:.5f},{pred_str}\n")

    def log_action(self, step: int, filename: str, rollout: int, sweep_dict: dict):
        row = f"{step},{filename},{rollout},{sweep_dict['PRE'][0]:.3f},{sweep_dict['PRE'][1]:.3f},{sweep_dict['POST'][0]:.3f},{sweep_dict['POST'][1]:.3f}\n"
        self._write_action_row(row)

    def save_hmap(self, base_dir: Path, prefix: str, fig, hmap_arr, x_grid, y_grid, layout="split"):
        """
        layout='flat': Saves both .png and .npz directly in base_dir
        layout='rollout': Saves .png in base_dir, but tucks .npz into base_dir/raw
        layout='split': Cleanly separates into base_dir/viz and base_dir/raw
        """
        base_dir.mkdir(parents=True, exist_ok=True)
        
        if layout == "flat":
            fig.savefig(base_dir / f"{prefix}_2D.png", bbox_inches='tight')
            np.savez(base_dir / f"{prefix}.npz", heightmap=hmap_arr, x_grid=x_grid, y_grid=y_grid)
        
        elif layout == "rollout":
            raw_dir = base_dir / "raw"
            raw_dir.mkdir(exist_ok=True)
            
            fig.savefig(base_dir / f"{prefix}_2D.png", bbox_inches='tight')
            np.savez(raw_dir / f"{prefix}.npz", heightmap=hmap_arr, x_grid=x_grid, y_grid=y_grid)
            
        elif layout == "split":
            viz_dir = base_dir / "viz"
            raw_dir = base_dir / "raw"
            viz_dir.mkdir(exist_ok=True)
            raw_dir.mkdir(exist_ok=True)
            
            fig.savefig(viz_dir / f"{prefix}_2D.png", bbox_inches='tight')
            np.savez(raw_dir / f"{prefix}.npz", heightmap=hmap_arr, x_grid=x_grid, y_grid=y_grid)

    def log_candidate(self, rollout_dir: Path, cand_idx: int, score: float, p_time: float, start, end, fig, hmap_arr, x_grid, y_grid):
        cand_dir = rollout_dir / "candidates"
        cand_dir.mkdir(exist_ok=True)
        
        csv_path = cand_dir / "sweep_candidates.csv"
        if not csv_path.exists():
            with open(csv_path, 'w') as f:
                f.write("candidate_idx,volume_score,prediction_time,start_x,start_y,end_x,end_y\n")

        with open(csv_path, 'a') as f:
            f.write(f"{cand_idx},{score:.5f},{p_time:.4f},{start[0]:.3f},{start[1]:.3f},{end[0]:.3f},{end[1]:.3f}\n")


        # Format to 11 chars total: 2 integer digits, 1 dot, 8 decimal digits.
        # Examples:
        # 0.00006341 -> "00.00006341"
        # 0.00036000 -> "00.00036000"
        # 1.5        -> "01.50000000"
        sort_score = f"{score:011.8f}"

        # Score first: lower is better, so sorting the files by name lists the best candidates first.
        prefix = f"score_{sort_score}_cand_{cand_idx:03d}"
        
        # File name: score first, or candidate index first
        sort_by_score = True  # Set to False to have candidate index first
        if sort_by_score:
            prefix = f"score_{sort_score}_cand_{cand_idx:03d}"
        else:
            prefix = f"cand_{cand_idx:03d}_score_{sort_score}"

        # Split candidates cleanly into viz/ and raw/
        self.save_hmap(cand_dir, prefix, fig, hmap_arr, x_grid, y_grid, layout="split")

# ==============================================================================
# 3. GUI MANAGER
# ==============================================================================

class SweepGUI:
    """Manages the Live Matplotlib Dashboard."""
    def __init__(self, n_rollout: int, k_rot: int):
        self.n_rollout = n_rollout
        self.k_rot = k_rot
        
        plt.ion()
        num_cols = 2 + n_rollout
        self.fig, self.axes = plt.subplots(2, num_cols, figsize=(4 * num_cols, 8))
        self.fig.canvas.manager.set_window_title(f'Sweep Planner ({n_rollout}-Step Rollout)')
        
        # Structure references
        self.ax_tar = self.axes[0, 0]
        self.ax_in = self.axes[0, 1]
        self.im_preds = []
        self.im_posts = []

        self.ax_tar.set_title("Goal (Target)")
        self.ax_in.set_title("Input: Waiting...")
        
        for d in range(n_rollout):
            ax_p = self.axes[0, 2 + d]
            ax_p.set_title(f"Pred: Rollout {d}")
            self.im_preds.append(ax_p.imshow(np.zeros((10, 10)))) # placeholder until the first image
            
            ax_po = self.axes[1, 2 + d]
            ax_po.set_title(f"Actual Post: Rollout {d}")
            self.im_posts.append(ax_po.imshow(np.zeros((10, 10))))

        self.axes[1, 0].set_visible(False)
        self.axes[1, 1].set_visible(False)
        
        for ax in self.axes.flatten(): ax.axis('off')
        plt.tight_layout()
        plt.show()

    def update_target(self, fig_target):
        img = np.rot90(fig_to_array(fig_target), k=self.k_rot)
        self.ax_tar.imshow(img)
        self._dummy_blank = np.zeros_like(img) # Save for resets
        
        # Init blank canvases properly now that we have target shape
        self.im_input = self.ax_in.imshow(self._dummy_blank)
        for im in self.im_preds + self.im_posts:
            im.set_data(self._dummy_blank)

    def new_scan(self, fig_input, title_stem: str):
        img = np.rot90(fig_to_array(fig_input), k=self.k_rot)
        self.im_input.set_data(img)
        self.ax_in.set_title(f"Input: {title_stem}")
        
        # Blank out old predictions and actuals
        for p, po in zip(self.im_preds, self.im_posts):
            p.set_data(self._dummy_blank)
            po.set_data(self._dummy_blank)
            
        self.refresh()

    def update_prediction(self, rollout: int, fig_pred):
        img = np.rot90(fig_to_array(fig_pred), k=self.k_rot)
        self.im_preds[rollout].set_data(img)
        self.refresh()

    def update_actual(self, rollout: int, fig_actual):
        img = np.rot90(fig_to_array(fig_actual), k=self.k_rot)
        self.im_posts[rollout].set_data(img)
        self.refresh()

    def refresh(self):
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.01)

    def save_freeze(self, filepath: Path):
        self.fig.savefig(filepath, bbox_inches='tight')


# ==============================================================================
# 4. CORE PLANNING LOOP
# ==============================================================================

def run_planner(config: dict, archive_root: Path, active_root: Path, input_iterator, n_rollout: int):
    # Config Unpacking
    sys_cfg = config['system']
    map_cfg = config['mapping']
    hbm_cfg = config['hbm']
    plan_cfg = config['planner']

    # Initialization
    storage = SessionStorage(archive_root, active_root)
    planner = get_planner(config)
    erosion_stencil = parse_erosion_stencil(hbm_cfg['erosion_stencil'])
    k_rot = (VIZ_ROTATE_DEG // 90) % 4
    gui = SweepGUI(n_rollout, k_rot)

    # Clearance region (from the config)
    clearance_cfg = plan_cfg.get('clearance_region', {'center': None, 'radius': None})
    cx, cy = clearance_cfg['center']
    c_radius = clearance_cfg['radius']

    # Setup Target Map
    target_hmap = elevationMap2D()
    target_hmap.setMetricConfig(map_width=map_cfg['map_width'], resolution=map_cfg['map_resolution'], label="target")
    # Use the YAML parameters and set elevation to 1.0 to create a clean mask
    target_hmap.overrideWithCircleElevation(const_elevation=1.0, xy_centers=[np.array([cx, cy])], radius=c_radius)
    
    # Extract the binary mask once and for all
    target_mask = (target_hmap.elevation_data > 0.5).astype(np.float32)
    
    x_grid = np.linspace(-map_cfg['map_width']/2, map_cfg['map_width']/2, target_hmap.elevation_data.shape[0])
    y_grid = np.linspace(-map_cfg['map_width']/2, map_cfg['map_width']/2, target_hmap.elevation_data.shape[1])

    fig_tar, _, _ = visualize_height_map(target_hmap.elevation_data, x_grid, y_grid, max_height=sys_cfg['viz_max_height'], title="TARGET", headless=True)
    
    # Save target directly into the session root
    storage.save_hmap(storage.root, "target_map", fig_tar, target_hmap.elevation_data, x_grid, y_grid, layout="flat")
    
    gui.update_target(fig_tar)
    plt.close(fig_tar)

    # Bounds & Init Test Sweeps
    bnd = plan_cfg.get('start_bounds', [[-0.1, -0.1], [0.1, 0.1]])
    xmin, ymin, xmax, ymax = bnd[0][0], bnd[0][1], bnd[1][0], bnd[1][1]
    test_sweeps = [
        (xmin, 0.0, xmax, 0.0), (xmax, 0.0, xmin, 0.0),
        (0.0, ymin, 0.0, ymax), (0.0, ymax, 0.0, ymin),
        (xmin, ymin, xmax, ymin), (xmax, ymin, xmin, ymin),
        (xmin, ymax, xmax, ymax), (xmax, ymax, xmin, ymax),
        (xmin, ymin, xmin, ymax), (xmin, ymax, xmin, ymin),
        (xmax, ymin, xmax, ymax), (xmax, ymax, xmax, ymin),
        (xmin, ymin, xmax, ymax), (xmax, ymax, xmin, ymin),
        (xmin, ymax, xmax, ymin), (xmax, ymin, xmin, ymax)
    ]
    storage.log_test_sweeps(test_sweeps)

    # Tracking Variables
    global_step = 0
    scan_idx = 0
    current_post_rollout = 0
    current_scan_dir = None
    cell_area = map_cfg['map_resolution'] ** 2

    # Best predicted score of the last plan (logged with the next POST scan)
    last_predicted_score = None

    # --- MAIN EVENT LOOP ---
    for file_type, physical_file in input_iterator:
        try:
            filename = physical_file.name
            
            # --- PROCESS POINT CLOUD ---
            raw_pcd = o3d.io.read_point_cloud(str(physical_file))
            hmap_arr, xg, yg = compute_height_map(
                raw_pcd, map_cfg['map_width'], map_cfg['map_width'], 
                resolution=map_cfg['map_resolution'], aggregation_method=map_cfg['agg_method'],
                height_cutoff=map_cfg['height_cutoff']
            )

            vol_score = compute_clearance_volume(hmap_arr, target_mask, cell_area)

            # ==========================================
            # EVENT: POST-SWEEP EVALUATION
            # ==========================================
            if file_type == "POST":
                logger.info(f"--- ACQUIRED POSTSWEEP: {filename} ---")
                
                if current_scan_dir:
                    # Place the POST physical file into the input folder
                    input_dir = current_scan_dir / "input"
                    input_dir.mkdir(exist_ok=True)
                    if physical_file.exists():
                        shutil.copy2(physical_file, input_dir / filename)
                    
                    storage.log_score(current_scan_dir, filename, vol_score, predicted_score=last_predicted_score, is_initial=False)
                
                if current_post_rollout < n_rollout:
                    fig_post, _, _ = visualize_height_map(hmap_arr, xg, yg, max_height=sys_cfg['viz_max_height'], title="Actual", headless=True)
                    
                    rollout_dir = current_scan_dir / f"rollout_{current_post_rollout}"
                    rollout_dir.mkdir(exist_ok=True)
                    
                    # Store the measured map
                    storage.save_hmap(rollout_dir, f"step_{global_step:04d}_MEASURED", fig_post, hmap_arr, xg, yg, layout="rollout")
                    
                    gui.update_actual(current_post_rollout, fig_post)
                    plt.close(fig_post)
                    
                    gui.save_freeze(current_scan_dir / "live_gui_freeze.png")
                    current_post_rollout += 1
                else:
                    logger.warning(f"POST file received but rollout max ({n_rollout}) reached.")
                continue

            # ==========================================
            # EVENT: PRE-SWEEP PLANNING & ROLLOUT
            # ==========================================
            logger.info(f"--- ACQUIRED PRESWEEP: {filename} ---")
            scan_idx += 1
            current_post_rollout = 0
            current_scan_dir = storage.create_scan_dir(scan_idx)
            
            # 1. Setup Input Folder inside the scan directory
            input_dir = current_scan_dir / "input"
            input_dir.mkdir(exist_ok=True)
            if physical_file.exists():
                shutil.copy2(physical_file, input_dir / filename)

            storage.log_score(current_scan_dir, filename, vol_score, predicted_score=None, is_initial=True)
            
            current_hmap = elevationMap2D()
            current_hmap.setMetricConfig(map_cfg['map_width'], map_cfg['map_resolution'], label="current")
            current_hmap.elevation_data = hmap_arr

            fig_in, _, _ = visualize_height_map(hmap_arr, xg, yg, max_height=sys_cfg['viz_max_height'], title="Input", headless=True)
            
            # Save Input directly into the input folder
            storage.save_hmap(input_dir, f"scan_{scan_idx:03d}_input", fig_in, hmap_arr, xg, yg, layout="flat")
            
            gui.new_scan(fig_in, f"Scan {scan_idx:03d}")
            plt.close(fig_in)
            gui.save_freeze(current_scan_dir / "live_gui_freeze.png")

            # --- N-STEP SIMULATION ROLLOUT ---
            for rollout in range(n_rollout):
                logger.info(f"Scan {scan_idx} | Planning Rollout {rollout}/{n_rollout} | Global Step: {global_step}")
                rollout_dir = current_scan_dir / f"rollout_{rollout}"
                rollout_dir.mkdir(exist_ok=True)

                # 1. Plan
                current_hmap.elevation_data = np.clip(current_hmap.elevation_data, 0.0, None)
                
                sweep_dict = planner.plan(current_hmap, target_mask, cell_area)
                
                # Save the planner's best predicted score for the next POST log
                if sweep_dict.get('candidates_info'):
                    last_predicted_score = min(cand['vol_in_zone'] for cand in sweep_dict['candidates_info'])
                else:
                    last_predicted_score = None
                
                # 2. Log Candidates
                for idx, cand in enumerate(sweep_dict.get('candidates_info', [])):
                    c_map = getattr(cand['predicted_map'], 'elevation_data', cand['predicted_map'])
                    s_xy, e_xy = cand['sweep'][:2], cand['sweep'][2:]
                    
                    # Candidate colour (black if none)
                    sweep_color = cand.get('color', 'black') 
                    
                    fig_c, _, _ = visualize_height_map(
                        c_map, xg, yg, 
                        sweep_to_viz={'PRE': s_xy, 'POST': e_xy}, 
                        sweep_width=hbm_cfg['sweep_width'], 
                        sweep_color=sweep_color,
                        max_height=sys_cfg['viz_max_height'], 
                        title="Cand", 
                        headless=True
                    )
                    
                    storage.log_candidate(rollout_dir, idx, cand['vol_in_zone'], cand['time_for_prediction'], s_xy, e_xy, fig_c, c_map, xg, yg)
                    plt.close(fig_c)

                # 3. Simulate Best Action
                storage.log_action(global_step, filename, rollout, sweep_dict)
                hbm_pred = elevationMap2D.sweepModelDisplacementAndErosion(
                    current_hmap, 
                    xy_sweep_start=sweep_dict['PRE'], 
                    xy_sweep_end=sweep_dict['POST'],
                    sweep_width=hbm_cfg['sweep_width'], 
                    angle_of_repose_deg=hbm_cfg['aor'],
                    dump_base_angle_deg=hbm_cfg['dump_base_angle_deg'], 
                    relative_height_tolerance=0.95,
                    N_iter_max=hbm_cfg['erosion_max_iters'], 
                    erosion_stencil=erosion_stencil, 
                )
                predicted_hmap = hbm_pred['elev_map_predicted']

                # 4. Save PRE and PREDICTED Simulation Visuals
                fig_pre, _, _ = visualize_height_map(current_hmap.elevation_data, xg, yg, max_height=sys_cfg['viz_max_height'], title="Pre Sim", headless=True)
                storage.save_hmap(rollout_dir, f"step_{global_step:04d}_PRE", fig_pre, current_hmap.elevation_data, xg, yg, layout="rollout")
                
                fig_post_sim, _, _ = visualize_height_map(predicted_hmap.elevation_data, xg, yg, sweep_to_viz=sweep_dict, sweep_width=hbm_cfg['sweep_width'], max_height=sys_cfg['viz_max_height'], title="Post Sim", headless=True)
                storage.save_hmap(rollout_dir, f"step_{global_step:04d}_PREDICTED", fig_post_sim, predicted_hmap.elevation_data, xg, yg, layout="rollout")

                gui.update_prediction(rollout, fig_post_sim)
                plt.close(fig_pre)
                plt.close(fig_post_sim)
                gui.save_freeze(current_scan_dir / "live_gui_freeze.png")

                # Advance Step
                current_hmap = predicted_hmap
                global_step += 1

        except Exception as e:
            logger.error(f"Error processing {physical_file.name}: {e}")

    logger.info("--- WORKFLOW FINISHED ---")


# ==============================================================================
# 5. CLI ENTRY POINT
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Robot Sweep Planner YAML-based Runner")
    parser.add_argument("--offline-input", type=str, help="Path to a single .ply file OR a directory.")
    parser.add_argument("--data-root", type=str, help="Root directory for the live experiment run.")
    parser.add_argument("--n-steps", type=int, default=None, help="Pushes planned per scan (default: system.n_steps in the config).")
    parser.add_argument("--pre-filter", type=str, default="PRESWEEP", help="Text in the name of a scan that triggers planning.")
    parser.add_argument("--post-filter", type=str, default="POSTSWEEP", help="Text in the name of a post-push scan (scored and shown).")
    parser.add_argument("--config", type=str, required=True, help="e.g. granular_pushing/planner/configs/grapes/grapes_PWE.yaml")
    parser.add_argument("--active-root", type=str, help="Live mode: folder shared with the robot (overrides system.active_root).")
    parser.add_argument("--archive-root", type=str, help="Where the planning session is written (overrides system.archive_root).")
    parser.add_argument("--set", nargs='*', help="Override config values, e.g. --set planner.clearance_region.radius=0.1")
    
    args = parser.parse_args()

    if args.offline_input and args.data_root:
        parser.error("Cannot use both --offline-input and --data-root.")
    if not args.offline_input and not args.data_root:
        parser.error("Must provide either --offline-input or --data-root.")

    is_live = args.data_root is not None
    yaml_path = Path(args.config)
    
    if not yaml_path.exists(): 
        parser.error(f"Config file not found: {args.config}")
    
    with open(yaml_path, 'r') as f: 
        config = ryaml.load(f)

    if args.set:
        for item in args.set:
            if '=' in item:
                update_nested_config_strict(config, *item.split('=', 1))
                
    if args.active_root: config['system']['active_root'] = args.active_root
    if args.archive_root: config['system']['archive_root'] = args.archive_root
    if args.n_steps is not None: config['system']['n_steps'] = args.n_steps
    n_steps = int(config['system']['n_steps'])

    # Clean Timestamped Session Folder
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = f"{date_str}_planning_session"
    
    if is_live:
        data_root = Path(args.data_root).expanduser()
        if not data_root.is_dir(): parser.error(f"Data root missing: {data_root}")
        
        source_dir = data_root / config['system'].get('ply_location', '')
        
        # Archive inside data_root (store_nested) or in system.archive_root
        if config['system'].get('store_nested', True):
            archive_base = data_root / "planner_archive" / folder_name
        else:
            archive_base = Path(config['system']['archive_root']) / folder_name
            
        active_paths = Path(config['system']['active_root'])
        
        active_input_dir = active_paths / "input"
        if active_input_dir.exists():
            shutil.rmtree(active_input_dir)
        active_input_dir.mkdir(parents=True, exist_ok=True)
        
        iterator = get_live_iterator(source_dir, active_input_dir, args.pre_filter, args.post_filter)
    else:
        archive_base = Path(config['system']['archive_root']) / folder_name
        active_paths = None
        source_path = Path(args.offline_input).expanduser()
        if not source_path.exists(): parser.error(f"Offline missing: {source_path}")
        iterator = get_offline_iterator(source_path)

    print(f"\n{'='*40}\nEXPERIMENT INITIALIZED\nFolder:  {folder_name}\nArchive: {archive_base}")
    if is_live: print(f"Active:  {active_paths}")
    print(f"Rollout: {n_steps} sweeps per physical scan\n{'='*40}\n")

    archive_base.mkdir(parents=True, exist_ok=True)
    shutil.copy2(yaml_path, archive_base / "source_config.yaml")
    with open(archive_base / "resolved_run_config.yaml", 'w') as f:
        ryaml.dump(config, f)

    run_planner(config, archive_base, active_paths, iterator, n_steps)

if __name__ == "__main__":
    main()
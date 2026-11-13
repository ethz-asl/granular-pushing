"""
Clearance score of one clearance run: the volume of material left inside the clearance region,
for the scan before the pushes and after every push. Written as clearance_scores_recomputed.csv.

The planner writes the same scores during the experiment (clearance_scores.csv), but rounded to
10 cm^3. This script recomputes them with the planner's functions and settings, at full precision.
It also works for runs whose post-push scans carry a different tag (--post-filter, e.g. POSTLIFT
for coffee_beans_PWE).

What it does:
    1. Reads the planner config of the run (--config, e.g. granular_pushing/planner/configs/<material>/<material>_<model>.yaml).
    2. Finds the scans (scan folders in run_*/, e.g. PRESWEEP_SCAN, POSTSWEEP_SCAN_1 ... _10):
         - the pre-push scan (name contains --pre-filter)   -> first row (the initial state)
         - all post-push scans (name contains --post-filter) -> one row per push, in scan order
    3. For each scan: point cloud -> heightmap -> volume inside the clearance circle.
       Same functions and same settings as run_planner.py.
    4. Writes clearance_scores_recomputed.csv into planner_archive/<session>/physical_scan_001/,
       or into --output-dir if given. This is the only file it creates. Nothing else is copied,
       moved or overwritten.

predicted_volume is always written as N/A (not used anywhere).

Precision: actual_volume is in m^3 (same unit as the planner), but with 8 decimals
instead of the planner's 5. 5 decimals = 10 cm^3 steps, which rounds small residuals
to 0. 8 decimals = 0.01 cm^3, finer than the sensor can resolve.

Usage:
    python scripts/clearance_scores.py <experiment_folder> --config <config.yaml> [--output-dir <folder>]
    e.g. for the run whose post-push scans are named POSTLIFT:
    python scripts/clearance_scores.py \
        $DATA_ROOT/Clearance_HBM/coffee_beans/coffee_beans_PWE \
        --config granular_pushing/planner/configs/coffee_beans/coffee_beans_PWE.yaml \
        --post-filter POSTLIFT

    Point it at the experiment folder. It uses the single run_* folder inside it.

    --pre-filter / --post-filter default to PRESWEEP / POSTSWEEP, same as run_planner.py.
"""

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
from ruamel.yaml import YAML

# Same imports the planner uses, so the numbers come out identical.
from model_eval_utils import compute_height_map, SCAN_POINT_CLOUD
from hbm_model import elevationMap2D, compute_clearance_volume


# ------------------------------------------------------------------
# 0. Command line: the experiment folder
# ------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Write clearance_scores_recomputed.csv for one clearance run.")
parser.add_argument("experiment_folder", type=str, help="e.g. .../Clearance_HBM/coffee_beans/coffee_beans_PWE")
parser.add_argument("--config", type=str, required=True,
                    help="Planner config of the run, e.g. granular_pushing/planner/configs/grapes/grapes_PWE.yaml.")
parser.add_argument("--pre-filter", type=str, default="PRESWEEP", help="Text in the pre-push scan's filename.")
parser.add_argument("--post-filter", type=str, default="POSTSWEEP", help="Text in the post-push scans' filenames.")
parser.add_argument("--output-dir", type=str, default=None,
                    help="Write the CSV into this folder instead of the run's physical_scan_001/ folder.")
args = parser.parse_args()

experiment_folder = Path(args.experiment_folder).expanduser()
pre_filter = args.pre_filter
post_filter = args.post_filter

# The experiment folder contains exactly one run_* folder. Use that one.
run_folders = sorted(p for p in experiment_folder.glob("run_*") if p.is_dir())
if len(run_folders) != 1:
    raise SystemExit(f"Expected exactly 1 run_* folder in {experiment_folder}, found {len(run_folders)}: "
                     f"{[p.name for p in run_folders]}")
run_folder = run_folders[0]
print(f"Run folder: {run_folder}")
print(f"Looking for: pre-push = *{pre_filter}*.ply, post-push = *{post_filter}*.ply")


# ------------------------------------------------------------------
# 1. Read the config and find the planning session (default output folder)
# ------------------------------------------------------------------
# There should be exactly one planning session inside planner_archive/.
session_folders = sorted((run_folder / "planner_archive").glob("*planning_session"))
if len(session_folders) != 1:
    raise SystemExit(f"Expected exactly 1 planning session in planner_archive/, found {len(session_folders)}.")
session_folder = session_folders[0]
print(f"Planning session: {session_folder.name}")

with open(Path(args.config).expanduser(), "r") as f:
    config = YAML().load(f)

map_width = config["mapping"]["map_width"]
map_resolution = config["mapping"]["map_resolution"]
aggregation_method = config["mapping"]["agg_method"]
height_cutoff = config["mapping"]["height_cutoff"]

circle_center_x, circle_center_y = config["planner"]["clearance_region"]["center"]
circle_radius = config["planner"]["clearance_region"]["radius"]

print(f"Reading scans from: {run_folder}")


# ------------------------------------------------------------------
# 2. Build the clearance-region mask (copied from run_planner.py)
# ------------------------------------------------------------------
target_map = elevationMap2D()
target_map.setMetricConfig(map_width=map_width, resolution=map_resolution, label="target")
target_map.overrideWithCircleElevation(const_elevation=1.0,
                                       xy_centers=[np.array([circle_center_x, circle_center_y])],
                                       radius=circle_radius)
target_mask = (target_map.elevation_data > 0.5).astype(np.float32)

cell_area = map_resolution ** 2


# ------------------------------------------------------------------
# 3. Collect the scans: 1 pre-push scan, then all post-push scans in scan order
# ------------------------------------------------------------------
# Scan folders in the run folder, e.g. PRESWEEP_SCAN, POSTSWEEP_SCAN_1 ... POSTSWEEP_SCAN_10,
# each with one point_cloud.ply.
def scan_number(scan_dir):
    # "POSTSWEEP_SCAN_10" -> 10 ("PRESWEEP_SCAN" -> 0).
    # Sorting by this number matters: sorting by name would put _10 before _2.
    last = scan_dir.name.split("_")[-1]
    return int(last) if last.isdigit() else 0

def scan_cloud(scan_dir):
    plys = list(scan_dir.glob(SCAN_POINT_CLOUD))
    if len(plys) != 1:
        raise SystemExit(f"Expected exactly 1 point cloud in {scan_dir}, found {len(plys)}")
    return plys[0]

pre_push_files = [scan_cloud(d) for d in sorted(run_folder.glob(f"*{pre_filter}*"), key=scan_number) if d.is_dir()]
post_push_files = [scan_cloud(d) for d in sorted(run_folder.glob(f"*{post_filter}*"), key=scan_number) if d.is_dir()]

if len(pre_push_files) != 1:
    raise SystemExit(f"Expected exactly 1 '{pre_filter}' scan, found {len(pre_push_files)}: "
                     f"{[p.name for p in pre_push_files]}")
if len(post_push_files) == 0:
    raise SystemExit(f"No '{post_filter}' scans found in {run_folder}. Wrong --post-filter?")

scans_in_order = pre_push_files + post_push_files
print(f"Found 1 '{pre_filter}' + {len(post_push_files)} '{post_filter}' scans.")


# ------------------------------------------------------------------
# 4. Compute the volume inside the clearance region for every scan
# ------------------------------------------------------------------
rows = []
for ply_path in scans_in_order:
    point_cloud = o3d.io.read_point_cloud(str(ply_path))

    heightmap, _, _ = compute_height_map(point_cloud, map_width, map_width,
                                         resolution=map_resolution,
                                         aggregation_method=aggregation_method,
                                         height_cutoff=height_cutoff)

    actual_volume = compute_clearance_volume(heightmap, target_mask, cell_area)

    rows.append((ply_path.parent.name, actual_volume))  # scan folder name, e.g. POSTSWEEP_SCAN_3
    print(f"  {ply_path.parent.name:50s} actual_volume = {actual_volume:.8f} m^3 ({actual_volume * 1e6:8.2f} cm^3)")


# ------------------------------------------------------------------
# 5. Write clearance_scores_recomputed.csv (same format as the planner, but more decimals)
# ------------------------------------------------------------------
scan_folder = session_folder / "physical_scan_001"
# Only ever writes this one new file. The planner's clearance_scores.csv is never touched.
if args.output_dir:
    scan_folder = Path(args.output_dir).expanduser()
    scan_folder.mkdir(parents=True, exist_ok=True)
csv_path = scan_folder / "clearance_scores_recomputed.csv"

with open(csv_path, "w") as f:
    f.write("filename,actual_volume,predicted_volume\n")
    for filename, actual_volume in rows:
        # 8 decimals in m^3 = 0.01 cm^3 resolution (the planner uses 5 = 10 cm^3).
        f.write(f"{filename},{actual_volume:.8f},N/A\n")

print(f"Wrote: {csv_path}")

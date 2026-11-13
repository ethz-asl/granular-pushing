"""
Evaluate the push models against recorded experiments.

Compares the heightmap-based model (HBM) and MPM with the measured scans
of one experiment folder (`--root`), in one of two modes:

  --lift  Bucket lift (calibration). Two scenarios per run:
            Settling:  start from the known bucket shape, predict the settled pile.
            Retention: start from the measured settled pile, check it stays put.
          Also estimates the angle of repose from the scans if `--aor` is not given.
  --push  Push experiments. Per run, two pushes are predicted:
            TaskA_Sweep1 and TaskB_Sweep2: one push each, from the measured map before it (1-step).
            TaskC_Chained: both pushes from the initial map (2-step).
          HBM variants (`--hbm-variants`): PS, PW, PSE, PWE.

MPM is simulated through mpm_runner.py, or loaded from an earlier run with `--mpm-root`.

Outputs (in the output folder): LIFT_SUMMARY_*.csv or SUMMARY_*.csv (IoU per model), METRICS_*.csv,
TIMINGS_*.csv, estimated_aor.txt, plane_fit.txt, height map images per run,
the MPM inputs/outputs (mpm_*_staging/), and the configuration (0_config_params.yaml).
See docs/REPRODUCE.md for the commands used in the paper.

Usage:
    python scripts/model_eval.py --lift --headless --root <bucketlift_folder> --aor <deg> \
        --erosion-stencil <stencil> <MPM flags> --output-dir <folder> --analysis-name <name>
    python scripts/model_eval.py --push --headless --root <push_folder> ...   (same flags)
"""
import os
import argparse
import yaml
import numpy as np
import pandas as pd
import logging
import sys
import collections
from scipy.spatial.transform import Rotation as RScipy
import time 

from mpm_runner import run_mpm_simulation

from hbm_model import (
    compute_error_metrics,
    elevationMap2D, generate_erosion_update_path,
    make_kernel_LUT, parse_erosion_stencil,
    erodeHeightMapNUMBA_Safe,
    erodeHeightMapNUMBA_Warmup,

)

from model_eval_utils import (
    normalize_volume_to_target,
    load_pc_from_path_and_key,
    compute_height_maps,
    transform_sweep,
    estimate_angle_of_repose,
    find_init_shapes, find_pointclouds, find_sweeps,
    fit_plane_with_reference, init_sim_map,
    transform_pointcloud,
    visualize_height_map,
)

def setup_logging(output_dir, verbose):
    """Sets up logging to both console and a file in the output directory."""
    log_level = logging.DEBUG if verbose else logging.INFO
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # clear previous handlers if running in interactive environment
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Console Handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(log_level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File Handler
    log_file = os.path.join(output_dir, "experiment_log.txt")
    fh = logging.FileHandler(log_file, mode='w')
    fh.setLevel(log_level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger

def get_run_config():
    """
    Parses CLI args, attaches static config data, and calculates derived paths.
    Returns an argparse.Namespace object for clean dot-notation access (cfg.root).
    """
    parser = argparse.ArgumentParser(description="Experiment Preprocessing and Simulation")

    # --- Group 1: I/O and System ---
    sys_group = parser.add_argument_group('System & I/O')
    sys_group.add_argument("--root", type=str, required=True, help="Path to EXPERIMENT_ROOT")
    sys_group.add_argument("--output-dir", type=str, default=None, help="Path to OUTPUT_DIR (overrides analysis-name)")
    sys_group.add_argument("--analysis-name", type=str, default="analysis", help="Base name of output folder")
    sys_group.add_argument("--headless", action='store_true', help="Run without visualization")
    sys_group.add_argument("--verbose", action='store_true', help="Enable verbose logging and viz")
    sys_group.add_argument("--random-seed", type=int, default=42, help="Random seed for reproducibility (e.g., erosion path generation)")

    # --- MPM options ---
    sys_group.add_argument("--mpm-root", type=str, default=None, help="Path to MPM simulation data")
    sys_group.add_argument("--mpm-friction-coeff", type=float, default=None, help="Friction coefficient")
    sys_group.add_argument("--mpm-critical-fraction", type=float, default=None, help="Critical fraction")
    sys_group.add_argument("--mpm-ground-friction", type=float, default=None, help="Ground friction")
    sys_group.add_argument("--mpm-yield-stress", type=float, default=None, help="Yield stress")
    sys_group.add_argument("--mpm-voxel-size", type=float, default=None, help="MPM voxel size (meters)")

    sys_group.add_argument("--mpm-density", type=float, default=None, help="Yield stress")

    sys_group.add_argument("--tuning-mpm", action='store_true', help="If we are tuning MPM parameters (skip some steps)")
    sys_group.add_argument("--finish-after-mpm-gen", action='store_true', help="Finish script after MPM data generation (skip analysis)")
    
    # --- Preprocessing ---
    prep_group = parser.add_argument_group('Preprocessing Options')
    prep_group.add_argument("--ransac-plane-distance-threshold", type=float, default=0.01, help="RANSAC plane fitting distance threshold (meters)")
    prep_group.add_argument("--ransac-plane-num-pts", type=int, default=3, help="RANSAC plane fitting number of points per iteration")
    prep_group.add_argument("--ransac-plane-iterations", type=int, default=1000, help="RANSAC plane fitting number of iterations")
    prep_group.add_argument("--ransac-plane-max-tilt-deg", type=float, default=10.0, help="RANSAC plane fitting max tilt (degrees)")

    # --- Map and physics ---
    map_group = parser.add_argument_group('Map & Physics Parameters')
    map_group.add_argument("--map-width", type=float, default=0.6, help="Width of height map (meters)")
    map_group.add_argument("--resolution", type=float, default=0.005, help="Resolution (meters per pixel)")
    map_group.add_argument("--height-cutoff", type=float, default=0.004, help="Ignore points below this height (meters)")
    map_group.add_argument("--aggregation-method", type=str, default="mean", choices=["max", "min", "mean", "median"], help="Height map aggregation method")
    map_group.add_argument("--aor", type=float, default=None, help="Angle of repose (degrees). If None, estimated from data")
    map_group.add_argument("--erosion-rel-height-tol", type=float, default=0.95, help="Erosion: relative height tolerance")
    map_group.add_argument("--finish-after-aor-estimation", action='store_true', help="Finish script after AoR estimation (skip rest)")

    # --- HBM erosion ---
    sim_group = parser.add_argument_group('Simulation Settings')
    sim_group.add_argument("--erosion-stencil", type=str, default=None, help="Erosion: neighbourhood stencil")
    sim_group.add_argument("--erosion-max-iters", type=int, default=5000, help="Erosion: max iterations")

    # --- Lift experiments ---
    lift_group = parser.add_argument_group('Lift Experiment Options')
    lift_group.add_argument("--lift", action='store_true', help="Experiment involves lifting")

    # --- Push experiments ---
    push_group = parser.add_argument_group('Push Experiment Options')
    push_group.add_argument("--push", action='store_true', help="Experiment involves pushing")
    push_group.add_argument("--sweep-width", type=float, default=0.0735, help="Width of sweeper (meters)")
    push_group.add_argument("--hbm-variants", type=str, nargs='+', default=["PS", "PW", "PSE", "PWE"], 
                            help="HBM variants for push experiments: PS, PW (wedge at 89 deg / at the angle of repose), PSE, PWE (+ erosion)")

    # --- Visualization ---
    viz_group = parser.add_argument_group('Visualization Options')
    viz_group.add_argument("--viz-height-limit", type=float, default=0.05, help="Max height for visualization color scale (meters)")
    viz_group.add_argument("--minimal-viz", action='store_true', help="Minimal visualization during processing")
    # Images: 2D PNG by default; --save-3d and --save-pdf add more
    viz_group.add_argument("--save-3d", action='store_true', help="Also save 3D views of the height maps (slow)")
    viz_group.add_argument("--save-pdf", action='store_true', help="Also save every height map image as PDF")
    viz_group.add_argument("--no-images", action='store_true', help="Save no height map images at all (numbers only, fastest)")
    viz_group.add_argument("--hmap-cmap", type=str, default="viridis", help="Colormap for height map visualization (any matplotlib colormap)")

    args = parser.parse_args()

    # --- Validation ---
    if args.resolution > 0.02 or args.resolution < 0.001:
        raise ValueError("Resolution should be between 0.001 and 0.02 meters per pixel.")
    
    if args.lift and args.push:
        raise ValueError("Experiment cannot be both LIFT and PUSH. Choose one!")

    if args.push and args.aor is None:
        raise ValueError("For push experiments, AoR must be specified by the user!")

    # --- Bucket shapes used to form the initial piles (dimensions in m) ---
    args.VALID_INIT_SHAPES = {
        'CYLINDER_L': {'type': 'cylinder', 'diameter': 90.3/1000, 'height': 220.5/1000, 'volume_liters': -1},
        'CYLINDER_S': {'type': 'cylinder', 'diameter': 90.3/1000, 'height': 125.27/1000, 'volume_liters': -1},
        'PRISM_L':    {'type': 'prism', 'side_length': 90/1000, 'height': 0.22, 'volume_liters': 0.48}
    }

    # --- Derived values ---
    erosion_stencil = parse_erosion_stencil(args.erosion_stencil)

    # Output folder: <output-dir>/<analysis-name> if --output-dir is given, else <root>/<analysis-name>_<timestamp>
    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d_%H%M")
    if args.analysis_name is None:
        folder_name = "default_analysis"
        folder_name += "_" + timestamp
    else:
        folder_name = args.analysis_name
        folder_name += "_" + timestamp
    if args.output_dir is not None:
        # fixed folder: <output-dir>/<analysis-name>, no timestamp
        args.OUTPUT_DIR = os.path.join(args.output_dir, args.analysis_name)
    else:
        args.OUTPUT_DIR = os.path.join(args.root, folder_name)

    return args, erosion_stencil

def main():
    # -------------------- Configuration --------------------
    cfg, EROSION_STENCIL = get_run_config()

    # Ensure Output Directory Exists
    if not os.path.exists(cfg.OUTPUT_DIR):
        os.makedirs(cfg.OUTPUT_DIR)

    # Setup Logging (Console + File)
    logger = setup_logging(cfg.OUTPUT_DIR, cfg.verbose)

    def save_hmap_image(*args, three_d_view=False, **kwargs):
        """Height map image for the run folder: 2D PNG by default (see --save-3d, --save-pdf, --no-images)."""
        if cfg.no_images:
            return
        visualize_height_map(*args, three_d_view=(three_d_view and cfg.save_3d), save_pdf=cfg.save_pdf, **kwargs)
    logger.info(f"Experiment initialized. Output dir: {cfg.OUTPUT_DIR}")

    # Save the full configuration next to the results
    config_path = os.path.join(cfg.OUTPUT_DIR, "0_config_params.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)
    logger.info(f"Configuration saved to {config_path}")

    cfg.view_angles = [(0,0)] # angles are (elevation, azimuth) in degrees
    # Metrics computed for every comparison (the paper reports IoU)
    cfg.req_metrics = ['iou', 'diff_max_height', 'max_abs_diff']

    logger.info("Configuration Parameters:")
    for key, value in vars(cfg).items():
        logger.info(f"  {key}: {value}")

    # -------------------- Numba warm-up (so HBM runtimes exclude compilation) --------------------
    time_for_jit_warmup = erodeHeightMapNUMBA_Warmup(random_seed=cfg.random_seed)
    # Second call: should be fast, confirms that the warm-up worked
    time_for_jit_warmup_verify = erodeHeightMapNUMBA_Warmup(random_seed=cfg.random_seed)

    # -------------------- Load the measured data --------------------

    # Pose of the bucket used to form the initial pile
    init_shapes = find_init_shapes(cfg.root, valid_init_shapes=cfg.VALID_INIT_SHAPES)
    logger.info(f"Found initial shapes for runs: {list(init_shapes.keys())}")

    # Scans of the empty workspace (used to fit the ground plane)
    clouds_empty = find_pointclouds(cfg.root, identifier="EMPTY", resolve_multiple='ILLEGAL')
    if len(clouds_empty) == 0:
        raise RuntimeError("No EMPTY scans found: Need it for plane estimation!")
    logger.info(f"Found EMPTY pointclouds for runs: {list(clouds_empty.keys())}")

    # Scans of the pile
    if cfg.lift:
        # Lift: the settled pile after lifting the bucket
        clouds_postlift = find_pointclouds(cfg.root, identifier="POSTLIFT", resolve_multiple='ALL')
        logger.info(f"LIFT: Found {len(clouds_postlift)} POSTLIFT pointclouds.")
    if cfg.push:
        # Push: before push 1, before push 2 (= after push 1), after push 2
        clouds_pre_sweep1 = find_pointclouds(cfg.root, identifier="PRESWEEP1_SCAN", resolve_multiple='ALL')
        clouds_pre_sweep2 = find_pointclouds(cfg.root, identifier="PRESWEEP2_SCAN", resolve_multiple='ALL')
        clouds_post_sweep2 = find_pointclouds(cfg.root, identifier="POSTSWEEP2_SCAN", resolve_multiple='ALL')
        assert len(clouds_pre_sweep1) == len(clouds_pre_sweep2) == len(clouds_post_sweep2), "Mismatched number of runs in PUSH pointclouds!"
        logger.info(f"PUSH: Found {len(clouds_pre_sweep1)} PRESWEEP1, {len(clouds_pre_sweep2)} PRESWEEP2, and {len(clouds_post_sweep2)} POSTSWEEP2 pointclouds.")

    # -------------------- Ground plane and coordinate frame --------------------
    # Fit the ground plane to the empty scan of run_01; all data is then expressed in this plane frame.

    plane_fit = fit_plane_with_reference(
        clouds_empty['run_01'], 
        xy_ref=np.array([0.0, 0.0]),  # note, cloud is already in workplane coords -> Use (x,y) = (0,0) as support point
        distance_threshold=cfg.ransac_plane_distance_threshold,
        ransac_n=cfg.ransac_plane_num_pts,
        num_iterations=cfg.ransac_plane_iterations,
        max_tilt_deg=cfg.ransac_plane_max_tilt_deg
    )
    # T_data_plane: plane frame -> data frame (the workplane frame of the scans)
    T_data_plane = np.eye(4)
    T_data_plane[:3, :3] = RScipy.from_quat(plane_fit['quaternion']).as_matrix()
    T_data_plane[:3, 3] = plane_fit['translation']
    with open(os.path.join(cfg.OUTPUT_DIR, "plane_fit.txt"), 'w') as f:
        f.write(f"Plane fit results:\n")
        f.write(f"  Quaternion (xyzw): {plane_fit['quaternion']}\n")
        f.write(f"  Translation (xyz): {plane_fit['translation']}\n")
        f.write(f"  Mean error distance (m): {plane_fit['mean_error_distance']}\n")
        f.write(f"  Stddev error distance (m): {plane_fit['stddev_error_distance']}\n")
    logger.info("Fitted plane to EMPTY scan and saved results.")

    # T_plane_data: data frame -> plane frame (scans, push poses and bucket pose are all given in the data frame)
    T_plane_data = np.linalg.inv(T_data_plane)

    # Move all scans into the plane frame (and drop points below the height cutoff)
    combined_clouds_for_viz = {}
    if cfg.lift:
        clouds_transformed = {}
        for run, pcd in clouds_postlift.items():
            clouds_transformed[run] = transform_pointcloud(pcd, T_plane_data, height_cutoff=cfg.height_cutoff)
        clouds_postlift = clouds_transformed    
        combined_clouds_for_viz.update({f"{run}_POSTLIFT": pcd for run, pcd in clouds_postlift.items()})
        logger.info("LIFT: Transformed POSTLIFT point clouds to plane frame.")
    if cfg.push:
        # presweep1
        clouds_transformed = {}
        for run, pcd in clouds_pre_sweep1.items():
            clouds_transformed[run] = transform_pointcloud(pcd, T_plane_data, height_cutoff=cfg.height_cutoff)
        clouds_pre_sweep1 = clouds_transformed
        combined_clouds_for_viz.update({f"{run}_PRESWEEP1": pcd for run, pcd in clouds_pre_sweep1.items()})
        # presweep2
        clouds_transformed = {}
        for run, pcd in clouds_pre_sweep2.items():
            clouds_transformed[run] = transform_pointcloud(pcd, T_plane_data, height_cutoff=cfg.height_cutoff)
        clouds_pre_sweep2 = clouds_transformed
        combined_clouds_for_viz.update({f"{run}_PRESWEEP2": pcd for run, pcd in clouds_pre_sweep2.items()})
        # postsweep2
        clouds_transformed = {}
        for run, pcd in clouds_post_sweep2.items():
            clouds_transformed[run] = transform_pointcloud(pcd, T_plane_data, height_cutoff=cfg.height_cutoff)
        clouds_post_sweep2 = clouds_transformed
        combined_clouds_for_viz.update({f"{run}_POSTSWEEP2": pcd for run, pcd in clouds_post_sweep2.items()})
        logger.info("PUSH: Transformed PUSH point clouds to plane frame.")

    # Push start/end points in the plane frame
    def extract_sweep_coords(sweep_data): 
        return [
            float(sweep_data['PRE'][0]), float(sweep_data['PRE'][1]), 
            float(sweep_data['POST'][0]), float(sweep_data['POST'][1])
        ]

    if cfg.push:
        sweeps_dict_raw = find_sweeps(cfg.root)
        sweeps_dict = {}
        for run_name, sweep_data in sweeps_dict_raw.items():
            sweep1_transformed = transform_sweep(sweep_data['sweep01'], T_cloud_data=T_plane_data)
            sweep2_transformed = transform_sweep(sweep_data['sweep02'], T_cloud_data=T_plane_data)
            sweeps_dict[run_name] = {
                'sweep01': sweep1_transformed,
                'sweep02': sweep2_transformed,
            }

    # From here on, all data is in the plane frame.

    # -------------------- Angle of repose --------------------

    # Estimated from the settled piles of a bucket lift unless given with --aor (required for push experiments)
    mean_aor, std_aor = None, None
    if cfg.aor is None:
        mean_aor, std_aor, cone_cloud_dict = estimate_angle_of_repose(clouds_postlift, init_shapes=init_shapes, T_cloud_data=T_plane_data)
        logger.info(f"Estimated AoR from REAL POSTLIFT clouds: {mean_aor:.2f}° ± {std_aor:.2f}°")
        with open(os.path.join(cfg.OUTPUT_DIR, "estimated_aor.txt"), 'w') as f:
            f.write(f"Estimated Angle of Repose (degrees): {mean_aor:.2f}\n")
            f.write(f"Estimated Stddev (degrees): {std_aor:.2f}\n")
        AOR_DEG = mean_aor
        logger.info(f"Using estimated AoR: {AOR_DEG:.2f}°")

        if cfg.finish_after_aor_estimation:
            logger.info("Finishing script after AoR estimation as per user request.")
            return
    else:
        AOR_DEG = cfg.aor
        logger.info(f"Using user-specified AoR: {AOR_DEG:.2f}°")

    # -------------------- Height maps of the measured scans --------------------
    # Returns {run_name: (height_map, x_edges, y_edges)}
    def get_height_maps(clouds, tag):
        return compute_height_maps(
            clouds, width_x=cfg.map_width, width_y=cfg.map_width, 
            resolution=cfg.resolution, height_cutoff=cfg.height_cutoff, 
            aggregation_method=cfg.aggregation_method, tag_postfix=tag
        )
    if cfg.lift:
        maps_real_postlift_settling = get_height_maps(clouds_postlift, "POSTLIFT")
        logger.info("LIFT: Computed real POSTLIFT height maps for LIFT experiment.")

    if cfg.push:
        maps_real_at_start     = get_height_maps(clouds_pre_sweep1, "PRESWEEP1")
        maps_real_after_sweep1 = get_height_maps(clouds_pre_sweep2, "PRESWEEP2")
        maps_real_after_sweep2 = get_height_maps(clouds_post_sweep2, "POSTSWEEP2")
        logger.info("PUSH: Computed real height maps for PUSH experiment.")

    # -------------------- MPM parameters (override mpm_default_config.yaml) --------------------
    config_overrides_mpm = {}
    if cfg.mpm_friction_coeff is not None:
        config_overrides_mpm["friction_coeff"] = cfg.mpm_friction_coeff

    if cfg.mpm_critical_fraction is not None:
        config_overrides_mpm["critical_fraction"] = cfg.mpm_critical_fraction

    if cfg.mpm_ground_friction is not None:
        config_overrides_mpm["groundplane_mu"] = cfg.mpm_ground_friction
    if cfg.mpm_yield_stress is not None:
        config_overrides_mpm["yield_stress"] = cfg.mpm_yield_stress
    if cfg.mpm_voxel_size is not None:
        config_overrides_mpm["voxel_size"] = cfg.mpm_voxel_size

    if cfg.mpm_density is not None:
        config_overrides_mpm["density"] = cfg.mpm_density

    # -------------------- Lift: HBM predictions and MPM data --------------------
    # In the MPM output, PRELIFT is the initialized pile and POSTLIFT the pile after settling.
    if cfg.lift:
        '''
        mpm_lift_staging/
        └── run_XX/              # run_01, run_02, etc.
            ├── settling/
            │   ├── data_to_mpm/      <- sim_heightmap_pre_settling.npz (Geometric Init)
            │   └── data_from_mpm/    <- *PRELIFT*.ply, *POSTLIFT*.ply
            └── retention/
                ├── data_to_mpm/      <- real_heightmap_post_settling.npz (Real Data Init)
                └── data_from_mpm/    <- *PRELIFT*.ply, *POSTLIFT*.ply        
        '''
        
        # 1. HBM predictions per run (Settling and Retention)
        maps_hbm_settling_init = {} # bucket shape before lifting (also the MPM start for Settling)
        maps_hbm_settling = {}      # HBM, Settling
        maps_hbm_retention = {}      # HBM, Retention
        logger.info("LIFT: Computing per-run HBM Settling and HBM Retention height maps...")
        
        times_for_settling = []
        times_for_retention = []

        erosion_steps_for_settling = []
        erosion_steps_for_retention = []

        for run_name, (map_real, _, _) in maps_real_postlift_settling.items():
            # --- Settling: bucket shape -> settled pile ---
            # Volume correction: the bucket shape is scaled to the volume measured in this run's settled pile
            run_vol = np.nansum(map_real) * (cfg.resolution**2)
            
            sim_res = init_sim_map(
                valid_init_shapes=cfg.VALID_INIT_SHAPES, 
                init_shapes=init_shapes, 
                resolution=cfg.resolution, 
                map_width=cfg.map_width, 
                angle_of_repose=AOR_DEG, 
                T_cloud_data=T_plane_data, 
                reference_volume=run_vol, 
                erosion_max_iters=cfg.erosion_max_iters, 
                erosion_stencil=EROSION_STENCIL,
                random_seed=cfg.random_seed
            )
            maps_hbm_settling_init[run_name] = sim_res['map_init_from_shapes']
            maps_hbm_settling[run_name] = sim_res['eroded_map']
            times_for_settling.append(sim_res['time_for_erosion'])
            erosion_steps_for_settling.append(sim_res['N_iter_erosion'])

            # --- Retention: measured settled pile -> erosion (should change little) ---
            sim_obj = elevationMap2D()
            sim_obj.setMetricConfig(
                map_width=cfg.map_width, 
                resolution=cfg.resolution,
                label='LIFT_RETENTION_SIM')
            sim_obj.elevation_data = map_real
            
            erosion_height_unit = sim_obj.computeHeightUnit(AOR_DEG, cfg.resolution)        
            paths = generate_erosion_update_path(*sim_obj.elevation_data.shape, cfg.erosion_max_iters, random=False, random_seed=cfg.random_seed)
            erosion_neighbors, erosion_neighbors_max = make_kernel_LUT(*sim_obj.elevation_data.shape, EROSION_STENCIL)
       
            time_for_retention = time.time()
            [stack, last_iter] = erodeHeightMapNUMBA_Safe(
                height_map=sim_obj.elevation_data, 
                paths=paths, 
                neighbors=erosion_neighbors, 
                max_neighbors=erosion_neighbors_max,
                height_unit=erosion_height_unit, 
                relative_height_tolerance=cfg.erosion_rel_height_tol,
                N_iter_max=cfg.erosion_max_iters,
                random_neighbor_order=True,
                random_seed=cfg.random_seed)
            times_for_retention.append(time.time() - time_for_retention)
            erosion_steps_for_retention.append(last_iter)
            
            maps_hbm_retention[run_name] = stack[:, :, last_iter]

        # 2. MPM: simulate (or load with --mpm-root)
        if cfg.mpm_root is None:
            logger.warning("LIFT: MPM root not provided; generating MPM data via subprocess (Per Run).")
            mpm_lift_staging_dir = os.path.join(cfg.OUTPUT_DIR, "mpm_lift_staging")
            os.makedirs(mpm_lift_staging_dir, exist_ok=True)

            for run_name in maps_real_postlift_settling.keys():
                logger.info(f"LIFT: Generating MPM data for: {run_name}")
                
                # Shared edges for this run
                x_edges = maps_real_postlift_settling[run_name][1]
                y_edges = maps_real_postlift_settling[run_name][2]

                # --- Settling: start from the (volume-matched) bucket shape ---
                task_coll_dir = os.path.join(mpm_lift_staging_dir, run_name, "settling")
                os.makedirs(os.path.join(task_coll_dir, "data_to_mpm"), exist_ok=True)
                
                hmap_path_coll = os.path.join(task_coll_dir, "data_to_mpm", "sim_heightmap_pre_settling_with_edges.npz")
                np.savez(hmap_path_coll, 
                    heightmap=maps_hbm_settling_init[run_name], 
                    x_edges=x_edges, y_edges=y_edges)

                run_mpm_simulation(
                    headless=cfg.headless,
                    config_overrides={
                        "log_directory": cfg.OUTPUT_DIR,
                        "exp_name": os.path.join(task_coll_dir, "data_from_mpm"),
                        "render_depth": True,
                        "apply_sweeps": False,
                        "init_from_heightmap": True,
                        "init_heightmap_path": hmap_path_coll,
                        **config_overrides_mpm  # This "unpacks" the pre-built dict into this one                        
                    }
                )

                # --- Retention: start from the measured settled pile ---
                task_pres_dir = os.path.join(mpm_lift_staging_dir, run_name, "retention")
                os.makedirs(os.path.join(task_pres_dir, "data_to_mpm"), exist_ok=True)

                hmap_path_pres = os.path.join(task_pres_dir, "data_to_mpm", "real_heightmap_post_settling_with_edges.npz")
                np.savez(hmap_path_pres, 
                    heightmap=maps_real_postlift_settling[run_name][0], 
                    x_edges=x_edges, y_edges=y_edges)

                if cfg.tuning_mpm:
                    # MPM tuning (--tuning-mpm) only uses Settling. To save compute, Retention is not simulated:
                    # the Settling result is copied instead, so Retention numbers from tuning runs are not meaningful.
                    import shutil
                    src_data = os.path.join(task_coll_dir, "data_from_mpm")
                    dst_data = os.path.join(task_pres_dir, "data_from_mpm")
                    
                    if os.path.exists(dst_data):
                        shutil.rmtree(dst_data) # Ensure a clean overwrite
                    shutil.copytree(src_data, dst_data)
                    logger.warning(f"LIFT: Tuning Mode - Copied SETTLING to RETENTION for {run_name}")
                else:
                    run_mpm_simulation(
                        headless=cfg.headless,
                        config_overrides={
                            "log_directory": cfg.OUTPUT_DIR,
                            "exp_name": os.path.join(task_pres_dir, "data_from_mpm"),
                            "render_depth": True,
                            "apply_sweeps": False,
                            "init_from_heightmap": True,
                            "init_heightmap_path": hmap_path_pres,
                            **config_overrides_mpm  # This "unpacks" the pre-built dict into this one
                        }
                    )
                    logger.info(f"LIFT: Completed MPM data generation for: {run_name}")
            logger.info("LIFT: Completed MPM data generation for all runs.")
        else:
            logger.info(f"LIFT: Loading MPM data from provided root: {cfg.mpm_root}")
            mpm_lift_staging_dir = cfg.mpm_root

        # 3. Load the MPM results per run
        clouds_mpm_lift_settling_pre = {}
        clouds_mpm_lift_settling_post = {}
        clouds_mpm_lift_retention_pre = {}
        clouds_mpm_lift_retention_post = {}
        
        for run_name in maps_real_postlift_settling.keys():
            path_coll = os.path.join(mpm_lift_staging_dir, run_name, "settling", "data_from_mpm")
            clouds_mpm_lift_settling_pre[run_name] = load_pc_from_path_and_key(path_coll, key="PRELIFT", extension='.ply')
            clouds_mpm_lift_settling_post[run_name] = load_pc_from_path_and_key(path_coll, key="POSTLIFT", extension='.ply')

            path_pres = os.path.join(mpm_lift_staging_dir, run_name, "retention", "data_from_mpm")
            clouds_mpm_lift_retention_pre[run_name] = load_pc_from_path_and_key(path_pres, key="PRELIFT", extension='.ply')
            clouds_mpm_lift_retention_post[run_name] = load_pc_from_path_and_key(path_pres, key="POSTLIFT", extension='.ply')

        logger.info(f"LIFT: Loaded MPM point clouds for {len(clouds_mpm_lift_settling_pre)} runs.")


        # 4. Height maps of the MPM results
        maps_mpm_lift_settling_pre = get_height_maps(clouds_mpm_lift_settling_pre, "MPM_SETTLING_PRE") 
        maps_mpm_lift_settling_post = get_height_maps(clouds_mpm_lift_settling_post, "MPM_SETTLING_POST")
        maps_mpm_lift_retention_pre = get_height_maps(clouds_mpm_lift_retention_pre, "MPM_RETENTION_PRE") 
        maps_mpm_lift_retention_post = get_height_maps(clouds_mpm_lift_retention_post, "MPM_RETENTION_POST")

        # Flip MPM maps left-right to match the orientation convention of the scans
        for target_dict in [maps_mpm_lift_settling_pre, maps_mpm_lift_settling_post, 
                            maps_mpm_lift_retention_pre, maps_mpm_lift_retention_post]:
            for r_name in target_dict:
                if target_dict[r_name] is not None:
                    target_dict[r_name] = (np.fliplr(target_dict[r_name][0]), *target_dict[r_name][1:])

        logger.info("Generated/Loaded MPM LIFT height maps for all runs (Settling & Retention).")
    
    # -------------------- Push: MPM data --------------------
    maps_mpm = collections.defaultdict(dict)

    if cfg.push:
        '''
        mpm_push_staging/
        └── run_XX/
            ├── taskA/
            │   ├── data_to_mpm/      <- real_height_map_at_start.npy
            │   └── data_from_mpm/    <- *PRESWEEP1*, *POSTSWEEP1*.ply
            ├── taskB/
            │   ├── data_to_mpm/      <- real_height_map_after_sweep1.npy
            │   └── data_from_mpm/    <- *PRESWEEP1*, *POSTSWEEP1*.ply
            └── taskC/
                ├── data_to_mpm/      <- real_height_map_at_start.npy
                └── data_from_mpm/    <- *PRESWEEP1*, *POSTSWEEP1*, *POSTSWEEP2*.ply
        '''
        mpm_push_staging_dir = None        
        if cfg.mpm_root is None:
            logger.info("PUSH: No external MPM root provided. Generating Per-Run MPM data (Geometric Init)...")
            
            mpm_push_staging_dir = os.path.join(cfg.OUTPUT_DIR, "mpm_push_staging")
            os.makedirs(mpm_push_staging_dir, exist_ok=True)
                        
            for run_name in sweeps_dict.keys():
                logger.info(f"MPM Gen for Run: {run_name}")
                sweep1_coords = extract_sweep_coords(sweeps_dict[run_name]['sweep01'])
                sweep2_coords = extract_sweep_coords(sweeps_dict[run_name]['sweep02'])

                x_edges = maps_real_at_start[run_name][1]
                y_edges = maps_real_at_start[run_name][2]

                # MPM start maps. Volume correction: each start map is scaled to the measured volume
                # of the scan the prediction is compared with (the depth sensor misses thin layers).
                # Task A: push 1 only
                mpm_init_hmap_taskA = normalize_volume_to_target(maps_real_at_start[run_name][0], maps_real_after_sweep1[run_name][0])
                data_to_mpm_folder_A = os.path.join(mpm_push_staging_dir, run_name, "taskA", "data_to_mpm")
                os.makedirs(data_to_mpm_folder_A, exist_ok=True)
                filename_A = os.path.join(data_to_mpm_folder_A, "real_heightmap_at_start_with_edges.npz")   
                np.savez(filename_A,
                    heightmap=mpm_init_hmap_taskA, 
                    x_edges=x_edges, 
                    y_edges=y_edges)

                # Task B: push 2 only
                mpm_init_hmap_taskB = normalize_volume_to_target(maps_real_after_sweep1[run_name][0], maps_real_after_sweep2[run_name][0])
                data_to_mpm_folder_B = os.path.join(mpm_push_staging_dir, run_name, "taskB", "data_to_mpm")
                os.makedirs(data_to_mpm_folder_B, exist_ok=True)
                filename_B = os.path.join(data_to_mpm_folder_B, "real_heightmap_after_sweep1_with_edges.npz")
                np.savez(filename_B,
                    heightmap=mpm_init_hmap_taskB, 
                    x_edges=x_edges, 
                    y_edges=y_edges)
                
                # Task C: both pushes
                mpm_init_hmap_taskC = normalize_volume_to_target(maps_real_at_start[run_name][0], maps_real_after_sweep2[run_name][0])
                data_to_mpm_folder_C = os.path.join(mpm_push_staging_dir, run_name, "taskC", "data_to_mpm")
                os.makedirs(data_to_mpm_folder_C, exist_ok=True)
                filename_C = os.path.join(data_to_mpm_folder_C, "real_heightmap_at_start_with_edges.npz")
                np.savez(filename_C,
                    heightmap=mpm_init_hmap_taskC, 
                    x_edges=x_edges, 
                    y_edges=y_edges)
                logger.info(f"Wrote real height maps to MPM data_to_mpm folders for {run_name}.")

                mpm_jobs = [
                    ([sweep1_coords],               f"{run_name}/taskA/", "real_heightmap_at_start_with_edges.npz"),
                    ([sweep2_coords],               f"{run_name}/taskB/", "real_heightmap_after_sweep1_with_edges.npz"),
                    ([sweep1_coords, sweep2_coords], f"{run_name}/taskC/", "real_heightmap_at_start_with_edges.npz"),
                ]

                for sweeps_list, subfolder, hmap_np_file_name in mpm_jobs:
                    path_to_file = os.path.join(mpm_push_staging_dir, subfolder, "data_to_mpm", hmap_np_file_name)
                    run_mpm_simulation(
                        headless=cfg.headless,
                        config_overrides={
                            "log_directory": cfg.OUTPUT_DIR,
                            "exp_name": os.path.join("mpm_push_staging", subfolder, "data_from_mpm"),
                            "render_depth": True,
                            "apply_sweeps": True,
                            "sweeps_xy": sweeps_list,
                            "init_from_heightmap": True,
                            "init_heightmap_path": path_to_file,
                            **config_overrides_mpm  # This "unpacks" the pre-built dict into this one
                        }
                    )
            
            logger.info("MPM Data Generation Complete.")
        else:
            mpm_push_staging_dir = cfg.mpm_root

        # Load the MPM results
        clouds_push_mpm_taskA = {
            "run_01": load_pc_from_path_and_key(folder_path=os.path.join(mpm_push_staging_dir, "run_01", "taskA", "data_from_mpm"), key="POSTSWEEP1", extension=".ply"),
            "run_02": load_pc_from_path_and_key(folder_path=os.path.join(mpm_push_staging_dir, "run_02", "taskA", "data_from_mpm"), key="POSTSWEEP1", extension=".ply"),
            "run_03": load_pc_from_path_and_key(folder_path=os.path.join(mpm_push_staging_dir, "run_03", "taskA", "data_from_mpm"), key="POSTSWEEP1", extension=".ply"),
        } # Corresponds to TaskA_Sweep1
        clouds_push_mpm_taskB = {
            "run_01": load_pc_from_path_and_key(folder_path=os.path.join(mpm_push_staging_dir, "run_01", "taskB", "data_from_mpm"), key="POSTSWEEP1", extension=".ply"),
            "run_02": load_pc_from_path_and_key(folder_path=os.path.join(mpm_push_staging_dir, "run_02", "taskB", "data_from_mpm"), key="POSTSWEEP1", extension=".ply"),
            "run_03": load_pc_from_path_and_key(folder_path=os.path.join(mpm_push_staging_dir, "run_03", "taskB", "data_from_mpm"), key="POSTSWEEP1", extension=".ply"),
        } # Corresponds to TaskB_Sweep2
        clouds_push_mpm_taskC = {
            "run_01": load_pc_from_path_and_key(folder_path=os.path.join(mpm_push_staging_dir, "run_01", "taskC", "data_from_mpm"), key="POSTSWEEP2", extension=".ply"),
            "run_02": load_pc_from_path_and_key(folder_path=os.path.join(mpm_push_staging_dir, "run_02", "taskC", "data_from_mpm"), key="POSTSWEEP2", extension=".ply"),
            "run_03": load_pc_from_path_and_key(folder_path=os.path.join(mpm_push_staging_dir, "run_03", "taskC", "data_from_mpm"), key="POSTSWEEP2", extension=".ply"),
        } # Corresponds to TaskC_Chained
        logger.info("Loaded MPM PUSH point clouds for all runs and tasks.")

        maps_push_mpm_taskA = get_height_maps(clouds_push_mpm_taskA, "MPM_TASKA")
        maps_push_mpm_taskB = get_height_maps(clouds_push_mpm_taskB, "MPM_TASKB")
        maps_push_mpm_taskC = get_height_maps(clouds_push_mpm_taskC, "MPM_TASKC")

        # Flip MPM maps left-right to match the orientation convention of the scans
        for run_name in maps_push_mpm_taskA.keys():
            maps_push_mpm_taskA[run_name] = (
                np.fliplr(maps_push_mpm_taskA[run_name][0]),
                maps_push_mpm_taskA[run_name][1],
                maps_push_mpm_taskA[run_name][2],
            )
            maps_push_mpm_taskB[run_name] = (
                np.fliplr(maps_push_mpm_taskB[run_name][0]),
                maps_push_mpm_taskB[run_name][1],
                maps_push_mpm_taskB[run_name][2],
            )
            maps_push_mpm_taskC[run_name] = (
                np.fliplr(maps_push_mpm_taskC[run_name][0]),
                maps_push_mpm_taskC[run_name][1],
                maps_push_mpm_taskC[run_name][2],
            )

    if cfg.finish_after_mpm_gen:
        logger.warning("Finish after MPM data generation flag is set. Exiting now.")
        return

    # -------------------- Lift: evaluation --------------------

    if cfg.lift:
        logger.info("Starting LIFT Experiment Analysis...")
        
        # Container for metrics
        metrics_storage = collections.defaultdict(dict)

        # =========================================================
        # PHASE 1: PROCESSING LOOP (Per Run)
        # =========================================================
        for run_name, (map_real, x_edges, y_edges) in maps_real_postlift_settling.items():
            logger.info(f"Processing Run: {run_name}")
            run_img_folder = os.path.join(cfg.OUTPUT_DIR, f"{run_name}_hm_images")
            os.makedirs(run_img_folder, exist_ok=True)

            # -----------------------------------------------------
            # STEP A: CONSOLIDATE MAPS
            # -----------------------------------------------------
            
            hbm_col_init = maps_hbm_settling_init.get(run_name)
            hbm_col_final = maps_hbm_settling.get(run_name)
            hbm_pres_final = maps_hbm_retention.get(run_name)

            # MPM dicts hold (map, x_edges, y_edges)
            mpm_col_pre_tuple = maps_mpm_lift_settling_pre.get(run_name)
            mpm_col_pre = mpm_col_pre_tuple[0] if mpm_col_pre_tuple else None
            
            mpm_col_post_tuple = maps_mpm_lift_settling_post.get(run_name)
            mpm_col_post = mpm_col_post_tuple[0] if mpm_col_post_tuple else None

            mpm_pres_pre_tuple = maps_mpm_lift_retention_pre.get(run_name)
            mpm_pres_pre = mpm_pres_pre_tuple[0] if mpm_pres_pre_tuple else None
            
            mpm_pres_post_tuple = maps_mpm_lift_retention_post.get(run_name)
            mpm_pres_post = mpm_pres_post_tuple[0] if mpm_pres_post_tuple else None

            maps_consolidated = {
                "Real": map_real,
                
                # --- Start maps (checks) ---
                "HBM_Settling_Prelift": hbm_col_init,
                "MPM_Settling_Prelift": mpm_col_pre,     # Should match HBM_Settling_Prelift (geom init)
                "MPM_Retention_Prelift": mpm_pres_pre,    # Should match Real (real init)
                
                # --- Settling ---
                "HBM_Settling": hbm_col_final,
                "MPM_Settling": mpm_col_post,
                
                # --- Retention ---
                "HBM_Retention": hbm_pres_final,
                "MPM_Retention": mpm_pres_post,
            }

            # -----------------------------------------------------
            # STEP B: ERROR & METRICS COMPUTATION
            # -----------------------------------------------------
            
            # Comparisons: label -> (reference, prediction). Predictions are compared without further volume scaling.
            # "Roundtrip" = MPM start map vs the height map it was initialized from (error of the MPM initialization).
            comparisons_def = {
                "Settling_Real_vs_HBM": ("Real", "HBM_Settling"),
                "Settling_Real_vs_MPM": ("Real", "MPM_Settling"),
                "Settling_MPM_Roundtrip_Error": ("HBM_Settling_Prelift", "MPM_Settling_Prelift"), 
                "Retention_Real_vs_HBM": ("Real", "HBM_Retention"),
                "Retention_Real_vs_MPM": ("Real", "MPM_Retention"),
                "Retention_MPM_Roundtrip_Error": ("Real", "MPM_Retention_Prelift"), 
            }

            for label, (key_GT, key_Pred) in comparisons_def.items():
                map_GT = maps_consolidated[key_GT]
                map_Pred = maps_consolidated[key_Pred]

                if map_GT is None or map_Pred is None: 
                    logger.warning(f"Skipping metrics for {label} due to missing data.")
                    continue
                
                res = compute_error_metrics(
                    mapA=map_GT, mapB=map_Pred, 
                    resolution=cfg.resolution,
                    requested=cfg.req_metrics,
                )
                
                metrics_storage[label][run_name] = res[2]

            # -----------------------------------------------------
            # STEP C: VISUALIZATION
            # -----------------------------------------------------
            
            for label, map_data in maps_consolidated.items():
                if map_data is not None:
                    save_hmap_image(
                        map_data, x_edges, y_edges, title=f"{label} ({run_name})", 
                        three_d_view=True, max_height=cfg.viz_height_limit, min_height=0,
                        cmap=cfg.hmap_cmap, 
                        headless=cfg.headless, view_angles=cfg.view_angles,
                        T_cloud_data=T_plane_data, # only used for shapes and sweeps viz, if used 
                        save_path=run_img_folder
                    )

        # =========================================================
        # PHASE 2: DUMP / STORE RESULTS
        # =========================================================
        
        for label, run_data in metrics_storage.items():
            if not run_data: continue
            
            df = pd.DataFrame.from_dict(run_data, orient='index').T
            df['Mean'] = df.mean(axis=1)
            df['StdDev'] = df.std(axis=1, ddof=1)
            
            outfile = os.path.join(cfg.OUTPUT_DIR, f"METRICS_{label}.csv")
            df.to_csv(outfile)
            logger.info(f"Saved metrics: {outfile}")

        # =========================================================
        # PHASE 3: AGGREGATE RESULTS
        # =========================================================

        logger.info("Aggregating LIFT Results...")
        
        # One row per comparison and run
        lift_long_data = []
        for label, run_dict in metrics_storage.items():
            for run_name, score_dict in run_dict.items():
                row = {
                    "Label": label,
                    "Run": run_name,
                    "EMD": score_dict.get('emd', np.nan),
                    "IoU": score_dict.get('iou', np.nan),
                    "MaxHDiff": score_dict.get('diff_max_height', np.nan),
                    'MaxAbsDiff': score_dict.get('max_abs_diff', np.nan),
                }
                lift_long_data.append(row)

        df_lift_long = pd.DataFrame(lift_long_data)

        if not df_lift_long.empty:
            long_outfile = os.path.join(cfg.OUTPUT_DIR, "LIFT_SUMMARY_LONG_FORMAT.csv")
            df_lift_long.to_csv(long_outfile, index=False)
            logger.info(f"Saved LIFT Long Format Summary: {long_outfile}")

            # Mean and std over the runs, per comparison (EMD is no longer computed; its columns stay empty)
            lift_grouped = df_lift_long.groupby(['Label'])[['EMD', 'IoU', 'MaxHDiff', 'MaxAbsDiff']].agg(['mean', 'std'])
            
            lift_grouped.columns = ['_'.join(col).strip() for col in lift_grouped.columns.values]
            
            desired_cols = []
            for metric in ['EMD', 'IoU', 'MaxHDiff', 'MaxAbsDiff']:
                for stat in ['mean', 'std']:
                    col_name = f"{metric}_{stat}"
                    if col_name in lift_grouped.columns:
                        desired_cols.append(col_name)
            
            df_lift_leaderboard = lift_grouped[desired_cols]

            leaderboard_outfile = os.path.join(cfg.OUTPUT_DIR, "LIFT_SUMMARY_LEADERBOARD.csv")
            df_lift_leaderboard.to_csv(leaderboard_outfile)
            logger.info(f"Saved LIFT Leaderboard Summary: {leaderboard_outfile}")

            # --- Lift runtimes and erosion iterations ---
            run_names_lift = list(maps_real_postlift_settling.keys())
            df_lift_times = pd.DataFrame({
                'Run': run_names_lift,
                'Settling_Time_sec': times_for_settling,
                'Retention_Time_sec': times_for_retention,
                'Warmup_sec': time_for_jit_warmup,
                'Warmup_Verify_sec': time_for_jit_warmup_verify
            }).set_index('Run')
            df_lift_times['Settling_Erosion_Steps'] = erosion_steps_for_settling
            df_lift_times['Retention_Erosion_Steps'] = erosion_steps_for_retention
            
            df_lift_times.loc['Mean'] = df_lift_times.mean()
            df_lift_times.loc['StdDev'] = df_lift_times.std(ddof=1)
            
            lift_timing_file = os.path.join(cfg.OUTPUT_DIR, "TIMINGS_LIFT.csv")
            df_lift_times.to_csv(lift_timing_file)
            logger.info(f"Saved LIFT Timings Summary: {lift_timing_file}")

    # -------------------- Push: HBM predictions and evaluation --------------------
    if cfg.push:
        logger.info("Starting PUSH Experiment Analysis...")

        # =========================================================
        # 1. SETUP MODELS & VARIANTS
        # =========================================================
        
        # HBM variants: PS (steep dump, 89°), PW (dump at the angle of repose),
        # PSE / PWE = the same with erosion. The CSV rows follow this order.
        HBM_VARIANTS = {}
        for variant_name in cfg.hbm_variants:
            if variant_name == "PS":
                HBM_VARIANTS[variant_name] = {"dump_at_aor": False, "apply_erosion": False}
            elif variant_name == "PW":
                HBM_VARIANTS[variant_name] = {"dump_at_aor": True,  "apply_erosion": False}
            elif variant_name == "PSE":
                HBM_VARIANTS[variant_name] = {"dump_at_aor": False, "apply_erosion": True}
            elif variant_name == "PWE":
                HBM_VARIANTS[variant_name] = {"dump_at_aor": True,  "apply_erosion": True}
            else:
                raise ValueError(f"Unknown HBM variant specified: {variant_name}")
        
        # Variants to plot (only the first one with --minimal-viz)
        VARIANTS_TO_VISUALIZE = list(HBM_VARIANTS.keys())
        if cfg.minimal_viz:
            VARIANTS_TO_VISUALIZE = [list(HBM_VARIANTS.keys())[0]]
        
        has_mpm = ('maps_mpm' in locals() and maps_mpm is not None)
        if has_mpm:
            logger.info("Will include MPM Baseline in PUSH Analysis.")
        else:
            maps_mpm = {} # Empty dict fallback

        # metrics[model][task][run] = {metric: value}
        metrics_storage = collections.defaultdict(lambda: collections.defaultdict(dict))
        timing_storage_push = collections.defaultdict(lambda: collections.defaultdict(dict))
        # =========================================================
        # 2. RUN LOOP (Per Real Trial)
        # =========================================================
        for run_name in maps_real_after_sweep2.keys():
            logger.info(f"Processing Run: {run_name}")
            run_img_folder = os.path.join(cfg.OUTPUT_DIR, f"{run_name}_hm_images")
            os.makedirs(run_img_folder, exist_ok=True)

            # --- A. Measured maps: before push 1, after push 1, after push 2 ---
            real_start, _, _ = maps_real_at_start[run_name]
            real_mid, _, _   = maps_real_after_sweep1[run_name]
            real_end, x_edges, y_edges = maps_real_after_sweep2[run_name]
            
            sweep1 = sweeps_dict[run_name]['sweep01']
            sweep2 = sweeps_dict[run_name]['sweep02']

            eval_tasks = []

            # Trivial baseline NOPUSH: "prediction" = the start map, scaled to the volume of the target scan.
            # Entries: (model, task, target, prediction, setup time, push time, erosion time, erosion iterations)
            eval_tasks.append(("NOPUSH", "TaskA_Sweep1", real_mid, normalize_volume_to_target(real_start, real_mid), np.nan, np.nan, np.nan, 0))
            eval_tasks.append(("NOPUSH", "TaskB_Sweep2", real_end, normalize_volume_to_target(real_mid, real_end), np.nan, np.nan, np.nan, 0))
            eval_tasks.append(("NOPUSH", "TaskC_Chained", real_end, normalize_volume_to_target(real_start, real_end), np.nan, np.nan, np.nan, 0))

            # --- B. MPM predictions (already volume-matched through their start maps) ---
            if has_mpm:
                vizzed_mpm = False
                for task_key, target_map, pred_map in [
                    ("TaskA_Sweep1", real_mid, maps_push_mpm_taskA[run_name][0]),
                    ("TaskB_Sweep2", real_end, maps_push_mpm_taskB[run_name][0]),
                    ("TaskC_Chained", real_end, maps_push_mpm_taskC[run_name][0])]:

                    eval_tasks.append(("MPM", task_key, target_map, pred_map, np.nan, np.nan, np.nan, 0))

                    if cfg.minimal_viz and (not vizzed_mpm):
                        viz_mpm = True
                    elif cfg.minimal_viz and vizzed_mpm:
                        viz_mpm = False
                    else:
                        viz_mpm = True
                    if viz_mpm:
                        save_hmap_image(
                            pred_map, x_edges, y_edges, title=f"MPM {task_key} Prediction ({run_name})", 
                            three_d_view=True, max_height=cfg.viz_height_limit,
                            T_cloud_data=T_plane_data, # only used for shapes and sweeps viz, if used 
                            cmap=cfg.hmap_cmap, 
                            headless=cfg.headless, save_path=run_img_folder
                        )
                        vizzed_mpm = True

            # --- C. HBM predictions ---
            for variant, params in HBM_VARIANTS.items():
                logger.info(f"Running HBM Variant: {variant}")
                # Predict one or more pushes in sequence from `start_map`
                def run_sim(start_map, target_vol_map, sweep_data_list):
                    if not isinstance(sweep_data_list, list):
                        sweep_data_list = [sweep_data_list]

                    sim_obj = elevationMap2D()
                    sim_obj.setMetricConfig(map_width=cfg.map_width, resolution=cfg.resolution, label=variant)
                    
                    # Volume correction (once, at the start): scale the start map to the measured volume of the target scan
                    sim_obj.elevation_data = normalize_volume_to_target(start_map.copy(), target_vol_map)
                    
                    time_for_setup = 0.0
                    time_for_push = 0.0
                    time_for_erosion = 0.0
                    erode_steps = 0
                    for i, sweep_data in enumerate(sweep_data_list):
                        # Dump angle: angle of repose (PW) or 89° (PS, steep); erosion only for PSE/PWE
                        d_ang = AOR_DEG if params['dump_at_aor'] else 89
                        iters = cfg.erosion_max_iters if params['apply_erosion'] else 0
                        
                        res = elevationMap2D.sweepModelDisplacementAndErosion(
                            sim_obj, 
                            xy_sweep_start=sweep_data['PRE'][:2], 
                            xy_sweep_end=sweep_data['POST'][:2], 
                            sweep_width=cfg.sweep_width, 
                            angle_of_repose_deg=AOR_DEG,
                            dump_base_angle_deg=d_ang, 
                            N_iter_max=iters, 
                            relative_height_tolerance=cfg.erosion_rel_height_tol, 
                            only_erosion=False, 
                            sweep_shortening=False,
                            erosion_random_neighbor_order=True, 
                            erosion_stencil=EROSION_STENCIL,
                            random_seed=cfg.random_seed 
                        )
                        time_for_setup += res['time_for_setup']
                        time_for_push += res['time_for_push']
                        time_for_erosion += res['time_for_erosion']
                        erode_steps += res['N_iter_actual']
                        
                        # The result is the start of the next push (no further volume scaling)
                        sim_obj.elevation_data = res['elev_map_predicted'].elevation_data

                    return sim_obj.elevation_data, time_for_setup, time_for_push, time_for_erosion, erode_steps
                
                # 1-step predictions
                pred_mid = run_sim(real_start, real_mid, [sweep1])            
                pred_end_short = run_sim(real_mid, real_end, [sweep2])        

                # 2-step prediction: both pushes from the initial map, volume matched to the final scan
                pred_end_chained = run_sim(real_start, real_end, [sweep1, sweep2])

                eval_tasks.append((variant, "TaskA_Sweep1", real_mid, pred_mid[0], pred_mid[1], pred_mid[2], pred_mid[3], pred_mid[4]))
                eval_tasks.append((variant, "TaskB_Sweep2", real_end, pred_end_short[0], pred_end_short[1], pred_end_short[2], pred_end_short[3], pred_end_short[4])) 
                eval_tasks.append((variant, "TaskC_Chained", real_end, pred_end_chained[0], pred_end_chained[1], pred_end_chained[2], pred_end_chained[3], pred_end_chained[4])) 

                # --- D. Metrics (predictions are compared without further volume scaling) ---
                for model_name, task_name, target_map, pred_map, t_setup, t_push, t_erode, N_erode_steps in eval_tasks:
                    logger.info(f"Evaluating Model: {model_name}, Task: {task_name}")
                                    
                    res = compute_error_metrics(
                        mapA=target_map, mapB=pred_map, 
                        resolution=cfg.resolution,
                        requested=cfg.req_metrics,
                    )
                    metrics_storage[model_name][task_name][run_name] = res[2]

                    # Runtimes (HBM only; NaN for MPM and NOPUSH)
                    if not np.isnan(t_push):
                        timing_storage_push[model_name][task_name][run_name] = {
                            'Setup_Time_sec': t_setup,
                            'Push_Time_sec': t_push,
                            'Erosion_Time_sec': t_erode,
                            'Push_and_Erosion_sec': t_push + t_erode,
                            'Total_Time_sec': t_setup + t_push + t_erode,
                            'Erosion_Steps': N_erode_steps
                        }

                # --- Plots (selected variants) ---
                if variant in VARIANTS_TO_VISUALIZE:
                    logger.info(f"Generating Visualizations for Variant: {variant}")
                    # TaskA: Prediction
                    save_hmap_image(pred_mid[0], x_edges, y_edges, title=f"Sim Sweep1 ({variant})",
                                            three_d_view=True, max_height=cfg.viz_height_limit, min_height=0.0, 
                                            T_cloud_data=T_plane_data, # only used for shapes, if used
                                            cmap=cfg.hmap_cmap, 
                                            headless=cfg.headless, save_path=run_img_folder)
                    # TaskB: Prediction
                    save_hmap_image(pred_end_short[0], x_edges, y_edges, title=f"Sim Sweep2 ({variant})",
                                            three_d_view=True, max_height=cfg.viz_height_limit, min_height=0.0, 
                                            T_cloud_data=T_plane_data, # only used for shapes, if used 
                                            cmap=cfg.hmap_cmap, 
                                            headless=cfg.headless, save_path=run_img_folder)

                    # TaskC: Prediction (Chained)
                    save_hmap_image(pred_end_chained[0], x_edges, y_edges, title=f"Sim Chained End ({variant})", 
                                         three_d_view=True, max_height=cfg.viz_height_limit, min_height=0.0, 
                                        T_cloud_data=T_plane_data, # only used for shapes, if used
                                            cmap=cfg.hmap_cmap, 
                                         headless=cfg.headless, save_path=run_img_folder)

            # --- Plots of the measured maps ---
            save_hmap_image(real_start, x_edges, y_edges, title=f"Real Start (Run {run_name})", 
                                 three_d_view=True, max_height=cfg.viz_height_limit,
                                T_cloud_data=T_plane_data, # only used for shapes, if used 
                                cmap=cfg.hmap_cmap,
                                 headless=cfg.headless, save_path=run_img_folder)
            save_hmap_image(real_mid, x_edges, y_edges, title=f"Real Mid (Run {run_name})",
                                    three_d_view=True, max_height=cfg.viz_height_limit,
                                T_cloud_data=T_plane_data, # only used for shapes, if used 
                                cmap=cfg.hmap_cmap,
                                    headless=cfg.headless, save_path=run_img_folder)
            save_hmap_image(real_end, x_edges, y_edges, title=f"Real End (Run {run_name})", 
                                 three_d_view=True, max_height=cfg.viz_height_limit,
                                T_cloud_data=T_plane_data, # only used for shapes, if used 
                                cmap=cfg.hmap_cmap,
                                 headless=cfg.headless, save_path=run_img_folder)

        # =========================================================
        # 3. AGGREGATION & SUMMARY
        # =========================================================
        logger.info("Aggregating Results...")

        # One row per model, task and run
        long_data = []
        for model_name, task_dict in metrics_storage.items():
            for task_name, run_dict in task_dict.items():
                for run_id, score_dict in run_dict.items():
                    row = {
                        "Model": model_name,
                        "Task": task_name,
                        "Run": run_id,
                        "EMD": score_dict.get('emd', np.nan),
                        "IoU": score_dict.get('iou', np.nan)
                    }
                    long_data.append(row)
        
        df_long = pd.DataFrame(long_data)
        
        # Read by plotting/plot_push_iou.py
        long_outfile = os.path.join(cfg.OUTPUT_DIR, "SUMMARY_LONG_FORMAT.csv")
        df_long.to_csv(long_outfile, index=False)
        logger.info(f"Saved Long Format Summary: {long_outfile}")

        # Leaderboard: one row per model, mean and std per task (EMD columns stay empty)
        if not df_long.empty:
            grouped = df_long.groupby(['Model', 'Task'])[['EMD', 'IoU']].agg(['mean', 'std'])
            
            grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
            
            df_flat = grouped.reset_index()
            df_leaderboard = df_flat.pivot(index='Model', columns='Task')
            
            # Column names: <task>_<metric>_<stat>
            df_leaderboard.columns = [f"{t}_{m}" for m, t in df_leaderboard.columns]
            
            # Row order: MPM, then the HBM variants, then anything else (NOPUSH)
            desired_order = []
            if has_mpm: desired_order.append("MPM")
            desired_order.extend(list(HBM_VARIANTS.keys()))
            
            existing_models = df_leaderboard.index.tolist()
            final_row_order = [m for m in desired_order if m in existing_models]
            final_row_order.extend([m for m in existing_models if m not in final_row_order])
            
            df_leaderboard = df_leaderboard.reindex(final_row_order)

            # Column order: per task and metric, mean then std
            unique_tasks = sorted(df_long['Task'].unique())
            metrics_list = ['EMD', 'IoU']
            stats_list = ['mean', 'std']
            
            final_col_order = []
            for task in unique_tasks:
                for metric in metrics_list:
                    for stat in stats_list:
                        col_name = f"{task}_{metric}_{stat}"
                        if col_name in df_leaderboard.columns:
                            final_col_order.append(col_name)
            
            df_leaderboard = df_leaderboard[final_col_order]
            
            leaderboard_outfile = os.path.join(cfg.OUTPUT_DIR, "SUMMARY_LEADERBOARD.csv")
            df_leaderboard.to_csv(leaderboard_outfile)
            logger.info(f"Saved Leaderboard Summary: {leaderboard_outfile}")

        # --- Push runtimes (HBM) ---
        timing_rows = []
        for model, t_dict in timing_storage_push.items():
            for task, r_dict in t_dict.items():
                for run_id, times in r_dict.items():
                    timing_rows.append({
                        'Model': model, 'Task': task, 'Run': run_id,
                        **times
                    })
                    
        if timing_rows:
            df_push_times = pd.DataFrame(timing_rows)
            
            # Per run
            long_file = os.path.join(cfg.OUTPUT_DIR, "TIMINGS_PUSH_LONG.csv")
            df_push_times.to_csv(long_file, index=False)
            
            # Mean over runs
            df_time_summary = df_push_times.groupby(['Model', 'Task']).mean(numeric_only=True).reset_index()
            summary_file = os.path.join(cfg.OUTPUT_DIR, "TIMINGS_PUSH_SUMMARY_MEAN.csv")
            df_time_summary.to_csv(summary_file, index=False)
            
if __name__ == "__main__":
    main()
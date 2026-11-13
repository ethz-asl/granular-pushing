"""
Utilities for evaluating granular-material models against recorded experiments.

Used by scripts/model_eval.py (bucket-lift and push evaluation) and
scripts/run_planner.py (clearance planner).

Expected experiment layout (details in docs/DATA.md):

    <experiment>/
        run_01/                        one folder per repetition (run_01, run_02, ...)
            <scan folder>/             one folder per scan; its name contains a tag, e.g.
                                         PRELIFT / POSTLIFT        (bucket lift)
                                         PRESWEEP1 / POSTSWEEP2    (pushes)
                                         POSITION_<SHAPE>_<SIZE>   (pose of the initial shape,
                                                                   e.g. POSITION_PRISM_L, POSITION_CYLINDER_S)
                                         PRESWEEP1_POSE / POSTSWEEP1_POSE  (spatula pose at push start/end)
                point_cloud.ply    (point cloud, cropped to the work area, in the workplane frame)
                transforms.yaml    (transforms of 'workplane', 'panda_cone_tip', 'panda_spatula_tip')

All geometry is expressed in the workplane frame, or in the ground-plane frame fitted
to the scans (see "Coordinate frames").
"""
import os
import re
import time
from collections import OrderedDict, defaultdict

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import yaml
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation as RScipy
from shapely.geometry import Polygon, box
from sklearn.linear_model import LinearRegression, RANSACRegressor

from hbm_model import (
    elevationMap2D, generate_erosion_update_path,
    make_kernel_LUT,
    erodeHeightMapNUMBA_Safe,
    )

# Force TrueType fonts in PDF/PS output.
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# ==== Reading experimental data ====

# File names inside every scan folder
SCAN_POINT_CLOUD = "point_cloud.ply"
SCAN_TRANSFORMS = "transforms.yaml"

def wait_for_stability(file_path, file_stability_wait=0.5):
    """Wait until the file at `file_path` has stopped growing (used for scans written while the planner runs)."""

    if not file_path.exists(): return False 
    last_size = -1
    while True:
        try:
            current_size = file_path.stat().st_size
            if current_size == last_size and current_size > 0: return True
            last_size = current_size
            time.sleep(file_stability_wait)
        except OSError: return False

def load_pc_from_path_and_key(folder_path, key, extension=""):
    """Load the first point cloud in `folder_path` whose filename contains `key` (and ends with `extension`, if given)."""

    # Normalize extension
    if extension and not extension.startswith("."):
        extension = "." + extension

    for filename in os.listdir(folder_path):
        if key in filename:
            if extension == "" or filename.endswith(extension):
                selected_ply_file = os.path.join(folder_path, filename)
                pcd = o3d.io.read_point_cloud(selected_ply_file)
                return pcd
    return None

def find_pointclouds(experiment_root, identifier="", resolve_multiple='ILLEGAL'):
    """
    Load one point cloud per run from the scans tagged with `identifier` (e.g. 'POSTLIFT').

    `resolve_multiple` decides what happens if a run has several matching scans:
    'ILLEGAL' (raise), 'FIRST', 'LAST' or 'ALL'.
    Returns {'run_01': pcd, 'run_02': pcd, ...}.
    """
    clouds_dict = {}
    if resolve_multiple not in ['ILLEGAL', 'FIRST', 'LAST', 'ALL']:
        raise ValueError("resolve_multiple must be one of 'ILLEGAL', 'FIRST', 'LAST', 'ALL'")
    for run_folder in sorted(os.listdir(experiment_root)):
        run_path = os.path.join(experiment_root, run_folder)
        if not os.path.isdir(run_path) or not run_folder.startswith("run_"):
            continue
        sorted_by_scans = run_path
        # Scan folders whose name contains the identifier
        identifier_folders = []
        for scan_folder in sorted(os.listdir(sorted_by_scans)):
            if os.path.isdir(os.path.join(sorted_by_scans, scan_folder)) and identifier in scan_folder.upper():
                identifier_folders.append(scan_folder)

        if len(identifier_folders) == 0:
            print(f"  Warning: No folders with identifier '{identifier}' found in {sorted_by_scans}, skipping run.")
            continue

        if len(identifier_folders) > 1:
            if resolve_multiple == 'ILLEGAL':
                raise RuntimeError(f"Multiple folders with identifier '{identifier}' found in {sorted_by_scans}: {identifier_folders}")
            elif resolve_multiple == 'FIRST':
                identifier_folders = [identifier_folders[0]]
            elif resolve_multiple == 'LAST':
                identifier_folders = [identifier_folders[-1]]
            elif resolve_multiple == 'ALL':
                # Note: the merged cloud is built but not used below; the first matching scan is loaded.
                new_cloud = o3d.geometry.PointCloud()
                for folder in identifier_folders:
                    folder_path = os.path.join(sorted_by_scans, folder)
                    ply_files = [f for f in os.listdir(folder_path) if f == SCAN_POINT_CLOUD]
                    if len(ply_files) != 1:
                        raise RuntimeError(f"Expected exactly one {SCAN_POINT_CLOUD}, but found {len(ply_files)} in {folder_path}")
                    selected_ply_file = os.path.join(folder_path, ply_files[0])
                    pcd = o3d.io.read_point_cloud(selected_ply_file)
                    new_cloud += pcd
            else:
                print(f"  Warning: Unknown resolve_multiple='{resolve_multiple}', defaulting to FIRST.")
                identifier_folders = [identifier_folders[0]]

        # Load the (first) matching scan
        folder_path = os.path.join(sorted_by_scans, identifier_folders[0])
        ply_files = [f for f in os.listdir(folder_path) if f == SCAN_POINT_CLOUD]
        if len(ply_files) != 1:
            raise RuntimeError(f"Expected exactly one {SCAN_POINT_CLOUD}, but found {len(ply_files)} in {folder_path}")
        selected_ply_file = os.path.join(folder_path, ply_files[0])
        pcd = o3d.io.read_point_cloud(selected_ply_file)
        key_name = run_folder.split("_")[0] + "_" + run_folder.split("_")[1]  # e.g. run_01
        clouds_dict[key_name] = pcd
    return clouds_dict

def find_init_shapes(experiment_root, valid_init_shapes=None):
    """
    Read the pose of each initial shape (e.g. PRISM_L) from the POSITION_* scans of run_01.

    Returns {shape_type: {'position_in_workplane': xyz, 'quat_xyzw_in_workplane': quat}}.
    """
    init_shapes = {}

    # The POSITION_* scans are stored in run_01 only
    run_folder = None
    for folder in sorted(os.listdir(experiment_root)):
        if os.path.isdir(os.path.join(experiment_root, folder)) and folder.startswith("run_01"):
            if run_folder is not None:
                raise RuntimeError(f"Multiple run_01 folders found in {experiment_root}")
            run_folder = folder
    if run_folder is None:
        raise RuntimeError(f"No run_01 folder found in {experiment_root}")

    scans_folder = os.path.join(experiment_root, run_folder)
    position_folders = []
    for scan_folder in sorted(os.listdir(scans_folder)):
        if os.path.isdir(os.path.join(scans_folder, scan_folder)) and "POSITION" in scan_folder.split("_"):
            position_folders.append(scan_folder)

    # Shape type = the two name parts after POSITION, e.g. scan_2_POSITION_CYLINDER_L_<timestamp> -> CYLINDER_L
    existing_shapes = []
    for pos_folder in position_folders:
        parts = pos_folder.split("_")
        pos_index = parts.index("POSITION")
        shape_type = "_".join(parts[pos_index + 1:pos_index + 3])
        if shape_type not in valid_init_shapes.keys():
            print(f"  Warning: {shape_type} not in valid_init_shapes, skipping")
            raise RuntimeError(f"Invalid shape type {shape_type} in folder {pos_folder}")
        existing_shapes.append(shape_type)

    if len(existing_shapes) != len(set(existing_shapes)):
        raise RuntimeError(f"Duplicate POSITION folders found in {experiment_root}: {existing_shapes}")

    # Shape pose = pose of the robot's cone tip (placed at the shape's center), expressed in the workplane frame
    shape_dict = {}
    for pos_folder in position_folders:
        parts = pos_folder.split("_")
        pos_index = parts.index("POSITION")
        shape_type = "_".join(parts[pos_index + 1:pos_index + 3])
        folder_path = os.path.join(scans_folder, pos_folder)
        yaml_file = os.path.join(folder_path, SCAN_TRANSFORMS)
        if not os.path.isfile(yaml_file):
            raise RuntimeError(f"Missing {SCAN_TRANSFORMS} in {folder_path}")

        workplane_tf = load_yaml_transform(yaml_file, 'workplane')
        panda_tip_tf = load_yaml_transform(yaml_file, 'panda_cone_tip')
        workplane_ssr = yaml_transform_to_ssr(workplane_tf)
        panda_tip_ssr = yaml_transform_to_ssr(panda_tip_tf)
        # Both are given in the world frame: T_wp_tip = inv(T_world_wp) @ T_world_tip
        workplane_ssr_inv = np.linalg.inv(workplane_ssr)
        panda_tip_in_workplane_ssr = workplane_ssr_inv @ panda_tip_ssr
        xyz_in_workplane = panda_tip_in_workplane_ssr[:3, 3]
        quat_in_workplane = RScipy.from_matrix(panda_tip_in_workplane_ssr[:3, :3]).as_quat()  # [x, y, z, w]
        init_shapes[shape_type] = {
            'position_in_workplane': xyz_in_workplane,
            'quat_xyzw_in_workplane': quat_in_workplane
        }

    return init_shapes
    
def find_sweeps(experiment_root):
    """
    Read the spatula tip position at the start (PRE) and end (POST) of every push, from the *SWEEP<n>_POSE scans.

    Returns {'run_01': {'sweep01': {'PRE': xyz, 'POST': xyz}, ...}, ...}, in the workplane frame and sorted.
    """
    sweep_pattern = re.compile(r"(PRE|POST)SWEEP(\d+)_POSE")
    run_pattern = re.compile(r"(run_\d+)")

    temp_runs = {}

    for run_folder in os.listdir(experiment_root):
        run_path = os.path.join(experiment_root, run_folder)
        if not os.path.isdir(run_path) or not run_folder.startswith("run_"):
            continue

        # Short run name, e.g. run_01
        m = run_pattern.match(run_folder)
        if not m:
            continue
        run_short = m.group(1)

        sorted_by_scans = run_path

        if run_short not in temp_runs:
            temp_runs[run_short] = {}

        for scan_folder in os.listdir(sorted_by_scans):
            full_path = os.path.join(sorted_by_scans, scan_folder)
            if not os.path.isdir(full_path):
                continue

            match = sweep_pattern.search(scan_folder)
            if not match:
                continue

            phase, sweep_num = match.groups()
            sweep_key = f"sweep{int(sweep_num):02d}"

            if sweep_key not in temp_runs[run_short]:
                temp_runs[run_short][sweep_key] = {}

            tf_path = os.path.join(full_path, SCAN_TRANSFORMS)
            spatula_tip_tf = yaml_transform_to_ssr(load_yaml_transform(tf_path, 'panda_spatula_tip'))
            workplane_tf = yaml_transform_to_ssr(load_yaml_transform(tf_path, 'workplane'))

            # T_wp_tip = inv(T_world_wp) @ T_world_tip
            tf_spatula_tip_in_wp = np.linalg.inv(workplane_tf) @ spatula_tip_tf

            pos_spatula_tip = tf_spatula_tip_in_wp[:3, 3]
            temp_runs[run_short][sweep_key][phase] = pos_spatula_tip

    # Sort runs and sweeps numerically, PRE before POST
    runs = OrderedDict()
    for run in sorted(temp_runs.keys(), key=lambda x: int(x.split("_")[1])):
        sweeps_ordered = OrderedDict()
        for sweep in sorted(temp_runs[run].keys(), key=lambda x: int(x.replace("sweep", ""))):
            phases = OrderedDict()
            for phase in ["PRE", "POST"]:
                if phase in temp_runs[run][sweep]:
                    phases[phase] = temp_runs[run][sweep][phase]
            sweeps_ordered[sweep] = phases
        runs[run] = sweeps_ordered

    return runs


def load_yaml_transform(file_path, key):
    """Load the transform named `key` (e.g. 'workplane', 'panda_cone_tip') from a transforms.yaml file."""
    with open(file_path, 'r') as f:
        data = yaml.safe_load(f)
    if key not in data['transforms']:
        raise KeyError(f"'{key}' not found in {file_path}")
    return data['transforms'][key]

# ==== Coordinate frames ====

def yaml_transform_to_ssr(yaml_tf):
    """Convert a transform from transforms.yaml ('translation' + quaternion 'rotation') to a 4x4 homogeneous matrix."""
    # SciPy expects quaternions in [x, y, z, w] order
    q = yaml_tf['rotation']
    quat = [q['x'], q['y'], q['z'], q['w']]
    rot = RScipy.from_quat(quat)

    t = np.array([yaml_tf['translation']['x'],
                yaml_tf['translation']['y'],
                yaml_tf['translation']['z']])

    tf_matrix = np.eye(4)
    tf_matrix[:3, :3] = rot.as_matrix()
    tf_matrix[:3, 3] = t
    return tf_matrix

def fit_plane_with_reference(pcd, xy_ref=None,
                                distance_threshold=0.01, ransac_n=3,
                                num_iterations=1000, max_tilt_deg=10.0):
    """
    Fit the ground plane to a scan (RANSAC) and return the plane frame plus fit statistics.

    The plane frame has its z-axis along the plane normal and x/y aligned with the world x/y.
    Its origin is the plane point below `xy_ref` (default: the mean x/y of the cloud).
    Raises if the plane is tilted more than `max_tilt_deg` from the world XY plane.
    """
    print(f"Fitting plane with RANSAC: distance_threshold={distance_threshold}, ransac_n={ransac_n}, num_iterations={num_iterations}, max_tilt_deg={max_tilt_deg}")

    plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold,
                                                ransac_n=ransac_n,
                                                num_iterations=num_iterations)
    a, b, c, d = plane_model
    plane_normal = np.array([a, b, c])
    plane_normal /= np.linalg.norm(plane_normal)

    # Reject planes tilted too far from the world XY plane
    world_z = np.array([0, 0, 1])
    tilt_angle = np.arccos(np.clip(np.dot(world_z, plane_normal), -1.0, 1.0))
    tilt_deg = np.degrees(tilt_angle)
    if tilt_deg > max_tilt_deg:
        raise ValueError(f"Plane tilted {tilt_deg:.2f}° > {max_tilt_deg}° from XY plane")

    # Frame origin in x/y
    if xy_ref is not None:
        xy_ref = np.asarray(xy_ref)
        if xy_ref.shape != (2,):
            raise ValueError("xy_ref must be a 2-element array [x, y]")
        x_ref, y_ref = xy_ref
    else:
        points = np.asarray(pcd.points)
        x_ref, y_ref = np.mean(points[:, :2], axis=0)

    # Frame origin in z: on the plane
    z_ref = -(a * x_ref + b * y_ref + d) / c
    translation = np.array([x_ref, y_ref, z_ref])

    # Frame axes: z = plane normal, x = world x projected onto the plane
    z_axis = plane_normal
    x_axis = np.array([1, 0, 0]) - np.dot([1, 0, 0], z_axis) * z_axis
    if np.linalg.norm(x_axis) < 1e-6:
        x_axis = np.array([0, 1, 0]) - np.dot([0, 1, 0], z_axis) * z_axis
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    R_mat = np.column_stack((x_axis, y_axis, z_axis))
    quaternion = RScipy.from_matrix(R_mat).as_quat()

    # Fit quality: signed point-to-plane distances of the inliers
    inlier_cloud = pcd.select_by_index(inliers)
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    points = np.asarray(inlier_cloud.points)
    errors = a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d
    abs_errors = np.abs(errors)
    mean_error = np.mean(errors)
    stddev_error = np.std(errors, ddof=1)  # sample std
    inlier_ratio = len(inliers) / len(pcd.points)
    
    return {
        'translation': translation,
        'quaternion': quaternion,
        'plane_normal': plane_normal,
        'inlier_cloud': inlier_cloud,
        'outlier_cloud': outlier_cloud,
        'error_distances': errors,
        'abs_error_distances': abs_errors,
        'mean_error_distance': mean_error,
        'stddev_error_distance': stddev_error}

def transform_pointcloud(pcd, T_out_in, height_cutoff=0.0):
    """
    Transform an Open3D point cloud with the 4x4 matrix `T_out_in` ('in' frame -> 'out' frame).

    If `height_cutoff` is not None, points below this z (in the 'out' frame) are removed.
    """

    if not isinstance(pcd, o3d.geometry.PointCloud):
        raise ValueError("pcd must be an Open3D PointCloud")
    if T_out_in.shape != (4, 4):
        raise ValueError("T_out_in must be a 4x4 transformation matrix")

    points = np.asarray(pcd.points)
    points_h = np.hstack((points, np.ones((points.shape[0], 1))))  # Nx4
    points_transformed_h = (T_out_in @ points_h.T).T  # Nx4

    pcd_transformed = o3d.geometry.PointCloud()
    pcd_transformed.points = o3d.utility.Vector3dVector(points_transformed_h[:, :3])

    if height_cutoff is not None:
        points_array = np.asarray(pcd_transformed.points)
        height_mask = points_array[:, 2] >= height_cutoff
        pcd_transformed = pcd_transformed.select_by_index(np.where(height_mask)[0])

    return pcd_transformed

def transform_sweep(sweep, T_cloud_data):
    """
    Transform a push's start and end point with the 4x4 matrix `T_cloud_data`; returns their x/y.

    `sweep` is either {'PRE': xyz, 'POST': xyz} or [start_xyz, end_xyz]; the result has the same form.
    """

    if isinstance(sweep, (list, tuple)):
        sweep_start = sweep[0]
        sweep_end = sweep[1]
    elif isinstance(sweep, dict):
        sweep_start = sweep["PRE"]
        sweep_end = sweep["POST"]
    else:
        raise ValueError("sweep must be a dict with 'PRE' and 'POST' keys or a list/tuple of two elements.")

    sweep_start_h = np.hstack((sweep_start, 1))
    sweep_start = (T_cloud_data @ sweep_start_h)[:2]
    sweep_end_h = np.hstack((sweep_end, 1))
    sweep_end = (T_cloud_data @ sweep_end_h)[:2]

    if isinstance(sweep, (list, tuple)):
        return [sweep_start, sweep_end]
    elif isinstance(sweep, dict):
        return {"PRE": sweep_start, "POST": sweep_end}

# ==== Angle of Repose (AOR) ====

def compute_cone_parameters(angle_of_repose_deg, volume):
    """Return height and base radius (m) of the cone with the given angle of repose (deg) and volume (m³)."""
    theta_rad = np.radians(angle_of_repose_deg)
    if angle_of_repose_deg <= 0 or angle_of_repose_deg >= 60:
        raise ValueError("Angle of repose must be in (0, 60) degrees for valid cone geometry.")
    tan_theta = np.tan(theta_rad)
    r = (3 * volume / (np.pi * tan_theta)) ** (1/3)
    h = r * tan_theta
    return h, r

def fit_cone_opening_angle(pcd, center_xy=None, z_min=0.0, use_ransac=False):
    """
    Fit a cone to a pile scan and return its opening angle and the angle of repose (both in degrees).

    The apex is taken near the highest points; the slope r/h is fitted by linear regression
    (optionally RANSAC). Also returns the pile points, a synthetic cone cloud for visualization,
    and the x/y of the apex.
    """
    points = np.asarray(pcd.points)

    # Keep only points above z_min (the pile)
    cone_mask = points[:, 2] > z_min
    cone_points_array = points[cone_mask]
    if len(cone_points_array) == 0:
        raise ValueError("No points left after applying z_min filter.")
    
    # Apex height: 99th percentile of z (more robust than the maximum)
    z_tip = np.percentile(cone_points_array[:, 2], 99)
    tip_mask = np.abs(cone_points_array[:, 2] - z_tip) < 1e-3
    if np.sum(tip_mask) == 0:
        raise ValueError("No points found near the cone tip height.")
    if center_xy is not None:
        print("  Using provided center_xy for cone base center.")
        x0, y0 = center_xy
    else:
        print("Estimating cone base center from points.")
        x0 = np.mean(cone_points_array[tip_mask, 0])
        y0 = np.mean(cone_points_array[tip_mask, 1])

    # Depth below the apex and radial distance from the apex axis
    h = z_tip - cone_points_array[:, 2]
    r = np.sqrt((cone_points_array[:, 0] - x0) ** 2 +
                    (cone_points_array[:, 1] - y0) ** 2)

    valid_mask = h > 1e-6
    h = h[valid_mask].reshape(-1, 1)
    r = r[valid_mask]

    if len(h) < 2:
        raise ValueError("Not enough valid points to fit cone.")

    # Fit r = slope * h (slope = tan of the half opening angle)
    if use_ransac:
        base_model = LinearRegression(fit_intercept=False)
        model = RANSACRegressor(base_model, min_samples=0.5, residual_threshold=np.median(r) * 0.1)
        model.fit(h, r)
        slope = model.estimator_.coef_[0]
    else:
        model = LinearRegression(fit_intercept=False).fit(h, r)
        slope = model.coef_[0]

    theta_cone = np.arctan(slope)

    cone_cloud = o3d.geometry.PointCloud()
    cone_cloud.points = o3d.utility.Vector3dVector(cone_points_array)

    # Synthetic cone with the fitted slope, for visualization only
    artificial_points = []
    n_z = 50
    n_ang = 100
    z_vals = np.linspace(cone_points_array[:, 2].min(), z_tip, n_z)

    for z in z_vals:
        h_val = z_tip - z
        r_val = h_val * slope
        for ang in np.linspace(0, 2 * np.pi, n_ang, endpoint=False):
            x = x0 + r_val * np.cos(ang)
            y = y0 + r_val * np.sin(ang)
            artificial_points.append([x, y, z])

    artificial_cone_cloud = o3d.geometry.PointCloud()
    artificial_cone_cloud.points = o3d.utility.Vector3dVector(np.array(artificial_points))

    # Angle of repose = 90° minus the half opening angle
    angle_of_repose_deg_center = 180 - 90 - np.degrees(theta_cone)

    return np.degrees(theta_cone), angle_of_repose_deg_center, cone_cloud, artificial_cone_cloud, np.array([x0, y0])

def estimate_angle_of_repose(clouds, init_shapes=None, T_cloud_data=None):
    """
    Estimate the angle of repose from one pile scan per run (e.g. the POSTLIFT scans of a bucket lift).

    Fits a cone to each scan; returns mean and sample std over the runs (degrees) and the pile point clouds.
    Only experiments with a single initial shape are supported.
    """

    if len(init_shapes) != 1:
        raise RuntimeError("AoR estimation only supported for single shape experiments.")
    shape_type = list(init_shapes.keys())[0]

    shape_position = init_shapes[shape_type]['position_in_workplane']
    if T_cloud_data is not None:
        shape_position_h = np.hstack((shape_position, 1))
        shape_position = (T_cloud_data @ shape_position_h)[:3]

    angles_per_run = []
    cone_cloud_dict = {}
    for run_name in sorted(clouds.keys()):
        pcd = clouds[run_name]
        cone_angle, aor, cone_cloud, artificial_cone, center_returned = fit_cone_opening_angle(
            pcd, z_min=0.01, use_ransac=True)
        angles_per_run.append(aor)
        cone_cloud_dict[run_name] = cone_cloud

    mean_angle = np.mean(angles_per_run)
    std_angle = np.std(angles_per_run, ddof=1)  # sample std

    return mean_angle, std_angle, cone_cloud_dict


# ==== Height mapping ====

def compute_height_map(pcd, width_x, width_y, resolution=0.001,
                        height_cutoff=-np.inf, aggregation_method="max"):
    """
    Rasterize a point cloud into a height map centered at the origin.

    Each cell takes the max/min/mean/median z of its points (`aggregation_method`); empty cells are 0.
    Returns (heights[rows, cols], x_edges, y_edges), or (None, None, None) if no points remain.
    """
    points = np.asarray(pcd.points)
    if points.shape[0] == 0:
        return None, None, None

    points = points[points[:, 2] >= height_cutoff]
    if points.shape[0] == 0:
        return None, None, None

    # Grid boundaries centered at origin
    x_min, y_min = -width_x / 2, -width_y / 2
    x_max, y_max = width_x / 2, width_y / 2

    # Cell index of every point (points outside the map are clipped to the border cells)
    ix = ((points[:, 0] - x_min) / resolution).astype(int)
    iy = ((points[:, 1] - y_min) / resolution).astype(int)

    cols = int(np.ceil(width_x / resolution))
    rows = int(np.ceil(width_y / resolution))
    ix = np.clip(ix, 0, cols - 1)
    iy = np.clip(iy, 0, rows - 1)

    cell_points = defaultdict(list)
    for x_idx, y_idx, z_val in zip(ix, iy, points[:, 2]):
        cell_points[(y_idx, x_idx)].append(z_val)

    elevation_data = np.full((rows, cols), np.nan, dtype=np.float32)
    for (y_idx, x_idx), values in cell_points.items():
        if aggregation_method == "max":
            elevation_data[y_idx, x_idx] = np.max(values)
        elif aggregation_method == "min":
            elevation_data[y_idx, x_idx] = np.min(values)
        elif aggregation_method == "mean":
            elevation_data[y_idx, x_idx] = np.mean(values)
        elif aggregation_method == "median":
            elevation_data[y_idx, x_idx] = np.median(values)
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation_method}")

    # Empty cells -> height 0
    elevation_data = np.nan_to_num(elevation_data, nan=0.0)

    x_edges = np.linspace(x_min, x_max, cols + 1)
    y_edges = np.linspace(y_min, y_max, rows + 1)

    return elevation_data, x_edges, y_edges

def compute_height_maps(cloud_dict, width_x, width_y, resolution=0.001,
                        height_cutoff=-np.inf, aggregation_method="mean", tag_postfix=""):
    """Apply compute_height_map to each cloud in `cloud_dict`; returns {run: (heights, x_edges, y_edges)}."""
    height_maps = {}
    for run_name in sorted(cloud_dict.keys()):
            pcd = cloud_dict[run_name]
            height_map, x_edges, y_edges = compute_height_map(
                pcd, width_x, width_y, resolution,
                height_cutoff, aggregation_method)
            height_maps[run_name] = (height_map, x_edges, y_edges)
    return height_maps

def normalize_volume_to_target(source_map, target_map):
    """
    Scale all heights of `source_map` so its total volume equals that of `target_map`.

    Compensates for material the depth sensor misses (e.g. thin layers) before maps are compared.
    """
    if source_map is None or target_map is None: return None
    vol_source = np.nansum(source_map)
    vol_target = np.nansum(target_map)
    if vol_source == 0: return source_map 
    return source_map * (vol_target / vol_source)

# ==== Model Initialization ====

def init_sim_map(valid_init_shapes=None, init_shapes=None, resolution=None, map_width=None, angle_of_repose=None, T_cloud_data=None, reference_volume=None, erosion_max_iters=None, erosion_stencil=None, random_seed=None):
    """
    Build the starting height maps of a bucket lift from the known shape and pose of the bucket.

    Returns:
        'map_init_from_shapes': the filled bucket (cylinder or prism) as a height map
        'eroded_map':      the filled bucket after HBM erosion (bucket lifted, material settled)
        plus the number of erosion iterations and their runtime.

    If `reference_volume` is given, the maps are scaled to that volume (see normalize_volume_to_target).
    Shapes are looked up in `valid_init_shapes`; positions are moved into the map frame with `T_cloud_data`.
    """

    # Filled bucket
    init_map = elevationMap2D()
    init_map.setMetricConfig(
        map_width=map_width,
        resolution=resolution, label='init_map')

    shapes = init_shapes.keys()

    # Supported setups: one shape (CYLINDER_S, CYLINDER_L, PRISM_L) or two cylinders
    if len(shapes) == 1:
        if "CYLINDER_S" in shapes:
            shape_info = valid_init_shapes["CYLINDER_S"]
            radius = shape_info['diameter'] / 2
            height = shape_info['height']
            shape_position = init_shapes["CYLINDER_S"]['position_in_workplane']
            if T_cloud_data is not None:
                shape_position_h = np.hstack((shape_position, 1))
                shape_position = (T_cloud_data @ shape_position_h)[:3]
            init_map.overrideWithCircleElevation(
                const_elevation=height,
                xy_centers=[np.array([shape_position[0], shape_position[1]])],
                radius=radius)

        elif "CYLINDER_L" in shapes:
            shape_info = valid_init_shapes["CYLINDER_L"]
            radius = shape_info['diameter'] / 2
            height = shape_info['height']
            shape_position = init_shapes["CYLINDER_L"]['position_in_workplane']
            if T_cloud_data is not None:
                shape_position_h = np.hstack((shape_position, 1))
                shape_position = (T_cloud_data @ shape_position_h)[:3]
            init_map.overrideWithCircleElevation(
                const_elevation=height,
                xy_centers=[np.array([shape_position[0], shape_position[1]])],
                radius=radius)
        elif "PRISM_L" in shapes:
            shape_info = valid_init_shapes["PRISM_L"]
            side_length = shape_info['side_length']
            height = shape_info['height']
            shape_position = init_shapes["PRISM_L"]['position_in_workplane']
            if T_cloud_data is not None:
                shape_position_h = np.hstack((shape_position, 1))
                shape_position = (T_cloud_data @ shape_position_h)[:3]
            bottom_left = np.array([shape_position[0] - side_length / 2, shape_position[1] - side_length / 2])
            top_right = np.array([shape_position[0] + side_length / 2, shape_position[1] + side_length / 2])
            init_map.overrideWithRectangleElevation(
                const_elevation=height,
                xy_bottomleft=bottom_left,
                xy_topright=top_right)
    elif len(shapes) == 2:
        if "CYLINDER_S1" in shapes and "CYLINDER_S2" in shapes:
            print("Simulating with two small cylinders")
            shape_s1_info = valid_init_shapes["CYLINDER_S1"]
            radius_s1 = shape_s1_info['diameter'] / 2
            height_s1 = shape_s1_info['height']
            shape_s2_info = valid_init_shapes["CYLINDER_S2"]
            radius_s2 = shape_s2_info['diameter'] / 2
            height_s2 = shape_s2_info['height']
            shape_position_s1 = init_shapes["CYLINDER_S1"]['position_in_workplane']
            shape_position_s2 = init_shapes["CYLINDER_S2"]['position_in_workplane']
            if T_cloud_data is not None:   
                shape_position_s1_h = np.hstack((shape_position_s1, 1))
                shape_position_s1 = (T_cloud_data @ shape_position_s1_h)[:3]
                shape_position_s2_h = np.hstack((shape_position_s2, 1))
                shape_position_s2 = (T_cloud_data @ shape_position_s2_h)[:3]
            init_map.overrideWithCircleElevation(
                const_elevation=[height_s1, height_s2],
                xy_centers=[np.array([shape_position_s1[0], shape_position_s1[1]]),
                            np.array([shape_position_s2[0], shape_position_s2[1]])],
                radius=radius_s1)
        elif "CYLINDER_L" in shapes and "CYLINDER_S" in shapes:
            print("Simulating with large and small cylinder")
            shape_info_L = valid_init_shapes["CYLINDER_L"]
            radius_L = shape_info_L['diameter'] / 2
            height_L = shape_info_L['height']
            shape_info_S = valid_init_shapes["CYLINDER_S"]
            radius_S = shape_info_S['diameter'] / 2
            height_S = shape_info_S['height']
            shape_position_L = init_shapes["CYLINDER_L"]['position_in_workplane']
            shape_position_S = init_shapes["CYLINDER_S"]['position_in_workplane']
            if T_cloud_data is not None:
                shape_position_L_h = np.hstack((shape_position_L, 1))
                shape_position_L = (T_cloud_data @ shape_position_L_h)[:3]
                shape_position_S_h = np.hstack((shape_position_S, 1))
                shape_position_S = (T_cloud_data @ shape_position_S_h)[:3]
            init_map.overrideWithCircleElevation(
                const_elevation=[height_L, height_S],
                xy_centers=[np.array([shape_position_L[0], shape_position_L[1]]),
                            np.array([shape_position_S[0], shape_position_S[1]])],
                radius=[radius_L, radius_S])

        else:
            raise RuntimeError("Invalid combination of two shapes for simulation.")
    else:
        raise RuntimeError("Simulation config currently only supported for 1 or 2 shapes.")        

    if reference_volume is not None:
        init_volume = init_map.elevation_data.sum() * (resolution ** 2)
        scale_factor = (reference_volume / init_volume)
        init_map.elevation_data *= scale_factor

    # HBM erosion of the filled bucket: the material settles to the angle of repose
    rs, cs = init_map.elevation_data.shape
    random_lookup_paths = generate_erosion_update_path(rs, cs, erosion_max_iters, random=False, random_seed=random_seed)

    erosion_height_unit = init_map.computeHeightUnit(angle_of_repose_deg=angle_of_repose, resolution=resolution)
    
    neighbors, max_neighbors = make_kernel_LUT(*init_map.elevation_data.shape, erosion_stencil)
    time_before_erosion = time.time()
    [height_map_stack, N_iter_actual] = erodeHeightMapNUMBA_Safe(
        init_map.elevation_data,
        paths=random_lookup_paths,
        neighbors=neighbors, max_neighbors=max_neighbors,
        height_unit = erosion_height_unit, 
        relative_height_tolerance=0.95,
        N_iter_max=erosion_max_iters,
        random_neighbor_order=True,
        random_seed=random_seed)
    time_for_erosion = time.time() - time_before_erosion
    eroded_map_arr = height_map_stack[:, :, N_iter_actual]
    print(f"  Eroded initial map for {N_iter_actual} iters in {time_for_erosion:.2f} sec")

    eroded_map = elevationMap2D()
    eroded_map.setMetricConfig(
        map_width=map_width,
        resolution=resolution,
        label='eroded_map')
    eroded_map.elevation_data = eroded_map_arr

    return {
        'map_init_from_shapes': init_map.elevation_data,
        'eroded_map': eroded_map.elevation_data,
        'N_iter_erosion': N_iter_actual,
        'time_for_erosion': time_for_erosion,}

# ==== Visualization ====

def fig_to_array(fig):
    """Render a Matplotlib figure and return it as an RGB image array."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    # RGBA -> RGB
    return np.array(renderer.buffer_rgba())[:, :, :3]

def visualize_height_map(
    height_map, x_edges, y_edges,
    title=None, three_d_view=False, min_height=0.0, max_height=1.0,
    valid_shapes=None, shapes_to_viz=None, sweep_to_viz=None, sweep_width=None, 
    T_cloud_data=None, headless=False, view_angles=[(45, 45)], cmap="viridis",
    save_path=None, black_bg=False, sweep_color="yellow",
    three_d_bars=False, clean_frame=False, exact_save_path=None, fast_render=False,
    spatula_pose=None,  # {"center": [x, y], "angle": rad, "width": W, "thickness": T}
    save_pdf=True  # also save a PDF next to each PNG
):
    """
    Plot a height map in 2D (optionally also in 3D) with optional overlays.

    Overlays: initial shapes (`shapes_to_viz`), the area swept by a push (`sweep_to_viz`, `sweep_width`)
    and the spatula (`spatula_pose`). `three_d_view` adds a surface plot, `three_d_bars` a bar plot.
    Saves to `exact_save_path` (one file) or into the folder `save_path` (2D/3D, PNG, and PDF if `save_pdf`).
    Returns (fig_2d, fig_3d or None, list of 3D view images).
    """
    # --- 2D plot ---
    if clean_frame:
        fig, ax = plt.subplots(figsize=(8, 8), facecolor='black' if black_bg else 'white')
    else:
        fig, ax = plt.subplots(figsize=(9, 5))

    im = ax.imshow(
        height_map,
        extent=(x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]),
        origin="lower",
        cmap=cmap,
        vmin=min_height,
        vmax=max_height,
        interpolation='nearest' if clean_frame else None
    )
    
    if not clean_frame:
        ax.set_title(title or "Height Map")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        cbar = fig.colorbar(im, ax=ax, fraction=0.2, pad=0.05, aspect=10)
        cbar.set_label("Height (m)", fontsize=26, labelpad=10)
        cbar.ax.tick_params(labelsize=24)
    else:
        ax.axis('off')

    x_min, x_max = x_edges[0], x_edges[-1]
    y_min, y_max = y_edges[0], y_edges[-1]

    # --- Spatula (2D) ---
    if spatula_pose is not None:
        cx, cy = spatula_pose["center"]
        angle = spatula_pose["angle"]
        W = spatula_pose["width"]
        T = spatula_pose["thickness"]

        corners_local = np.array([
            [-T/2, -W/2],
            [ T/2, -W/2],
            [ T/2,  W/2],
            [-T/2,  W/2]
        ])
        R = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle),  np.cos(angle)]
        ])
        corners_global = (corners_local @ R.T) + np.array([cx, cy])

        patch = patches.Polygon(corners_global, edgecolor="orange", facecolor="orange", alpha=0.8, linewidth=2, zorder=10)
        ax.add_patch(patch)

    # --- Initial shapes ---
    if shapes_to_viz is not None:
        for shape_name, shape_pose in shapes_to_viz.items():
            shape_info = valid_shapes[shape_name]
            shape_position = shape_pose["position_in_workplane"]

            if T_cloud_data is not None:
                shape_position_h = np.hstack((shape_position, 1))
                shape_position = (T_cloud_data @ shape_position_h)[:3]

            if shape_info["type"] == "cylinder":
                patch = patches.Circle((shape_position[0], shape_position[1]), radius=shape_info["diameter"] / 2, edgecolor="red", facecolor="none", linewidth=2)
                ax.add_patch(patch)

            elif shape_info["type"] == "prism":
                p_angle = np.degrees(RScipy.from_quat(shape_pose["quat_xyzw_in_workplane"]).as_euler("zxy")[0])
                patch = patches.Rectangle(
                    (shape_position[0] - shape_info["side_length"] / 2, shape_position[1] - shape_info["side_length"] / 2),
                    width=shape_info["side_length"], height=shape_info["side_length"], angle=p_angle, edgecolor="red", facecolor="none", linewidth=2
                )
                ax.add_patch(patch)


    # --- Swept area of the push, clipped to the map ---
    if sweep_to_viz is not None:
        start = sweep_to_viz["PRE"]
        end = sweep_to_viz["POST"]
        sweep_vec = end - start
        sweep_length = np.linalg.norm(sweep_vec)

        if sweep_length > 1e-6:
            angle = np.arctan2(sweep_vec[1], sweep_vec[0])
            center = (start + end) / 2
            dx, dy = sweep_length / 2, sweep_width / 2
            corners = np.array([[-dx, -dy], [ dx, -dy], [ dx,  dy], [-dx,  dy]])
            R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle),  np.cos(angle)]])
            corners_world = (corners @ R.T) + center

            poly = Polygon(corners_world)
            map_bounds = box(x_min, y_min, x_max, y_max)
            clipped = poly.intersection(map_bounds)

            if not clipped.is_empty:
                if clipped.geom_type == 'Polygon':
                    patch = patches.Polygon(np.array(clipped.exterior.coords), edgecolor="black", facecolor=sweep_color, alpha=0.3, linewidth=2)
                    ax.add_patch(patch)
                elif clipped.geom_type == 'MultiPolygon':
                    for p in clipped.geoms:
                        patch = patches.Polygon(np.array(p.exterior.coords), edgecolor="black", facecolor=sweep_color, alpha=0.1, linewidth=2)
                        ax.add_patch(patch)
                ax.autoscale()

    # --- Optional 3D plot ---
    fig_3d = None
    if three_d_view or three_d_bars:
        if clean_frame:
            fig_3d = plt.figure(figsize=(8, 8), facecolor='black' if black_bg else 'white')
        else:
            fig_3d = plt.figure(figsize=(8, 6))
            
        ax_3d = fig_3d.add_subplot(111, projection="3d")
        X, Y = np.meshgrid((x_edges[:-1] + x_edges[1:]) / 2, (y_edges[:-1] + y_edges[1:]) / 2)
        
        if three_d_view:
            ax_3d.plot_surface(X, Y, height_map, cmap=cmap, vmin=min_height, vmax=max_height, alpha=1.0)

        elif three_d_bars:
            dx, dy = x_edges[1] - x_edges[0], y_edges[1] - y_edges[0]
            x_flat, y_flat, dz_flat = X.flatten(), Y.flatten(), height_map.flatten()
            dz_bars = dz_flat.copy()
            min_thickness = 0.0005 
            dz_bars[dz_bars < min_thickness] = min_thickness
            
            norm = plt.Normalize(vmin=min_height, vmax=max_height)
            colors = plt.get_cmap(cmap)(norm(dz_bars))

            ax_3d.bar3d(x_flat - dx/2, y_flat - dy/2, np.zeros_like(x_flat), dx, dy, dz_bars, color=colors, edgecolor='none', shade=True)
            ax_3d.view_init(elev=view_angles[0][0], azim=view_angles[0][1])
            ax_3d.set_xlim(x_min, x_max)
            ax_3d.set_ylim(y_min, y_max)
            ax_3d.set_zlim(0, max_height)
            ax_3d.axis('off')        

        # --- Spatula (3D) ---
        if spatula_pose is not None:
            cx, cy = spatula_pose["center"]
            angle = spatula_pose["angle"]
            W = spatula_pose["width"]
            T = spatula_pose["thickness"]
            H_spatula = max_height if max_height > 0.0 else 0.15

            corners_local = np.array([[-T/2, -W/2], [ T/2, -W/2], [ T/2,  W/2], [-T/2,  W/2]])
            R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle),  np.cos(angle)]])
            corners_2d = (corners_local @ R.T) + np.array([cx, cy])

            verts_bottom = [(x, y, 0) for x, y in corners_2d]
            verts_top = [(x, y, H_spatula) for x, y in corners_2d]

            faces = []
            for i in range(4):
                faces.append([verts_bottom[i], verts_bottom[(i+1)%4], verts_top[(i+1)%4], verts_top[i]])
            faces.append(verts_bottom)
            faces.append(verts_top)

            ax_3d.add_collection3d(Poly3DCollection(faces, facecolors='orange', edgecolors='black', alpha=0.8, linewidths=1))

        if not clean_frame:
            ax_3d.set_xlabel("X (m)")
            ax_3d.set_ylabel("Y (m)")
            ax_3d.set_zlabel("Height (m)")
            ax_3d.set_title(f"3D View: {title}")
        else:
            ax_3d.axis('off')
            
        ax_3d.set_box_aspect((1, 1, 1.2))
        ax_3d.set_zlim(0, x_edges[-1] - x_edges[0]/2.0)
        ax_3d.grid(False)

        if black_bg:
            ax_3d.set_facecolor((0, 0, 0))
            ax_3d.xaxis.set_pane_color((0, 0, 0, 1))
            ax_3d.yaxis.set_pane_color((0, 0, 0, 1))
            ax_3d.zaxis.set_pane_color((0, 0, 0, 1))

        fig_3d.tight_layout()
        
        td_view_imgs = []
        for elev, azimuth in view_angles:
            ax_3d.view_init(elev=elev, azim=azimuth)
            plt.draw()
            try:
                img = fig_to_array(fig_3d)
                td_view_imgs.append(img)
            except NameError:
                pass
    else:
        td_view_imgs = []

    if not headless:
        plt.show()

    # --- Saving ---
    if exact_save_path is not None:
        os.makedirs(os.path.dirname(exact_save_path), exist_ok=True)
        dpi_val = 80 if fast_render else 240
        target_fig = fig_3d if (three_d_view or three_d_bars) else fig
        
        target_fig.savefig(exact_save_path, dpi=dpi_val, bbox_inches='tight', pad_inches=0, facecolor=target_fig.get_facecolor())
        
        if not fast_render and save_pdf:
            pdf_path = os.path.splitext(exact_save_path)[0] + ".pdf"
            try:
                target_fig.savefig(pdf_path, format='pdf', bbox_inches='tight', pad_inches=0, facecolor=target_fig.get_facecolor())
            except Exception as e:
                print(f"  [Warning] PDF generation failed for {pdf_path}: {e}")
                
    elif save_path is not None:
        if title is None:
            print("Warning: No title provided, using 'height_map.png' as filename, might overwrite existing files.")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        fname_2d = os.path.join(save_path, f"{title.replace(' ', '_')}_2D.png")
        fname_3d = os.path.join(save_path, f"{title.replace(' ', '_')}_3D.png")
        fig.savefig(fname_2d, facecolor=fig.get_facecolor())
        if fig_3d is not None:
            fig_3d.savefig(fname_3d, facecolor=fig_3d.get_facecolor())
            
        if save_pdf:
            fname_2d_pdf = os.path.join(save_path, f"{title.replace(' ', '_')}_2D.pdf")
            fname_3d_pdf = os.path.join(save_path, f"{title.replace(' ', '_')}_3D.pdf")
            fig.savefig(fname_2d_pdf, facecolor=fig.get_facecolor())
            if fig_3d is not None:
                fig_3d.savefig(fname_3d_pdf, facecolor=fig_3d.get_facecolor())

    if headless:
        plt.close(fig)
        if fig_3d is not None:
            plt.close(fig_3d)

    return fig, fig_3d, td_view_imgs
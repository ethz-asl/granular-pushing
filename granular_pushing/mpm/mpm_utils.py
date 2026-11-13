"""
Helpers for the MPM simulation (mpm_sim.py): particle emission from a height map,
the spatula, the push trajectory, the run summary and depth snapshots.
"""
import math
import numpy as np
from PIL import Image
from typing import List, Tuple, Dict
import logging

import newton
import warp as wp

def emit_particles(
    builder: newton.ModelBuilder,
    logger: logging.Logger,
    init_heightmap_path: str = None,
    density: float = None,
    particles_per_voxel: int = None,
    voxel_size: float = None,
    jitter_scale: float = None,) -> None:
    """
    Fill the height map in `init_heightmap_path` (.npz with 'heightmap', 'x_edges', 'y_edges') with particles.

    `particles_per_voxel` particles per MPM voxel, stacked in layers up to the local height and
    jittered by `jitter_scale` x particle spacing (fixed seed 42). Returns {'particle_radius': r}.
    """
    logger.info("Emitting particles into the simulation.")
    
    logger.info(f"Initializing particles from heightmap: {init_heightmap_path}")
    
    # A. Load the height map (x_edges / y_edges in world coordinates)
    data = np.load(init_heightmap_path)
    height_map = data['heightmap']
    x_edges = data['x_edges']
    y_edges = data['y_edges']
    
    # B. Square map: derive its size
    assert height_map.shape[0] == height_map.shape[1], "Heightmap must be square (NxN)."
    
    n_cells = height_map.shape[0]
    
    start_x = x_edges[0]
    start_y = y_edges[0]
    map_width = x_edges[-1] - x_edges[0]
    
    logger.info(f"Map Origin: ({start_x:.2f}, {start_y:.2f}) | Width: {map_width:.2f}m | Grid: {n_cells}x{n_cells}")

    # C. Particle size and mass
    voxel_vol = voxel_size ** 3
    vol_per_particle = voxel_vol / particles_per_voxel

    # particle_spacing = side length of the cube each particle occupies
    particle_spacing = vol_per_particle ** (1.0/3.0)
    # The solver assumes V = (2*r)^3, so r must be half the side length
    particle_radius = particle_spacing / 2.0

    mass_per_particle = vol_per_particle * density

    # D. Particle positions
    points = []
    rng = np.random.default_rng(42)

    # Grid in world coordinates over the whole map
    x_coords = np.arange(start_x, start_x + map_width, particle_spacing)
    y_coords = np.arange(start_y, start_y + map_width, particle_spacing)
    
    inv_width = 1.0 / map_width
  
    # 1. 2D grid of particle columns ('ij' indexing: x along the first axis)
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
    flat_x = X.ravel()
    flat_y = Y.ravel()

    # 2. Height of the map at each column
    u = (flat_x - start_x) * inv_width
    v = (flat_y - start_y) * inv_width
    
    cols = np.clip(u * n_cells, 0, n_cells - 1).astype(int)
    rows = np.clip(v * n_cells, 0, n_cells - 1).astype(int)
    floor_heights = height_map[rows, cols]

    # 3. Stack particles layer by layer up to the local height
    valid_x = []
    valid_y = []
    valid_z = []

    max_h = np.max(floor_heights)
    z_cursor = particle_radius # first layer rests on the ground
    
    while z_cursor < max_h:
        mask = floor_heights > z_cursor
        
        if np.any(mask):
            valid_x.append(flat_x[mask])
            valid_y.append(flat_y[mask])
            valid_z.append(np.full(np.count_nonzero(mask), z_cursor))
        
        z_cursor += particle_spacing

    # 4. Concatenate and jitter
    if valid_x:
        p_x = np.concatenate(valid_x)
        p_y = np.concatenate(valid_y)
        p_z = np.concatenate(valid_z)
        
        count = len(p_x)
        
        # Random jitter breaks up the regular lattice of the initial positions
        # (Newton's example jitters by one particle spacing; here jitter_scale x spacing)
        jitter_mag = particle_spacing * jitter_scale
        
        p_x += rng.uniform(-0.5, 0.5, size=count) * jitter_mag
        p_y += rng.uniform(-0.5, 0.5, size=count) * jitter_mag
        p_z += rng.uniform(-0.5, 0.5, size=count) * jitter_mag 
        
        points = np.column_stack((p_x, p_y, p_z)).tolist()

    # E. Add the particles to the model
    if points:
        pts_np = np.array(points)
        count = len(pts_np)
        builder.add_particles(
            pos=pts_np.tolist(),
            vel=np.zeros_like(pts_np).tolist(),
            mass=[mass_per_particle] * count,
            radius=[particle_radius] * count,
        )
        logger.info(f"Emitted {count} particles from heightmap.")
    else:
        logger.warning("Heightmap loaded but 0 particles emitted.")
    return {'particle_radius': particle_radius}        
    
def add_spatula(
    builder: newton.ModelBuilder,
    start_pos: wp.vec3=None,
    start_rot:wp.quat=wp.quat_identity(),
    spatula_height:float=None,
    spatula_width:float=None,
    spatula_thickness:float=None,) -> int:
    """Add the spatula as a kinematic box body (mass 0) resting on the ground; returns its body index."""

    # Lift so the bottom rests on the ground plane
    start_pos[2] += spatula_height / 2.0  
    
    shape_cfg = newton.ModelBuilder.ShapeConfig(mu=0.5)
    
    # Mass 0 makes the body kinematic: it follows the prescribed trajectory and is not pushed back by the particles
    body_index = builder.add_body(
        xform=wp.transform(p=start_pos, q=start_rot),
        mass=0.0,
    )

    # Box shape attached to the body
    builder.add_shape_box(
        body=body_index,
        cfg=shape_cfg,
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), start_rot), # shape transform is relative to the body's center of mass (COM)
        hx=spatula_thickness / 2.0, hy=spatula_width / 2.0, hz=spatula_height / 2.0,
    )
    
    return body_index

def yaw_to_quaternion(yaw: float) -> np.ndarray:
    """Convert a yaw angle (rotation about z) to a quaternion [x, y, z, w]."""
    cos_yaw = np.cos(yaw * 0.5)
    sin_yaw = np.sin(yaw * 0.5)
    q = np.array([0.0, 0.0, sin_yaw, cos_yaw])  # [x, y, z, w]
    return q

def generate_sweep_trajectory(
    sweeps_xy: List[Tuple[np.ndarray, np.ndarray]],
    sim_dT: float,
    settling_dT: float,
    sweep_speed: float,
    body_height: float,
    default_pose: np.ndarray,
    start_time: float = 0.0,
) -> Tuple[Dict[int, np.ndarray], Dict[str, float]]:
    """
    Spatula pose for every simulation step of a sequence of pushes, and the times of the snapshot events.

    For each push: wait `settling_dT` at the parked pose (`MPM_PRESWEEP<n>` snapshot at the start of
    this wait), jump to the start pose, move in a straight line at `sweep_speed`, jump back to the
    parked pose and wait `settling_dT` again (`MPM_POSTSWEEP<n>` snapshot at the end).
    Poses are [x, y, z, qx, qy, qz, qw].
    """
    SETTLING_STEPS = round(settling_dT / sim_dT)
    trajectory_map = {}
    event_timestamps = {}
    
    current_step = round(start_time / sim_dT)

    for i, (P_start_xy, P_end_xy) in enumerate(sweeps_xy):
        sweep_idx = i + 1
        
        # --- 1. Settle with the spatula parked; snapshot before the push ---
        event_timestamps[f'MPM_PRESWEEP{sweep_idx}'] = current_step * sim_dT
        for _ in range(SETTLING_STEPS):
            trajectory_map[current_step] = default_pose
            current_step += 1

        # --- 2. Start and end pose of the push ---
        P_delta_xy = P_end_xy - P_start_xy
        D = np.linalg.norm(P_delta_xy)
        yaw_rad = np.arctan2(P_delta_xy[1], P_delta_xy[0])
        q_orientation = yaw_to_quaternion(yaw_rad)

        P_start_pose = np.concatenate([P_start_xy, [body_height], q_orientation])
        P_end_pose = np.concatenate([P_end_xy, [body_height], q_orientation])

        # Jump to the start pose
        trajectory_map[current_step] = P_start_pose
        current_step += 1

        # --- 3. Push: straight line at constant speed ---
        sweep_duration_steps = max(1, round((D / sweep_speed) / sim_dT))
        velocity_per_step = (P_end_pose - P_start_pose) / sweep_duration_steps

        for step_offset in range(1, sweep_duration_steps + 1):
            trajectory_map[current_step] = P_start_pose + step_offset * velocity_per_step
            current_step += 1

        # --- 4. Jump back to the parked pose (out of the camera view) ---
        trajectory_map[current_step] = default_pose
        current_step += 1

        # --- 5. Settle without the spatula ---
        for _ in range(SETTLING_STEPS):
            trajectory_map[current_step] = default_pose
            current_step += 1
        
        # --- 6. Snapshot after the push ---
        event_timestamps[f'MPM_POSTSWEEP{sweep_idx}'] = current_step * sim_dT

    return trajectory_map, event_timestamps

def print_simulation_report(logger: logging.Logger, 
                            options, 
                            sim_dt: float, 
                            sim_substeps: int, 
                            max_sim_steps: int, 
                            particle_radius: float, 
                            raycast_snapshot_steps: dict, 
                            event_timestamps: dict = None, 
                            viewer = None,
                            log_directory: str = None):
    """Log a summary of the run: grid and time step, stability checks (CFL, tunneling) and the event timeline."""
    if event_timestamps is None:
        event_timestamps = {}

    # --- 1. Calculate Safety Metrics (Physics Checks) ---
    # Load heightmap to determine max potential energy (drop height)
    try:
        _h_data = np.load(options.init_heightmap_path)
        _h_drop = np.max(_h_data['heightmap'])
    except Exception as e:
        logger.warning(f"Could not load heightmap for safety check: {e}")
        _h_drop = 1.0 # Default fallback
    
    # Calculate theoretical maximum velocity (Free fall: v = sqrt(2gh))
    _v_max = np.sqrt(2 * 9.81 * _h_drop)
    
    # CFL Condition: Measures if a particle moves further than one grid cell in a single step.
    _cfl = (_v_max * sim_dt) / options.voxel_size
    
    # Tunneling Ratio: Measures if the tool moves further than a particle's radius in one step.
    _tun = (options.sweep_speed * sim_dt) / particle_radius
    
    # --- 2. Build Timeline List ---
    # Initialize with Start and End
    _timeline = [(0.0, "START", "Simulation Start")]
    _timeline.append((max_sim_steps * sim_dt, "END", "Simulation End"))
    
    # Add Raycast Snapshots
    for _s, _n in raycast_snapshot_steps.items():
        _timeline.append((_s * sim_dt, "SNAP", f"Snapshot: {_n}"))
        
    # Add Trajectory Events (if any)
    if options.apply_sweeps:
        for _n, _t in event_timestamps.items():
            _timeline.append((_t, "MOVE", f"Event: {_n}"))
    
    # Sort events chronologically
    _timeline.sort(key=lambda x: x[0])

    # --- 3. Print Unified Report ---
    logger.info("\n" + "="*65)
    logger.info(f" MPM SIMULATION CONFIGURATION: {options.exp_name}")
    logger.info("="*65)
    
    logger.info(f" [PHYSICS & GRID]")
    logger.info(f"   + Voxel Size:    {options.voxel_size:.4f} m")
    logger.info(f"   + Particle Rad:  {particle_radius*1000:.2f} mm")
    logger.info(f"   + Time Step:     {sim_dt*1000:.3f} ms (Substeps: {sim_substeps})")
    
    # Handle viewer frames print
    if viewer and hasattr(viewer, 'num_frames'):
        logger.info(f"   + Total Time:    {viewer.num_frames} frames ({max_sim_steps * sim_dt:.2f}s)")
    else:
        logger.info(f"   + Total Time:    {max_sim_steps * sim_dt:.2f}s")

    logger.info(f" [SAFETY CHECKS]")
    logger.info(f"   + Max Velocity:  {_v_max:.2f} m/s (Max Drop: {_h_drop:.2f}m)")
    logger.info(f"   + CFL Ratio:     {_cfl:.4f}  [{'FAIL' if _cfl > 1.0 else 'OK'}]  (Req < 1.0)")
    logger.info(f"   + Tunneling:     {_tun:.4f}  [{'FAIL' if _tun > 0.5 else 'OK'}]  (Req < 0.5)")
    
    if _cfl > 1.0 or _tun > 0.5:
            logger.warning(f"   >>> WARNING: SIMULATION UNSAFE. CHECK RATIOS ABOVE. <<<")

    logger.info(f" [TIMELINE]")
    for i, (_t, _type, _desc) in enumerate(_timeline):
        # Select ASCII icon based on type
        if _type == "START": _icon = "(S)"
        elif _type == "END": _icon = "(E)"
        elif _type == "SNAP": _icon = "[O]" # Camera lens shape
        else: _icon = "-->" # Movement arrow
        
        logger.info(f"   {_t:6.2f}s | {_icon} {_desc}")
        
        # Print visual connector '|' only if there is a time gap > 0.1s
        if i < len(_timeline) - 1 and (_timeline[i+1][0] - _t > 0.1):
            logger.info(f"          |")
    # print where logs are saved
    logger.info(f"\n Logs and outputs saved to: {log_directory}")
    logger.info("="*65 + "\n")

def depth_scan(
        raycast_sensor,
        state,
        tag, 
        camera_poses_dict=None, 
        log_directory=None):
    """
    Take a depth image of the particles and save it as PNG and point cloud (PLY): depth_image_<tag>[_<view>].

    Without `camera_poses_dict`, one scan from the current sensor pose (points in the camera frame);
    otherwise one scan per view, transformed to the world frame.
    """
    if camera_poses_dict is None:
        raycast_sensor.eval(state, include_particles=True)
        depth_img_arr = raycast_sensor.get_depth_image_numpy()
        fname = f"{log_directory}/depth_image_{tag}"
        identity_transform = np.eye(4)
        save_depth_image_and_cloud(depth_img_arr, fname, fov_radians=raycast_sensor.fov_radians, T_world_cam=identity_transform)
    else:
        for view in camera_poses_dict.keys():
            raycast_sensor.update_camera_pose(
                position=camera_poses_dict[view]['camera_position'],
                direction=camera_poses_dict[view]['camera_direction'],
                up=camera_poses_dict[view]['camera_up'],
            )
            raycast_sensor.eval(state, include_particles=True)
            depth_img_arr = raycast_sensor.get_depth_image_numpy()
            # Camera -> world transform
            T_world_cam = np.eye(4)
            nx_cam = raycast_sensor.camera_right
            ny_cam = raycast_sensor.camera_up
            nz_cam = raycast_sensor.camera_direction
            T_world_cam[0:3, 0] = np.array( nx_cam )
            T_world_cam[0:3, 1] = np.array( ny_cam )
            T_world_cam[0:3, 2] = np.array( np.array(nz_cam) )  # camera direction is -Z in camera frame
            T_world_cam[0:3, 3] = np.array( raycast_sensor.camera_position )
            fname = f"{log_directory}/depth_image_{tag}_{view}"
            save_depth_image_and_cloud(depth_img_arr, fname, fov_radians=raycast_sensor.fov_radians, T_world_cam=T_world_cam)

def save_depth_image_and_cloud(depth_image: np.ndarray, 
                               filename: str, 
                               fov_radians: float, 
                               T_world_cam: np.ndarray = None) -> None:
    """
    Saves a depth image png and a corrected PLY pointcloud.
    
    Args:
        depth_image: (H, W) array of Euclidean distances (-1.0 for no hit)
        filename: Output path prefix (e.g. "output/scan_01")
        fov_radians: Vertical Field of View in radians (MUST match sensor)
        T_world_cam: 4x4 Transformation matrix (World <- Camera)
    """
    # 1. Save Visual PNG
    # Normalize depth for visualization (0-255)
    viz_data = depth_image.copy()
    valid_mask = viz_data > 0
    
    if np.any(valid_mask):
        min_d, max_d = viz_data[valid_mask].min(), viz_data[valid_mask].max()
        # Invert so close = bright
        viz_data[valid_mask] = 255 - ((viz_data[valid_mask] - min_d) / (max_d - min_d + 1e-6) * 255)
    
    viz_data[~valid_mask] = 0 # Black background
    Image.fromarray(viz_data.astype(np.uint8)).save(f"{filename}.png")

    # 2. Generate Corrected Point Cloud
    height, width = depth_image.shape
    
    # CALCULATE FOCAL LENGTHS (Based on Sensor Math)
    # Sensor uses: scale = tan(fov/2)
    fov_scale = math.tan(fov_radians / 2.0)
    
    fy = (height / 2.0) / fov_scale
    fx = fy * (width / height)  # Assuming square pixels
    
    cx = width / 2.0
    cy = height / 2.0

    # Pixel coordinates (corner-aligned, as used for the paper's data)
    u_grid, v_grid = np.meshgrid(np.arange(width), np.arange(height))

    # Flatten
    u = u_grid.flatten()
    v = v_grid.flatten()
    d_euclidean = depth_image.flatten() # 'd' is Euclidean distance
    
    # Filter valid points
    valid = d_euclidean > 0
    if not np.any(valid):
        print("No valid points to save.")
        return

    u, v, d = u[valid], v[valid], d_euclidean[valid]
    
    # Calculate normalized image coordinates (x', y', 1)
    x_norm = (u - cx) / fx
    y_norm = (v - cy) / fy
    
    # The sensor measures the Euclidean ray length d; the point cloud needs the depth Z
    # (perpendicular distance to the image plane).
    # Relationship: Z = d / ||ray_vector||
    # ||ray_vector|| = sqrt(x_norm^2 + y_norm^2 + 1)
    
    ray_lengths = np.sqrt(x_norm**2 + y_norm**2 + 1.0)
    z_planar = d / ray_lengths
    
    # Reproject using Planar Z
    x_cam = x_norm * z_planar
    y_cam = y_norm * z_planar
    z_cam = z_planar
    
    points_cam = np.stack((x_cam, y_cam, z_cam), axis=-1)

    # Transform to World
    if T_world_cam is not None:
        # Homogeneous transform
        pts_h = np.ones((len(points_cam), 4))
        pts_h[:, :3] = points_cam
        points_3d = (T_world_cam @ pts_h.T).T[:, :3]
    else:
        points_3d = points_cam
        
    # 3. Write PLY
    header = f"""ply
            format ascii 1.0
            element vertex {len(points_3d)}
            property float x
            property float y
            property float z
            end_header
        """
    with open(f"{filename}.ply", "w") as f:
        f.write(header)
        np.savetxt(f, points_3d, fmt="%.4f")
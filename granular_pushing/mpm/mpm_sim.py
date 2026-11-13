# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified from newton/examples/mpm/example_mpm_granular.py (Newton commit e5f6493):
# particles initialized from a height map, a kinematic spatula that executes pushes,
# depth-camera snapshots at scheduled events, configuration via YAML, and runtime telemetry.

"""
MPM simulation of a granular pile, optionally with pushes, using Newton's implicit MPM solver.

Usually called through mpm_runner.run_mpm_simulation. Directly:
    python mpm_sim.py [--config <yaml>] [--set key=value ...] [--viewer null|gl]
Settings come from mpm_default_config.yaml (or --config) and are overridden by --set.

Writes into <log_directory>/<exp_name>/:
  - depth images and point clouds (*.png, *.ply) at the scheduled events
    (MPM_PRELIFT, MPM_POSTLIFT, MPM_PRESWEEP<n>, MPM_POSTSWEEP<n>)
  - resolved_sim_config.yaml (the settings actually used)
  - raw_telemetry.csv / raw_telemetry_meta.json / raw_telemetry_trajectory.json
    (timings for the runtime comparison, see plotting/plot_push_times.py)
"""

import argparse
import logging
import math
import numpy as np
import os
import sys
import time

import json
import csv

from datetime import datetime
from pathlib import Path

from ruamel.yaml import YAML

import warp as wp
import newton
import newton.examples
from newton.sensors import SensorRaycast
from newton.solvers import SolverImplicitMPM
from newton.viewer import ViewerNull

from mpm_utils import (
    emit_particles,
    add_spatula,
    generate_sweep_trajectory,
    print_simulation_report,
    depth_scan
)

# YAML handler (ruamel keeps the formatting when the resolved config is archived)
ryaml = YAML()
ryaml.preserve_quotes = True
ryaml.indent(mapping=2, sequence=4, offset=2)
ryaml.representer.add_representer(type(None), lambda self, data: self.represent_scalar('tag:yaml.org,2002:null', 'null'))
def safe_float_representer(self, data):
    return self.represent_float(data)
ryaml.representer.add_representer(float, safe_float_representer)
ryaml.representer.add_representer(np.float64, safe_float_representer)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("MPM_Sim")

class Example:
    """Newton example: pile + optional spatula, MPM solver, depth sensor, event schedule and telemetry."""
    def __init__(self, viewer, options, log_directory, script_start_time) -> None:
        self.script_start_time = script_start_time
        # ----------------------------------------------------------------------
        # 1. CONFIGURATION & TIMING
        # ----------------------------------------------------------------------
        self.options = options
        self.log_directory = log_directory
        
        # Simulation timing parameters
        self.fps = self.options.fps
        self.sim_substeps = self.options.substeps
        self.frame_dt = 1.0 / self.fps
        self.sim_dt = self.frame_dt / self.sim_substeps
        
        # Incremental state variables
        self.sim_time = 0.0
        self.physics_step_counter = 0
        self.telemetry_data = []

        # Interaction Parameters
        self.mpm_settling_time = self.options.mpm_settling_time
        self.sweep_speed = wp.float32(self.options.sweep_speed)
        self.spatula_safe_pose = wp.vec3(0.8, 0.0, 0.0)

        # ----------------------------------------------------------------------
        # 2. SCENE FABRICATION (Geometry & Bodies)
        # ----------------------------------------------------------------------
        builder = newton.ModelBuilder()
        builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=self.options.groundplane_mu))

        # A. Emit Particles
        particle_radius = emit_particles(
            builder,
            logger,
            init_heightmap_path=self.options.init_heightmap_path,
            density=self.options.density,
            particles_per_voxel=self.options.particles_per_voxel,
            voxel_size=self.options.voxel_size,
            jitter_scale=self.options.jitter_scale,
        )['particle_radius']

        # B. Spatula (kinematic body, parked far away until the first push)
        self.spatula_body_id = -1
        if self.options.apply_sweeps:
            spatula_init_pose = wp.vec3(100.0, 100.0, 100.0) 
            self.spatula_body_id = add_spatula(
                builder,
                start_pos=spatula_init_pose,
                spatula_height=self.options.spatula_height,
                spatula_width=self.options.spatula_width,
                spatula_thickness=self.options.spatula_thickness
            )

        # C. Finalize Newton Model
        self.model = builder.finalize()
        self.model.particle_mu = self.options.friction_coeff
        self.model.set_gravity(self.options.gravity)

        # ----------------------------------------------------------------------
        # 3. MPM SETUP
        # ----------------------------------------------------------------------
        # Disable Newton's internal physics to use MPM
        self.model.particle_ke = None
        self.model.particle_kd = None
        self.model.particle_cohesion = None
        self.model.particle_adhesion = None

        # MPM solver: every config key that matches a solver option is passed on (see mpm_default_config.yaml)
        mpm_options = SolverImplicitMPM.Options()
        for key in vars(options):
            if hasattr(mpm_options, key):
                setattr(mpm_options, key, getattr(options, key))

        # Initialize Solver and State
        mpm_model = SolverImplicitMPM.Model(self.model, mpm_options)
        self.solver = SolverImplicitMPM(mpm_model, mpm_options)
        
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.solver.enrich_state(self.state_0)
        self.solver.enrich_state(self.state_1)

        # ----------------------------------------------------------------------
        # 4. VIEWER AND DEPTH SENSOR
        # ----------------------------------------------------------------------
        # A. Viewer (a null viewer when headless)
        self.viewer = viewer
        self.viewer.set_model(self.model)
        self.viewer.show_particles = True
        self.viewer.set_camera(
            pos=wp.vec3(self.options.viewer_pos), 
            pitch=self.options.viewer_pitch_deg, 
            yaw=self.options.viewer_yaw_deg
        )

        # B. Depth sensor (top view)
        self.raycast_poses = {
            "TOPVIEW": {
                "camera_position": self.options.camera_pos,
                "camera_direction": self.options.camera_direction,
                "camera_up": self.options.camera_up
            },
        }
        
        self.raycast_sensor = SensorRaycast(
            model=self.model,
            camera_position=self.raycast_poses["TOPVIEW"]["camera_position"],
            camera_direction=self.raycast_poses["TOPVIEW"]["camera_direction"],
            camera_up=self.raycast_poses["TOPVIEW"]["camera_up"],
            fov_radians=np.deg2rad(self.options.depth_fov_deg),
            width=self.options.depth_res,
            height=self.options.depth_res,
            max_distance=3.0,
        )

        # ======================================================================
        # 5. EVENT SCHEDULE (snapshots and spatula trajectory)
        # ======================================================================
        self.sweep_trajectory = {}
        event_timestamps = {}
        snapshot_schedule = {}

        # A. Snapshots before and after the initial settling
        snapshot_schedule[0.0] = "MPM_PRELIFT"
        snapshot_schedule[self.mpm_settling_time] = "MPM_POSTLIFT"

        # B. Push trajectory and its snapshot events
        if self.options.apply_sweeps and self.spatula_body_id >= 0:
            self.sweeps_xy = [
                (np.array(row[:2]), np.array(row[2:])) 
                for row in self.options.sweeps_xy
            ]
            half_height = self.options.spatula_height / 2.0
            
            # Parked pose between pushes
            default_pose = np.concatenate([
                np.array([self.spatula_safe_pose[0], self.spatula_safe_pose[1], half_height]),
                np.array([1.0, 0.0, 0.0, 0.0]) 
            ])
            
            self.sweep_trajectory, event_timestamps = generate_sweep_trajectory(
                sweeps_xy=self.sweeps_xy,
                sim_dT=self.sim_dt,
                settling_dT=self.mpm_settling_time,
                sweep_speed=self.sweep_speed,
                body_height=half_height,
                default_pose=default_pose,
            )
            
            for event, t in event_timestamps.items():
                snapshot_schedule[t] = event

        # C. Event times -> simulation steps
        self.raycast_snapshot_steps = {}
        for t, event_name in snapshot_schedule.items():
            step = int(round(t / self.sim_dt))
            self.raycast_snapshot_steps[step] = event_name
            logger.info(f"Scheduled event '{event_name}' at {t:.4f}s (step {step})")

        # D. Headless: run until shortly after the last event
        if isinstance(self.viewer, ViewerNull) and self.raycast_snapshot_steps:
            last_event_step = max(self.raycast_snapshot_steps.keys())
            required_steps = last_event_step + 100
            
            self.options.num_frames = math.ceil(required_steps / self.sim_substeps)
            # self.viewer.set_new_num_frames(self.options.num_frames)
            self.viewer.num_frames = self.options.num_frames
            logger.info(f"Headless Mode: Script requires {self.options.num_frames} frames.")

        self.max_sim_steps = self.options.num_frames * self.sim_substeps

        # ----------------------------------------------------------------------
        # 6. EXECUTION PREP
        # ----------------------------------------------------------------------
        os.makedirs(self.log_directory, exist_ok=True)
        
        # Depth scan of the initial state
        if self.options.render_depth:
            depth_scan(
                raycast_sensor=self.raycast_sensor,
                state=self.state_0,
                tag='initial',
                camera_poses_dict=None, 
                log_directory=self.log_directory
            )

        print_simulation_report(
            logger, 
            options, 
            sim_dt=self.sim_dt, 
            sim_substeps=self.sim_substeps, 
            max_sim_steps=self.max_sim_steps, 
            particle_radius=particle_radius, 
            raycast_snapshot_steps=self.raycast_snapshot_steps, 
            event_timestamps=event_timestamps if self.options.apply_sweeps else None, 
            viewer=self.viewer,
            log_directory=self.log_directory
        )
        
        self.example_init_time = time.time()

        logger.info("Simulation initialized. Starting main loop.")

    def step(self):
            self.simulate()
            self.sim_time += self.frame_dt

    def simulate(self):       
        for _ in range(self.sim_substeps):
            step_idx = self.physics_step_counter

            # 1. Snapshots and spatula pose (before the step)
            if self.options.render_depth:
                self._capture_sensors(step_idx)
            if self.options.apply_sweeps:
                self._apply_kinematics(step_idx)

            # 2. MPM step
            self.solver.step(self.state_0, self.state_1, None, None, self.sim_dt)
            self.solver.project_outside(self.state_1, self.state_1, self.sim_dt)

            # 3. Telemetry, after the step: wall-clock time, sim time, step, kinetic energy
            # (kinetic energy is skipped in timed runs, as it needs a GPU -> CPU copy)
            ke = np.sum(np.square(self.state_0.particle_qd.numpy())) if not self.options.timed_run else None
            
            self.telemetry_data.append((
                time.time(), 
                self.sim_time, 
                step_idx, 
                ke
            ))

            # 4. Swap states
            self.state_0, self.state_1 = self.state_1, self.state_0

            self.physics_step_counter += 1

    def render(self):
        """Called by Newton's run loop (newton.examples.run)."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def save_telemetry(self):
        """Write raw_telemetry.csv, raw_telemetry_trajectory.json and raw_telemetry_meta.json."""
        if not self.telemetry_data:
            logger.warning("No telemetry data captured. Simulation may not have run.")
            return

        # 1. Per-step timings
        csv_path = os.path.join(self.log_directory, "raw_telemetry.csv")
        with open(csv_path, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "sim_time", "step", "kinetic_energy"])
            writer.writerows(self.telemetry_data)

        # 2. Spatula trajectory (pushes only)
        if getattr(self, "sweep_trajectory", None):
            safe_trajectory = {}
            for step_key, pose_array in self.sweep_trajectory.items():
                safe_trajectory[int(step_key)] = pose_array.tolist() if isinstance(pose_array, np.ndarray) else list(pose_array)
            
            traj_path = os.path.join(self.log_directory, "raw_telemetry_trajectory.json")
            with open(traj_path, "w") as f:
                json.dump(safe_trajectory, f, indent=4)

        # 3. Run metadata
        meta_data = {
            "script_start_time": getattr(self, "script_start_time", 0.0),
            "example_init_time": getattr(self, "example_init_time", 0.0),
            "apply_sweeps": getattr(self.options, "apply_sweeps", False) if hasattr(self, "options") else False,
            "spatula_safe_pose": [float(x) for x in self.spatula_safe_pose] if hasattr(self, "spatula_safe_pose") else [0.0, 0.0],
            "raycast_snapshot_steps": list(self.raycast_snapshot_steps.keys()) if hasattr(self, "raycast_snapshot_steps") else []
        }

        meta_path = os.path.join(self.log_directory, "raw_telemetry_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta_data, f, indent=4)

        logger.info(f"Raw telemetry and metadata saved to {self.log_directory}")

    def _capture_sensors(self, step_idx):
        """Checks if the current step requires a sensor snapshot."""
        if (self.options.render_depth) and (step_idx in self.raycast_snapshot_steps):
            logger.info(f"Capturing raycast depth image for event '{self.raycast_snapshot_steps[step_idx]}' at step {step_idx}")
            depth_scan(
                raycast_sensor=self.raycast_sensor,
                state=self.state_0,
                tag=f"{str(step_idx).zfill(4)}_{self.raycast_snapshot_steps[step_idx]}", 
                camera_poses_dict=self.raycast_poses, 
                log_directory=self.log_directory
            )
            logger.info(f"Captured raycast depth image for event '{self.raycast_snapshot_steps[step_idx]}' at step {step_idx}")

    def _apply_kinematics(self, step_idx):
        """Applies kinematic updates (e.g. Spatula trajectory) to the state."""
        # Fast exit if no spatula or no event scheduled for this step
        if self.spatula_body_id < 0 or step_idx not in self.sweep_trajectory:
            return

        # 1. Target pose
        target_pose = self.sweep_trajectory[step_idx]
        target_pos = wp.vec3(target_pose[0], target_pose[1], target_pose[2])
        target_rot = wp.quat(target_pose[3], target_pose[4], target_pose[5], target_pose[6])
        
        # 2. Set the pose (body_q)
        new_transform = wp.transform(p=target_pos, q=target_rot)
        wp.copy(
            dest=self.state_0.body_q, 
            dest_offset=self.spatula_body_id, 
            src=wp.array([new_transform], dtype=wp.transform, device=self.state_0.body_q.device), 
            src_offset=0, 
            count=1
        )

        # 3. Set the velocity (body_qd): push speed along the spatula's x-axis, in the world frame
        local_x_axis = wp.vec3(1.0, 0.0, 0.0)
        world_x_axis_dir = wp.quat_rotate(target_rot, local_x_axis)
        linear_vel = self.sweep_speed * world_x_axis_dir
        
        new_velocity = wp.spatial_vector(
            linear_vel[0], linear_vel[1], linear_vel[2],
            0.0, 0.0, 0.0
        )
        wp.copy(
            dest=self.state_0.body_qd, 
            dest_offset=self.spatula_body_id, 
            src=wp.array([new_velocity], dtype=wp.spatial_vector, device=self.state_0.body_qd.device), 
            src_offset=0, 
            count=1
        )

if __name__ == "__main__":
    script_start_time = time.time()
    script_dir = Path(__file__).parent.resolve()
    default_config_path = script_dir / "mpm_default_config.yaml"
    
    # Read --config and --set first, then leave only Newton's own arguments in sys.argv
    # (Newton's parser would otherwise fail on --config / --set)
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=str(default_config_path))
    pre_parser.add_argument("--set", nargs='*', help="Override via key=value")    
    my_args, newton_raw_args = pre_parser.parse_known_args()
    sys.argv = [sys.argv[0]] + newton_raw_args
    
    # Load the (flat) YAML config
    yaml_path = Path(my_args.config)
    if not yaml_path.exists():
        logger.error(f"Config file not found: {yaml_path}")
        sys.exit(1)

    logger.info(f"Loading config from: {yaml_path}")
    with open(yaml_path, 'r') as f:
        config = ryaml.load(f)

    # --set key=value overrides
    if my_args.set:
        for item in my_args.set:
            if '=' in item:
                key, val = item.split('=', 1)
                try:
                    # safe_load only converts the value to the right type (int, float, bool, list)
                    import yaml as pyyaml 
                    config[key] = pyyaml.safe_load(val) 
                    logger.info(f"Override: {key} = {config[key]}")
                except Exception as e:
                    logger.error(f"Override Error on {key}: {e}")

    # Output folder: <log_directory>/<exp_name>
    base_log_dir = Path(config.get('log_directory', '/tmp/newton_sim')).expanduser().resolve()
    exp_name = config.get('exp_name', 'experiment')
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    resolved_log_path = base_log_dir / f"{exp_name}"
    resolved_log_path.mkdir(parents=True, exist_ok=True)
  
    # Newton viewer and arguments
    parser = newton.examples.create_parser()
    viewer, args = newton.examples.init(parser)

    # Config keys become attributes of args (the Example reads them from there)
    for key, value in config.items():
        setattr(args, key, value)

    # Save the settings actually used
    archive_path = resolved_log_path / "resolved_sim_config.yaml"
    with open(archive_path, 'w') as f:
        ryaml.dump(config, f)
    logger.info(f"Archive created at: {archive_path}")

    example = Example(viewer, args, log_directory=str(resolved_log_path), script_start_time=script_start_time)    

    try:    
        newton.examples.run(example, args)
    finally:
        # Raw telemetry (used for the runtime comparison)
        example.save_telemetry()

    print("Exiting example.")
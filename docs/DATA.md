# Dataset

Robot-arm experiments with three granular materials (grape stones, coffee beans, rolled oats): point cloud scans of the piles, robot poses and planner logs, from three experiments:

| Experiment | Used for |
|---|---|
| Bucket lift | Calibrating the models |
| Push | Evaluating how well the models predict the pile shape after a push |
| Clearance | Clearing a target region with the HBM-based planner |

Download: [doi.org/10.3929/ethz-c-000808988](https://doi.org/10.3929/ethz-c-000808988) (ETH Research Collection). Unpack it into `data/` in the repository, so that `data/BucketLift`, `data/Push` and `data/Clearance_HBM` exist.

## Data collection

This section describes the setup, the materials and how each experiment was carried out.

### Hardware and software

All experiments used the same robot setup:

| | |
|---|---|
| Robot | Franka Emika Panda, 7-DOF |
| Tool | Plastic spatula, 7 cm wide |
| Sensor | [pmd flexx2-VGA](https://3d.pmdtec.com/en/3d-cameras/flexx2/) time-of-flight camera, wrist-mounted |
| Workspace | Flat table, 60 cm × 60 cm |
| Software | [ROS 1 Noetic](https://wiki.ros.org/noetic) on Ubuntu 20.04: robot control, scanning (pmd ROS driver) and logging. Pushes are straight-line motions planned with [MoveIt](https://moveit.ai/) and the Pilz industrial motion planner. |

### Materials

We used three dry, cohesionless materials that form stable piles:

| Material | Bulk density |
|---|---|
| Grape stones | 689 kg/m³ |
| Coffee beans | 377 kg/m³ |
| Rolled oats | 477 kg/m³ |

The bulk densities were measured by filling a container of known volume and weighing it.

### Procedure

All scans are taken from about 70 cm above the table. Each scan is saved under a tag that says what it shows (in brackets below). Pushes are called *sweeps* in the tags and in the code.

**Bucket lift** (3 runs per material)
1. Scan the empty table, used to fit the table plane (`EMPTY_SCAN`).
2. Place a cuboid bucket (9 cm × 9 cm × 22 cm) at a fixed position and record its pose with a pointer on the robot (`POSITION_PRISM_L`, `run_01` only).
3. Fill the bucket with material and scan it (`PRELIFT_SCAN`).
4. Lift the bucket with the robot; the material settles into a pile. Scan the pile (`POSTLIFT_SCAN`).

**Push** (3 runs per material, inward and outward)
1. Scan the empty table (`EMPTY_SCAN`) and record the bucket pose (`POSITION_CYLINDER_L`, `run_01` only).
2. Fill a cylindrical bucket (9.03 cm diameter, 22.05 cm high) and scan it (`PRELIFT_SCAN`).
3. Lift the bucket to form a pile and scan it before the first push (`PRESWEEP1_SCAN`).
4. Push the pile in a straight line at 0.1 m/s, recording the spatula pose at the start and end of the push (`PRESWEEP1_POSE`, `POSTSWEEP1_POSE`). *Inward* pushes move material toward the pile's centre (20 cm per push), *outward* pushes move it away from the centre (15 cm per push).
5. Retract the spatula, let the material settle and scan (`PRESWEEP2_SCAN`).
6. Do the second push the same way (`PRESWEEP2_POSE`, `POSTSWEEP2_POSE`) and scan again (`POSTSWEEP2_SCAN`).

**Clearance** (1 run per material and HBM variant: PS, PW, PSE, PWE)
1. Form a pile with the cylindrical bucket, centred on a circular target region (radius 7.5 cm).
2. Scan the pile (`PRESWEEP_SCAN`). From this single scan, the planner plans a sequence of 10 pushes.
3. Execute the pushes open-loop, one after the other, and scan after each push (`POSTSWEEP_SCAN_1` ... `POSTSWEEP_SCAN_10`). These scans only measure progress and do not change the plan.

## Folder structure and naming

The dataset has one folder per experiment, one subfolder per material (and push direction or model), one `run_XX` folder per repetition, and one folder per scan, named by its tag.

```
data/
├── BucketLift/
│   └── <material>/                         grapes, coffee_beans, oats
│       └── run_01/ ... run_03/
│           ├── EMPTY_SCAN/
│           ├── POSITION_PRISM_L/           run_01 only
│           ├── PRELIFT_SCAN/
│           └── POSTLIFT_SCAN/
├── Push/
│   └── <material>_inward/, <material>_outward/
│       └── run_01/ ... run_03/
│           ├── EMPTY_SCAN/
│           ├── POSITION_CYLINDER_L/        run_01 only
│           ├── PRELIFT_SCAN/
│           ├── PRESWEEP1_SCAN/
│           ├── PRESWEEP1_POSE/, POSTSWEEP1_POSE/
│           ├── PRESWEEP2_SCAN/
│           ├── PRESWEEP2_POSE/, POSTSWEEP2_POSE/
│           └── POSTSWEEP2_SCAN/
└── Clearance_HBM/
    └── <material>/<material>_<model>/      model: PS, PW, PSE, PWE
        └── run_01/
            ├── PRESWEEP_SCAN/
            ├── POSTSWEEP_SCAN_1/ ... POSTSWEEP_SCAN_10/
            └── planner_archive/planning_session/
                ├── actions_log.csv
                └── physical_scan_001/clearance_scores.csv
```

Every scan folder holds two files (`POSITION_PRISM_L` only `transforms.yaml`):
- `point_cloud.ply`: the point cloud (x, y, z in metres), cropped to the work area and expressed in the workplane frame (the table).
- `transforms.yaml`: the robot's frames at the time of the scan, as translation (metres) and rotation (quaternion w, x, y, z). The code uses `workplane` (table), `panda_spatula_tip` (push start and end) and `panda_cone_tip` (bucket position). For the `*_POSE` and `POSITION_*` folders, only this file is used.

The clearance logs:
- `actions_log.csv`: the planned pushes, with start and end point in metres (workplane frame). The first 16 rows (`INIT_TEST`) are test pushes recorded before planning and are not part of the plan.
- `clearance_scores.csv`: the volume left in the target region after each push, as computed during the experiment (m³, rounded to 10 cm³).

The planner settings of each clearance run are in the code repository: `granular_pushing/planner/configs/<material>/<material>_<model>.yaml`.

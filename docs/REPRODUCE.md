# Reproducing the Paper Results

This guide walks through the results of the paper step by step: calibration on the bucket-lift data, push prediction, runtimes and the clearance planner. Before you start, set up the container ([README](../README.md#setup)) and download the dataset ([DATA.md](DATA.md)).

## How to use this guide

- Run everything inside the container (`docker/run.sh`), from the repo root (`/workspace`).
- Work through the sections top to bottom **in one terminal**: the variables you `export` only live in that shell.
- Code blocks are templates. Replace every `<...>` with your own value.
- Under each template, a collapsed **Example** holds our values for grapes. Paste it instead of the template to run directly on our data.
- Each step writes into its own fixed folder under `RESULTS_DIR` (e.g. `$RESULTS_DIR/aor`). Later steps read from there, so you never copy folder names by hand.
- For what a script's flags do, run it with `--help`.
- Steps that simulate MPM (MPM tuning, push evaluation, timed runs) can take a while; run nothing else on the GPU meanwhile.

## Preliminaries

| Variable | Meaning |
|---|---|
| `DATA_ROOT` | Folder containing the experiment folders `BucketLift`, `Push`, `Clearance_HBM` (see [DATA.md](DATA.md)) |
| `MATERIAL_NAME` | Short name for the material (free to choose) |
| `MATERIAL_DENSITY_KGM3` | Bulk density in kg/m³. For our materials, see [DATA.md](DATA.md#materials). For your own material: fill a container of known volume, weigh it, divide |
| `RESULTS_DIR` | Where all results for this material are written (calibration and evaluation). Inside the container `/workspace/outputs/<material_name>`, i.e. `outputs/<material_name>` in the repo on your machine |

```bash
export DATA_ROOT=<data_root>
export MATERIAL_NAME=<material_name>
export MATERIAL_DENSITY_KGM3=<density>
export RESULTS_DIR=/workspace/outputs/$MATERIAL_NAME
```

<details><summary>Example: grapes (our data)</summary>

```bash
export DATA_ROOT=/workspace/data
export MATERIAL_NAME=grapes
export MATERIAL_DENSITY_KGM3=689
export RESULTS_DIR=/workspace/outputs/$MATERIAL_NAME
```

</details>

## Calibration from Bucket Lift Data

> **In the paper:** Sec. V-A, *Model Calibration* (Figs. 8 and 10).

Point to the bucket-lift experiment folder (relative to `DATA_ROOT`):
```bash
export BUCKETLIFT_PATH=$DATA_ROOT/<bucketlift_folder>
```

<details><summary>Example: grapes (our data)</summary>

```bash
export BUCKETLIFT_PATH=$DATA_ROOT/BucketLift/grapes
```

</details>

### HBM

> **In the paper:** Sec. V-A, *HBM Calibration*.

#### Angle of Repose

Estimate the angle of repose from the measured piles:
```bash
python scripts/model_eval.py --lift --headless \
    --root $BUCKETLIFT_PATH \
    --erosion-stencil lrud-1 \
    --finish-after-aor-estimation \
    --output-dir $RESULTS_DIR --analysis-name aor
```

The estimate is printed at the end of the run and saved in `$RESULTS_DIR/aor/estimated_aor.txt`. Round it to whole degrees and set it:
```bash
export HBM_AOR_DEG=<aor_deg>
```

<details><summary>Example: grapes (our data)</summary>

```bash
export HBM_AOR_DEG=23
```

</details>

#### Stencil

The erosion stencil sets the neighbourhood that HBM erosion compares heights over (type and size, e.g. `bloc-5` = 5×5 block).
We found `bloc-5` to be a good all-rounder and use it for all materials:
```bash
export HBM_STENCIL=bloc-5
```

#### Stencil sweep (optional)

Compare stencil types and sizes on your data, e.g. to check whether a different one fits better. Three steps:

1. Create a placeholder MPM run. The sweep only compares HBM stencils, but the evaluation script expects MPM results; this run provides them with placeholder parameters.
```bash
python scripts/model_eval.py --lift --headless \
    --root $BUCKETLIFT_PATH \
    --erosion-stencil lrud-1 \
    --finish-after-mpm-gen \
    --mpm-density $MATERIAL_DENSITY_KGM3 \
    --output-dir $RESULTS_DIR/stencil_sweep --analysis-name placeholder_mpm
```

2. Run HBM once per stencil:
```bash
python scripts/hbm_erosion_stencil_sweep.py run \
    --EXP-DATA-DIR $BUCKETLIFT_PATH \
    --MPM-DUMMY-RUN $RESULTS_DIR/stencil_sweep/placeholder_mpm \
    --sweep-name stencil_sweep \
    --aor $HBM_AOR_DEG \
    --output-dir $RESULTS_DIR/stencil_sweep/runs
```

3. Collect the results and plot them:
```bash
python scripts/hbm_erosion_stencil_sweep.py aggregate \
    --EXP-DATA-DIR $BUCKETLIFT_PATH \
    --MPM-DUMMY-RUN $RESULTS_DIR/stencil_sweep/placeholder_mpm \
    --sweep-name stencil_sweep \
    --aor $HBM_AOR_DEG \
    --path $RESULTS_DIR/stencil_sweep/runs

python plotting/plot_bucketlift_hbm_stencils.py \
    --input $RESULTS_DIR/stencil_sweep/runs/aggregated_results.csv \
    --out-dir $RESULTS_DIR/plots/stencil_sweep
```
`aggregated_results.csv` lists the IoU per stencil type and size. Ignore its MPM numbers (placeholder parameters).

**In the paper:** Fig. 8 is this plot for coffee beans.

To continue with a different stencil than `bloc-5`, set `HBM_STENCIL` to it before the next steps.

### MPM

> **In the paper:** Sec. V-A, *MPM*.

Tune the four MPM parameters (ground friction, material friction, critical fraction, voxel size) on the bucket-lift data (about 100 MPM simulations).
```bash
python scripts/tune_mpm.py \
    --DATA-ROOT $BUCKETLIFT_PATH \
    --AOR $HBM_AOR_DEG \
    --EROSION-STENCIL $HBM_STENCIL \
    --DENSITY $MATERIAL_DENSITY_KGM3 \
    --TUNING-NAME mpm_tuning \
    --output-dir $RESULTS_DIR/mpm_tuning
```
Tuning only simulates Settling; the Retention numbers inside the tuning folder are copies and shouldn't be used.

Rank the tested parameter sets:
```bash
python scripts/tune_mpm_rank.py --csv_log $RESULTS_DIR/mpm_tuning/optimization_history.csv
head -2 $RESULTS_DIR/mpm_tuning/optimization_history_analysis/ranked_results.csv
```

Set the parameters from the top row:
```bash
export MPM_ARGS="--mpm-ground-friction <ground_friction> --mpm-friction-coeff <friction> --mpm-critical-fraction <critical_fraction> --mpm-voxel-size <voxel_size> --mpm-density $MATERIAL_DENSITY_KGM3"
```

<details><summary>Example: grapes (our data)</summary>

```bash
export MPM_ARGS="--mpm-ground-friction 0.571401 --mpm-friction-coeff 0.686414 --mpm-critical-fraction 0.676798 --mpm-voxel-size 0.005 --mpm-density $MATERIAL_DENSITY_KGM3"
```

</details>

### Model Fit to Bucket Lift Data

Evaluate all models (HBM variants and MPM) against the measured bucket lift. This simulates HBM and MPM again with the calibrated parameters; no earlier simulation results are reused.
```bash
python scripts/model_eval.py --lift --headless \
    --root $BUCKETLIFT_PATH \
    --aor $HBM_AOR_DEG \
    --erosion-stencil $HBM_STENCIL \
    $MPM_ARGS \
    --output-dir $RESULTS_DIR --analysis-name lift_fit
```
This writes `LIFT_SUMMARY_LEADERBOARD.csv` (IoU per model, Settling and Retention) and `LIFT_SUMMARY_LONG_FORMAT.csv` into `$RESULTS_DIR/lift_fit`.

Optional, plot the comparison:

```bash
python plotting/plot_bucketlift_model_fit.py \
    --inputs $RESULTS_DIR/lift_fit/LIFT_SUMMARY_LEADERBOARD.csv \
    --out-dir $RESULTS_DIR/plots/lift_fit
```

**In the paper:** Fig. 10 is this plot with all three materials pooled: pass one `LIFT_SUMMARY_LONG_FORMAT.csv` per material to `--inputs`; the script then computes mean and standard deviation over all runs together.


## Push Evaluation

> **In the paper:** Sec. V-B, *Push Prediction Fidelity* (Fig. 12).

Uses the calibrated parameters from the previous section: `HBM_AOR_DEG`, `HBM_STENCIL` and `MPM_ARGS` must still be set.

Each material has two push experiments, each with two consecutive pushes: *inward* pushes material toward the pile's centre (20 cm per push), *outward* pushes it away from the centre (15 cm per push). Point to both folders (relative to `DATA_ROOT`):
```bash
export PUSH_IN_PATH=$DATA_ROOT/<push_inward_folder>
export PUSH_OUT_PATH=$DATA_ROOT/<push_outward_folder>
```

<details><summary>Example: grapes (our data)</summary>

```bash
export PUSH_IN_PATH=$DATA_ROOT/Push/grapes_inward
export PUSH_OUT_PATH=$DATA_ROOT/Push/grapes_outward
```

</details>

Predict each recorded push with all models (HBM variants PS, PW, PSE, PWE, and MPM) and compare with the measured result.
```bash
python scripts/model_eval.py --push --headless \
    --root $PUSH_IN_PATH \
    --aor $HBM_AOR_DEG \
    --erosion-stencil $HBM_STENCIL \
    $MPM_ARGS \
    --output-dir $RESULTS_DIR --analysis-name push_in

python scripts/model_eval.py --push --headless \
    --root $PUSH_OUT_PATH \
    --aor $HBM_AOR_DEG \
    --erosion-stencil $HBM_STENCIL \
    $MPM_ARGS \
    --output-dir $RESULTS_DIR --analysis-name push_out
```
Each run writes `SUMMARY_LONG_FORMAT.csv` and `SUMMARY_LEADERBOARD.csv` (IoU per model, for 1-step and 2-step predictions) into `$RESULTS_DIR/push_in` and `$RESULTS_DIR/push_out`. The HBM runtimes for the runtime comparison are written there too (`TIMINGS_PUSH_*.csv`).

Optional, plot the prediction fidelity of both experiments:
```bash
python plotting/plot_push_iou.py \
    --inputs $RESULTS_DIR/push_in $RESULTS_DIR/push_out \
    --tags $MATERIAL_NAME-in $MATERIAL_NAME-out \
    --models PS PW PSE PWE MPM \
    --out-dir $RESULTS_DIR/plots/push_iou
```

**In the paper:** Fig. 12 is this plot with all six push experiments (three materials, inward and outward): pass all six folders to `--inputs`, with one tag each.

## Runtime Comparison

> **In the paper:** Sec. V-C, *Runtime Comparison* (Fig. 13).

Needs the two push evaluations from the previous section (`$RESULTS_DIR/push_in` and `$RESULTS_DIR/push_out`). The HBM runtimes are already there (`TIMINGS_PUSH_LONG.csv`, `TIMINGS_PUSH_SUMMARY_MEAN.csv`). These files contain **only the HBM variants**: the MPM runtimes are measured separately (below) and stored per push in `mpm_push_staging/run_XX/taskY/timed_run/raw_telemetry.csv` (one row per simulation step). The MPM runs from the push evaluation are not suitable for timing: they also compute extra outputs (depth renders, kinetic energy). So each MPM push is simulated a second time with the same settings, with these outputs switched off.

Re-run every MPM push as a timed run. Each one writes `raw_telemetry.csv` into a `timed_run` folder next to its `data_from_mpm`. Run nothing else on the GPU meanwhile, so the times are not distorted.
```bash
for task_dir in $RESULTS_DIR/push_in/mpm_push_staging/*/task* \
                $RESULTS_DIR/push_out/mpm_push_staging/*/task*; do
  echo ">>> $task_dir"
  python granular_pushing/mpm/mpm_sim.py \
    --config $task_dir/data_from_mpm/resolved_sim_config.yaml \
    --set render_depth=false timed_run=true exp_name=timed_run log_directory=$task_dir \
    --viewer null
done
```

Collect the HBM and MPM runtimes and plot them:
```bash
python plotting/plot_push_times.py \
    --inputs $RESULTS_DIR/push_in $RESULTS_DIR/push_out \
    --tags ${MATERIAL_NAME}_Inward ${MATERIAL_NAME}_Outward \
    --out-dir $RESULTS_DIR/plots/runtime
```
The plots (PDF) and the runtimes per push (CSV) are written into `$RESULTS_DIR/plots/runtime/Comparison`. Only these CSVs (`Comparison/CSV_Data/`) have HBM and MPM side by side, one total time per push and model. The MPM time counts only the pushing motion itself (setup and settling are left out).

**In the paper:** Fig. 13 is this plot with all six push experiments (three materials, inward and outward): pass all six folders to `--inputs`, with one tag each.

## Clearance Planner

> **In the paper:** Sec. V-D, *Clearance Task* (Fig. 15).

The planner takes one scan of the pile and plans a sequence of pushes that clear the material out of a circular region. It plans **open-loop**: all pushes are planned in one go from that single scan, each on the map predicted by the HBM after the previous push. The robot then executes them one after the other. After every push the robot takes another scan; these scans are only used to score the result, not to re-plan.

In our data, every clearance run is one folder per material and model. Inside:
- `run_01/PRESWEEP_SCAN/`: the scan the planner used. `run_01/POSTSWEEP_SCAN_1` ... `_10`: the scans after each push. Each holds `point_cloud.ply` (cropped to the work area, in the workplane frame).
- `run_01/planner_archive/planning_session/`: what the planner did during the experiment: `actions_log.csv` (the planned pushes) and `physical_scan_001/clearance_scores.csv` (the volume left after each push, as computed during the experiment).

In both CSVs, the scan file name column (`source_filename`, `filename`) holds the names the scans had during the experiment (e.g. `scan_4_POSTSWEEP_SCAN_<timestamp>.ply`). In the published data, the scans are in folders named by tag, in the same order (`PRESWEEP_SCAN`, `POSTSWEEP_SCAN_1` ... `_10`).

The planner settings of each run (model, clearance region, candidate pushes) are in the repository, one file per run: `granular_pushing/planner/configs/<material>/<material>_<model>.yaml`. They differ only in the HBM parameters (AoR, dump angle, erosion iterations).

Point to one clearance run (relative to `DATA_ROOT`):
```bash
export CLEARANCE_PATH=$DATA_ROOT/<clearance_run_folder>
export CLEARANCE_CONFIG=granular_pushing/planner/configs/<material>/<material>_<model>.yaml
export CLEARANCE_RUN=$(ls -d $CLEARANCE_PATH/run_*)
export CLEARANCE_SESSION=$(ls -d $CLEARANCE_RUN/planner_archive/*planning_session)
```

<details><summary>Example: grapes, PWE (our data)</summary>

```bash
export CLEARANCE_PATH=$DATA_ROOT/Clearance_HBM/grapes/grapes_PWE
export CLEARANCE_CONFIG=granular_pushing/planner/configs/grapes/grapes_PWE.yaml
export CLEARANCE_RUN=$(ls -d $CLEARANCE_PATH/run_*)
export CLEARANCE_SESSION=$(ls -d $CLEARANCE_RUN/planner_archive/*planning_session)
```

</details>

### Re-run the planner on the recorded scan

Runs the planner offline (no robot) on the recorded PRESWEEP scan, with the exact settings of the experiment (`--config` is the config of this run, so the model and the clearance region are the same). The number of pushes to plan is `system.n_steps` in the config (10, one per `POSTSWEEP_SCAN_*` folder; `--n-steps` overrides it). Opens a live window, so run it with the GUI.
```bash
python scripts/run_planner.py \
    --offline-input $CLEARANCE_RUN/PRESWEEP_SCAN/point_cloud.ply \
    --config $CLEARANCE_CONFIG \
    --archive-root $RESULTS_DIR/planner
```
Writes a new `<timestamp>_planning_session` folder into `$RESULTS_DIR/planner`, with the same layout as the recorded one.

Its `actions_log.csv` should list the same pushes as the one recorded during the experiment (`$CLEARANCE_SESSION/actions_log.csv`).

### Clearance score and plot

For every scan of the run (PRESWEEP first, then each POSTSWEEP), compute the volume of material left inside the clearance region, with the same functions and settings as the planner. Writes `clearance_scores_recomputed.csv` (one row per scan, volume in m³).
```bash
python scripts/clearance_scores.py $CLEARANCE_PATH \
    --config $CLEARANCE_CONFIG \
    --output-dir $RESULTS_DIR/clearance/$(basename $CLEARANCE_PATH)
```
The planner also logged these volumes during the experiment (`$CLEARANCE_SESSION/physical_scan_001/clearance_scores.csv`), but rounded to 10 cm³, so small leftovers appear as 0. The recomputed values are not rounded.

Plot the remaining volume over the pushes:
```bash
python plotting/plot_clearance_score.py \
    --inputs $RESULTS_DIR/clearance/$(basename $CLEARANCE_PATH)/clearance_scores_recomputed.csv \
    --tags $MATERIAL_NAME/<model> \
    --out-dir $RESULTS_DIR/plots/clearance
```
`<model>` is the model of the run (PS, PW, PSE or PWE; for our example `grapes/PWE`).

**In the paper:** Fig. 15 is this plot for all 12 runs (three materials, four models): compute the scores for each run, then pass all 12 CSVs to `--inputs`, with one tag `<material>/<model>` each.

### Running the planner on a robot

The planner talks to the robot only through files, so it works with any robot stack. Our robot-side code (ROS, motion planning, scanning) is not part of this repository. The published data is offline data: during the experiments the robot delivered its scans into one flat folder (`system.ply_location`); in the published dataset they are sorted into scan folders as described above.

```bash
python scripts/run_planner.py \
    --data-root <run_folder> \
    --config granular_pushing/planner/configs/<material>/<material>_<model>.yaml \
    --active-root <shared_folder>
```

What happens:
1. The robot scans the pile and saves the point cloud (`.ply`, in the work-plane frame) into `<run_folder>/<system.ply_location>`, with `PRESWEEP` in the filename.
2. The planner sees the new file, plans `system.n_steps` pushes and writes them to `<shared_folder>/actions_log.csv`: one row per push, start and end point (`start_x, start_y, end_x, end_y`) in metres, work-plane frame.
3. The robot executes the pushes in order. After each one it saves a scan with `POSTSWEEP` in the filename. The planner scores it (volume left in the clearance region) and shows it next to its prediction in the live window.
4. Everything is archived in `<run_folder>/planner_archive/<timestamp>_planning_session/`.

Settings in the config:
- `hbm`: the model (AoR, erosion stencil and iterations, dump angle) and the pusher width. For a new material, copy one of the configs and set its AoR.
- `planner.clearance_region`: centre and radius of the region to clear.
- `planner.clearance.radial_config`: the candidate pushes (radial pushes around a centre point).
- `planner.start_bounds`: corners of the area for the 16 boundary test pushes at the start of `actions_log.csv` (`INIT_TEST`). `planner.end_bounds` is only used by the random planner (`planner_type: random`).
- `system`: number of pushes (`n_steps`) and folders (`active_root`, `archive_root`, `ply_location`).

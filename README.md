# Robotic Pushing of Granular Materials

This repository contains the code for our paper

> **Robotic Pushing of Granular Materials: Comparing Heightmap-based Models to MPM**  
> Nikhilesh Alatur, Helen Oleynikova, Giulio Schiavi, Roland Siegwart, Michael Pantic  
> IEEE Robotics and Automation Letters (RA-L), 2026

The experimental data is available on the ETH Research Collection: [doi.org/10.3929/ethz-c-000808988](https://doi.org/10.3929/ethz-c-000808988). Its structure is described in [DATA.md](docs/DATA.md).

Please note that this is the code we used to run the evaluations in the paper, not a standalone simulation library with a stable API. However, the heightmap-based model in `granular_pushing/hbm/hbm_model.py` and the MPM pipeline in `granular_pushing/mpm/` (using [Newton](https://github.com/newton-physics/newton)) are self-contained enough to serve as a starting point for your own projects.

## Setup

### Prerequisites

- Linux with an NVIDIA GPU and a driver supporting CUDA 12 (driver 525 or newer)
- [Docker Engine](https://docs.docker.com/engine/install/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), configured for Docker

Tested on:

| | |
|---|---|
| OS | Ubuntu 20.04 |
| GPU | NVIDIA RTX 3080 |
| CPU / RAM | Intel i9-11900KF / 32 GB |
| NVIDIA driver | 550.54.14 (CUDA 12.4) |

### Docker

Everything runs in an Ubuntu 22.04 Docker container with CUDA and Newton.

```bash
git clone https://github.com/ethz-asl/granular-pushing.git
cd granular-pushing
docker/run.sh
```

The first call builds the image (a few minutes) and then opens a shell in the container. To check that it sees your GPU, run `python -c "import warp as wp; wp.init(); print(wp.get_cuda_devices())"` in that shell; it should print e.g. `['cuda:0']`.

| Command | What it does |
|---|---|
| `docker/run.sh` | Starts the container and opens a shell in it (`/workspace` = this repository). Further calls, e.g. from other terminals, attach to the same container. GUI windows (Open3D, Newton viewer) appear on your desktop if started from a desktop session. |
| `docker/run.sh stop` | Stops the container. Nothing is lost: code, data and outputs live in the mounted repository. |

### Software versions

The paper's results were produced with Newton commit `e5f6493` on the Warp nightly build `1.12.0.dev20260115`. As that build is no longer available, this repository uses the release `warp-lang 1.12.0`, with which we verified that the results are reproduced.

## Reproducing the results

1. Download the dataset from the [ETH Research Collection](https://doi.org/10.3929/ethz-c-000808988) and unpack it into `data/` ([DATA.md](docs/DATA.md)).
2. Follow [REPRODUCE.md](docs/REPRODUCE.md), which walks through every result of the paper step by step: model calibration, push prediction, runtimes and the clearance planner.

## Citation

If you use this code or the data in your work, please cite:

```bibtex
@article{alatur2026granular,
  title   = {Robotic Pushing of Granular Materials: Comparing Heightmap-based Models to {MPM}},
  author  = {Alatur, Nikhilesh and Oleynikova, Helen and Schiavi, Giulio and Siegwart, Roland and Pantic, Michael},
  journal = {IEEE Robotics and Automation Letters},
  year    = {2026}
}
```

## Use of AI

We used Gemini and Claude as coding assistants while developing this code.

## License

This code is released under the Apache License 2.0, see [LICENSE](LICENSE).

# Environment for granular-pushing: Python 3.11, Newton (MPM) on warp, and the analysis packages.
# Only pyproject.toml and uv.lock go into the image; the repo itself is mounted at runtime (docker/run.sh).
FROM nvidia/cuda:12.0.1-base-ubuntu22.04

# System libraries: git, OpenGL/OpenMP runtime for Open3D,
# and the X11/OpenGL client libraries for GUI windows (Open3D, Newton viewer), shown via docker/run.sh.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git ca-certificates libgl1 libgomp1 libglib2.0-0 \
        libegl1 libglu1-mesa libxrandr2 libxinerama1 libxcursor1 libxi6 libxxf86vm1 \
    && rm -rf /var/lib/apt/lists/*

# uv manages Python and all packages.
COPY --from=ghcr.io/astral-sh/uv:0.11.32 /uv /uvx /usr/local/bin/

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PATH=/opt/venv/bin:$PATH

# Build the environment once, into the image. Rebuilt only when pyproject.toml or uv.lock change.
WORKDIR /opt/env
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project \
    && chmod -R a+rX /opt/venv /opt/python

WORKDIR /workspace
# Module folders on the path, so files import each other by plain name (e.g. `from mpm_utils import ...`).
ENV PYTHONPATH=/workspace/granular_pushing/hbm:/workspace/granular_pushing/mpm:/workspace/granular_pushing/eval:/workspace/granular_pushing/planner
CMD ["bash"]

#!/usr/bin/env bash
# One long-running container per machine; every call works inside it.
#   docker/run.sh                 -> shell in the container (starts it if needed; more terminals join the same one)
#   docker/run.sh python x.py     -> run a command in the container
#   docker/run.sh stop            -> stop and remove the container
# The container keeps running after you exit the shell, until `docker/run.sh stop` or a reboot.
# Nothing is lost when it stops: the repo (code, data/, outputs/, caches in .home/) lives on the host.
set -e

# What it does, step by step:
#   1. Build the image from docker/Dockerfile (cached; rebuilt only when dependencies change).
#   2. If no container is running (or it runs an older image), start one in the background with:
#      GPU access, your own user (files in outputs/ belong to you), HOME=/workspace/.home for caches,
#      the X socket for GUI windows, and the repo mounted at /workspace (plus DATA_DIR, if set).
#   3. Open a shell (or run the given command) in that container, with this terminal's DISPLAY.

REPO="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE=granular-pushing
NAME=granular-pushing

if [ "$1" = "stop" ]; then
    docker rm -f "$NAME" > /dev/null 2>&1 || true
    echo "Container $NAME stopped."
    exit 0
fi

# Build the image (cached: seconds; first time or after a dependency change: a few minutes).
if ! docker image inspect "$IMAGE" > /dev/null 2>&1; then
    echo "Building image $IMAGE (first time: a few minutes)..."
fi
DOCKER_BUILDKIT=1 docker build -q -t "$IMAGE" -f "$REPO/docker/Dockerfile" "$REPO" > /dev/null

# If the container runs an older image (dependencies changed), restart it.
RUNNING_IMAGE="$(docker inspect -f '{{.Image}}' "$NAME" 2> /dev/null || true)"
CURRENT_IMAGE="$(docker image inspect -f '{{.Id}}' "$IMAGE")"
if [ -n "$RUNNING_IMAGE" ] && [ "$RUNNING_IMAGE" != "$CURRENT_IMAGE" ]; then
    echo "Image changed: restarting container $NAME (other open shells in it will close)."
    docker rm -f "$NAME" > /dev/null
    RUNNING_IMAGE=""
fi

if [ -z "$RUNNING_IMAGE" ]; then
    mkdir -p "$REPO/.home"

    # --user drops your supplementary groups. Some hosts restrict /dev/nvidia* to a group
    # (e.g. root:1001, mode 660), so pass the GPU device's group on explicitly.
    GPU_GROUP=()
    [ -e /dev/nvidia0 ] && GPU_GROUP=(--group-add "$(stat -c %g /dev/nvidia0)")

    # GUI: always mount the X socket folder, so a shell attached later from a desktop or VNC session
    # can open windows, even if the container was started from a plain CLI/ssh session.
    X11=()
    [ -d /tmp/.X11-unix ] && X11=(-v /tmp/.X11-unix:/tmp/.X11-unix:ro)

    # Optional: mount an existing data folder as /workspace/data, e.g. DATA_DIR=~/old/data docker/run.sh
    DATA=()
    [ -n "$DATA_DIR" ] && DATA=(-v "$(realpath "$DATA_DIR"):/workspace/data")

    # Run as your own user so files written to outputs/ belong to you.
    # HOME inside the repo (gitignored .home/) keeps caches between runs, e.g. warp's compiled kernels.
    # The host's passwd/group files (read-only) give your user and groups their names inside the container.
    # NVIDIA_DRIVER_CAPABILITIES=all also gives the container the driver's OpenGL libraries (GPU-rendered GUI).
    docker run -d --rm --name "$NAME" --gpus all \
        -e NVIDIA_DRIVER_CAPABILITIES=all \
        --user "$(id -u):$(id -g)" "${GPU_GROUP[@]}" \
        -e HOME=/workspace/.home \
        -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro \
        "${X11[@]}" "${DATA[@]}" \
        -v "$REPO":/workspace \
        "$IMAGE" sleep infinity > /dev/null
    echo "Started container $NAME. It keeps running; stop it with: docker/run.sh stop"
fi

# Each shell gets the display of the terminal it was opened from (desktop, VNC, or none over ssh).
GUI=()
if [ -n "$DISPLAY" ]; then
    xhost "+SI:localuser:$(id -un)" > /dev/null 2>&1 || true
    GUI=(-e DISPLAY="$DISPLAY")
fi

[ $# -eq 0 ] && set -- bash
TTY=""; [ -t 0 ] && TTY="-it"
exec docker exec $TTY "${GUI[@]}" "$NAME" "$@"

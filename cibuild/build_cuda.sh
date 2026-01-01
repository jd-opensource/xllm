#!/bin/bash
set -e

function error() {
  echo "Require build command, e.g. python setup.py build --device cuda"
  exit 1
}

IMAGE="quay.io/jd_xllm/xllm-ai:xllm-dev-cuda-x86"

RUN_OPTS=(
  --rm
  -t
  --privileged
  --ipc=host
  --network=host
  --pid=host
  --shm-size '128gb'
  -v /export/home:/export/home
  -v /export/home/cuda_vcpkg_cache:/root/.cache/vcpkg # cached vcpkg installed dir
  -w /export/home
)

CMD="$*"
[[ -z "${CMD}" ]] && error

[[ ! -x $(command -v docker) ]] && echo "ERROR: 'docker' command is missing." && exit 1

docker run "${RUN_OPTS[@]}" "${IMAGE}" bash -c "set -euo pipefail; cd $(pwd); ${CMD}"

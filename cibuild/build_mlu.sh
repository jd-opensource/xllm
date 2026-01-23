#!/bin/bash
set -e

function error() {
  echo "Require build command, e.g. python setup.py build"
  exit 1
}

IMAGE="xllm:v25.12.0-torch2.9.1-torchmlu1.30.2-ubuntu22.04-py310-xllm-x86"

RUN_OPTS=(
  --rm
  -t
  --privileged
  --ipc=host
  --network=host
  --pid=host
  --shm-size '128gb'
  -v /export/home:/export/home
  -v /usr/bin/cnmon:/usr/bin/cnmon
  -v /export/home/mlu_vcpkg_cache:/root/.cache/vcpkg # cached vcpkg installed dir
  -w /export/home
)

CMD="$*"
[[ -z "${CMD}" ]] && error

[[ ! -x $(command -v docker) ]] && echo "ERROR: 'docker' command is missing." && exit 1

docker run "${RUN_OPTS[@]}" "${IMAGE}" bash -c "set -euo pipefail; cd $(pwd); ${CMD}"

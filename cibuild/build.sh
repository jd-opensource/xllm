#!/bin/bash
set -e

function error() {
  echo "Require build command, e.g. python setup.py build"
  exit 1
}

IMAGE="quay.io/jd_xllm/xllm-ai:0.6.0-dev-hb-rc2-x86"

RUN_OPTS=(
  --rm
  -t
  --privileged
  --ipc=host
  --network=host
  --device=/dev/davinci0
  --device=/dev/davinci_manager
  --device=/dev/devmm_svm
  --device=/dev/hisi_hdc
  -v /var/queue_schedule:/var/queue_schedule
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver
  -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi
  -v /usr/local/sbin/:/usr/local/sbin/
  -v /export/home:/export/home
  -v /export/home/vcpkg-cache:/root/.cache/vcpkg
  -w /export/home
)

CMD="$*"
[[ -z "${CMD}" ]] && error

[[ ! -x $(command -v docker) ]] && echo "ERROR: 'docker' command is missing." && exit 1

docker run "${RUN_OPTS[@]}" "${IMAGE}" bash -c "set -euo pipefail; cd $(pwd); ${CMD}"

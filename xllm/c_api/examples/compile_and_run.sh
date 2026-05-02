#!/usr/bin/env bash
# 使用工程内构建的 libxllm.so 编译 multi_rec_completions 并运行（参数原样传给可执行文件）。
set -euo pipefail

XLLM_REPO_ROOT="${XLLM_REPO_ROOT:-/export/home/liuhan37/xllm-so/xllm}"
LIB_DIR="${XLLM_REPO_ROOT}/build/xllm/core/server"
LIB_SO="${LIB_DIR}/libxllm.so"
C_API_INCLUDE="${XLLM_REPO_ROOT}/xllm/c_api"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if [[ ! -f "${LIB_SO}" ]]; then
  echo "error: ${LIB_SO} not found. Build the project first, e.g.:" >&2
  echo "  cd ${XLLM_REPO_ROOT} && python setup.py build --generate-so true" >&2
  exit 1
fi

g++ multi_rec_completions.cpp -o multi_rec_completions \
  -I/usr/local/xllm/include \
  -I"${C_API_INCLUDE}" \
  -L"${LIB_DIR}" -lxllm -Wl,-rpath="${LIB_DIR}"

export ASCEND_RT_VISIBLE_DEVICES=0,1

exec ./multi_rec_completions "$@"
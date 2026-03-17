# MiniMax-M2.5 CANN 8.5 Baseline

Updated: 2026-03-17

This note records the current MiniMax-M2.5 baseline on the rebased CANN 8.5 /
PyTorch-NPU 2.7.1 branch.

## 1. Main Changes Against `upstream/main`

- Native MiniMax model support was added under
  `xllm/models/llm/npu/minimax_m2.h` and wired into model registration.
- MiniMax-specific NPU decoder integration was added under
  `xllm/core/layers/npu/npu_minimax_m2_decoder_layer_impl.*` and
  `xllm/core/layers/npu/loader/minimax_m2_decoder_loader.*`.
- MiniMax router and MoE behavior were rebuilt on the NPU torch path in
  `xllm/core/layers/npu_torch/fused_moe.*`, including MiniMax-specific routing
  semantics and grouped-vs-reference debug hooks.
- ACL-graph decode integration for MiniMax was extended in
  `xllm/core/runtime/acl_graph_executor_impl.*`,
  `xllm/core/distributed_runtime/llm_engine.cpp`,
  `xllm/core/runtime/worker_impl.cpp`, and
  `xllm/core/scheduler/profile/profile_manager.*`.
- BF16-friendly streaming checkpoint load was added in
  `xllm/core/framework/hf_model_loader.*` so the bf16 checkpoint does not keep
  all shards resident at once during startup.
- An offline fp8-to-bf16 checkpoint converter was added at
  `tools/dequant_minimax_fp8.py`.
- Multi-rank startup on the new stack was stabilized with:
  `xllm/core/distributed_runtime/remote_worker.cpp`,
  `xllm/core/common/global_flags.cpp`,
  `xllm/core/distributed_runtime/collective_service.cpp`, and
  `xllm/core/util/net.cpp`.

## 2. Start Script And Settings

- The current local runtime baseline uses:
  `MODEL_PATH=/models/MiniMax-M2.5-bf16`
  `DP_SIZE=1`
  `EP_SIZE=1`
  `NNODES=16`
  `XLLM_MINIMAX_EP_MOE_REFERENCE=0`
  `XLLM_MINIMAX_NATIVE_DECODE_ATTN=1`
  `XLLM_MINIMAX_NATIVE_DECODE_MOE=1`
  `XLLM_GRAPH_WARMUP_PREFILL_TOKENS=1024`
  `--enable_graph=true`
- This is effectively pure `tp=16` for both attention and MoE:
  `attn_tp=16`, `moe_tp=16`.
- The correctness-first EP debug setting is still:
  `MODEL_PATH=/models/MiniMax-M2.5-bf16`
  `DP_SIZE=2`
  `EP_SIZE=2`
  `XLLM_MINIMAX_EP_MOE_REFERENCE=1`
- The bf16 checkpoint is the preferred runtime checkpoint because it matches
  the fp8-dequant runtime path but starts much faster than the original fp8
  checkpoint.

### Reproduce The Baseline

Below is the baseline starting script:

```bash
#!/bin/bash

set -eo pipefail

cleanup_root_log_dir() {
  local path="$1"
  rm -rf "$path" 2>/dev/null || sudo -n rm -rf "$path" 2>/dev/null || true
}

##### 1，配置依赖路径相关环境变量
export PYTHON_INCLUDE_PATH="$(python3 -c 'from sysconfig import get_paths; print(get_paths()["include"])')"
export PYTHON_LIB_PATH="$(python3 -c 'from sysconfig import get_paths; print(get_paths()["include"])')"
export PYTORCH_NPU_INSTALL_PATH=/usr/local/libtorch_npu/
export PYTORCH_INSTALL_PATH="$(python3 -c 'import torch, os; print(os.path.dirname(os.path.abspath(torch.__file__)))')"
export LIBTORCH_ROOT="$(python3 -c 'import torch, os; print(os.path.dirname(os.path.abspath(torch.__file__)))')"

export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/xllm/op_api/lib/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/libtorch_npu/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=/usr/lib64/libjemalloc.so.2:$LD_PRELOAD

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

##### 2，配置日志相关环境变量
cleanup_root_log_dir /root/atb/log/
cleanup_root_log_dir /root/ascend/log/

rm -rf core.*
export GLOG_logtostderr=1
export GLOG_logbufsecs=0
export ASDOPS_LOG_LEVEL=ERROR
export ASDOPS_LOG_TO_STDOUT=1
export ASDOPS_LOG_TO_FILE=1

##### 3，配置性能、通信相关环境变量
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export NPU_MEMORY_FRACTION=0.96
export ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE=3
export ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1

export OMP_NUM_THREADS=12
export ALLOW_INTERNAL_FORMAT=1

export ATB_LAYER_INTERNAL_TENSOR_REUSE=1
export ATB_LLM_ENABLE_AUTO_TRANSPOSE=0
export ATB_CONVERT_NCHW_TO_AND=1
export ATB_LAUNCH_KERNEL_WITH_TILING=1
export ATB_OPERATION_EXECUTE_ASYNC=2
export ATB_CONTEXT_WORKSPACE_SIZE=0
export INF_NAN_MODE_ENABLE=1
export HCCL_EXEC_TIMEOUT=0
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_IF_BASE_PORT=2864

###############
# 启动参数
###############

BATCH_SIZE=${BATCH_SIZE:-256}
MAX_TOKENS_PER_BATCH=${MAX_TOKENS_PER_BATCH:-32768}
MODEL_PATH=${MODEL_PATH:-"/models/MiniMax-M2.5-bf16"}
MODEL_ID=${MODEL_ID:-"minimax-m2.5"}
XLLM_PATH=./build/xllm/core/server/xllm

## 单机默认自动探测本机IP；若跨机部署请显式导出 LOCAL_HOST / MASTER_NODE_ADDR
#LOCAL_HOST=${LOCAL_HOST:-"$(hostname -I 2>/dev/null | awk '{print $1}')"}
#if [[ -z "${LOCAL_HOST}" ]]; then
#  LOCAL_HOST="127.0.0.1"
#fi
LOCAL_HOST=${LOCAL_HOST:-"127.0.0.1"}

START_PORT=${START_PORT:-18994}
MASTER_PORT=${MASTER_PORT:-28994}
MASTER_NODE_ADDR=${MASTER_NODE_ADDR:-"${LOCAL_HOST}:${MASTER_PORT}"}
START_DEVICE=0
LOG_DIR=${LOG_DIR:-"logs"}
NNODES=${NNODES:-16}
# Default to the current CANN 8.5 MiniMax bf16 baseline: pure TP16 with ACL
# graph enabled. EP correctness debugging can still override these from the
# shell before invoking the script.
DP_SIZE=${DP_SIZE:-1}
EP_SIZE=${EP_SIZE:-1}
EXPERT_PARALLEL_DEGREE=${EXPERT_PARALLEL_DEGREE:-0}

if (( NNODES % DP_SIZE != 0 )); then
  echo "NNODES=$NNODES must be divisible by DP_SIZE=$DP_SIZE" >&2
  exit 1
fi
if (( NNODES % EP_SIZE != 0 )); then
  echo "NNODES=$NNODES must be divisible by EP_SIZE=$EP_SIZE" >&2
  exit 1
fi

ATTN_TP_SIZE=$((NNODES / DP_SIZE))
MOE_TP_SIZE=$((NNODES / EP_SIZE))
echo "Parallel config: dp=${DP_SIZE}, ep=${EP_SIZE}, attn_tp=${ATTN_TP_SIZE}, moe_tp=${MOE_TP_SIZE}, expert_parallel_degree=${EXPERT_PARALLEL_DEGREE}, max_tokens_per_batch=${MAX_TOKENS_PER_BATCH}, warmup_prefill_tokens=${XLLM_GRAPH_WARMUP_PREFILL_TOKENS:-1024}"

# feat switch
export XLLM_MINIMAX_NATIVE_DECODE_ATTN=${XLLM_MINIMAX_NATIVE_DECODE_ATTN:-1}
export XLLM_MINIMAX_NATIVE_DECODE_MOE=${XLLM_MINIMAX_NATIVE_DECODE_MOE:-1}
# TP16 baseline keeps EP disabled, so leave the reference EP path off by
# default. When debugging EP correctness specifically, override this to `1`
# together with `DP_SIZE=2 EP_SIZE=2`.
export XLLM_MINIMAX_EP_MOE_REFERENCE=${XLLM_MINIMAX_EP_MOE_REFERENCE:-0}
export XLLM_MINIMAX_NATIVE_DECODE_MOE_MIN_BATCH=${XLLM_MINIMAX_NATIVE_DECODE_MOE_MIN_BATCH:-8}
export XLLM_MINIMAX_WARMUP_DECODE_BUCKETS=${XLLM_MINIMAX_WARMUP_DECODE_BUCKETS:-1}
export XLLM_GRAPH_WARMUP_PREFILL_TOKENS=${XLLM_GRAPH_WARMUP_PREFILL_TOKENS:-1024}
export XLLM_STREAM_HF_STATE_DICT_LOAD=${XLLM_STREAM_HF_STATE_DICT_LOAD:-1}

mkdir -p "$LOG_DIR"

for (( i=0; i<$NNODES; i++ ))
do
  PORT=$((START_PORT + i))
  DEVICE=$((START_DEVICE + i))
  LOG_FILE="$LOG_DIR/node_$i.log"
  nohup numactl -C $((i*40))-$((i*40+39)) "$XLLM_PATH" \
    --model "$MODEL_PATH" \
    --model_id "$MODEL_ID" \
    --host "$LOCAL_HOST" \
    --port "$PORT" \
    --devices="npu:$DEVICE" \
    --master_node_addr="$MASTER_NODE_ADDR" \
    --nnodes="$NNODES" \
    --node_rank="$i" \
    --dp_size="$DP_SIZE" \
    --ep_size="$EP_SIZE" \
    --expert_parallel_degree="$EXPERT_PARALLEL_DEGREE" \
    --max_memory_utilization=0.86 \
    --max_tokens_per_batch="$MAX_TOKENS_PER_BATCH" \
    --max_seqs_per_batch="$BATCH_SIZE" \
    --communication_backend=hccl \
    --enable_chunked_prefill=true \
    --enable_prefix_cache=true \
    --enable_schedule_overlap=true \
    --enable_graph=true \
    --enable_atb_spec_kernel=false \
    > $LOG_FILE 2>&1 &
done

echo "MiniMax server launch commands submitted. Logs in: $LOG_DIR"
```

## 3. Performance And Correctness

- Pure `tp=16` bf16 is the current baseline for performance work.
- Measured single-request result on the current graph-enabled TP16 path:
  prompt: `Write a short two-line poem about graphs.`
  `max_tokens=64`
  `ttft: 4595.0ms`
  `avg tpot: 18.2ms`
  `generation speed: 55.9 tok/s`
- The TP16 path completed ACL graph warmup and captured decode buckets
  `1/2/4/8/16` without the idle-DP fallback seen in `dp=2, ep=2`.
- The TP16 output was coherent reasoning-style text, not garbage bytes.
- `dp=2, ep=2` grouped EP MoE with ACL graph enabled is still not acceptable:
  the same small live request produced garbage output and only about `7 tok/s`.
- `dp=2, ep=2` reference EP MoE remains the correctness-first path, but ACL
  graph is disabled there and single-request decode is much slower.
- The remaining high-priority work is:
  keep ACL graph on while preserving correctness,
  support full 192k context,
  keep chunked prefill and prefix cache healthy,
  and make the performance target hold on the intended serving topology.

## 4. Offline FP8 -> BF16 Dequant Script

- Script path: `tools/dequant_minimax_fp8.py`
- Purpose: convert `/models/MiniMax-M2.5` into a reusable bf16 checkpoint cache
  so server restarts avoid repeated fp8 dequant during load.
- Usage:

```bash
python tools/dequant_minimax_fp8.py \
  --input-dir /models/MiniMax-M2.5 \
  --output-dir /models/MiniMax-M2.5-bf16
```

- The script:
  reads MiniMax fp8 safetensors,
  applies the per-block `weight_scale_inv`,
  writes bf16 safetensors,
  updates `config.json`,
  and regenerates `model.safetensors.index.json`.

#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUNNER_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$RUNNER_ROOT/.." && pwd)"

export COREDUMP_DIR="$SCRIPT_DIR/core_dump"

\rm -rf $COREDUMP_DIR
mkdir -p "$COREDUMP_DIR"
ulimit -c unlimited

if [ -w /proc/sys/kernel/core_pattern ]; then
    echo "$COREDUMP_DIR/core.%e.%p.%t" > /proc/sys/kernel/core_pattern
fi

# =============================================================
# Fixed Env
# =============================================================
export PYTHON_INCLUDE_PATH="$(python3 -c 'from sysconfig import get_paths; print(get_paths()["include"])')"
export PYTHON_LIB_PATH="$(python3 -c 'from sysconfig import get_paths; print(get_paths()["include"])')"
export PYTORCH_NPU_INSTALL_PATH=/usr/local/libtorch_npu/
export PYTORCH_INSTALL_PATH="$(python3 -c 'import torch, os; print(os.path.dirname(os.path.abspath(torch.__file__)))')"
export LIBTORCH_ROOT="$PYTORCH_INSTALL_PATH"
export LD_LIBRARY_PATH=/usr/local/libtorch_npu/lib:$LD_LIBRARY_PATH
export ATB_RUNNER_POOL_SIZE=64
export ASDOPS_LOG_TO_STDOUT=1
export ASDOPS_LOG_LEVEL=ERROR
export ASDOPS_LOG_TO_FILE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export NPU_MEMORY_FRACTION=0.98
export ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE=3
export ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1
export ATB_STREAM_SYNC_EVERY_OPERATION_ENABLE=0
export OMP_NUM_THREADS=12
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_IF_BASE_PORT=43432



usage() {
    cat <<'EOF'
Usage:
  run.sh <prefill_nodes> <decode_nodes> [start_store] [store_force_restart] [store_mode]

Args:
  prefill_nodes        Number of PREFILL workers.
  decode_nodes         Number of DECODE workers.
  start_store          Optional, default: 1 (1/0, true/false).
  store_force_restart  Optional, default: 1 (1/0, true/false).
  store_mode           Optional, default: rh2d (rh2d/rh2h).
EOF
}

log() {
    echo "[run_demo] $*"
}

die() {
    echo "[run_demo][ERROR] $*" >&2
    exit 1
}

to_bool01() {
    case "${1:-}" in
        1|true|TRUE|yes|YES)
            echo "1"
            ;;
        0|false|FALSE|no|NO)
            echo "0"
            ;;
        *)
            die "Invalid boolean value: $1 (use 0/1 or true/false)"
            ;;
    esac
}

require_file() {
    [ -f "$1" ] || die "File not found: $1"
}

require_path() {
    [ -e "$1" ] || die "Path not found: $1"
}

require_exec() {
    [ -x "$1" ] || die "Executable not found: $1"
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    usage
    exit 0
fi

PREFILL_NODES="${1:-}"
DECODE_NODES="${2:-}"
START_STORE_RAW="${3:-1}"
STORE_FORCE_RESTART_RAW="${4:-1}"
STORE_MODE_RAW="${5:-rh2d}"

[ -n "$PREFILL_NODES" ] || die "prefill_nodes is required"
[ -n "$DECODE_NODES" ] || die "decode_nodes is required"
[[ "$PREFILL_NODES" =~ ^[0-9]+$ ]] || die "prefill_nodes must be a non-negative integer"
[[ "$DECODE_NODES" =~ ^[0-9]+$ ]] || die "decode_nodes must be a non-negative integer"

START_STORE="$(to_bool01 "$START_STORE_RAW")"
STORE_FORCE_RESTART="$(to_bool01 "$STORE_FORCE_RESTART_RAW")"
STORE_MODE="$(printf '%s' "$STORE_MODE_RAW" | tr '[:upper:]' '[:lower:]')"
case "$STORE_MODE" in
    rh2d|rh2h)
        ;;
    *)
        die "Invalid store_mode: $STORE_MODE_RAW (use rh2d/rh2h)"
        ;;
esac

# =============================================================
# Manual Config Section
# Change these values directly for your machine/environment.
# Keep parameters grouped by component.
# =============================================================
PYTHON_BIN="python3"

# [deployment ip]
# LOCAL_NODE_IP: current machine IP (where this script runs).
# GLOBAL_MASTER_IP: machine IP hosting global-unique components:
#   etcd + mooncake master + memfabric config store.
# Single-machine deployment: keep them the same.
# Split deployment: only change GLOBAL_MASTER_IP.
LOCAL_NODE_IP="127.0.0.1"
GLOBAL_MASTER_IP="$LOCAL_NODE_IP"

# [xllm worker binary + model]
XLLM_BIN="$REPO_ROOT/xllm/build/xllm/core/server/xllm"
MODEL_PATH="$REPO_ROOT/models/Qwen3-32B"
START_DEVICE=0
START_PORT=8100
CLIENT_START_PORT=8200
TRANSFER_START_PORT=8300

# [xllm-service binary + ports]
XLLM_SERVICE_BIN="$REPO_ROOT/xllm-service/build/xllm_service/xllm_master_serving"
SERVICE_HTTP_PORT=8501
SERVICE_RPC_PORT=8502

# [etcd]
ETCD_RUN_SCRIPT="$SCRIPT_DIR/etcd/run.sh"
ETCD_ADDR="$GLOBAL_MASTER_IP:8400"

# [kvcache store]
STORE_BASE_DIR="$SCRIPT_DIR/kvcache_store"
STORE_RUN_SCRIPT="$STORE_BASE_DIR/run_store.sh"
STORE_CONFIG_RH2D="$STORE_BASE_DIR/client-rh2d.json"
STORE_CONFIG_RH2H="$STORE_BASE_DIR/client-rh2h.json"
STORE_MASTER_BIN="/usr/local/bin/mooncake_master"
STORE_CLIENT_SCRIPT="$STORE_BASE_DIR/distributed_client.py"
STORE_LOG_DIR="$STORE_BASE_DIR"
STORE_HOST_IP="$LOCAL_NODE_IP"
STORE_PROTOCOL="ub"
STORE_METADATA_SERVER="P2PHANDSHAKE"
STORE_MASTER_SERVER_ADDRESS="$GLOBAL_MASTER_IP:8000"
MF_STORE_URL="$GLOBAL_MASTER_IP:7777"
STORE_LOCAL_HOSTNAME="$LOCAL_NODE_IP:8002"

# [logs]
LOG_DIR="$SCRIPT_DIR/logs"

# [xllm distributed addresses]
PREFILL_DISAGG_PD_PORT=9110
DECODE_DISAGG_PD_PORT=9220
PREFILL_MASTER_NODE_ADDR="$STORE_HOST_IP:8600"
DECODE_MASTER_NODE_ADDR="$STORE_HOST_IP:8700"

# [feature flags]
ENABLE_STORE=true
CACHE_UPLOAD=true
PREFIX_CACHE=true
SCHEDULE_OVERLAP=true
CHUNKED_PREFILL=false
# =============================================================

# store_mode-derived defaults.
STORE_CONFIG=""
STORE_RUNTIME_CONFIG="$STORE_LOG_DIR/client-runtime-${STORE_MODE}.json"
MF_USE_LOCAL_HOST_ENV=0
FACTOR=0
LAYERS_WISE_BATCHS=1
case "$STORE_MODE" in
    rh2h)
        STORE_CONFIG="$STORE_CONFIG_RH2H"
        MF_USE_LOCAL_HOST_ENV=1
        FACTOR=3
        LAYERS_WISE_BATCHS=4
        ;;
    rh2d)
        STORE_CONFIG="$STORE_CONFIG_RH2D"
        MF_USE_LOCAL_HOST_ENV=0
        FACTOR=0
        LAYERS_WISE_BATCHS=1
        ;;
esac

if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
    # shellcheck disable=SC1091
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi

if [ -f /usr/local/memfabric_hybrid/set_env.sh ]; then
    # shellcheck disable=SC1091
    source /usr/local/memfabric_hybrid/set_env.sh
    export MEMCACHE_ROOT=/usr/local/memfabric_hybrid/latest/aarch64-linux/
    export MF_OP_TYPE=device_sdma
    export MF_STORE_URL="$MF_STORE_URL"
    export MF_DRAM_SIZE=0
    export MF_MAX_DRAM_SIZE=268435456000
    export MF_LOG_LEVEL=1
    export MF_USE_LOCAL_HOST="$MF_USE_LOCAL_HOST_ENV"
fi

export ASCEND_RT_VISIBLE_DEVICES="0,1,2,3,4"


unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

mkdir -p "$LOG_DIR"
mkdir -p "$STORE_LOG_DIR"

require_exec "$XLLM_BIN"
require_path "$MODEL_PATH"
require_file "$ETCD_RUN_SCRIPT"
require_file "$STORE_RUN_SCRIPT"
require_file "$STORE_CONFIG"


start_store_if_needed() {
    if [ "$START_STORE" != "1" ]; then
        log "Skip store startup (start_store=$START_STORE)."
        return
    fi

    require_file "$STORE_CLIENT_SCRIPT"

    log "Starting store via run_store.sh ..."
    MOONCAKE_MASTER_BIN="$STORE_MASTER_BIN" \
    CLIENT_SCRIPT="$STORE_CLIENT_SCRIPT" \
    LOG_DIR="$STORE_LOG_DIR" \
    sh "$STORE_RUN_SCRIPT" "$STORE_CONFIG" both "$STORE_FORCE_RESTART"
}

stop_old_xllm_processes() {
    pkill -9 xllm >/dev/null 2>&1 || true
    sleep 2
}

run_etcd_cleanup() {
    log "Running etcd helper script: $ETCD_RUN_SCRIPT"
    (
        cd "$(dirname "$ETCD_RUN_SCRIPT")"
        sh "$(basename "$ETCD_RUN_SCRIPT")"
    )
}

start_xllm_service() {
    log "Starting xllm-service ..."
    ENABLE_DECODE_RESPONSE_TO_SERVICE=true \
    nohup "$XLLM_SERVICE_BIN" \
        --etcd_addr="$ETCD_ADDR" \
        --http_server_port="$SERVICE_HTTP_PORT" \
        --rpc_server_port="$SERVICE_RPC_PORT" \
        --tokenizer_path="$MODEL_PATH" \
        --load_balance_policy="CAR" >"$LOG_DIR/service.log" 2>&1 &
    log "xllm-service pid=$!, log=$LOG_DIR/service.log"
}

start_prefill_workers() {
    log "Starting $PREFILL_NODES prefill workers ..."
    for ((i = 0; i < PREFILL_NODES; i++)); do
        port=$((START_PORT + i))
        client_port=$((CLIENT_START_PORT + i))
        device=$((START_DEVICE + i))
        transfer_port=$((TRANSFER_START_PORT + i))
        node_log_dir="$LOG_DIR/prefill_$i"
        node_log_file="$LOG_DIR/prefill_$i.log"

        rm -rf "$node_log_dir"
        mkdir -p "$node_log_dir"

        nohup "$XLLM_BIN" \
            --model "$MODEL_PATH" \
            --devices="npu:$device" \
            --port "$port" \
            --nnodes="$PREFILL_NODES" \
            --max_memory_utilization=0.8 \
            --max_tokens_per_batch=100000 \
            --max_seqs_per_batch=256 \
            --enable_mla=true \
            --block_size=128 \
            --communication_backend="hccl" \
            --enable_prefix_cache="$PREFIX_CACHE" \
            --enable_chunked_prefill="$CHUNKED_PREFILL" \
            --enable_schedule_overlap="$SCHEDULE_OVERLAP" \
            --node_rank="$i" \
            --host_blocks_factor="$FACTOR" \
            --enable_disagg_pd=true \
            --instance_role=PREFILL \
            --disagg_pd_port="$PREFILL_DISAGG_PD_PORT" \
            --etcd_addr="$ETCD_ADDR" \
            --transfer_listen_port="$transfer_port" \
            --master_node_addr="$PREFILL_MASTER_NODE_ADDR" \
            --enable_cache_upload="$CACHE_UPLOAD" \
            --enable_kvcache_store="$ENABLE_STORE" \
            --store_protocol="$STORE_PROTOCOL" \
            --store_metadata_server="$STORE_METADATA_SERVER" \
            --store_master_server_address="$STORE_MASTER_SERVER_ADDRESS" \
            --store_local_hostname="$STORE_HOST_IP:$client_port" \
            --logtostderr=false \
            --prefetch_timeout=100 \
            --prefetch_bacth_size=2 \
            --layers_wise_copy_batchs="$LAYERS_WISE_BATCHS" \
            -log_dir="$node_log_dir" >"$node_log_file" 2>&1 &
    done
}

start_decode_workers() {
    log "Starting $DECODE_NODES decode workers ..."
    for ((j = 0; j < DECODE_NODES; j++)); do
        idx=$((PREFILL_NODES + j))
        port=$((START_PORT + idx))
        client_port=$((CLIENT_START_PORT + idx))
        device=$((START_DEVICE + idx))
        transfer_port=$((TRANSFER_START_PORT + idx))
        node_log_dir="$LOG_DIR/decode_$idx"
        node_log_file="$LOG_DIR/decode_$idx.log"

        rm -rf "$node_log_dir"
        mkdir -p "$node_log_dir"

        nohup "$XLLM_BIN" \
            --model "$MODEL_PATH" \
            --devices="npu:$device" \
            --port "$port" \
            --nnodes="$DECODE_NODES" \
            --max_memory_utilization=0.7 \
            --max_tokens_per_batch=100000 \
            --max_seqs_per_batch=256 \
            --enable_mla=false \
            --block_size=128 \
            --communication_backend="hccl" \
            --enable_prefix_cache="$PREFIX_CACHE" \
            --enable_chunked_prefill="$CHUNKED_PREFILL" \
            --enable_schedule_overlap="$SCHEDULE_OVERLAP" \
            --node_rank="$j" \
            --host_blocks_factor="$FACTOR" \
            --enable_disagg_pd=true \
            --instance_role=DECODE \
            --disagg_pd_port="$DECODE_DISAGG_PD_PORT" \
            --etcd_addr="$ETCD_ADDR" \
            --transfer_listen_port="$transfer_port" \
            --master_node_addr="$DECODE_MASTER_NODE_ADDR" \
            --enable_cache_upload="$CACHE_UPLOAD" \
            --enable_kvcache_store="$ENABLE_STORE" \
            --store_protocol="$STORE_PROTOCOL" \
            --store_metadata_server="$STORE_METADATA_SERVER" \
            --store_master_server_address="$STORE_MASTER_SERVER_ADDRESS" \
            --store_local_hostname="$STORE_HOST_IP:$client_port" \
            --logtostderr=false \
            --offload_batch=1 \
            -log_dir="$node_log_dir" >"$node_log_file" 2>&1 &
    done
}

log "Config summary:"
log "  PREFILL_NODES=$PREFILL_NODES, DECODE_NODES=$DECODE_NODES"
log "  MODEL_PATH=$MODEL_PATH"
log "  LOCAL_NODE_IP=$LOCAL_NODE_IP, GLOBAL_MASTER_IP=$GLOBAL_MASTER_IP"
log "  ETCD_ADDR=$ETCD_ADDR"
log "  STORE_MODE=$STORE_MODE"
log "  STORE_CONFIG=$STORE_CONFIG"
log "  STORE_RUNTIME_CONFIG=$STORE_RUNTIME_CONFIG"
log "  STORE_MASTER_SERVER_ADDRESS=$STORE_MASTER_SERVER_ADDRESS, MF_STORE_URL=$MF_STORE_URL"
log "  FACTOR=$FACTOR, LAYERS_WISE_BATCHS=$LAYERS_WISE_BATCHS"

log "  LOG_DIR=$LOG_DIR"

start_store_if_needed
stop_old_xllm_processes
run_etcd_cleanup
start_xllm_service
start_prefill_workers
start_decode_workers

log "Startup done. Check logs under: $LOG_DIR"

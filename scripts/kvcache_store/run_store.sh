#!/usr/bin/env bash

set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
XLLM_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MOONCAKE_MASTER_BIN="${MOONCAKE_MASTER_BIN:-/usr/local/bin/mooncake_master}"
CLIENT_SCRIPT="${CLIENT_SCRIPT:-$SCRIPT_DIR/distributed_client.py}"
LOG_DIR="${LOG_DIR:-$SCRIPT_DIR}"
DEPENDENCIES_SCRIPT="${DEPENDENCIES_SCRIPT:-$XLLM_ROOT/third_party/dependencies.sh}"

usage() {
    cat <<'EOF'
Usage:
  run_store.sh <config_path> [start_target] [force_restart]

Args:
  config_path    Required JSON config path.
  start_target   Optional, default: both
                 master | client | both
  force_restart  Optional, default: 0
                 0: if running then skip start
                 1: always kill first, then restart

Env overrides:
  MOONCAKE_MASTER_BIN    Master binary path (default: /usr/local/bin/mooncake_master)
  CLIENT_SCRIPT          Client script path (default: <script_dir>/distributed_client.py)
  LOG_DIR                Log directory (default: <script_dir>)
  PYTHON_BIN             Python executable (default: python)
EOF
}

log() {
    echo "[run_store] $*"
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    usage
    exit 0
fi

CONFIG_PATH="${1:-}"
START_TARGET_RAW="${2:-both}"
FORCE_RESTART_RAW="${3:-0}"

if [ -z "$CONFIG_PATH" ]; then
    usage
    exit 1
fi

START_TARGET="$(printf '%s' "$START_TARGET_RAW" | tr '[:upper:]' '[:lower:]')"
case "$START_TARGET" in
    master|client|both)
        ;;
    *)
        log "Invalid start_target: $START_TARGET_RAW"
        usage
        exit 1
        ;;
esac

case "$FORCE_RESTART_RAW" in
    1|true|TRUE|yes|YES)
        FORCE_RESTART=1
        ;;
    0|false|FALSE|no|NO|"")
        FORCE_RESTART=0
        ;;
    *)
        log "Invalid force_restart: $FORCE_RESTART_RAW (use 0/1)"
        usage
        exit 1
        ;;
esac

PYTHON_BIN="${PYTHON_BIN:-python}"
MASTER_LOG=""
CLIENT_LOG=""
MASTER_METRICS_PORT="${MASTER_METRICS_PORT:-9001}"
CLIENT_READY_CHECK_URL="${CLIENT_READY_CHECK_URL:-http://127.0.0.1:${MASTER_METRICS_PORT}/get_all_segments}"
CLIENT_READY_SEGMENT="${CLIENT_READY_SEGMENT:-127.0.0.1:8002}"
CLIENT_READY_TIMEOUT_SEC="${CLIENT_READY_TIMEOUT_SEC:-120}"
CLIENT_READY_INTERVAL_SEC="${CLIENT_READY_INTERVAL_SEC:-1}"
CLIENT_STOP_TIMEOUT_SEC="${CLIENT_STOP_TIMEOUT_SEC:-30}"
CLIENT_STOP_INTERVAL_SEC="${CLIENT_STOP_INTERVAL_SEC:-1}"

if [ ! -f "$CONFIG_PATH" ]; then
    log "config file not found: $CONFIG_PATH"
    exit 1
fi

if [ -z "$LOG_DIR" ]; then
    log "LOG_DIR is required"
    exit 1
fi
LOG_DIR="${LOG_DIR%/}"
MASTER_LOG="$LOG_DIR/mooncake_master_service.log"
CLIENT_LOG="$LOG_DIR/mooncake_client.log"
mkdir -p "$LOG_DIR"

if [ "$START_TARGET" = "master" ] || [ "$START_TARGET" = "both" ]; then
    if [ -z "$MOONCAKE_MASTER_BIN" ]; then
        log "MOONCAKE_MASTER_BIN is required"
        exit 1
    fi
    if [ ! -e "$MOONCAKE_MASTER_BIN" ]; then
        log "mooncake_master not found: $MOONCAKE_MASTER_BIN"
        log "Please install dependencies first by running: bash $DEPENDENCIES_SCRIPT mooncake <a3|a2|auto>"
        exit 1
    fi
    if [ ! -x "$MOONCAKE_MASTER_BIN" ]; then
        log "mooncake_master exists but is not executable: $MOONCAKE_MASTER_BIN"
        exit 1
    fi
fi

if [ "$START_TARGET" = "client" ] || [ "$START_TARGET" = "both" ]; then
    if [ -z "$CLIENT_SCRIPT" ]; then
        log "CLIENT_SCRIPT is required"
        exit 1
    fi
    if [ ! -f "$CLIENT_SCRIPT" ]; then
        log "client script not found: $CLIENT_SCRIPT"
        exit 1
    fi
fi

PROTOCOL="ub"

# Prevent proxy from affecting local metadata/service calls.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

if [ "$PROTOCOL" = "ub" ]; then
    export MEMCACHE_ROOT="${MEMCACHE_ROOT:-/usr/local/memfabric_hybrid/latest/aarch64-linux/}"
    if [ -f /usr/local/memfabric_hybrid/set_env.sh ]; then
        # shellcheck disable=SC1091
        source /usr/local/memfabric_hybrid/set_env.sh
    else
        log "WARN: /usr/local/memfabric_hybrid/set_env.sh not found; continue."
        log "memfabric_hybrid not found: $MEMCACHE_ROOT"
        log "Please install dependencies first by running: bash $DEPENDENCIES_SCRIPT memfabric"
        exit 1
    fi
fi

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"

is_master_running() {
    pgrep -f "$MOONCAKE_MASTER_BIN" >/dev/null 2>&1
}

is_client_running() {
    pgrep -f "$CLIENT_SCRIPT" >/dev/null 2>&1
}

stop_master() {
    pkill -f "$MOONCAKE_MASTER_BIN" >/dev/null 2>&1 || true
}

wait_for_client_exit() {
    local deadline=$((SECONDS + CLIENT_STOP_TIMEOUT_SEC))
    while is_client_running; do
        if [ "$SECONDS" -ge "$deadline" ]; then
            return 1
        fi
        sleep "$CLIENT_STOP_INTERVAL_SEC"
    done
    return 0
}

stop_client() {
    if ! is_client_running; then
        log "Client process not running, skip stop."
        return 0
    fi

    log "Stopping client process..."
    pkill -f "$CLIENT_SCRIPT" >/dev/null 2>&1 || true
    if wait_for_client_exit; then
        log "Client process stopped."
        return 0
    fi

    log "Client did not exit within ${CLIENT_STOP_TIMEOUT_SEC}s, sending SIGKILL."
    pkill -9 -f "$CLIENT_SCRIPT" >/dev/null 2>&1 || true
    if wait_for_client_exit; then
        log "Client process stopped after SIGKILL."
        return 0
    fi

    log "Client process still exists after SIGKILL."
    return 1
}

start_master() {
    log "Starting store master..."
    nohup "$MOONCAKE_MASTER_BIN" \
        --rpc_port=8000 \
        --enable_http_metadata_server=true \
        --http_metadata_server_port=8001 \
        --http_metadata_server_host=0.0.0.0 \
        --metrics_port="$MASTER_METRICS_PORT" \
        --enable_ha=false \
        --etcd_endpoints="http://127.0.0.1:8400" >"$MASTER_LOG" 2>&1 &
    log "Master started, pid=$!, log=$MASTER_LOG"
}

start_client() {
    log "Starting store client with config: $CONFIG_PATH"
    nohup "$PYTHON_BIN" "$CLIENT_SCRIPT" --config "$CONFIG_PATH" >"$CLIENT_LOG" 2>&1 &
    log "Client started, pid=$!, log=$CLIENT_LOG"
}

wait_for_client_registration() {
    local last_response=""
    local deadline=$((SECONDS + CLIENT_READY_TIMEOUT_SEC))

    log "Waiting client registration on master: url=$CLIENT_READY_CHECK_URL, expect=$CLIENT_READY_SEGMENT, timeout=${CLIENT_READY_TIMEOUT_SEC}s"
    while [ "$SECONDS" -lt "$deadline" ]; do
        last_response="$(curl -s "$CLIENT_READY_CHECK_URL" 2>/dev/null || true)"
        if printf '%s\n' "$last_response" | grep -Fq "$CLIENT_READY_SEGMENT"; then
            log "Client registration confirmed: $CLIENT_READY_SEGMENT"
            return 0
        fi

        if ! is_client_running; then
            log "Client process exited before registration completed."
            [ -n "$last_response" ] && log "Last response: $last_response"
            return 1
        fi
        sleep "$CLIENT_READY_INTERVAL_SEC"
    done

    log "Timed out waiting for client registration."
    [ -n "$last_response" ] && log "Last response: $last_response"
    return 1
}

if [ "$FORCE_RESTART" = "1" ]; then
    log "force_restart=1, kill then restart."
    if [ "$START_TARGET" = "master" ] || [ "$START_TARGET" = "both" ]; then
        stop_master
    fi
    if [ "$START_TARGET" = "client" ] || [ "$START_TARGET" = "both" ]; then
        stop_client
    fi
fi

if [ "$START_TARGET" = "master" ] || [ "$START_TARGET" = "both" ]; then
    if [ "$FORCE_RESTART" = "0" ] && is_master_running; then
        log "Master process already exists, skip start."
    else
        start_master
    fi
fi

if [ "$START_TARGET" = "client" ] || [ "$START_TARGET" = "both" ]; then
    if [ "$FORCE_RESTART" = "0" ] && is_client_running; then
        log "Client process already exists, skip start."
    else
        start_client
    fi
    wait_for_client_registration
fi

log "Store startup sequence finished."

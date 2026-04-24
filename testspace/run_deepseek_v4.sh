#!/bin/bash
 
# 配置参数
MODEL_PATH="/mnt/cfs/9n-das-admin/llm_models/mlu/DeepSeek-V4-Flash-BF16"           # 模型路径
MASTER_NODE_ADDR="0.0.0.0:25555"             # Master 节点地址（需全局一致）
PORT=7442                                         # 服务起始端口
START_DEVICE=0                                    # 起始 MLU 逻辑设备号
LOG_DIR="log"                                     # 日志目录
NNODES=8                                          # 节点数量（当前脚本启动 2 个进程）
WORLD_SIZE=8

export XLLM_PROCESS_GROUP_ASYNC_TIMEOUT_SECONDS=10

# 创建日志目录
mkdir -p "$LOG_DIR"
 
# 数据集参数设置
max_tokens_per_batch=4096
max_seqs_per_batch=8

for ((i = 0; i < NNODES; i++)); do
    DEVICE=$((START_DEVICE + i))
    LOG_FILE="${LOG_DIR}/node_${i}.log"
    xllm \
        --model "${MODEL_PATH}" \
        --devices="mlu:${DEVICE}" \
        --port "${PORT}" \
        --host="0.0.0.0" \
        --master_node_addr="${MASTER_NODE_ADDR}" \
        --nnodes="${WORLD_SIZE}" \
        --max_memory_utilization=0.93 \
        --max_tokens_per_batch="${max_tokens_per_batch}" \
        --max_seqs_per_batch="${max_seqs_per_batch}" \
        --block_size=1 \
        --max_cache_size=0 \
        --enable_prefix_cache=false \
        --enable_chunked_prefill=false \
        --enable_schedule_overlap=true \
        --node_rank="${i}" \
        --enable_shm=false \
	      --enable_graph=false \
        --expert_parallel_degree=1 \
        --ep_size=8 \
	      > "${LOG_FILE}" 2>&1 &
done

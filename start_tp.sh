# 1. 环境变量设置
export PYTHON_INCLUDE_PATH="$(python3 -c 'from sysconfig import get_paths; print(get_paths()["include"])')"
export PYTHON_LIB_PATH="$(python3 -c 'from sysconfig import get_paths; print(get_paths()["include"])')"
export PYTORCH_NPU_INSTALL_PATH=/usr/local/libtorch_npu/  # NPU 版 PyTorch 路径
export PYTORCH_INSTALL_PATH="$(python3 -c 'import torch, os; print(os.path.dirname(os.path.abspath(torch.__file__)))')"  # PyTorch 安装路径
export LIBTORCH_ROOT="$PYTORCH_INSTALL_PATH"  # LibTorch 路径
export LD_LIBRARY_PATH=/usr/local/libtorch_npu/lib:$LD_LIBRARY_PATH  # 添加 NPU 库路径

# 2. 加载环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh 
source /usr/local/Ascend/nnal/atb/set_env.sh


export ASDOPS_LOG_TO_STDOUT=1
export ASDOPS_LOG_LEVEL=ERROR
export ASDOPS_LOG_TO_FILE=1
# export ASCEND_SLOG_PRINT_TO_STDOUT=1
# export ASCEND_GLOBAL_LOG_LEVEL=0
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export NPU_MEMORY_FRACTION=0.98
export ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE=3
export ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1
export OMP_NUM_THREADS=12
export HCCL_CONNECT_TIMEOUT=7200
export INF_NAN_MODE_ENABLE=0
export INF_NAN_MODE_FORCE_DISABLE=1

#export ASCEND_LAUNCH_BLOCKING=1
#export ASCEND_HOST_LOG_FILE_NUM=1000
#export ASCEND_GLOBAL_EVENT_ENABLE=1
#export HCCL_ENTRY_LOG_ENABLE=1
#export ASCEND_MODULE_LOG_LEVEL=HCCL=0
# 3. 清理旧日志
\rm -rf core.*
\rm -rf log/*.log

# 4. 启动分布式服务
MASTER_NODE_ADDR="127.0.0.1:8888"                  # Master 节点地址（需全局一致）
START_PORT=18018                                   # 服务起始端口
START_DEVICE=4                                     # 起始 NPU 逻辑设备号
LOG_DIR="log"                                      # 日志目录
NNODES=2                                           # 节点数（当前脚本启动 2 个进程）

export HCCL_IF_BASE_PORT=43432  # HCCL 通信基础端口

for (( i=0; i<2; i++ ))
do
  PORT=$((START_PORT + i))
  DEVICE=$((START_DEVICE + i))
  LOG_FILE="$LOG_DIR/dit_node_tp_$i.log"
  ./build/xllm/core/server/xllm \
    --model="/export/home/models/flux2/" \
    --max_memory_utilization=0.6 \
    --backend="dit" \
    --tp_size=2 \
    --devices="npu:$DEVICE" \
    --master_node_addr=$MASTER_NODE_ADDR \
    --nnodes=$NNODES \
    --port $PORT \
    --communication_backend="hccl" \
    --enable_prefix_cache=false \
    --enable_chunked_prefill=false \
    --enable_schedule_overlap=false \
    --use_contiguous_input_buffer=false \
    --dit_debug_print=true \
    --enable-shm=true \
    --node_rank=$i > $LOG_FILE 2>&1 &
done
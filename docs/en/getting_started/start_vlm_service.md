# VLM Service Startup

This document describes how to start a VLM model service based on the xLLM inference engine.

## Single Device
Start the service by executing the following command in the main directory of the `xllm` project:
```bash
ASCEND_RT_VISIBLE_DEVICES=0 ./build/xllm/core/server/xllm --model=/path/to/Qwen2.5-VL-7B-Instruct  --port=12345  --max_memory_utilization 0.90 --enable_shm true
```

## Multiple Devices

Start the service:
```shell
bash start_qwen_vl.sh
```
The start_qwen_vl.sh script is as follows:
```bash title="start_qwen_vl.sh" linenums="1"
# 1. Environment variable setup
export PYTHON_INCLUDE_PATH="$(python3 -c 'from sysconfig import get_paths; print(get_paths()["include"])')"
export PYTHON_LIB_PATH="$(python3 -c 'from sysconfig import get_paths; print(get_paths()["include"])')"
export PYTORCH_NPU_INSTALL_PATH=/usr/local/libtorch_npu/  # NPU version PyTorch path
export PYTORCH_INSTALL_PATH="$(python3 -c 'import torch, os; print(os.path.dirname(os.path.abspath(torch.__file__)))')"  # PyTorch installation path
export LIBTORCH_ROOT="$PYTORCH_INSTALL_PATH"  # LibTorch path
export LD_LIBRARY_PATH=/usr/local/libtorch_npu/lib:$LD_LIBRARY_PATH  # Add NPU library path

# 2. Load npu environment
source /usr/local/Ascend/ascend-toolkit/set_env.sh 
source /usr/local/Ascend/nnal/atb/set_env.sh 
export ASCEND_RT_VISIBLE_DEVICES=10,11
export ASDOPS_LOG_TO_STDOUT=1
export ASDOPS_LOG_LEVEL=ERROR
export ASDOPS_LOG_TO_FILE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export NPU_MEMORY_FRACTION=0.98
export ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE=3
export ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1
export OMP_NUM_THREADS=24
export HCCL_CONNECT_TIMEOUT=7200
export INF_NAN_MODE_ENABLE=0

# 3. Clean up old logs
\rm -rf core.*

# 4. Start distributed service
MODEL_PATH="/path/to/your/Qwen2-7B-Instruct"  # Model path
MASTER_NODE_ADDR="127.0.0.1:9748"                  # Master node address (must be globally consistent)
START_PORT=18000                                   # Service starting port
START_DEVICE=0                                     # Starting NPU logical device number
LOG_DIR="log"                                      # Log directory
NNODES=2                                           # Number of nodes (this script starts 2 processes)

export HCCL_IF_BASE_PORT=43432  # HCCL communication base port

for (( i=0; i<$NNODES; i++ ))
do
  PORT=$((START_PORT + i))
  DEVICE=$((START_DEVICE + i))
  LOG_FILE="$LOG_DIR/node_$i.log"
  ./xllm/build/xllm/core/server/xllm \
    --model $MODEL_PATH \
    --devices="npu:$DEVICE" \
    --port $PORT \
    --master_node_addr=$MASTER_NODE_ADDR \
    --nnodes=$NNODES \
    --max_memory_utilization=0.86 \
    --max_tokens_per_batch=40000 \
    --max_seqs_per_batch=256 \
    --enable_mla=false \
    --block_size=128 \
    --communication_backend="lccl" \
    --enable_prefix_cache=false \
    --enable_chunked_prefill=false \
    --enable_schedule_overlap=false \
    --enable_shm=true \
    --node_rank=$i  &
done
```
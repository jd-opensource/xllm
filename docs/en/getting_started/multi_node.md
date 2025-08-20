# Multi-Node Deployment

Start the service:
```shell
bash start_qwen.sh
```

The start_qwen.sh script is as follows:
```bash title="start_qwen.sh" linenums="1"
# 1. Environment variable setup
export PYTHON_INCLUDE_PATH="$(python3 -c 'from sysconfig import get_paths; print(get_paths()["include"])')"
export PYTHON_LIB_PATH="$(python3 -c 'from sysconfig import get_paths; print(get_paths()["include"])')"
export PYTORCH_NPU_INSTALL_PATH=/usr/local/libtorch_npu/  # NPU version PyTorch path
export PYTORCH_INSTALL_PATH="$(python3 -c 'import torch, os; print(os.path.dirname(os.path.abspath(torch.__file__)))')"  # PyTorch installation path
export LIBTORCH_ROOT="$PYTORCH_INSTALL_PATH"  # LibTorch path
export LD_LIBRARY_PATH=/usr/local/libtorch_npu/lib:$LD_LIBRARY_PATH  # Add NPU library path

# 2. Load Ascend environment
source /usr/local/Ascend/ascend-toolkit/set_env.sh  # Load CANN toolchain
source /usr/local/Ascend/nnal/atb/set_env.sh       # Load ATB acceleration library
export ASCEND_RT_VISIBLE_DEVICES=10,11             # Specify visible NPU devices (physical cards 10 and 11)
export ASDOPS_LOG_TO_STDOUT=1   # Output ASDOPS logs to standard output (terminal)
export ASDOPS_LOG_LEVEL=ERROR   # Set ASDOPS log level, only output logs of specified level and above
export ASDOPS_LOG_TO_FILE=1     # Write ASDOPS logs to default path
# export HCCL_BUFFSIZE=1024
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True  # Allow dynamic expansion of video memory
export NPU_MEMORY_FRACTION=0.98                    # Video memory utilization upper limit 98%
export ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE=3          # ATB memory allocation algorithm
export ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1            # Global memory allocation
export OMP_NUM_THREADS=12   # OpenMP thread count (recommended to match CPU core count)
export HCCL_CONNECT_TIMEOUT=7200    # HCCL connection timeout (2 hours)
export INF_NAN_MODE_ENABLE=0

# 3. Clean up old logs
\rm -rf /root/atb/log/
\rm -rf /root/ascend/log/
\rm -rf core.*

# 4. Start distributed service
MODEL_PATH="/path/to/your/Qwen2-7B-Instruct"  # Model path
MASTER_NODE_ADDR="127.0.0.1:9748"                  # Master node address (must be globally consistent)
START_PORT=18000                                   # Service starting port
START_DEVICE=0                                     # Starting NPU logical device number
LOG_DIR="log"                                      # Log directory
NNODES=2                                           # Number of nodes (this script starts 2 processes)

export HCCL_IF_BASE_PORT=43432  # HCCL communication base port
export FOLLY_DEBUG_MEMORYIDLER_DISABLE_UNMAP=1  # Disable memory release (improves stability)

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
    --communication_backend="hccl" \
    --enable_prefix_cache=false \
    --enable_chunked_prefill=true \
    --enable_schedule_overlap=true \
    --node_rank=$i  &
done
```

Two nodes are used here, which can be configured using `--nnodes=$NNODES` and `--node_rank=$i`.
NPU Device can also be set using the `ASCEND_RT_VISIBLE_DEVICES` environment variable.

The client test command is the same as in the previous chapter [Client Invocation](./single_node.md#client).
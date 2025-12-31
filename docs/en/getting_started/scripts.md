# Startup Scripts

To help users quickly launch xLLM for rapid inference on different models, we provide optimized startup configurations for various mainstream open-source large language models that we have tested experimentally:

## Qwen2 / 3

```bash title="Qwen2/3 Startup Script" linenums="1" 
export PYTHON_INCLUDE_PATH="$(python3 -c 'from sysconfig import get_paths; print(get_paths()["include"])')"
export PYTHON_LIB_PATH="$(python3 -c 'from sysconfig import get_paths; print(get_paths()["include"])')"
export PYTORCH_NPU_INSTALL_PATH=/usr/local/libtorch_npu/
export PYTORCH_INSTALL_PATH="$(python3 -c 'import torch, os; print(os.path.dirname(os.path.abspath(torch.__file__)))')"
export LIBTORCH_ROOT="$(python3 -c 'import torch, os; print(os.path.dirname(os.path.abspath(torch.__file__)))')"
export LD_LIBRARY_PATH=/usr/local/libtorch_npu/lib:$LD_LIBRARY_PATH

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export ASDOPS_LOG_TO_STDOUT=1
export ASDOPS_LOG_LEVEL=ERROR
export ATB_LOG_TO_STDOUT=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export NPU_MEMORY_FRACTION=0.98
export ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE=3
export ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1

export OMP_NUM_THREADS=12

export HCCL_CONNECT_TIMEOUT=7200
export HCCL_OP_EXPANSION_MODE="AIV"

\rm -rf /root/atb/log/
\rm -rf /root/ascend/log/
\rm -rf core.*

MODEL_PATH="/export/home/weinan/weights/Qwen3-8B"
MASTER_NODE_ADDR="11.87.48.253:9590"
START_PORT=14830
START_DEVICE=0
LOG_DIR="log"
NNODES=4
WORLD_SIZE=4

export HCCL_IF_BASE_PORT=43439

for (( i=0; i<$NNODES; i++ ))
do
  PORT=$((START_PORT + i))
  DEVICE=$((START_DEVICE + i))
  LOG_FILE="$LOG_DIR/node_$i.log"
  ./xllm \
    --model $MODEL_PATH \
    --port $PORT \
    --devices="npu:$DEVICE" \
    --master_node_addr=$MASTER_NODE_ADDR \
    --nnodes=$WORLD_SIZE \
    --node_rank=$i \
    --max_memory_utilization=0.9 \
    --max_tokens_per_batch=20000 \
    --max_seqs_per_batch=3000 \
    --block_size=128 \
    --enable_prefix_cache=false \
    --enable_chunked_prefill=false \
    --communication_backend="lccl" \
    --enable_schedule_overlap=true \
    --enable_mla=false \
    --dp_size=1 \
    --ep_size=1 \
    > $LOG_FILE 2>&1 &
done
```

For different model sizes, only modify the following parameters in the script: `MODEL_PATH: weight path`, `START_DEVICE: starting device`, `NNODES: number of local devices`, `WORLD_SIZE: total number of devices`.

For random datasets generated based on ShareGPT, with input/output length limited to 2048 and TTFT limited to 50ms:

| Model Name | Number of devices | Single device Throughput | 
|:---------:|:---------:|:---------:|
|Qwen3-0.6B|      1 | 2946.02 tokens/s|
|Qwen3-1.7B |     1 | 2619.74 tokens/s|
|Qwen3-4B   |      1  |1628.13 tokens/s|
|Qwen3-8B  |       1  |1304.92 tokens/s|
|Qwen3-14B |      4  |951.6 tokens/s|
|Qwen3-32B |      8  |430.7 tokens/s|

## Deepseek

```bash title="Deepseek Startup Script" linenums="1" 
export PYTHON_INCLUDE_PATH="$(python3 -c 'from sysconfig import get_paths; print(get_paths()["include"])')"
export PYTHON_LIB_PATH="$(python3 -c 'from sysconfig import get_paths; print(get_paths()["include"])')"
export PYTORCH_NPU_INSTALL_PATH=/usr/local/libtorch_npu/
export PYTORCH_INSTALL_PATH="$(python3 -c 'import torch, os; print(os.path.dirname(os.path.abspath(torch.__file__)))')"
export LIBTORCH_ROOT="$(python3 -c 'import torch, os; print(os.path.dirname(os.path.abspath(torch.__file__)))')"
export LD_LIBRARY_PATH=/usr/local/libtorch_npu/lib:$LD_LIBRARY_PATH

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export ASDOPS_LOG_TO_STDOUT=1
export ASDOPS_LOG_LEVEL=ERROR
export ATB_LOG_TO_STDOUT=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export NPU_MEMORY_FRACTION=0.98
export ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE=3
export ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1
export OMP_NUM_THREADS=12
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_OP_EXPANSION_MODE="AIV"

\rm -rf /root/atb/log/
\rm -rf /root/ascend/log/
\rm -rf core.*

MODEL_PATH="/export/home/weinan/weights/DeepSeek-V3"
MASTER_NODE_ADDR="11.87.48.253:9590"
START_PORT=14830
START_DEVICE=0
LOG_DIR="log"
NNODES=16
WORLD_SIZE=16

export HCCL_IF_BASE_PORT=43439

for (( i=0; i<$NNODES; i++ ))
do
  PORT=$((START_PORT + i))
  DEVICE=$((START_DEVICE + i))
  LOG_FILE="$LOG_DIR/node_$i.log"
  ./xllm \
    --model $MODEL_PATH \
    --port $PORT \
    --devices="npu:$DEVICE" \
    --master_node_addr=$MASTER_NODE_ADDR \
    --nnodes=$WORLD_SIZE \
    --node_rank=$i \
    --max_memory_utilization=0.9 \
    --max_tokens_per_batch=20000 \
    --max_seqs_per_batch=3000 \
    --block_size=128 \
    --enable_prefix_cache=false \
    --enable_chunked_prefill=false \
    --communication_backend="hccl" \
    --enable_schedule_overlap=true \
    --enable_mla=true \
    --ep_size=4 \
    --dp_size=4 \
    > $LOG_FILE 2>&1 &
done
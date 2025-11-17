# VLM启动服务
本文档介绍基于xLLM推理引擎启动VLM模型服务

## 单卡
启动服务，在`xllm`工程主目录中执行下面命令：
```bash
ASCEND_RT_VISIBLE_DEVICES=0 ./build/xllm/core/server/xllm --model=/path/to/Qwen2.5-VL-7B-Instruct  --port=12345  --max_memory_utilization 0.90 --devices auto --enable_shm true
```

## 多卡

启动服务:
```shell
bash start_qwen_vl.sh
```

start_qwen_vl.sh 脚本如下:
```bash title="start_qwen_vl.sh" linenums="1"
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

# 3. 清理旧日志
\rm -rf core.*

# 4. 启动分布式服务
MODEL_PATH="/path/to/your/Qwen2.5-VL-7B-Instruct"  # 模型路径
MASTER_NODE_ADDR="127.0.0.1:9748"                  # Master 节点地址（需全局一致）
START_PORT=18000                                   # 服务起始端口
START_DEVICE=0                                     # 起始 NPU 逻辑设备号
LOG_DIR="log"                                      # 日志目录
NNODES=2                                           # 节点数（当前脚本启动 2 个进程）

export HCCL_IF_BASE_PORT=43432  # HCCL 通信基础端口

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
    --enable_shm true \
    --node_rank=$i  &
done
```
这里使用了两个节点，可以通过 `--nnodes=$NNODES` 和`--node_rank=$i`来设置。
同时可以通过 `ASCEND_RT_VISIBLE_DEVICES` 环境变量设置NPU Device。

客户端测试参考[client.md](./client.md)。

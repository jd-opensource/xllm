# 多机部署
该示例为两机32卡启动示例，第一台机器服务:
```shell
bash start_deepseek_machine_1.sh
```
start_deepseek_machine_1.sh 脚本如下:
```bash title="start_deepseek_machine_1.sh" linenums="1"
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
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export NPU_MEMORY_FRACTION=0.98
export ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE=3
export ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1
export OMP_NUM_THREADS=12
export HCCL_CONNECT_TIMEOUT=7200
export INF_NAN_MODE_ENABLE=0

# 3. 清理旧日志
\rm -rf core.*

# 4. 启动分布式服务

MODEL_PATH="/path/to/your/DeepSeek-R1"             # 模型路径
MASTER_NODE_ADDR="123.123.123.123:9748"            # Master 节点地址（需全局一致）
LOCAL_HOST=123.123.123.123                         # 本机服务启动IP
START_PORT=18000                                   # 服务起始端口
START_DEVICE=0                                     # 起始 NPU 逻辑设备号
LOG_DIR="log"                                      # 日志目录
LOCAL_NODES=16                                     # 单机节点数（当前脚本启动 16 个进程）
NNODES=32                                          # 总卡数（该示例为2机32卡）

export HCCL_IF_BASE_PORT=43432  # HCCL 通信基础端口

for (( i=0; i<$LOCAL_NODES; i++ ))
do
  PORT=$((START_PORT + i))
  DEVICE=$((START_DEVICE + i))
  LOG_FILE="$LOG_DIR/node_$i.log"
  ./xllm/build/xllm/core/server/xllm \
    --model $MODEL_PATH \
    --host $LOCAL_HOST \
    --port $PORT \
    --devices="npu:$DEVICE" \
    --master_node_addr=$MASTER_NODE_ADDR \
    --nnodes=$NNODES \
    --max_memory_utilization=0.86 \
    --max_tokens_per_batch=40000 \
    --max_seqs_per_batch=256 \
    --enable_mla=true \
    --block_size=128 \
    --enable_prefix_cache=false \
    --enable_chunked_prefill=false \
    --communication_backend="hccl" \
    --enable_schedule_overlap=true \
    --rank_tablefile=./ranktable_2s_32p.json \
    --node_rank=$i  &
done
```

启动第二台机器服务:

```shell
bash start_deepseek_machine_2.sh
```
start_deepseek_machine_2.sh 脚本如下:
```bash title="start_deepseek_machine_2.sh" linenums="1"
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
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export NPU_MEMORY_FRACTION=0.98
export ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE=3
export ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1
export OMP_NUM_THREADS=12
export HCCL_CONNECT_TIMEOUT=7200
export INF_NAN_MODE_ENABLE=0

# 3. 清理旧日志
\rm -rf core.*

# 4. 启动分布式服务

MODEL_PATH="/path/to/your/DeepSeek-R1"             # 模型路径
MASTER_NODE_ADDR="123.123.123.123:9748"            # Master 节点地址（需全局一致）
LOCAL_HOST=456.456.456.456                         # 本机服务启动IP
START_PORT=18000                                   # 服务起始端口
START_DEVICE=0                                     # 起始 NPU 逻辑设备号
LOG_DIR="log"                                      # 日志目录
LOCAL_NODES=16                                     # 单机节点数（当前脚本启动 16 个进程）
NNODES=32                                          # 总卡数（该示例为2机32卡）

export HCCL_IF_BASE_PORT=43432  # HCCL 通信基础端口

for (( i=0; i<$LOCAL_NODES; i++ ))
do
  PORT=$((START_PORT + i))
  DEVICE=$((START_DEVICE + i))
  LOG_FILE="$LOG_DIR/node_$i.log"
  ./xllm/build/xllm/core/server/xllm \
    --model $MODEL_PATH \
    --host $LOCAL_HOST \
    --port $PORT \
    --devices="npu:$DEVICE" \
    --master_node_addr=$MASTER_NODE_ADDR \
    --nnodes=$NNODES \
    --max_memory_utilization=0.86 \
    --max_tokens_per_batch=40000 \
    --max_seqs_per_batch=256 \
    --enable_mla=true \
    --block_size=128 \
    --enable_prefix_cache=false \
    --enable_chunked_prefill=false \
    --communication_backend="hccl" \
    --enable_schedule_overlap=true \
    --rank_tablefile=./ranktable_2s_32p.json \
    --node_rank=$((i + LOCAL_NODES))  &
done
```
这里使用了两台机器，可以通过 `--nnodes`设置总卡数，`--node_rank`为全局rank id。
`--rank_tablefile=./ranktable_2s_32p.json`为构建npu通信域所需文件，可参考[ranktable 生成](https://gitee.com/mindspore/models/blob/master/utils/hccl_tools/README.md)生成。

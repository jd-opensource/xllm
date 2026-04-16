# 使用 xLLM 在 Ascend A3设备 推理 GLM-5.0-W8A8 基座模型

+ 源码地址：https://github.com/jd-opensource/xllm

+ 国内可用: https://gitcode.com/xLLM-AI/xllm

+ 权重下载: [modelscope-GLM-5-W8A8](https://www.modelscope.cn/models/Eco-Tech/GLM-5-W8A8-xLLM/files)
  
## 1.拉取镜像环境

首先下载xLLM提供的镜像：

```bash
# A2 x86
docker pull quay.io/jd_xllm/xllm-ai:xllm-dev-a2-x86-20260306
# A2 arm
docker pull quay.io/jd_xllm/xllm-ai:xllm-dev-a2-arm-20260306
# A3 arm
docker pull quay.io/jd_xllm/xllm-ai:xllm-dev-a3-arm-20260306
```

**注意**: A2 机器性能未进行压测。

然后创建对应的容器

```bash
sudo docker run -it --ipc=host -u 0 --privileged --name mydocker --network=host \
 -v /var/queue_schedule:/var/queue_schedule \
 -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
 -v /usr/local/Ascend/add-ons/:/usr/local/Ascend/add-ons/ \
 -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
 -v /var/log/npu/conf/slog/slog.conf:/var/log/npu/conf/slog/slog.conf \
 -v /var/log/npu/slog/:/var/log/npu/slog \
 -v ~/.ssh:/root/.ssh  \
 -v /var/log/npu/profiling/:/var/log/npu/profiling \
 -v /var/log/npu/dump/:/var/log/npu/dump \
 -v /runtime/:/runtime/ -v /etc/hccn.conf:/etc/hccn.conf \
 -v /export/home:/export/home \
 -v /home/:/home/  \
 -w /export/home \
 quay.io/jd_xllm/xllm-ai:xllm-dev-hb-rc2-x86
```

## 2.拉取源码并编译

下载官方仓库与模块依赖：

```bash
git clone https://github.com/jd-opensource/xllm
cd xllm 
git checkout preview/glm-5
git submodule init
git submodule update
```

下载安装依赖:

```bash
pip install --upgrade pre-commit
yum install numactl
```

执行编译，在`build/`下生成可执行文件`build/xllm/core/server/xllm`：

```bash
python setup.py build
```

## 3.启动模型

### 若机器为重启后初次拉起服务，需先执行以下脚本对device进行初始化

#若不执行且npu未初始化可能导致xllm进程拉起失败

```bash
python -c "import torch_npu
for i in range(16):torch_npu.npu.set_device(i)"
```

### 环境变量

```bash
##### 1， 配置依赖路径相关环境变量
# export PYTHON_INCLUDE_PATH="$(python3 -c 'from sysconfig import get_paths; print(get_paths()["include"])')"
# export PYTHON_LIB_PATH="$(python3 -c 'from sysconfig import get_paths; print(get_paths()["include"])')"
# export PYTORCH_NPU_INSTALL_PATH=/usr/local/libtorch_npu/
# export PYTORCH_INSTALL_PATH="$(python3 -c 'import torch, os; print(os.path.dirname(os.path.abspath(torch.__file__)))')"
# export LIBTORCH_ROOT="$(python3 -c 'import torch, os; print(os.path.dirname(os.path.abspath(torch.__file__)))')"

# export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/xllm/op_api/lib/:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=/usr/local/libtorch_npu/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=/usr/lib64/libjemalloc.so.2:$LD_PRELOAD

# source /usr/local/Ascend/ascend-toolkit/set_env.sh
# source /usr/local/Ascend/nnal/atb/set_env.sh

##### 2， 配置日志相关环境变量
rm -rf /root/ascend/log/
rm -rf core.*

##### 3. 配置性能、通信相关环境变量
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
export HCCL_EXEC_TIMEOUT=300
export HCCL_CONNECT_TIMEOUT=300
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_IF_BASE_PORT=2864
```

## 启动命令 - GLM-5 （W8A8权重可单机拉起）

```bash
BATCH_SIZE=256
#推理最大batch数量
XLLM_PATH="./myxllm/xllm/build/xllm/core/server/xllm"
#推理入口文件路径（上一步中编译产物）
MODEL_PATH=/path/to/GLM-5-W8A8/
#模型路径（此处为int8量化的Glm-5）
DRAFT_MODEL_PATH=/path/to/GLM-5-W8A8/GLM-5-W8A8-MTP/
#Glm-5 导出的mtp权重

MASTER_NODE_ADDR="11.87.49.110:10015"
LOCAL_HOST="11.87.49.110"
# Service Port
START_PORT=18994
START_DEVICE=0
LOG_DIR="logs"
NNODES=16

for (( i=0; i<$NNODES; i++ ))
do
  PORT=$((START_PORT + i))
  DEVICE=$((START_DEVICE + i))
  LOG_FILE="$LOG_DIR/node_$i.log"
  nohup numactl -C $((DEVICE*40))-$((DEVICE*40+39)) $XLLM_PATH \
    --model $MODEL_PATH \
    --port $PORT \
    --devices="npu:$DEVICE" \
    --master_node_addr=$MASTER_NODE_ADDR \
    --nnodes=$NNODES \
    --node_rank=$i \
    --max_memory_utilization=0.85 \
    --max_tokens_per_batch=8192 \
    --max_seqs_per_batch=32 \
    --block_size=128 \
    --enable_prefix_cache=false \
    --enable_chunked_prefill=true \
    --communication_backend="hccl" \
    --enable_schedule_overlap=true \
    --enable_graph=true \
    --enable_graph_mode_decode_no_padding=true \
    --draft_model=$DRAFT_MODEL_PATH \
    --draft_devices="npu:$DEVICE" \
    --num_speculative_tokens=1 \
    --ep_size=8 \
    --dp_size=1 \
    > $LOG_FILE 2>&1 &
done

# numactl -C xxxxx          亲和性绑核(NUMA亲和性查询命令： npu-smi info -t topo)
#--max_memory_utilization   单卡最大显存占用比例
#--max_tokens_per_batch     单batch最大token数  （主要限制prefill）
#--max_seqs_per_batch       单batch最大请求数   （主要限制decoe）
#--communication_backend    通信backend 可选(hccl / lccl) 此处建议hccl
#--enable_schedule_overlap  开启异步调度
#--enable_prefix_cache      开启prefix_cache
#--enable_chunked_prefill   开启chunked_prefill
#--enable_graph             开启aclgraph
#--draft_model              mtp - mtp权重路径
#--draft_devices            mtp - mtp推理设备(与主模型同一)
#--num_speculative_tokens   mtp - 预测token数
```

日志出现"Brpc Server Started"表示服务成功拉起。

## 其他可选环境变量

```bash
#开启确定性计算
export LCCL_DETERMINISTIC=1
export HCCL_DETERMINISTIC=true
export ATB_MATMUL_SHUFFLE_K_ENABLE=0

# #开启动态profiling模式
# export PROFILING_MODE=dynamic
# \rm -rf ~/dynamic_profiling_socket_*
```

## 启动命令 - 双机拉起样例

### Node0 (master)

```bash
MASTER_NODE_ADDR="11.87.49.110:19990"
LOCAL_HOST="11.87.49.110"
START_PORT=15890
START_DEVICE=0
LOG_DIR="logs"
NNODES=32
LOCAL_NODES=16
export HCCL_IF_BASE_PORT=48439
unset HCCL_OP_EXPANSION_MODE

for (( i=0; i<$LOCAL_NODES; i++ ))do
  PORT=$((START_PORT + i))
  DEVICE=$((START_DEVICE + i));  LOG_FILE="$LOG_DIR/node_$i.log"
  nohup numactl -C $((DEVICE*40))-$((DEVICE*40+39)) $XLLM_PATH \    --model $MODEL_PATH \
    --host $LOCAL_HOST \
    --port $PORT \
    --devices="npu:$DEVICE" \
    --master_node_addr=$MASTER_NODE_ADDR \
    --nnodes=$NNODES \
    --node_rank=$i \
    --max_memory_utilization=0.85 \
    --max_tokens_per_batch=8192 \
    --max_seqs_per_batch=4 \
    --block_size=128 \
    --enable_prefix_cache=false \
    --enable_chunked_prefill=true \
    --communication_backend="hccl" \
    --enable_schedule_overlap=true \
    --enable_graph=true \
    --enable_graph_mode_decode_no_padding=true \
    --ep_size=16 \
    --dp_size=1 \
    --rank_tablefile=/yourPath/ranktable.json \
    > $LOG_FILE 2>&1 &
done
```

#### Node1 (worker)

```bash
MASTER_NODE_ADDR="11.87.49.110:19990"
LOCAL_HOST="11.87.49.111"
START_PORT=15890
START_DEVICE=0
LOG_DIR="logs"
NNODES=32
LOCAL_NODES=16
export HCCL_IF_BASE_PORT=48439
unset HCCL_OP_EXPANSION_MODE

for (( i=0; i<$LOCAL_NODES; i++ ))do
  PORT=$((START_PORT + i))
  DEVICE=$((START_DEVICE + i));  LOG_FILE="$LOG_DIR/node_$i.log"
  nohup numactl -C $((DEVICE*40))-$((DEVICE*40+39)) $XLLM_PATH \    --model $MODEL_PATH \
    --host $LOCAL_HOST \
    --port $PORT \
    --devices="npu:$DEVICE" \
    --master_node_addr=$MASTER_NODE_ADDR \
    --nnodes=$NNODES \
    --node_rank=$((i + LOCAL_NODES)) \
    --max_memory_utilization=0.85 \
    --max_tokens_per_batch=8192 \
    --max_seqs_per_batch=4 \
    --block_size=128 \
    --enable_prefix_cache=false \
    --enable_chunked_prefill=true \
    --communication_backend="hccl" \
    --enable_schedule_overlap=true \
    --enable_graph=true \
    --enable_graph_mode_decode_no_padding=true \
    --ep_size=16 \
    --dp_size=1 \
    --rank_tablefile=/yourPath/ranktable.json \
    > $LOG_FILE 2>&1 &
done
```

#### ranktable样例

 ranktable配置指导：https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/hccl/hcclug/hcclug_000014.html

```json
{
    "version": "1.0",
    "server_count": "2",
    "server_list": [
        {
            "server_id": "11.87.49.110",
            "device": [
                {
                    "device_id": "0",
                    "device_ip": "11.86.23.210",
                    "rank_id": "0"
                },
                ...
                {
                    "device_id": "7",
                    "device_ip": "11.86.23.217",
                    "rank_id": "7"
                }
            ],
            "host_nic_ip": "reserve"
        },
        {
            "server_id": "11.87.49.111",
            "device": [
                {
                    "device_id": "0",
                    "device_ip": "11.87.63.202",
                    "rank_id": "8"
                },
                ...
                {
                    "device_id": "7",
                    "device_ip": "11.87.63.209",
                    "rank_id": "15"
                }
            ],
            "host_nic_ip": "reserve"
        }
    ],
    "status": "completed"
}
```


## device NUMA亲和性查看

命令：

```bash
npu-smi info -t topo
```

前述命令中

```bash
numactl -C $((DEVICE*12))-$((DEVICE*12+11))
```

表示该进程绑在对应亲和的核上，可根据机器具体情况修改绑定的核id

## EX3.Glm-5 权重量化 

### 安装msmodelslim

```bash
git clone https://gitcode.com/shenxiaolong/msmodelslim.git 
cd msmodelslim
bash install.sh
```

### 修改tokenizer_config.json

```bash
  "extra_special_tokens" 
    改成 "additional_special_tokens"

  "tokenizer_class": "TokenizersBackend" 
    改成 "tokenizer_class": "PreTrainedTokenizer"
```

### 基于GLM-5-BF16 权重量化W8A8权重

```bash
### 预处理mtp相关权重
python example/GLM5/extract_mtp.py --model-dir ${model_path}

#指定transformers版本
pip install transformers==4.48.2

#量化执行（生成量化权重）
msmodelslim quant  --model_path ${model_path}  --save_path ${save_path}  --model_type DeepSeek-V3.2  --quant_type w8a8  --trust_remote_code True

#拷贝chat_template文件
cp ${model_path}/chat_template.jinja ${save_path}

#量化mtp权重导出（用于xllm推理）
python example/GLM5/export_mtp.py --input-dir  ${int8_save_path} --output-dir  ${mtp_save_path}
```

## PD分离

### etcd\xllm-service 安装

#### PD分离部署

`xllm`支持PD分离部署，这需要与另一个开源库[xllm service](https://github.com/jd-opensource/xllm-service)配套使用。

##### xLLM Service依赖

首先，我们下载安装`xllm service`，与安装编译`xllm`类似：

```bash
git clone https://github.com/jd-opensource/xllm-service
cd xllm_service
git submodule init
git submodule update
```

##### etcd安装

`xllm_service`依赖[etcd](https://github.com/etcd-io/etcd)，使用etcd官方提供的[安装脚本](https://github.com/etcd-io/etcd/releases)进行安装，其脚本提供的默认安装路径是`/tmp/etcd-download-test/etcd`，我们可以手动修改其脚本中的安装路径，也可以运行完脚本之后手动迁移：

```bash
mv /tmp/etcd-download-test/etcd /path/to/your/etcd
```

##### xLLM Service编译

先应用patch:

```bash
sh prepare.sh
```

再执行编译:

```bash
mkdir -p build
cd build
cmake ..
make -j 8
cd ..
```

!!! warning "可能的错误"
    这里能会遇到关于`boost-locale`和`boost-interprocess`的安装错误：`vcpkg-src/packages/boost-locale_x64-linux/include: No such     file or directory`,`/vcpkg-src/packages/boost-interprocess_x64-linux/include: No such file or directory`
    我们使用`vcpkg`重新安装这些包:
    ```bash
    /path/to/vcpkg remove boost-locale boost-interprocess
    /path/to/vcpkg install boost-locale:x64-linux
    /path/to/vcpkg install boost-interprocess:x64-linux
    ```

### PD分离运行

启动etcd:

```bash
./etcd-download-test/etcd --listen-peer-urls 'http://localhost:2390'  --listen-client-urls 'http://localhost:2389' --advertise-client-urls  'http://localhost:2391'
```

跨机配置时，etcd参考如下：

```bash
/tmp/etcd-download-test/etcd --listen-peer-urls 'http://0.0.0.0:3390' --listen-client-urls 'http://0.0.0.0:3389' --advertise-client-urls 'http://11.87.191.82:3389'
```

启动xllm service:

```bash
ENABLE_DECODE_RESPONSE_TO_SERVICE=true ./xllm_master_serving --etcd_addr="127.0.0.1:12389" --http_server_port 28888 --rpc_server_port 28889 --tokenizer_path=/export/home/models/GLM-5-W8A8/
```

跨机配置时，启动xllm service:

```bash
ENABLE_DECODE_RESPONSE_TO_SERVICE=true ../xllm-service/build/xllm_service/xllm_master_serving --etcd_addr="11.87.191.82:3389" --http_server_port 38888 --rpc_server_port 38889 --tokenizer_path=/export/home/models/GLM-5-W8A8/
```
- 启动Prefill实例
```bash
  BATCH_SIZE=256
  #推理最大batch数量
  XLLM_PATH="./myxllm/xllm/build/xllm/core/server/xllm"
  #推理入口文件路径（上一步中编译产物）
  MODEL_PATH=/export/home/models/GLM-5-w8a8/
  #模型路径（此处为int量化的Glm-5）
  DRAFT_MODEL_PATH=/export/home/models/GLM-5-MTP/
  
  MASTER_NODE_ADDR="$master_ip:$master_port"
  LOCAL_HOST="$local_ip"
  # Service Port
  START_PORT=18994
  START_DEVICE=0
  LOG_DIR="logs"
  NNODES=16
  
  for (( i=0; i<$NNODES; i++ ))
  do
    PORT=$((START_PORT + i))
    DEVICE=$((START_DEVICE + i))
    LOG_FILE="$LOG_DIR/node_$i.log"
    nohup numactl -C $((i*40))-$((i*40+39)) $XLLM_PATH \
      --model $MODEL_PATH  --model_id glmmoe \
      --host $LOCAL_HOST \
      --port $PORT \
      --devices="npu:$DEVICE" \
      --master_node_addr=$MASTER_NODE_ADDR \
      --nnodes=$NNODES \
      --node_rank=$i \
      --max_memory_utilization=0.86 \
      --max_tokens_per_batch=5000 \
      --max_seqs_per_batch=$BATCH_SIZE \
      --communication_backend=hccl \
      --enable_schedule_overlap=true \
      --enable_prefix_cache=false \
      --enable_chunked_prefill=false \
      --enable_graph=true \
      --draft_model $DRAFT_MODEL_PATH \
      --draft_devices="npu:$DEVICE" \
      --num_speculative_tokens 1 \
      --enable_disagg_pd=true \
      --instance_role=PREFILL \
      --etcd_addr=$LOCAL_HOST:3389 \
      --transfer_listen_port=$((36100 + i)) \
      --disagg_pd_port=8877 \
      > $LOG_FILE 2>&1 &
  done
  
  #--etcd_addr=$LOCAL_HOST:3389  参考etcd中advertise-client-urls的配置
  #--instance_role=DECODE     PD配置，DECODE\PREFILL
  ```

- 启动Decode实例
  
  ```bash
    BATCH_SIZE=256
  #推理最大batch数量
  XLLM_PATH="./myxllm/xllm/build/xllm/core/server/xllm"
  #推理入口文件路径（上一步中编译产物）
  MODEL_PATH=/export/home/models/GLM-5-w8a8/
  #模型路径（此处为int量化的Glm-5）
  DRAFT_MODEL_PATH=/export/home/models/GLM-5-MTP/
  
  MASTER_NODE_ADDR="$master_ip:$master_port"
  LOCAL_HOST="$local_ip"
  # Service Port
  START_PORT=18994
  START_DEVICE=0
  LOG_DIR="logs"
  NNODES=16
  
  for (( i=0; i<$NNODES; i++ ))
  do
    PORT=$((START_PORT + i))
    DEVICE=$((START_DEVICE + i))
    LOG_FILE="$LOG_DIR/node_$i.log"
    nohup numactl -C $((i*40))-$((i*40+39)) $XLLM_PATH \
      --model $MODEL_PATH  --model_id glmmoe \
      --host $LOCAL_HOST \
      --port $PORT \
      --devices="npu:$DEVICE" \
      --master_node_addr=$MASTER_NODE_ADDR \
      --nnodes=$NNODES \
      --node_rank=$i \
      --max_memory_utilization=0.86 \
      --max_tokens_per_batch=5000 \
      --max_seqs_per_batch=$BATCH_SIZE \
      --communication_backend=hccl \
      --enable_schedule_overlap=true \
      --enable_prefix_cache=false \
      --enable_chunked_prefill=false \
      --enable_graph=true \
      --draft_model $DRAFT_MODEL_PATH \
      --draft_devices="npu:$DEVICE" \
      --num_speculative_tokens 1 \
      --enable_disagg_pd=true \
      --instance_role=DECODE \
      --etcd_addr=$LOCAL_HOST:3389 \
      --transfer_listen_port=$((36100 + i)) \
      --disagg_pd_port=8877 \
      > $LOG_FILE 2>&1 &
  done
  
  #--etcd_addr=$LOCAL_HOST:3389  参考etcd中advertise-client-urls的配置
  #--instance_role=DECODE     PD配置，DECODE\PREFILL
  ```

# GLM5/CP 特性压测数据（最优配置）
##  压测环境
* 硬件： Ascend 910C（A3） / 4 Pods
* 主模型：GLM5-W8A8 
* 草稿模型：GLM5-W8A8-MTP
* PD分离配置：
  * P实例：cp_size = 16，dp_size = 1, ep_size = 1
  * D实例：dp_size = 2， ep_size = 32
* xllm 版本：release/v0.9.0（9be308aec60ea4a2dd799ee021ea42d608f4e67c - lastcommit）

## PD分离服务启动脚本
### PD分离4机配置  
#### prefill双节点配置
##### prefill节点1
```
#!/bin/bash
set -e

rm -rf core.*
rm -rf ~/ascend/log

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export HCCL_IF_BASE_PORT=43432

#export ASCEND_GLOBAL_LOG_LEVEL=1
#export MINDIE_LOG_TO_STDOUT=1

#export LCCL_DETERMINISTIC=1
#export HCCL_DETERMINISTIC=true
#export ATB_MATMUL_SHUFFLE_K_ENABLE=0

#export ASCEND_LAUNCH_BLOCKING=1
#export ATB_STREAM_SYNC_EVERY_KERNEL_ENABLE=1

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export NPU_MEMORY_FRACTION=0.96
export ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE=3
export ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1
export ATB_LAYER_INTERNAL_TENSOR_REUSE=1
export ATB_CONTEXT_WORKSPACE_SIZE=0


MODEL_PATH="/export/home/models/GLM-5-final-w8a8/"
#MODEL_PATH="/export/home/models/DeepSeek-V3.2-w8a8/"
#DRAFT_MODEL_PATH="/export/home/models/DeepSeek-V3.2-w8a8-mtp"
DRAFT_MODEL_PATH="/export/home/models/GLM-5-final-w8a8-MTP/"
MASTER_NODE_ADDR="$master_ip:$master_port"
START_PORT=48000
START_DEVICE=0
LOG_DIR="log"
NNODES=32
LOCAL_NODES=16
LOCAL_HOST="$local_ip"

mkdir -p $LOG_DIR


    #--draft_model $DRAFT_MODEL_PATH \
    #--draft_devices="npu:$DEVICE" \
    #--num_speculative_tokens 3 \

for (( i=0; i<$LOCAL_NODES; i++ ))
do
  PORT=$((START_PORT + i))
  DEVICE=$((START_DEVICE + i))
  LOG_FILE="$LOG_DIR/node_$i.log"
  nohup numactl -C $((DEVICE*40))-$((DEVICE*40+39)) /export/home/shifengmin.3/workspace/lt_xllm/build/xllm/core/server/xllm \
    --model $MODEL_PATH \
    --devices="npu:$DEVICE" \
    --port $PORT \
    --host $LOCAL_HOST \
    --master_node_addr=$MASTER_NODE_ADDR \
    --draft_model $DRAFT_MODEL_PATH \ # 草稿模型
    --draft_devices="npu:$DEVICE" \
    --num_speculative_tokens 3 \ # 采样率
    --nnodes=$NNODES \
    --max_memory_utilization=0.7 \ # 现存使用率
    --block_size=128 \
    --max_seqs_per_batch=9000 \
    --max_tokens_per_batch=67000 \
    --communication_backend="hccl" \
    --enable_prefix_cache=false \
    --enable_chunked_prefill=false \
    --enable_schedule_overlap=false \
    --enable_disagg_pd=true \ # 开启PD分离
    --instance_role=PREFILL \
    --etcd_addr=$etcd_addr:$etcd_port \
    --transfer_listen_port=$((26000+i)) \
    --disagg_pd_port=7777 \
    --cp_size 16 \ # 开启CP
    --dp_size 1 \
    --ep_size 1 \
    --node_rank=$i \
    --rank_tablefile=/export/home/shifengmin.3/workspace/ranktable_9899_new.json \ # prefill双机卡间通信路由表
    > $LOG_FILE 2>&1 &

done

tail -f log/node_0.log
```

##### prefill节点2
```
#!/bin/bash
set -e

rm -rf core.*
rm -rf ~/ascend/log

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export HCCL_IF_BASE_PORT=43432

#export ASCEND_GLOBAL_LOG_LEVEL=1
#export MINDIE_LOG_TO_STDOUT=1

#export LCCL_DETERMINISTIC=1
#export HCCL_DETERMINISTIC=true
#export ATB_MATMUL_SHUFFLE_K_ENABLE=0

#export ASCEND_LAUNCH_BLOCKING=1
#export ATB_STREAM_SYNC_EVERY_KERNEL_ENABLE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export NPU_MEMORY_FRACTION=0.96
export ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE=3
export ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1
export ATB_LAYER_INTERNAL_TENSOR_REUSE=1
export ATB_CONTEXT_WORKSPACE_SIZE=0

MODEL_PATH="/export/home/models/GLM-5-final-w8a8/"
DRAFT_MODEL_PATH="/export/home/models/GLM-5-final-w8a8-MTP/"
MASTER_NODE_ADDR="$master_ip:$master_port"
START_PORT=48000
START_DEVICE=0
LOG_DIR="log"
NNODES=32
LOCAL_NODES=16
LOCAL_HOST="$local_ip"

mkdir -p $LOG_DIR


    #--draft_model $DRAFT_MODEL_PATH \
    #--draft_devices="npu:$DEVICE" \
    #--num_speculative_tokens 3 \

for (( i=0; i<$LOCAL_NODES; i++ ))
do
  PORT=$((START_PORT + i))
  DEVICE=$((START_DEVICE + i))
  LOG_FILE="$LOG_DIR/node_$i.log"
  nohup numactl -C $((DEVICE*40))-$((DEVICE*40+39)) /export/home/shifengmin.3/workspace/lt_xllm/build/xllm/core/server/xllm \
    --model $MODEL_PATH \
    --devices="npu:$DEVICE" \
    --port $PORT \
    --host $LOCAL_HOST \
    --master_node_addr=$MASTER_NODE_ADDR \
    --draft_model $DRAFT_MODEL_PATH \ # 草稿模型地址
    --draft_devices="npu:$DEVICE" \
    --num_speculative_tokens 3 \ # 采样率
    --nnodes=$NNODES \
    --max_memory_utilization=0.7 \ # 显存使用率
    --block_size=128 \
    --max_seqs_per_batch=9000 \
    --max_tokens_per_batch=67000 \
    --communication_backend="hccl" \
    --enable_prefix_cache=false \
    --enable_chunked_prefill=false \
    --enable_schedule_overlap=false \
    --enable_disagg_pd=true \ # 开启PD分离
    --instance_role=PREFILL \
    --etcd_addr=$etcd_addr:$etcd_port \
    --transfer_listen_port=$((26100+i)) \
    --disagg_pd_port=7777 \
    --cp_size 16 \ # 开启CP
    --dp_size 1 \ 
    --ep_size 1 \
    --node_rank=$((i+LOCAL_NODES)) \
    --rank_tablefile=/export/home/shifengmin.3/workspace/ranktable_9899_new.json \
    > $LOG_FILE 2>&1 &

done

tail -f log/node_0.log
```
#### decode 双机配置
##### decode节点1
```
#!/bin/bash
set -e

rm -rf core.*
rm -rf ~/ascend/log

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export HCCL_IF_BASE_PORT=43432

#export ASCEND_GLOBAL_LOG_LEVEL=1
#export MINDIE_LOG_TO_STDOUT=1

#export LCCL_DETERMINISTIC=1
#export HCCL_DETERMINISTIC=true
#export ATB_MATMUL_SHUFFLE_K_ENABLE=0

#export ASCEND_LAUNCH_BLOCKING=1
#export ATB_STREAM_SYNC_EVERY_KERNEL_ENABLE=1

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export NPU_MEMORY_FRACTION=0.96
export ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE=3
export ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1
export ATB_LAYER_INTERNAL_TENSOR_REUSE=1
export ATB_CONTEXT_WORKSPACE_SIZE=0


MODEL_PATH="/export/home/models/GLM-5-final-w8a8/"
DRAFT_MODEL_PATH="/export/home/models/GLM-5-final-w8a8-MTP/"
MASTER_NODE_ADDR="$master_ip:$master_port"
START_PORT=48000
START_DEVICE=0
LOG_DIR="log"
NNODES=32
LOCAL_NODES=16
LOCAL_HOST="$local_ip"

mkdir -p $LOG_DIR


    #--draft_model $DRAFT_MODEL_PATH \
    #--draft_devices="npu:$DEVICE" \
    #--num_speculative_tokens 3 \

for (( i=0; i<$LOCAL_NODES; i++ ))
do
  PORT=$((START_PORT + i))
  DEVICE=$((START_DEVICE + i))
  LOG_FILE="$LOG_DIR/node_$i.log"
  nohup numactl -C $((DEVICE*40))-$((DEVICE*40+39)) /export/home/shifengmin.3/workspace/lt_xllm/build/xllm/core/server/xllm \
    --model $MODEL_PATH \ # GLM5.0权重
    --devices="npu:$DEVICE" \
    --port $PORT \
    --host $LOCAL_HOST \
    --master_node_addr=$MASTER_NODE_ADDR \ 
    --draft_model $DRAFT_MODEL_PATH \ # MTP权重
    --draft_devices="npu:$DEVICE" \
    --num_speculative_tokens 3 \ # 采样率
    --nnodes=$NNODES \
    --max_memory_utilization=0.80 \
    --block_size=128 \
    --max_seqs_per_batch=9000 \
    --communication_backend="hccl" \
    --enable_prefix_cache=false \
    --enable_chunked_prefill=false \
    --enable_schedule_overlap=true \ # 开启异步调度
    --enable_shm=true \ # 开启共享内存
    --enable_graph=false \
    --enable_graph_mode_decode_no_padding=false \
    --enable_disagg_pd=true \ # 开启PD分离
    --instance_role=DECODE \
    --etcd_addr=$etcd_addr:$etcd_port \
    --transfer_listen_port=$((26000+i)) \
    --disagg_pd_port=7777 \
    --dp_size 2 \ # dp并行
    --ep_size 32 \ # EP并行
    --node_rank=$i \
    --rank_tablefile=/export/home/shifengmin.3/workspace/ranktable_8382_new.json \ # 设置卡间通信
    > $LOG_FILE 2>&1 &

done

tail -f log/node_0.log
```
##### decode节点-2
```
#!/bin/bash
set -e

rm -rf core.*
rm -rf ~/ascend/log

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export HCCL_IF_BASE_PORT=43432

#export ASCEND_GLOBAL_LOG_LEVEL=1
#export MINDIE_LOG_TO_STDOUT=1

#export LCCL_DETERMINISTIC=1
#export HCCL_DETERMINISTIC=true
#export ATB_MATMUL_SHUFFLE_K_ENABLE=0

#export ASCEND_LAUNCH_BLOCKING=1
#export ATB_STREAM_SYNC_EVERY_KERNEL_ENABLE=1

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export NPU_MEMORY_FRACTION=0.96
export ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE=3
export ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1
export ATB_LAYER_INTERNAL_TENSOR_REUSE=1
export ATB_CONTEXT_WORKSPACE_SIZE=0


MODEL_PATH="/export/home/models/GLM-5-final-w8a8/"
DRAFT_MODEL_PATH="/export/home/models/GLM-5-final-w8a8-MTP/"
MASTER_NODE_ADDR="$master_ip:$master_port"
START_PORT=48000
START_DEVICE=0
LOG_DIR="log"
NNODES=32
LOCAL_NODES=16
LOCAL_HOST="$local_ip"

mkdir -p $LOG_DIR


    #--draft_model $DRAFT_MODEL_PATH \
    #--draft_devices="npu:$DEVICE" \
    #--num_speculative_tokens 3 \

for (( i=0; i<$LOCAL_NODES; i++ ))
do
  PORT=$((START_PORT + i))
  DEVICE=$((START_DEVICE + i))
  LOG_FILE="$LOG_DIR/node_$i.log"
  nohup numactl -C $((DEVICE*40))-$((DEVICE*40+39)) /export/home/shifengmin.3/workspace/lt_xllm/build/xllm/core/server/xllm \
    --model $MODEL_PATH \
    --devices="npu:$DEVICE" \
    --port $PORT \
    --host $LOCAL_HOST \
    --master_node_addr=$MASTER_NODE_ADDR \
    --draft_model $DRAFT_MODEL_PATH \ # 草稿模型
    --draft_devices="npu:$DEVICE" \
    --num_speculative_tokens 3 \ # 采样率
    --nnodes=$NNODES \
    --max_memory_utilization=0.80 \ # 现存使用率 80%
    --block_size=128 \
    --max_seqs_per_batch=9000 \
    --communication_backend="hccl" \
    --enable_prefix_cache=false \
    --enable_chunked_prefill=false \
    --enable_schedule_overlap=true \ # 开启异步调度
    --enable_shm=true \ # 开启共享内存
    --enable_graph=false \
    --enable_graph_mode_decode_no_padding=false \
    --enable_disagg_pd=true \ # PD分离
    --instance_role=DECODE \ #  decode 节点
    --etcd_addr=$etcd_ip:$etcd_port \
    --transfer_listen_port=$((26100+i)) \
    --disagg_pd_port=7777 \
    --dp_size 2 \ # 开启dp
    --ep_size 32 \ # 开启ep
    --node_rank=$((i+LOCAL_NODES)) \
    --rank_tablefile=/export/home/shifengmin.3/workspace/ranktable_8382_new.json \ # 双机间通信路由表
    > $LOG_FILE 2>&1 &

done

tail -f log/node_0.log
```

## 压测
### 自定义数据集 - 输入输出配置
Modified Location：/benchmark/ais_bench/datasets/synthetic/synthetic_config.py
```
#
# [Uniform均匀分布] -- "Method" : "uniform"
#   - MinValue: 最小值，范围为 [1, 2**20]
#   - MaxValue: 最大值, 范围为 [1, 2**20], 可等于MinValue
#
# [Gaussian高斯分布] -- "Method" : "gaussian"
#   - Mean    : 平均值, 范围为 [-3.0e38, 3.0e38]，分布中心位置
#   - Var     : 方差, 范围为[0, 3.0e38]，控制数据分散程度
#   - MinValue: 最小值, 范围为 [1, 2**20], 可低于Mean
#   - MaxValue: 最大值, 范围为 [1, 2**20], 可高于Mean, 可等于MinValue
#
# [Zipf齐夫分布] -- "Method" : "zipf"
#   - Alpha   : 形状参数, 范围为(1.0,10.0], 值越大分布越均匀
#   - MinValue: 最小值, 范围为 [1, 2**20]
#   - MaxValue: 最大值, 范围为 [1, 2**20], 需大于MinValue
"""
synthetic_config = {
    "Type":"tokenid",   # [tokenid/string]，生成的随机数据集类型，支持固定长度的随机tokenid，和随机长度的string，两种类型的数据集
    "RequestCount": 10, # 生成的请求条数，应与模型侧配置文件中的 decode_batch_size 一致
    "TrustRemoteCode": False, #是否信任远端代码，tokenid模式下需要加载tokenizer生成tokenid，默认为Fasle
    "StringConfig" : {  # string类型的随机数据集的配置相关项，请参考以上注释处："StringConfig中的随机生成方法参数说明"
        "Input" : {     # 每条请求的输入长度
            "Method": "uniform",
            "Params": {"MinValue": 16384, "MaxValue": 16384}
        },
        "Output" : {    # 每条请求的输出长度
            "Method": "gaussian",
            "Params": {"Mean": 1024, "Var": 0, "MinValue": 1024, "MaxValue": 1024}
        }
    },
    "TokenIdConfig" : { # tokenid类型的随机数据集的配置相关项
        "RequestSize": 16384 # 每条请求的长度，即每条请求中token id的个数，应与模型侧配置文件中的 input_seq_len 一致
    }
}
```
### ais_bench客户端设置
更新位置：/benchmark/ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py
```
from ais_bench.benchmark.models import VLLMCustomAPIChatStream
from ais_bench.benchmark.utils.model_postprocessors import extract_non_reasoning_content

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChatStream,
        abbr='vllm-api-stream-chat',
        path="[$GLM5_weight]", # GLM5 w8a8权重
        model="[$GLM5_mtp_weight]", # GLM5-MTP权重
        request_rate = 0,
        retry = 1,
        host_ip = "[$server_ip]", # 推理服务ip
        host_port = [$server_port], # 推理服务port 
        max_out_len = 1024, # token输出数量
        batch_size=1,
        trust_remote_code=False,
        generation_kwargs = dict(
            temperature = 0,
            top_k = -1,
            top_p = 1,
            seed = None,
            repetition_penalty = 1.03,
            ignore_eos=True,
        ),
        pred_postprocessor=dict(type=extract_non_reasoning_content)
    )
]
```

### ais_bench客户端发起压测
```
ais_bench --models vllm_api_stream_chat --datasets synthetic_gen -m perf
```

## 压测数据
### 32k/2k
* TTFT： P99  3.36/s
* TPOT：P99 42/ms

```
╒══════════════════════════╤═════════╤═════════════════╤═════════════════╤═════════════════╤═════════════════╤═════════════════╤═════════════════╤═════════════════╤═════╕
│ Performance Parameters   │ Stage   │ Average         │ Min             │ Max             │ Median          │ P75             │ P90             │ P99             │  N  │
╞══════════════════════════╪═════════╪═════════════════╪═════════════════╪═════════════════╪═════════════════╪═════════════════╪═════════════════╪═════════════════╪═════╡
│ E2EL                     │ total   │ 77142.7 ms      │ 63874.8 ms      │ 89564.7 ms      │ 78482.0 ms      │ 82704.9 ms      │ 84642.2 ms      │ 89072.4 ms      │ 10  │
├──────────────────────────┼─────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┼─────┤
│ TTFT                     │ total   │ 3221.8 ms       │ 3179.5 ms       │ 3375.2 ms       │ 3198.3 ms       │ 3230.2 ms       │ 3255.4 ms       │ 3363.2 ms       │ 10  │
├──────────────────────────┼─────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┼─────┤
│ TPOT                     │ total   │ 36.1 ms         │ 29.6 ms         │ 42.2 ms         │ 36.8 ms         │ 38.8 ms         │ 39.8 ms         │ 42.0 ms         │ 10  │
├──────────────────────────┼─────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┼─────┤
│ ITL                      │ total   │ 88.0 ms         │ 0.0 ms          │ 1866.4 ms       │ 89.9 ms         │ 92.0 ms         │ 93.9 ms         │ 110.2 ms        │ 10  │
├──────────────────────────┼─────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┼─────┤
│ InputTokens              │ total   │ 32744.3         │ 32629.0         │ 32923.0         │ 32733.5         │ 32801.25        │ 32867.2         │ 32917.42        │ 10  │
├──────────────────────────┼─────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┼─────┤
│ OutputTokens             │ total   │ 2048.0          │ 2048.0          │ 2048.0          │ 2048.0          │ 2048.0          │ 2048.0          │ 2048.0          │ 10  │
├──────────────────────────┼─────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┼─────┤
│ OutputTokenThroughput    │ total   │ 26.8428 token/s │ 22.8662 token/s │ 32.0627 token/s │ 26.0953 token/s │ 27.7358 token/s │ 31.9176 token/s │ 32.0482 token/s │ 10  │
╘══════════════════════════╧═════════╧═════════════════╧═════════════════╧═════════════════╧═════════════════╧═════════════════╧═════════════════╧═════════════════╧═════╛
╒══════════════════════════╤═════════╤════════════════════╕
│ Common Metric            │ Stage   │ Value              │
╞══════════════════════════╪═════════╪════════════════════╡
│ Benchmark Duration       │ total   │ 771444.3698 ms     │
├──────────────────────────┼─────────┼────────────────────┤
│ Total Requests           │ total   │ 10                 │
├──────────────────────────┼─────────┼────────────────────┤
│ Failed Requests          │ total   │ 0                  │
├──────────────────────────┼─────────┼────────────────────┤
│ Success Requests         │ total   │ 10                 │
├──────────────────────────┼─────────┼────────────────────┤
│ Concurrency              │ total   │ 1.0                │
├──────────────────────────┼─────────┼────────────────────┤
│ Max Concurrency          │ total   │ 1                  │
├──────────────────────────┼─────────┼────────────────────┤
│ Request Throughput       │ total   │ 0.013 req/s        │
├──────────────────────────┼─────────┼────────────────────┤
│ Total Input Tokens       │ total   │ 327443             │
├──────────────────────────┼─────────┼────────────────────┤
│ Prefill Token Throughput │ total   │ 10163.3049 token/s │
├──────────────────────────┼─────────┼────────────────────┤
│ Total Generated Tokens   │ total   │ 20480              │
├──────────────────────────┼─────────┼────────────────────┤
│ Input Token Throughput   │ total   │ 424.4545 token/s   │
├──────────────────────────┼─────────┼────────────────────┤
│ Output Token Throughput  │ total   │ 26.5476 token/s    │
├──────────────────────────┼─────────┼────────────────────┤
│ Total Token Throughput   │ total   │ 451.0021 token/s   │
╘══════════════════════════╧═════════╧═════════════════
```


  
  需要注意：

- PD分离需要读取`/etc/hccn.conf`文件，确保将物理机上的该文件映射到了容器中

- `etcd_addr`需与`xllm_service`的`etcd_addr`相同
  测试命令和上面类似，注意`curl http://localhost:{PORT}/v1/chat/completions ...`的`PORT`选择为启动xLLM service的`http_server_port`。

- 多机部署P或者Q时(例如部署两个P)，需要增加--rank_tablefile来完成通信。

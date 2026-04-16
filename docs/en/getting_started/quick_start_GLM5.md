# Using xLLM to Infer the GLM-5.0-W8A8 Base Model on Ascend A3 Devices

+ Source Code: https://github.com/jd-opensource/xllm

+ Available in China: https://gitcode.com/xLLM-AI/xllm

+ Weights Download: [modelscope-GLM-5-W8A8](https://www.modelscope.cn/models/Eco-Tech/GLM-5-W8A8-xLLM/files)
  
## 1. Pull the Docker Image Environment

First, download the image provided by xLLM:

```bash
# A2 x86
docker pull quay.io/jd_xllm/xllm-ai:xllm-dev-a2-x86-20260306
# A2 arm
docker pull quay.io/jd_xllm/xllm-ai:xllm-dev-a2-arm-20260306
# A3 arm
docker pull quay.io/jd_xllm/xllm-ai:xllm-dev-a3-arm-20260306
```

**Note**: The performance of A2 machines has not been stress-tested.

Then create the corresponding container:

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

## 2. Pull Source Code and Compile

Download the official repository and module dependencies:

```bash
git clone https://github.com/jd-opensource/xllm
cd xllm 
git checkout preview/glm-5
git submodule init
git submodule update
```

Install dependencies:

```bash
pip install --upgrade pre-commit
yum install numactl
```

Execute compilation to generate the executable `build/xllm/core/server/xllm` under `build/`:

```bash
python setup.py build
```

## 3. Start the Model

### If the machine is starting the service for the first time after a reboot, initialize the device by executing the following script first

# Failure to execute this and uninitialized NPU may cause the xllm process startup to fail

```bash
python -c "import torch_npu
for i in range(16):torch_npu.npu.set_device(i)"
```

### Environment Variables

```bash
##### 1. Configure environment variables related to dependency paths
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

##### 2. Configure environment variables related to logging
rm -rf /root/ascend/log/
rm -rf core.*

##### 3. Configure environment variables related to performance and communication
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

## Startup Command - GLM-5 (W8A8 weights can be started on a single machine)

```bash
BATCH_SIZE=256
# Maximum inference batch size
XLLM_PATH="./myxllm/xllm/build/xllm/core/server/xllm"
# Path to the inference entry file (compiled artifact from the previous step)
MODEL_PATH=/path/to/GLM-5-W8A8/
# Model path (here is the int8 quantized Glm-5)
DRAFT_MODEL_PATH=/path/to/GLM-5-W8A8/GLM-5-W8A8-MTP/
# MTP weights exported from Glm-5

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

# numactl -C xxxxx          Affinity binding to cores (NUMA affinity query command: npu-smi info -t topo)
#--max_memory_utilization   Maximum VRAM usage ratio per card
#--max_tokens_per_batch     Maximum number of tokens per batch (mainly limits prefill)
#--max_seqs_per_batch       Maximum number of requests per batch (mainly limits decode)
#--communication_backend    Communication backend options (hccl / lccl), hccl is recommended here
#--enable_schedule_overlap  Enable asynchronous scheduling
#--enable_prefix_cache      Enable prefix_cache
#--enable_chunked_prefill   Enable chunked_prefill
#--enable_graph             Enable aclgraph
#--draft_model              mtp - path to mtp weights
#--draft_devices            mtp - mtp inference device (same as the main model)
#--num_speculative_tokens   mtp - number of predicted tokens
```

The appearance of "Brpc Server Started" in the logs indicates that the service has started successfully.

## Other Optional Environment Variables

```bash
# Enable deterministic computation
export LCCL_DETERMINISTIC=1
export HCCL_DETERMINISTIC=true
export ATB_MATMUL_SHUFFLE_K_ENABLE=0

# # Enable dynamic profiling mode
# export PROFILING_MODE=dynamic
# \rm -rf ~/dynamic_profiling_socket_*
```

## Startup Command - Two-Machine Startup Example

### Node0 (master)

```bash
MASTER_NODE_ADDR="$master_ip:$master_port"
LOCAL_HOST="$local_ip"
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
MASTER_NODE_ADDR="$master_ip:$master_port"
LOCAL_HOST="$local_ip"
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

#### ranktable Example

 ranktable configuration guide: https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/hccl/hcclug/hcclug_000014.html

```json
{
    "version": "1.0",
    "server_count": "2",
    "server_list": [
        {
            "server_id": "$server_id",
            "device": [
                {
                    "device_id": "0",
                    "device_ip": "$device_ip_0",
                    "rank_id": "0"
                },
                ...
                {
                    "device_id": "7",
                    "device_ip": "$device_ip_7",
                    "rank_id": "7"
                }
            ],
            "host_nic_ip": "reserve"
        },
        {
            "server_id": "$server_id",
            "device": [
                {
                    "device_id": "0",
                    "device_ip": "$device_ip_0",
                    "rank_id": "8"
                },
                ...
                {
                    "device_id": "7",
                    "device_ip": "$device_ip_7",
                    "rank_id": "15"
                }
            ],
            "host_nic_ip": "reserve"
        }
    ],
    "status": "completed"
}
```


## Viewing Device NUMA Affinity

Command:

```bash
npu-smi info -t topo
```

In the preceding commands

```bash
numactl -C $((DEVICE*12))-$((DEVICE*12+11))
```

indicates that the process is bound to the corresponding affine cores. You can modify the bound core IDs according to the specific machine configuration.

## EX3. Glm-5 Weight Quantization 

### Install msmodelslim

```bash
git clone https://gitcode.com/shenxiaolong/msmodelslim.git 
cd msmodelslim
bash install.sh
```

### Modify tokenizer_config.json

```bash
  "extra_special_tokens" 
    Change to "additional_special_tokens"

  "tokenizer_class": "TokenizersBackend" 
    Change to "tokenizer_class": "PreTrainedTokenizer"
```

### Quantize GLM-5-BF16 Weights to W8A8 Weights

```bash
### Preprocess mtp related weights
python example/GLM5/extract_mtp.py --model-dir ${model_path}

# Specify transformers version
pip install transformers==4.48.2

# Execute quantization (generate quantized weights)
msmodelslim quant  --model_path ${model_path}  --save_path ${save_path}  --model_type DeepSeek-V3.2  --quant_type w8a8  --trust_remote_code True

# Copy chat_template file
cp ${model_path}/chat_template.jinja ${save_path}

# Export quantized mtp weights (for xllm inference)
python example/GLM5/export_mtp.py --input-dir  ${int8_save_path} --output-dir  ${mtp_save_path}
```

## PD Separation

### Installation of etcd\xllm-service

#### PD Separation Deployment

`xllm` supports PD separation deployment, which requires using another open-source library [xllm service](https://github.com/jd-opensource/xllm-service).

##### xLLM Service Dependencies

First, we download and install `xllm service`, similar to installing and compiling `xllm`:

```bash
git clone https://github.com/jd-opensource/xllm-service
cd xllm_service
git submodule init
git submodule update
```

##### etcd Installation

`xllm_service` depends on [etcd](https://github.com/etcd-io/etcd). Use the [installation script](https://github.com/etcd-io/etcd/releases) provided by the official etcd team for installation. The default installation path provided by the script is `/tmp/etcd-download-test/etcd`. You can manually modify the installation path in the script or manually move it after running the script:

```bash
mv /tmp/etcd-download-test/etcd /path/to/your/etcd
```

##### xLLM Service Compilation

Apply the patch first:

```bash
sh prepare.sh
```

Then execute compilation:

```bash
mkdir -p build
cd build
cmake ..
make -j 8
cd ..
```

!!! warning "Possible Errors"
    You may encounter installation errors regarding `boost-locale` and `boost-interprocess`: `vcpkg-src/packages/boost-locale_x64-linux/include: No such     file or directory`,`/vcpkg-src/packages/boost-interprocess_x64-linux/include: No such file or directory`
    Use `vcpkg` to reinstall these packages:
    ```bash
    /path/to/vcpkg remove boost-locale boost-interprocess
    /path/to/vcpkg install boost-locale:x64-linux
    /path/to/vcpkg install boost-interprocess:x64-linux
    ```

### Running PD Separation

Start etcd:

```bash
./etcd-download-test/etcd --listen-peer-urls 'http://localhost:2390'  --listen-client-urls 'http://localhost:2389' --advertise-client-urls  'http://localhost:2391'
```

For cross-machine configuration, refer to the following for etcd:

```bash
/tmp/etcd-download-test/etcd --listen-peer-urls 'http://0.0.0.0:3390' --listen-client-urls 'http://0.0.0.0:3389' --advertise-client-urls 'http://11.87.191.82:3389'
```

Start xllm service:

```bash
ENABLE_DECODE_RESPONSE_TO_SERVICE=true ./xllm_master_serving --etcd_addr="127.0.0.1:12389" --http_server_port 28888 --rpc_server_port 28889 --tokenizer_path=/export/home/models/GLM-5-W8A8/
```

For cross-machine configuration, start xllm service:

```bash
ENABLE_DECODE_RESPONSE_TO_SERVICE=true ../xllm-service/build/xllm_service/xllm_master_serving --etcd_addr="$etcd_ip:$etcd_port" --http_server_port 38888 --rpc_server_port 38889 --tokenizer_path=/export/home/models/GLM-5-W8A8/
```
- Start Prefill Instance
```bash
  BATCH_SIZE=256
  # Maximum inference batch size
  XLLM_PATH="./myxllm/xllm/build/xllm/core/server/xllm"
  # Path to the inference entry file (compiled artifact from the previous step)
  MODEL_PATH=/export/home/models/GLM-5-w8a8/
  # Model path (here is the int quantized Glm-5)
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
  
  #--etcd_addr=$LOCAL_HOST:3389  Refer to the advertise-client-urls configuration in etcd
  #--instance_role=DECODE     PD configuration, DECODE\PREFILL
  ```

- Start Decode Instance
  
  ```bash
    BATCH_SIZE=256
  # Maximum inference batch size
  XLLM_PATH="./myxllm/xllm/build/xllm/core/server/xllm"
  # Path to the inference entry file (compiled artifact from the previous step)
  MODEL_PATH=/export/home/models/GLM-5-w8a8/
  # Model path (here is the int quantized Glm-5)
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
  
  #--etcd_addr=$LOCAL_HOST:3389  Refer to the advertise-client-urls configuration in etcd
  #--instance_role=DECODE     PD configuration, DECODE\PREFILL
  ```

# GLM5/CP Feature Stress Test Data (Optimal Configuration)
##  Stress Test Environment
* Hardware: Ascend 910C (A3) / 4 Pods
* Main Model: GLM5-W8A8 
* Draft Model: GLM5-W8A8-MTP
* PD Separation Configuration:
  * P Instance: cp_size = 16，dp_size = 1, ep_size = 1
  * D Instance: dp_size = 2， ep_size = 32
* xllm Version: release/v0.9.0（9be308aec60ea4a2dd799ee021ea42d608f4e67c - lastcommit）

## PD Separation Service Startup Script
### PD Separation 4-Machine Configuration  
#### Prefill Dual-Node Configuration
##### Prefill Node 1
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
    --draft_model $DRAFT_MODEL_PATH \ # Draft model
    --draft_devices="npu:$DEVICE" \
    --num_speculative_tokens 3 \ # Sampling rate
    --nnodes=$NNODES \
    --max_memory_utilization=0.7 \ # VRAM usage rate
    --block_size=128 \
    --max_seqs_per_batch=9000 \
    --max_tokens_per_batch=67000 \
    --communication_backend="hccl" \
    --enable_prefix_cache=false \
    --enable_chunked_prefill=false \
    --enable_schedule_overlap=false \
    --enable_disagg_pd=true \ # Enable PD separation
    --instance_role=PREFILL \
    --etcd_addr=$etcd_addr:$etcd_port \
    --transfer_listen_port=$((26000+i)) \
    --disagg_pd_port=7777 \
    --cp_size 16 \ # Enable CP
    --dp_size 1 \
    --ep_size 1 \
    --node_rank=$i \
    --rank_tablefile=/export/home/shifengmin.3/workspace/ranktable_9899_new.json \ # Prefill dual-machine inter-card communication routing table
    > $LOG_FILE 2>&1 &

done

tail -f log/node_0.log
```

##### Prefill Node 2
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
    --draft_model $DRAFT_MODEL_PATH \ # Draft model path
    --draft_devices="npu:$DEVICE" \
    --num_speculative_tokens 3 \ # Sampling rate
    --nnodes=$NNODES \
    --max_memory_utilization=0.7 \ # VRAM usage rate
    --block_size=128 \
    --max_seqs_per_batch=9000 \
    --max_tokens_per_batch=67000 \
    --communication_backend="hccl" \
    --enable_prefix_cache=false \
    --enable_chunked_prefill=false \
    --enable_schedule_overlap=false \
    --enable_disagg_pd=true \ # Enable PD separation
    --instance_role=PREFILL \
    --etcd_addr=$etcd_addr:$etcd_port \
    --transfer_listen_port=$((26100+i)) \
    --disagg_pd_port=7777 \
    --cp_size 16 \ # Enable CP
    --dp_size 1 \ 
    --ep_size 1 \
    --node_rank=$((i+LOCAL_NODES)) \
    --rank_tablefile=/export/home/shifengmin.3/workspace/ranktable_9899_new.json \
    > $LOG_FILE 2>&1 &

done

tail -f log/node_0.log
```
#### Decode Dual-Machine Configuration
##### Decode Node 1
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
    --model $MODEL_PATH \ # GLM5.0 weights
    --devices="npu:$DEVICE" \
    --port $PORT \
    --host $LOCAL_HOST \
    --master_node_addr=$MASTER_NODE_ADDR \ 
    --draft_model $DRAFT_MODEL_PATH \ # MTP weights
    --draft_devices="npu:$DEVICE" \
    --num_speculative_tokens 3 \ # Sampling rate
    --nnodes=$NNODES \
    --max_memory_utilization=0.80 \
    --block_size=128 \
    --max_seqs_per_batch=9000 \
    --communication_backend="hccl" \
    --enable_prefix_cache=false \
    --enable_chunked_prefill=false \
    --enable_schedule_overlap=true \ # Enable asynchronous scheduling
    --enable_shm=true \ # Enable shared memory
    --enable_graph=false \
    --enable_graph_mode_decode_no_padding=false \
    --enable_disagg_pd=true \ # Enable PD separation
    --instance_role=DECODE \
    --etcd_addr=$etcd_addr:$etcd_port \
    --transfer_listen_port=$((26000+i)) \
    --disagg_pd_port=7777 \
    --dp_size 2 \ # dp parallelism
    --ep_size 32 \ # EP parallelism
    --node_rank=$i \
    --rank_tablefile=/export/home/shifengmin.3/workspace/ranktable_8382_new.json \ # Set inter-card communication
    > $LOG_FILE 2>&1 &

done

tail -f log/node_0.log
```
##### Decode Node-2
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
    --draft_model $DRAFT_MODEL_PATH \ # Draft model
    --draft_devices="npu:$DEVICE" \
    --num_speculative_tokens 3 \ # Sampling rate
    --nnodes=$NNODES \
    --max_memory_utilization=0.80 \ # VRAM usage rate 80%
    --block_size=128 \
    --max_seqs_per_batch=9000 \
    --communication_backend="hccl" \
    --enable_prefix_cache=false \
    --enable_chunked_prefill=false \
    --enable_schedule_overlap=true \ # Enable asynchronous scheduling
    --enable_shm=true \ # Enable shared memory
    --enable_graph=false \
    --enable_graph_mode_decode_no_padding=false \
    --enable_disagg_pd=true \ # PD separation
    --instance_role=DECODE \ #  decode node
    --etcd_addr=$etcd_ip:$etcd_port \
    --transfer_listen_port=$((26100+i)) \
    --disagg_pd_port=7777 \
    --dp_size 2 \ # Enable dp
    --ep_size 32 \ # Enable ep
    --node_rank=$((i+LOCAL_NODES)) \
    --rank_tablefile=/export/home/shifengmin.3/workspace/ranktable_8382_new.json \ # Dual-machine inter-communication routing table
    > $LOG_FILE 2>&1 &

done

tail -f log/node_0.log
```

## Stress Testing
### Custom Dataset - Input/Output Configuration
Modified Location：/benchmark/ais_bench/datasets/synthetic/synthetic_config.py
```
#
# [Uniform Distribution] -- "Method" : "uniform"
#   - MinValue: Minimum value, range [1, 2**20]
#   - MaxValue: Maximum value, range [1, 2**20], can be equal to MinValue
#
# [Gaussian Distribution] -- "Method" : "gaussian"
#   - Mean    : Mean, range [-3.0e38, 3.0e38], center of distribution
#   - Var     : Variance, range [0, 3.0e38], controls data dispersion
#   - MinValue: Minimum value, range [1, 2**20], can be lower than Mean
#   - MaxValue: Maximum value, range [1, 2**20], can be higher than Mean, can be equal to MinValue
#
# [Zipf Distribution] -- "Method" : "zipf"
#   - Alpha   : Shape parameter, range (1.0,10.0], larger values result in more uniform distribution
#   - MinValue: Minimum value, range [1, 2**20]
#   - MaxValue: Maximum value, range [1, 2**20], must be greater than MinValue
"""
synthetic_config = {
    "Type":"tokenid",   # [tokenid/string], type of generated random dataset, supports fixed-length random tokenid and random-length string datasets
    "RequestCount": 10, # Number of generated requests, should match decode_batch_size in the model-side configuration file
    "TrustRemoteCode": False, # Whether to trust remote code, tokenizer needs to be loaded to generate tokenid in tokenid mode, default is False
    "StringConfig" : {  # Configuration items for string-type random dataset, please refer to the comments above: "Parameter description for random generation methods in StringConfig"
        "Input" : {     # Input length of each request
            "Method": "uniform",
            "Params": {"MinValue": 16384, "MaxValue": 16384}
        },
        "Output" : {    # Output length of each request
            "Method": "gaussian",
            "Params": {"Mean": 1024, "Var": 0, "MinValue": 1024, "MaxValue": 1024}
        }
    },
    "TokenIdConfig" : { # Configuration items for tokenid-type random dataset
        "RequestSize": 16384 # Length of each request, i.e., the number of token ids in each request, should match input_seq_len in the model-side configuration file
    }
}
```
### ais_bench Client Settings
Update Location：/benchmark/ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py
```
from ais_bench.benchmark.models import VLLMCustomAPIChatStream
from ais_bench.benchmark.utils.model_postprocessors import extract_non_reasoning_content

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChatStream,
        abbr='vllm-api-stream-chat',
        path="[$GLM5_weight]", # GLM5 w8a8 weights
        model="[$GLM5_mtp_weight]", # GLM5-MTP weights
        request_rate = 0,
        retry = 1,
        host_ip = "[$server_ip]", # Inference service IP
        host_port = [$server_port], # Inference service port 
        max_out_len = 1024, # Number of output tokens
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

### ais_bench Client Initiates Stress Test
```
ais_bench --models vllm_api_stream_chat --datasets synthetic_gen -m perf
```

## Stress Test Data
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
│ Total Generated Tokens   │
```

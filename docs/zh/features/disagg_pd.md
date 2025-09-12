# PD分离
## 背景
LLM在线推理服务通常需要满足TTFT和TPOT两项性能指标，而传统的Contiguous Batching调度策略将Prefill和Decode请求混合在一起调度，导致P和D会互相抢占计算资源，影响性能指标无法最大程度的利用计算资源。为解决上述矛盾，将Prefill和Decode两阶段拆分到独立的计算资源并行执行，从而同时降低TTFT和TPOT并提升吞吐量。
## 功能介绍
xLLM PD分离功能主要通过以下三个模块实现：

- **etcd**: 存储实例信息等元数据
- **xLLM Service**: 调度请求和管理所有计算实例
- **xLLM**: 请求计算实例

整体架构图如下：
![xLLM PD分离架构图](../../assets/pd_architecture.jpg)
## 功能使用示例
### 使用准备
#### 安装相关依赖
- **xLLM**: 参见[安装编译](../getting_started/compile.md)
- **xLLM Service**: 参见[PD分离部署](../getting_started/PD_disagg.md)

#### 获取环境信息
部署xLLM PD分离需要获取机器的Device IP以创建相关通信资源，可以在当前AI Server执行指令`cat /etc/hccn.conf | grep address`获取Device IP，例如：
```
address_0=xx.xx.xx.xx
address_1=xx.xx.xx.xx
```
address_xx即为Device IP。
### 启动PD分离服务
1. 启动etcd
```
./etcd
```
2. 启动xLLM Service
```bash
ENABLE_DECODE_RESPONSE_TO_SERVICE=true ./xllm_master_serving --etcd_addr="127.0.0.1:12389" --http_server_port 28888 --rpc_server_port 28889 --tokenizer_path=/path/to/tokenizer_config_dir/
```
3. 启动xLLM

4. 以Qwen2-7B为例
    - 启动Prefill实例
        ``` shell linenums="1" hl_lines="3 9 10"
        ./xllm --model=Qwen2-7B-Instruct \
               --port=8010 \
               --devices="npu:0" \
               --master_node_addr="127.0.0.1:18888" \
               --enable_prefix_cache=false \
               --enable_chunked_prefill=false \
               --enable_disagg_pd=true \
               --instance_role=PREFILL \
               --etcd_addr=127.0.0.1:12389 \
               --device_ip=xx.xx.xx.xx \ # 替换为实际的Device IP
               --transfer_listen_port=26000 \
               --disagg_pd_port=7777 \
               --node_rank=0 \
               --nnodes=1
        ```
    - 启动Decode实例
        ``` shell linenums="1" hl_lines="3 9 10"
        ./xllm --model=Qwen2-7B-Instruct \
               --port=8020 \
               --devices="npu:1" \
               --master_node_addr="127.0.0.1:18898" \
               --enable_prefix_cache=false \
               --enable_chunked_prefill=false \
               --enable_disagg_pd=true \
               --instance_role=DECODE \
               --etcd_addr=127.0.0.1:12389 \
               --device_ip=xx.xx.xx.xx \ # 替换为实际的Device IP
               --transfer_listen_port=26100 \
               --disagg_pd_port=7787 \
               --node_rank=0 \
               --nnodes=1
        ```
    需要注意：
    
    - PD分离在指定NPU Device的时候，需要对应的Device IP，这个每张卡是不一样的
  
    - `etcd_addr`需与`xllm_service`的`etcd_addr`相同

!!! warning "注意事项"
    PD分离目前不支持开启prefix cache及chunked prefill功能，需要通过以下参数关闭
    ``` shell
    --enable_prefix_cache=false
    --enable_chunked_prefill=false
    ```

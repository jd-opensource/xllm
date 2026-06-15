# xLLM CLI 参数详解

## 1. 概述

xLLM 使用 gflags 进行命令行参数解析，所有参数可以通过以下方式设置：

```bash
# 方式 1: 命令行参数
xllm --model /path/to/model --port 8010 --device_id 0

# 方式 2: 配置文件 (JSON)
xllm --config_json_file config.json

# 方式 3: 环境变量
export XLLM_MODEL=/path/to/model
xllm
```

参数优先级：**命令行 > 配置文件 > 默认值**

---

## 2. 模型配置参数

### 2.1 核心模型参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| **`--model`** | string | "" | **[必需]** 模型路径，包含权重文件和 config.json 的目录 |
| **`--model_id`** | string | "" | 模型名称标识符，用于服务路由和日志 |
| **`--backend`** | string | "" | 模型类型：`llm` (文本)、`vlm` (多模态)、`dit` (扩散)、`rec` (推荐) |
| **`--task`** | string | "generate" | 任务类型：`generate` (生成)、`embed` (嵌入)、`mm_embed` (多模态嵌入) |

**示例**：
```bash
# LLM 文本生成
xllm --model /models/Qwen2-7B --backend llm --task generate

# VLM 多模态
xllm --model /models/Qwen2-VL --backend vlm --task generate

# Embedding 模型
xllm --model /models/bge-large --backend llm --task embed
```

### 2.2 设备参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| **`--device_id`** | int32 | -1 | **[推荐]** 设备 ID，如 `0` 使用第一个 GPU/NPU |
| **`--devices`** | string | "" | **[已废弃]** 设备列表，如 `npu:0,npu:1,npu:2` |

**示例**：
```bash
# 单卡
xllm --model /models/Qwen2-7B --device_id 0

# 多卡（建议使用 --device_id 多次启动）
xllm --model /models/Qwen2-7B --device_id 0 --port 8010 &
xllm --model /models/Qwen2-7B --device_id 1 --port 8011 &
```

### 2.3 多模态参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| **`--limit_image_per_prompt`** | int32 | 8 | 每个请求最大图片数量 (仅 VLM) |
| **`--max_encoder_cache_size`** | int64 | 0 | Encoder cache 大小 (MB)，0 表示禁用 |
| **`--use_audio_in_video`** | bool | false | 视频输入时是否同时处理音频 |
| **`--enable_return_mm_full_embeddings`** | bool | false | 返回完整 VLM embeddings (ViT + sequence) |

**示例**：
```bash
# VLM 配置
xllm --model /models/Qwen2-VL --backend vlm \
     --limit_image_per_prompt 16 \
     --max_encoder_cache_size 1024
```

### 2.4 工具调用和推理

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| **`--tool_call_parser`** | string | "" | Tool call 解析器：`auto`、`qwen25`、`qwen3`、`deepseekv3`、`glm45` |
| **`--reasoning_parser`** | string | "" | Reasoning 解析器：`auto`、`qwen3`、`deepseek-r1`、`glm5` |
| **`--enable_qwen3_reranker`** | bool | false | 启用 Qwen3 reranker 功能 |
| **`--use_cpp_chat_template`** | bool | true | 使用 C++ chat template (比 Jinja 更快) |

**示例**：
```bash
# 启用 tool call
xllm --model /models/Qwen2-7B \
     --tool_call_parser qwen25

# 启用 reasoning
xllm --model /models/DeepSeek-R1 \
     --reasoning_parser deepseek-r1
```

---

## 3. KV Cache 配置参数

### 3.1 内存管理

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| **`--block_size`** | int32 | 128 | KV cache block 大小 (token 数) |
| **`--max_cache_size`** | int64 | 0 | 最大 KV cache 大小 (bytes)，0 表示自动计算 |
| **`--max_memory_utilization`** | double | 0.8 | GPU/NPU 内存利用率上限 |
| **`--kv_cache_dtype`** | string | "auto" | KV cache 数据类型：`auto` (跟随模型)、`int8` (量化) |
| **`--enable_prefix_cache`** | bool | true | 启用 prefix cache (共享相同前缀) |
| **`--enable_xtensor`** | bool | false | 启用 xtensor (连续内存池) |
| **`--phy_page_granularity_size`** | int64 | 2MB | Physical page 大小 (仅 enable_xtensor=true) |

**示例**：
```bash
# 手动指定 KV cache 大小 (8GB)
xllm --model /models/Qwen2-7B \
     --max_cache_size 8589934592 \
     --max_memory_utilization 0.85

# 启用 INT8 KV cache 量化
xllm --model /models/Qwen2-7B \
     --kv_cache_dtype int8
```

**内存分配公式**：
```
可用内存 = GPU_total_memory * max_memory_utilization
KV cache = 可用内存 - model_weights - activation_memory
Block 数量 = KV cache / (block_size * hidden_dim * dtype_size * 2 * n_layers)
```

---

## 4. 调度器配置参数

### 4.1 Batch 配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| **`--max_tokens_per_batch`** | int32 | 10240 | 每个 batch 最大 token 数 |
| **`--max_seqs_per_batch`** | int32 | 1024 | 每个 batch 最大序列数 |
| **`--enable_chunked_prefill`** | bool | true | 启用 chunked prefill (分块处理长 prompt) |
| **`--max_tokens_per_chunk_for_prefill`** | int32 | 512 | Chunked prefill 每个 chunk 的 token 数 |
| **`--enable_schedule_overlap`** | bool | true | 启用调度重叠 (prefill 和 decode 并行) |

**示例**：
```bash
# 高吞吐配置 (大 batch)
xllm --model /models/Qwen2-7B \
     --max_tokens_per_batch 20480 \
     --max_seqs_per_batch 512

# 低延迟配置 (小 batch)
xllm --model /models/Qwen2-7B \
     --max_tokens_per_batch 4096 \
     --max_seqs_per_batch 128
```

### 4.2 调度策略

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| **`--priority_strategy`** | string | "fcfs" | 优先级策略：`fcfs` (先来先服务)、`priority` (按优先级) |
| **`--enable_online_preempt_offline`** | bool | true | Online 请求可抢占 offline 请求 |
| **`--prefill_scheduling_memory_usage_threshold`** | double | 0.8 | Prefill 内存阈值 |
| **`--use_zero_evict`** | bool | false | 启用 zero evict 策略 |

---

## 5. 并行配置参数

### 5.1 并行策略

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| **`--dp_size`** | int32 | 1 | 数据并行大小 (DP) |
| **`--ep_size`** | int32 | 1 | 专家并行大小 (EP，用于 MoE) |
| **`--cp_size`** | int32 | 1 | Context 并行大小 (CP，用于长序列) |
| **`--tp_size`** | int64 | 1 | Tensor 并行大小 (TP，仅 DiT) |
| **`--sp_size`** | int64 | 1 | Sequence 并行大小 (SP，仅 DiT) |
| **`--kv_split_size`** | int32 | 1 | KV cache 分片大小 |
| **`--enable_prefill_sp`** | bool | false | 启用 prefill-only sequence parallel |
| **`--enable_multi_stream_parallel`** | bool | false | 启用多流并行 (计算通信重叠) |

**示例**：
```bash
# MoE 模型 EP 配置
xllm --model /models/DeepSeek-V3 \
     --ep_size 8 \
     --device_id 0,1,2,3,4,5,6,7

# Context parallel for long sequence
xllm --model /models/Qwen2-7B \
     --cp_size 4 \
     --device_id 0,1,2,3
```

### 5.2 通信配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| **`--communication_backend`** | string | "hccl" | NPU 通信后端：`hccl`、`lccl` |
| **`--rank_tablefile`** | string | "" | HCCL rank table 文件路径 |

---

## 6. 推测解码配置参数

### 6.1 基础推测解码

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| **`--draft_model`** | string | "" | Draft 模型路径 |
| **`--draft_devices`** | string | "" | Draft 模型设备 |
| **`--num_speculative_tokens`** | int32 | 0 | 推测 token 数量 |
| **`--speculative_algorithm`** | string | "MTP" | 推测算法：`MTP`、`Suffix` |
| **`--enable_atb_spec_kernel`** | bool | false | 启用 ATB speculative kernel |

**示例**：
```bash
# MTP 推测解码
xllm --model /models/Qwen2-7B \
     --draft_model /models/Qwen2-7B-draft \
     --num_speculative_tokens 4 \
     --speculative_algorithm MTP

# Suffix 推测解码 (无需 draft model)
xllm --model /models/Qwen2-7B \
     --num_speculative_tokens 8 \
     --speculative_algorithm Suffix
```

### 6.2 Suffix 推测解码参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| **`--speculative_suffix_cache_max_depth`** | int32 | 64 | Suffix cache 最大深度 |
| **`--speculative_suffix_max_spec_factor`** | double | 1.0 | 最大推测因子 |
| **`--speculative_suffix_max_spec_offset`** | double | 0.0 | 最大推测偏移 |
| **`--speculative_suffix_min_token_prob`** | double | 0.1 | 最小 token 概率阈值 |
| **`--speculative_suffix_max_cached_requests`** | int32 | -1 | 最大缓存请求数 (-1 表示无限) |
| **`--speculative_suffix_use_tree_spec`** | bool | false | 使用 tree speculation |

---

## 7. EPLB 配置参数 (MoE 专家负载均衡)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| **`--enable_eplb`** | bool | false | 启用 expert parallel load balance |
| **`--redundant_experts_num`** | int32 | 1 | 每个设备的冗余专家数量 |
| **`--eplb_update_interval`** | int64 | 1000 | EPLB 更新间隔 (steps) |
| **`--eplb_update_threshold`** | double | 0.8 | EPLB 更新阈值 |
| **`--expert_parallel_degree`** | int32 | 0 | 专家并行度 |

**示例**：
```bash
# MoE EPLB 配置
xllm --model /models/DeepSeek-V3 \
     --enable_eplb \
     --redundant_experts_num 2 \
     --eplb_update_interval 500
```

---

## 8. 服务配置参数

### 8.1 HTTP/RPC 服务

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| **`--host`** | string | "" | 服务主机名 |
| **`--port`** | int32 | 8010 | HTTP 服务端口 |
| **`--num_threads`** | int32 | 8 | 处理请求的线程数 |
| **`--max_concurrent_requests`** | int32 | 100 | 最大并发请求数 |
| **`--num_request_handling_threads`** | int32 | 4 | 请求处理线程数 |
| **`--num_response_handling_threads`** | int32 | 4 | 响应处理线程数 |
| **`--rpc_channel_timeout_ms`** | int32 | 5000 | RPC 超时时间 (ms) |
| **`--max_reconnect_count`** | int32 | 3 | 最大重连次数 |
| **`--health_check_interval_ms`** | int32 | 30000 | 健康检查间隔 (ms) |

**示例**：
```bash
# 高并发服务
xllm --model /models/Qwen2-7B \
     --port 8010 \
     --num_threads 16 \
     --max_concurrent_requests 500

# 多实例部署
xllm --model /models/Qwen2-7B --port 8010 --device_id 0 &
xllm --model /models/Qwen2-7B --port 8011 --device_id 1 &
```

---

## 9. 分布式配置参数

### 9.1 多节点配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| **`--nnodes`** | int32 | 1 | 节点数量 |
| **`--node_rank`** | int32 | 0 | 当前节点 rank |
| **`--master_node_addr`** | string | "" | Master 节点地址 |
| **`--enable_service_routing`** | bool | false | 启用服务路由 |

**示例**：
```bash
# Node 0 (Master)
xllm --model /models/Qwen2-7B \
     --nnodes 2 --node_rank 0 \
     --master_node_addr 192.168.1.100:8010

# Node 1 (Worker)
xllm --model /models/Qwen2-7B \
     --nnodes 2 --node_rank 1 \
     --master_node_addr 192.168.1.100:8010 \
     --enable_service_routing
```

### 9.2 ETCD 配置 (服务发现)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| **`--etcd_addr`** | string | "" | ETCD 地址，如 `http://127.0.0.1:2379` |
| **`--etcd_namespace`** | string | "" | ETCD namespace |
| **`--instance_name`** | string | "" | 实例名称 |
| **`--heart_beat_interval`** | double | 0.5 | 心跳间隔 (s) |

---

## 10. P-D Disaggregation 配置参数

### 10.1 Prefill-Decode 分离

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| **`--enable_disagg_pd`** | bool | false | 启用 P-D disaggregation |
| **`--instance_role`** | string | "default" | 实例角色：`default`、`prefill`、`decode` |
| **`--kv_cache_transfer_mode`** | string | "PUSH" | KV cache 传输模式：`PUSH`、`PULL` |
| **`--transfer_listen_port`** | int32 | 26000 | KV cache 传输监听端口 |
| **`--disagg_pd_port`** | int32 | 7777 | Disagg PD 服务端口 |

**示例**：
```bash
# Prefill 实例
xllm --model /models/Qwen2-7B \
     --enable_disagg_pd \
     --instance_role prefill \
     --port 8010

# Decode 实例
xllm --model /models/Qwen2-7B \
     --enable_disagg_pd \
     --instance_role decode \
     --port 8011 \
     --master_node_addr prefill-host:8010
```

---

## 11. KV Cache Store 配置参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| **`--enable_kvcache_store`** | bool | false | 启用 KV cache 存储 |
| **`--enable_cache_upload`** | bool | false | 启用 cache 上传 |
| **`--host_blocks_factor`** | double | 0.0 | Host blocks 因子 |
| **`--prefetch_timeout`** | uint32 | 0 | Prefetch 超时 (ms) |
| **`--prefetch_batch_size`** | uint32 | 2 | Prefetch batch 大小 |
| **`--layers_wise_copy_batchs`** | uint32 | 4 | Layer-wise H2D copy batch 数 |
| **`--store_protocol`** | string | "tcp" | 存储协议 |
| **`--store_master_server_address`** | string | "" | 存储服务器地址 |

---

## 12. 性能优化配置参数

### 12.1 Graph 优化

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| **`--enable_graph`** | bool | false | 启用 CUDA Graph / ACL Graph |
| **`--enable_graph_mode_decode_no_padding`** | bool | false | Graph decode 无 padding |
| **`--enable_prefill_piecewise_graph`** | bool | false | Piecewise graph for prefill |
| **`--max_tokens_for_graph_mode`** | int32 | 2048 | Graph 模式最大 token 数 |
| **`--acl_graph_decode_batch_size_limit`** | int32 | 0 | ACL graph decode batch size 上限 |

**示例**：
```bash
# 启用 Graph 优化
xllm --model /models/Qwen2-7B \
     --enable_graph \
     --enable_graph_mode_decode_no_padding \
     --max_tokens_for_graph_mode 4096
```

### 12.2 Load 优化

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| **`--enable_manual_loader`** | bool | false | 启用手动加载器 |
| **`--enable_rolling_load`** | bool | false | 启用 rolling load (层权重动态加载) |
| **`--rolling_load_num_cached_layers`** | int32 | 2 | Rolling load 缓存层数 |
| **`--rolling_load_num_rolling_slots`** | int32 | 2 | Rolling load slots 数 |

**示例**：
```bash
# Rolling load (适合大模型低显存)
xllm --model /models/DeepSeek-V3 \
     --enable_rolling_load \
     --rolling_load_num_cached_layers 4 \
     --rolling_load_num_rolling_slots 4
```

---

## 13. Profile 配置参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| **`--enable_profile_step_time`** | bool | false | 启用 step time profiling |
| **`--enable_profile_token_budget`** | bool | false | 启用 token budget profiling |
| **`--enable_latency_aware_schedule`** | bool | false | 启用延迟感知调度 |
| **`--profile_max_prompt_length`** | int32 | 2048 | Profile 最大 prompt 长度 |
| **`--max_global_ttft_ms`** | int32 | MAX | 全局 TTFT 上限 (ms) |
| **`--max_global_tpot_ms`** | int32 | MAX | 全局 TPOT 上限 (ms) |
| **`--enable_profile_kv_blocks`** | bool | true | Profile KV blocks |
| **`--disable_ttft_profiling`** | bool | false | 禁用 TTFT profiling |
| **`--enable_forward_interruption`** | bool | false | 启用 forward 中断 |
| **`--profile_dir`** | string | "" | Profile 输出目录 |

**示例**：
```bash
# 启用性能 profiling
xllm --model /models/Qwen2-7B \
     --enable_profile_step_time \
     --enable_latency_aware_schedule \
     --profile_dir ./profile_results
```

---

## 14. Kernel 配置参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| **`--npu_kernel_backend`** | string | "AUTO" | NPU kernel backend：`AUTO`、`ATB`、`TORCH` |
| **`--enable_customize_mla_kernel`** | bool | false | 启用自定义 MLA kernel |
| **`--enable_intralayer_addnorm`** | bool | false | 启用 intra-layer add norm |
| **`--enable_interlayer_addnorm`** | bool | false | 启用 inter-layer add norm |
| **`--enable_split_rmsnorm_rope`** | bool | false | 启用 split RMSNorm + RoPE |
| **`--enable_fused_mc2`** | int32 | 0 | 启用 fused MC2 |

---

## 15. Beam Search 配置参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| **`--enable_beam_search_kernel`** | bool | false | 启用 beam search kernel |
| **`--beam_width`** | int32 | 1 | Beam width |
| **`--enable_block_copy_kernel`** | bool | false | 启用 block copy kernel |
| **`--enable_topk_sorted`** | bool | false | 启用 top-k sorted |

---

## 16. 推荐模型配置参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| **`--enable_rec_prefill_only`** | bool | false | 启用推荐 prefill-only 模式 |
| **`--enable_xattention_one_stage`** | bool | false | 启用 cross-attention 单阶段 |
| **`--max_decode_rounds`** | int32 | 1 | 最大 decode 轮数 |
| **`--enable_constrained_decoding`** | bool | false | 启用约束解码 |
| **`--enable_convert_tokens_to_item`** | bool | false | 启用 token 转 item |
| **`--rec_worker_max_concurrency`** | uint32 | 1 | Rec worker 最大并发 |

---

## 17. 其他配置参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| **`--enable_offline_inference`** | bool | false | 启用离线推理模式 |
| **`--spawn_worker_path`** | string | "" | Worker binary 路径 |
| **`--enable_shm`** | bool | false | 启用共享内存通信 |
| **`--input_shm_size`** | uint64 | 1024 | Input SHM 大小 (KB) |
| **`--output_shm_size`** | uint64 | 128 | Output SHM 大小 (KB) |
| **`--random_seed`** | int32 | -1 | 随机种子 (-1 表示随机) |
| **`--flashinfer_workspace_buffer_size`** | int32 | 128MB | Flashinfer workspace 大小 |

---

## 18. 配置文件示例

### 18.1 JSON 配置文件

```json
{
  "model": "/models/Qwen2-7B",
  "backend": "llm",
  "task": "generate",
  "device_id": 0,
  "port": 8010,
  
  "block_size": 128,
  "max_memory_utilization": 0.85,
  "enable_prefix_cache": true,
  
  "max_tokens_per_batch": 10240,
  "max_seqs_per_batch": 512,
  "enable_chunked_prefill": true,
  
  "enable_graph": true,
  "enable_schedule_overlap": true,
  
  "num_threads": 8,
  "max_concurrent_requests": 200
}
```

**使用方式**：
```bash
xllm --config_json_file config.json
```

### 18.2 多模态模型配置

```json
{
  "model": "/models/Qwen2-VL-7B",
  "backend": "vlm",
  "task": "generate",
  "device_id": 0,
  
  "limit_image_per_prompt": 16,
  "max_encoder_cache_size": 2048,
  
  "block_size": 128,
  "max_memory_utilization": 0.9,
  
  "tool_call_parser": "qwen25"
}
```

### 18.3 高吞吐配置

```json
{
  "model": "/models/Qwen2-7B",
  "backend": "llm",
  "device_id": 0,
  
  "max_tokens_per_batch": 20480,
  "max_seqs_per_batch": 1024,
  "enable_chunked_prefill": true,
  "max_tokens_per_chunk_for_prefill": 1024,
  
  "enable_schedule_overlap": true,
  "enable_prefix_cache": true,
  
  "num_threads": 16,
  "max_concurrent_requests": 500
}
```

---

## 19. 参数分类总结

### 19.1 必需参数

| 参数 | 说明 |
|------|------|
| `--model` | 模型路径 |
| `--device_id` | 设备 ID |

### 19.2 核心性能参数

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--max_memory_utilization` | 内存利用率 | 0.85-0.9 |
| `--max_tokens_per_batch` | Batch token 数 | 10240-20480 |
| `--enable_chunked_prefill` | Chunked prefill | true |
| `--enable_schedule_overlap` | 调度重叠 | true |
| `--enable_prefix_cache` | Prefix cache | true |

### 19.3 高级优化参数

| 参数 | 说明 | 适用场景 |
|------|------|----------|
| `--enable_graph` | Graph 优化 | Decode 模式加速 |
| `--enable_rolling_load` | Rolling load | 大模型低显存 |
| `--enable_eplb` | Expert load balance | MoE 模型 |
| `--num_speculative_tokens` | 推测解码 | 加速 token 生成 |

### 19.4 分布式参数

| 参数 | 说明 | 适用场景 |
|------|------|----------|
| `--nnodes` | 多节点 | 分布式推理 |
| `--enable_disagg_pd` | P-D 分离 | Prefill-Decode disaggregation |
| `--dp_size` | 数据并行 | 多卡吞吐提升 |
| `--ep_size` | 专家并行 | MoE 模型 |

---

## 20. 常见使用场景

### 20.1 单卡部署

```bash
xllm --model /models/Qwen2-7B \
     --device_id 0 \
     --port 8010 \
     --max_memory_utilization 0.85
```

### 20.2 多卡数据并行

```bash
# 4 卡 DP
xllm --model /models/Qwen2-7B \
     --dp_size 4 \
     --device_id 0 \
     --port 8010 \
     --enable_service_routing \
     --etcd_addr http://127.0.0.1:2379
```

### 20.3 MoE 模型部署

```bash
xllm --model /models/DeepSeek-V3 \
     --backend llm \
     --device_id 0,1,2,3,4,5,6,7 \
     --ep_size 8 \
     --enable_eplb \
     --redundant_experts_num 2
```

### 20.4 多模态模型部署

```bash
xllm --model /models/Qwen2-VL \
     --backend vlm \
     --device_id 0 \
     --limit_image_per_prompt 20 \
     --max_encoder_cache_size 2048
```

### 20.5 推测解码

```bash
# MTP 推测解码
xllm --model /models/Qwen2-7B \
     --draft_model /models/Qwen2-7B-draft \
     --num_speculative_tokens 5

# Suffix 推测解码
xllm --model /models/Qwen2-7B \
     --speculative_algorithm Suffix \
     --num_speculative_tokens 8
```

---

## 21. 参数调优建议

### 21.1 吞吐优先

```bash
--max_tokens_per_batch 20480
--max_seqs_per_batch 512
--enable_chunked_prefill true
--enable_schedule_overlap true
--enable_prefix_cache true
--num_threads 16
```

### 21.2 延迟优先

```bash
--max_tokens_per_batch 4096
--max_seqs_per_batch 128
--enable_chunked_prefill true
--max_tokens_per_chunk_for_prefill 256
--enable_schedule_overlap false
```

### 21.3 内存受限

```bash
--max_memory_utilization 0.7
--enable_rolling_load true
--rolling_load_num_cached_layers 2
--block_size 64
```

---

## 22. 常见问题排查

### Q1: `--model` 参数必需吗？

**是的**，必须指定模型路径，路径中必须包含：
- `config.json` (模型配置)
- 权重文件 (`*.safetensors` 或 `*.bin`)
- tokenizer 文件

### Q2: `--devices` 和 `--device_id` 区别？

`--devices` 已废弃，推荐使用 `--device_id`：
```bash
# 旧方式 (已废弃)
xllm --devices npu:0,npu:1

# 新方式 (推荐)
xllm --device_id 0  # 单卡
# 多卡使用 DP 配置
xllm --device_id 0 --dp_size 2
```

### Q3: 如何确定 `--max_memory_utilization`？

建议值：
- CUDA: 0.85-0.9
- NPU: 0.8-0.85
- MLU: 0.85

过高可能导致 OOM，过低浪费显存。

### Q4: `--enable_chunked_prefill` vs `--enable_schedule_overlap`？

- `enable_chunked_prefill`: 将长 prompt 分块处理
- `enable_schedule_overlap`: Prefill 和 Decode 并行执行

两者可以同时启用以最大化吞吐。

### Q5: 如何选择推测解码算法？

| 算法 | 适用场景 | 特点 |
|------|---------|------|
| **MTP** | 有 draft model | 高准确性，需额外模型 |
| **Suffix** | 无 draft model | 无需额外模型，基于 suffix cache |

---

## 23. 完整参数列表速查

| 分类 | 核心参数 |
|------|---------|
| **模型** | `model`, `backend`, `task`, `device_id` |
| **内存** | `block_size`, `max_memory_utilization`, `max_cache_size` |
| **调度** | `max_tokens_per_batch`, `max_seqs_per_batch`, `enable_chunked_prefill` |
| **并行** | `dp_size`, `ep_size`, `cp_size` |
| **优化** | `enable_graph`, `enable_prefix_cache`, `enable_schedule_overlap` |
| **服务** | `port`, `num_threads`, `max_concurrent_requests` |
| **推测** | `draft_model`, `num_speculative_tokens`, `speculative_algorithm` |
| **分布式** | `nnodes`, `enable_disagg_pd`, `enable_service_routing` |

---

## 24. 总结

xLLM 提供了丰富的 CLI 参数，覆盖：
- **模型配置**: 模型加载、设备管理、多模态
- **内存管理**: KV cache、prefix cache、内存利用率
- **调度策略**: Batch 配置、chunked prefill、优先级
- **并行策略**: DP/EP/CP/TP/SP
- **性能优化**: Graph、rolling load、推测解码
- **分布式**: 多节点、P-D disaggregation、服务路由

**推荐配置模板**：

```bash
# 通用高性能配置
xllm --model /models/YOUR_MODEL \
     --backend llm \
     --device_id 0 \
     --port 8010 \
     --max_memory_utilization 0.85 \
     --enable_prefix_cache \
     --enable_chunked_prefill \
     --enable_schedule_overlap \
     --num_threads 8
```
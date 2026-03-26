---
hide:
  - navigation
---


# 服务启动参数

xLLM使用gflags来管理服务启动参数，具体的参数含义如下：

## 常用参数
| 参数名称 | 类型 | 默认值 | 其他值 | 参数含义 | 其他 |
|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|
| `master_node_addr` | `string` | "127.0.0.1:19888" | ip:port | master节点rpc server的监听地址 | [详情](./features/basics.md) |
| `host` | `string` | "" | 当前device所在的机器ip | 当前device用于通信的host ip，每个device上会启动一个rpc server，用于多卡之间通信 |  |
| `port` | `int32` | 8010 | 任意可用的端口 | 与host参数配套使用，组合后用于device间的rpc通信 |  |
| `model` | `string` | "" |  | 模型所在的路径 |  |
| `devices` | `string` | "npu:0" |  | 指定当前进程使用的NPU设备 |  |
| `nnodes` | `int32` | 1 |  | 当前服务所使用的device总数 |  |
| `node_rank` | `int32` | 0 | 0 ~ device总数减1 | 每个device的rank id |  |
| `max_memory_utilization` | `double` | 0.8 | 0-1之间 | 模型权重和KV Cache一起可用的最大device memory占比 |  |
| `max_tokens_per_batch` | `int32` | 10240 |  | 每个step可计算的最大token数量 |  |
| `max_seqs_per_batch` | `int32` | 1024 |  | 每个step可计算的最大sequence数量 |  |
| `enable_chunked_prefill` | `bool` | true | false | 是否开启chunked prefill |  |
| `enable_prefill_sp` | `bool` | false | true | 是否开启 prefill 阶段的 sequence parallel | 支持 `enable_chunked_prefill=true`，但仅限纯 prefill batch（`PREFILL` / `CHUNKED_PREFILL`）；`MIXED` 与 `DECODE` batch 不会进入 sequence parallel。 |
| `enable_schedule_overlap` | `bool` | false | true | 是否开启异步调度 | [详情](./features/async_schedule.md) |
| `enable_prefix_cache` | `bool` | true | false | 是否开启prefix cache（DeepSeek暂不支持） |  |
| `communication_backend` | `string` | "hccl" | "lccl" | 通信操作采用的后端 |  |
| `block_size` | `int32` | 128 |  | KV Cache存储的block size大小 |  |
| `task` | `string` | "generate" | "embed", "mm_embed" | 服务类型，生成式、embedding或多模态embedding |  |
| `max_cache_size` | `int64` | 0 |  | 可使用的KV Cache大小，单位byte |  |
| `kv_cache_dtype` | `string` | "auto" | "int8" | KV Cache数据类型。"auto"表示与模型dtype对齐（不量化），"int8"启用INT8量化以节省约50%显存。仅MLU后端支持 |  |

## MOE模型相关参数
| 参数名称 | 类型 | 默认值 | 其他值 | 参数含义 | 其他 |
|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|
| `dp_size` | `int32` | 1 | 2的指数 | Attention部分的dp规模大小 |  |
| `ep_size` | `int32` | 1 | 2的指数 | MoE部分的ep规模大小 |  |
| `enable_mla` | `bool` | false | true | 是否开启MLA |  |
| `expert_parallel_degree` | `int32` | 0 | 1,2 | ep并行相关参数，gflag默认值为0；当`ep_size > 1`且未显式配置时，部分NPU MoE实现会按EP Level 1处理；当`ep_size=devices`总数时可设置为2（使用all2all通信） |  |


## PD分离相关参数
| 参数名称 | 类型 | 默认值 | 其他值 | 参数含义 | 其他 |
|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|
| `enable_disagg_pd` | `bool` | false | true | 是否启动PD分离 | [详情](./features/disagg_pd.md) |
| `disagg_pd_port` | `int32` | 7777 | 任意可用的端口 | 启用PD分离后配置，对应每张卡上启动的pd分离rpc server的监听端口号 |  |
| `instance_role` | `string` | DEFAULT | PREFILL DECODE MIX | 默认情况下为DEFAULT，开启PD分离后需要配置为PREFILL、DECODE或者MIX |  |
| `kv_cache_transfer_mode` | `string` | "PUSH" | "PULL" | PD分离传输KV Cache的模式。PUSH模式：Prefill逐层向Decode传输；PULL模式：Decode一次性拉取Prefill的KV Cache |  |
| `transfer_listen_port` | `int32` | 26000 | 任意可用的端口 | 启用PD分离后配置，对应每张卡上KV Cache Transfer的监听端口 |  |


## MTP相关参数
| 参数名称 | 类型 | 默认值 | 其他值 | 参数含义 | 其他 |
|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|
| `draft_model` | `string` | "" |  | MTP模型所在的路径 | [详情](./features/mtp.md) |
| `draft_devices` | `string` | "npu:0" | 与`devices`格式保持一致，如`npu:0`或`npu:0,npu:1` | 与devices设置保持一致 |  |
| `num_speculative_tokens` | `int32` | 0 | 任意整数，建议1或者2 | MTP模型每次step输出token的个数 |  |


## 图执行相关参数
| 参数名称 | 类型 | 默认值 | 其他值 | 参数含义 | 其他 |
|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|
| `enable_graph` | `bool` | false | true | 是否启用图执行模式优化decode阶段性能。仅用于decode阶段，对prefill阶段不生效。支持ACL Graph（NPU）、MLU Graph。 | [详情](./features/graph_mode.md) |
| `enable_graph_mode_decode_no_padding` | `bool` | false | true | decode阶段按实际`num_tokens`建图，而不是按padding后的shape建图 |  |
| `enable_prefill_piecewise_graph` | `bool` | false | true | 是否启用prefill阶段的分段Graph。attention以eager执行，其他算子进入图捕获。 |  |
| `max_tokens_for_graph_mode` | `int32` | 2048 | 任意大于等于0的整数 | 图执行模式最大token数。为0表示不限制。 |  |


## 配套xLLM-service使用的参数
| 参数名称 | 类型 | 默认值 | 其他值 | 参数含义 | 其他 |
|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|
| `etcd_addr` | `string` | "" | ip:port | etcd的rpc server监听地址 |  |
| `enable_service_routing` | `bool` | false | true | 请求是否来自xllm service，当使用xllm service管理xllm实例时使用 |  |

## 其他参数
| 参数名称 | 类型 | 默认值 | 其他值 | 参数含义 | 其他 |
|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|
| `max_concurrent_requests` | `int32` | 200 | 任意大于等于0的整数 | 限流用，限制实例中正在处理的总请求数；设置为0表示不限流 |  |
| `model_id` | `string` | "" |  | 模型名称，非路径 |  |
| `num_request_handling_threads` | `int32` | 4 | 任意大于0的整数 | 处理输入请求的线程池大小 |  |
| `num_response_handling_threads` | `int32` | 4 | 任意大于0的整数 | 处理输出的线程池大小 |  |
| `prefill_scheduling_memory_usage_threshold` | `double` | 0.95 | 0-1之间的值 | 当kv cache使用量达到该阈值时，暂停prefill请求的调度 |  |
| `rank_tablefile` | `string` | "" |  | 创建通信域的配置文件,多机场景需要 |  |

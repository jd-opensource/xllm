---
hide:
  - navigation
---


# 服务启动参数

xLLM使用gflags来管理服务启动参数，具体的参数含义如下：

## 常用参数
| 参数名称 | 类型 | 默认值 | 其他值 | 参数含义 | 其他 |
|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|
| `master_node_addr` | `string` | "" | ip:port | master节点rpc server的监听地址 | [详情](./features/basics.md) |
| `host` | `string` | "" | 当前device所在的机器ip | 当前device用于通信的host ip，每个device上会启动一个rpc server，用于多卡之间通信 |  |
| `port` | `int32` | 8080 | 任意可用的端口 | 与host参数配套使用，组合后用于device间的rpc通信 |  |
| `model` | `string` | "" |  | 模型所在的路径 |  |
| `devices` | `string` | "npu:0" |  | 指定当前进程使用的NPU设备 |  |
| `nnodes` | `int32` | 1 |  | 当前服务所使用的device总数 |  |
| `node_rank` | `int32` | 0 | 0 ~ device总数减1 | 每个device的rank id |  |
| `max_memory_utilization` | `double` | 0.9 | 0-1之间 | 模型权重和KV Cache一起可用的最大device memory占比 |  |
| `max_tokens_per_batch` | `int32` | int32的最大值 |  | 每个step可计算的最大token数量 |  |
| `max_seqs_per_batch` | `int32` | 256 |  | 每个step可计算的最大sequence数量 |  |
| `enable_chunked_prefill` | `bool` | true | false | 是否开启chunked prefill |  |
| `enable_schedule_overlap` | `bool` | false | true | 是否开启异步调度 | [详情](./features/async_schedule.md) |
| `enable_prefix_cache` | `bool` | true | false | 是否开启prefix cache（DeepSeek暂不支持） |  |
| `communication_backend` | `string` | "hccl" | "lccl" | 通信操作采用的后端 |  |
| `block_size` | `int32` | 128 |  | KV Cache存储的block size大小 |  |
| `task` | `string` | "generate" | "embed" | 服务类型，生成式或embedding |  |
| `max_cache_size` | `int64` | 0 |  | 可使用的KV Cache大小，单位byte |  |

## MOE模型相关参数
| 参数名称 | 类型 | 默认值 | 其他值 | 参数含义 | 其他 |
|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|
| `dp_size` | `int32` | 1 | 2的指数 | Attention部分的dp规模大小 |  |
| `ep_size` | `int32` | 1 | 2的指数 | MoE部分的ep规模大小 |  |
| `enable_mla` | `bool` | false | true | 是否开启MLA |  |
| `expert_parallel_degree` | `int32` | 0 | 1,2 | ep并行相关参数，不采用ep时默认值为0，启用ep时默认值为1，当ep_size=devices总数时可以设置为2（使用all2all通信） |  |


## PD分离相关参数
| 参数名称 | 类型 | 默认值 | 其他值 | 参数含义 | 其他 |
|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|
| `enable_disagg_pd` | `bool` | false | true | 是否启动PD分离 | [详情](./features/disagg_pd.md) |
| `disagg_pd_port` | `int32` | 7777 | 任意可用的端口 | 启用PD分离后配置，对应每张卡上启动的pd分离rpc server的监听端口号 |  |
| `instance_role` | `string` | DEFAULT | PREFILL DECODE | 默认情况下为DEFAULT，开启PD分离后需要配置为PREFILL或者DECODE |  |
| `kv_cache_transfer_mode` | `string` | "PUSH" | "PULL" | PD分离传输KV Cache的模式。PUSH模式：Prefill逐层向Decode传输；PULL模式：Decode一次性拉取Prefill的KV Cache |  |
| `device_ip` | `string` | "" | 当前device的ip | PD分离需要获取机器的Device IP以创建相关通信资源，可以在当前AI Server执行指令cat /etc/hccn.conf | grep address获取Device IP |  |
| `transfer_listen_port` | `int32` | 26000 | 任意可用的端口 | 启用PD分离后配置，对应每张卡上KV Cache Transfer的监听端口 |  |


## MTP相关参数
| 参数名称 | 类型 | 默认值 | 其他值 | 参数含义 | 其他 |
|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|
| `draft_model` | `string` | "" | ip:port | MTP模型所在的路径 | [详情](./features/mtp.md) |
| `draft_devices` | `string` | "" | ip:port | 与devices设置保持一致 |  |
| `num_speculative_tokens` | `int32` | 0 | 任意整数，建议1或者2 | MTP模型每次step输出token的个数 |  |


## 配套xLLM-service使用的参数
| 参数名称 | 类型 | 默认值 | 其他值 | 参数含义 | 其他 |
|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|
| `xservice_addr` | `string` | "" | ip:port | xllm service的rpc server监听地址 |  |
| `etcd_addr` | `string` | "" | ip:port | etcd的rpc server监听地址 |  |
| `rank_tablefile` | `string` | "" |  | 创建通信域的配置文件,多机场景需要 |  |

## 其他参数
| 参数名称 | 类型 | 默认值 | 其他值 | 参数含义 | 其他 |
|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|
| `max_concurrent_requests` | `int32` | 0 | 任意大于0的整数 | 限流用，限制实例中正在处理的总请求数 |  |
| `model_id` | `string` | "" | ip:port | 模型名称，非路径 |  |
| `num_request_handling_threads` | `int32` | 4 | 任意大于0的整数 | 处理输入请求的线程池大小 |  |
| `num_response_handling_threads` | `int32` | 4 | 任意大于0的整数 | 处理输出的线程池大小 |  |
| `prefill_scheduling_memory_usage_threshold` | `double` | 0.95 | 0-1之间的值 | 当kv cache使用量达到该阈值时，暂停prefill请求的调度 |  |
| `num_response_handling_threads` | `int32` | 4 | 任意大于0的整数 | 处理输出的线程池大小 |  |
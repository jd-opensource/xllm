---
hide:
  - navigation
---


# Service Startup Parameters

xLLM uses gflags to manage service startup parameters. The specific parameter meanings are as follows:

## Common Parameters
| Parameter Name | Data Type | Default Value | Other Values | Description | Notes |
|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|
| `master_node_addr` | `string` | "" | ip:port | The listening address of the master node's rpc server | [Details](./features/basics.md) |
| `host` | `string` | "" | The machine IP where the current device is located | The host IP used by the current device for communication. An rpc server is started on each device for multi-device communication. |  |
| `port` | `int32` | 8080 | Any available port | Used in conjunction with the `host` parameter. The combination is used for rpc communication between devices. |  |
| `model` | `string` | "" |  | Path to the model. |  |
| `devices` | `string` | "npu:0" |  | Specifies the NPU devices used by the current process. |  |
| `nnodes` | `int32` | 1 |  | The total number of devices used by the current service. |  |
| `node_rank` | `int32` | 0 | 0 ~ (total devices - 1) | The rank id of each device. |  |
| `max_memory_utilization` | `double` | 0.9 | Between 0-1 | The maximum proportion of device memory available for model weights and KV Cache combined. |  |
| `max_tokens_per_batch` | `int32` | Maximum value of int32 |  | The maximum number of tokens that can be computed per step. |  |
| `max_seqs_per_batch` | `int32` | 256 |  | The maximum number of sequences that can be computed per step. |  |
| `enable_chunked_prefill` | `bool` | true | false | Whether to enable chunked prefill. |  |
| `enable_schedule_overlap` | `bool` | false | true | Whether to enable asynchronous scheduling. | [Details](./features/async_schedule.md) |
| `enable_prefix_cache` | `bool` | true | false | Whether to enable prefix cache (not supported by DeepSeek currently). |  |
| `communication_backend` | `string` | "hccl" | "lccl" | The backend used for communication operations. |  |
| `block_size` | `int32` | 128 |  | The block size for KV Cache storage. |  |
| `task` | `string` | "generate" | "embed" | Service type: generation or embedding. |  |
| `max_cache_size` | `int64` | 0 |  | The usable KV Cache size in bytes. |  |

## MoE Model Related Parameters
| Parameter Name | Type | Default Value | Other Values | Description | Notes |
|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|
| `dp_size` | `int32` | 1 | Power of 2 | The dp scale size for the Attention part. |  |
| `ep_size` | `int32` | 1 | Power of 2 | The ep scale size for the MoE part. |  |
| `enable_mla` | `bool` | false | true | Whether to enable MLA. |  |
| `expert_parallel_degree` | `int32` | 0 | 1,2 | Parameter related to ep parallelism. Defaults to 0 when ep is not used, and to 1 when ep is enabled. Can be set to 2 when `ep_size` equals the total number of devices (uses all2all communication). |  |


## P-D Separation Related Parameters
| Parameter Name | Type | Default Value | Other Values | Description | Notes |
|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|
| `enable_disagg_pd` | `bool` | false | true | Whether to enable P-D separation. | [Details](./features/disagg_pd.md) |
| `disagg_pd_port` | `int32` | 7777 | Any available port | Configuration when P-D separation is enabled. Corresponds to the listening port number of the pd separation rpc server started on each card. |  |
| `instance_role` | `string` | DEFAULT | PREFILL, DECODE | Defaults to DEFAULT. Must be configured as PREFILL or DECODE when P-D separation is enabled. |  |
| `kv_cache_transfer_mode` | `string` | "PUSH" | "PULL" | The mode for transferring KV Cache in P-D separation. PUSH mode: Prefill transmits layer by layer to Decode; PULL mode: Decode pulls the KV Cache from Prefill in one go. |  |
| `device_ip` | `string` | "" | The IP of the current device | P-D separation requires obtaining the device's Device IP to create related communication resources. You can run the command `cat /etc/hccn.conf | grep address` on the current AI Server to get the Device IP. |  |
| `transfer_listen_port` | `int32` | 26000 | Any available port | Configuration when P-D separation is enabled. Corresponds to the listening port for KV Cache Transfer on each card. |  |


## MTP Related Parameters
| Parameter Name | Type | Default Value | Other Values | Description | Notes |
|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|
| `draft_model` | `string` | "" |  | Path to the MTP model. | [Details](./features/mtp.md) |
| `draft_devices` | `string` | "" |  | Should be set consistently with the `devices` parameter. |  |
| `num_speculative_tokens` | `int32` | 0 | Any integer, suggestion 1 or 2 | The number of tokens output by the MTP model per step. |  |


## Parameters for Use with xLLM-service
| Parameter Name | Type | Default Value | Other Values | Description | Notes |
|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|
| `xservice_addr` | `string` | "" | ip:port | The listening address of the xllm service's rpc server. |  |
| `etcd_addr` | `string` | "" | ip:port | The listening address of the etcd's rpc server. |  |
| `rank_tablefile` | `string` | "" |  | Configuration file for creating the communication domain. Required for multi-node scenarios. |  |

## Other Parameters
| Parameter Name | Type | Default Value | Other Values | Description | Notes |
|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|
| `max_concurrent_requests` | `int32` | 0 | Any integer greater than 0 | For rate limiting, restricts the total number of requests being processed in the instance. |  |
| `model_id` | `string` | "" |  | Model name, not a path. |  |
| `num_request_handling_threads` | `int32` | 4 | Any integer greater than 0 | The thread pool size for handling input requests. |  |
| `num_response_handling_threads` | `int32` | 4 | Any integer greater than 0 | The thread pool size for handling outputs. |  |
| `prefill_scheduling_memory_usage_threshold` | `double` | 0.95 | Value between 0-1 | When kv cache usage reaches this threshold, scheduling of prefill requests is paused. |  |
| `num_response_handling_threads` | `int32` | 4 | Any integer greater than 0 | The thread pool size for handling outputs. |  |
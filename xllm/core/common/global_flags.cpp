#include "global_flags.h"

#include <limits>

#include "brpc/reloadable_flags.h"

// --- xllm service config ---

DEFINE_string(host, "", "Host name for brpc server.");

DEFINE_int32(port, 8080, "Port for brpc server.");

DEFINE_int32(idle_timeout_s,
             -1,
             "Connection will be closed if there is no read/write operations "
             "during the last `idle_timeout_s`");

DEFINE_int32(num_threads, 32, "Number of threads to process requests");

DEFINE_int32(max_concurrency,
             0,
             "Limit number of requests processed in parallel");

DEFINE_int32(
    max_concurrent_requests,
    0,
    "Maximum number of concurrent requests the xllm service can handle.");

BRPC_VALIDATE_GFLAG(max_concurrent_requests, brpc::NonNegativeInteger);

// --- model serving config ---

DEFINE_string(model_id, "", "hf model name.");

DEFINE_string(model, "", "Name or path of the huggingface model to use.");

DEFINE_string(backend,
              "llm",
              "Choose the backend model type. 'llm' for text-only, "
              "'vlm' for multimodal (text and images).");

DEFINE_string(task,
              "generate",
              "The task to use the model for. generate/embed.");

DEFINE_string(devices,
              "npu:0",
              "Devices to run the model on, e.g. npu:0, npu:0,npu:1.");

DEFINE_string(draft_model, "", "draft hf model path to the model file.");

DEFINE_string(draft_devices,
              "npu:0",
              "Devices to run the draft model on, e.g. npu:0, npu:0,npu:1.");

DEFINE_int32(block_size,
             128,
             "Number of slots per kv cache block. Default is 128.");

DEFINE_int64(max_cache_size,
             0,
             "Max gpu memory size for kv cache. Default is 0, which means "
             "cache size is caculated by available memory.");

DEFINE_double(max_memory_utilization,
              0.9,
              "The fraction of GPU memory to be used for model inference, "
              "including model weights and kv cache.");

DEFINE_int32(max_tokens_per_batch,
             std::numeric_limits<int32_t>::max(),
             "Max number of tokens per batch.");

DEFINE_int32(max_seqs_per_batch, 256, "Max number of sequences per batch.");

DEFINE_int32(max_tokens_per_chunk_for_prefill,
             2048,
             "Max number of token per chunk in prefill stage.");

DEFINE_int32(num_speculative_tokens, 0, "Number of speculative tokens.");

DEFINE_int32(num_handling_threads, 4, "Number of handling threads.");

DEFINE_int32(num_response_handling_threads,
             4,
             "Number of response handling threads.");

DEFINE_bool(enable_chunked_prefill, true, "Whether to enable chunked prefill.");

DEFINE_int32(dp_size, 1, "Data parallel size for MLA attention.");

DEFINE_int32(ep_size, 1, "Expert parallel size for MoE model.");

DEFINE_bool(enable_schedule_overlap,
            false,
            "Whether to enable schedule overlap.");

DEFINE_double(prefill_scheduling_memory_usage_threshold,
              0.95,
              "The memory usage threshold during prefill scheduling.");

DEFINE_string(communication_backend, "hccl", "npu communication backend.");

DEFINE_bool(enable_eplb, false, "Whether to use ep load balance.");

DEFINE_int64(eplb_update_rate, 1000, "eplb update rate.");

DEFINE_double(eplb_update_threshold, 0.8, "eplb update threshold.");

DEFINE_string(rank_tablefile, "", "atb hccl rank table file.");

DEFINE_int32(expert_parallel_degree, 0, "ep degree");

DEFINE_bool(enable_mla,
            false,
            "whether to enable multi-head latent attention.");

// --- prefix cache config ---

DEFINE_bool(enable_prefix_cache,
            true,
            "enable the prefix cache for the block manager");

// --- serving on multi-nodes config ---

DEFINE_string(master_node_addr,
              "127.0.0.1:19888",
              "The master address for multi-node distributed serving(e.g. "
              "10.18.1.1:9999).");

DEFINE_int32(nnodes, 1, "The number of multi-nodes.");

DEFINE_int32(node_rank, 0, "The node rank.");

// --- disaggregated prefill and decode config ---

DEFINE_string(xservice_addr, "", "xservice server address.");

DEFINE_bool(enable_disagg_pd,
            false,
            "Enable disaggregated prefill and decode execution.");

DEFINE_int32(disagg_pd_port, 7777, "Port for brpc disagg pd server.");

DEFINE_string(instance_role,
              "DEFAULT",
              "The role of instance(e.g. DEFAULT, PREFILL, DECODE).");

DEFINE_string(kv_cache_transfer_type,
              "LlmDataDist",
              "The type of kv cache transfer(e.g. LlmDataDist, HCCL).");

DEFINE_string(kv_cache_transfer_mode,
              "PUSH",
              "The mode of kv cache transfer(e.g. PUSH, PULL).");

DEFINE_string(device_ip, "", "The device ip.");

DEFINE_int32(transfer_listen_port, 26000, "The KVCacheTranfer listen port.");

// --- worker server config ---

DEFINE_int32(max_connect_count,
             40,
             "The max count for worker try to connect to server.");

DEFINE_int32(sleep_time_second,
             3,
             "The sleep time for worker try to connect to server next time.");

// --- kernel config ---

DEFINE_bool(disable_custom_kernels, false, "disable all custom kernels");

DEFINE_bool(enable_atb_comm_multiprocess,
            false,
            "whether to use multiprocess mode.");

// --- function call config ---

DEFINE_string(tool_call_parser,
              "",
              "Specify the parser for handling tool-call interactions. "
              "Options include: 'qwen25'");

DEFINE_bool(enable_atb_spec_kernel,
            false,
            "whether to use ATB speculative kernel.");

// --- service routing config ---

DEFINE_string(etcd_addr, "", "etcd adderss for save instance meta info");

DEFINE_bool(enable_service_routing, false, "whether to use etcd.");

DEFINE_int32(heart_beat_interval, 3, "heart beat interval");

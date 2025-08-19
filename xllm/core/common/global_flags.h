#pragma once

#include <gflags/gflags.h>

constexpr int64_t GB = int64_t(1024) * 1024 * 1024;

DECLARE_string(host);

DECLARE_int32(port);

DECLARE_int32(disagg_pd_port);

DECLARE_int32(idle_timeout_s);

DECLARE_int32(num_threads);

DECLARE_int32(max_concurrency);

DECLARE_string(model_id);

DECLARE_string(model);

DECLARE_string(backend);

DECLARE_string(task);

DECLARE_string(devices);

DECLARE_string(draft_model);

DECLARE_string(draft_devices);

DECLARE_int32(block_size);

DECLARE_int64(max_cache_size);

DECLARE_double(max_memory_utilization);

DECLARE_bool(enable_prefix_cache);

DECLARE_int32(max_tokens_per_batch);

DECLARE_int32(max_seqs_per_batch);

DECLARE_int32(max_tokens_per_chunk_for_prefill);

DECLARE_int32(num_speculative_tokens);

DECLARE_int32(num_handling_threads);

DECLARE_int32(num_response_handling_threads);

DECLARE_string(communication_backend);

DECLARE_bool(enable_eplb);

DECLARE_int64(eplb_update_rate);

DECLARE_double(eplb_update_threshold);

DECLARE_string(rank_tablefile);

DECLARE_bool(enable_mla);

DECLARE_bool(enable_chunked_prefill);

DECLARE_string(master_node_addr);

DECLARE_bool(enable_disagg_pd);

DECLARE_int32(nnodes);

DECLARE_int32(node_rank);

DECLARE_int32(dp_size);

DECLARE_int32(ep_size);

DECLARE_string(xservice_addr);

DECLARE_string(instance_role);

DECLARE_string(kv_cache_transfer_type);

DECLARE_string(kv_cache_transfer_mode);

DECLARE_string(device_ip);

DECLARE_int32(transfer_listen_port);

DECLARE_int32(max_concurrent_requests);

DECLARE_bool(enable_schedule_overlap);

DECLARE_double(prefill_scheduling_memory_usage_threshold);

DECLARE_int32(expert_parallel_degree);

DECLARE_int32(max_connect_count);

DECLARE_int32(sleep_time_second);

DECLARE_bool(disable_custom_kernels);

DECLARE_bool(enable_atb_comm_multiprocess);

DECLARE_string(tool_call_parser);

DECLARE_bool(enable_atb_spec_kernel);

DECLARE_string(etcd_addr);

DECLARE_bool(enable_service_routing);

DECLARE_int32(heart_beat_interval);

DECLARE_int32(chunked_match_frequency);

DECLARE_bool(use_zero_evict);

/* Copyright 2025 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <gflags/gflags.h>

constexpr int64_t GB = int64_t(1024) * 1024 * 1024;

DECLARE_string(host);

DECLARE_int32(port);

DECLARE_int32(disagg_pd_port);

DECLARE_int32(rpc_idle_timeout_s);

DECLARE_int32(num_threads);

DECLARE_string(model_id);

DECLARE_string(model);

DECLARE_string(backend);

DECLARE_string(task);

DECLARE_string(devices);

DECLARE_string(draft_model);

DECLARE_string(draft_devices);

DECLARE_int32(limit_image_per_prompt);

DECLARE_int32(block_size);

DECLARE_int64(max_cache_size);

DECLARE_double(max_memory_utilization);

DECLARE_bool(enable_prefix_cache);

DECLARE_bool(enable_cache_upload);

DECLARE_uint32(murmur_hash3_seed);

DECLARE_int32(max_tokens_per_batch);

DECLARE_int32(max_seqs_per_batch);

DECLARE_int32(max_tokens_per_chunk_for_prefill);

DECLARE_int32(num_speculative_tokens);

DECLARE_int32(num_request_handling_threads);

DECLARE_int32(num_response_handling_threads);

DECLARE_string(communication_backend);

DECLARE_bool(enable_eplb);

DECLARE_int32(redundant_experts_num);

DECLARE_int64(eplb_update_interval);

DECLARE_double(eplb_update_threshold);

DECLARE_string(rank_tablefile);

DECLARE_bool(enable_mla);

DECLARE_bool(enable_acl_graph);

DECLARE_int32(max_seq_len_for_graph_mode);

DECLARE_bool(enable_chunked_prefill);

DECLARE_string(master_node_addr);

DECLARE_bool(enable_disagg_pd);

DECLARE_bool(enable_pd_ooc);

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

DECLARE_int32(max_reconnect_count);

DECLARE_bool(enable_atb_comm_multiprocess);

DECLARE_string(tool_call_parser);

DECLARE_bool(enable_atb_spec_kernel);

DECLARE_bool(enable_block_copy_kernel);

DECLARE_string(etcd_addr);

DECLARE_bool(enable_service_routing);

DECLARE_double(heart_beat_interval);

DECLARE_int32(etcd_ttl);

DECLARE_int32(rpc_channel_timeout_ms);

DECLARE_int32(chunked_match_frequency);

DECLARE_bool(use_zero_evict);

DECLARE_int32(max_decode_token_per_sequence);

DECLARE_string(priority_strategy);

DECLARE_bool(enable_online_preempt_offline);

DECLARE_double(host_blocks_factor);

DECLARE_bool(enable_kvcache_store);

DECLARE_string(store_protocol);

DECLARE_string(store_master_server_entry);

DECLARE_string(store_metadata_connstring);

DECLARE_bool(enable_multi_stream_parallel);

DECLARE_int32(micro_batch_num);

DECLARE_bool(enable_profile_step_time);

DECLARE_bool(enable_profile_token_budget);

DECLARE_bool(enable_latency_aware_schedule);

DECLARE_int32(profile_max_prompt_length);

DECLARE_bool(enable_profile_kv_blocks);

DECLARE_bool(disable_ttft_profiling);

DECLARE_bool(enable_forward_interruption);

DECLARE_int32(max_global_ttft_ms);

DECLARE_int32(max_global_tpot_ms);

DECLARE_int32(max_requests_per_batch);

DECLARE_bool(enable_continuous_kvcache);

DECLARE_int64(phy_page_granularity_size);

DECLARE_int64(cache_size_per_token);

DECLARE_int64(buffer_size_per_seq);

DECLARE_bool(enable_beam_search_kernel);

DECLARE_bool(enable_qwen3_reranker);

DECLARE_string(reasoning_parser);

DECLARE_bool(enable_shm);

DECLARE_bool(enable_prefetch_weight);

DECLARE_bool(enable_rec_prefill_only);

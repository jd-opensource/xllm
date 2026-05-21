/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include "core/framework/config/parallel_config.h"

#include "core/common/global_flags.h"
#include "core/framework/config/config_json_utils.h"

DEFINE_int32(dp_size, 1, "Data parallel size for MLA attention.");

DEFINE_int32(ep_size, 1, "Expert parallel size for MoE model.");

DEFINE_int32(cp_size, 1, "Context parallel size for DSA attention.");

DEFINE_int32(kv_split_size,
             0,
             "KV-cache split width. 0 falls back to cp_size (legacy); 1 means "
             "no KV split (each CP rank stores full KV, skips prefix "
             "AllGather); other K (K divides cp_size) means KV is sharded "
             "across K ranks while token-CP still uses cp_size.");

DEFINE_int32(prefill_kv_split_size,
             0,
             "KV-cache split width of the remote prefill instance. Set on "
             "decode nodes in PD mode so D can match P logical block layout "
             "(link_cluster stride and remote_blocks_ids expansion). 0 falls "
             "back to local cp_size.");

DEFINE_int64(tp_size, 1, "Tensor parallelism size, only used for DiT model.");

DEFINE_int64(sp_size, 1, "Sequence parallelism size, only used for DiT model.");

DEFINE_int64(cfg_size,
             1,
             "Classifier-free guidiance parallelism size, only used for DiT "
             "model.");

DEFINE_string(
    communication_backend,
    "hccl",
    "NPU communication backend.(e.g. lccl, hccl). When enable dp, use hccl.");

DEFINE_bool(enable_prefill_sp,
            false,
            "Whether to enable prefill-only sequence parallel.");

DEFINE_bool(
    enable_multi_stream_parallel,
    false,
    "Whether to enable computation communication parallel by two streams "
    "and two micro batches in prefill stage.");

DEFINE_int32(micro_batch_num,
             1,
             "Default use two micro batches for multi-stream parallel.");

DEFINE_bool(
    enable_dp_balance,
    false,
    "Whether to enable dp load balance, if true, sequences within a single "
    "dp batch will be shuffled.");

namespace xllm {

void ParallelConfig::from_flags() {
  dp_size(FLAGS_dp_size)
      .ep_size(FLAGS_ep_size)
      .cp_size(FLAGS_cp_size)
      .kv_split_size(FLAGS_kv_split_size)
      .prefill_kv_split_size(FLAGS_prefill_kv_split_size)
      .tp_size(FLAGS_tp_size)
      .sp_size(FLAGS_sp_size)
      .cfg_size(FLAGS_cfg_size)
      .communication_backend(FLAGS_communication_backend)
      .enable_prefill_sp(FLAGS_enable_prefill_sp)
      .enable_multi_stream_parallel(FLAGS_enable_multi_stream_parallel)
      .micro_batch_num(FLAGS_micro_batch_num)
      .enable_dp_balance(FLAGS_enable_dp_balance);
}

void ParallelConfig::from_json(const JsonReader& json) {
  dp_size(json.value_or<int32_t>("dp_size", dp_size()))
      .ep_size(json.value_or<int32_t>("ep_size", ep_size()))
      .cp_size(json.value_or<int32_t>("cp_size", cp_size()))
      .tp_size(json.value_or<int64_t>("tp_size", tp_size()))
      .sp_size(json.value_or<int64_t>("sp_size", sp_size()))
      .cfg_size(json.value_or<int64_t>("cfg_size", cfg_size()))
      .communication_backend(json.value_or<std::string>(
          "communication_backend", communication_backend()))
      .enable_prefill_sp(
          json.value_or<bool>("enable_prefill_sp", enable_prefill_sp()))
      .enable_multi_stream_parallel(json.value_or<bool>(
          "enable_multi_stream_parallel", enable_multi_stream_parallel()))
      .micro_batch_num(
          json.value_or<int32_t>("micro_batch_num", micro_batch_num()))
      .enable_dp_balance(
          json.value_or<bool>("enable_dp_balance", enable_dp_balance()));
}

ParallelConfig& ParallelConfig::get_instance() {
  static ParallelConfig config;
  return config;
}

void ParallelConfig::initialize() {
  from_flags();
  if (const auto& json_config = config::get_parsed_json_config()) {
    from_json(*json_config);
  }
}

}  // namespace xllm

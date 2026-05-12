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

DEFINE_int32(dp_size, 1, "Data parallel size for MLA attention.");

DEFINE_int32(ep_size, 1, "Expert parallel size for MoE model.");

DEFINE_int32(cp_size, 1, "Context parallel size for DSA attention.");

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

ParallelConfig ParallelConfig::from_flags() {
  ParallelConfig config;
  config.dp_size(FLAGS_dp_size)
      .ep_size(FLAGS_ep_size)
      .cp_size(FLAGS_cp_size)
      .tp_size(FLAGS_tp_size)
      .sp_size(FLAGS_sp_size)
      .cfg_size(FLAGS_cfg_size)
      .communication_backend(FLAGS_communication_backend)
      .enable_prefill_sp(FLAGS_enable_prefill_sp)
      .enable_multi_stream_parallel(FLAGS_enable_multi_stream_parallel)
      .micro_batch_num(FLAGS_micro_batch_num)
      .enable_dp_balance(FLAGS_enable_dp_balance);
  return config;
}

ParallelConfig& ParallelConfig::get_instance() {
  static ParallelConfig config = ParallelConfig::from_flags();
  return config;
}

void ParallelConfig::reload_from_flags() {
  ParallelConfig::get_instance() = ParallelConfig::from_flags();
}

}  // namespace xllm

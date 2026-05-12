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

#include "core/framework/config/execution_config.h"

#include "core/common/global_flags.h"

DEFINE_bool(
    enable_graph,
    false,
    "Whether to enable graph execution for decode phase. When enabled, "
    "the engine uses graph mode (CUDA Graph for GPU, ACL Graph for NPU, "
    "or MLU Graph) to optimize decode performance by reducing kernel "
    "launch overhead and device idle time.");

DEFINE_bool(enable_graph_mode_decode_no_padding,
            false,
            "Whether to enable graph execution for decode phase without "
            "padding. If true, graph will be caputured with every actual num "
            "tokens, as stride is 1.");

DEFINE_bool(enable_prefill_piecewise_graph,
            false,
            "Whether to enable piecewise CUDA graph for prefill phase. "
            "When enabled, attention operations use eager mode while other "
            "operations are captured in CUDA graphs.");

DEFINE_bool(enable_graph_vmm_pool,
            true,
            "Whether to enable VMM-backed CUDA graph memory pool for "
            "multi-shape graph memory reuse.");

DEFINE_int32(max_tokens_for_graph_mode,
             2048,
             "Maximum number of tokens for graph execution. "
             "If 0, no limit is applied.");

DEFINE_bool(enable_shm,
            false,
            "Whether to enable shared memory for executing model.");

DEFINE_bool(use_contiguous_input_buffer,
            true,
            "Whether to use contiguous device input buffer for executing "
            "model. Currently only effective when enable_shm is true.");

DEFINE_uint64(input_shm_size,
              1024,
              "Input shared memory size, default is 1GB.");

DEFINE_uint64(output_shm_size,
              128,
              "Output shared memory size, default is 128MB.");

DEFINE_int32(random_seed, -1, "Random seed for random number generator.");

DEFINE_bool(enable_customize_mla_kernel, false, "enable customize mla kernel");

#if defined(USE_NPU)
DEFINE_string(npu_kernel_backend,
              "AUTO",
              "NPU kernel backend. Supported options: AUTO, ATB, TORCH.");

DEFINE_bool(enable_intralayer_addnorm,
            false,
            "enable fused intralayer addnorm ops.");
#endif

namespace xllm {

ExecutionConfig ExecutionConfig::from_flags() {
  ExecutionConfig config;
  config.enable_graph(FLAGS_enable_graph)
      .enable_graph_mode_decode_no_padding(
          FLAGS_enable_graph_mode_decode_no_padding)
      .enable_prefill_piecewise_graph(FLAGS_enable_prefill_piecewise_graph)
      .enable_graph_vmm_pool(FLAGS_enable_graph_vmm_pool)
      .max_tokens_for_graph_mode(FLAGS_max_tokens_for_graph_mode)
      .enable_shm(FLAGS_enable_shm)
      .use_contiguous_input_buffer(FLAGS_use_contiguous_input_buffer)
      .input_shm_size(FLAGS_input_shm_size)
      .output_shm_size(FLAGS_output_shm_size)
      .random_seed(FLAGS_random_seed)
      .enable_customize_mla_kernel(FLAGS_enable_customize_mla_kernel);
#if defined(USE_NPU)
  config.npu_kernel_backend(FLAGS_npu_kernel_backend)
      .enable_intralayer_addnorm(FLAGS_enable_intralayer_addnorm);
#endif
  return config;
}

ExecutionConfig& ExecutionConfig::get_instance() {
  static ExecutionConfig config = ExecutionConfig::from_flags();
  return config;
}

void ExecutionConfig::reload_from_flags() {
  ExecutionConfig::get_instance() = ExecutionConfig::from_flags();
}

}  // namespace xllm

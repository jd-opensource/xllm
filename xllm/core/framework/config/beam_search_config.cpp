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

#include "core/framework/config/beam_search_config.h"

#include "core/common/global_flags.h"

DEFINE_bool(enable_beam_search_kernel,
            false,
            "Whether to enable beam search kernel.");

DEFINE_int32(beam_width, 1, "Beam width for beam search.");

#if defined(USE_NPU) || defined(USE_CUDA)
DEFINE_bool(enable_block_copy_kernel,
            true,
            "Whether to use block copy kernel on supported backends.");
#else
DEFINE_bool(enable_block_copy_kernel,
            false,
            "Whether to use block copy kernel on supported backends.");
#endif

DEFINE_bool(enable_topk_sorted,
            true,
            "Whether to enable sorted output for topk.");

namespace xllm {

BeamSearchConfig BeamSearchConfig::from_flags() {
  BeamSearchConfig config;
  config.enable_beam_search_kernel(FLAGS_enable_beam_search_kernel)
      .beam_width(FLAGS_beam_width)
      .enable_block_copy_kernel(FLAGS_enable_block_copy_kernel)
      .enable_topk_sorted(FLAGS_enable_topk_sorted);
  return config;
}

BeamSearchConfig& BeamSearchConfig::get_instance() {
  static BeamSearchConfig config = BeamSearchConfig::from_flags();
  return config;
}

void BeamSearchConfig::reload_from_flags() {
  BeamSearchConfig::get_instance() = BeamSearchConfig::from_flags();
}

}  // namespace xllm

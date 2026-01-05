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

#include "layers/common/attention_metadata.h"

namespace xllm {
namespace layer {
namespace flashinfer {

void update_plan_info(std::shared_ptr<PlanInfo> plan_info,
                      const std::string& backend,
                      const AttentionMetadata& attn_meta,
                      c10::ScalarType query_dtype,
                      c10::ScalarType key_dtype,
                      c10::ScalarType output_dtype,
                      int32_t head_dim_qk,
                      int32_t head_dim_vo,
                      int32_t num_qo_heads,
                      int32_t num_kv_heads,
                      int32_t block_size,
                      int32_t window_size_left,
                      bool enable_cuda_graph,
                      bool causal,
                      bool use_tensor_core);

}  // namespace flashinfer
}  // namespace layer
}  // namespace xllm

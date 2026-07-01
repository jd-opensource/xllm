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

#pragma once

#include <torch/torch.h>

#include <optional>
#include <string>
#include <utility>

namespace xllm {
namespace kernel {

struct MateGatedDeltaRulePrefillParams;
struct MateGatedDeltaRuleDecodeParams;

namespace cuda {

std::string get_mate_gdn_prefill_uri(int64_t num_q_heads,
                                     int64_t num_v_heads,
                                     torch::ScalarType dtype);

std::string get_mate_gdn_decode_uri(int64_t num_q_heads,
                                    int64_t num_v_heads,
                                    torch::ScalarType dtype);

std::pair<torch::Tensor, torch::Tensor> mate_gated_delta_rule_prefill(
    MateGatedDeltaRulePrefillParams& params);

torch::Tensor mate_gated_delta_rule_decode(
    MateGatedDeltaRuleDecodeParams& params);

// Single-launch in-house CUDA decode kernel; drop-in alternative to the mate
// FFI decode. Reuses the same param struct so the layer can swap between
// implementations behind a flag. See fused_gdn_decode.cu for semantics.
torch::Tensor fused_gated_delta_rule_decode(
    MateGatedDeltaRuleDecodeParams& params);

}  // namespace cuda
}  // namespace kernel
}  // namespace xllm

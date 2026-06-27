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

#include <glog/logging.h>

#include "kernels/mlu/chunk_gated_delta_rule.h"
#include "kernels/mlu/mlu_ops_api.h"

namespace xllm {
namespace kernel {
namespace mlu {

namespace {

void fail_mlu_triton_aot_skipped(const char* op_name) {
  LOG(FATAL) << op_name
             << " requires MLU Triton AOT kernels, but this build was created "
             << "with XLLM_SKIP_MLU_TRITON_AOT enabled.";
}

}  // namespace

torch::Tensor causal_conv1d_fn(const torch::Tensor&,
                               const torch::Tensor&,
                               const torch::Tensor&,
                               const torch::Tensor&,
                               const torch::Tensor&,
                               const torch::Tensor&,
                               int32_t,
                               const std::optional<torch::Tensor>&,
                               const std::optional<torch::Tensor>&,
                               const std::optional<torch::Tensor>&,
                               const std::optional<torch::Tensor>&,
                               const std::optional<torch::Tensor>&,
                               bool) {
  fail_mlu_triton_aot_skipped("causal_conv1d_fn");
  return torch::Tensor();
}

std::pair<torch::Tensor, torch::Tensor> fused_recurrent_gated_delta_rule(
    const torch::Tensor&,
    const torch::Tensor&,
    const torch::Tensor&,
    const torch::Tensor&,
    const std::optional<torch::Tensor>&,
    const std::optional<torch::Tensor>&,
    bool,
    const std::optional<torch::Tensor>&,
    const std::optional<torch::Tensor>&,
    const std::optional<torch::Tensor>&,
    bool) {
  fail_mlu_triton_aot_skipped("fused_recurrent_gated_delta_rule");
  return {};
}

std::pair<torch::Tensor, torch::Tensor>
fused_recurrent_gated_delta_rule_packed_decode(const torch::Tensor&,
                                               const torch::Tensor&,
                                               const torch::Tensor&,
                                               const torch::Tensor&,
                                               const torch::Tensor&,
                                               double,
                                               torch::Tensor&,
                                               const torch::Tensor&,
                                               bool) {
  fail_mlu_triton_aot_skipped("fused_recurrent_gated_delta_rule_packed_decode");
  return {};
}

torch::Tensor causal_conv1d_update_decode(const torch::Tensor&,
                                          torch::Tensor&,
                                          const torch::Tensor&,
                                          const std::optional<torch::Tensor>&,
                                          const torch::Tensor&,
                                          int32_t,
                                          const std::optional<torch::Tensor>&,
                                          int32_t,
                                          const std::optional<torch::Tensor>&,
                                          const std::optional<torch::Tensor>&,
                                          const std::optional<torch::Tensor>&) {
  fail_mlu_triton_aot_skipped("causal_conv1d_update_decode");
  return torch::Tensor();
}

std::pair<torch::Tensor, torch::Tensor> fused_gdn_gating(const torch::Tensor&,
                                                         const torch::Tensor&,
                                                         const torch::Tensor&,
                                                         const torch::Tensor&,
                                                         float,
                                                         float) {
  fail_mlu_triton_aot_skipped("fused_gdn_gating");
  return {};
}

ChunkGatedDeltaRuleImpl::ChunkGatedDeltaRuleImpl(int64_t num_k_heads,
                                                 int64_t num_v_heads)
    : total_core_num_(0),
      num_k_heads_(num_k_heads),
      num_v_heads_(num_v_heads),
      algo_id_(0),
      chunk_algo_id_(0) {}

std::tuple<torch::Tensor, torch::Tensor> ChunkGatedDeltaRuleImpl::forward(
    torch::Tensor&,
    torch::Tensor&,
    torch::Tensor&,
    torch::Tensor&,
    torch::Tensor&,
    torch::Tensor&,
    torch::Tensor&,
    torch::Tensor&,
    bool,
    bool) {
  fail_mlu_triton_aot_skipped("ChunkGatedDeltaRuleImpl::forward");
  return {};
}

}  // namespace mlu
}  // namespace kernel
}  // namespace xllm

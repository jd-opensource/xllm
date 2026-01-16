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

#include "mlu_ops_api.h"

namespace xllm::kernel::mlu {
void fused_mla_q(const torch::Tensor& input,
                 torch::Tensor& output,
                 torch::Tensor& output_scale,
                 const std::optional<torch::Tensor>& output_norm,
                 const torch::Tensor& gamma,
                 const std::optional<torch::Tensor>& smooth_quant_scale,
                 const torch::Tensor& weight_b,
                 const torch::Tensor& weight_b_scale,
                 const torch::Tensor& weight_c,
                 const torch::Tensor& sin,
                 const torch::Tensor& cos,
                 const torch::Tensor& position_id,
                 const std::string& quant_mode,
                 double eps,
                 bool interleaved) {
  tmo::torch_api::fused_mla_q(input,
                              output,
                              output_scale,
                              output_norm,
                              gamma,
                              smooth_quant_scale,
                              weight_b,
                              weight_b_scale,
                              weight_c,
                              sin,
                              cos,
                              position_id,
                              quant_mode,
                              eps,
                              interleaved);
}

void fused_mla_kv(const torch::Tensor& input_kv,
                  const torch::Tensor& sin,
                  const torch::Tensor& cos,
                  const torch::Tensor& position_id,
                  const torch::Tensor& gamma,
                  const torch::Tensor& kv_cache,
                  const std::optional<torch::Tensor>& kv_cache_scale,
                  const std::optional<torch::Tensor>& slot_mapping,
                  const std::optional<torch::Tensor>& cache_bs_id,
                  const std::optional<torch::Tensor>& cache_seq_offset,
                  const std::string& quant_mode,
                  bool is_paged_cache,
                  double eps,
                  bool interleaved) {
  tmo::torch_api::fused_mla_kv(input_kv,
                               sin,
                               cos,
                               position_id,
                               gamma,
                               kv_cache,
                               kv_cache_scale,
                               slot_mapping,
                               cache_bs_id,
                               cache_seq_offset,
                               quant_mode,
                               is_paged_cache,
                               eps,
                               interleaved);
}

}  // namespace xllm::kernel::mlu
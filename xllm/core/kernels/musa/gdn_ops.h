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

// Qwen3.5 hybrid Gated DeltaNet + partial-RoPE + gated-norm op declarations
// on the CUDA (incl. MUSA-as-CUDA) backend.
//
// These are the kernels invoked by ``xllm::kernel::ops_api`` dispatchers
// when the model is one of the Qwen3.5 / Qwen3-Next hybrid architectures.
// Two backing implementations are envisaged:
//
//   1. FlashInfer-style mate bridge SOs loaded via TVM-FFI under
//      ``FLASHINFER_OPS_PATH`` (preferred; see ``mate/csrc/integrations/
//      flashinfer/``).
//   2. Hand-written MUSA kernels shipped through the same SO layout
//      (fallback / baseline).
//
// At M1 these are libtorch reference kernels (correctness-first). M2 can
// replace hot paths with mate TVM-FFI SOs (``gated_delta_rule_decode``, etc.).
#pragma once

#include <torch/torch.h>

#include <optional>
#include <tuple>
#include <utility>

namespace xllm {
namespace kernel {

struct CausalConv1dUpdateParams;
struct ChunkGatedDeltaRuleParams;
struct FusedGdnGatingParams;
struct FusedQkvzbaSplitReshapeParams;
struct FusedRecurrentGatedDeltaRuleParams;
struct GatedLayerNormParams;
struct PartialRotaryEmbeddingParams;

namespace cuda {

torch::Tensor l2_norm(torch::Tensor& x, double eps);

std::pair<torch::Tensor, torch::Tensor> fused_gdn_gating(
    FusedGdnGatingParams& params);

std::pair<torch::Tensor, torch::Tensor> gdn_gating(const torch::Tensor& a,
                                                   const torch::Tensor& b,
                                                   const torch::Tensor& A_log,
                                                   const torch::Tensor& dt_bias,
                                                   double sp_beta,
                                                   double threshold);

std::pair<torch::Tensor, torch::Tensor> fused_recurrent_gated_delta_rule(
    FusedRecurrentGatedDeltaRuleParams& params);

torch::Tensor causal_conv1d_update(CausalConv1dUpdateParams& params);

torch::Tensor gated_layer_norm(GatedLayerNormParams& params);

std::pair<torch::Tensor, torch::Tensor> partial_rotary_embedding(
    PartialRotaryEmbeddingParams& params);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
fused_qkvzba_split_reshape_cat(FusedQkvzbaSplitReshapeParams& params);

std::pair<torch::Tensor, torch::Tensor> chunk_gated_delta_rule(
    ChunkGatedDeltaRuleParams& params);

torch::Tensor recurrent_gated_delta_rule(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    torch::Tensor& state,
    const std::optional<torch::Tensor>& beta,
    const std::optional<double> scale,
    const std::optional<torch::Tensor>& actual_seq_lengths,
    const std::optional<torch::Tensor>& ssm_state_indices,
    const std::optional<torch::Tensor>& num_accepted_tokens,
    const std::optional<torch::Tensor>& g,
    const std::optional<torch::Tensor>& gk);

}  // namespace cuda
}  // namespace kernel
}  // namespace xllm

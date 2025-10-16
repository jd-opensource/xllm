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

#include <torch/torch.h>

#include "dense_mlp.h"
#include "framework/model/model_args.h"
#include "framework/model/model_input_params.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"
#include "framework/state_dict/utils.h"
#include "linear.h"
namespace xllm {
namespace layer {

class FusedMoEImpl : public torch::nn::Module {
 public:
  FusedMoEImpl() = default;
  FusedMoEImpl(int num_experts,
               int top_k,
               int num_expert_group,
               int topk_group,
               double route_scale,
               int hidden_size,
               int intermediate_size,
               int n_shared_experts,
               bool is_gated,
               bool has_score_bias,
               bool has_bias,
               bool skip_bias_add,
               int renormalize,
               const std::string& hidden_act,
               const std::string& scoring_func,
               const std::string& topk_method,
               const QuantArgs& quant_args,
               const ParallelArgs& parallel_args,
               const torch::TensorOptions& options);

  torch::Tensor forward_expert(
      const torch::Tensor& hidden_states,
      const torch::Tensor& router_logits,
      const std::optional<torch::Tensor>& shared_output);
  torch::Tensor forward(const torch::Tensor& hidden_states,
                        const ModelInputParams& input_params);
  void load_state_dict(const StateDict& state_dict);

 private:
  int num_experts_;
  int topk_;
  int num_expert_group_;
  int topk_group_;
  double route_scale_;
  int hidden_size_;
  int intermediate_size_;
  int n_shared_experts_;
  bool is_gated_;
  bool has_score_bias_;
  bool has_bias_;
  bool skip_bias_add_;
  int renormalize_;
  std::string hidden_act_;
  std::string scoring_func_;
  bool is_smoothquant_;

  int num_experts_per_rank_;
  int start_expert_id_;

  ReplicatedLinear gate_{nullptr};
  DenseMLP shared_experts_{nullptr};

  QuantArgs quant_args_;
  ParallelArgs parallel_args_;
  torch::TensorOptions options_;
  ProcessGroup* tp_pg_;

  DEFINE_FUSED_WEIGHT(w13);
  DEFINE_FUSED_WEIGHT(w1);
  DEFINE_FUSED_WEIGHT(w3);
  DEFINE_FUSED_WEIGHT(w2);
  DEFINE_WEIGHT(e_score_correction_bias);
  DEFINE_FUSED_WEIGHT(w13_scale);
  DEFINE_FUSED_WEIGHT(w1_scale);
  DEFINE_FUSED_WEIGHT(w3_scale);
  DEFINE_FUSED_WEIGHT(w2_scale);
  DEFINE_FUSED_WEIGHT(input_smooth);
  DEFINE_FUSED_WEIGHT(act_smooth);

  void pack_params();
  torch::Tensor map_param_data(const std::vector<torch::Tensor>& param_list);
  void load_w13(const StateDict& state_dict, int idx, bool is_gated);
  void load_w2(const StateDict& state_dict, int idx);
  void load_e_score_correction_bias(const StateDict& state_dict);
  void load_experts(const StateDict& state_dict);
};
TORCH_MODULE(FusedMoE);

}  // namespace layer
}  // namespace xllm

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

#include "fused_moe.h"

#include <glog/logging.h>

#include <algorithm>
#include <cctype>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "framework/parallel_state/parallel_state.h"
#include "kernels/ops_api.h"

namespace xllm {
namespace layer {

namespace {
// Generic local tensor helpers.
torch::Tensor get_tensor_with_weight_suffix(const StateDict& state_dict,
                                            const std::string& tensor_name) {
  auto tensor = state_dict.get_tensor(tensor_name);
  if (!tensor.defined()) {
    tensor = state_dict.get_tensor(tensor_name + ".weight");
  }
  return tensor;
}

torch::Tensor slice_expert_weights(const torch::Tensor& weight,
                                   int64_t start_expert_id,
                                   int64_t num_experts_per_rank) {
  return weight
      .slice(0, start_expert_id, start_expert_id + num_experts_per_rank)
      .contiguous();
}

std::string tensor_debug_info(const torch::Tensor& tensor) {
  if (!tensor.defined()) {
    return "undefined";
  }
  std::ostringstream os;
  os << "dtype=" << c10::toString(tensor.scalar_type())
     << ", shape=" << tensor.sizes() << ", device=" << tensor.device();
  return os.str();
}

std::optional<std::string> resolve_moe_quant_method(
    const QuantArgs& quant_args,
    const StateDict& state_dict) {
  // resolve quant type by first expert.
  static const std::vector<std::vector<std::string>> kExpertPrefixGroups = {
      {"experts.0.gate_proj", "experts.0.up_proj", "experts.0.down_proj"},
      {"experts.0.w1", "experts.0.w3", "experts.0.w2"}};
  std::optional<std::string> first_quant;
  for (const auto& local_prefix_group : kExpertPrefixGroups) {
    if (auto quant = quant_args.get_quant_method_from_prefixes(
            state_dict, local_prefix_group);
        quant.has_value()) {
      if (!first_quant.has_value()) {
        first_quant = quant;
      } else {
        CHECK_EQ(first_quant.value(), quant.value())
            << "Experts have different quant type in same layer: "
            << first_quant.value() << " vs " << quant.value();
      }
    }
  }
  if (!first_quant.has_value()) {
    auto quantize_type = quant_args.quantize_type();
    std::transform(
        quantize_type.begin(),
        quantize_type.end(),
        quantize_type.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (quantize_type == "w4a8_dynamic") {
      first_quant = quantize_type;
    }
  }
  return first_quant;
}

bool is_w8a8_dynamic_quant_method(
    const std::optional<std::string>& quant_method) {
  return quant_method.has_value() && quant_method.value() == "w8a8_dynamic";
}

bool is_w4a8_dynamic_quant_method(
    const std::optional<std::string>& quant_method) {
  return quant_method.has_value() && quant_method.value() == "w4a8_dynamic";
}

bool is_supported_dynamic_moe_quant_method(
    const std::optional<std::string>& quant_method) {
  return is_w8a8_dynamic_quant_method(quant_method) ||
         is_w4a8_dynamic_quant_method(quant_method);
}

bool should_apply_shared_expert_gate(
    const torch::nn::Linear& shared_expert_gate,
    bool is_deepseek_v4,
    bool shared_expert_gate_is_loaded) {
  if (!shared_expert_gate) {
    return false;
  }
  if (is_deepseek_v4) {
    return shared_expert_gate_is_loaded;
  }
  return true;
}

// Qwen3.5-MoE fused checkpoint fallback helpers.
bool load_fused_gate_up_fallback(const StateDict& state_dict,
                                 int64_t rank,
                                 int64_t world_size,
                                 int64_t start_expert_id,
                                 int64_t num_experts_per_rank,
                                 torch::Tensor& w13) {
  auto fused_gate_up =
      get_tensor_with_weight_suffix(state_dict, "gate_up_proj");
  if (!fused_gate_up.defined()) {
    return false;
  }

  if (world_size > 1) {
    CHECK_EQ(fused_gate_up.size(1) % 2, 0)
        << "gate_up_proj dim1 must be even, got " << fused_gate_up.size(1);
    const int64_t full_intermediate = fused_gate_up.size(1) / 2;
    CHECK_EQ(full_intermediate % world_size, 0)
        << "gate_up_proj intermediate dim is not divisible by world_size";
    const int64_t inter_shard = full_intermediate / world_size;

    auto gate_full = fused_gate_up.slice(1, 0, full_intermediate);
    auto up_full =
        fused_gate_up.slice(1, full_intermediate, full_intermediate * 2);
    auto gate_shard =
        gate_full.slice(1, rank * inter_shard, (rank + 1) * inter_shard);
    auto up_shard =
        up_full.slice(1, rank * inter_shard, (rank + 1) * inter_shard);
    fused_gate_up = torch::cat({gate_shard, up_shard}, 1);
  }

  auto gate_up_slice = slice_expert_weights(
      fused_gate_up, start_expert_id, num_experts_per_rank);
  CHECK_EQ(w13.sizes(), gate_up_slice.sizes())
      << "weight size mismatch for " << state_dict.prefix()
      << "experts.gate_up_proj";
  w13.copy_(gate_up_slice);
  return true;
}

bool load_fused_down_fallback(const StateDict& state_dict,
                              int64_t rank,
                              int64_t world_size,
                              int64_t start_expert_id,
                              int64_t num_experts_per_rank,
                              torch::Tensor& w2) {
  auto fused_down = get_tensor_with_weight_suffix(state_dict, "down_proj");
  if (!fused_down.defined()) {
    return false;
  }

  if (world_size > 1) {
    CHECK_EQ(fused_down.size(2) % world_size, 0)
        << "down_proj dim2 is not divisible by world_size";
    const int64_t down_shard = fused_down.size(2) / world_size;
    fused_down =
        fused_down.slice(2, rank * down_shard, (rank + 1) * down_shard);
  }

  auto down_slice =
      slice_expert_weights(fused_down, start_expert_id, num_experts_per_rank);
  CHECK_EQ(w2.sizes(), down_slice.sizes())
      << "weight size mismatch for " << state_dict.prefix()
      << "experts.down_proj";
  w2.copy_(down_slice);
  return true;
}

bool load_fused_up_scale_fallback(const StateDict& state_dict,
                                  int64_t rank,
                                  int64_t world_size,
                                  int64_t start_expert_id,
                                  int64_t num_experts_per_rank,
                                  torch::Tensor& w13_scale) {
  auto fused_gate_up_scale = state_dict.get_tensor("gate_up_proj.weight_scale");
  if (!fused_gate_up_scale.defined()) {
    return false;
  }

  if (world_size > 1) {
    CHECK_GE(fused_gate_up_scale.dim(), 2)
        << "gate_up_proj.weight_scale dim must be >= 2, got "
        << fused_gate_up_scale.dim();
    CHECK_EQ(fused_gate_up_scale.size(1) % 2, 0)
        << "gate_up_proj.weight_scale dim1 must be even, got "
        << fused_gate_up_scale.size(1);
    const int64_t full_intermediate = fused_gate_up_scale.size(1) / 2;
    CHECK_EQ(full_intermediate % world_size, 0)
        << "gate_up_proj.weight_scale intermediate dim is not divisible by "
           "world_size";
    const int64_t inter_shard = full_intermediate / world_size;

    auto gate_full = fused_gate_up_scale.slice(1, 0, full_intermediate);
    auto up_full =
        fused_gate_up_scale.slice(1, full_intermediate, full_intermediate * 2);
    auto gate_shard =
        gate_full.slice(1, rank * inter_shard, (rank + 1) * inter_shard);
    auto up_shard =
        up_full.slice(1, rank * inter_shard, (rank + 1) * inter_shard);
    fused_gate_up_scale = torch::cat({gate_shard, up_shard}, 1);
  }

  auto gate_up_scale_slice = slice_expert_weights(
      fused_gate_up_scale, start_expert_id, num_experts_per_rank);
  if (gate_up_scale_slice.sizes() != w13_scale.sizes() &&
      gate_up_scale_slice.numel() == w13_scale.numel()) {
    gate_up_scale_slice = gate_up_scale_slice.reshape(w13_scale.sizes());
  }
  CHECK_EQ(w13_scale.sizes(), gate_up_scale_slice.sizes())
      << "weight size mismatch for " << state_dict.prefix()
      << "experts.gate_up_proj.weight_scale";
  w13_scale.copy_(gate_up_scale_slice);
  return true;
}

bool load_fused_down_scale_fallback(const StateDict& state_dict,
                                    int64_t start_expert_id,
                                    int64_t num_experts_per_rank,
                                    torch::Tensor& w2_scale) {
  auto fused_down_scale = state_dict.get_tensor("down_proj.weight_scale");
  if (!fused_down_scale.defined()) {
    return false;
  }

  auto down_scale_slice = slice_expert_weights(
      fused_down_scale, start_expert_id, num_experts_per_rank);
  if (down_scale_slice.sizes() != w2_scale.sizes() &&
      down_scale_slice.numel() == w2_scale.numel()) {
    down_scale_slice = down_scale_slice.reshape(w2_scale.sizes());
  }
  CHECK_EQ(w2_scale.sizes(), down_scale_slice.sizes())
      << "weight size mismatch for " << state_dict.prefix()
      << "experts.down_proj.weight_scale";
  w2_scale.copy_(down_scale_slice);
  return true;
}

}  // namespace

FusedMoEImpl::FusedMoEImpl(const ModelArgs& model_args,
                           const FusedMoEArgs& moe_args,
                           const QuantArgs& quant_args,
                           const ParallelArgs& parallel_args,
                           const torch::TensorOptions& options)
    : num_total_experts_(model_args.n_routed_experts()),
      topk_(model_args.num_experts_per_tok()),
      num_expert_group_(model_args.n_group()),
      topk_group_(model_args.topk_group()),
      route_scale_(model_args.routed_scaling_factor()),
      hidden_size_(model_args.hidden_size()),
      n_shared_experts_(model_args.n_shared_experts()),
      is_gated_(moe_args.is_gated),
      skip_gate_load_(moe_args.skip_gate_load),
      is_deepseek_v4_(model_args.model_type() == "deepseek_v4"),
      renormalize_(model_args.norm_topk_prob() ? 1 : 0),
      hidden_act_(model_args.hidden_act()),
      scoring_func_(model_args.scoring_func()),
      quant_args_(quant_args),
      parallel_args_(parallel_args),
      options_(options),
      tp_pg_(parallel_args.tp_group_) {
  const int64_t num_experts = num_total_experts_;
  const int64_t intermediate_size =
      static_cast<int64_t>(model_args.moe_intermediate_size());
  const std::string& topk_method = model_args.topk_method();
  int64_t ep_size = parallel_args.ep_size();
  int64_t ep_rank = 0;
  if (ep_size > 1) {
    ep_rank = parallel_args.moe_ep_group_->rank();
    tp_pg_ = parallel_args.moe_tp_group_;
  }

  // calculate the number of experts per rank
  num_experts_per_rank_ = num_experts / ep_size;
  start_expert_id_ = ep_rank * num_experts_per_rank_;

  if (topk_method == "noaux_tc") {
    e_score_correction_bias_ = register_parameter(
        "e_score_correction_bias", torch::empty({num_experts}, options), false);
  }

  gate_ = register_module(
      "gate_proj",
      ReplicatedLinear(hidden_size_, num_experts, false, quant_args, options));
  if (n_shared_experts_ > 0) {
    /*
    The shared_experts are usually implemented using the RowParallelLinear
    layer. Typically, this output serves as the enable_result_reduction results
    for the module. If only tensor parallelism is applied, immediate
    reduction of the shared_experts output isn't necessary; instead, we perform
    the reduction once at the end of the MoE operation.
    */
    shared_experts_ =
        register_module("shared_experts",
                        DenseMLP(hidden_size_,
                                 intermediate_size * n_shared_experts_,
                                 is_gated_,
                                 false,
                                 hidden_act_,
                                 /*enable_result_reduction=*/false,
                                 quant_args,
                                 tp_pg_,
                                 options));
    shared_expert_gate_ = register_module(
        "shared_expert_gate",
        torch::nn::Linear(
            torch::nn::LinearOptions(hidden_size_, 1).bias(false)));
    shared_expert_gate_->weight.set_data(
        shared_expert_gate_->weight.to(options));
  }

  // create weight buffer
  const int64_t world_size = tp_pg_->world_size();
  CHECK_GT(world_size, 0) << "invalid MoE tp world_size: " << world_size;
  CHECK_GT(intermediate_size, 0)
      << "invalid moe_intermediate_size: " << intermediate_size;
  CHECK_EQ(intermediate_size % world_size, 0)
      << "moe_intermediate_size(" << intermediate_size
      << ") must be divisible by MoE tp world_size(" << world_size << ")";
  int64_t local_intermediate_size = intermediate_size / world_size;
  local_intermediate_size_ = local_intermediate_size;
  LOG(INFO) << "[MOE_LOAD_DEBUG][FusedMoEInit] hidden_size=" << hidden_size_
            << ", intermediate_size=" << intermediate_size
            << ", local_intermediate_size=" << local_intermediate_size
            << ", model_args.intermediate_size="
            << model_args.intermediate_size()
            << ", model_args.moe_intermediate_size="
            << model_args.moe_intermediate_size()
            << ", tp_group(rank/world)=" << tp_pg_->rank() << "/"
            << tp_pg_->world_size() << ", ep_size=" << ep_size
            << ", ep_rank=" << ep_rank
            << ", num_experts_per_rank=" << num_experts_per_rank_
            << ", start_expert_id=" << start_expert_id_ << ", quant_method="
            << (quant_args_.quant_method().empty()
                    ? "<empty>"
                    : quant_args_.quant_method());
  if (quant_args_.quant_method() == kQuantMethodAscendInt4) {
    CHECK_EQ(hidden_size_ % 2, 0)
        << "Ascend int4 FusedMoE expects even hidden_size, got "
        << hidden_size_;
    // Ascend int4 stores two int4 values in one int8 slot. Initialize the
    // packed layout up front; load_state_dict can still re-register when the
    // resolved per-weight quant method disagrees with this model-level hint.
    w13_ = register_parameter(
        "w13",
        torch::empty(
            {num_experts_per_rank_, local_intermediate_size, hidden_size_},
            options_.dtype(torch::kInt8)),
        false);
    w2_ = register_parameter(
        "w2",
        torch::empty(
            {num_experts_per_rank_, hidden_size_ / 2, local_intermediate_size},
            options_.dtype(torch::kInt8)),
        false);
  } else if (quant_args_.quant_method() == kQuantMethodAscendInt8) {
    // Ascend int8 uses the normal MoE weight shape with int8 storage.
    // During load_state_dict, the weight will be lazily re-registered to the
    // appropriate dtype based on the resolved quant method.
    w13_ = register_parameter(
        "w13",
        torch::empty(
            {num_experts_per_rank_, local_intermediate_size * 2, hidden_size_},
            options_.dtype(torch::kInt8)),
        false);
    w2_ = register_parameter(
        "w2",
        torch::empty(
            {num_experts_per_rank_, hidden_size_, local_intermediate_size},
            options_.dtype(torch::kInt8)),
        false);
  } else {
    w13_ = register_parameter(
        "w13",
        torch::empty(
            {num_experts_per_rank_, local_intermediate_size * 2, hidden_size_},
            options_),
        false);
    w2_ = register_parameter(
        "w2",
        torch::empty(
            {num_experts_per_rank_, hidden_size_, local_intermediate_size},
            options_),
        false);
  }
}

void FusedMoEImpl::validate_resolved_quant_method() const {
  if (!resolved_moe_quant_method_.has_value()) {
    return;
  }
  if (is_w8a8_dynamic_quant_method(resolved_moe_quant_method_)) {
    if (!is_gated_ || (hidden_act_ != "silu" && hidden_act_ != "swiglu")) {
      LOG(WARNING) << "W8A8_DYNAMIC FusedMoE currently uses dequant+swiglu "
                      "path, but got is_gated="
                   << (is_gated_ ? "true" : "false")
                   << ", hidden_act=" << hidden_act_
                   << ". This may cause activation behavior mismatch.";
    }
  } else if (is_w4a8_dynamic_quant_method(resolved_moe_quant_method_)) {
    CHECK(is_gated_ && (hidden_act_ == "silu" || hidden_act_ == "swiglu"))
        << "W4A8_DYNAMIC FusedMoE currently assumes gated SiLU/SwiGLU. "
        << "got is_gated=" << (is_gated_ ? "true" : "false")
        << ", hidden_act=" << hidden_act_;
    CHECK_GE(quant_args_.group_size(), 0)
        << "W4A8_DYNAMIC group_size must be >= 0, got "
        << quant_args_.group_size();
    CHECK_EQ(quant_args_.quant_version(), "1.0.0")
        << "W4A8_DYNAMIC only supports quant_version 1.0.0, got "
        << (quant_args_.quant_version().empty() ? "<empty>"
                                                : quant_args_.quant_version());
    CHECK_LE(tp_pg_->world_size(), 16)
        << "W4A8_DYNAMIC version 1.0.0 does not support MoE TP > 16, "
        << "got " << tp_pg_->world_size();
  } else {
    LOG(FATAL) << "Unsupported MoE quant_method for NPU FusedMoE: "
               << resolved_moe_quant_method_.value();
  }
}

void FusedMoEImpl::ensure_quant_weight_layout() {
  std::vector<weight::LazyParameterSpec> specs;
  specs.reserve(18);
  auto push = [&](torch::Tensor& tensor,
                  bool& tensor_is_loaded,
                  const char* name,
                  std::vector<int64_t> sizes,
                  const torch::TensorOptions& tensor_options) {
    specs.push_back(weight::LazyParameterSpec{
        &tensor, &tensor_is_loaded, name, std::move(sizes), tensor_options});
  };

  const auto fp32_options = options_.dtype(torch::kFloat32);
  if (is_w8a8_dynamic_quant_method(resolved_moe_quant_method_)) {
    auto w2_scale_options = options_.dtype() == torch::kBFloat16
                                ? options_.dtype(torch::kBFloat16)
                                : options_.dtype(torch::kFloat32);
    push(w13_,
         w13_is_loaded_,
         "w13",
         {num_experts_per_rank_, local_intermediate_size_ * 2, hidden_size_},
         options_.dtype(torch::kInt8));
    push(w2_,
         w2_is_loaded_,
         "w2",
         {num_experts_per_rank_, hidden_size_, local_intermediate_size_},
         options_.dtype(torch::kInt8));
    push(w13_scale_,
         w13_scale_is_loaded_,
         "w13_scale",
         {num_experts_per_rank_, local_intermediate_size_ * 2},
         fp32_options);
    push(w2_scale_,
         w2_scale_is_loaded_,
         "w2_scale",
         {num_experts_per_rank_, hidden_size_},
         w2_scale_options);
    weight::ensure_parameter_storage(this, specs);
    return;
  }

  if (!is_w4a8_dynamic_quant_method(resolved_moe_quant_method_)) {
    return;
  }

  const int64_t w13_weight_out = local_intermediate_size_;
  const int64_t w2_weight_out = hidden_size_ / 2;
  CHECK_EQ(hidden_size_ % 2, 0)
      << "W4A8_DYNAMIC version 1.0.0 expects even hidden_size, got "
      << hidden_size_;

  push(w13_,
       w13_is_loaded_,
       "w13",
       {num_experts_per_rank_, w13_weight_out, hidden_size_},
       options_.dtype(torch::kInt8));
  push(w2_,
       w2_is_loaded_,
       "w2",
       {num_experts_per_rank_, w2_weight_out, local_intermediate_size_},
       options_.dtype(torch::kInt8));

  push(w13_scale_,
       w13_scale_is_loaded_,
       "w13_scale",
       {num_experts_per_rank_, local_intermediate_size_ * 2, 1},
       fp32_options);
  push(w2_scale_,
       w2_scale_is_loaded_,
       "w2_scale",
       {num_experts_per_rank_, hidden_size_, 1},
       fp32_options);

  if (quant_args_.group_size() > 0) {
    CHECK_EQ(hidden_size_ % quant_args_.group_size(), 0)
        << "W4A8_DYNAMIC hidden_size must be divisible by group_size, got "
        << hidden_size_ << " and group_size=" << quant_args_.group_size();
    CHECK_EQ(local_intermediate_size_ % quant_args_.group_size(), 0)
        << "W4A8_DYNAMIC local_intermediate_size must be divisible by "
        << "group_size, got " << local_intermediate_size_
        << " and group_size=" << quant_args_.group_size();
    push(w13_scale_second_,
         w13_scale_second_is_loaded_,
         "w13_scale_second",
         {num_experts_per_rank_,
          local_intermediate_size_ * 2,
          hidden_size_ / quant_args_.group_size()},
         fp32_options);
    push(w2_scale_second_,
         w2_scale_second_is_loaded_,
         "w2_scale_second",
         {num_experts_per_rank_,
          hidden_size_,
          local_intermediate_size_ / quant_args_.group_size()},
         fp32_options);
  }

  push(w13_scale_bias_,
       w13_scale_bias_is_loaded_,
       "w13_scale_bias",
       {num_experts_per_rank_, local_intermediate_size_ * 2, 1},
       fp32_options);
  push(w2_scale_bias_,
       w2_scale_bias_is_loaded_,
       "w2_scale_bias",
       {num_experts_per_rank_, hidden_size_, 16 / tp_pg_->world_size()},
       fp32_options);

  weight::ensure_parameter_storage(this, specs);
}

void FusedMoEImpl::resolve_quant_method_from_state_dict(
    const StateDict& state_dict) {
  resolved_moe_quant_method_ =
      resolve_moe_quant_method(quant_args_, state_dict);
  LOG(INFO) << "[QUANT_DEBUG][FusedMoELoad] prefix=" << state_dict.prefix()
            << ", resolved_moe_quant_method="
            << resolved_moe_quant_method_.value_or("<none>")
            << ", quant_desc_count=" << quant_args_.quant_descs().size();
  if (is_supported_dynamic_moe_quant_method(resolved_moe_quant_method_)) {
    validate_resolved_quant_method();
    ensure_quant_weight_layout();
  } else if (quant_args_.quant_method() == kQuantMethodAscendInt4 ||
             quant_args_.quant_method() == kQuantMethodAscendInt8 ||
             !quant_args_.quant_descs().empty()) {
    // The constructor may have used an Ascend quantized layout as a
    // model-level hint, but the actual per-weight method resolved to an
    // unsupported/non-quantized path. Restore the unquantized MoE layout so
    // load_experts can copy checkpoint weights correctly.
    std::vector<weight::LazyParameterSpec> specs;
    specs.reserve(2);
    auto push = [&](torch::Tensor& tensor,
                    bool& tensor_is_loaded,
                    const char* name,
                    std::vector<int64_t> sizes) {
      specs.push_back(weight::LazyParameterSpec{
          &tensor, &tensor_is_loaded, name, std::move(sizes), options_});
    };
    push(w13_,
         w13_is_loaded_,
         "w13",
         {num_experts_per_rank_, local_intermediate_size_ * 2, hidden_size_});
    push(w2_,
         w2_is_loaded_,
         "w2",
         {num_experts_per_rank_, hidden_size_, local_intermediate_size_});
    weight::ensure_parameter_storage(this, specs);
  }
}

torch::Tensor FusedMoEImpl::select_experts(
    const torch::Tensor& hidden_states_2d,
    const torch::Tensor& router_logits_2d,
    SelectedExpertInfo& selected_expert_info) {
  torch::Tensor topk_weights;
  torch::Tensor topk_ids;
  std::optional<torch::Tensor> e_score_correction_bias = std::nullopt;
  if (e_score_correction_bias_.defined()) {
    e_score_correction_bias = e_score_correction_bias_;
  }
  if (preselected_experts_.has_value()) {
    const auto& selected = preselected_experts_.value();
    topk_weights = selected.first.reshape({-1, topk_});
    topk_ids = selected.second.reshape({-1, topk_}).to(torch::kInt32);
    CHECK_EQ(topk_weights.size(0), hidden_states_2d.size(0))
        << "preselected topk_weights token count mismatch, expected "
        << hidden_states_2d.size(0) << ", got " << topk_weights.size(0);
    CHECK_EQ(topk_ids.size(0), hidden_states_2d.size(0))
        << "preselected topk_ids token count mismatch, expected "
        << hidden_states_2d.size(0) << ", got " << topk_ids.size(0);
    topk_weights = topk_weights.to(hidden_states_2d.dtype());
  } else {
    // prepare the parameters for select_experts
    xllm::kernel::MoeFusedTopkParams moe_active_topk_params;
    moe_active_topk_params.input = router_logits_2d;
    moe_active_topk_params.topk = topk_;
    // moe_active_topk_params.num_expert_group = num_expert_group_;
    // moe_active_topk_params.topk_group = topk_group_;
    moe_active_topk_params.normalize = static_cast<bool>(renormalize_);
    // moe_active_topk_params.normed_by = "topk_logit";
    moe_active_topk_params.scoring_func = scoring_func_;
    // moe_active_topk_params.route_scale = route_scale_;
    // moe_active_topk_params.e_score_correction_bias = e_score_correction_bias;
    std::tie(topk_weights, topk_ids) =
        xllm::kernel::moe_active_topk(moe_active_topk_params);
    topk_ids = topk_ids.to(torch::kInt32);
  }

  xllm::kernel::MoeInitRoutingV2Params moe_init_routing_params;
  moe_init_routing_params.x = hidden_states_2d;
  moe_init_routing_params.expert_idx = topk_ids;
  moe_init_routing_params.scale = std::nullopt;
  moe_init_routing_params.offset = std::nullopt;
  moe_init_routing_params.active_num = hidden_states_2d.size(0) * topk_;
  moe_init_routing_params.expert_capacity = 0;
  moe_init_routing_params.expert_num = num_experts_per_rank_;
  moe_init_routing_params.drop_pad_mode = 0;
  moe_init_routing_params.expert_tokens_num_type = 1;
  moe_init_routing_params.expert_tokens_num_flag = true;
  moe_init_routing_params.row_idx_type = 0;
  std::vector<int64_t> expert_range = {
      start_expert_id_, start_expert_id_ + num_experts_per_rank_};
  moe_init_routing_params.active_expert_range = expert_range;
  moe_init_routing_params.quant_mode = -1;
  // TODO: NPU moe_init_routing_v2 is equivalent to moe_gen_idx +
  // moe_expand_input (and the token_count/cusum outputs) on other backends.
  auto [expand_hidden_states, expand_row_ids, group_list, dynamic_scale] =
      xllm::kernel::moe_init_routing_v2(moe_init_routing_params);
  (void)dynamic_scale;

  // collect the selected tensor
  selected_expert_info.reduce_weight = topk_weights;
  selected_expert_info.combine_idx = expand_row_ids;
  selected_expert_info.token_count_slice = group_list.to(torch::kInt64);
  selected_expert_info.cusum_token_count = group_list;
  return expand_hidden_states;
}

torch::Tensor FusedMoEImpl::forward_expert(
    const torch::Tensor& hidden_states,
    const torch::Tensor& router_logits,
    const std::optional<torch::Tensor>& shared_output) {
  // prepare the parameters for MoE computation
  torch::IntArrayRef hidden_states_shape = hidden_states.sizes();
  torch::ScalarType hidden_states_dtype = hidden_states.dtype().toScalarType();
  torch::Tensor hidden_states_2d =
      hidden_states.reshape({-1, hidden_states.size(-1)});
  torch::Tensor router_logits_2d =
      router_logits.reshape({-1, router_logits.size(-1)});

  // Step 1-3: select experts
  SelectedExpertInfo selected_expert_info;
  torch::Tensor expand_hidden_states =
      select_experts(hidden_states_2d, router_logits_2d, selected_expert_info);

  torch::Tensor gemm1_out;
  torch::Tensor gemm2_out;
  if (is_w8a8_dynamic_quant_method(resolved_moe_quant_method_)) {
    CHECK(w13_scale_is_loaded_ && w13_scale_.defined())
        << "w13_scale is required for W8A8 fused MoE.";
    CHECK(w2_scale_is_loaded_ && w2_scale_.defined())
        << "w2_scale is required for W8A8 fused MoE.";

    // Step 4: dynamic quant on expanded inputs.
    xllm::kernel::NpuQuantizeParams quant_params;
    quant_params.input = expand_hidden_states;
    torch::Tensor quantized_expand_hidden_states;
    std::optional<torch::Tensor> pertoken_scale;
    std::tie(quantized_expand_hidden_states, pertoken_scale) =
        xllm::kernel::dynamic_quant(quant_params);
    CHECK(pertoken_scale.has_value() && pertoken_scale->defined())
        << "dynamic_quant must return per-token scale for W8A8 fused MoE.";

    // Step 5: first grouped matmul (int32 output expected for dequant+swiglu).
    if (w13_.size(1) != quantized_expand_hidden_states.size(1)) {
      w13_ = w13_.transpose(1, 2);
    }

    {
      std::vector<torch::Tensor> x_list = {quantized_expand_hidden_states};
      std::vector<torch::Tensor> weight_list = {w13_};
      xllm::kernel::GroupGemmParams group_gemm_params;
      group_gemm_params.x_list = torch::TensorList(x_list);
      group_gemm_params.weight_list = torch::TensorList(weight_list);
      group_gemm_params.group_list = selected_expert_info.token_count_slice;
      group_gemm_params.split_item = 2;
      group_gemm_params.group_type = 0;
      group_gemm_params.group_list_type = 1;
      group_gemm_params.output_dtype = torch::kInt32;
      gemm1_out = xllm::kernel::group_gemm(group_gemm_params);
    }

    // Step 6: fused dequant + swiglu + quant.
    torch::Tensor act_quantized;
    torch::Tensor act_scale;
    {
      xllm::kernel::DequantSwigluQuantParams params;
      params.x = gemm1_out;
      params.weight_scale = w13_scale_;
      params.activation_scale = pertoken_scale.value();
      params.group_index = selected_expert_info.token_count_slice;
      params.activate_left = true;
      params.quant_mode = 1;
      std::tie(act_quantized, act_scale) =
          xllm::kernel::dequant_swiglu_quant(params);
    }

    // Step 7: second grouped matmul (dequant to hidden dtype).
    if (w2_.size(1) != act_quantized.size(1)) {
      w2_ = w2_.transpose(1, 2);
    }
    {
      std::vector<torch::Tensor> x_list = {act_quantized};
      std::vector<torch::Tensor> weight_list = {w2_};
      std::vector<torch::Tensor> scale_list = {w2_scale_};
      std::vector<torch::Tensor> per_token_scale_list = {act_scale};
      xllm::kernel::GroupGemmParams group_gemm_params;
      group_gemm_params.x_list = torch::TensorList(x_list);
      group_gemm_params.weight_list = torch::TensorList(weight_list);
      group_gemm_params.scale_list = torch::TensorList(scale_list);
      group_gemm_params.per_token_scale_list =
          torch::TensorList(per_token_scale_list);
      group_gemm_params.group_list = selected_expert_info.token_count_slice;
      group_gemm_params.split_item = 2;
      group_gemm_params.group_type = 0;
      group_gemm_params.group_list_type = 1;
      group_gemm_params.output_dtype = hidden_states_dtype;
      gemm2_out = xllm::kernel::group_gemm(group_gemm_params);
    }
  } else if (is_w4a8_dynamic_quant_method(resolved_moe_quant_method_)) {
    preprocess_w4a8_dynamic_weights();
    CHECK(w4a8_dynamic_preprocessed_)
        << "W4A8_DYNAMIC fused MoE weights were not preprocessed. Check "
        << "whether all W4A8 weight/scale tensors have been loaded and "
        << "whether the NPU preprocess implementation completed.";
    CHECK(w13_scale_is_loaded_ && w13_scale_.defined())
        << "w13_scale is required for W4A8_DYNAMIC fused MoE.";
    CHECK(w2_scale_is_loaded_ && w2_scale_.defined())
        << "w2_scale is required for W4A8_DYNAMIC fused MoE.";
    CHECK(w13_scale_bias_is_loaded_ && w13_scale_bias_.defined())
        << "w13_scale_bias is required for W4A8_DYNAMIC fused MoE.";
    CHECK(w2_scale_bias_is_loaded_ && w2_scale_bias_.defined())
        << "w2_scale_bias is required for W4A8_DYNAMIC fused MoE.";
    // Match vllm-ascend's current W4A8_DYNAMIC TODO path. Revisit once the
    // real grouped matmul operator contract can report the desired dtype.
    const auto w4a8_group_gemm_output_dtype = torch::kBFloat16;

    xllm::kernel::NpuQuantizeParams quant_params;
    quant_params.input = expand_hidden_states;
    torch::Tensor quantized_expand_hidden_states;
    std::optional<torch::Tensor> pertoken_scale;
    std::tie(quantized_expand_hidden_states, pertoken_scale) =
        xllm::kernel::dynamic_quant(quant_params);
    CHECK(pertoken_scale.has_value() && pertoken_scale->defined())
        << "dynamic_quant must return per-token scale for W4A8_DYNAMIC "
        << "fused MoE.";

    // W4A8_DYNAMIC weights are expected to be transposed/NZ-converted and
    // packed by preprocess_w4a8_dynamic_weights(), matching vllm-ascend's
    // process_weights_after_loading path. Do not do the ad-hoc transpose used
    // by the unquantized/W8A8 layout fallback.
    {
      std::vector<torch::Tensor> x_list = {quantized_expand_hidden_states};
      std::vector<torch::Tensor> weight_list = {w13_};
      std::vector<torch::Tensor> scale_list = {w13_scale_};
      std::vector<torch::Tensor> per_token_scale_list = {
          pertoken_scale.value()};
      std::vector<torch::Tensor> bias_list = {w13_scale_bias_};
      xllm::kernel::GroupGemmParams group_gemm_params;
      group_gemm_params.x_list = torch::TensorList(x_list);
      group_gemm_params.weight_list = torch::TensorList(weight_list);
      group_gemm_params.scale_list = torch::TensorList(scale_list);
      group_gemm_params.bias_list = torch::TensorList(bias_list);
      group_gemm_params.per_token_scale_list =
          torch::TensorList(per_token_scale_list);
      group_gemm_params.group_list = selected_expert_info.token_count_slice;
      group_gemm_params.split_item = 2;
      group_gemm_params.group_type = 0;
      group_gemm_params.group_list_type = 1;
      group_gemm_params.output_dtype = w4a8_group_gemm_output_dtype;
      gemm1_out = xllm::kernel::group_gemm(group_gemm_params);
    }

    torch::Tensor act_out;
    xllm::kernel::ActivationParams activation_params;
    activation_params.input = gemm1_out;
    activation_params.output = act_out;
    activation_params.act_mode = hidden_act_;
    activation_params.is_gated = is_gated_;
    xllm::kernel::active(activation_params);
    act_out = activation_params.output;

    torch::Tensor act_quantized;
    std::optional<torch::Tensor> act_scale;
    quant_params = xllm::kernel::NpuQuantizeParams{};
    quant_params.input = act_out;
    std::tie(act_quantized, act_scale) =
        xllm::kernel::dynamic_quant(quant_params);
    CHECK(act_scale.has_value() && act_scale->defined())
        << "dynamic_quant must return activation scale after W4A8_DYNAMIC "
        << "SwiGLU.";

    {
      std::vector<torch::Tensor> x_list = {act_quantized};
      std::vector<torch::Tensor> weight_list = {w2_};
      std::vector<torch::Tensor> scale_list = {w2_scale_};
      std::vector<torch::Tensor> per_token_scale_list = {act_scale.value()};
      std::vector<torch::Tensor> bias_list = {w2_scale_bias_};
      xllm::kernel::GroupGemmParams group_gemm_params;
      group_gemm_params.x_list = torch::TensorList(x_list);
      group_gemm_params.weight_list = torch::TensorList(weight_list);
      group_gemm_params.scale_list = torch::TensorList(scale_list);
      group_gemm_params.bias_list = torch::TensorList(bias_list);
      group_gemm_params.per_token_scale_list =
          torch::TensorList(per_token_scale_list);
      group_gemm_params.group_list = selected_expert_info.token_count_slice;
      group_gemm_params.split_item = 2;
      group_gemm_params.group_type = 0;
      group_gemm_params.group_list_type = 1;
      group_gemm_params.output_dtype = w4a8_group_gemm_output_dtype;
      gemm2_out = xllm::kernel::group_gemm(group_gemm_params);
    }
  } else {
    // Step 4: group gemm 1
    {
      xllm::kernel::GroupGemmParams group_gemm_params;
      group_gemm_params.a = expand_hidden_states;
      if (w13_.size(1) != expand_hidden_states.size(1)) {
        w13_ = w13_.transpose(1, 2);
      }
      group_gemm_params.b = w13_;
      group_gemm_params.group_list = selected_expert_info.token_count_slice;
      group_gemm_params.split_item = 2;
      group_gemm_params.group_type = 0;
      group_gemm_params.group_list_type = 1;
      gemm1_out = xllm::kernel::group_gemm(group_gemm_params);
    }

    // Step 5: activation
    torch::Tensor act_out;

    xllm::kernel::ActivationParams activation_params;
    activation_params.input = gemm1_out;
    activation_params.output = act_out;
    activation_params.act_mode = hidden_act_;
    activation_params.is_gated = is_gated_;
    xllm::kernel::active(activation_params);
    act_out = activation_params.output;

    // Step 6: group gemm 2
    {
      xllm::kernel::GroupGemmParams group_gemm_params;
      group_gemm_params.a = act_out;
      if (w2_.size(1) != act_out.size(1)) {
        w2_ = w2_.transpose(1, 2);
      }
      group_gemm_params.b = w2_;
      group_gemm_params.group_list = selected_expert_info.token_count_slice;
      group_gemm_params.split_item = 2;
      group_gemm_params.group_type = 0;
      group_gemm_params.group_list_type = 1;
      gemm2_out = xllm::kernel::group_gemm(group_gemm_params);
    }
  }

  // Step 8: combine the intermediate results and get the final hidden states
  torch::Tensor final_hidden_states;
  xllm::kernel::MoeCombineResultParams moe_combine_params;
  moe_combine_params.input = gemm2_out;
  moe_combine_params.reduce_weight = selected_expert_info.reduce_weight;
  moe_combine_params.gather_ids = selected_expert_info.combine_idx;
  final_hidden_states = xllm::kernel::moe_combine_result(moe_combine_params);
  if (is_deepseek_v4_) {
    final_hidden_states = final_hidden_states * route_scale_;
  }
  if (shared_output.has_value()) {
    final_hidden_states = final_hidden_states + shared_output.value();
  }
  // reshape the final hidden states to the original shape
  final_hidden_states = final_hidden_states.reshape(hidden_states_shape);

  if (tp_pg_->world_size() > 1) {
    final_hidden_states = parallel_state::reduce(final_hidden_states, tp_pg_);
  }
  if (parallel_args_.ep_size() > 1) {
    final_hidden_states = parallel_state::reduce(final_hidden_states,
                                                 parallel_args_.moe_ep_group_);
  }
  return final_hidden_states;
}

torch::Tensor FusedMoEImpl::forward(const torch::Tensor& hidden_states,
                                    const ModelInputParams& input_params) {
  auto input = hidden_states;
  bool need_slice = false;
  if (parallel_args_.dp_size() > 1 && parallel_args_.ep_size() > 1) {
    input = parallel_state::gather(input,
                                   parallel_args_.dp_local_process_group_,
                                   input_params.dp_global_token_nums);
    need_slice = true;
  }

  std::optional<torch::Tensor> shared_output = std::nullopt;
  if (n_shared_experts_ > 0) {
    shared_output = shared_experts_(input);
    if (should_apply_shared_expert_gate(shared_expert_gate_,
                                        is_deepseek_v4_,
                                        shared_expert_gate_is_loaded_)) {
      auto gate = torch::sigmoid(shared_expert_gate_->forward(input));
      if (shared_output.has_value()) {
        torch::Tensor res = gate * shared_output.value();
        shared_output = res;
      }
    }
  }
  auto router_logits = gate_(input);
  auto output = forward_expert(input, router_logits, shared_output);

  if (need_slice) {
    const auto& dp_tokens = input_params.dp_global_token_nums;
    const int64_t dp_rank = parallel_args_.dp_local_process_group_->rank();
    auto start =
        std::accumulate(dp_tokens.begin(), dp_tokens.begin() + dp_rank, 0);
    auto end = start + dp_tokens[dp_rank];
    output = output.slice(0, start, end);
  }
  return output;
}

torch::Tensor FusedMoEImpl::forward_with_selected_experts(
    const torch::Tensor& hidden_states,
    const torch::Tensor& topk_weights,
    const torch::Tensor& topk_ids,
    const ModelInputParams& input_params) {
  auto input = hidden_states;
  auto selected_topk_weights = topk_weights;
  auto selected_topk_ids = topk_ids;
  bool need_slice = false;
  if (parallel_args_.dp_size() > 1 && parallel_args_.ep_size() > 1) {
    input = parallel_state::gather(input,
                                   parallel_args_.dp_local_process_group_,
                                   input_params.dp_global_token_nums);
    selected_topk_weights =
        parallel_state::gather(selected_topk_weights,
                               parallel_args_.dp_local_process_group_,
                               input_params.dp_global_token_nums);
    selected_topk_ids =
        parallel_state::gather(selected_topk_ids,
                               parallel_args_.dp_local_process_group_,
                               input_params.dp_global_token_nums);
    need_slice = true;
  }

  auto hidden_rows = input.reshape({-1, input.size(-1)}).size(0);
  auto weights_2d = selected_topk_weights.reshape({-1, topk_});
  auto ids_2d = selected_topk_ids.reshape({-1, topk_});
  CHECK_EQ(weights_2d.size(0), hidden_rows)
      << "topk_weights token count mismatch, expected " << hidden_rows
      << ", got " << weights_2d.size(0);
  CHECK_EQ(ids_2d.size(0), hidden_rows)
      << "topk_ids token count mismatch, expected " << hidden_rows << ", got "
      << ids_2d.size(0);

  preselected_experts_ =
      std::make_pair(weights_2d.to(input.dtype()), ids_2d.to(torch::kInt32));

  std::optional<torch::Tensor> shared_output = std::nullopt;
  if (n_shared_experts_ > 0) {
    shared_output = shared_experts_(input);
    if (should_apply_shared_expert_gate(shared_expert_gate_,
                                        is_deepseek_v4_,
                                        shared_expert_gate_is_loaded_)) {
      auto gate = torch::sigmoid(shared_expert_gate_->forward(input));
      if (shared_output.has_value()) {
        torch::Tensor res = gate * shared_output.value();
        shared_output = res;
      }
    }
  }

  auto router_shape = input.sizes().vec();
  router_shape.back() = num_total_experts_;
  auto router_logits = torch::empty(router_shape, input.options());
  auto output = forward_expert(input, router_logits, shared_output);
  preselected_experts_ = std::nullopt;

  if (need_slice) {
    const auto& dp_tokens = input_params.dp_global_token_nums;
    const int64_t dp_rank = parallel_args_.dp_local_process_group_->rank();
    auto start =
        std::accumulate(dp_tokens.begin(), dp_tokens.begin() + dp_rank, 0);
    auto end = start + dp_tokens[dp_rank];
    output = output.slice(0, start, end);
  }
  return output;
}

void FusedMoEImpl::load_e_score_correction_bias(const StateDict& state_dict) {
  if (e_score_correction_bias_.defined() &&
      !e_score_correction_bias_is_loaded_) {
    LOAD_WEIGHT(e_score_correction_bias);
  }
}

void FusedMoEImpl::preprocess_w4a8_dynamic_weights() {
  if (w4a8_dynamic_preprocessed_ ||
      !is_w4a8_dynamic_quant_method(resolved_moe_quant_method_)) {
    return;
  }

  CHECK_EQ(quant_args_.quant_version(), "1.0.0")
      << "W4A8_DYNAMIC only supports quant_version 1.0.0, got "
      << (quant_args_.quant_version().empty() ? "<empty>"
                                              : quant_args_.quant_version());
  const bool base_loaded = w13_is_loaded_ && w2_is_loaded_ &&
                           w13_scale_is_loaded_ && w2_scale_is_loaded_;
  if (!base_loaded) {
    LOG(INFO) << "[QUANT_DEBUG][FusedMoE][W4A8Dynamic] wait for base "
              << "weights/scales before preprocess. w13_loaded="
              << (w13_is_loaded_ ? "true" : "false")
              << ", w2_loaded=" << (w2_is_loaded_ ? "true" : "false")
              << ", w13_scale_loaded="
              << (w13_scale_is_loaded_ ? "true" : "false")
              << ", w2_scale_loaded="
              << (w2_scale_is_loaded_ ? "true" : "false");
    return;
  }
  if (quant_args_.group_size() > 0 &&
      !(w13_scale_second_is_loaded_ && w2_scale_second_is_loaded_)) {
    LOG(INFO) << "[QUANT_DEBUG][FusedMoE][W4A8Dynamic] wait for "
              << "weight_scale_second before preprocess.";
    return;
  }
  if (!(w13_scale_bias_is_loaded_ && w2_scale_bias_is_loaded_)) {
    LOG(INFO) << "[QUANT_DEBUG][FusedMoE][W4A8Dynamic] wait for scale_bias "
              << "before preprocess.";
    return;
  }

  xllm::kernel::W4A8DynamicMoePreprocessParams params;
  params.w13_weight = w13_;
  params.w2_weight = w2_;
  params.w13_weight_scale = w13_scale_;
  params.w2_weight_scale = w2_scale_;
  params.w13_weight_scale_second =
      w13_scale_second_is_loaded_
          ? std::optional<torch::Tensor>(w13_scale_second_)
          : std::nullopt;
  params.w2_weight_scale_second =
      w2_scale_second_is_loaded_
          ? std::optional<torch::Tensor>(w2_scale_second_)
          : std::nullopt;
  params.w13_scale_bias = w13_scale_bias_is_loaded_
                              ? std::optional<torch::Tensor>(w13_scale_bias_)
                              : std::nullopt;
  params.w2_scale_bias = w2_scale_bias_is_loaded_
                             ? std::optional<torch::Tensor>(w2_scale_bias_)
                             : std::nullopt;
  params.group_size = quant_args_.group_size();

  torch::Tensor processed_w13;
  torch::Tensor processed_w2;
  torch::Tensor processed_w13_scale;
  torch::Tensor processed_w2_scale;
  std::optional<torch::Tensor> processed_w13_scale_bias;
  std::optional<torch::Tensor> processed_w2_scale_bias;
  std::tie(processed_w13,
           processed_w2,
           processed_w13_scale,
           processed_w2_scale,
           processed_w13_scale_bias,
           processed_w2_scale_bias) =
      xllm::kernel::w4a8_dynamic_moe_preprocess(params);

  w13_.set_data(processed_w13);
  w2_.set_data(processed_w2);
  w13_scale_.set_data(processed_w13_scale);
  w2_scale_.set_data(processed_w2_scale);
  if (processed_w13_scale_bias.has_value() &&
      processed_w13_scale_bias->defined()) {
    w13_scale_bias_.set_data(processed_w13_scale_bias.value());
    w13_scale_bias_is_loaded_ = true;
  }
  if (processed_w2_scale_bias.has_value() &&
      processed_w2_scale_bias->defined()) {
    w2_scale_bias_.set_data(processed_w2_scale_bias.value());
    w2_scale_bias_is_loaded_ = true;
  }
  w4a8_dynamic_preprocessed_ = true;
}

void FusedMoEImpl::load_experts(const StateDict& state_dict) {
  LOG(INFO) << "[MOE_LOAD_DEBUG][FusedMoE.load_experts] prefix="
            << state_dict.prefix() << ", expected_w13=" << w13_.sizes()
            << ", expected_w2=" << w2_.sizes()
            << ", tp_group(rank/world)=" << tp_pg_->rank() << "/"
            << tp_pg_->world_size() << ", start_expert_id=" << start_expert_id_
            << ", num_experts_per_rank=" << num_experts_per_rank_;
  const int64_t rank = tp_pg_->rank();
  const int64_t world_size = tp_pg_->world_size();
  const int64_t start_expert_id = start_expert_id_;
  const int64_t num_experts_per_rank = num_experts_per_rank_;
  std::vector<std::string> prefixes = {"gate_proj.", "up_proj."};
  LOAD_MOE_FUSED_WEIGHT("weight", w1, w3, w13);
  if (!w13_is_loaded_) {
    prefixes = {"w1.", "w3."};
    LOAD_MOE_FUSED_WEIGHT("weight", w1, w3, w13);
  }
  LOAD_MOE_WEIGHT("down_proj.", "weight", w2, 1);
  if (!w2_is_loaded_) {
    LOAD_MOE_WEIGHT("w2.", "weight", w2, 1);
  }
  // Some Qwen3.5-MoE checkpoints store expert weights in fused tensors
  // (gate_up_proj / down_proj). Fall back to this format when split
  // gate_proj/up_proj tensors are absent.
  if (!w13_is_loaded_) {
    w13_is_loaded_ = load_fused_gate_up_fallback(state_dict,
                                                 rank,
                                                 world_size,
                                                 start_expert_id,
                                                 num_experts_per_rank,
                                                 w13_);
  }
  if (!w2_is_loaded_) {
    w2_is_loaded_ = load_fused_down_fallback(state_dict,
                                             rank,
                                             world_size,
                                             start_expert_id,
                                             num_experts_per_rank,
                                             w2_);
  }

  if (is_w8a8_dynamic_quant_method(resolved_moe_quant_method_)) {
    prefixes = {"gate_proj.", "up_proj."};
    LOAD_MOE_FUSED_WEIGHT("weight_scale", w1_scale, w3_scale, w13_scale);
    if (!w13_scale_is_loaded_) {
      prefixes = {"w1.", "w3."};
      LOAD_MOE_FUSED_WEIGHT("weight_scale", w1_scale, w3_scale, w13_scale);
    }
    LOAD_MOE_WEIGHT("down_proj.", "weight_scale", w2_scale, -1);
    if (!w2_scale_is_loaded_) {
      LOAD_MOE_WEIGHT("w2.", "weight_scale", w2_scale, -1);
    }
    if (!w13_scale_is_loaded_) {
      w13_scale_is_loaded_ = load_fused_up_scale_fallback(state_dict,
                                                          rank,
                                                          world_size,
                                                          start_expert_id,
                                                          num_experts_per_rank,
                                                          w13_scale_);
    }
    if (!w2_scale_is_loaded_) {
      w2_scale_is_loaded_ = load_fused_down_scale_fallback(
          state_dict, start_expert_id, num_experts_per_rank, w2_scale_);
    }
  }

  if (is_w4a8_dynamic_quant_method(resolved_moe_quant_method_)) {
    prefixes = {"gate_proj.", "up_proj."};
    LOAD_MOE_FUSED_WEIGHT("weight_scale", w1_scale, w3_scale, w13_scale);
    if (!w13_scale_is_loaded_) {
      prefixes = {"w1.", "w3."};
      LOAD_MOE_FUSED_WEIGHT("weight_scale", w1_scale, w3_scale, w13_scale);
    }
    LOAD_MOE_WEIGHT("down_proj.", "weight_scale", w2_scale, -1);
    if (!w2_scale_is_loaded_) {
      LOAD_MOE_WEIGHT("w2.", "weight_scale", w2_scale, -1);
    }

    if (quant_args_.group_size() > 0) {
      prefixes = {"gate_proj.", "up_proj."};
      LOAD_MOE_FUSED_WEIGHT("weight_scale_second",
                            w1_scale_second,
                            w3_scale_second,
                            w13_scale_second);
      if (!w13_scale_second_is_loaded_) {
        prefixes = {"w1.", "w3."};
        LOAD_MOE_FUSED_WEIGHT("weight_scale_second",
                              w1_scale_second,
                              w3_scale_second,
                              w13_scale_second);
      }
      LOAD_MOE_WEIGHT("down_proj.", "weight_scale_second", w2_scale_second, 1);
      if (!w2_scale_second_is_loaded_) {
        LOAD_MOE_WEIGHT("w2.", "weight_scale_second", w2_scale_second, 1);
      }
    }

    prefixes = {"gate_proj.", "up_proj."};
    LOAD_MOE_FUSED_WEIGHT(
        "scale_bias", w1_scale_bias, w3_scale_bias, w13_scale_bias);
    if (!w13_scale_bias_is_loaded_) {
      prefixes = {"w1.", "w3."};
      LOAD_MOE_FUSED_WEIGHT(
          "scale_bias", w1_scale_bias, w3_scale_bias, w13_scale_bias);
    }
    LOAD_MOE_WEIGHT("down_proj.", "scale_bias", w2_scale_bias, 1);
    if (!w2_scale_bias_is_loaded_) {
      LOAD_MOE_WEIGHT("w2.", "scale_bias", w2_scale_bias, 1);
    }
  }

  LOG(INFO) << "[QUANT_DEBUG][FusedMoE.load_experts] prefix="
            << state_dict.prefix() << ", quant_method="
            << (quant_args_.quant_method().empty() ? "<empty>"
                                                   : quant_args_.quant_method())
            << ", resolved_moe_quant_method="
            << resolved_moe_quant_method_.value_or("<none>")
            << ", is_w8a8_dynamic="
            << (is_w8a8_dynamic_quant_method(resolved_moe_quant_method_)
                    ? "true"
                    : "false")
            << ", is_w4a8_dynamic="
            << (is_w4a8_dynamic_quant_method(resolved_moe_quant_method_)
                    ? "true"
                    : "false")
            << ", w13_loaded=" << (w13_is_loaded_ ? "true" : "false")
            << ", w2_loaded=" << (w2_is_loaded_ ? "true" : "false") << ", w13={"
            << tensor_debug_info(w13_) << "}"
            << ", w2={" << tensor_debug_info(w2_) << "}";

  if (is_w8a8_dynamic_quant_method(resolved_moe_quant_method_)) {
    LOG(INFO) << "[QUANT_DEBUG][FusedMoE.load_experts][W8A8Dynamic] prefix="
              << state_dict.prefix() << ", w13_scale_loaded="
              << (w13_scale_is_loaded_ ? "true" : "false")
              << ", w2_scale_loaded="
              << (w2_scale_is_loaded_ ? "true" : "false") << ", w13_scale={"
              << tensor_debug_info(w13_scale_) << "}"
              << ", w2_scale={" << tensor_debug_info(w2_scale_) << "}";
  }
  if (is_w4a8_dynamic_quant_method(resolved_moe_quant_method_)) {
    LOG(INFO)
        << "[QUANT_DEBUG][FusedMoE.load_experts][W4A8Dynamic] prefix="
        << state_dict.prefix() << ", group_size=" << quant_args_.group_size()
        << ", quant_version="
        << (quant_args_.quant_version().empty() ? "<empty>"
                                                : quant_args_.quant_version())
        << ", w13_scale_loaded=" << (w13_scale_is_loaded_ ? "true" : "false")
        << ", w2_scale_loaded=" << (w2_scale_is_loaded_ ? "true" : "false")
        << ", w13_scale_second_loaded="
        << (w13_scale_second_is_loaded_ ? "true" : "false")
        << ", w2_scale_second_loaded="
        << (w2_scale_second_is_loaded_ ? "true" : "false")
        << ", w13_scale_bias_loaded="
        << (w13_scale_bias_is_loaded_ ? "true" : "false")
        << ", w2_scale_bias_loaded="
        << (w2_scale_bias_is_loaded_ ? "true" : "false");
  }
}

void FusedMoEImpl::load_state_dict(const StateDict& state_dict) {
  if (state_dict.size() == 0) {
    return;
  }
  resolve_quant_method_from_state_dict(state_dict);

  if (n_shared_experts_ > 0) {
    auto shared_expert_state =
        state_dict.get_dict_with_prefix("shared_expert.");
    if (shared_expert_state.size() == 0) {
      shared_expert_state = state_dict.get_dict_with_prefix("shared_experts.");
    }
    if (shared_expert_state.size() > 0) {
      if (shared_expert_state.get_tensor("w1.weight").defined() ||
          shared_expert_state.get_tensor("w1.qweight").defined()) {
        shared_experts_->load_state_dict(
            shared_expert_state, {"w1.", "w3."}, "w2.");
      } else {
        shared_experts_->load_state_dict(shared_expert_state);
      }
    }
    auto weight = state_dict.get_tensor("shared_expert_gate.weight");
    if (!weight.defined()) {
      weight = state_dict.get_tensor("shared_experts_gate.weight");
    }
    if (weight.defined()) {
      weight = weight.reshape({weight.size(0), -1});
      DCHECK_EQ(shared_expert_gate_->weight.sizes(), weight.sizes())
          << "proj weight size mismatch for " << name();
      shared_expert_gate_->weight.data().copy_(weight);
      shared_expert_gate_is_loaded_ = true;
    }
  }

  // Skip internal gate loading when an external gate module (e.g.
  // DeepseekV4Gate) already handles expert routing.  In that path the decoder
  // layer calls forward_with_selected_experts(), so this internal gate_ is
  // never executed and loading its weights would be redundant.
  if (!skip_gate_load_) {
    gate_->load_state_dict(state_dict.get_dict_with_prefix("gate."));
    load_e_score_correction_bias(state_dict.get_dict_with_prefix("gate."));
  }
  load_experts(state_dict.get_dict_with_prefix("experts."));
  preprocess_w4a8_dynamic_weights();
}

}  // namespace layer
}  // namespace xllm

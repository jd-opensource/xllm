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

#include "glm4_moe_decoder_loader.h"

#include <glog/logging.h>
#include <torch/torch.h>
#include <torch_npu/csrc/core/npu/NPUCachingAllocator.h>
#include <torch_npu/csrc/core/npu/NPUException.h>

#include <map>
#include <unordered_map>

#include "core/layers/npu/npu_glm4_moe_decoder_layer.h"

namespace xllm {
namespace layer {

enum DecoderLayerTensorId : int {
  IN_INPUT_NORM_WEIGHT = 0,
  IN_INPUT_NORM_BIAS = 1,
  IN_INPUT_NORM_NEW_WEIGHT = 2,
  IN_INPUT_NORM_NEW_BIAS = 3,

  IN_QKV_WEIGHT_0 = 4,
  IN_QKV_BIAS_0 = 5,
  IN_QKV_DESCALE_0 = 6,
  IN_QKV_OFFSET_0 = 7,
  IN_QKV_SCALE_0 = 8,
  IN_QKV_COMPRESS_IDX_0 = 9,

  IN_QKV_WEIGHT_1 = 10,
  IN_QKV_BIAS_1 = 11,
  IN_QKV_DESCALE_1 = 12,
  IN_QKV_OFFSET_1 = 13,
  IN_QKV_SCALE_1 = 14,
  IN_QKV_COMPRESS_IDX_1 = 15,

  IN_QKV_WEIGHT_2 = 16,
  IN_QKV_BIAS_2 = 17,
  IN_QKV_DESCALE_2 = 18,
  IN_QKV_OFFSET_2 = 19,
  IN_QKV_SCALE_2 = 20,
  IN_QKV_COMPRESS_IDX_2 = 21,

  IN_QKV_DENSE_WEIGHT = 22,
  IN_QKV_DENSE_BIAS = 23,
  IN_QKV_DENSE_DESCALE = 24,
  IN_QKV_DENSE_OFFSET = 25,
  IN_QKV_DENSE_SCALE = 26,
  IN_QKV_DENSE_COMPRESS_IDX = 27,

  IN_POST_ATTN_NORM_WEIGHT = 28,
  IN_POST_ATTN_NORM_BIAS = 29,
  IN_POST_ATTN_NORM_NEW_WEIGHT = 30,
  IN_POST_ATTN_NORM_NEW_BIAS = 31,

  IN_MLP_GATEUP_WEIGHT_SHARED_EXPERT = 32,
  IN_MLP_GATEUP_BIAS_SHARED_EXPERT = 33,
  IN_MLP_GATEUP_DESCALE_SHARED_EXPERT = 34,
  IN_MLP_GATEUP_OFFSET_SHARED_EXPERT = 35,
  IN_MLP_GATEUP_SCALE_SHARED_EXPERT = 36,
  IN_MLP_GATEUP_COMPRESS_IDX_SHARED_EXPERT = 37,

  IN_MLP_DOWN_WEIGHT_SHARED_EXPERT = 38,
  IN_MLP_DOWN_BIAS_SHARED_EXPERT = 39,
  IN_MLP_DOWN_DESCALE_SHARED_EXPERT = 40,
  IN_MLP_DOWN_OFFSET_SHARED_EXPERT = 41,
  IN_MLP_DOWN_SCALE_SHARED_EXPERT = 42,
  IN_MLP_DOWN_COMPRESS_IDX_SHARED_EXPERT = 43,

  IN_SHARED_EXPERT_GATE_WEIGHT = 44,
  IN_SHARED_EXPERT_GATE_BIAS = 45,
  IN_SHARED_EXPERT_GATE_DESCALE = 46,
  IN_SHARED_EXPERT_GATE_OFFSET = 47,
  IN_SHARED_EXPERT_GATE_SCALE = 48,
  IN_SHARED_EXPERT_GATE_COMPRESS_IDX = 49,

  BLOCK_SPARSE_MOE_GATE_WEIGHT = 50,
  BLOCK_SPARSE_MOE_GATE_BIAS = 51,
  BLOCK_SPARSE_MOE_GATE_DESCALE = 52,
  BLOCK_SPARSE_MOE_GATE_OFFSET = 53,
  BLOCK_SPARSE_MOE_GATE_SCALE = 54,
  BLOCK_SPARSE_MOE_GATE_COMPRESS_IDX = 55,

  IN_MLP_GATEUP_WEIGHT = 56,
  IN_MLP_GATEUP_BIAS = 57,
  IN_MLP_GATEUP_DESCALE = 58,
  IN_MLP_GATEUP_OFFSET = 59,
  IN_MLP_GATEUP_SCALE = 60,
  IN_MLP_GATEUP_COMPRESS_IDX = 61,

  IN_MLP_DOWN_WEIGHT = 62,
  IN_MLP_DOWN_BIAS = 63,
  IN_MLP_DOWN_DESCALE = 64,
  IN_MLP_DOWN_OFFSET = 65,
  IN_MLP_DOWN_SCALE = 66,
  IN_MLP_DOWN_COMPRESS_IDX = 67,

  Q_NORM_WEIGHT = 68,
  K_NORM_WEIGHT = 69
};

static std::unordered_map<std::string, int> WEIGHT_MAPPING = {
    {"input_layernorm.weight", IN_INPUT_NORM_WEIGHT},

    {"self_attn.q_proj.weight", IN_QKV_WEIGHT_0},
    {"self_attn.q_proj.bias", IN_QKV_BIAS_0},

    {"self_attn.k_proj.weight", IN_QKV_WEIGHT_1},
    {"self_attn.k_proj.bias", IN_QKV_BIAS_1},

    {"self_attn.v_proj.weight", IN_QKV_WEIGHT_2},
    {"self_attn.v_proj.bias", IN_QKV_BIAS_2},

    {"self_attn.o_proj.weight", IN_QKV_DENSE_WEIGHT},

    {"post_attention_layernorm.weight", IN_POST_ATTN_NORM_WEIGHT},

    // mlp or shared expert
    {"mlp.gate_proj.weight", IN_MLP_GATEUP_WEIGHT_SHARED_EXPERT},

    {"mlp.up_proj.weight", IN_MLP_GATEUP_WEIGHT_SHARED_EXPERT},

    {"mlp.down_proj.weight", IN_MLP_DOWN_WEIGHT_SHARED_EXPERT},

    {"mlp.shared_experts.gate_proj.weight", IN_MLP_GATEUP_WEIGHT_SHARED_EXPERT},

    {"mlp.shared_experts.up_proj.weight", IN_MLP_GATEUP_WEIGHT_SHARED_EXPERT},

    {"mlp.shared_experts.down_proj.weight", IN_MLP_DOWN_WEIGHT_SHARED_EXPERT},

    // MoE Gate
    {"mlp.gate.weight", BLOCK_SPARSE_MOE_GATE_WEIGHT},
    {"mlp.gate.e_score_correction_bias", BLOCK_SPARSE_MOE_GATE_BIAS},

    // Expert MLP - Gate/Up projections
    {"gate_proj.weight", IN_MLP_GATEUP_WEIGHT},
    {"up_proj.weight", IN_MLP_GATEUP_WEIGHT},

    // Expert MLP - Down projection
    {"down_proj.weight", IN_MLP_DOWN_WEIGHT},

};

static std::unordered_map<std::string, int> WEIGHT_MAPPING_W8A8 = {
    {"input_layernorm.weight", IN_INPUT_NORM_WEIGHT},
    {"input_layernorm.bias", IN_INPUT_NORM_NEW_BIAS},

    {"self_attn.q_proj.weight", IN_QKV_WEIGHT_0},
    {"self_attn.q_proj.deq_scale", IN_QKV_DESCALE_0},
    {"self_attn.q_proj.quant_bias", IN_QKV_BIAS_0},
    {"self_attn.q_proj.input_offset", IN_QKV_OFFSET_0},
    {"self_attn.q_proj.input_scale", IN_QKV_SCALE_0},

    {"self_attn.k_proj.weight", IN_QKV_WEIGHT_1},
    {"self_attn.k_proj.deq_scale", IN_QKV_DESCALE_1},
    {"self_attn.k_proj.quant_bias", IN_QKV_BIAS_1},

    {"self_attn.v_proj.weight", IN_QKV_WEIGHT_2},
    {"self_attn.v_proj.deq_scale", IN_QKV_DESCALE_2},
    {"self_attn.v_proj.quant_bias", IN_QKV_BIAS_2},

    {"self_attn.o_proj.weight", IN_QKV_DENSE_WEIGHT},
    {"self_attn.o_proj.quant_bias", IN_QKV_DENSE_BIAS},
    {"self_attn.o_proj.deq_scale", IN_QKV_DENSE_DESCALE},
    {"self_attn.o_proj.weight_offset", IN_QKV_DENSE_OFFSET},
    {"self_attn.o_proj.weight_scale", IN_QKV_DENSE_SCALE},

    {"post_attention_layernorm.weight", IN_POST_ATTN_NORM_WEIGHT},
    {"post_attention_layernorm.bias", IN_POST_ATTN_NORM_NEW_BIAS},

    // mlp
    {"mlp.gate_proj.weight", IN_MLP_GATEUP_WEIGHT_SHARED_EXPERT},
    {"mlp.gate_proj.weight_offset", IN_MLP_GATEUP_OFFSET_SHARED_EXPERT},
    {"mlp.gate_proj.weight_scale", IN_MLP_GATEUP_SCALE_SHARED_EXPERT},

    {"mlp.up_proj.weight", IN_MLP_GATEUP_WEIGHT_SHARED_EXPERT},
    {"mlp.up_proj.weight_offset", IN_MLP_GATEUP_OFFSET_SHARED_EXPERT},
    {"mlp.up_proj.weight_scale", IN_MLP_GATEUP_SCALE_SHARED_EXPERT},

    {"mlp.down_proj.weight", IN_MLP_DOWN_WEIGHT_SHARED_EXPERT},
    {"mlp.down_proj.weight_offset", IN_MLP_DOWN_OFFSET_SHARED_EXPERT},
    {"mlp.down_proj.weight_scale", IN_MLP_DOWN_SCALE_SHARED_EXPERT},

    // shared expert
    {"mlp.shared_experts.gate_proj.weight", IN_MLP_GATEUP_WEIGHT_SHARED_EXPERT},
    {"mlp.shared_experts.gate_proj.weight_offset",
     IN_MLP_GATEUP_OFFSET_SHARED_EXPERT},
    {"mlp.shared_experts.gate_proj.weight_scale",
     IN_MLP_GATEUP_SCALE_SHARED_EXPERT},

    {"mlp.shared_experts.up_proj.weight", IN_MLP_GATEUP_WEIGHT_SHARED_EXPERT},
    {"mlp.shared_experts.up_proj.weight_offset",
     IN_MLP_GATEUP_OFFSET_SHARED_EXPERT},
    {"mlp.shared_experts.up_proj.weight_scale",
     IN_MLP_GATEUP_SCALE_SHARED_EXPERT},

    {"mlp.shared_experts.down_proj.weight", IN_MLP_DOWN_WEIGHT_SHARED_EXPERT},
    {"mlp.shared_experts.down_proj.weight_offset",
     IN_MLP_DOWN_OFFSET_SHARED_EXPERT},
    {"mlp.shared_experts.down_proj.weight_scale",
     IN_MLP_DOWN_SCALE_SHARED_EXPERT},

    // MoE Gate
    {"mlp.gate.weight", BLOCK_SPARSE_MOE_GATE_WEIGHT},
    {"mlp.gate.e_score_correction_bias", BLOCK_SPARSE_MOE_GATE_BIAS},

    {"gate_proj.weight", IN_MLP_GATEUP_WEIGHT},
    {"gate_proj.weight_offset", IN_MLP_GATEUP_OFFSET},
    {"gate_proj.weight_scale", IN_MLP_GATEUP_SCALE},
    {"up_proj.weight", IN_MLP_GATEUP_WEIGHT},
    {"up_proj.weight_offset", IN_MLP_GATEUP_OFFSET},
    {"up_proj.weight_scale", IN_MLP_GATEUP_SCALE},

    {"down_proj.weight", IN_MLP_DOWN_WEIGHT},
    {"down_proj.weight_offset", IN_MLP_DOWN_OFFSET},
    {"down_proj.weight_scale", IN_MLP_DOWN_SCALE},
};

static const std::unordered_map<std::string, std::vector<int>>
    SPECIAL_MULTI_ASSIGN_W8A8 = {
        {"input_layernorm.weight",
         {IN_INPUT_NORM_WEIGHT, IN_INPUT_NORM_NEW_WEIGHT}},
        {"post_attention_layernorm.weight",
         {IN_POST_ATTN_NORM_WEIGHT, IN_POST_ATTN_NORM_NEW_WEIGHT}},
};

static const std::map<int, int> WEIGHT_SHARD = {
    {IN_QKV_WEIGHT_0, 0},
    {IN_QKV_BIAS_0, 0},
    {IN_QKV_WEIGHT_1, 0},
    {IN_QKV_BIAS_1, 0},
    {IN_QKV_WEIGHT_2, 0},
    {IN_QKV_BIAS_2, 0},
    {IN_QKV_DENSE_WEIGHT, 1},
    {IN_MLP_GATEUP_WEIGHT_SHARED_EXPERT, 0},
    {IN_MLP_DOWN_WEIGHT_SHARED_EXPERT, 1},
    {IN_MLP_GATEUP_WEIGHT, 0},
    {IN_MLP_DOWN_WEIGHT, 1},
};

static const std::map<int, int> WEIGHT_SHARD_W8A8 = {
    {IN_QKV_WEIGHT_0, 0},
    {IN_QKV_BIAS_0, 0},
    {IN_QKV_DESCALE_0, 0},
    {IN_QKV_WEIGHT_1, 0},
    {IN_QKV_BIAS_1, 0},
    {IN_QKV_DESCALE_1, 0},
    {IN_QKV_WEIGHT_2, 0},
    {IN_QKV_BIAS_2, 0},
    {IN_QKV_DESCALE_2, 0},
    {IN_QKV_DENSE_WEIGHT, 1},
    {IN_MLP_GATEUP_WEIGHT_SHARED_EXPERT, 0},
    {IN_MLP_GATEUP_OFFSET_SHARED_EXPERT, 0},
    {IN_MLP_GATEUP_SCALE_SHARED_EXPERT, 0},
    {IN_MLP_DOWN_WEIGHT_SHARED_EXPERT, 1},
    {IN_MLP_GATEUP_WEIGHT, 0},
    {IN_MLP_GATEUP_OFFSET, 0},
    {IN_MLP_GATEUP_SCALE, 0},
    {IN_MLP_DOWN_WEIGHT, 1},
};

Glm4MoeDecoderLoader::Glm4MoeDecoderLoader(
    uint64_t weight_count,
    const ModelContext& context,
    int32_t layer_id,
    int32_t prefill_param_firstKDenseReplace)
    : BaseLoader(weight_count, context),
      layer_id_(layer_id),
      prefill_param_firstKDenseReplace_(prefill_param_firstKDenseReplace) {
  auto model_args = context.get_model_args();
  auto parallel_args = context.get_parallel_args();
  auto options = context.get_tensor_options();

  tensor_placeholder_ = torch::zeros({1}).to(options);

  if (model_args.use_qk_norm()) {
    weight_count_ = weight_count = 70;
    WEIGHT_MAPPING_W8A8["self_attn.q_norm.weight"] = Q_NORM_WEIGHT;
    WEIGHT_MAPPING_W8A8["self_attn.k_norm.weight"] = K_NORM_WEIGHT;
    WEIGHT_MAPPING["self_attn.q_norm.weight"] = Q_NORM_WEIGHT;
    WEIGHT_MAPPING["self_attn.k_norm.weight"] = K_NORM_WEIGHT;
  }

  at_weight_tensors_.resize(weight_count_);

  num_experts_ = model_args.num_experts();
  ep_size_ = parallel_args.ep_size();
  ep_local_tp_size_ = parallel_args.world_size() / ep_size_;
  CHECK_EQ(parallel_args.world_size(), ep_size_ * ep_local_tp_size_);
  ep_local_tp_rank_ = parallel_args.rank() % ep_local_tp_size_;
  num_experts_per_partition_ = model_args.num_experts() / ep_size_;
  ep_rank_ = parallel_args.rank() / ep_local_tp_size_;
  start_expert_id_ = ep_rank_ * num_experts_per_partition_;
  end_expert_id_ = start_expert_id_ + num_experts_per_partition_ - 1;

  dp_size_ = parallel_args.dp_size();
  dp_local_tp_size_ = parallel_args.world_size() / dp_size_;
  CHECK_EQ(parallel_args.world_size(), dp_size_ * dp_local_tp_size_);
  dp_local_tp_rank_ = parallel_args.rank() % dp_local_tp_size_;

  n_kv_heads_ = static_cast<int32_t>(model_args.n_kv_heads().value());
}

void Glm4MoeDecoderLoader::resize_experts_weights(int num_of_device_experts) {
  experts_weights_["gate_proj.weight"] =
      std::vector<torch::Tensor>(num_of_device_experts);
  experts_weights_["up_proj.weight"] =
      std::vector<torch::Tensor>(num_of_device_experts);
  experts_weights_["down_proj.weight"] =
      std::vector<torch::Tensor>(num_of_device_experts);
  if (quantize_type_.compare("w8a8_dynamic") == 0) {
    experts_weights_["gate_proj.weight_offset"] =
        std::vector<torch::Tensor>(num_of_device_experts);
    experts_weights_["up_proj.weight_offset"] =
        std::vector<torch::Tensor>(num_of_device_experts);
    experts_weights_["down_proj.weight_offset"] =
        std::vector<torch::Tensor>(num_of_device_experts);
    experts_weights_["gate_proj.weight_scale"] =
        std::vector<torch::Tensor>(num_of_device_experts);
    experts_weights_["up_proj.weight_scale"] =
        std::vector<torch::Tensor>(num_of_device_experts);
    experts_weights_["down_proj.weight_scale"] =
        std::vector<torch::Tensor>(num_of_device_experts);
  }
}

void Glm4MoeDecoderLoader::load_state_dict(const StateDict& state_dict) {
  for (const auto& [name, tensor] : state_dict) {
    bool is_sharded = false;
    int index = 0;

    if (absl::StartsWith(name, "mlp.experts")) {
      process_expert_weights(state_dict, name, tensor);
      continue;
    }
    if (absl::StartsWith(name, "mlp.shared_experts")) {
      process_shared_expert_weights(state_dict, name, tensor);
      continue;
    }
    if (absl::StartsWith(name, "mlp") && !absl::StrContains(name, "gate.")) {
      process_mlp_common_weights(state_dict, name, tensor);
      continue;
    }

    process_general_weights(state_dict, name, tensor);
  }
}

void Glm4MoeDecoderLoader::verify_loaded_weights() const {
  for (const auto& [name, index] : WEIGHT_MAPPING) {
    if (name == "down_proj.weight" || name == "gate_proj.weight" ||
        name == "up_proj.weight" || name == "mlp.gate.weight" ||
        name == "mlp.gate.e_score_correction_bias") {
      continue;
    }
    CHECK(at_weight_tensors_[index].sizes() != std::vector<int64_t>({0}))
        << layer_id_ << "-weight is not loaded for " << name;
  }
}

void Glm4MoeDecoderLoader::merge_loaded_weights() {
  merge_shared_experts_weights();
  if (layer_id_ >= prefill_param_firstKDenseReplace_) {
    merge_experts_weights();
  }
  at_weight_tensors_[IN_QKV_WEIGHT_0] =
      torch::cat({at_weight_tensors_[IN_QKV_WEIGHT_0],
                  at_weight_tensors_[IN_QKV_WEIGHT_1],
                  at_weight_tensors_[IN_QKV_WEIGHT_2]},
                 0)
          .contiguous();
  at_weight_tensors_[IN_QKV_WEIGHT_1] =
      torch::zeros({1}, torch::kFloat16).to(device_);
  at_weight_tensors_[IN_QKV_WEIGHT_2] =
      torch::zeros({1}, torch::kFloat16).to(device_);

  at_weight_tensors_[IN_QKV_BIAS_0] =
      at_weight_tensors_[IN_QKV_BIAS_0].squeeze();
  at_weight_tensors_[IN_QKV_BIAS_1] =
      at_weight_tensors_[IN_QKV_BIAS_1].squeeze();
  at_weight_tensors_[IN_QKV_BIAS_2] =
      at_weight_tensors_[IN_QKV_BIAS_2].squeeze();

  at_weight_tensors_[IN_QKV_BIAS_0] =
      torch::cat({at_weight_tensors_[IN_QKV_BIAS_0],
                  at_weight_tensors_[IN_QKV_BIAS_1],
                  at_weight_tensors_[IN_QKV_BIAS_2]},
                 0)
          .contiguous();
  at_weight_tensors_[IN_QKV_BIAS_1] =
      torch::zeros({1}, torch::kFloat16).to(device_);
  at_weight_tensors_[IN_QKV_BIAS_2] =
      torch::zeros({1}, torch::kFloat16).to(device_);

  if (quantize_type_.compare("w8a8_dynamic") == 0) {
    at_weight_tensors_[IN_QKV_DESCALE_0] =
        at_weight_tensors_[IN_QKV_DESCALE_0].squeeze();
    at_weight_tensors_[IN_QKV_DESCALE_1] =
        at_weight_tensors_[IN_QKV_DESCALE_1].squeeze();
    at_weight_tensors_[IN_QKV_DESCALE_2] =
        at_weight_tensors_[IN_QKV_DESCALE_2].squeeze();

    at_weight_tensors_[IN_QKV_DESCALE_0] =
        torch::cat({at_weight_tensors_[IN_QKV_DESCALE_0],
                    at_weight_tensors_[IN_QKV_DESCALE_1],
                    at_weight_tensors_[IN_QKV_DESCALE_2]},
                   0)
            .contiguous();

    at_weight_tensors_[IN_QKV_DESCALE_1] =
        torch::zeros({1}, torch::kFloat16).to(device_);
    at_weight_tensors_[IN_QKV_DESCALE_2] =
        torch::zeros({1}, torch::kFloat16).to(device_);

    at_weight_tensors_[IN_QKV_DENSE_BIAS] =
        torch::zeros({1}, torch::kFloat16).to(device_);
    at_weight_tensors_[IN_QKV_DENSE_DESCALE] =
        torch::zeros({1}, torch::kFloat16).to(device_);

    at_weight_tensors_[IN_QKV_OFFSET_0] =
        at_weight_tensors_[IN_QKV_OFFSET_0].to(torch::kInt8).to(device_);
    at_weight_tensors_[IN_QKV_OFFSET_1] =
        torch::zeros({1}, torch::kFloat16).to(device_);
    at_weight_tensors_[IN_QKV_OFFSET_2] =
        torch::zeros({1}, torch::kFloat16).to(device_);
    at_weight_tensors_[IN_QKV_DENSE_OFFSET] =
        at_weight_tensors_[IN_QKV_DENSE_OFFSET].contiguous().view(-1);

    at_weight_tensors_[IN_QKV_SCALE_1] =
        torch::zeros({1}, torch::kFloat16).to(device_);
    at_weight_tensors_[IN_QKV_SCALE_2] =
        torch::zeros({1}, torch::kFloat16).to(device_);
    at_weight_tensors_[IN_QKV_DENSE_SCALE] =
        at_weight_tensors_[IN_QKV_DENSE_SCALE].contiguous().view(-1);
  }
}

void Glm4MoeDecoderLoader::process_expert_weights(const StateDict& state_dict,
                                                  const std::string& name,
                                                  const torch::Tensor& tensor) {
  int expert_index = extract_expert_index(name);
  if (expert_index < start_expert_id_ || expert_index > end_expert_id_) {
    return;
  }

  const std::string suffix = extract_endswith(name);
  const auto& weight_mapping = (quantize_type_.compare("w8a8_dynamic") == 0)
                                   ? WEIGHT_MAPPING_W8A8
                                   : WEIGHT_MAPPING;
  const auto& shard_map = (quantize_type_.compare("w8a8_dynamic") == 0)
                              ? WEIGHT_SHARD_W8A8
                              : WEIGHT_SHARD;
  const int index = get_mapped_index(suffix, weight_mapping);
  const int local_index = expert_index % num_experts_per_partition_;
  const bool is_sharded = shard_map.count(index);

  std::lock_guard<std::mutex> lock(experts_mutex_);
  torch::Tensor tmp_tensor = is_sharded
                                 ? get_sharded_tensor(state_dict,
                                                      name,
                                                      shard_map.at(index),
                                                      ep_local_tp_rank_,
                                                      ep_local_tp_size_)
                                 : tensor;

  experts_weights_[suffix][local_index] = tmp_tensor.clone();
}

void Glm4MoeDecoderLoader::process_shared_expert_weights(
    const StateDict& state_dict,
    const std::string& name,
    const torch::Tensor& tensor) {
  torch::Tensor tmp_tensor;
  const auto& weight_mapping = (quantize_type_.compare("w8a8_dynamic") == 0)
                                   ? WEIGHT_MAPPING_W8A8
                                   : WEIGHT_MAPPING;
  const auto& shard_map = (quantize_type_.compare("w8a8_dynamic") == 0)
                              ? WEIGHT_SHARD_W8A8
                              : WEIGHT_SHARD;
  std::lock_guard<std::mutex> lock(shared_experts_mutex_);
  const int index = get_mapped_index(name, weight_mapping);
  if (index == -1) {
    return;
  }

  const bool is_sharded = shard_map.count(index);
  tmp_tensor = is_sharded
                   ? get_sharded_tensor(state_dict, name, shard_map.at(index))
                         .to(device_)
                   : tensor.to(device_);

  if (absl::StrContains(name, "down_proj")) {
    at_weight_tensors_[index] = tmp_tensor;
  } else {
    shared_experts_weights_[name] = tmp_tensor;
  }
}

void Glm4MoeDecoderLoader::process_mlp_common_weights(
    const StateDict& state_dict,
    const std::string& name,
    const torch::Tensor& tensor) {
  const auto& weight_mapping = (quantize_type_.compare("w8a8_dynamic") == 0)
                                   ? WEIGHT_MAPPING_W8A8
                                   : WEIGHT_MAPPING;
  const auto& shard_map = (quantize_type_.compare("w8a8_dynamic") == 0)
                              ? WEIGHT_SHARD_W8A8
                              : WEIGHT_SHARD;
  const int index = get_mapped_index(name, weight_mapping);
  const bool is_sharded = shard_map.count(index);

  std::lock_guard<std::mutex> lock(shared_experts_mutex_);

  torch::Tensor tmp_tensor = is_sharded
                                 ? get_sharded_tensor(state_dict,
                                                      name,
                                                      shard_map.at(index),
                                                      dp_local_tp_rank_,
                                                      dp_local_tp_size_)
                                       .to(device_)
                                 : tensor.to(device_);
  if (absl::StrContains(name, "down_proj")) {
    at_weight_tensors_[index] = tmp_tensor;
  } else {
    shared_experts_weights_[name] = tmp_tensor;
  }
}

void Glm4MoeDecoderLoader::process_general_weights(
    const StateDict& state_dict,
    const std::string& name,
    const torch::Tensor& tensor) {
  const auto& weight_mapping = (quantize_type_.compare("w8a8_dynamic") == 0)
                                   ? WEIGHT_MAPPING_W8A8
                                   : WEIGHT_MAPPING;
  const auto& shard_map = (quantize_type_.compare("w8a8_dynamic") == 0)
                              ? WEIGHT_SHARD_W8A8
                              : WEIGHT_SHARD;

  if (weight_mapping.find(name) == weight_mapping.end()) {
    return;
  }

  const int index = get_mapped_index(name, weight_mapping);
  const bool is_sharded = shard_map.count(index);
  torch::Tensor tmp_tensor;
  int32_t tp_rank = dp_local_tp_rank_;
  int32_t tp_size = dp_local_tp_size_;
  if (index == IN_QKV_WEIGHT_1 || index == IN_QKV_WEIGHT_2 ||
      index == IN_QKV_BIAS_1 || index == IN_QKV_BIAS_2 ||
      index == IN_QKV_DESCALE_1 || index == IN_QKV_DESCALE_2) {
    if (n_kv_heads_ < dp_local_tp_size_) {
      int32_t repeat_times = (dp_local_tp_size_ / n_kv_heads_);
      tp_rank = tp_rank / repeat_times;
      tp_size = n_kv_heads_;
    }
  }
  if (is_sharded) {
    tmp_tensor = get_sharded_tensor(
                     state_dict, name, shard_map.at(index), tp_rank, tp_size)
                     .to(device_);
  } else {
    tmp_tensor = tensor.to(device_);
  }
  if (index == BLOCK_SPARSE_MOE_GATE_BIAS) {
    auto min_val = tmp_tensor.min();
    tmp_tensor = tmp_tensor - min_val;
  }
  correct_tensor_dtype(tmp_tensor, name);
  if (quantize_type_.compare("w8a8_dynamic") == 0) {
    auto it = SPECIAL_MULTI_ASSIGN_W8A8.find(name);
    if (it != SPECIAL_MULTI_ASSIGN_W8A8.end()) {
      for (int idx : it->second) {
        at_weight_tensors_[idx] = tmp_tensor;
      }
      return;
    }
  }
  at_weight_tensors_[index] = tmp_tensor;
}

torch::Tensor Glm4MoeDecoderLoader::get_sharded_tensor(
    const StateDict& state_dict,
    const std::string& name,
    int dim) {
  if (parallel_args_.world_size() > 1) {
    return state_dict.get_sharded_tensor(
        name, dim, parallel_args_.rank(), parallel_args_.world_size());
  } else {
    return state_dict.get_tensor(name);
  }
}

torch::Tensor Glm4MoeDecoderLoader::get_sharded_tensor(
    const StateDict& state_dict,
    const std::string& name,
    int dim,
    int loacal_tp_rank,
    int local_tp_size) {
  if (local_tp_size > 1) {
    return state_dict.get_sharded_tensor(
        name, dim, loacal_tp_rank, local_tp_size);
  } else {
    return state_dict.get_tensor(name);
  }
}

std::string Glm4MoeDecoderLoader::extract_endswith(const std::string& input) {
  std::vector<std::string> parts;
  std::stringstream ss(input);
  std::string part;
  while (std::getline(ss, part, '.')) {
    parts.push_back(part);
  }
  if (parts.size() < 2) {
    return "";
  }
  std::string result = parts[parts.size() - 2] + "." + parts[parts.size() - 1];

  return result;
}

int Glm4MoeDecoderLoader::extract_expert_index(const std::string& name) {
  std::string prefix = "experts.";
  size_t pos = name.find(prefix);
  if (pos != std::string::npos) {
    pos += prefix.length();
    size_t end_pos = pos;
    while (end_pos < name.length() && std::isdigit(name[end_pos])) {
      ++end_pos;
    }
    if (end_pos > pos) {
      return std::stoi(name.substr(pos, end_pos - pos));
    }
  }

  return -1;
}

void Glm4MoeDecoderLoader::merge_shared_experts_weights() {
  auto merge_and_clear = [this](int index,
                                torch::Tensor& shared_experts_gate,
                                torch::Tensor& shared_experts_up) {
    at_weight_tensors_[index] =
        torch::cat({shared_experts_gate, shared_experts_up}, 0)
            .to(device_)
            .contiguous();
    shared_experts_gate = tensor_placeholder_;
    shared_experts_up = tensor_placeholder_;
  };

  if (layer_id_ >= prefill_param_firstKDenseReplace_) {
    merge_and_clear(
        IN_MLP_GATEUP_WEIGHT_SHARED_EXPERT,
        shared_experts_weights_["mlp.shared_experts.gate_proj.weight"],
        shared_experts_weights_["mlp.shared_experts.up_proj.weight"]);
    if (quantize_type_ == "w8a8_dynamic") {
      merge_and_clear(
          IN_MLP_GATEUP_OFFSET_SHARED_EXPERT,
          shared_experts_weights_["mlp.shared_experts.gate_proj.weight_offset"],
          shared_experts_weights_["mlp.shared_experts.up_proj.weight_offset"]);
      merge_and_clear(
          IN_MLP_GATEUP_SCALE_SHARED_EXPERT,
          shared_experts_weights_["mlp.shared_experts.gate_proj.weight_scale"],
          shared_experts_weights_["mlp.shared_experts.up_proj.weight_scale"]);
      at_weight_tensors_[IN_MLP_GATEUP_OFFSET_SHARED_EXPERT] =
          at_weight_tensors_[IN_MLP_GATEUP_OFFSET_SHARED_EXPERT].squeeze();
      at_weight_tensors_[IN_MLP_GATEUP_SCALE_SHARED_EXPERT] =
          at_weight_tensors_[IN_MLP_GATEUP_SCALE_SHARED_EXPERT].squeeze();
      at_weight_tensors_[IN_MLP_DOWN_OFFSET_SHARED_EXPERT] =
          at_weight_tensors_[IN_MLP_DOWN_OFFSET_SHARED_EXPERT].squeeze();
      at_weight_tensors_[IN_MLP_DOWN_SCALE_SHARED_EXPERT] =
          at_weight_tensors_[IN_MLP_DOWN_SCALE_SHARED_EXPERT].squeeze();
    }
  } else {
    merge_and_clear(IN_MLP_GATEUP_WEIGHT_SHARED_EXPERT,
                    shared_experts_weights_["mlp.gate_proj.weight"],
                    shared_experts_weights_["mlp.up_proj.weight"]);
    if (quantize_type_ == "w8a8_dynamic") {
      merge_and_clear(IN_MLP_GATEUP_OFFSET_SHARED_EXPERT,
                      shared_experts_weights_["mlp.gate_proj.weight_offset"],
                      shared_experts_weights_["mlp.up_proj.weight_offset"]);
      merge_and_clear(IN_MLP_GATEUP_SCALE_SHARED_EXPERT,
                      shared_experts_weights_["mlp.gate_proj.weight_scale"],
                      shared_experts_weights_["mlp.up_proj.weight_scale"]);
      at_weight_tensors_[IN_MLP_GATEUP_OFFSET_SHARED_EXPERT] =
          at_weight_tensors_[IN_MLP_GATEUP_OFFSET_SHARED_EXPERT].squeeze();
      at_weight_tensors_[IN_MLP_GATEUP_SCALE_SHARED_EXPERT] =
          at_weight_tensors_[IN_MLP_GATEUP_SCALE_SHARED_EXPERT].squeeze();
    }
  }
}

void Glm4MoeDecoderLoader::merge_experts_weights() {
  try {
    torch::Tensor mlp_gateup_weight;
    if (quantize_type_.compare("w8a8_dynamic") == 0) {
      mlp_gateup_weight =
          merge_experts_weights(experts_weights_["gate_proj.weight"],
                                experts_weights_["up_proj.weight"],
                                /*transpose=*/true);

      at_weight_tensors_[IN_MLP_GATEUP_OFFSET] =
          merge_experts_weights(experts_weights_["gate_proj.weight_offset"],
                                experts_weights_["up_proj.weight_offset"]);
      at_weight_tensors_[IN_MLP_GATEUP_SCALE] =
          merge_experts_weights(experts_weights_["gate_proj.weight_scale"],
                                experts_weights_["up_proj.weight_scale"]);
      at_weight_tensors_[IN_MLP_GATEUP_WEIGHT] =
          at_npu::native::npu_format_cast(mlp_gateup_weight, 29);
    } else {
      mlp_gateup_weight =
          merge_experts_weights(experts_weights_["gate_proj.weight"],
                                experts_weights_["up_proj.weight"],
                                /*transpose=*/false);
      at_weight_tensors_[IN_MLP_GATEUP_WEIGHT] =
          at_npu::native::npu_format_cast(mlp_gateup_weight, 2).contiguous();
    }
  } catch (const std::exception& e) {
    LOG(ERROR) << "[ERROR] Exception in gateup weight processing: " << e.what();
    throw;
  }

  if (experts_weights_.count("down_proj.weight") > 0) {
    auto& down_weight = experts_weights_["down_proj.weight"];
  }

  try {
    torch::Tensor mlp_down_weight =
        merge_experts_weights(experts_weights_["down_proj.weight"],
                              /*transpose=*/false);

    at_weight_tensors_[IN_MLP_DOWN_WEIGHT] =
        at_npu::native::npu_format_cast(mlp_down_weight, 2).contiguous();

    if (quantize_type_.compare("w8a8_dynamic") == 0) {
      at_weight_tensors_[IN_MLP_DOWN_OFFSET] =
          merge_experts_weights(experts_weights_["down_proj.weight_offset"]);
      at_weight_tensors_[IN_MLP_DOWN_SCALE] =
          merge_experts_weights(experts_weights_["down_proj.weight_scale"]);
    }
  } catch (const std::exception& e) {
    LOG(ERROR) << "[ERROR] Exception in down weight processing: " << e.what();
    throw;
  }
}

torch::Tensor Glm4MoeDecoderLoader::merge_experts_weights(
    std::vector<torch::Tensor>& experts,
    bool transpose) {
  torch::Tensor merged_tensor = torch::stack(experts, 0).to(device_);
  if (transpose) {
    merged_tensor = merged_tensor.transpose(1, 2);
  }
  merged_tensor = merged_tensor.contiguous();
  experts.clear();

  return merged_tensor;
}

torch::Tensor Glm4MoeDecoderLoader::merge_experts_weights(
    std::vector<torch::Tensor>& experts_gate,
    std::vector<torch::Tensor>& experts_up,
    bool transpose) {
  for (size_t i = 0; i < experts_up.size(); ++i) {
    experts_gate[i] = torch::cat({experts_gate[i], experts_up[i]}, 0);
  }
  torch::Tensor merged_tensor = torch::stack(experts_gate, 0).to(device_);
  if (transpose) {
    merged_tensor = merged_tensor.transpose(1, 2);
  }
  merged_tensor = merged_tensor.contiguous();
  experts_gate.clear();
  experts_up.clear();

  return merged_tensor;
}

int Glm4MoeDecoderLoader::get_mapped_index(
    const std::string& name,
    const std::unordered_map<std::string, int>& mapping) {
  const auto it = mapping.find(name);
  if (it == mapping.end()) {
    LOG(ERROR) << "Missing mapping for: " << name;
    return -1;
  }
  return it->second;
}

}  // namespace layer
}  // namespace xllm
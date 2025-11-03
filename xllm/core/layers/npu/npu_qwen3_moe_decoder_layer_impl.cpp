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

#include "npu_qwen3_moe_decoder_layer_impl.h"

#include <gflags/gflags.h>

#include "common/global_flags.h"

namespace xllm {
namespace layer {

enum DecoderLayerTensorId : int {
  IN_INPUT_NORM_WEIGHT = 0,  // [2048]
  IN_INPUT_NORM_BIAS = 1,
  IN_INPUT_NORM_NEW_WEIGHT = 2,
  IN_INPUT_NORM_NEW_BIAS = 3,

  IN_QKV_WEIGHT_0 = 4,  // [4096, 2048]
  IN_QKV_BIAS_0 = 5,
  IN_QKV_DESCALE_0 = 6,
  IN_QKV_OFFSET_0 = 7,
  IN_QKV_SCALE_0 = 8,
  IN_QKV_COMPRESS_IDX_0 = 9,

  IN_QKV_WEIGHT_1 = 10,  // [512, 2048]
  IN_QKV_BIAS_1 = 11,
  IN_QKV_DESCALE_1 = 12,
  IN_QKV_OFFSET_1 = 13,
  IN_QKV_SCALE_1 = 14,
  IN_QKV_COMPRESS_IDX_1 = 15,

  IN_QKV_WEIGHT_2 = 16,  // [512, 2048]
  IN_QKV_BIAS_2 = 17,
  IN_QKV_DESCALE_2 = 18,
  IN_QKV_OFFSET_2 = 19,
  IN_QKV_SCALE_2 = 20,
  IN_QKV_COMPRESS_IDX_2 = 21,

  IN_ATTENTION_OUT_WEIGHT = 22,  // [2048, 4096]
  IN_ATTENTION_OUT_BIAS = 23,
  IN_ATTENTION_OUT_DESCALE = 24,
  IN_ATTENTION_OUT_OFFSET = 25,
  IN_ATTENTION_OUT_SCALE = 26,
  IN_ATTENTION_OUT_COMPRESS_IDX = 27,

  IN_Q_NORM_WEIGHT = 28,  // [128]
  IN_K_NORM_WEIGHT = 29,  // [128]

  IN_SELFATTENTION_OUT_NORM_WEIGHT = 30,  // [2048]
  IN_SELFATTENTION_OUT_NORM_BIAS = 31,
  IN_SELFATTENTION_OUT_NEW_NORM_WEIGHT = 32,
  IN_SELFATTENTION_OUT_NEW_NORM_BIAS = 33,

  IN_BLOCK_SPARSE_MOE_GATE_WEIGHT = 34,  // [128, 2048]
  IN_BLOCK_SPARSE_MOE_GATE_BIAS = 35,
  IN_BLOCK_SPARSE_MOE_GATE_DESCALE = 36,
  IN_BLOCK_SPARSE_MOE_GATE_OFFSET = 37,
  IN_BLOCK_SPARSE_MOE_GATE_SCALE = 38,
  IN_BLOCK_SPARSE_MOE_GATE_COMPRESS_IDX = 39,

  IN_MLP_GATEUP_WEIGHT_EXPERT = 40,
  IN_MLP_GATEUP_BIAS_EXPERT = 41,
  IN_MLP_GATEUP_DESCALE_EXPERT = 42,
  IN_MLP_GATEUP_OFFSET_EXPERT = 43,
  IN_MLP_GATEUP_SCALE_EXPERT = 44,
  IN_MLP_GATEUP_COMPRESS_IDX_EXPERT = 45,

  IN_MLP_DOWN_WEIGHT_EXPERT = 46,  // [2048, 768]
  IN_MLP_DOWN_BIAS_EXPERT = 47,
  IN_MLP_DOWN_DESCALE_EXPERT = 48,
  IN_MLP_DOWN_OFFSET_EXPERT = 49,
  IN_MLP_DOWN_SCALE_EXPERT = 50,
  IN_MLP_DOWN_COMPRESS_IDX_EXPERT = 51,

  IN_MLP_SHARED_GATEUP_WEIGHT = 52,
  IN_MLP_SHARED_DOWN_WEIGHT = 53,
  IN_MLP_SHARED_EXPERT_GATE = 54,
};

static const uint64_t WEIGHT_COUNT_PER_LAYER = 55;

static const std::unordered_map<std::string, int> WEIGHT_MAPPING = {
    {"input_layernorm.weight", IN_INPUT_NORM_WEIGHT},

    {"self_attn.q_proj.weight", IN_QKV_WEIGHT_0},

    {"self_attn.k_proj.weight", IN_QKV_WEIGHT_1},

    {"self_attn.v_proj.weight", IN_QKV_WEIGHT_2},

    {"self_attn.o_proj.weight", IN_ATTENTION_OUT_WEIGHT},

    {"self_attn.q_norm.weight", IN_Q_NORM_WEIGHT},
    {"self_attn.k_norm.weight", IN_K_NORM_WEIGHT},

    {"post_attention_layernorm.weight", IN_SELFATTENTION_OUT_NORM_WEIGHT},

    // MoE Gate
    {"mlp.gate.weight", IN_BLOCK_SPARSE_MOE_GATE_WEIGHT},

    // Expert MLP - Gate/Up projections
    {"gate_proj.weight", IN_MLP_GATEUP_WEIGHT_EXPERT},

    {"up_proj.weight", IN_MLP_GATEUP_WEIGHT_EXPERT},

    // Expert MLP - Down projection
    {"down_proj.weight", IN_MLP_DOWN_WEIGHT_EXPERT},

};

static const std::unordered_map<std::string, int> WEIGHT_MAPPING_W8A8 = {
    {"input_layernorm.weight", IN_INPUT_NORM_WEIGHT},
    {"input_layernorm.bias", IN_INPUT_NORM_NEW_BIAS},

    {"self_attn.q_proj.weight", IN_QKV_WEIGHT_0},
    {"self_attn.q_proj.bias", IN_QKV_BIAS_0},
    {"self_attn.q_proj.deq_scale", IN_QKV_DESCALE_0},
    {"self_attn.q_proj.weight_offset", IN_QKV_OFFSET_0},
    {"self_attn.q_proj.weight_scale", IN_QKV_SCALE_0},

    {"self_attn.k_proj.weight", IN_QKV_WEIGHT_1},
    {"self_attn.k_proj.bias", IN_QKV_BIAS_1},
    {"self_attn.k_proj.deq_scale", IN_QKV_DESCALE_1},
    {"self_attn.k_proj.weight_offset", IN_QKV_OFFSET_1},
    {"self_attn.k_proj.weight_scale", IN_QKV_SCALE_1},

    {"self_attn.v_proj.weight", IN_QKV_WEIGHT_2},
    {"self_attn.v_proj.bias", IN_QKV_BIAS_2},
    {"self_attn.v_proj.deq_scale", IN_QKV_DESCALE_2},
    {"self_attn.v_proj.weight_offset", IN_QKV_OFFSET_2},
    {"self_attn.v_proj.weight_scale", IN_QKV_SCALE_2},

    {"self_attn.o_proj.weight", IN_ATTENTION_OUT_WEIGHT},
    {"self_attn.o_proj.quant_bias", IN_ATTENTION_OUT_BIAS},
    {"self_attn.o_proj.deq_scale", IN_ATTENTION_OUT_DESCALE},
    {"self_attn.o_proj.weight_offset", IN_ATTENTION_OUT_OFFSET},
    {"self_attn.o_proj.weight_scale", IN_ATTENTION_OUT_SCALE},

    {"self_attn.q_norm.weight", IN_Q_NORM_WEIGHT},
    {"self_attn.k_norm.weight", IN_K_NORM_WEIGHT},

    {"post_attention_layernorm.weight", IN_SELFATTENTION_OUT_NORM_WEIGHT},
    {"post_attention_layernorm.bias", IN_SELFATTENTION_OUT_NEW_NORM_BIAS},

    // MoE Gate
    {"mlp.gate.weight", IN_BLOCK_SPARSE_MOE_GATE_WEIGHT},

    {"gate_proj.weight", IN_MLP_GATEUP_WEIGHT_EXPERT},
    {"gate_proj.weight_offset", IN_MLP_GATEUP_OFFSET_EXPERT},
    {"gate_proj.weight_scale", IN_MLP_GATEUP_SCALE_EXPERT},
    {"up_proj.weight", IN_MLP_GATEUP_WEIGHT_EXPERT},
    {"up_proj.weight_offset", IN_MLP_GATEUP_OFFSET_EXPERT},
    {"up_proj.weight_scale", IN_MLP_GATEUP_SCALE_EXPERT},

    {"down_proj.weight", IN_MLP_DOWN_WEIGHT_EXPERT},
    {"down_proj.weight_offset", IN_MLP_DOWN_OFFSET_EXPERT},
    {"down_proj.weight_scale", IN_MLP_DOWN_SCALE_EXPERT},
};

static const std::unordered_map<std::string, std::vector<int>>
    SPECIAL_MULTI_ASSIGN_W8A8 = {
        {"input_layernorm.weight",
         {IN_INPUT_NORM_WEIGHT, IN_INPUT_NORM_NEW_WEIGHT}},
        {"post_attention_layernorm.weight",
         {IN_SELFATTENTION_OUT_NORM_WEIGHT,
          IN_SELFATTENTION_OUT_NEW_NORM_WEIGHT}},
};

static const std::map<int, int> WEIGHT_SHARD = {
    {IN_QKV_WEIGHT_0, 0},
    {IN_QKV_WEIGHT_1, 0},
    {IN_QKV_WEIGHT_2, 0},
    {IN_ATTENTION_OUT_WEIGHT, 1},
    {IN_MLP_GATEUP_WEIGHT_EXPERT, 0},
    {IN_MLP_DOWN_WEIGHT_EXPERT, 1},
};

static const std::map<int, int> WEIGHT_SHARD_W8A8 = {
    {IN_QKV_WEIGHT_0, 0},
    {IN_QKV_OFFSET_0, 0},
    {IN_QKV_SCALE_0, 0},
    {IN_QKV_WEIGHT_1, 0},
    {IN_QKV_OFFSET_1, 0},
    {IN_QKV_SCALE_1, 0},
    {IN_QKV_WEIGHT_2, 0},
    {IN_QKV_OFFSET_2, 0},
    {IN_QKV_SCALE_2, 0},
    {IN_ATTENTION_OUT_WEIGHT, 1},
    {IN_MLP_GATEUP_WEIGHT_EXPERT, 0},
    {IN_MLP_GATEUP_OFFSET_EXPERT, 0},
    {IN_MLP_GATEUP_SCALE_EXPERT, 0},
    {IN_MLP_DOWN_WEIGHT_EXPERT, 1},
};

NpuQwen3MoeDecoderLayerImpl::NpuQwen3MoeDecoderLayerImpl(
    const ModelContext& context,
    const int32_t layer_id)
    : NpuBaseLayer(context),
      device_id_(context.get_tensor_options().device().index()),
      layer_id_(layer_id),
      num_speculative_tokens_(
          context.get_model_args().num_speculative_tokens()) {
  auto model_args = context.get_model_args();
  auto parallel_args = context.get_parallel_args();
  auto options = context.get_tensor_options();

  num_experts_ = model_args.num_experts();
  ep_size_ = parallel_args.ep_size();
  ep_local_tp_size_ = parallel_args.world_size() / ep_size_;
  CHECK_EQ(parallel_args.world_size(), ep_size_ * ep_local_tp_size_);
  ep_local_tp_rank_ = parallel_args.rank() % ep_local_tp_size_;
  num_experts_per_partition_ = model_args.num_experts() / ep_size_;
  ep_rank_ = parallel_args.rank() / ep_local_tp_size_;
  // int ep_rank = prefill_param_.rank /  ep_local_tp_size_;
  start_expert_id_ = ep_rank_ * num_experts_per_partition_;
  end_expert_id_ = start_expert_id_ + num_experts_per_partition_ - 1;

  dp_size_ = parallel_args.dp_size();
  dp_local_tp_size_ = parallel_args.world_size() / dp_size_;
  CHECK_EQ(parallel_args.world_size(), dp_size_ * dp_local_tp_size_);
  dp_local_tp_rank_ = parallel_args.rank() % dp_local_tp_size_;

  param_from_args(prefill_param_, model_args, parallel_args, true);
  param_from_args(decode_param_, model_args, parallel_args, false);
  initialize_tensors(options);
}

void NpuQwen3MoeDecoderLayerImpl::initialize_tensors(
    const torch::TensorOptions& options) {
  // initializ placeholder
  at_weight_tensors_.resize(WEIGHT_COUNT_PER_LAYER);
  atb_weight_tensors_.resize(WEIGHT_COUNT_PER_LAYER);
  placeholder_vec_ = {1};
  int_tensor_placeholder_ = torch::ones({1}).to(torch::kInt32).to(device_);
  slot_tensor_placeholder_ = torch::full({1}, 0).to(torch::kInt32).to(device_);
  block_tables_placeholder_ =
      torch::zeros({1, 1}).to(torch::kInt32).to(device_);
  tensor_placeholder_ = torch::zeros({1}).to(options);
  resize_experts_weights(num_experts_per_partition_);
  one_hot_ = torch::tensor({1}, torch::kInt32).to(device_);
  zero_hot_ = torch::tensor({0}, torch::kInt32).to(device_);
  at_start_expert_id_ =
      torch::tensor({start_expert_id_}, torch::kInt64).to(device_);
  at_in_device_expert_count_ =
      torch::tensor({num_experts_per_partition_ - 1}, torch::kInt64)
          .to(device_);
  expert_group_ = torch::tensor({1}, torch::dtype(torch::kInt32)).to(device_);
  initialize_weight_tensors(options);
}

void NpuQwen3MoeDecoderLayerImpl::param_from_args(
    atb_speed::qwen::MoeDecoderLayerParam& param,
    const ModelArgs& args,
    const ParallelArgs& parallel_args,
    bool is_prefill) {
  initialize_basic_parameters(param, args, parallel_args, is_prefill);
  initialize_attention_parameters(param, args, parallel_args);
  initialize_mlp_parameters(param, args, parallel_args);
  initialize_parallel_parameters(param, parallel_args);
  initialize_quantization_parameters(param);
}

void NpuQwen3MoeDecoderLayerImpl::resize_experts_weights(
    int num_of_device_experts) {
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

void NpuQwen3MoeDecoderLayerImpl::initialize_weight_tensors(
    const torch::TensorOptions& options) {
  for (int i = 0; i < WEIGHT_COUNT_PER_LAYER; ++i) {
    at_weight_tensors_[i] = torch::zeros({1}).to(options);
  }
}

void NpuQwen3MoeDecoderLayerImpl::initialize_basic_parameters(
    atb_speed::qwen::MoeDecoderLayerParam& param,
    const ModelArgs& args,
    const ParallelArgs& parallel_args,
    bool is_prefill) {
  param.isFA = false;
  param.isPrefill = is_prefill;
  param.isBF16 = args.dtype() == "bfloat16";
  param.enableSwiGLU = true;
  param.enableLcoc = is_prefill;  // false;

  param.mlpLinearTransposeType = {-1, -1, -1, -1};

  if (quantize_type_.empty()) {
    param.moeLinearTransposeType = std::vector<int>{1, 1, -1, 1};
  } else {
    param.moeLinearTransposeType = std::vector<int>{1, 0, -1, 1};
  }
  param.normEps = args.rms_norm_eps();
  param.rank = parallel_args.rank();
  param.backend = FLAGS_communication_backend;
  // param.rankTableFile = FLAGS_rank_tablefile;

  param.layerId = layer_id_;
  param.numHiddenLayers = 0;
  param.enableIntraLayerAddNorm = false;
  param.enableInterLayerAddNorm = false;
  if (quantize_type_.empty()) {
    param.enableGMMSwigluQuant = false;
  } else {
    param.enableGMMSwigluQuant =
        (is_prefill && parallel_args.world_size() > 16) || !is_prefill;
  }

  param.enableSpeculate = false;                    // MTP
  param.enableSwiGLUQuantForSharedExperts = false;  // TODO

  param.useQKNorm = true;
  param.rmsnormQKNorm = true;
  param.hiddenSizePerAttentionHead = args.head_dim();
  std::optional<long int> optionalValue = args.n_kv_heads();
  param.numKeyValueHeadsPerRank =
      static_cast<int>(optionalValue.value()) / parallel_args.world_size();
  param.numAttentionHeadsPerRank = args.n_heads() / dp_local_tp_size_;

  param.attnLinearTransposeType = {1, -1, -1, 1, -1, -1};
  param.worldSize = parallel_args.world_size();

  if (is_prefill) {
    param.enableAclnnRmsNorm = quantize_type_.empty();
  }
}

void NpuQwen3MoeDecoderLayerImpl::initialize_attention_parameters(
    atb_speed::qwen::MoeDecoderLayerParam& param,
    const ModelArgs& args,
    const ParallelArgs& parallel_args) {
  param.enableFA3 = false;           // TODO
  param.enableKvQuantLayer = false;  // TODO
}

void NpuQwen3MoeDecoderLayerImpl::initialize_mlp_parameters(
    atb_speed::qwen::MoeDecoderLayerParam& param,
    const ModelArgs& args,
    const ParallelArgs& parallel_args) {
  param.hasSharedExpert = (args.n_shared_experts() > 0);
  param.hasSharedExpertGate = false;
  param.processLogits = "normalization";
  param.numOfSelectedExperts = {args.num_experts_per_tok()};

  param.expertParallelDegree = 1;
  param.enableFusedRouting = 1;
  param.numOfExperts = args.num_experts();
  param.maskStartIdx = 0;
  param.routingMethod = "softMaxTopK";

  param.quantGroupSize = 0;

  param.enableInitQuant = false;
  param.enableSwigluQuant = false;
  param.enableCVOverlap = false;  // TODO
}

void NpuQwen3MoeDecoderLayerImpl::initialize_parallel_parameters(
    atb_speed::qwen::MoeDecoderLayerParam& param,
    const ParallelArgs& parallel_args) {
  param.lmHeadLocalTp = 0;
  param.mapping = parallel_args.mapping();
  param.maxDecodeDpTokenSize = 0;  // TODO
}

void NpuQwen3MoeDecoderLayerImpl::initialize_quantization_parameters(
    atb_speed::qwen::MoeDecoderLayerParam& param) {
  if (quantize_type_.empty()) {
    param.packQuantType = {static_cast<int>(PackType::ALL_FP),
                           static_cast<int>(PackType::ALL_FP)};
    param.attnLinearQuantType = {static_cast<int>(LinearType::FP),
                                 static_cast<int>(LinearType::INVALID),
                                 static_cast<int>(LinearType::INVALID),
                                 static_cast<int>(LinearType::FP),
                                 static_cast<int>(LinearType::INVALID),
                                 static_cast<int>(LinearType::INVALID)};
    param.mlpLinearQuantType = {static_cast<int>(LinearType::INVALID),
                                static_cast<int>(LinearType::INVALID),
                                static_cast<int>(LinearType::INVALID),
                                static_cast<int>(LinearType::INVALID)};

    param.moeLinearQuantType = {static_cast<int>(LinearType::FP),
                                static_cast<int>(LinearType::FP),
                                static_cast<int>(LinearType::INVALID),
                                static_cast<int>(LinearType::FP)};
  } else {
    param.packQuantType = {static_cast<int>(PackType::ALL_W8A8_DYNAMIC_ANTI),
                           static_cast<int>(PackType::ALL_W8A8_DYNAMIC_ANTI)};
    param.attnLinearQuantType = {static_cast<int>(LinearType::INT),
                                 static_cast<int>(LinearType::INVALID),
                                 static_cast<int>(LinearType::INVALID),
                                 static_cast<int>(LinearType::INT),
                                 static_cast<int>(LinearType::INVALID),
                                 static_cast<int>(LinearType::INVALID)};
    param.mlpLinearQuantType = {static_cast<int>(LinearType::INVALID),
                                static_cast<int>(LinearType::INVALID),
                                static_cast<int>(LinearType::INVALID),
                                static_cast<int>(LinearType::INVALID)};
    param.moeLinearQuantType = {static_cast<int>(LinearType::FP),
                                static_cast<int>(LinearType::INT),
                                static_cast<int>(LinearType::INVALID),
                                static_cast<int>(LinearType::INT)};
  }
}

void NpuQwen3MoeDecoderLayerImpl::load_state_dict(const StateDict& state_dict) {
  for (const auto& [name, tensor] : state_dict) {
    bool is_sharded = false;
    int index = 0;

    if (absl::StartsWith(name, "mlp.experts")) {
      process_expert_weights(state_dict, name, tensor);
      continue;
    }

    if (absl::StartsWith(name, "mlp") && !absl::StrContains(name, "gate.")) {
      process_mlp_common_weights(state_dict, name, tensor);
      continue;
    }

    process_general_weights(state_dict, name, tensor);
  }
}

int NpuQwen3MoeDecoderLayerImpl::get_mapped_index(
    const std::string& name,
    const std::unordered_map<std::string, int>& mapping) {
  const auto it = mapping.find(name);
  if (it == mapping.end()) {
    LOG(ERROR) << "Missing mapping for: " << name;
    return -1;
  }

  return it->second;
}

void NpuQwen3MoeDecoderLayerImpl::process_expert_weights(
    const StateDict& state_dict,
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

  torch::Tensor tmp_tensor = is_sharded
                                 ? get_sharded_tensor(state_dict,
                                                      name,
                                                      shard_map.at(index),
                                                      ep_local_tp_rank_,
                                                      ep_local_tp_size_)
                                 : tensor;

  experts_weights_[suffix][local_index] = tmp_tensor.clone();
}

void NpuQwen3MoeDecoderLayerImpl::process_mlp_common_weights(
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

void NpuQwen3MoeDecoderLayerImpl::process_general_weights(
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

  if (is_sharded) {
    tmp_tensor = get_sharded_tensor(state_dict,
                                    name,
                                    shard_map.at(index),
                                    dp_local_tp_rank_,
                                    dp_local_tp_size_)
                     .to(device_);
  } else {
    tmp_tensor = tensor.to(device_);
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

torch::Tensor NpuQwen3MoeDecoderLayerImpl::get_sharded_tensor(
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

torch::Tensor NpuQwen3MoeDecoderLayerImpl::get_sharded_tensor(
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

std::string NpuQwen3MoeDecoderLayerImpl::extract_endswith(
    const std::string& input) {
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

int NpuQwen3MoeDecoderLayerImpl::extract_expert_index(const std::string& name) {
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

void NpuQwen3MoeDecoderLayerImpl::verify_loaded_weights(
    const std::string& prefix) const {
  for (const auto& [name, index] : WEIGHT_MAPPING) {
    if (name == "down_proj.weight" || name == "gate_proj.weight" ||
        name == "up_proj.weight") {
      continue;
    }
    CHECK(at_weight_tensors_[index].sizes() != std::vector<int64_t>({1}))
        << "weight is not loaded for " << name;
  }
}

void NpuQwen3MoeDecoderLayerImpl::merge_loaded_weights() {
  merge_experts_weights();
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

  if (quantize_type_.compare("w8a8_dynamic") == 0) {
    at_weight_tensors_[IN_QKV_BIAS_0] =
        torch::zeros({1}, torch::kFloat16).to(device_);
    at_weight_tensors_[IN_QKV_BIAS_1] =
        torch::zeros({1}, torch::kFloat16).to(device_);
    at_weight_tensors_[IN_QKV_BIAS_2] =
        torch::zeros({1}, torch::kFloat16).to(device_);
    at_weight_tensors_[IN_ATTENTION_OUT_BIAS] =
        torch::zeros({1}, torch::kFloat16).to(device_);

    at_weight_tensors_[IN_QKV_DESCALE_0] =
        torch::zeros({1}, torch::kFloat16).to(device_);
    at_weight_tensors_[IN_QKV_DESCALE_1] =
        torch::zeros({1}, torch::kFloat16).to(device_);
    at_weight_tensors_[IN_QKV_DESCALE_2] =
        torch::zeros({1}, torch::kFloat16).to(device_);
    at_weight_tensors_[IN_ATTENTION_OUT_DESCALE] =
        torch::zeros({1}, torch::kFloat16).to(device_);

    at_weight_tensors_[IN_QKV_OFFSET_0] =
        torch::cat({at_weight_tensors_[IN_QKV_OFFSET_0],
                    at_weight_tensors_[IN_QKV_OFFSET_1],
                    at_weight_tensors_[IN_QKV_OFFSET_2]},
                   0)
            .contiguous()
            .view(-1);
    at_weight_tensors_[IN_QKV_OFFSET_1] =
        torch::zeros({1}, torch::kFloat16).to(device_);
    at_weight_tensors_[IN_QKV_OFFSET_2] =
        torch::zeros({1}, torch::kFloat16).to(device_);
    at_weight_tensors_[IN_ATTENTION_OUT_OFFSET] =
        at_weight_tensors_[IN_ATTENTION_OUT_OFFSET].contiguous().view(-1);

    at_weight_tensors_[IN_QKV_SCALE_0] =
        torch::cat({at_weight_tensors_[IN_QKV_SCALE_0],
                    at_weight_tensors_[IN_QKV_SCALE_1],
                    at_weight_tensors_[IN_QKV_SCALE_2]},
                   0)
            .contiguous()
            .view(-1);
    at_weight_tensors_[IN_QKV_SCALE_1] =
        torch::zeros({1}, torch::kFloat16).to(device_);
    at_weight_tensors_[IN_QKV_SCALE_2] =
        torch::zeros({1}, torch::kFloat16).to(device_);
    at_weight_tensors_[IN_ATTENTION_OUT_SCALE] =
        at_weight_tensors_[IN_ATTENTION_OUT_SCALE].contiguous().view(-1);
  }

  c10_npu::NPUCachingAllocator::emptyCache();
  for (int i = 0; i < WEIGHT_COUNT_PER_LAYER; ++i) {
    atb_weight_tensors_[i] =
        atb_speed::Utils::AtTensor2Tensor(at_weight_tensors_[i]);
  }
  init_layer();
}

torch::Tensor NpuQwen3MoeDecoderLayerImpl::convert_fp16_to_int64(
    const torch::Tensor& fp16_tensor) {
  auto float_tensor = fp16_tensor.to(torch::kFloat32);
  auto int32_tensor = float_tensor.view(torch::kInt32);
  auto int64_tensor = int32_tensor.to(torch::kInt64);
  return int64_tensor;
}

void NpuQwen3MoeDecoderLayerImpl::convert_descaled_weights_to_float() {
  auto convert_to_float = [this](int index) {
    at_weight_tensors_[index] = at_weight_tensors_[index].to(torch::kFloat32);
  };
  convert_to_float(IN_ATTENTION_OUT_DESCALE);
}

void NpuQwen3MoeDecoderLayerImpl::merge_experts_weights() {
  if (experts_weights_.count("gate_proj.weight") > 0) {
    auto& gate_weight = experts_weights_["gate_proj.weight"];
  }

  if (experts_weights_.count("up_proj.weight") > 0) {
    auto& up_weight = experts_weights_["up_proj.weight"];
  }

  try {
    torch::Tensor mlp_gateup_weight;
    if (quantize_type_.compare("w8a8_dynamic") == 0) {
      mlp_gateup_weight =
          merge_experts_weights(experts_weights_["gate_proj.weight"],
                                experts_weights_["up_proj.weight"],
                                /*transpose=*/true);
      at_weight_tensors_[IN_MLP_GATEUP_OFFSET_EXPERT] =
          merge_experts_weights(experts_weights_["gate_proj.weight_offset"],
                                experts_weights_["up_proj.weight_offset"]);
      at_weight_tensors_[IN_MLP_GATEUP_SCALE_EXPERT] =
          merge_experts_weights(experts_weights_["gate_proj.weight_scale"],
                                experts_weights_["up_proj.weight_scale"]);
    } else {
      mlp_gateup_weight =
          merge_experts_weights(experts_weights_["gate_proj.weight"],
                                experts_weights_["up_proj.weight"],
                                /*transpose=*/false);
    }
    at_weight_tensors_[IN_MLP_GATEUP_WEIGHT_EXPERT] =
        at_npu::native::npu_format_cast(mlp_gateup_weight, 2).contiguous();
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

    at_weight_tensors_[IN_MLP_DOWN_WEIGHT_EXPERT] =
        at_npu::native::npu_format_cast(mlp_down_weight, 2).contiguous();

    if (quantize_type_.compare("w8a8_dynamic") == 0) {
      at_weight_tensors_[IN_MLP_DOWN_OFFSET_EXPERT] =
          merge_experts_weights(experts_weights_["down_proj.weight_offset"]);
      at_weight_tensors_[IN_MLP_DOWN_SCALE_EXPERT] =
          merge_experts_weights(experts_weights_["down_proj.weight_scale"]);
    }
  } catch (const std::exception& e) {
    LOG(ERROR) << "[ERROR] Exception in down weight processing: " << e.what();
    throw;
  }
}

torch::Tensor NpuQwen3MoeDecoderLayerImpl::merge_experts_weights(
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

torch::Tensor NpuQwen3MoeDecoderLayerImpl::merge_experts_weights(
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

int64_t NpuQwen3MoeDecoderLayerImpl::init_layer() {
  name_ = "qwen3_moe_decoder_layer " + std::to_string(layer_id_);
  model_name_ = "Qwen3_Moe";
  CHECK_OPERATION_STATUS_RETURN(init_node(prefill_node_, prefill_param_));
  CHECK_OPERATION_STATUS_RETURN(init_node(decode_node_, decode_param_));

  return atb::NO_ERROR;
}

int64_t NpuQwen3MoeDecoderLayerImpl::init_node(
    atb_speed::Model::Node& node,
    atb_speed::qwen::MoeDecoderLayerParam& param) {
  atb::Operation* operation = nullptr;
  atb_speed::qwen::MoeDecoderLayer(param, &operation);
  node.operation.reset(operation);
  if (node.operation == nullptr) {
    LOG(ERROR) << "node.operation is null";
    return -1;
  }
  if (node.operation->GetInputNum() < 1) {
    LOG(ERROR) << "Can not resize number which is smaller than 1";
    return -1;
  }
  node.inTensors.resize(node.operation->GetInputNum());
  node.outTensors.resize(1);
  size_t inTensorId = 1;

  for (size_t weightTensorId = 0; weightTensorId < WEIGHT_COUNT_PER_LAYER;
       ++weightTensorId) {
    node.inTensors.at(weightTensorId) = &atb_weight_tensors_[weightTensorId];
  }

  node.variantPack.inTensors.reserve(node.inTensors.size());
  node.variantPack.inTensors.resize(node.inTensors.size());
  node.variantPack.outTensors.reserve(1);
  node.variantPack.outTensors.resize(1);

  return atb::NO_ERROR;
}

torch::Tensor NpuQwen3MoeDecoderLayerImpl::forward(
    torch::Tensor& x,
    torch::Tensor& cos_pos,
    torch::Tensor& sin_pos,
    torch::Tensor& attn_mask,
    KVCache& kv_cache,
    const ModelInputParams& input_params,
    torch::Tensor& expert_array,
    aclrtEvent* event,
    std::atomic<bool>* event_flag,
    int node_id) {
  atb::Status st;
  if (input_params.global_empty_kv_cache) {
    build_node_variant_pack(prefill_node_,
                            x,
                            cos_pos,
                            sin_pos,
                            attn_mask,
                            kv_cache,
                            input_params,
                            expert_array,
                            true);
    st = execute_node(prefill_node_, node_id, {event}, {event_flag});
    LOG_IF(FATAL, st != 0) << model_name_
                           << "excute prefill layer fail, error code: " << st;
  } else {
    build_node_variant_pack(decode_node_,
                            x,
                            cos_pos,
                            sin_pos,
                            /*attn_mask*/ tensor_placeholder_,
                            kv_cache,
                            input_params,
                            expert_array,
                            false);
    st = execute_node(decode_node_, node_id + 1000, {event}, {event_flag});
    LOG_IF(FATAL, st != 0) << model_name_
                           << "excute decode layer fail, error code: " << st;
  }

  return tensor_placeholder_;
}

void NpuQwen3MoeDecoderLayerImpl::build_node_variant_pack(
    atb_speed::Model::Node& node,
    torch::Tensor& x,
    torch::Tensor& cos_pos,
    torch::Tensor& sin_pos,
    torch::Tensor& attn_mask,
    KVCache& kv_cache,
    const ModelInputParams& input_params,
    torch::Tensor& expert_array,
    bool is_prefill) {
  internal_tensor_ = atb_speed::Utils::AtTensor2Tensor(x);
  int32_t input_idx = 0;
  auto& dp_ep_padding = input_params.dp_ep_padding_data;

  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER) = internal_tensor_;
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 1) =
      atb_speed::Utils::AtTensor2Tensor(expert_array);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 2) =
      atb_speed::Utils::AtTensor2Tensor(expert_group_);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 3) =
      atb_speed::Utils::AtTensor2Tensor(one_hot_);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 4) =
      atb_speed::Utils::AtTensor2Tensor(zero_hot_);

  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 5) =
      atb_speed::Utils::AtTensor2Tensor(cos_pos);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 6) =
      atb_speed::Utils::AtTensor2Tensor(sin_pos);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 7) =
      atb_speed::Utils::AtTensor2Tensor(attn_mask);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 8) =
      atb_speed::Utils::AtTensor2Tensor(kv_cache.get_k_cache());
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 9) =
      atb_speed::Utils::AtTensor2Tensor(kv_cache.get_v_cache());

  if (!input_params.block_tables.defined() ||
      input_params.block_tables.storage().data() == nullptr) {
    node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 10) =
        atb_speed::Utils::AtTensor2Tensor(int_tensor_placeholder_);
    node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 10).hostData =
        const_cast<int32_t*>(placeholder_vec_.data());
  } else {
    node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 10) =
        atb_speed::Utils::AtTensor2Tensor(input_params.kv_seq_lens);
    node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 10).hostData =
        const_cast<int32_t*>(input_params.kv_seq_lens_vec.data());
  }
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 11) =
      atb_speed::Utils::AtTensor2Tensor(tensor_placeholder_);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 12) =
      atb_speed::Utils::AtTensor2Tensor(tensor_placeholder_);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 12).hostData =
      const_cast<int32_t*>(placeholder_vec_.data());
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 13) =
      atb_speed::Utils::AtTensor2Tensor(tensor_placeholder_);
  if (!input_params.block_tables.defined() ||
      input_params.block_tables.storage().data() == nullptr) {
    node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 14) =
        atb_speed::Utils::AtTensor2Tensor(block_tables_placeholder_);
  } else {
    node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 14) =
        atb_speed::Utils::AtTensor2Tensor(input_params.block_tables);
  }
  if (!input_params.block_tables.defined() ||
      input_params.block_tables.storage().data() == nullptr) {
    node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 15) =
        atb_speed::Utils::AtTensor2Tensor(slot_tensor_placeholder_);
  } else {
    node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 15) =
        atb_speed::Utils::AtTensor2Tensor(input_params.new_cache_slots);
  }

  for (size_t i = 0; i < WEIGHT_COUNT_PER_LAYER; ++i) {
    CHECK_THROW(node.inTensors.at(i) == nullptr,
                model_name_ << " inTensor " << i << " is NULL");
    node.variantPack.inTensors.at(i) = *node.inTensors.at(i);
  }

  node.variantPack.outTensors.at(0) = internal_tensor_;
}

}  // namespace layer
}  // namespace xllm

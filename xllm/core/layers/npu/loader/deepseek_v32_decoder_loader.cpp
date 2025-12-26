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
#include "deepseek_v32_decoder_loader.h"

#include <torch_npu/csrc/core/npu/NPUFormat.h>

#include <iostream>

namespace xllm {
namespace layer {

enum DecoderLayerTensorId : int {
  IN_INPUT_NORM_WEIGHT = 0,
  IN_INPUT_NORM_BIAS = 1,
  IN_INPUT_NORM_NEW_WEIGHT = 2,
  IN_INPUT_NORM_NEW_BIAS = 3,

  IN_Q_PROJ_A_WEIGHT = 4,
  IN_Q_PROJ_A_BIAS = 5,
  IN_Q_PROJ_A_DESCALE = 6,
  IN_Q_PROJ_A_OFFSET = 7,
  IN_Q_PROJ_A_SCALE = 8,
  IN_Q_PROJ_A_COMPRESS_IDX = 9,
  IN_Q_PROJ_A_LAYERNORM_WEIGHT = 10,
  IN_Q_PROJ_A_LAYERNORM_BIAS = 11,

  IN_Q_PROJ_B_WEIGHT = 12,
  IN_Q_PROJ_B_BIAS = 13,
  IN_Q_PROJ_B_DESCALE = 14,
  IN_Q_PROJ_B_OFFSET = 15,
  IN_Q_PROJ_B_SCALE = 16,
  IN_Q_PROJ_B_COMPRESS_IDX = 17,

  IN_KV_PROJ_WITH_MQA_WEIGHT = 18,
  IN_KV_PROJ_WITH_MQA_BIAS = 19,
  IN_KV_PROJ_WITH_MQA_DESCALE = 20,
  IN_KV_PROJ_WITH_MQA_OFFSET = 21,
  IN_KV_PROJ_WITH_MQA_SCALE = 22,
  IN_KV_PROJ_WITH_MQA_COMPRESS_IDX = 23,

  IN_KV_PROJ_A_LAYERNORM_WEIGHT = 24,
  IN_KV_PROJ_A_LAYERNORM_BIAS = 25,

  IN_K_PROJ_B_FOR_Q_WEIGHT = 26,
  IN_K_PROJ_B_FOR_Q_BIAS = 27,
  IN_K_PROJ_B_FOR_Q_DESCALE = 28,
  IN_K_PROJ_B_FOR_Q_OFFSET = 29,
  IN_K_PROJ_B_FOR_Q_SCALE = 30,
  IN_K_PROJ_B_FOR_Q_COMPRESS_IDX = 31,

  IN_V_PROJ_B_FOR_O_WEIGHT = 32,
  IN_V_PROJ_B_FOR_O_BIAS = 33,
  IN_V_PROJ_B_FOR_O_DESCALE = 34,
  IN_V_PROJ_B_FOR_O_OFFSET = 35,
  IN_V_PROJ_B_FOR_O_SCALE = 36,
  IN_V_PROJ_B_FOR_O_COMPRESS_IDX = 37,

  IN_ATTENTION_OUT_WEIGHT = 38,
  IN_ATTENTION_OUT_BIAS = 39,
  IN_ATTENTION_OUT_DESCALE = 40,
  IN_ATTENTION_OUT_OFFSET = 41,
  IN_ATTENTION_OUT_SCALE = 42,
  IN_ATTENTION_OUT_COMPRESS_IDX = 43,

  IN_SELFATTENTION_OUT_NORM_WEIGHT = 44,
  IN_SELFATTENTION_OUT_NORM_BIAS = 45,
  IN_SELFATTENTION_OUT_NEW_NORM_WEIGHT = 46,
  IN_SELFATTENTION_OUT_NEW_NORM_BIAS = 47,

  IN_MLP_GATEUP_WEIGHT_SHARED_EXPERT = 48,
  IN_MLP_GATEUP_BIAS_SHARED_EXPERT = 49,
  IN_MLP_GATEUP_DESCALE_SHARED_EXPERT = 50,
  IN_MLP_GATEUP_OFFSET_SHARED_EXPERT = 51,
  IN_MLP_GATEUP_SCALE_SHARED_EXPERT = 52,
  IN_MLP_GATEUP_COMPRESS_IDX_SHARED_EXPERT = 53,

  IN_MLP_DOWN_WEIGHT_SHARED_EXPERT = 54,
  IN_MLP_DOWN_BIAS_SHARED_EXPERT = 55,
  IN_MLP_DOWN_DESCALE_SHARED_EXPERT = 56,
  IN_MLP_DOWN_OFFSET_SHARED_EXPERT = 57,
  IN_MLP_DOWN_SCALE_SHARED_EXPERT = 58,
  IN_MLP_DOWN_COMPRESS_IDX_SHARED_EXPERT = 59,

  IN_SHARED_EXPERT_GATE_WEIGHT = 60,
  IN_SHARED_EXPERT_GATE_BIAS = 61,
  IN_SHARED_EXPERT_GATE_DESCALE = 62,
  IN_SHARED_EXPERT_GATE_OFFSET = 63,
  IN_SHARED_EXPERT_GATE_SCALE = 64,
  IN_SHARED_EXPERT_GATE_COMPRESS_IDX = 65,

  IN_BLOCK_SPARSE_MOE_GATE_WEIGHT = 66,
  IN_BLOCK_SPARSE_MOE_GATE_BIAS = 67,
  IN_BLOCK_SPARSE_MOE_GATE_DESCALE = 68,
  IN_BLOCK_SPARSE_MOE_GATE_OFFSET = 69,
  IN_BLOCK_SPARSE_MOE_GATE_SCALE = 70,
  IN_BLOCK_SPARSE_MOE_GATE_COMPRESS_IDX = 71,

  IN_MLP_GATEUP_WEIGHT_EXPERT = 72,
  IN_MLP_GATEUP_BIAS_EXPERT = 73,
  IN_MLP_GATEUP_DESCALE_EXPERT = 74,
  IN_MLP_GATEUP_OFFSET_EXPERT = 75,
  IN_MLP_GATEUP_SCALE_EXPERT = 76,
  IN_MLP_GATEUP_COMPRESS_IDX_EXPERT = 77,

  IN_MLP_DOWN_WEIGHT_EXPERT = 78,
  IN_MLP_DOWN_BIAS_EXPERT = 79,
  IN_MLP_DOWN_DESCALE_EXPERT = 80,
  IN_MLP_DOWN_OFFSET_EXPERT = 81,
  IN_MLP_DOWN_SCALE_EXPERT = 82,
  IN_MLP_DOWN_COMPRESS_IDX_EXPERT = 83,

  IN_INDEXER_WQ_B_WEIGHT = 84,
  IN_INDEXER_WQ_B_BIAS = 85,
  IN_INDEXER_WQ_B_DESCALE = 86,
  IN_INDEXER_WQ_B_OFFSET = 87,
  IN_INDEXER_WQ_B_SCALE = 88,
  IN_INDEXER_WQ_B_COMPRESS_IDX = 89,

  IN_INDEXER_WK_WEIGHT = 90,
  IN_INDEXER_WK_BIAS = 91,
  IN_INDEXER_WK_DESCALE = 92,
  IN_INDEXER_WK_OFFSET = 93,
  IN_INDEXER_WK_SCALE = 94,
  IN_INDEXER_WK_COMPRESS_IDX = 95,

  IN_INDEXER_K_NORM_WEIGHT = 96,
  IN_INDEXER_K_NORM_BIAS = 97,

  IN_INDEXER_PROJ_WEIGHT = 98,
  IN_INDEXER_PROJ_BIAS = 99,
  IN_INDEXER_PROJ_DESCALE = 100,
  IN_INDEXER_PROJ_OFFSET = 101,
  IN_INDEXER_PROJ_SCALE = 102,
  IN_INDEXER_PROJ_COMPRESS_IDX = 103,
  IN_Q_PROJ_A_RECOMPUTE_WEIGHT = 104,
  IN_Q_PROJ_A_RECOMPUTE_BIAS = 105,
  IN_Q_PROJ_A_RECOMPUTE_DESCALE = 106,
  IN_Q_PROJ_A_RECOMPUTE_OFFSET = 107,
  IN_Q_PROJ_A_RECOMPUTE_SCALE = 108,
  IN_Q_PROJ_A_RECOMPUTE_COMPRESS_IDX = 109,
};

static std::vector<std::pair<int, std::string>> WEIGHT_MAPPING = {};

static const std::unordered_map<std::string, int> WEIGHT_MAPPING_W8A8 = {
    {"input_layernorm.weight", IN_INPUT_NORM_WEIGHT},
    {"input_layernorm.bias", IN_INPUT_NORM_BIAS},

    {"self_attn.q_a_proj.weight", IN_Q_PROJ_A_WEIGHT},
    {"self_attn.q_a_proj.quant_bias", IN_Q_PROJ_A_BIAS},
    {"self_attn.q_a_proj.deq_scale", IN_Q_PROJ_A_DESCALE},
    {"self_attn.q_a_proj.input_offset", IN_Q_PROJ_A_OFFSET},
    {"self_attn.q_a_proj.input_scale", IN_Q_PROJ_A_SCALE},
    {"self_attn.q_a_layernorm.weight", IN_Q_PROJ_A_LAYERNORM_WEIGHT},
    {"self_attn.q_a_layernorm.bias", IN_Q_PROJ_A_LAYERNORM_BIAS},

    {"self_attn.q_proj.weight", IN_Q_PROJ_B_WEIGHT},
    {"self_attn.q_b_proj.weight", IN_Q_PROJ_B_WEIGHT},
    {"self_attn.q_b_proj.quant_bias", IN_Q_PROJ_B_BIAS},
    {"self_attn.q_b_proj.input_scale", IN_Q_PROJ_B_SCALE},
    {"self_attn.q_b_proj.deq_scale", IN_Q_PROJ_B_DESCALE},
    {"self_attn.q_b_proj.input_offset", IN_Q_PROJ_B_OFFSET},

    {"self_attn.kv_a_proj_with_mqa.weight", IN_KV_PROJ_WITH_MQA_WEIGHT},
    {"self_attn.kv_a_proj_with_mqa.quant_bias", IN_KV_PROJ_WITH_MQA_BIAS},
    {"self_attn.kv_a_proj_with_mqa.deq_scale", IN_KV_PROJ_WITH_MQA_DESCALE},
    {"self_attn.kv_a_proj_with_mqa.input_offset", IN_KV_PROJ_WITH_MQA_OFFSET},
    {"self_attn.kv_a_proj_with_mqa.input_scale", IN_KV_PROJ_WITH_MQA_SCALE},

    {"self_attn.kv_a_layernorm.weight", IN_KV_PROJ_A_LAYERNORM_WEIGHT},
    {"self_attn.kv_a_layernorm.bias", IN_KV_PROJ_A_LAYERNORM_BIAS},

    {"self_attn.kv_b_proj.weight", IN_K_PROJ_B_FOR_Q_WEIGHT},  // merge
    // {"self_attn.kv_b_proj.weight", IN_V_PROJ_B_FOR_O_WEIGHT},  // merge

    {"self_attn.o_proj.weight", IN_ATTENTION_OUT_WEIGHT},
    {"self_attn.o_proj.quant_bias", IN_ATTENTION_OUT_BIAS},
    {"self_attn.o_proj.deq_scale", IN_ATTENTION_OUT_DESCALE},
    {"self_attn.o_proj.input_offset", IN_ATTENTION_OUT_OFFSET},
    {"self_attn.o_proj.input_scale", IN_ATTENTION_OUT_SCALE},

    {"self_attn.indexer.wq_b.weight", IN_INDEXER_WQ_B_WEIGHT},
    {"self_attn.indexer.wk.weight", IN_INDEXER_WK_WEIGHT},
    {"self_attn.indexer.k_norm.weight", IN_INDEXER_K_NORM_WEIGHT},
    {"self_attn.indexer.k_norm.bias", IN_INDEXER_K_NORM_BIAS},
    {"self_attn.indexer.weights_proj.weight", IN_INDEXER_PROJ_WEIGHT},

    {"post_attention_layernorm.weight", IN_SELFATTENTION_OUT_NORM_WEIGHT},
    {"post_attention_layernorm.bias", IN_SELFATTENTION_OUT_NORM_BIAS},

    {"mlp.gate_proj.weight", IN_MLP_GATEUP_WEIGHT_SHARED_EXPERT},
    {"mlp.gate_proj.weight_offset", IN_MLP_GATEUP_OFFSET_SHARED_EXPERT},
    {"mlp.gate_proj.weight_scale", IN_MLP_GATEUP_SCALE_SHARED_EXPERT},

    {"mlp.up_proj.weight", IN_MLP_GATEUP_WEIGHT_SHARED_EXPERT},
    {"mlp.up_proj.weight_offset", IN_MLP_GATEUP_OFFSET_SHARED_EXPERT},
    {"mlp.up_proj.weight_scale", IN_MLP_GATEUP_SCALE_SHARED_EXPERT},

    {"mlp.down_proj.weight", IN_MLP_DOWN_WEIGHT_SHARED_EXPERT},
    {"mlp.down_proj.weight_offset", IN_MLP_DOWN_OFFSET_SHARED_EXPERT},
    {"mlp.down_proj.weight_scale", IN_MLP_DOWN_SCALE_SHARED_EXPERT},

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

    {"mlp.gate.weight", IN_BLOCK_SPARSE_MOE_GATE_WEIGHT},
    {"mlp.gate.e_score_correction_bias", IN_BLOCK_SPARSE_MOE_GATE_BIAS},

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

static const std::unordered_map<std::string, int>
    WEIGHT_MAPPING_W8A8_RECOMPUTE = {
        {"self_attn.q_a_proj.weight", IN_Q_PROJ_A_RECOMPUTE_WEIGHT},
        {"self_attn.q_a_proj.quant_bias", IN_Q_PROJ_A_RECOMPUTE_BIAS},
        {"self_attn.q_a_proj.deq_scale", IN_Q_PROJ_A_RECOMPUTE_DESCALE},
        {"self_attn.q_a_proj.input_offset", IN_Q_PROJ_A_RECOMPUTE_OFFSET},
        {"self_attn.q_a_proj.input_scale", IN_Q_PROJ_A_RECOMPUTE_SCALE},
};

static const std::map<int, int> WEIGHT_SHARD = {};

static const std::map<int, int> WEIGHT_SHARD_W8A8 = {
    {IN_Q_PROJ_B_WEIGHT, 0},
    {IN_Q_PROJ_B_BIAS, 0},
    {IN_Q_PROJ_B_DESCALE, 0},
    {IN_K_PROJ_B_FOR_Q_WEIGHT, 0},
    {IN_V_PROJ_B_FOR_O_WEIGHT, 0},
    {IN_ATTENTION_OUT_WEIGHT, 1},
    {IN_MLP_GATEUP_WEIGHT_SHARED_EXPERT, 0},
    {IN_MLP_GATEUP_OFFSET_SHARED_EXPERT, 0},
    {IN_MLP_GATEUP_SCALE_SHARED_EXPERT, 0},
    {IN_MLP_DOWN_WEIGHT_SHARED_EXPERT, 1},
    {IN_MLP_GATEUP_WEIGHT_EXPERT, 0},
    {IN_MLP_GATEUP_OFFSET_EXPERT, 0},
    {IN_MLP_GATEUP_SCALE_EXPERT, 0},
    {IN_MLP_DOWN_WEIGHT_EXPERT, 1},
};

static std::vector<int> SQUEEZE_WEIGHT_VEC = {
    IN_MLP_GATEUP_OFFSET_SHARED_EXPERT,
    IN_MLP_GATEUP_SCALE_SHARED_EXPERT,
    IN_MLP_DOWN_OFFSET_SHARED_EXPERT,
    IN_MLP_DOWN_SCALE_SHARED_EXPERT};

static std::vector<std::string> LINEAR_FOR_ROPE = {
    "self_attn.q_b_proj.weight",
    "self_attn.q_b_proj.quant_bias",
    "self_attn.q_b_proj.deq_scale",
    "self_attn.kv_a_proj_with_mqa.weight",
    "self_attn.kv_a_proj_with_mqa.quant_bias",
    "self_attn.kv_a_proj_with_mqa.deq_scale",
};

DeekseekV32DecoderLoader::DeekseekV32DecoderLoader(
    uint64_t weight_count,
    const ModelContext& context,
    int32_t layer_id,
    int32_t prefill_firstKDenseReplace,
    int32_t prefill_numOfDeviceExperts,
    int32_t prefill_qkRopeHeadDim,
    int32_t prefill_numAttentionHeadsPerRank,
    int32_t decode_worldSize,
    int32_t qk_nope_head_dim,
    int32_t kv_lora_rank,
    int32_t num_key_value_heads,
    int32_t v_head_dim,
    bool prefill_isBF16,
    bool decode_isBF16)
    : BaseLoader(weight_count, context),
      layer_id_(layer_id),
      prefill_firstKDenseReplace_(prefill_firstKDenseReplace),
      prefill_numOfDeviceExperts_(prefill_numOfDeviceExperts),
      prefill_qkRopeHeadDim_(prefill_qkRopeHeadDim),
      prefill_numAttentionHeadsPerRank_(prefill_numAttentionHeadsPerRank),
      decode_worldSize_(decode_worldSize),
      qk_nope_head_dim_(qk_nope_head_dim),
      kv_lora_rank_(kv_lora_rank),
      num_key_value_heads_(num_key_value_heads),
      v_head_dim_(v_head_dim),
      prefill_isBF16_(prefill_isBF16),
      decode_isBF16_(decode_isBF16) {
  auto model_args = context.get_model_args();
  auto options = context.get_tensor_options();

  rank_ = parallel_args_.rank();
  first_k_dense_replace_ = model_args.first_k_dense_replace();
  n_layers_ = model_args.n_layers();
  num_experts_ = model_args.n_routed_experts();
  localWorldSize_ = parallel_args_.mapping().localWorldSize();
  ep_size_ = parallel_args_.ep_size();
  ep_local_tp_size_ = parallel_args_.world_size() / ep_size_;
  CHECK_EQ(parallel_args_.world_size(), ep_size_ * ep_local_tp_size_);
  ep_local_tp_rank_ = parallel_args_.rank() % ep_local_tp_size_;
  num_experts_per_partition_ = model_args.n_routed_experts() / ep_size_;
  redundant_experts_num_ = FLAGS_redundant_experts_num;
  if (FLAGS_enable_eplb) {
    num_experts_per_partition_ += redundant_experts_num_;
  }
  ep_rank_ = parallel_args_.rank() / ep_local_tp_size_;
  start_expert_id_ = ep_rank_ * num_experts_per_partition_;
  end_expert_id_ = start_expert_id_ + num_experts_per_partition_ - 1;
  initialize_tensors(options);
  initialize_weight_tensors(options);
}

void DeekseekV32DecoderLoader::initialize_tensors(
    const torch::TensorOptions& options) {
  tensor_placeholder_ = torch::zeros({1}).to(options);
  reserve_experts_weights(prefill_numOfDeviceExperts_);
  initialize_device_expert_list(decode_worldSize_, num_experts_per_partition_);
}

void DeekseekV32DecoderLoader::load_state_dict(const StateDict& state_dict) {
  for (const auto& [name, tensor] : state_dict) {
    bool is_sharded = false;
    int index = 0;

    if (absl::EndsWith(name, "self_attn.kv_b_proj.weight")) {
      index = WEIGHT_MAPPING_W8A8.at(name);
      set_kv_weight(state_dict, name, index, WEIGHT_SHARD_W8A8.at(index));
      continue;
    }

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

void DeekseekV32DecoderLoader::verify_loaded_weights(
    const std::string& prefix) const {
  for (const auto& [index, name] : WEIGHT_MAPPING) {
    CHECK(at_weight_tensors_[index].sizes() != std::vector<int64_t>({1}))
        << "weight is not loaded for " << prefix + name;
  }
}

int DeekseekV32DecoderLoader::extract_expert_index(const std::string& name) {
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

void DeekseekV32DecoderLoader::process_expert_weights(
    const StateDict& state_dict,
    const std::string& name,
    const torch::Tensor& tensor) {
  // Step 1: Early checks and basic info extraction
  int expert_index = extract_expert_index(name);
  const std::string suffix = extract_endswith(name);
  const int index = get_mapped_index(suffix, WEIGHT_MAPPING_W8A8);
  if (index == -1) {
    return;
  }

  const bool is_sharded = WEIGHT_SHARD_W8A8.count(index);
  const bool needs_eplb = FLAGS_enable_eplb && (rank_ % localWorldSize_ ==
                                                expert_index % localWorldSize_);

  // Step 2: Check if expert is in partition
  const int start_idx = ep_rank_ * num_experts_per_partition_;
  const int end_idx = (ep_rank_ + 1) * num_experts_per_partition_;
  const int safe_end =
      std::min(end_idx, static_cast<int>(device_expert_list_.size()));

  auto it = std::find(device_expert_list_.begin() + start_idx,
                      device_expert_list_.begin() + safe_end,
                      expert_index);
  const bool in_partition = it != device_expert_list_.begin() + safe_end;

  // Early return if neither EPLB nor partition needs this expert
  if (!needs_eplb && !in_partition) {
    return;
  }

  // Step 3: Process tensor
  torch::Tensor processed_tensor;
  {
    std::lock_guard<std::mutex> lock(experts_mutex_);
    processed_tensor = is_sharded
                           ? get_sharded_tensor(state_dict,
                                                name,
                                                WEIGHT_SHARD_W8A8.at(index),
                                                ep_local_tp_rank_,
                                                ep_local_tp_size_)
                           : tensor;

    if (!decode_isBF16_) {
      if (absl::EndsWith(name, "_offset")) {
        processed_tensor = processed_tensor.to(torch::kFloat16);
      } else if (absl::EndsWith(name, "_scale")) {
        processed_tensor = processed_tensor.to(torch::kFloat32);
      }
    }
  }

  // Step 4: Handle EPLB case
  if (needs_eplb) {
    std::lock_guard<std::mutex> lock(experts_mutex_);
    std::string shm_key = get_expert_shm_key(layer_id_, expert_index, suffix);
    shared_buffer_->add_tensor(expert_index,
                               layer_id_ - first_k_dense_replace_,
                               shm_key,
                               processed_tensor.contiguous());
  }

  // Step 5: Handle partition case
  if (in_partition) {
    std::vector<size_t> matches_pos;
    for (auto iter = it; iter != device_expert_list_.begin() + safe_end;
         ++iter) {
      if (*iter == expert_index) {
        matches_pos.emplace_back(
            std::distance(device_expert_list_.begin(), iter) - start_idx);
      }
    }

    if (!matches_pos.empty()) {
      std::lock_guard<std::mutex> lock(experts_mutex_);
      for (auto pos : matches_pos) {
        experts_weights_[suffix][pos] = processed_tensor.clone();
      }
    }
  }
}

void DeekseekV32DecoderLoader::initialize_weight_tensors(
    const torch::TensorOptions& options) {
  for (int i = 0; i < weight_count_; ++i) {
    at_weight_tensors_[i] = torch::zeros({1}).to(options);
  }

  if (FLAGS_enable_eplb) {
    const int64_t size =
        50LL * 1024LL * 1024LL * int64_t(n_layers_ - first_k_dense_replace_);
    shared_buffer_ = std::make_unique<ExpertBufferManager>(
        num_experts_, n_layers_ - first_k_dense_replace_, size);
  }
}

void DeekseekV32DecoderLoader::convert_offsets_to_int8() {
  auto convert_to_int8 = [this](int index) {
    at_weight_tensors_[index] =
        at_weight_tensors_[index].to(torch::kInt8).to(device_);
  };
  convert_to_int8(IN_Q_PROJ_A_OFFSET);
  convert_to_int8(IN_Q_PROJ_B_OFFSET);
  convert_to_int8(IN_KV_PROJ_WITH_MQA_OFFSET);
  convert_to_int8(IN_ATTENTION_OUT_OFFSET);
}

void DeekseekV32DecoderLoader::handle_device_specific_bias() {
  if (dp_local_tp_rank_ != 0) {
    torch::Tensor original_tensor = at_weight_tensors_[IN_ATTENTION_OUT_BIAS];
    at_weight_tensors_[IN_ATTENTION_OUT_BIAS] =
        torch::zeros(original_tensor.sizes(),
                     torch::TensorOptions()
                         .dtype(original_tensor.dtype())
                         .device(original_tensor.device()));
  }
}

std::string DeekseekV32DecoderLoader::extract_endswith(
    const std::string& input) {
  std::vector<std::string> parts;
  std::stringstream ss(input);
  std::string part;
  while (std::getline(ss, part, '.')) {
    parts.emplace_back(part);
  }
  if (parts.size() < 2) {
    return "";
  }
  std::string result = parts[parts.size() - 2] + "." + parts[parts.size() - 1];
  return result;
}

torch::Tensor DeekseekV32DecoderLoader::get_sharded_tensor(
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

torch::Tensor DeekseekV32DecoderLoader::get_sharded_tensor(
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

int DeekseekV32DecoderLoader::get_mapped_index(
    const std::string& name,
    const std::unordered_map<std::string, int>& mapping) {
  const auto it = mapping.find(name);
  if (it == mapping.end()) {
    LOG(WARNING) << "Parameter '" << name
                 << "' not found in mapping and will not be used.";
    return -1;
  }
  return it->second;
}

void DeekseekV32DecoderLoader::squeeze_experts_weights() {
  for (const auto& index : SQUEEZE_WEIGHT_VEC) {
    if (at_weight_tensors_[index].dim() > 1) {
      at_weight_tensors_[index] = at_weight_tensors_[index].squeeze();
    }
  }
}

void DeekseekV32DecoderLoader::process_general_weights(
    const StateDict& state_dict,
    const std::string& name,
    const torch::Tensor& tensor) {
  const int index = get_mapped_index(name, WEIGHT_MAPPING_W8A8);
  if (index == -1) {
    return;
  }
  const bool is_sharded = WEIGHT_SHARD_W8A8.count(index);
  torch::Tensor tmp_tensor;

  tmp_tensor = is_sharded ? get_sharded_tensor(state_dict,
                                               name,
                                               WEIGHT_SHARD_W8A8.at(index),
                                               dp_local_tp_rank_,
                                               dp_local_tp_size_)
                                .to(device_)
                          : tensor.to(device_);

  correct_tensor_dtype(tmp_tensor, name);
  at_weight_tensors_[index] = tmp_tensor;
  if (absl::StartsWith(name, "self_attn.q_a_proj")) {
    const int index_re = get_mapped_index(name, WEIGHT_MAPPING_W8A8_RECOMPUTE);
    torch::Tensor tmp_tensor_re = tensor.to(device_);
    at_weight_tensors_[index_re] = tmp_tensor_re;
  }
  if (layer_id_ != 61 && absl::StrContains(name, "layernorm.weight")) {
    at_weight_tensors_[index + 1] = torch::zeros_like(tmp_tensor);
  }
}

void DeekseekV32DecoderLoader::process_mlp_common_weights(
    const StateDict& state_dict,
    const std::string& name,
    const torch::Tensor& tensor) {
  const int index = get_mapped_index(name, WEIGHT_MAPPING_W8A8);
  if (index == -1) {
    return;
  }
  const bool is_sharded = WEIGHT_SHARD_W8A8.count(index);
  std::lock_guard<std::mutex> lock(shared_experts_mutex_);

  torch::Tensor tmp_tensor =
      is_sharded ? get_sharded_tensor(state_dict,
                                      name,
                                      WEIGHT_SHARD_W8A8.at(index),
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

void DeekseekV32DecoderLoader::merge_experts_weights() {
  torch::Tensor mlp_gateup_weight =
      merge_experts_weights(experts_weights_["gate_proj.weight"],
                            experts_weights_["up_proj.weight"],
                            device_,
                            /*transpose=*/true);
  at_weight_tensors_[IN_MLP_GATEUP_WEIGHT_EXPERT] =
      at_npu::native::npu_format_cast(mlp_gateup_weight, 29);
  // at_weight_tensors_[IN_MLP_GATEUP_WEIGHT_EXPERT] =
  //     at_npu::native::npu_format_cast(mlp_gateup_weight, 2).contiguous();
  if (quantize_type_ == "w8a8_dynamic") {
    at_weight_tensors_[IN_MLP_GATEUP_OFFSET_EXPERT] =
        merge_experts_weights(experts_weights_["gate_proj.weight_offset"],
                              experts_weights_["up_proj.weight_offset"],
                              device_);
    at_weight_tensors_[IN_MLP_GATEUP_SCALE_EXPERT] =
        merge_experts_weights(experts_weights_["gate_proj.weight_scale"],
                              experts_weights_["up_proj.weight_scale"],
                              device_);
  }

#if defined(USE_A3)
  torch::Tensor mlp_down_weight =
      merge_experts_weights(experts_weights_["down_proj.weight"],
                            device_,
                            /*transpose=*/false);
  // at_weight_tensors_[IN_MLP_DOWN_WEIGHT_EXPERT] =
  //     at_npu::native::npu_format_cast(mlp_down_weight, 29);
  at_weight_tensors_[IN_MLP_DOWN_WEIGHT_EXPERT] =
      at_npu::native::npu_format_cast(mlp_down_weight, 2).contiguous();
#else
  // TODO: xllm ops's GMM need to support MTP.
  if (decode_isBF16_ && false) {
    torch::Tensor mlp_down_weight =
        merge_experts_weights(experts_weights_["down_proj.weight"],
                              device_,
                              /*transpose=*/true);
    at_weight_tensors_[IN_MLP_DOWN_WEIGHT_EXPERT] =
        at_npu::native::npu_format_cast(mlp_down_weight, 29);
  } else {
    torch::Tensor mlp_down_weight =
        merge_experts_weights(experts_weights_["down_proj.weight"],
                              device_,
                              /*transpose=*/false);
    at_weight_tensors_[IN_MLP_DOWN_WEIGHT_EXPERT] =
        at_npu::native::npu_format_cast(mlp_down_weight, 2).contiguous();
  }
#endif
  if (quantize_type_ == "w8a8_dynamic") {
    at_weight_tensors_[IN_MLP_DOWN_OFFSET_EXPERT] = merge_experts_weights(
        experts_weights_["down_proj.weight_offset"], device_);
    at_weight_tensors_[IN_MLP_DOWN_SCALE_EXPERT] = merge_experts_weights(
        experts_weights_["down_proj.weight_scale"], device_);
  }
}

torch::Tensor DeekseekV32DecoderLoader::merge_experts_weights(
    std::vector<torch::Tensor>& experts,
    at::Device device,
    bool transpose) {
  torch::Tensor merged_tensor = torch::stack(experts, 0).to(device);
  if (transpose) {
    merged_tensor = merged_tensor.transpose(1, 2);
  }
  merged_tensor = merged_tensor.contiguous();
  experts.clear();
  return merged_tensor;
}

torch::Tensor DeekseekV32DecoderLoader::merge_experts_weights(
    std::vector<torch::Tensor>& experts_gate,
    std::vector<torch::Tensor>& experts_up,
    at::Device device,
    bool transpose) {
  for (size_t i = 0; i < experts_up.size(); ++i) {
    experts_gate[i] = torch::cat({experts_gate[i], experts_up[i]}, 0);
  }

  torch::Tensor merged_tensor = torch::stack(experts_gate, 0).to(device);

  if (transpose) {
    merged_tensor = merged_tensor.transpose(1, 2);
  }

  merged_tensor = merged_tensor.contiguous();
  experts_gate.clear();
  experts_up.clear();
  return merged_tensor;
}

void DeekseekV32DecoderLoader::process_shared_expert_weights(
    const StateDict& state_dict,
    const std::string& name,
    const torch::Tensor& tensor) {
  torch::Tensor tmp_tensor;
  std::lock_guard<std::mutex> lock(shared_experts_mutex_);
  const int index = get_mapped_index(name, WEIGHT_MAPPING_W8A8);
  if (index == -1) {
    return;
  }
  if (FLAGS_expert_parallel_degree == 2) {
    tmp_tensor = tensor.to(device_);
  } else {
    const bool is_sharded = WEIGHT_SHARD_W8A8.count(index);
    tmp_tensor = is_sharded ? get_sharded_tensor(
                                  state_dict, name, WEIGHT_SHARD_W8A8.at(index))
                                  .to(device_)
                            : tensor.to(device_);
  }
  if (absl::StrContains(name, "down_proj")) {
    at_weight_tensors_[index] = tmp_tensor;
  } else {
    shared_experts_weights_[name] = tmp_tensor;
  }
}

void DeekseekV32DecoderLoader::set_kv_weight(const StateDict& state_dict,
                                             const std::string& tensor_name,
                                             int weight_position,
                                             int dim) {
  torch::Tensor mutable_tensor;
  if (parallel_args_.world_size() <= 1) {
    mutable_tensor = state_dict.get_tensor(tensor_name).to(device_);
    correct_tensor_dtype(mutable_tensor, tensor_name);
  } else {
    mutable_tensor =
        get_sharded_tensor(
            state_dict, tensor_name, dim, dp_local_tp_rank_, dp_local_tp_size_)
            .to(device_);
    // mutable_tensor = get_sharded_tensor(state_dict, tensor_name, dim);
    correct_tensor_dtype(mutable_tensor, tensor_name);
  }

  torch::Tensor kv_b_proj_weight =
      mutable_tensor.reshape({num_key_value_heads_ / dp_local_tp_size_,
                              qk_nope_head_dim_ + v_head_dim_,
                              kv_lora_rank_});
  torch::Tensor k_b_proj_preprocessed =
      kv_b_proj_weight.slice(1, 0, qk_nope_head_dim_).contiguous();
  torch::Tensor v_b_proj_preprocessed =
      kv_b_proj_weight
          .slice(1, qk_nope_head_dim_, qk_nope_head_dim_ + v_head_dim_)
          .transpose(1, 2)
          .contiguous();
  at_weight_tensors_[weight_position] = k_b_proj_preprocessed.to(device_);
  at_weight_tensors_[weight_position + 6] = v_b_proj_preprocessed.to(device_);
}

void DeekseekV32DecoderLoader::preprocess_linear_for_rope() {
  for (const auto& name : LINEAR_FOR_ROPE) {
    if (quantize_type_ == "") {
      if (!absl::EndsWith(name, "weight")) {
        continue;
      }
    }
    int index = WEIGHT_MAPPING_W8A8.at(name);
    at_weight_tensors_[index] =
        view_tensor(at_weight_tensors_[index], name, true);
    at_weight_tensors_[index] = trans_rope_weight(at_weight_tensors_[index]);
    at_weight_tensors_[index] =
        (!absl::EndsWith(name, "weight"))
            ? view_tensor(at_weight_tensors_[index], name, false).flatten()
            : view_tensor(at_weight_tensors_[index], name, false);
  }
}

torch::Tensor DeekseekV32DecoderLoader::view_tensor(torch::Tensor weight,
                                                    const std::string& name,
                                                    bool pre_view) {
  if (absl::StrContains(name, "q_b_proj")) {
    if (pre_view) {
      return weight
          .view({prefill_numAttentionHeadsPerRank_,
                 qk_nope_head_dim_ + prefill_qkRopeHeadDim_,
                 -1})
          .contiguous();
    } else {
      return weight
          .view({prefill_numAttentionHeadsPerRank_ *
                     (qk_nope_head_dim_ + prefill_qkRopeHeadDim_),
                 -1})
          .contiguous();
    }
  } else if (absl::StrContains(name, "kv_a_proj_with_mqa")) {
    return weight.view({kv_lora_rank_ + prefill_qkRopeHeadDim_, -1})
        .contiguous();
  }
  return weight;
}

torch::Tensor DeekseekV32DecoderLoader::trans_rope_weight(
    torch::Tensor weight) {
  int64_t d = weight.size(-2);
  int64_t rope_dim = prefill_qkRopeHeadDim_;
  torch::Tensor weight_1 =
      weight.slice(-2, d - rope_dim, torch::indexing::None, 2).contiguous();

  torch::Tensor weight_2 =
      weight.slice(-2, d - rope_dim + 1, torch::indexing::None, 2).contiguous();

  torch::Tensor combined = torch::cat({weight_1, weight_2}, -2);

  weight.slice(-2, d - rope_dim, d).copy_(combined);

  return weight.contiguous();
}

void DeekseekV32DecoderLoader::initialize_device_expert_list(
    int num_device,
    int num_device_expert) {
  int32_t num_device_route_expert = num_device_expert;
  if (FLAGS_enable_eplb) {
    num_device_route_expert = num_device_expert - redundant_experts_num_;
  }
  for (int i = 0; i < num_device * num_device_route_expert; ++i) {
    device_expert_list_.emplace_back(i);
    if (FLAGS_enable_eplb && (i + 1) % num_device_route_expert == 0) {
      for (int redundant_expert = 0; redundant_expert < redundant_experts_num_;
           ++redundant_expert)
        device_expert_list_.emplace_back(i);
    }
  }
}

torch::Tensor DeekseekV32DecoderLoader::convert_fp16_to_int64(
    const torch::Tensor& fp16_tensor) {
  auto float_tensor = fp16_tensor.to(torch::kFloat32);
  auto int32_tensor = float_tensor.view(torch::kInt32);
  auto int64_tensor = int32_tensor.to(torch::kInt64);
  return int64_tensor;
}

void DeekseekV32DecoderLoader::convert_descaled_weights_to_float() {
  auto convert_to_float = [this](int index) {
    at_weight_tensors_[index] = at_weight_tensors_[index].to(torch::kFloat32);
  };
  convert_to_float(IN_Q_PROJ_A_DESCALE);
  convert_to_float(IN_Q_PROJ_B_DESCALE);
  convert_to_float(IN_KV_PROJ_WITH_MQA_DESCALE);
  convert_to_float(IN_ATTENTION_OUT_DESCALE);
}

void DeekseekV32DecoderLoader::reserve_experts_weights(
    int num_of_device_experts) {
  experts_weights_.clear();
  std::vector<std::string> weight_names = {
      "gate_proj.weight", "up_proj.weight", "down_proj.weight"};
  if (quantize_type_ == "w8a8_dynamic") {
    weight_names.emplace_back("gate_proj.weight_offset");
    weight_names.emplace_back("up_proj.weight_offset");
    weight_names.emplace_back("down_proj.weight_offset");
    weight_names.emplace_back("gate_proj.weight_scale");
    weight_names.emplace_back("up_proj.weight_scale");
    weight_names.emplace_back("down_proj.weight_scale");
  }
  std::lock_guard<std::mutex> lock(experts_mutex_);
  for (const auto& weight_name : weight_names) {
    experts_weights_[weight_name] =
        std::vector<torch::Tensor>(num_of_device_experts);
  }
}

std::string DeekseekV32DecoderLoader::get_expert_shm_key(
    int32_t layer_id,
    int32_t expert_index,
    const std::string& suffix) {
  std::string shm_key =
      "layer_" + std::to_string(layer_id - first_k_dense_replace_) + "_" +
      "expert_" + std::to_string(expert_index) + "_" + suffix;
  return shm_key;
}

void DeekseekV32DecoderLoader::merge_shared_experts_weights() {
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

  if (layer_id_ >= prefill_firstKDenseReplace_) {
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
    }
  }
}

void DeekseekV32DecoderLoader::merge_loaded_weights() {
  if (quantize_type_ == "w8a8_dynamic") {
    if (prefill_isBF16_) {
      convert_descaled_weights_to_float();
    }
    convert_offsets_to_int8();
    handle_device_specific_bias();
  }

  merge_shared_experts_weights();
  if (layer_id_ >= prefill_firstKDenseReplace_) {
    merge_experts_weights();
  }

  squeeze_experts_weights();

  preprocess_linear_for_rope();

  at_weight_tensors_[IN_Q_PROJ_A_WEIGHT] =
      torch::cat({at_weight_tensors_[IN_KV_PROJ_WITH_MQA_WEIGHT],
                  at_weight_tensors_[IN_Q_PROJ_A_WEIGHT]},
                 0)
          .contiguous();
  if (quantize_type_ == "w8a8_dynamic") {
    at_weight_tensors_[IN_Q_PROJ_A_BIAS] =
        torch::cat({at_weight_tensors_[IN_KV_PROJ_WITH_MQA_BIAS],
                    at_weight_tensors_[IN_Q_PROJ_A_BIAS]},
                   0)
            .contiguous();
    at_weight_tensors_[IN_Q_PROJ_A_DESCALE] =
        torch::cat({at_weight_tensors_[IN_KV_PROJ_WITH_MQA_DESCALE],
                    at_weight_tensors_[IN_Q_PROJ_A_DESCALE]},
                   0)
            .contiguous();
  }

  at_weight_tensors_[IN_Q_PROJ_A_WEIGHT] = at_npu::native::npu_format_cast(
      at_weight_tensors_[IN_Q_PROJ_A_WEIGHT], 29);
  at_weight_tensors_[IN_Q_PROJ_A_RECOMPUTE_WEIGHT] =
      at_npu::native::npu_format_cast(
          at_weight_tensors_[IN_Q_PROJ_A_RECOMPUTE_WEIGHT], 29);
  at_weight_tensors_[IN_Q_PROJ_B_WEIGHT] = at_npu::native::npu_format_cast(
      at_weight_tensors_[IN_Q_PROJ_B_WEIGHT], 29);

  at_weight_tensors_[IN_KV_PROJ_WITH_MQA_WEIGHT] = tensor_placeholder_;
  at_weight_tensors_[IN_KV_PROJ_WITH_MQA_BIAS] = tensor_placeholder_;
  at_weight_tensors_[IN_KV_PROJ_WITH_MQA_DESCALE] = tensor_placeholder_;
  at_weight_tensors_[IN_KV_PROJ_WITH_MQA_OFFSET] = tensor_placeholder_;
  at_weight_tensors_[IN_KV_PROJ_WITH_MQA_SCALE] = tensor_placeholder_;
  if (FLAGS_expert_parallel_degree != 2) {
    at_weight_tensors_[IN_BLOCK_SPARSE_MOE_GATE_WEIGHT] =
        torch::roll(at_weight_tensors_[IN_BLOCK_SPARSE_MOE_GATE_WEIGHT],
                    {-1 * ep_rank_ * num_experts_per_partition_},
                    {0})
            .contiguous();
    at_weight_tensors_[IN_BLOCK_SPARSE_MOE_GATE_BIAS] =
        torch::roll(at_weight_tensors_[IN_BLOCK_SPARSE_MOE_GATE_BIAS],
                    {-1 * ep_rank_ * num_experts_per_partition_},
                    {0})
            .contiguous();
  }
  // at_weight_tensors_[IN_MLP_DOWN_WEIGHT_SHARED_EXPERT] =
  // at_weight_tensors_[IN_MLP_DOWN_WEIGHT_SHARED_EXPERT].transpose(0, 1);
  at_weight_tensors_[IN_BLOCK_SPARSE_MOE_GATE_WEIGHT] =
      at_weight_tensors_[IN_BLOCK_SPARSE_MOE_GATE_WEIGHT].to(torch::kFloat32);
  at_weight_tensors_[IN_MLP_GATEUP_SCALE_EXPERT] =
      at_weight_tensors_[IN_MLP_GATEUP_SCALE_EXPERT].to(torch::kBFloat16);
  at_weight_tensors_[IN_MLP_GATEUP_SCALE_SHARED_EXPERT] =
      at_weight_tensors_[IN_MLP_GATEUP_SCALE_SHARED_EXPERT].to(
          torch::kBFloat16);
  at_weight_tensors_[IN_MLP_DOWN_SCALE_EXPERT] =
      at_weight_tensors_[IN_MLP_DOWN_SCALE_EXPERT].to(torch::kBFloat16);
  at_weight_tensors_[IN_MLP_DOWN_SCALE_SHARED_EXPERT] =
      at_weight_tensors_[IN_MLP_DOWN_SCALE_SHARED_EXPERT].to(torch::kBFloat16);
  if (quantize_type_ == "w8a8_dynamic") {
    // at_weight_tensors_[IN_BLOCK_SPARSE_MOE_GATE_WEIGHT] =
    //     at_weight_tensors_[IN_BLOCK_SPARSE_MOE_GATE_WEIGHT].to(torch::kFloat32);
    if (!prefill_isBF16_) {
      at_weight_tensors_[IN_Q_PROJ_A_DESCALE] =
          convert_fp16_to_int64(at_weight_tensors_[IN_Q_PROJ_A_DESCALE]);
      at_weight_tensors_[IN_Q_PROJ_A_RECOMPUTE_DESCALE] = convert_fp16_to_int64(
          at_weight_tensors_[IN_Q_PROJ_A_RECOMPUTE_DESCALE]);
      at_weight_tensors_[IN_Q_PROJ_B_DESCALE] =
          convert_fp16_to_int64(at_weight_tensors_[IN_Q_PROJ_B_DESCALE]);
      at_weight_tensors_[IN_ATTENTION_OUT_DESCALE] =
          convert_fp16_to_int64(at_weight_tensors_[IN_ATTENTION_OUT_DESCALE]);

      at_weight_tensors_[IN_MLP_GATEUP_OFFSET_SHARED_EXPERT] =
          at_weight_tensors_[IN_MLP_GATEUP_OFFSET_SHARED_EXPERT].to(
              torch::kFloat16);
      at_weight_tensors_[IN_MLP_GATEUP_SCALE_SHARED_EXPERT] =
          at_weight_tensors_[IN_MLP_GATEUP_SCALE_SHARED_EXPERT].to(
              torch::kFloat32);
      at_weight_tensors_[IN_MLP_DOWN_SCALE_SHARED_EXPERT] =
          at_weight_tensors_[IN_MLP_DOWN_SCALE_SHARED_EXPERT].to(
              torch::kFloat32);
      at_weight_tensors_[IN_MLP_GATEUP_OFFSET_EXPERT] =
          at_weight_tensors_[IN_MLP_GATEUP_OFFSET_EXPERT].to(torch::kFloat16);
      at_weight_tensors_[IN_MLP_DOWN_OFFSET_EXPERT] =
          at_weight_tensors_[IN_MLP_DOWN_OFFSET_EXPERT].to(torch::kFloat16);
      at_weight_tensors_[IN_MLP_DOWN_SCALE_EXPERT] =
          at_weight_tensors_[IN_MLP_DOWN_SCALE_EXPERT].to(torch::kFloat32);
    }
  }
}

}  // namespace layer
}  // namespace xllm

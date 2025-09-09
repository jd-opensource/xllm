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

#include "deepseek_v2_decoder_layer.h"

#include <gflags/gflags.h>

#include <utility>

#include "common/global_flags.h"

DECLARE_string(rank_tablefile);
DECLARE_string(communication_backend);
DECLARE_int32(expert_parallel_degree);
DECLARE_bool(enable_continuous_kvcache);

namespace xllm::hf {

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
};

static const uint64_t WEIGHT_COUNT_PER_LAYER = 84;

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

DeepseekV2DecoderImpl::DeepseekV2DecoderImpl(const Context& context,
                                             const int32_t layer_id,
                                             const float sm_scale)
    : ATBBase(context),
      device_id_(context.get_tensor_options().device().index()),
      layer_id_(layer_id),
      sm_scale_(sm_scale),
      num_speculative_tokens_(
          context.get_model_args().num_speculative_tokens()) {
  auto parallel_args = context.get_parallel_args();
  auto model_args = context.get_model_args();
  auto options = context.get_tensor_options();
  rank_ = parallel_args.rank();
  first_k_dense_replace_ = model_args.first_k_dense_replace();
  n_layers_ = model_args.n_layers();
  num_experts_ = model_args.n_routed_experts();
  localWorldSize_ = parallel_args.mapping().localWorldSize();
  ep_size_ = parallel_args.ep_size();
  ep_local_tp_size_ = parallel_args.world_size() / ep_size_;
  CHECK_EQ(parallel_args.world_size(), ep_size_ * ep_local_tp_size_);
  ep_local_tp_rank_ = parallel_args.rank() % ep_local_tp_size_;
  num_experts_per_partition_ = model_args.n_routed_experts() / ep_size_;
  redundant_experts_num_ = FLAGS_redundant_experts_num;
  if (FLAGS_enable_eplb) {
    num_experts_per_partition_ += redundant_experts_num_;
  }
  ep_rank_ = parallel_args.rank() / ep_local_tp_size_;
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

void DeepseekV2DecoderImpl::initialize_tensors(
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

  reserve_experts_weights(prefill_param_.numOfDeviceExperts);
  expert_group_ = torch::arange(1024, torch::kInt32).to(device_);
  one_hot_ = torch::tensor({1}, torch::kInt32).to(device_);
  zero_hot_ = torch::tensor({0}, torch::kInt32).to(device_);
  at_start_expert_id_ =
      torch::tensor({start_expert_id_}, torch::kInt64).to(device_);
  at_in_device_expert_count_ =
      torch::tensor({num_experts_per_partition_ - 1}, torch::kInt64)
          .to(device_);
  initialize_weight_tensors(options);
  initialize_device_expert_list(decode_param_.worldSize,
                                num_experts_per_partition_);
  if (FLAGS_enable_eplb) {
    auto layer_expert_routing_map_ =
        build_expert_routing_map(device_expert_list_);
    std::vector<torch::Tensor> tensors_vec;
    for (int i = 0; i < n_layers_ - first_k_dense_replace_; i++) {
      tensors_vec.emplace_back(layer_expert_routing_map_);
    }
    expert_routing_map_ = torch::stack(tensors_vec, 0);
  }
}

void DeepseekV2DecoderImpl::initialize_device_expert_list(
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

void DeepseekV2DecoderImpl::param_from_args(
    atb_speed::deepseekV2::DecoderLayerParam& param,
    const ModelArgs& args,
    const ParallelArgs& parallel_args,
    bool is_prefill) {
  initialize_basic_parameters(param, args, parallel_args, is_prefill);
  initialize_attention_parameters(param, args, parallel_args);
  initialize_mlp_parameters(param, args, parallel_args);
  initialize_parallel_parameters(param, parallel_args);
  initialize_quantization_parameters(param);
  initialize_kimi_k2_parameters(param, args, is_prefill);
}

void DeepseekV2DecoderImpl::reserve_experts_weights(int num_of_device_experts) {
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
    ;
  }
}

void DeepseekV2DecoderImpl::initialize_weight_tensors(
    const torch::TensorOptions& options) {
  for (int i = 0; i < WEIGHT_COUNT_PER_LAYER; ++i) {
    at_weight_tensors_[i] = torch::zeros({1}).to(options);
  }
  if (FLAGS_enable_eplb) {
    const int64_t size =
        50LL * 1024LL * 1024LL * int64_t(n_layers_ - first_k_dense_replace_);
    shared_buffer_ = std::make_unique<ExpertBufferManager>(
        num_experts_, n_layers_ - first_k_dense_replace_, size);
  }
}

void DeepseekV2DecoderImpl::initialize_basic_parameters(
    atb_speed::deepseekV2::DecoderLayerParam& param,
    const ModelArgs& args,
    const ParallelArgs& parallel_args,
    bool is_prefill) {
  param.isFA = false;
  param.isPrefill = is_prefill;
  param.isBF16 = args.dtype() == "bfloat16";
  param.enableSwiGLU = true;
  param.enableLcoc = false;
  // TODO: modify xllm_atb_layers
  // param.enableContinuousKvCache = FLAGS_enable_continuous_kvcache;

  param.attnLinearTransposeType = {1, 1, 1, 1, 1, 1};
  param.mlpLinearTransposeType = {1, -1, 1, -1};

  param.moeLinearTransposeType = (layer_id_ < args.first_k_dense_replace())
                                     ? std::vector<int>{-1, -1, -1, -1}
                                     : std::vector<int>{1, 0, -1, 1};

  param.worldSize = parallel_args.world_size();
  param.normEps = args.rms_norm_eps();
  param.numAttentionHeadsPerRank = args.n_heads() / dp_local_tp_size_;
  param.hiddenSizePerAttentionHead = args.hidden_size() / args.n_heads();
  std::optional<long int> optionalValue = args.n_kv_heads();
  param.numKeyValueHeadsPerRank = 1;
  // static_cast<int>(optionalValue.value()) / param.worldSize;
  param.rank = parallel_args.rank();
  param.backend = FLAGS_communication_backend;
  param.rankTableFile = FLAGS_rank_tablefile;

  param.layerId = layer_id_;
  param.numHiddenLayers = args.n_layers();
  param.enableIntraLayerAddNorm = false;
  param.enableInterLayerAddNorm = false;
  if (quantize_type_ == "") {
    param.enableGMMSwigluQuant = false;
  } else {
    param.enableGMMSwigluQuant =
        (is_prefill && parallel_args.world_size() > 16) || !is_prefill;
  }
  param.enableDpOut = false;  // TODO
  if (num_speculative_tokens_ == 0) {
    param.enableSpeculate = false;  // MTP
  } else {
    param.enableSpeculate = true;
  }
  param.maskfree = true;                            // TODO
  param.enableSwiGLUQuantForSharedExperts = false;  // TODO
#if defined(USE_A3)
  param.scaledTopk = -1;
  param.enableATBGateMatmul = 1;
  param.enableLcocAll2All = 1;
#endif
  num_key_value_heads_ = static_cast<int>(args.n_kv_heads().value());
  qk_nope_head_dim_ = args.qk_nope_head_dim();
  v_head_dim_ = args.v_head_dim();
  kv_lora_rank_ = args.kv_lora_rank();
  qk_rope_head_dim_ = args.qk_rope_head_dim();
}

void DeepseekV2DecoderImpl::initialize_attention_parameters(
    atb_speed::deepseekV2::DecoderLayerParam& param,
    const ModelArgs& args,
    const ParallelArgs& parallel_args) {
  param.qLoraRank = args.q_lora_rank();
  // NOTE: The operation in this conditional is theoretically compatible with
  // DeepSeek, but we add this specific check to ensure DeepSeek behavior
  // remains unchanged
  if (args.model_type() != "kimi_k2") {
    param.headNum = args.n_heads();
  }
  param.qkNopeHeadDim = args.qk_nope_head_dim();
  param.qkRopeHeadDim = args.qk_rope_head_dim();
  param.kvLoraRank = args.kv_lora_rank();
  param.softmaxScale = sm_scale_;
  if (quantize_type_ == "w8a8_dynamic" && num_speculative_tokens_ == 0) {
    param.enableMlaPreprocess = param.isBF16 ? false : true;
  } else {
    param.enableMlaPreprocess = false;
  }

  param.enableFA3 = false;           // TODO
  param.isNzCache = false;           // TODO
  param.enableKvQuantLayer = false;  // TODO
}

void DeepseekV2DecoderImpl::initialize_mlp_parameters(
    atb_speed::deepseekV2::DecoderLayerParam& param,
    const ModelArgs& args,
    const ParallelArgs& parallel_args) {
  param.hasSharedExpert = (args.n_shared_experts() > 0);
  param.hasSharedExpertGate = false;
  param.processLogits = "normScaling";
  param.routedScalingFactor = args.routed_scaling_factor();
  param.numOfSelectedExperts = {args.num_experts_per_tok()};

  if (ep_size_ > 1) {
    param.expertParallelDegree = std::max(FLAGS_expert_parallel_degree, 1);
  } else {
    param.expertParallelDegree = 0;
  }

  param.deviceExpert.resize(num_experts_per_partition_);
  // param.deviceExpert.resize(args.n_routed_experts());
  std::iota(
      param.deviceExpert.begin(), param.deviceExpert.end(), start_expert_id_);
  param.numOfExperts = args.n_routed_experts();
  param.numOfDeviceExperts = num_experts_per_partition_;
  param.maskStartIdx = 0;
  param.firstKDenseReplace = args.first_k_dense_replace();
  // param.numOfSharedExperts = args.n_shared_experts();
  param.numOfSharedExperts = 2;
  param.routingMethod = "noAuxTc";
  param.numOfGroups = args.n_group();
  param.topkGroups = atb::SVector<int>{args.topk_group()};
  param.isDynamicEp = param.expertParallelDegree == 2 ? true : false;

  param.quantGroupSize = 0;
  if (quantize_type_ == "") {
    param.enableInitQuant = false;
    param.enableSwigluQuant = false;
  } else {
    param.enableInitQuant = true;
    param.enableSwigluQuant = param.isPrefill && !param.enableGMMSwigluQuant;
  }
  param.enableFusedTopk = true;

  param.enableCVOverlap = false;           // TODO
  param.enableExpertCumSumOutput = false;  // TODO
  param.enableLoadBalance = false;         // TODO
  param.enableEPWB = false;                // TODO
  param.numOfRedundantExpert = 0;          // TODO
  param.enableInfNan = param.isPrefill;    // TODO

  param.dispatchAndCombineHcclComm = parallel_args.dispatchAndCombineHcclComm();
  param.dispatchAndCombinecommDomain =
      parallel_args.dispatchAndCombinecommDomain();

  if (layer_id_ >= param.firstKDenseReplace) {
    param.enableQkvdownDp = false;
    param.enableSharedExpertDp = false;
    param.enableGatingDp = false;
    if (FLAGS_enable_eplb) {
      param.enableExpertCumSumOutput = param.isPrefill ? false : true;
      param.enableEPWB = true;
      param.numOfRedundantExpert = ep_size_ * redundant_experts_num_;
    }
  }
  if (layer_id_ < param.firstKDenseReplace) {
    param.isDenseLayer = true;
  }
}

void DeepseekV2DecoderImpl::initialize_kimi_k2_parameters(
    atb_speed::deepseekV2::DecoderLayerParam& param,
    const ModelArgs& args,
    bool is_prefill) {
  if (args.model_type() != "kimi_k2") {
    return;
  }
  // NOTE: These operations are theoretically applicable to DeepSeek as well,
  // but we only apply them to kimi_k2 to ensure DeepSeek behavior remains
  // unchanged
  param.enableInfNan = true;
  param.enableFusedTopk = (args.topk_method() == "noaux_tc" &&
                           args.n_group() * 32 >= args.n_routed_experts());
  param.maskfree = is_prefill;
  // TODO: Pending confirmation whether kimi_k2 model supports
  // enable_gmmswigluquant set to true
  bool enable_gmmswigluquant = false;
  param.enableSwigluQuant =
      quantize_type_ == "w8a8_dynamic" && !enable_gmmswigluquant;
  param.enableGMMSwigluQuant = enable_gmmswigluquant;
}

void DeepseekV2DecoderImpl::initialize_parallel_parameters(
    atb_speed::deepseekV2::DecoderLayerParam& param,
    const ParallelArgs& parallel_args) {
  param.lmHeadLocalTp = dp_local_tp_size_;
  param.enableSharedExpertOverlap = false;  // TODO

  param.enableAllToAllMC2 = (param.expertParallelDegree == 2);
  param.enableGatherPreNorm = true;
  param.enableExtraOprojTp = false;  // TODO
  param.isMlpFullTP = false;         // TODO
  param.mapping = parallel_args.mapping();
  param.maxDecodeDpTokenSize = 0;  // TODO
}

void DeepseekV2DecoderImpl::initialize_quantization_parameters(
    atb_speed::deepseekV2::DecoderLayerParam& param) {
  if (quantize_type_ == "") {
    param.moePackQuantType = static_cast<int>(PackType::ALL_FP);
    param.packQuantType = {static_cast<int>(PackType::ALL_FP),
                           static_cast<int>(PackType::ALL_FP)};
    param.attnLinearQuantType = {static_cast<int>(LinearType::FP),
                                 static_cast<int>(LinearType::FP),
                                 static_cast<int>(LinearType::FP),
                                 static_cast<int>(LinearType::FP),
                                 static_cast<int>(LinearType::FP),
                                 static_cast<int>(LinearType::FP)};
    param.mlpLinearQuantType = {static_cast<int>(LinearType::FP),
                                static_cast<int>(LinearType::INVALID),
                                static_cast<int>(LinearType::FP),
                                static_cast<int>(LinearType::INVALID)};
    if (layer_id_ < param.firstKDenseReplace) {
      param.moeLinearQuantType = {static_cast<int>(LinearType::INVALID),
                                  static_cast<int>(LinearType::INVALID),
                                  static_cast<int>(LinearType::INVALID),
                                  static_cast<int>(LinearType::INVALID)};
    } else {
      param.moeLinearQuantType = {static_cast<int>(LinearType::FP),
                                  static_cast<int>(LinearType::FP),
                                  static_cast<int>(LinearType::INVALID),
                                  static_cast<int>(LinearType::FP)};
    }
  } else {
    param.moePackQuantType = static_cast<int>(PackType::ALL_W8A8_DYNAMIC);
    param.packQuantType = {static_cast<int>(PackType::MIX_W8A8),
                           static_cast<int>(PackType::ALL_W8A8_DYNAMIC)};
    param.attnLinearQuantType = {static_cast<int>(LinearType::INT),
                                 static_cast<int>(LinearType::INT),
                                 static_cast<int>(LinearType::FP),
                                 static_cast<int>(LinearType::FP),
                                 static_cast<int>(LinearType::FP),
                                 static_cast<int>(LinearType::INT)};
    param.mlpLinearQuantType = {static_cast<int>(LinearType::INT),
                                static_cast<int>(LinearType::INVALID),
                                static_cast<int>(LinearType::INT),
                                static_cast<int>(LinearType::INVALID)};
    if (layer_id_ < param.firstKDenseReplace) {
      param.moeLinearQuantType = {static_cast<int>(LinearType::INVALID),
                                  static_cast<int>(LinearType::INVALID),
                                  static_cast<int>(LinearType::INVALID),
                                  static_cast<int>(LinearType::INVALID)};
    } else {
      param.moeLinearQuantType = {static_cast<int>(LinearType::FP),
                                  static_cast<int>(LinearType::INT),
                                  static_cast<int>(LinearType::INVALID),
                                  static_cast<int>(LinearType::INT)};
    }
  }
}

void DeepseekV2DecoderImpl::load_state_dict(const StateDict& state_dict) {
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

int DeepseekV2DecoderImpl::get_mapped_index(
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

std::string DeepseekV2DecoderImpl::get_expert_shm_key(int32_t layer_id,
                                                      int32_t expert_index,
                                                      std::string suffix) {
  std::string shm_key =
      "layer_" + std::to_string(layer_id - first_k_dense_replace_) + "_" +
      "expert_" + std::to_string(expert_index) + "_" + suffix;
  return shm_key;
}

void DeepseekV2DecoderImpl::process_expert_weights(
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

    if (!decode_param_.isBF16) {
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

void DeepseekV2DecoderImpl::process_shared_expert_weights(
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

void DeepseekV2DecoderImpl::process_mlp_common_weights(
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

void DeepseekV2DecoderImpl::process_general_weights(
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
}

void DeepseekV2DecoderImpl::set_kv_weight(const StateDict& state_dict,
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

void DeepseekV2DecoderImpl::preprocess_linear_for_rope() {
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

torch::Tensor DeepseekV2DecoderImpl::view_tensor(torch::Tensor weight,
                                                 const std::string& name,
                                                 bool pre_view) {
  if (absl::StrContains(name, "q_b_proj")) {
    if (pre_view) {
      return weight
          .view({prefill_param_.numAttentionHeadsPerRank,
                 qk_nope_head_dim_ + prefill_param_.qkRopeHeadDim,
                 -1})
          .contiguous();
    } else {
      return weight
          .view({prefill_param_.numAttentionHeadsPerRank *
                     (qk_nope_head_dim_ + prefill_param_.qkRopeHeadDim),
                 -1})
          .contiguous();
    }
  } else if (absl::StrContains(name, "kv_a_proj_with_mqa")) {
    return weight.view({kv_lora_rank_ + prefill_param_.qkRopeHeadDim, -1})
        .contiguous();
  }
  return weight;
}

torch::Tensor DeepseekV2DecoderImpl::trans_rope_weight(torch::Tensor weight) {
  int64_t d = weight.size(-2);
  int64_t rope_dim = prefill_param_.qkRopeHeadDim;
  torch::Tensor weight_1 =
      weight.slice(-2, d - rope_dim, torch::indexing::None, 2).contiguous();

  torch::Tensor weight_2 =
      weight.slice(-2, d - rope_dim + 1, torch::indexing::None, 2).contiguous();

  torch::Tensor combined = torch::cat({weight_1, weight_2}, -2);

  weight.slice(-2, d - rope_dim, d).copy_(combined);

  return weight.contiguous();
}

torch::Tensor DeepseekV2DecoderImpl::get_sharded_tensor(
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

torch::Tensor DeepseekV2DecoderImpl::get_sharded_tensor(
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

std::string DeepseekV2DecoderImpl::extract_endswith(const std::string& input) {
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

int DeepseekV2DecoderImpl::extract_expert_index(const std::string& name) {
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

void DeepseekV2DecoderImpl::verify_loaded_weights(
    const std::string& prefix) const {
  for (const auto& [index, name] : WEIGHT_MAPPING) {
    CHECK(at_weight_tensors_[index].sizes() != std::vector<int64_t>({1}))
        << "weight is not loaded for " << prefix + name;
  }
}

void DeepseekV2DecoderImpl::merge_loaded_weights() {
  if (quantize_type_ == "w8a8_dynamic") {
    if (prefill_param_.isBF16) {
      convert_descaled_weights_to_float();
    }
    convert_offsets_to_int8();
    handle_device_specific_bias();
  }

  merge_shared_experts_weights();
  if (layer_id_ >= prefill_param_.firstKDenseReplace) {
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
  if (quantize_type_ == "w8a8_dynamic") {
    at_weight_tensors_[IN_BLOCK_SPARSE_MOE_GATE_WEIGHT] =
        at_weight_tensors_[IN_BLOCK_SPARSE_MOE_GATE_WEIGHT].to(torch::kFloat32);
    if (!prefill_param_.isBF16) {
      at_weight_tensors_[IN_Q_PROJ_A_DESCALE] =
          convert_fp16_to_int64(at_weight_tensors_[IN_Q_PROJ_A_DESCALE]);
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
      at_weight_tensors_[IN_MLP_GATEUP_SCALE_EXPERT] =
          at_weight_tensors_[IN_MLP_GATEUP_SCALE_EXPERT].to(torch::kFloat32);
      at_weight_tensors_[IN_MLP_DOWN_OFFSET_EXPERT] =
          at_weight_tensors_[IN_MLP_DOWN_OFFSET_EXPERT].to(torch::kFloat16);
      at_weight_tensors_[IN_MLP_DOWN_SCALE_EXPERT] =
          at_weight_tensors_[IN_MLP_DOWN_SCALE_EXPERT].to(torch::kFloat32);
    }
  }
  c10_npu::NPUCachingAllocator::emptyCache();
  for (int i = 0; i < WEIGHT_COUNT_PER_LAYER; ++i) {
    atb_weight_tensors_[i] =
        atb_speed::Utils::AtTensor2Tensor(at_weight_tensors_[i]);
  }
  init_layer();
}

torch::Tensor DeepseekV2DecoderImpl::convert_fp16_to_int64(
    const torch::Tensor& fp16_tensor) {
  auto float_tensor = fp16_tensor.to(torch::kFloat32);
  auto int32_tensor = float_tensor.view(torch::kInt32);
  auto int64_tensor = int32_tensor.to(torch::kInt64);
  return int64_tensor;
}

void DeepseekV2DecoderImpl::convert_descaled_weights_to_float() {
  auto convert_to_float = [this](int index) {
    at_weight_tensors_[index] = at_weight_tensors_[index].to(torch::kFloat32);
  };
  convert_to_float(IN_Q_PROJ_A_DESCALE);
  convert_to_float(IN_Q_PROJ_B_DESCALE);
  convert_to_float(IN_KV_PROJ_WITH_MQA_DESCALE);
  convert_to_float(IN_ATTENTION_OUT_DESCALE);
}

void DeepseekV2DecoderImpl::convert_offsets_to_int8() {
  auto convert_to_int8 = [this](int index) {
    at_weight_tensors_[index] =
        at_weight_tensors_[index].to(torch::kInt8).to(device_);
  };
  convert_to_int8(IN_Q_PROJ_A_OFFSET);
  convert_to_int8(IN_Q_PROJ_B_OFFSET);
  convert_to_int8(IN_KV_PROJ_WITH_MQA_OFFSET);
  convert_to_int8(IN_ATTENTION_OUT_OFFSET);
}

void DeepseekV2DecoderImpl::handle_device_specific_bias() {
  if (dp_local_tp_rank_ != 0) {
    torch::Tensor original_tensor = at_weight_tensors_[IN_ATTENTION_OUT_BIAS];
    at_weight_tensors_[IN_ATTENTION_OUT_BIAS] =
        torch::zeros(original_tensor.sizes(),
                     torch::TensorOptions()
                         .dtype(original_tensor.dtype())
                         .device(original_tensor.device()));
  }
}

void DeepseekV2DecoderImpl::merge_shared_experts_weights() {
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

  if (layer_id_ >= prefill_param_.firstKDenseReplace) {
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

void DeepseekV2DecoderImpl::merge_experts_weights() {
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
  at_weight_tensors_[IN_MLP_DOWN_WEIGHT_EXPERT] =
      at_npu::native::npu_format_cast(mlp_down_weight, 2).contiguous();
#else
  if (decode_param_.isBF16) {
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

torch::Tensor DeepseekV2DecoderImpl::merge_experts_weights(
    std::vector<torch::Tensor>& experts,
    at::Device device,
    bool transpose) {
  torch::Tensor merged_tensor = torch::stack(experts, 0).to(device);
  if (transpose) {
    merged_tensor = merged_tensor.transpose(1, 2);
  }
  merged_tensor = merged_tensor.contiguous();
  return merged_tensor;
}

torch::Tensor DeepseekV2DecoderImpl::merge_experts_weights(
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
  return merged_tensor;
}

void DeepseekV2DecoderImpl::merge_and_copy_gate_up_weights(
    torch::Tensor&
        target_buffer,  // [num_experts, hidden_dim, gate_dim + up_dim]
    const std::vector<torch::Tensor>& experts_gate,  // [gate_dim, hidden_dim]
    const std::vector<torch::Tensor>& experts_up,    // [up_dim, hidden_dim]
    bool do_transpose) {
  const int64_t num_experts = experts_gate.size();
  const int64_t gate_dim = experts_gate[0].size(0);
  const int64_t up_dim = experts_up[0].size(0);
  const int64_t hidden_dim = experts_gate[0].size(1);

  target_buffer = at_npu::native::npu_format_cast(target_buffer.contiguous(), 2)
                      .reshape({num_experts, gate_dim + up_dim, hidden_dim});

  for (int64_t index = 0; index < num_experts; ++index) {
    target_buffer[index].slice(0, 0, gate_dim).copy_(experts_gate[index]);

    target_buffer[index]
        .slice(0, gate_dim, gate_dim + up_dim)
        .copy_(experts_up[index]);
  }

  if (do_transpose) {
    target_buffer = target_buffer.transpose(1, 2).contiguous();
    ;
  }
}

void DeepseekV2DecoderImpl::merge_and_copy_down_weights(
    torch::Tensor& target_buffer,
    const std::vector<torch::Tensor>& experts_down) {
  const int64_t num_experts = experts_down.size();

  for (int64_t index = 0; index < num_experts; ++index) {
    target_buffer[index].copy_(experts_down[index]);
  }
}

void DeepseekV2DecoderImpl::prepare_expert_weight(
    const std::vector<int32_t>& expert_list) {
  expert_routing_map_buffer_ = build_expert_routing_map(expert_list);
  auto& expert_buffer = ExpertBuffer::Instance();

  const int32_t num_local_experts = num_experts_per_partition_;
  const int32_t hidden_dim =
      at_weight_tensors_[IN_MLP_GATEUP_WEIGHT_EXPERT].size(1);
  const int32_t combined_dim =
      at_weight_tensors_[IN_MLP_GATEUP_WEIGHT_EXPERT].size(2);
  const int32_t gate_dim = combined_dim / 2;

  expert_buffer.initialize_or_reuse(
      /*gateup_weight_shape*/ {num_experts_per_partition_,
                               hidden_dim,
                               combined_dim},
      /*gateup_offset_shape*/ {num_experts_per_partition_, combined_dim, 1},
      /*gateup_scale_shape*/ {num_experts_per_partition_, combined_dim, 1},
      /*down_weight_shape*/ {num_experts_per_partition_, hidden_dim, gate_dim},
      /*down_offset_shape*/ {num_experts_per_partition_, hidden_dim, 1},
      /*down_scale_shape*/ {num_experts_per_partition_, hidden_dim, 1},
      at_weight_tensors_[IN_MLP_GATEUP_WEIGHT_EXPERT].options(),
      at_weight_tensors_[IN_MLP_GATEUP_OFFSET_EXPERT].options(),
      at_weight_tensors_[IN_MLP_GATEUP_SCALE_EXPERT].options()

  );

  const int start_expert_idx = num_experts_per_partition_ * ep_rank_;
  const int end_expert_idx = start_expert_idx + num_experts_per_partition_ - 1;

  for (const auto& pair : experts_weights_) {
    for (int expert_idx = start_expert_idx; expert_idx <= end_expert_idx;
         ++expert_idx) {
      std::string shm_key =
          get_expert_shm_key(layer_id_, expert_list[expert_idx], pair.first);
      experts_weights_[pair.first][expert_idx - start_expert_idx] =
          shared_buffer_->get_tensor(expert_list[expert_idx],
                                     layer_id_ - first_k_dense_replace_,
                                     shm_key);
      // experts_weights_[pair.first][expert_idx] =
      // shared_buffer_->get_tensors(shm_key);
    }
  }

  merge_and_copy_gate_up_weights(expert_buffer.gateup_weight,
                                 experts_weights_["gate_proj.weight"],
                                 experts_weights_["up_proj.weight"],
                                 /*do_transpose=*/true);

  merge_and_copy_gate_up_weights(expert_buffer.gateup_offset,
                                 experts_weights_["gate_proj.weight_offset"],
                                 experts_weights_["up_proj.weight_offset"]);

  merge_and_copy_gate_up_weights(expert_buffer.gateup_scale,
                                 experts_weights_["gate_proj.weight_scale"],
                                 experts_weights_["up_proj.weight_scale"]);

  merge_and_copy_down_weights(expert_buffer.down_weight,
                              experts_weights_["down_proj.weight"]);

  merge_and_copy_down_weights(expert_buffer.down_offset,
                              experts_weights_["down_proj.weight_offset"]);

  merge_and_copy_down_weights(expert_buffer.down_scale,
                              experts_weights_["down_proj.weight_scale"]);

  expert_buffer.gateup_weight =
      at_npu::native::npu_format_cast(expert_buffer.gateup_weight, 29);
}

torch::Tensor DeepseekV2DecoderImpl::build_expert_routing_map(
    std::vector<int32_t> expert_lists) {
  std::unordered_map<int64_t, std::vector<int64_t>> expert_routing_map;

  for (int64_t i = 0; i < expert_lists.size(); ++i) {
    int64_t v = expert_lists[i];
    expert_routing_map[v].emplace_back(i);
  }

  std::vector<int64_t> keys;
  std::vector<int32_t> values;
  for (auto& [key, indices] : expert_routing_map) {
    int num_of_duplications = indices.size();
    int selected_index = ep_rank_ % num_of_duplications;
    indices = {indices[selected_index]};

    keys.emplace_back(key);
    values.emplace_back(static_cast<int32_t>(indices[0]));
  }

  int64_t map_size = expert_routing_map.size();
  auto options = torch::TensorOptions().dtype(torch::kInt32);
  auto input = torch::zeros({map_size}, options);

  auto index_tensor = torch::tensor(keys, torch::kInt64);
  auto value_tensor = torch::tensor(values, torch::kInt32);
  auto result = input.scatter(0, index_tensor, value_tensor).to(device_);
  // result = result.reshape({ep_size_,result.size(0)/ep_size_}).contiguous();
  return result;
}

void DeepseekV2DecoderImpl::update_expert_weight() {
  auto& expert_buffer = ExpertBuffer::Instance();
  const auto tensor_pairs = {
      std::make_pair(IN_MLP_GATEUP_WEIGHT_EXPERT,
                     std::ref(expert_buffer.gateup_weight)),
      std::make_pair(IN_MLP_GATEUP_OFFSET_EXPERT,
                     std::ref(expert_buffer.gateup_offset)),
      std::make_pair(IN_MLP_GATEUP_SCALE_EXPERT,
                     std::ref(expert_buffer.gateup_scale)),
      std::make_pair(IN_MLP_DOWN_WEIGHT_EXPERT,
                     std::ref(expert_buffer.down_weight)),
      std::make_pair(IN_MLP_DOWN_OFFSET_EXPERT,
                     std::ref(expert_buffer.down_offset)),
      std::make_pair(IN_MLP_DOWN_SCALE_EXPERT,
                     std::ref(expert_buffer.down_scale))};
  for (auto& [index, buffer_tensor] : tensor_pairs) {
    std::swap(at_weight_tensors_[index], buffer_tensor);
    atb_weight_tensors_[index] =
        atb_speed::Utils::AtTensor2Tensor(at_weight_tensors_[index]);
    prefill_node_.inTensors.at(index) = &atb_weight_tensors_[index];
    decode_node_.inTensors.at(index) = &atb_weight_tensors_[index];
  }
  expert_routing_map_[layer_id_ - first_k_dense_replace_] =
      expert_routing_map_buffer_;
  expert_routing_map_ = expert_routing_map_.contiguous();
}

void DeepseekV2DecoderImpl::squeeze_experts_weights() {
  for (const auto& index : SQUEEZE_WEIGHT_VEC) {
    if (at_weight_tensors_[index].dim() > 1) {
      at_weight_tensors_[index] = at_weight_tensors_[index].squeeze();
    }
  }
}

int64_t DeepseekV2DecoderImpl::init_layer() {
  ATBBase::name_ = "deepseek_v2_decoder_layer " + std::to_string(layer_id_);
  model_name_ = "DeepSeek_V2";
  CHECK_OPERATION_STATUS_RETURN(init_node(prefill_node_, prefill_param_));
  CHECK_OPERATION_STATUS_RETURN(init_node(decode_node_, decode_param_));
  return atb::NO_ERROR;
}

int64_t DeepseekV2DecoderImpl::init_node(
    atb_speed::Model::Node& node,
    atb_speed::deepseekV2::DecoderLayerParam& param) {
  atb::Operation* operation = nullptr;
  atb_speed::deepseekV2::DecoderLayer(param, &operation);
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

  if (FLAGS_enable_eplb && layer_id_ >= decode_param_.firstKDenseReplace &&
      !param.isPrefill) {
    node.outTensors.resize(2);
  } else {
    node.outTensors.resize(1);
  }

  size_t inTensorId = 1;

  for (size_t weightTensorId = 0; weightTensorId < WEIGHT_COUNT_PER_LAYER;
       ++weightTensorId) {
    node.inTensors.at(weightTensorId) = &atb_weight_tensors_[weightTensorId];
  }

  node.variantPack.inTensors.reserve(node.inTensors.size());
  node.variantPack.inTensors.resize(node.inTensors.size());

  if (FLAGS_enable_eplb && layer_id_ >= decode_param_.firstKDenseReplace &&
      !param.isPrefill) {
    node.variantPack.outTensors.reserve(2);
    node.variantPack.outTensors.resize(2);  // TODO
  } else {
    node.variantPack.outTensors.reserve(1);
    node.variantPack.outTensors.resize(1);
  }
  return atb::NO_ERROR;
}

torch::Tensor DeepseekV2DecoderImpl::forward(
    torch::Tensor& x,
    torch::Tensor& cos_pos,
    torch::Tensor& sin_pos,
    torch::Tensor& attn_mask,
    KVCache& kv_cache,
    const ModelInputParams& input_params,
    atb::Context* context,
    AtbWorkspace& workspace,
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
                            true);
    st = execute_node(
        prefill_node_, context, workspace, node_id, event, event_flag);
    LOG_IF(FATAL, st != 0) << model_name_
                           << "excute decode layer fail, error code: " << st;
  } else {
    build_node_variant_pack(decode_node_,
                            x,
                            cos_pos,
                            sin_pos,
                            /*attn_mask*/ tensor_placeholder_,
                            kv_cache,
                            input_params,
                            false);
    st = execute_node(
        decode_node_, context, workspace, node_id + 1000, event, event_flag);
    LOG_IF(FATAL, st != 0) << model_name_
                           << "excute decode layer fail, error code: " << st;
  }
  return tensor_placeholder_;
}

void DeepseekV2DecoderImpl::build_node_variant_pack(
    atb_speed::Model::Node& node,
    torch::Tensor& x,
    torch::Tensor& cos_pos,
    torch::Tensor& sin_pos,
    torch::Tensor& attn_mask,
    KVCache& kv_cache,
    const ModelInputParams& input_params,
    bool is_prefill) {
  internal_tensor_ = atb_speed::Utils::AtTensor2Tensor(x);
  // final_hidden_states_ = torch::zeros_like(x);
  int32_t input_idx = 0;
  auto& dp_ep_padding = input_params.dp_ep_padding_data;

  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER) = internal_tensor_;
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 1) =
      atb_speed::Utils::AtTensor2Tensor(dp_ep_padding.expert_array());
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 2) =
      atb_speed::Utils::AtTensor2Tensor(expert_group_);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 3) =
      atb_speed::Utils::AtTensor2Tensor(one_hot_);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 4) =
      atb_speed::Utils::AtTensor2Tensor(zero_hot_);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 5) =
      atb_speed::Utils::AtTensor2Tensor(tensor_placeholder_);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 6) =
      atb_speed::Utils::AtTensor2Tensor(cos_pos);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 7) =
      atb_speed::Utils::AtTensor2Tensor(sin_pos);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 8) =
      atb_speed::Utils::AtTensor2Tensor(attn_mask);

  if (!FLAGS_enable_continuous_kvcache) {
    node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 9) =
        atb_speed::Utils::AtTensor2Tensor(kv_cache.get_k_cache());
    node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 10) =
        atb_speed::Utils::AtTensor2Tensor(kv_cache.get_v_cache());
  } else {
    node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 9) =
        ATBBase::XTensor2Tensor(kv_cache.get_k_xtensor());
    node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 10) =
        ATBBase::XTensor2Tensor(kv_cache.get_v_xtensor());
  }

  if ((!input_params.block_tables.defined() ||
       input_params.block_tables.storage().data() == nullptr) &&
      !FLAGS_enable_continuous_kvcache) {
    node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 11) =
        atb_speed::Utils::AtTensor2Tensor(int_tensor_placeholder_);
    node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 11).hostData =
        const_cast<int32_t*>(placeholder_vec_.data());
  } else {
    node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 11) =
        atb_speed::Utils::AtTensor2Tensor(input_params.kv_seq_lens);
    node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 11).hostData =
        const_cast<int32_t*>(input_params.kv_seq_lens_vec.data());
  }

  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 12) =
      atb_speed::Utils::AtTensor2Tensor(tensor_placeholder_);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 13) =
      atb_speed::Utils::AtTensor2Tensor(tensor_placeholder_);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 13).hostData =
      const_cast<int32_t*>(placeholder_vec_.data());
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 14) =
      atb_speed::Utils::AtTensor2Tensor(tensor_placeholder_);

  if (!FLAGS_enable_continuous_kvcache) {
    if (!input_params.block_tables.defined() ||
        input_params.block_tables.storage().data() == nullptr) {
      node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 15) =
          atb_speed::Utils::AtTensor2Tensor(block_tables_placeholder_);
      node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 16) =
          atb_speed::Utils::AtTensor2Tensor(slot_tensor_placeholder_);
    } else {
      node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 15) =
          atb_speed::Utils::AtTensor2Tensor(input_params.block_tables);
      node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 16) =
          atb_speed::Utils::AtTensor2Tensor(input_params.new_cache_slots);
    }
  } else {
    node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 15) =
        atb_speed::Utils::AtTensor2Tensor(input_params.kv_cache_start_offsets);
    node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 16) =
        atb_speed::Utils::AtTensor2Tensor(input_params.new_cache_slot_offsets);
  }

  if (num_speculative_tokens_ > 0 && !is_prefill) {
    if ((!input_params.block_tables.defined() ||
         input_params.block_tables.storage().data() == nullptr) &&
        !FLAGS_enable_continuous_kvcache) {
      node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 17) =
          atb_speed::Utils::AtTensor2Tensor(int_tensor_placeholder_);
      node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 17).hostData =
          const_cast<int32_t*>(placeholder_vec_.data());
    } else {
      node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 17) =
          atb_speed::Utils::AtTensor2Tensor(input_params.q_seq_lens);
      node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 17).hostData =
          const_cast<int32_t*>(input_params.q_seq_lens_vec.data());
    }
  } else {
    node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 17) =
        atb_speed::Utils::AtTensor2Tensor(tensor_placeholder_);
  }

  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 18) =
      atb_speed::Utils::AtTensor2Tensor(dp_ep_padding.attn_padding_idx());
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 19) =
      atb_speed::Utils::AtTensor2Tensor(dp_ep_padding.attn_unpadding_idx());
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 20) =
      atb_speed::Utils::AtTensor2Tensor(dp_ep_padding.ffn_padding_idx());
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 21) =
      atb_speed::Utils::AtTensor2Tensor(dp_ep_padding.ffn_unpadding_idx());
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 22) =
      atb_speed::Utils::AtTensor2Tensor(
          dp_ep_padding.lm_head_skip_padding_token_indices());
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 23) =
      atb_speed::Utils::AtTensor2Tensor(dp_ep_padding.gather_prenorm_idx());
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 24) =
      atb_speed::Utils::AtTensor2Tensor(at_start_expert_id_);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 25) =
      atb_speed::Utils::AtTensor2Tensor(at_in_device_expert_count_);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 26) =
      atb_speed::Utils::AtTensor2Tensor(dp_ep_padding.padding_idx());
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 27) =
      atb_speed::Utils::AtTensor2Tensor(dp_ep_padding.un_padding_idx());
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 28) =
      atb_speed::Utils::AtTensor2Tensor(dp_ep_padding.dynamic_ep_idx());
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 29) =
      atb_speed::Utils::AtTensor2Tensor(dp_ep_padding.moe_idx());

  for (size_t i = 0; i < WEIGHT_COUNT_PER_LAYER; ++i) {
    CHECK_THROW(node.inTensors.at(i) == nullptr,
                model_name_ << " inTensor " << i << " is NULL");
    node.variantPack.inTensors.at(i) = *node.inTensors.at(i);
  }

  node.variantPack.outTensors.at(0) = internal_tensor_;

  if (FLAGS_enable_eplb && layer_id_ >= decode_param_.firstKDenseReplace) {
    node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 30) =
        atb_speed::Utils::AtTensor2Tensor(expert_routing_map_);
    if (!is_prefill) {
      node.variantPack.outTensors.at(1) = atb_speed::Utils::AtTensor2Tensor(
          input_params
              .expert_load_data[layer_id_ - decode_param_.firstKDenseReplace]);
    }
  }
}

DeepseekV2Decoder::DeepseekV2Decoder(const Context& context,
                                     const int32_t layer_id,
                                     const float sm_scale)
    : ModuleHolder(
          create_deepseek_v2_decoder_layer(context, layer_id, sm_scale)) {}

std::shared_ptr<DeepseekV2DecoderImpl> create_deepseek_v2_decoder_layer(
    const Context& context,
    const int32_t layer_id,
    const float sm_scale) {
  return std::make_shared<DeepseekV2DecoderImpl>(context, layer_id, sm_scale);
}

}  // namespace xllm::hf
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

#include "npu_glm4_moe_decoder_layer.h"

#include "common/global_flags.h"

DECLARE_string(rank_tablefile);
DECLARE_string(communication_backend);
DECLARE_int32(expert_parallel_degree);

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

static uint64_t WEIGHT_COUNT_PER_LAYER = 68;

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

Glm4MoeDecoderImpl::Glm4MoeDecoderImpl(const ModelContext& context,
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
  start_expert_id_ = ep_rank_ * num_experts_per_partition_;
  end_expert_id_ = start_expert_id_ + num_experts_per_partition_ - 1;

  dp_size_ = parallel_args.dp_size();
  dp_local_tp_size_ = parallel_args.world_size() / dp_size_;
  CHECK_EQ(parallel_args.world_size(), dp_size_ * dp_local_tp_size_);
  dp_local_tp_rank_ = parallel_args.rank() % dp_local_tp_size_;

  n_kv_heads_ = static_cast<int32_t>(model_args.n_kv_heads().value());

  param_from_args(prefill_param_, model_args, parallel_args, true);
  param_from_args(decode_param_, model_args, parallel_args, false);

  initialize_tensors(options);
}

void Glm4MoeDecoderImpl::initialize_tensors(
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
  expert_group_ = torch::arange(1024, torch::kInt32).to(device_);
  one_hot_ = torch::tensor({1}, torch::kInt32).to(device_);
  zero_hot_ = torch::tensor({0}, torch::kInt32).to(device_);
  at_start_expert_id_ =
      torch::tensor({start_expert_id_}, torch::kInt64).to(device_);
  at_in_device_expert_count_ =
      torch::tensor({num_experts_per_partition_ - 1}, torch::kInt64)
          .to(device_);
  initialize_weight_tensors(options);
}

void Glm4MoeDecoderImpl::param_from_args(atb_speed::moe::MoeLayerParam& param,
                                         const ModelArgs& args,
                                         const ParallelArgs& parallel_args,
                                         bool is_prefill) {
  initialize_basic_parameters(param, args, parallel_args, is_prefill);
  initialize_attention_parameters(param, args, parallel_args);
  initialize_mlp_parameters(param, args, parallel_args);
  initialize_parallel_parameters(param, parallel_args);
  initialize_quantization_parameters(param);
}

void Glm4MoeDecoderImpl::resize_experts_weights(int num_of_device_experts) {
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

void Glm4MoeDecoderImpl::initialize_weight_tensors(
    const torch::TensorOptions& options) {
  for (int i = 0; i < WEIGHT_COUNT_PER_LAYER; ++i) {
    at_weight_tensors_[i] = torch::zeros({1}).to(options);
  }
}

void Glm4MoeDecoderImpl::initialize_basic_parameters(
    atb_speed::moe::MoeLayerParam& param,
    const ModelArgs& args,
    const ParallelArgs& parallel_args,
    bool is_prefill) {
  param.isFA = false;
  param.isPrefill = is_prefill;
  param.isBF16 = args.dtype() == "bfloat16";
  param.enableSwiGLU = true;
  param.enableLcoc = is_prefill;  // false;

  param.mlpLinearTransposeType = {1, -1, 1, -1};

  param.enableSplitFuse =
      (FLAGS_enable_chunked_prefill || FLAGS_enable_prefix_cache) && is_prefill;

  // not support MTP model yet
  param.enableAclGraph =
      FLAGS_enable_acl_graph && !is_prefill && args.n_layers() > 1;

  param.moeLinearTransposeType = (layer_id_ < args.first_k_dense_replace())
                                     ? std::vector<int>{-1, -1, -1, -1}
                                     : std::vector<int>{1, 1, -1, 1};

  param.normEps = args.rms_norm_eps();
  // param.rank = parallel_args.rank();
  param.backend = FLAGS_communication_backend;
  // param.rankTableFile = FLAGS_rank_tablefile;

  param.layerId = layer_id_;
  param.numHiddenLayers = args.n_layers();
  if (quantize_type_.empty()) {
    param.enableGMMSwigluQuant = false;
  } else {
    param.enableGMMSwigluQuant =
        (is_prefill && parallel_args.world_size() > 16) || !is_prefill;
  }

  param.enableSpeculate = false;                    // MTP
  param.enableSwiGLUQuantForSharedExperts = false;  // TODO

  param.useQKNorm = args.use_qk_norm();
  if (args.use_qk_norm()) {
    WEIGHT_COUNT_PER_LAYER = 70;
    WEIGHT_MAPPING_W8A8["self_attn.q_norm.weight"] = Q_NORM_WEIGHT;
    WEIGHT_MAPPING_W8A8["self_attn.k_norm.weight"] = K_NORM_WEIGHT;
    WEIGHT_MAPPING["self_attn.q_norm.weight"] = Q_NORM_WEIGHT;
    WEIGHT_MAPPING["self_attn.k_norm.weight"] = K_NORM_WEIGHT;
  }
  param.hiddenSizePerAttentionHead = args.head_dim();
  std::optional<long int> optionalValue = args.n_kv_heads();
  param.numKeyValueHeadsPerRank = std::max(
      1, static_cast<int>(optionalValue.value()) / parallel_args.world_size());

  param.numAttentionHeadsPerRank = args.n_heads() / dp_local_tp_size_;

  param.linearTransposeType = {1, -1, -1, 1, -1, -1, -1};
  // param.worldSize = parallel_args.world_size();
}

void Glm4MoeDecoderImpl::initialize_attention_parameters(
    atb_speed::moe::MoeLayerParam& param,
    const ModelArgs& args,
    const ParallelArgs& parallel_args) {
  param.linearHasBias = {true, false, false, false};
  // param.enableFA3 = false;           // TODO
  // param.enableKvQuantLayer = false;  // TODO
}

void Glm4MoeDecoderImpl::initialize_mlp_parameters(
    atb_speed::moe::MoeLayerParam& param,
    const ModelArgs& args,
    const ParallelArgs& parallel_args) {
  param.hasSharedExpert = (args.n_shared_experts() > 0);
  param.hasSharedExpertGate = false;
  param.processLogits = "normScaling";
  param.numOfSelectedExperts = {args.num_experts_per_tok()};

  param.expertParallelDegree = 0;
  param.enableFusedRouting = true;
  param.numOfSharedExperts = args.n_shared_experts();
  param.numOfExperts = args.num_experts();
  param.numOfDeviceExperts = args.num_experts();
  param.routedScalingFactor = args.routed_scaling_factor();
  param.deviceExpert.resize(num_experts_per_partition_);
  param.firstKDenseReplace = args.first_k_dense_replace();
  param.numOfGroups = args.n_group();
  param.topkGroups = atb::SVector<int>{args.topk_group()};
  param.isDenseLayer = param.layerId < param.firstKDenseReplace;
  param.enableDispatchCombineV2 = true;
  // param.deviceExpert.resize(args.n_routed_experts());
  std::iota(
      param.deviceExpert.begin(), param.deviceExpert.end(), start_expert_id_);
  // param.maskStartIdx = 0;
  param.routingMethod = "noAuxTc";

  // param.quantGroupSize = 0;
  param.enableInitQuant = false;
  param.enableSwigluQuant = false;
  param.enableFusedTopk = false;
  param.enableCVOverlap = false;  // TODO
}

void Glm4MoeDecoderImpl::initialize_parallel_parameters(
    atb_speed::moe::MoeLayerParam& param,
    const ParallelArgs& parallel_args) {
  param.lmHeadLocalTp = dp_local_tp_size_;
  param.mapping = parallel_args.mapping();
  param.tensorParallelInfo = {parallel_args.rank(),
                              parallel_args.world_size(),
                              FLAGS_communication_backend,
                              FLAGS_rank_tablefile,
                              nullptr,
                              ""};

  param.PrintParam();
  param.maxDecodeDpTokenSize = 0;  // TODO
}

void Glm4MoeDecoderImpl::initialize_quantization_parameters(
    atb_speed::moe::MoeLayerParam& param) {
  if (quantize_type_.empty()) {
    param.packQuantType = {static_cast<int>(PackType::ALL_FP),
                           static_cast<int>(PackType::ALL_FP)};
    param.linearQuantType = {static_cast<int>(LinearType::FP),
                             static_cast<int>(LinearType::INVALID),
                             static_cast<int>(LinearType::INVALID),
                             static_cast<int>(LinearType::FP),
                             static_cast<int>(LinearType::INVALID),
                             static_cast<int>(LinearType::INVALID)};
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
    param.kvQuantHasOffset = 1;
    param.enableGMMSwigluQuant = 1;
    param.enableInitQuant = 1;
    param.moePackQuantType = static_cast<int>(PackType::PACK_QUANT_UNDEFINED);
    param.packQuantType = {static_cast<int>(PackType::ALL_W8A8_ANTI),
                           static_cast<int>(PackType::ALL_W8A8_DYNAMIC_ANTI)};
    param.linearQuantType = {static_cast<int>(LinearType::INT),
                             static_cast<int>(LinearType::INVALID),
                             static_cast<int>(LinearType::INVALID),
                             static_cast<int>(LinearType::FP),
                             static_cast<int>(LinearType::INVALID),
                             static_cast<int>(LinearType::INVALID),
                             static_cast<int>(LinearType::INVALID)};

    if (layer_id_ < param.firstKDenseReplace) {
      param.moeLinearQuantType = {static_cast<int>(LinearType::INVALID),
                                  static_cast<int>(LinearType::INVALID),
                                  static_cast<int>(LinearType::INVALID),
                                  static_cast<int>(LinearType::INVALID)};

      param.mlpLinearQuantType = {static_cast<int>(LinearType::INT),
                                  static_cast<int>(LinearType::INVALID),
                                  static_cast<int>(LinearType::FP),
                                  static_cast<int>(LinearType::INVALID)};
    } else {
      param.moeLinearQuantType = {static_cast<int>(LinearType::INT),
                                  static_cast<int>(LinearType::INVALID),
                                  static_cast<int>(LinearType::INT),
                                  static_cast<int>(LinearType::INVALID)};

      param.mlpLinearQuantType = {static_cast<int>(LinearType::INT),
                                  static_cast<int>(LinearType::INVALID),
                                  static_cast<int>(LinearType::INT),
                                  static_cast<int>(LinearType::INVALID)};
    }
  }
}

void Glm4MoeDecoderImpl::load_state_dict(const StateDict& state_dict) {
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

int Glm4MoeDecoderImpl::get_mapped_index(
    const std::string& name,
    const std::unordered_map<std::string, int>& mapping) {
  const auto it = mapping.find(name);
  if (it == mapping.end()) {
    LOG(ERROR) << "Missing mapping for: " << name;
    return -1;
  }

  return it->second;
}

void Glm4MoeDecoderImpl::process_expert_weights(const StateDict& state_dict,
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

void Glm4MoeDecoderImpl::process_shared_expert_weights(
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

void Glm4MoeDecoderImpl::process_mlp_common_weights(
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

void Glm4MoeDecoderImpl::process_general_weights(const StateDict& state_dict,
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

torch::Tensor Glm4MoeDecoderImpl::get_sharded_tensor(
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

torch::Tensor Glm4MoeDecoderImpl::get_sharded_tensor(
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

std::string Glm4MoeDecoderImpl::extract_endswith(const std::string& input) {
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

int Glm4MoeDecoderImpl::extract_expert_index(const std::string& name) {
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

void Glm4MoeDecoderImpl::verify_loaded_weights(
    const std::string& prefix) const {
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

void Glm4MoeDecoderImpl::merge_loaded_weights() {
  merge_shared_experts_weights();
  if (layer_id_ >= prefill_param_.firstKDenseReplace) {
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
  c10_npu::NPUCachingAllocator::emptyCache();
  for (int i = 0; i < WEIGHT_COUNT_PER_LAYER; ++i) {
    atb_weight_tensors_[i] =
        atb_speed::Utils::AtTensor2Tensor(at_weight_tensors_[i]);
  }
  init_layer();
}

torch::Tensor Glm4MoeDecoderImpl::convert_fp16_to_int64(
    const torch::Tensor& fp16_tensor) {
  auto float_tensor = fp16_tensor.to(torch::kFloat32);
  auto int32_tensor = float_tensor.view(torch::kInt32);
  auto int64_tensor = int32_tensor.to(torch::kInt64);
  return int64_tensor;
}

void Glm4MoeDecoderImpl::convert_descaled_weights_to_float() {
  auto convert_to_float = [this](int index) {
    at_weight_tensors_[index] = at_weight_tensors_[index].to(torch::kFloat32);
  };
  convert_to_float(IN_QKV_DENSE_DESCALE);
}
void Glm4MoeDecoderImpl::merge_shared_experts_weights() {
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

void Glm4MoeDecoderImpl::merge_experts_weights() {
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

torch::Tensor Glm4MoeDecoderImpl::merge_experts_weights(
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

torch::Tensor Glm4MoeDecoderImpl::merge_experts_weights(
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

int64_t Glm4MoeDecoderImpl::init_layer() {
  NpuBaseLayer::name_ = "glm4_moe_decoder_layer " + std::to_string(layer_id_);
  model_name_ = "Glm4_Moe";
  CHECK_OPERATION_STATUS_RETURN(init_node(prefill_node_, prefill_param_));
  CHECK_OPERATION_STATUS_RETURN(init_node(decode_node_, decode_param_));

  return atb::NO_ERROR;
}

int64_t Glm4MoeDecoderImpl::init_node(atb_speed::Model::Node& node,
                                      atb_speed::moe::MoeLayerParam& param) {
  atb::Operation* operation = nullptr;
  atb_speed::glm::MoeDecoderLayer<atb::infer::RmsNormParam> decoder_layer(
      param);
  decoder_layer.BuildGraph(&operation);
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

torch::Tensor Glm4MoeDecoderImpl::forward(
    torch::Tensor& x,
    torch::Tensor& cos_pos,
    torch::Tensor& sin_pos,
    torch::Tensor& attn_mask,
    KVCache& kv_cache,
    const ModelInputParams& input_params,
    torch::Tensor& expert_array,
    std::vector<aclrtEvent*> event,
    std::vector<std::atomic<bool>*> event_flag,
    int node_id) {
  atb::Status st;
  bool is_prefill = input_params.decode_seq_range.second !=
                    input_params.q_seq_lens.size(0) - 1;
  if (is_prefill) {
    build_node_variant_pack(prefill_node_,
                            x,
                            cos_pos,
                            sin_pos,
                            attn_mask,
                            kv_cache,
                            input_params,
                            expert_array,
                            true);
    st = execute_node(prefill_node_, node_id, event, event_flag);
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
    st = execute_node(decode_node_, node_id + 1000, event, event_flag);
    LOG_IF(FATAL, st != 0) << model_name_
                           << "excute decode layer fail, error code: " << st;
  }

  return tensor_placeholder_;
}

void Glm4MoeDecoderImpl::build_node_variant_pack(
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
  auto& dp_ep_padding = input_params.dp_ep_padding_data;

  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER) = internal_tensor_;
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 1) =
      atb_speed::Utils::AtTensor2Tensor(cos_pos);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 2) =
      atb_speed::Utils::AtTensor2Tensor(sin_pos);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 3) =
      atb_speed::Utils::AtTensor2Tensor(attn_mask);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 4) =
      atb_speed::Utils::AtTensor2Tensor(kv_cache.get_k_cache());
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 5) =
      atb_speed::Utils::AtTensor2Tensor(kv_cache.get_v_cache());

  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 6) =
      atb_speed::Utils::AtTensor2Tensor(input_params.kv_seq_lens);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 6).hostData =
      const_cast<int32_t*>(input_params.kv_seq_lens_vec.data());
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 7) =
      atb_speed::Utils::AtTensor2Tensor(tensor_placeholder_);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 7).hostData =
      const_cast<int32_t*>(placeholder_vec_.data());
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 8) =
      atb_speed::Utils::AtTensor2Tensor(tensor_placeholder_);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 9) =
      atb_speed::Utils::AtTensor2Tensor(input_params.block_tables);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 10) =
      atb_speed::Utils::AtTensor2Tensor(input_params.new_cache_slots);

  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 11) =
      atb_speed::Utils::AtTensor2Tensor(expert_array);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 12) =
      atb_speed::Utils::AtTensor2Tensor(expert_group_);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 13) =
      atb_speed::Utils::AtTensor2Tensor(one_hot_);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 14) =
      atb_speed::Utils::AtTensor2Tensor(zero_hot_);

  int32_t input_idx = WEIGHT_COUNT_PER_LAYER + 15;

  if (is_prefill &&
      (FLAGS_enable_chunked_prefill || FLAGS_enable_prefix_cache)) {
    node.variantPack.inTensors.at(input_idx) =
        atb_speed::Utils::AtTensor2Tensor(input_params.q_seq_lens);
    node.variantPack.inTensors.at(input_idx).hostData =
        const_cast<int32_t*>(input_params.q_seq_lens_vec.data());
    input_idx++;
  }

  node.variantPack.inTensors.at(input_idx++) =
      atb_speed::Utils::AtTensor2Tensor(tensor_placeholder_);
  node.variantPack.inTensors.at(input_idx++) =
      atb_speed::Utils::AtTensor2Tensor(tensor_placeholder_);
  node.variantPack.inTensors.at(input_idx++) =
      atb_speed::Utils::AtTensor2Tensor(tensor_placeholder_);
  node.variantPack.inTensors.at(input_idx++) =
      atb_speed::Utils::AtTensor2Tensor(tensor_placeholder_);
  node.variantPack.inTensors.at(input_idx++) =
      atb_speed::Utils::AtTensor2Tensor(tensor_placeholder_);
  node.variantPack.inTensors.at(input_idx++) =
      atb_speed::Utils::AtTensor2Tensor(tensor_placeholder_);
  node.variantPack.inTensors.at(input_idx++) =
      atb_speed::Utils::AtTensor2Tensor(at_start_expert_id_);
  node.variantPack.inTensors.at(input_idx++) =
      atb_speed::Utils::AtTensor2Tensor(at_in_device_expert_count_);

  node.variantPack.inTensors.at(input_idx++) =
      atb_speed::Utils::AtTensor2Tensor(tensor_placeholder_);
  node.variantPack.inTensors.at(input_idx++) =
      atb_speed::Utils::AtTensor2Tensor(tensor_placeholder_);

  if (FLAGS_enable_acl_graph && !is_prefill &&
      input_params.graph_buffer.tiling_data.defined()) {
    node.variantPack.inTensors.at(input_idx++) =
        atb_speed::Utils::AtTensor2Tensor(
            input_params.graph_buffer.tiling_data);
  }

  for (size_t i = 0; i < WEIGHT_COUNT_PER_LAYER; ++i) {
    CHECK_THROW(node.inTensors.at(i) == nullptr,
                model_name_ << " inTensor " << i << " is NULL");
    node.variantPack.inTensors.at(i) = *node.inTensors.at(i);
  }

  node.variantPack.outTensors.at(0) = internal_tensor_;
}

Glm4MoeDecoder::Glm4MoeDecoder(const ModelContext& context,
                               const int32_t layer_id)
    : ModuleHolder(create_glm4_moe_decoder_layer(context, layer_id)) {}

std::shared_ptr<Glm4MoeDecoderImpl> create_glm4_moe_decoder_layer(
    const ModelContext& context,
    int32_t layer_id) {
  return std::make_shared<Glm4MoeDecoderImpl>(context, layer_id);
}

}  // namespace layer
}  // namespace xllm

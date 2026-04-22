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

#include "npu_minimax_m2_decoder_layer_impl.h"

#include <gflags/gflags.h>

#include <algorithm>
#include <numeric>
#include <unordered_set>

#include "common/global_flags.h"

namespace xllm {
namespace layer {

static uint64_t WEIGHT_COUNT_PER_LAYER = 68;

namespace {

std::vector<size_t> build_op_weight_tensor_ids(
    const atb_speed::glm::MoeLayerParam& param) {
  std::vector<size_t> ids;
  ids.reserve(70);

  // Base GLM MoE weights:
  // 0-3   : input norm
  // 4-27  : attention
  // 28-31 : post-attention norm
  for (size_t i = 0; i < 32; ++i) {
    ids.push_back(i);
  }

  // 32-49 are only present in the kernel input list when shared experts or
  // dense-layer replacement are enabled.
  if (param.hasSharedExpert || param.isDenseLayer) {
    for (size_t i = 32; i < 50; ++i) {
      ids.push_back(i);
    }
  }

  // 50-67 : routed MoE gate / expert weights.
  for (size_t i = 50; i < 68; ++i) {
    ids.push_back(i);
  }

  // 68-69 : q/k norm.
  if (param.useQKNorm) {
    ids.push_back(68);
    ids.push_back(69);
  }

  return ids;
}

size_t get_runtime_input_offset(const atb_speed::Model::Node& node) {
  size_t offset = 0;
  while (offset < node.inTensors.size() &&
         node.inTensors.at(offset) != nullptr) {
    ++offset;
  }
  return offset;
}

}  // namespace

NpuMiniMaxM2DecoderLayerImpl::NpuMiniMaxM2DecoderLayerImpl(
    const ModelContext& context,
    const int32_t layer_id)
    : BaseLayer(context),
      device_id_(context.get_tensor_options().device().index()),
      layer_id_(layer_id),
      num_speculative_tokens_(
          context.get_model_args().num_speculative_tokens()) {
  auto model_args = context.get_model_args();
  auto parallel_args = context.get_parallel_args();
  auto options = context.get_tensor_options();

  use_glm_qk_norm_ =
      model_args.use_qk_norm() && model_args.qk_norm_type() != "per_layer";
  if (model_args.use_qk_norm() && !use_glm_qk_norm_ && layer_id_ == 0 &&
      parallel_args.rank() == 0) {
    LOG(WARNING)
        << "MiniMax-M2 uses qk_norm_type="
        << (model_args.qk_norm_type().empty() ? "<empty>"
                                              : model_args.qk_norm_type())
        << ", but the current GLM-based NPU fused path only supports per-head "
           "Q/K RMSNorm. Disabling fused Q/K norm for MiniMax to avoid the "
           "warmup crash. Generation correctness remains incomplete until a "
           "MiniMax-native attention path is implemented.";
  }
  WEIGHT_COUNT_PER_LAYER = use_glm_qk_norm_ ? 70 : 68;

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

  param_from_args(prefill_param_, model_args, parallel_args, true);
  param_from_args(decode_param_, model_args, parallel_args, false);
  loader_ = std::make_unique<MiniMaxM2DecoderLoader>(
      WEIGHT_COUNT_PER_LAYER, context, use_glm_qk_norm_);
  initialize_tensors(options);
}

void NpuMiniMaxM2DecoderLayerImpl::initialize_tensors(
    const torch::TensorOptions& options) {
  atb_weight_tensors_.resize(WEIGHT_COUNT_PER_LAYER);
  placeholder_vec_ = {1};
  int_tensor_placeholder_ = torch::ones({1}).to(torch::kInt32).to(device_);
  slot_tensor_placeholder_ = torch::full({1}, 0).to(torch::kInt32).to(device_);
  block_tables_placeholder_ =
      torch::zeros({1, 1}).to(torch::kInt32).to(device_);
  tensor_placeholder_ = torch::zeros({1}).to(options);
  loader_->resize_experts_weights(num_experts_per_partition_);
  expert_group_ = torch::arange(1024, torch::kInt32).to(device_);
  one_hot_ = torch::tensor({1}, torch::kInt32).to(device_);
  zero_hot_ = torch::tensor({0}, torch::kInt32).to(device_);
  at_start_expert_id_ =
      torch::tensor({start_expert_id_}, torch::kInt64).to(device_);
  at_in_device_expert_count_ =
      torch::tensor({num_experts_per_partition_ - 1}, torch::kInt64)
          .to(device_);
}

void NpuMiniMaxM2DecoderLayerImpl::param_from_args(
    atb_speed::glm::MoeLayerParam& param,
    const ModelArgs& args,
    const ParallelArgs& parallel_args,
    bool is_prefill) {
  initialize_basic_parameters(param, args, parallel_args, is_prefill);
  initialize_attention_parameters(param, args, parallel_args);
  initialize_mlp_parameters(param, args, parallel_args);
  initialize_parallel_parameters(param, parallel_args);
  initialize_quantization_parameters(param);
}

void NpuMiniMaxM2DecoderLayerImpl::initialize_basic_parameters(
    atb_speed::glm::MoeLayerParam& param,
    const ModelArgs& args,
    const ParallelArgs& parallel_args,
    bool is_prefill) {
  param.isFA = false;
  param.isPrefill = is_prefill;
  param.isBF16 = args.dtype() == "bfloat16";
  param.enableSwiGLU = true;
  param.enableLcoc = is_prefill;

  param.mlpLinearTransposeType = {-1, -1, -1, -1};

  // The current MiniMax bring-up reuses the GLM noAuxTc MoE path. Keep the
  // prefill input layout conservative until MiniMax has a dedicated router
  // implementation, otherwise warmup can bind an extra split-fuse q_len input
  // that this path does not instantiate.
  param.enableSplitFuse = false;
  param.enableAclGraphPagedAttention = FLAGS_enable_graph && !is_prefill;

  if (quantize_type_.empty()) {
    param.moeLinearTransposeType = std::vector<int>{1, 1, -1, 1};
  } else {
    param.moeLinearTransposeType = std::vector<int>{1, 0, -1, 1};
  }
  param.normEps = args.rms_norm_eps();
  param.backend = FLAGS_communication_backend;

  param.layerId = layer_id_;
  param.numHiddenLayers = args.n_layers();
  if (quantize_type_.empty()) {
    param.enableGMMSwigluQuant = false;
  } else {
    param.enableGMMSwigluQuant =
        (is_prefill && parallel_args.world_size() > 16) || !is_prefill;
  }

  param.enableSpeculate = false;
  param.enableSwiGLUQuantForSharedExperts = false;

  // The reused GLM MoE kernel only supports per-head q/k RMSNorm, while
  // MiniMax-M2 stores per-layer q/k RMSNorm tensors.
  param.useQKNorm = use_glm_qk_norm_;
  param.hiddenSizePerAttentionHead = args.head_dim();
  std::optional<long int> optional_value = args.n_kv_heads();
  param.numKeyValueHeadsPerRank = std::max(
      1, static_cast<int>(optional_value.value()) / parallel_args.world_size());
  param.numAttentionHeadsPerRank = args.n_heads() / dp_local_tp_size_;

  param.linearTransposeType = {1, -1, -1, 1, -1, -1, -1};
}

void NpuMiniMaxM2DecoderLayerImpl::initialize_attention_parameters(
    atb_speed::glm::MoeLayerParam& param,
    const ModelArgs& args,
    const ParallelArgs& parallel_args) {
  param.linearHasBias = {args.attention_bias(), false, false, false};
}

void NpuMiniMaxM2DecoderLayerImpl::initialize_mlp_parameters(
    atb_speed::glm::MoeLayerParam& param,
    const ModelArgs& args,
    const ParallelArgs& parallel_args) {
  param.hasSharedExpert = (args.n_shared_experts() > 0);
  param.hasSharedExpertGate = false;
  param.processLogits = "normScaling";
  param.numOfSelectedExperts = {args.num_experts_per_tok()};

  param.expertParallelDegree =
      ep_size_ > 1 ? std::max(FLAGS_expert_parallel_degree, 1) : 0;
  param.enableFusedRouting = true;
  param.numOfSharedExperts = args.n_shared_experts();
  param.numOfExperts = args.num_experts();
  param.numOfDeviceExperts = num_experts_per_partition_;
  param.routedScalingFactor = args.routed_scaling_factor();
  param.deviceExpert.resize(num_experts_per_partition_);
  param.firstKDenseReplace = args.first_k_dense_replace();
  param.numOfGroups = std::max(args.n_group(), 1);
  param.topkGroups = atb::SVector<int>{std::max(args.topk_group(), 1)};
  param.isDenseLayer = false;
  param.enableDispatchCombineV2 = true;
  std::iota(
      param.deviceExpert.begin(), param.deviceExpert.end(), start_expert_id_);
  param.routingMethod = "noAuxTc";

  param.quantGroupSize = 0;
  param.enableInitQuant = false;
  param.enableSwigluQuant = false;
  param.enableFusedTopk = false;
  param.enableCVOverlap = false;
}

void NpuMiniMaxM2DecoderLayerImpl::initialize_parallel_parameters(
    atb_speed::glm::MoeLayerParam& param,
    const ParallelArgs& parallel_args) {
  param.lmHeadLocalTp = dp_local_tp_size_;
  param.mapping = parallel_args.mapping();
  param.tensorParallelInfo = {parallel_args.rank(),
                              parallel_args.world_size(),
                              FLAGS_communication_backend,
                              FLAGS_rank_tablefile,
                              nullptr,
                              ""};

  param.maxDecodeDpTokenSize = 0;
}

void NpuMiniMaxM2DecoderLayerImpl::initialize_quantization_parameters(
    atb_speed::glm::MoeLayerParam& param) {
  if (quantize_type_.empty()) {
    param.packQuantType = {static_cast<int>(PackType::ALL_FP),
                           static_cast<int>(PackType::ALL_FP)};
    param.linearQuantType = {static_cast<int>(LinearType::FP),
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
    param.linearQuantType = {static_cast<int>(LinearType::INT),
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

void NpuMiniMaxM2DecoderLayerImpl::merge_loaded_weights() {
  loader_->merge_loaded_weights();
  auto& at_weight_tensors = loader_->get_at_weight_tensors();
  c10_npu::NPUCachingAllocator::emptyCache();
  for (int i = 0; i < WEIGHT_COUNT_PER_LAYER; ++i) {
    atb_weight_tensors_[i] =
        atb_speed::Utils::AtTensor2Tensor(at_weight_tensors[i]);
  }
  init_layer();
}

int64_t NpuMiniMaxM2DecoderLayerImpl::init_layer() {
  name_ = "minimax_m2_decoder_layer " + std::to_string(layer_id_);
  model_name_ = "MiniMaxM2";
  CHECK_OPERATION_STATUS_RETURN(init_node(prefill_node_, prefill_param_));
  CHECK_OPERATION_STATUS_RETURN(init_node(decode_node_, decode_param_));

  return atb::NO_ERROR;
}

int64_t NpuMiniMaxM2DecoderLayerImpl::init_node(
    atb_speed::Model::Node& node,
    atb_speed::glm::MoeLayerParam& param) {
  atb::Operation* operation = nullptr;
  const auto op_weight_tensor_ids = build_op_weight_tensor_ids(param);
  atb_speed::glm::MoeDecoderLayer<atb::infer::RmsNormParam> decoder_layer(
      param);
  decoder_layer.BuildGraph(&operation);
  node.operation.reset(operation);
  CHECK_NOTNULL(node.operation);
  CHECK_GT(node.operation->GetInputNum(), 0);
  CHECK_LE(op_weight_tensor_ids.size(), node.operation->GetInputNum());
  if (layer_id_ == 0) {
    LOG(INFO) << "MiniMaxM2 " << (param.isPrefill ? "prefill" : "decode")
              << " node input count=" << node.operation->GetInputNum()
              << ", loader weight count=" << WEIGHT_COUNT_PER_LAYER
              << ", op weight count=" << op_weight_tensor_ids.size()
              << ", enableSplitFuse=" << param.enableSplitFuse;
  }
  node.inTensors.resize(node.operation->GetInputNum());
  node.outTensors.resize(1);

  for (size_t input_tensor_id = 0;
       input_tensor_id < op_weight_tensor_ids.size();
       ++input_tensor_id) {
    node.inTensors.at(input_tensor_id) =
        &atb_weight_tensors_[op_weight_tensor_ids.at(input_tensor_id)];
  }

  node.variantPack.inTensors.reserve(node.inTensors.size());
  node.variantPack.inTensors.resize(node.inTensors.size());
  node.variantPack.outTensors.reserve(1);
  node.variantPack.outTensors.resize(1);

  return atb::NO_ERROR;
}

torch::Tensor NpuMiniMaxM2DecoderLayerImpl::forward(
    torch::Tensor& x,
    torch::Tensor& cos_pos,
    torch::Tensor& sin_pos,
    torch::Tensor& attn_mask,
    KVCache& kv_cache,
    const ModelInputParams& input_params,
    aclrtEvent* event,
    std::atomic<bool>* event_flag,
    int node_id) {
  atb::Status st;
  if (!input_params.batch_forward_type.is_decode()) {
    build_node_variant_pack(prefill_node_,
                            x,
                            cos_pos,
                            sin_pos,
                            attn_mask,
                            kv_cache,
                            input_params,
                            true);
    st = execute_node(prefill_node_, node_id, event, event_flag);
    LOG_IF(FATAL, st != 0) << model_name_
                           << " excute prefill layer fail, error code: " << st;
  } else {
    build_node_variant_pack(decode_node_,
                            x,
                            cos_pos,
                            sin_pos,
                            tensor_placeholder_,
                            kv_cache,
                            input_params,
                            false);
    st = execute_node(decode_node_, node_id + 1000, event, event_flag);
    LOG_IF(FATAL, st != 0) << model_name_
                           << " excute decode layer fail, error code: " << st;
  }

  return tensor_placeholder_;
}

void NpuMiniMaxM2DecoderLayerImpl::build_node_variant_pack(
    atb_speed::Model::Node& node,
    torch::Tensor& x,
    torch::Tensor& cos_pos,
    torch::Tensor& sin_pos,
    torch::Tensor& attn_mask,
    KVCache& kv_cache,
    const ModelInputParams& input_params,
    bool is_prefill) {
  internal_tensor_ = atb_speed::Utils::AtTensor2Tensor(x);
  const size_t runtime_offset = get_runtime_input_offset(node);
  const bool enable_split_fuse = is_prefill ? prefill_param_.enableSplitFuse
                                            : decode_param_.enableSplitFuse;
  node.variantPack.inTensors.at(runtime_offset) = internal_tensor_;
  node.variantPack.inTensors.at(runtime_offset + 1) =
      atb_speed::Utils::AtTensor2Tensor(cos_pos);
  node.variantPack.inTensors.at(runtime_offset + 2) =
      atb_speed::Utils::AtTensor2Tensor(sin_pos);
  node.variantPack.inTensors.at(runtime_offset + 3) =
      atb_speed::Utils::AtTensor2Tensor(attn_mask);
  node.variantPack.inTensors.at(runtime_offset + 4) =
      atb_speed::Utils::AtTensor2Tensor(kv_cache.get_k_cache());
  node.variantPack.inTensors.at(runtime_offset + 5) =
      atb_speed::Utils::AtTensor2Tensor(kv_cache.get_v_cache());

  if (!input_params.kv_seq_lens.defined() ||
      input_params.kv_seq_lens.storage().data() == nullptr) {
    node.variantPack.inTensors.at(runtime_offset + 6) =
        atb_speed::Utils::AtTensor2Tensor(int_tensor_placeholder_);
    node.variantPack.inTensors.at(runtime_offset + 6).hostData =
        const_cast<int32_t*>(placeholder_vec_.data());
  } else {
    node.variantPack.inTensors.at(runtime_offset + 6) =
        atb_speed::Utils::AtTensor2Tensor(input_params.kv_seq_lens);
    node.variantPack.inTensors.at(runtime_offset + 6).hostData =
        const_cast<int32_t*>(input_params.kv_seq_lens_vec.data());
  }
  node.variantPack.inTensors.at(runtime_offset + 7) =
      atb_speed::Utils::AtTensor2Tensor(tensor_placeholder_);
  node.variantPack.inTensors.at(runtime_offset + 7).hostData =
      const_cast<int32_t*>(placeholder_vec_.data());
  node.variantPack.inTensors.at(runtime_offset + 8) =
      atb_speed::Utils::AtTensor2Tensor(tensor_placeholder_);
  if (!input_params.block_tables.defined() ||
      input_params.block_tables.storage().data() == nullptr) {
    node.variantPack.inTensors.at(runtime_offset + 9) =
        atb_speed::Utils::AtTensor2Tensor(block_tables_placeholder_);
    node.variantPack.inTensors.at(runtime_offset + 10) =
        atb_speed::Utils::AtTensor2Tensor(slot_tensor_placeholder_);
  } else {
    node.variantPack.inTensors.at(runtime_offset + 9) =
        atb_speed::Utils::AtTensor2Tensor(input_params.block_tables);
    node.variantPack.inTensors.at(runtime_offset + 10) =
        atb_speed::Utils::AtTensor2Tensor(input_params.new_cache_slots);
  }

  node.variantPack.inTensors.at(runtime_offset + 11) =
      atb_speed::Utils::AtTensor2Tensor(input_params.expert_array);
  node.variantPack.inTensors.at(runtime_offset + 12) =
      atb_speed::Utils::AtTensor2Tensor(expert_group_);
  node.variantPack.inTensors.at(runtime_offset + 13) =
      atb_speed::Utils::AtTensor2Tensor(one_hot_);
  node.variantPack.inTensors.at(runtime_offset + 14) =
      atb_speed::Utils::AtTensor2Tensor(zero_hot_);

  int32_t input_idx = runtime_offset + 15;
  if (is_prefill && enable_split_fuse) {
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

  if (FLAGS_enable_graph && !is_prefill &&
      input_params.graph_buffer.tiling_data.defined()) {
    node.variantPack.inTensors.at(input_idx++) =
        atb_speed::Utils::AtTensor2Tensor(
            input_params.graph_buffer.tiling_data);
  }

  for (size_t i = 0; i < runtime_offset; ++i) {
    CHECK_THROW(node.inTensors.at(i) == nullptr,
                model_name_ << " inTensor " << i << " is NULL");
    node.variantPack.inTensors.at(i) = *node.inTensors.at(i);
  }

  node.variantPack.outTensors.at(0) = internal_tensor_;
}

}  // namespace layer
}  // namespace xllm

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

#include "npu_qwen3_omni_code2wav_transformer_layer_impl.h"

#include <atb/atb_infer.h>
#include <atb/types.h>
#include <glog/logging.h>
#include <mstx/ms_tools_ext.h>

#include <iostream>
#include <map>

#include "npu_base_layer.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUException.h"

namespace xllm {
namespace layer {

enum TransformerLayerTensorId : int {
  IN_INPUT_NORM_WEIGHT = 0,
  IN_QKV_WEIGHT = 1,
  IN_ATTENTION_OUT_WEIGHT = 2,
  IN_SELF_ATTN_LAYER_SCALE_SCALE = 3,
  IN_SELFATTENTION_OUT_NORM_WEIGHT = 4,
  IN_MLP_GATEUP_WEIGHT = 5,
  IN_MLP_DOWN_WEIGHT = 6,
  IN_MLP_LAYER_SCALE_SCALE = 7,
  IN_MLP_GATE_WEIGHT = 8,  // tmp weight, atb actually not use
  IN_MLP_UP_WEIGHT = 9,    // tmp weight, atb actually not use
  IN_QKV_WEIGHT_Q = 10,    // tmp weight, atb actually not use
  IN_QKV_WEIGHT_K = 11,    // tmp weight, atb actually not use
  IN_QKV_WEIGHT_V = 12,    // tmp weight, atb actually not use
};

static const uint64_t WEIGHT_COUNT_PER_LAYER = 13;

static const std::unordered_map<std::string, int> WEIGHT_MAPPING = {
    {"input_layernorm.weight", IN_INPUT_NORM_WEIGHT},

    {"self_attn.q_proj.weight", IN_QKV_WEIGHT_Q},

    {"self_attn.k_proj.weight", IN_QKV_WEIGHT_K},

    {"self_attn.v_proj.weight", IN_QKV_WEIGHT_V},

    {"self_attn.o_proj.weight", IN_ATTENTION_OUT_WEIGHT},

    {"post_attention_layernorm.weight", IN_SELFATTENTION_OUT_NORM_WEIGHT},

    {"self_attn_layer_scale.scale", IN_SELF_ATTN_LAYER_SCALE_SCALE},

    {"mlp.gate_proj.weight", IN_MLP_GATE_WEIGHT},

    {"mlp.up_proj.weight", IN_MLP_UP_WEIGHT},

    {"mlp.down_proj.weight", IN_MLP_DOWN_WEIGHT},

    {"mlp_layer_scale.scale", IN_MLP_LAYER_SCALE_SCALE}};

static const std::unordered_map<int, int> WEIGHT_SHARD = {
    {IN_QKV_WEIGHT_Q, 0},
    {IN_QKV_WEIGHT_K, 0},
    {IN_QKV_WEIGHT_V, 0},
    {IN_ATTENTION_OUT_WEIGHT, 1},
    {IN_MLP_GATE_WEIGHT, 0},
    {IN_MLP_UP_WEIGHT, 0},
    {IN_MLP_DOWN_WEIGHT, 1}};

void Qwen3OmniCode2WavTransformerLayerImpl::param_from_args(
    atb_speed::qwen3_omni::Code2WavTransformerLayerParam& param,
    const ModelArgs& args,
    const ParallelArgs& parallel_args) {
  param.isBF16 = args.dtype() == "bfloat16";
  param.rmsNormEps = args.rms_norm_eps();
  param.worldSize = parallel_args.world_size();
  param.numAttentionHeadsPerRank =
      args.mm_num_attention_heads() / param.worldSize;
  param.hiddenSizePerAttentionHead =
      args.mm_hidden_size() / args.mm_num_attention_heads();
  std::optional<long int> optionalValue = args.mm_num_attention_heads();
  param.numKeyValueHeadsPerRank =
      static_cast<int>(optionalValue.value()) / param.worldSize;
  param.rank = parallel_args.rank();
  param.backend = "lccl";
  param.enableLogN = false;
}

Qwen3OmniCode2WavTransformerLayerImpl::Qwen3OmniCode2WavTransformerLayerImpl(
    const ModelContext& context,
    const int32_t layer_id)
    : BaseLayer(context) {
  auto model_args = context.get_model_args();
  auto parallel_args = context.get_parallel_args();
  auto options = context.get_tensor_options();
  param_from_args(encode_param_, model_args, parallel_args);

  at_weight_tensors_.resize(WEIGHT_COUNT_PER_LAYER);
  atb_weight_tensors_.resize(WEIGHT_COUNT_PER_LAYER);
  dtype_ = c10::typeMetaToScalarType(options.dtype());
  device_id_ = options.device().index();
  placeholder_ = atb_speed::Utils::AtTensor2Tensor(
      torch::zeros({1}).to(device_).to(dtype_));
  at_placeholder_ = torch::zeros({1}).to(device_).to(dtype_);
  for (int i = 0; i < WEIGHT_COUNT_PER_LAYER; ++i) {
    at_weight_tensors_[i] = torch::zeros({1}).to(options);
  }
}

Qwen3OmniCode2WavTransformerLayerImpl::
    ~Qwen3OmniCode2WavTransformerLayerImpl() {};

void Qwen3OmniCode2WavTransformerLayerImpl::verify_loaded_weights(
    const std::string& prefix) const {
  for (const auto& [name, index] : WEIGHT_MAPPING) {
    CHECK(at_weight_tensors_[index].sizes() != std::vector<int64_t>({1}))
        << "weight is not loaded for " << name;
  }
}

void Qwen3OmniCode2WavTransformerLayerImpl::merge_loaded_weights() {
  // get_weights_col_packed_qkv();
  auto new_qkv_weight = torch::cat({at_weight_tensors_[IN_QKV_WEIGHT_Q],
                                    at_weight_tensors_[IN_QKV_WEIGHT_K],
                                    at_weight_tensors_[IN_QKV_WEIGHT_V]},
                                   0);
  at_weight_tensors_[IN_QKV_WEIGHT] = new_qkv_weight;
  at_weight_tensors_[IN_QKV_WEIGHT_Q] = torch::zeros({1}).to(device_);
  at_weight_tensors_[IN_QKV_WEIGHT_K] = torch::zeros({1}).to(device_);
  at_weight_tensors_[IN_QKV_WEIGHT_V] = torch::zeros({1}).to(device_);

  auto new_mlp_weight = torch::cat({at_weight_tensors_[IN_MLP_GATE_WEIGHT],
                                    at_weight_tensors_[IN_MLP_UP_WEIGHT]},
                                   0);
  at_weight_tensors_[IN_MLP_GATEUP_WEIGHT] = new_mlp_weight;
  at_weight_tensors_[IN_MLP_GATE_WEIGHT] = torch::zeros({1}).to(device_);
  at_weight_tensors_[IN_MLP_UP_WEIGHT] = torch::zeros({1}).to(device_);

  c10_npu::NPUCachingAllocator::emptyCache();
  for (int i = 0; i < WEIGHT_COUNT_PER_LAYER; ++i) {
    atb_weight_tensors_[i] =
        atb_speed::Utils::AtTensor2Tensor(at_weight_tensors_[i]);
  }

  init_layer();
}

void Qwen3OmniCode2WavTransformerLayerImpl::get_weights_col_packed_qkv() {
  int rank = encode_param_.rank;
  int worldSize = encode_param_.worldSize;
  // split qkv weight
  qkv_weight = torch::chunk(at_weight_tensors_[IN_QKV_WEIGHT], 3, 0);
  // weight
  at_weight_tensors_[IN_QKV_WEIGHT_Q] =
      (qkv_weight[0].chunk(worldSize, 0))[rank];
  at_weight_tensors_[IN_QKV_WEIGHT_K] =
      (qkv_weight[1].chunk(worldSize, 0))[rank];
  at_weight_tensors_[IN_QKV_WEIGHT_V] =
      (qkv_weight[2].chunk(worldSize, 0))[rank];
}

void Qwen3OmniCode2WavTransformerLayerImpl::load_state_dict(
    const StateDict& state_dict) {
  for (const auto& [name, index] : WEIGHT_MAPPING) {
    if (WEIGHT_SHARD.find(index) != WEIGHT_SHARD.end()) {
      set_weight(state_dict, name, index, WEIGHT_SHARD.at(index));
    } else {
      set_weight(state_dict, name, index);
    }
  }
}

int64_t Qwen3OmniCode2WavTransformerLayerImpl::init_layer() {
  name_ = "qwen3_omni_code2wav_transformer_layer";
  model_name_ = "qwen3_omni_code2wav";
  CHECK_OPERATION_STATUS_RETURN(init_node(encode_node_, encode_param_));
  return atb::NO_ERROR;
}

int64_t Qwen3OmniCode2WavTransformerLayerImpl::init_node(
    atb_speed::Model::Node& node,
    atb_speed::qwen3_omni::Code2WavTransformerLayerParam& param) {
  atb::Operation* operation = nullptr;
  atb_speed::qwen3_omni::Qwen3Omni_Code2wav_TransformerLayer(param, &operation);
  node.operation.reset(operation);
  if (node.operation == nullptr) {
    LOG(ERROR) << "node.operation is null";
    return -1;
  }
  auto x = node.operation->GetInputNum();
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

torch::Tensor Qwen3OmniCode2WavTransformerLayerImpl::forward(
    torch::Tensor& x,
    torch::Tensor& cos_pos,
    torch::Tensor& sin_pos,
    torch::Tensor& attn_mask,
    torch::Tensor& cu_seqlen,
    std::vector<int>& cu_seqlen_vec,
    ModelInputParams& input_params,
    int node_id,
    aclrtEvent* event,
    std::atomic<bool>* event_flag) {
  atb::Status st;

  build_node_variant_pack(encode_node_,
                          x,
                          cos_pos,
                          sin_pos,
                          attn_mask,
                          cu_seqlen,
                          cu_seqlen_vec,
                          input_params,
                          true);
  st = execute_node(encode_node_, node_id);
  LOG_IF(FATAL, st != 0) << model_name_
                         << "excute transformer layer fail, error code: " << st;
  return x;
}

void Qwen3OmniCode2WavTransformerLayerImpl::build_node_variant_pack(
    atb_speed::Model::Node& node,
    torch::Tensor& x,
    torch::Tensor& cos_pos,
    torch::Tensor& sin_pos,
    torch::Tensor& attn_mask,
    torch::Tensor& cu_seqlen,
    std::vector<int>& cu_seqlen_vec,
    ModelInputParams& input_params,
    bool is_prefill) {
  internal_tensors_ = atb_speed::Utils::AtTensor2Tensor(x);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER) = internal_tensors_;
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 1) =
      atb_speed::Utils::AtTensor2Tensor(cos_pos);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 2) =
      atb_speed::Utils::AtTensor2Tensor(sin_pos);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 3) =
      atb_speed::Utils::AtTensor2Tensor(cu_seqlen);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 3).hostData =
      cu_seqlen_vec.data();
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 4) =
      atb_speed::Utils::AtTensor2Tensor(attn_mask);

  for (size_t i = 0; i < WEIGHT_COUNT_PER_LAYER; ++i) {
    CHECK_THROW(node.inTensors.at(i) == nullptr,
                model_name_ << "inTensor " << i << "is NULL");
    node.variantPack.inTensors.at(i) = *node.inTensors.at(i);
  }

  node.variantPack.outTensors.at(0) = internal_tensors_;
}

}  // namespace layer
}  // namespace xllm

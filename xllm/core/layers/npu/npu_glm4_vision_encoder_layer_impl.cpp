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

// copy from qwen3 vl, please follow its modifications
#include "npu_glm4_vision_encoder_layer_impl.h"

#include <glog/logging.h>
#include <mstx/ms_tools_ext.h>

#include <iostream>
#include <map>

#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUException.h"
#include "xllm_kernels/models/glm4v/glm4v_encoder.h"

namespace xllm {
namespace layer {

enum Glm4VisionEncoderLayerTensorId : int {
  IN_INPUT_NORM_WEIGHT = 0,
  IN_POST_NORM_WEIGHT,
  IN_QKV_WEIGHT,
  IN_ATTN_PROJ_WEIGHT,
  IN_LINEAR_GATE_UP_WEIGHT,
  IN_LINEAR_DOWN_WEIGHT,
  IN_LINEAR_UP_WEIGHT,
  IN_LINEAR_GATE_WEIGHT
};

const uint64_t WEIGHT_COUNT_PER_LAYER = 8;

static std::vector<std::pair<int, std::string>> WEIGHT_MAPPING = {
    {IN_INPUT_NORM_WEIGHT, "norm1.weight"},
    {IN_POST_NORM_WEIGHT, "norm2.weight"},
    {IN_QKV_WEIGHT, "attn.qkv.weight"},
    {IN_ATTN_PROJ_WEIGHT, "attn.proj.weight"},
    {IN_LINEAR_GATE_WEIGHT, "mlp.gate_proj.weight"},
    {IN_LINEAR_UP_WEIGHT, "mlp.up_proj.weight"},
    {IN_LINEAR_DOWN_WEIGHT, "mlp.down_proj.weight"}};

// {weight,dim}
// IN_QKV_WEIGHT SHARD artificially in merge_loaded_weights
static std::map<int, int> WEIGHT_SHARD = {{IN_ATTN_PROJ_WEIGHT, 1},
                                          {IN_LINEAR_UP_WEIGHT, 0},
                                          {IN_LINEAR_GATE_WEIGHT, 0},
                                          {IN_LINEAR_DOWN_WEIGHT, 1}};
// TODO: check shape with atb log -- HW pxy

void NpuGlm4VisionEncoderLayerImpl::param_from_args(
    atb_speed::glm::VisionEncoderLayerParam& param,
    const ModelArgs& args,
    const ParallelArgs& parallel_args) {
  param.isBF16 = args.dtype() == "bfloat16";
  param.supportLcoc = false;
  param.supportLora = false;
  param.loraEnableGMM = false;
  param.enableLogN = false;
  param.backend = "hccl";
  param.rank = parallel_args.rank();
  param.worldSize = parallel_args.world_size();

  param.quantType = 0;
  param.quantGroupSize = 64;

  param.numAttentionHeadsPerRank =
      args.mm_num_attention_heads() / param.worldSize;
  param.hiddenSizePerAttentionHead =
      args.mm_hidden_size() / args.mm_num_attention_heads();
  std::optional<long int> optionalValue = args.mm_num_attention_heads();
  param.numKeyValueHeadsPerRank =
      static_cast<int>(optionalValue.value()) / param.worldSize;

  param.rmsNormEps = args.rms_norm_eps();
}

NpuGlm4VisionEncoderLayerImpl::NpuGlm4VisionEncoderLayerImpl(
    const ModelContext& context)
    : BaseLayer(context) {
  auto model_args = context.get_model_args();
  auto parallel_args = context.get_parallel_args();
  auto options = context.get_tensor_options();
  param_from_args(encode_param_, model_args, parallel_args);
  at_weight_tensors_.resize(WEIGHT_COUNT_PER_LAYER);
  atb_weight_tensors_.resize(WEIGHT_COUNT_PER_LAYER);
  dtype_ = c10::typeMetaToScalarType(options.dtype());
  device_id_ = options.device().index();
  placeholder_ =
      atb_speed::Utils::AtTensor2Tensor(torch::zeros({1}).to(device_).to(
          dtype_));  // seems not to be used -- HW pxy
  at_placeholder_ = torch::zeros({1}).to(device_).to(dtype_);
  for (int i = 0; i < WEIGHT_COUNT_PER_LAYER; ++i) {
    at_weight_tensors_[i] = torch::zeros({1}).to(options);
  }
}

void NpuGlm4VisionEncoderLayerImpl::verify_loaded_weights() const {
  for (const auto& [index, name] : WEIGHT_MAPPING) {
    CHECK(at_weight_tensors_[index].sizes() != std::vector<int64_t>({1}))
        << "weight is not loaded for " << name;
  }
}

void NpuGlm4VisionEncoderLayerImpl::merge_loaded_weights() {
  if (encode_param_.worldSize > 1) {
    // spilt pack qkv weight when enable tp
    get_weights_col_packed_qkv();
  }

  at_weight_tensors_[IN_LINEAR_GATE_UP_WEIGHT] =
      torch::cat({at_weight_tensors_[IN_LINEAR_GATE_WEIGHT],
                  at_weight_tensors_[IN_LINEAR_UP_WEIGHT]},
                 0);
  at_weight_tensors_[IN_LINEAR_GATE_WEIGHT] = torch::empty({}, device_);
  at_weight_tensors_[IN_LINEAR_UP_WEIGHT] = torch::empty({}, device_);

  c10_npu::NPUCachingAllocator::emptyCache();
  for (int i = 0; i < WEIGHT_COUNT_PER_LAYER; ++i) {
    atb_weight_tensors_[i] =
        atb_speed::Utils::AtTensor2Tensor(at_weight_tensors_[i]);
  }

  init_layer();
}

// tp spilt weight
void NpuGlm4VisionEncoderLayerImpl::get_weights_col_packed_qkv() {
  int rank = encode_param_.rank;
  int worldSize = encode_param_.worldSize;
  // split qkv weight
  auto qkv_weight = torch::chunk(at_weight_tensors_[IN_QKV_WEIGHT], 3, 0);
  // get local weight and merge
  auto new_qkv_weight = torch::cat({(qkv_weight[0].chunk(worldSize, 0))[rank],
                                    (qkv_weight[1].chunk(worldSize, 0))[rank],
                                    (qkv_weight[2].chunk(worldSize, 0))[rank]},
                                   0);
  at_weight_tensors_[IN_QKV_WEIGHT] = new_qkv_weight;
}

void NpuGlm4VisionEncoderLayerImpl::load_state_dict(
    const StateDict& state_dict) {
  for (const auto& [index, name] : WEIGHT_MAPPING) {
    if (WEIGHT_SHARD.find(index) != WEIGHT_SHARD.end()) {
      set_weight(state_dict, name, index, WEIGHT_SHARD[index]);
    } else {
      set_weight(state_dict, name, index);
    }
  }
}

int64_t NpuGlm4VisionEncoderLayerImpl::init_layer() {
  name_ = "glm4_vision_encoder_layer";
  model_name_ = "glm4v";
  CHECK_OPERATION_STATUS_RETURN(init_node(encode_node_, encode_param_));
  return atb::NO_ERROR;
}

int64_t NpuGlm4VisionEncoderLayerImpl::init_node(
    atb_speed::Model::Node& node,
    atb_speed::glm::VisionEncoderLayerParam& param) {
  atb::Operation* operation = nullptr;
  atb_speed::glm::Glm4v_EncoderLayer(param, &operation);
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

torch::Tensor NpuGlm4VisionEncoderLayerImpl::forward(
    torch::Tensor& x,
    torch::Tensor& cos_pos,
    torch::Tensor& sin_pos,
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
                          cu_seqlen,
                          cu_seqlen_vec,
                          input_params,
                          true);
  // mstxRangeEnd(id);
  st = execute_node(encode_node_, node_id);
  LOG_IF(FATAL, st != 0) << model_name_
                         << "excute encode layer fail, error code: " << st;
  return x;
}

void NpuGlm4VisionEncoderLayerImpl::build_node_variant_pack(
    atb_speed::Model::Node& node,
    torch::Tensor& x,
    torch::Tensor& cos_pos,
    torch::Tensor& sin_pos,
    torch::Tensor& cu_seqlen,
    std::vector<int>& cu_seqlen_vec,
    ModelInputParams& input_params,
    bool is_prefill) {
  internal_tensors_ = atb_speed::Utils::AtTensor2Tensor(x);

  auto actual_weight_num = WEIGHT_COUNT_PER_LAYER - 2;
  for (size_t i = 0; i < actual_weight_num; ++i) {
    CHECK_THROW(node.inTensors.at(i) == nullptr,
                model_name_ << "inTensor " << i << "is NULL");
    node.variantPack.inTensors.at(i) = *node.inTensors.at(i);
    // LOG(INFO) << model_name_ << "inTensors[" << i << "]:"
    //               << atb_speed::TensorUtil::TensorToString(
    //                      node.variantPack.inTensors.at(i));
  }
  node.variantPack.inTensors.at(actual_weight_num) = internal_tensors_;
  node.variantPack.inTensors.at(actual_weight_num + 1) =
      atb_speed::Utils::AtTensor2Tensor(cos_pos);
  node.variantPack.inTensors.at(actual_weight_num + 2) =
      atb_speed::Utils::AtTensor2Tensor(sin_pos);
  node.variantPack.inTensors.at(actual_weight_num + 3) =
      atb_speed::Utils::AtTensor2Tensor(cu_seqlen);
  node.variantPack.inTensors.at(actual_weight_num + 3).hostData =
      cu_seqlen_vec.data();

  node.variantPack.outTensors.at(0) = internal_tensors_;
}

}  // namespace layer
}  // namespace xllm

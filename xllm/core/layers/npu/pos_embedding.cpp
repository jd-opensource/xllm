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

#include "pos_embedding.h"

#include <glog/logging.h>

namespace xllm::hf {
std::shared_ptr<AtbRotaryEmbeddingImpl> create_pos_embedding_layer(
    const Context& context) {
  return std::make_shared<AtbRotaryEmbeddingImpl>(context);
}

AtbRotaryEmbeddingImpl::AtbRotaryEmbeddingImpl(const Context& context)
    : ATBBase(context) {
  atOutTensors_.resize(2);
  dtype_ = c10::typeMetaToScalarType(context.get_tensor_options().dtype());
  init_layer();
}

int64_t AtbRotaryEmbeddingImpl::init_layer() {
  ATBBase::name_ = "rotary_embedding_layer";
  modelName_ = "llm";
  runTaskFunc_ = std::bind(&AtbRotaryEmbeddingImpl::run_task,
                           this,
                           std::placeholders::_1,
                           std::placeholders::_2);
  CHECK_OPERATION_STATUS_RETURN(init_node(embedding_node_));

  return atb::NO_ERROR;
}

int64_t AtbRotaryEmbeddingImpl::init_node(atb_speed::Model::Node& node) {
  atb::Operation* operation = nullptr;
  CHECK_OPERATION_STATUS_RETURN(
      atb_speed::common::PositionalEmbeddingGatherV2(&operation));
  node.operation.reset(operation);
  if (node.operation == nullptr) {
    LOG(ERROR) << "node.operation is null";
    return -1;
  }

  if (node.operation->GetInputNum() < 1) {
    LOG(ERROR) << "Get unexpected input num: " << node.operation->GetInputNum();
    return -1;
  }

  if (node.operation->GetOutputNum() < 1) {
    LOG(ERROR) << "Get unexpected output num: "
               << node.operation->GetOutputNum();
    return -1;
  }

  node.inTensors.resize(node.operation->GetInputNum());
  node.outTensors.resize(node.operation->GetOutputNum());

  // node.inTensors.at(0) = &atb_weight_tensors_[0];

  node.variantPack.inTensors.reserve(node.inTensors.size());
  node.variantPack.inTensors.resize(node.inTensors.size());
  node.variantPack.outTensors.reserve(node.outTensors.size());
  node.variantPack.outTensors.resize(node.outTensors.size());

  return atb::NO_ERROR;
}

torch::Tensor AtbRotaryEmbeddingImpl::forward(const torch::Tensor& cos_sin_pos,
                                              const torch::Tensor& position,
                                              atb::Context* context,
                                              AtbWorkspace& workspace,
                                              int nodeId) {
  atb::Status st;
  build_node_variant_pack(embedding_node_, cos_sin_pos, position);
  st = execute_node(embedding_node_, context, workspace, nodeId);
  LOG_IF(FATAL, st != 0) << modelName_
                         << "infer shape fail, error code: " << st;

  return atOutTensors_.at(0);
}

void AtbRotaryEmbeddingImpl::build_node_variant_pack(
    atb_speed::Model::Node& node,
    const torch::Tensor& cos_sin_pos,
    const torch::Tensor& position) {
  internal_cos_sin_pos = atb_speed::Utils::AtTensor2Tensor(cos_sin_pos);
  internal_position = atb_speed::Utils::AtTensor2Tensor(position);

  atb::SVector<atb::TensorDesc> inTensorDescs;
  inTensorDescs.reserve(node.operation->GetInputNum());
  inTensorDescs.resize(node.operation->GetInputNum());

  atb::SVector<atb::TensorDesc> outTensorDescs;
  outTensorDescs.reserve(node.operation->GetOutputNum());
  outTensorDescs.resize(node.operation->GetOutputNum());

  node.variantPack.inTensors.at(0) = internal_position;
  inTensorDescs.at(0) = internal_position.desc;
  node.variantPack.inTensors.at(1) = internal_cos_sin_pos;
  inTensorDescs.at(1) = internal_cos_sin_pos.desc;

  node.operation->InferShape(inTensorDescs, outTensorDescs);

  at::Tensor embedding =
      atb_speed::Utils::CreateAtTensorFromTensorDesc(outTensorDescs.at(0));
  atOutTensors_.at(0) = embedding;
  node.variantPack.outTensors.at(0) =
      atb_speed::Utils::AtTensor2Tensor(atOutTensors_.at(0));
}

AtbRotaryEmbedding::AtbRotaryEmbedding(const Context& context)
    : ModuleHolder(create_pos_embedding_layer(context)) {}

}  // namespace xllm::hf

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

#include "npu_word_embedding_impl.h"

#include <gflags/gflags.h>
#include <glog/logging.h>
// DECLARE_string(rank_tablefile);
DECLARE_string(communication_backend);
namespace xllm {
namespace layer {

void NpuWordEmbeddingImpl::param_from_args(
    atb_speed::common::WordEmbeddingParam& param,
    const xllm::ModelArgs& args,
    const xllm::ParallelArgs& parallel_args) {
  param.unpadInputs = true;
  if (dp_size_ > 1) {
    param.tensorParallelInfo.rank = dp_local_tp_rank_;
    param.tensorParallelInfo.worldSize = dp_local_tp_size_;
    param.tensorParallelInfo.backend = "lccl";
  } else if (parallel_args.world_size() != 1) {
    // param.tensorParallelInfo = {parallel_args.rank(),
    // parallel_args.world_size(), "lccl"};
    param.tensorParallelInfo = {
        parallel_args.rank(), parallel_args.world_size(), "lccl"};
  }
  // param.linearParallelParam.tensorParallelInfo.backend =
  // FLAGS_communication_backend;
  param.tensorParallelInfo.commDomain = std::to_string(dp_rank_);
  // param.tensorParallelInfo.rankTableFile = FLAGS_rank_tablefile;
}

NpuWordEmbeddingImpl::NpuWordEmbeddingImpl(const ModelContext& context)
    : NpuBaseLayer(context) {
  auto model_args = context.get_model_args();
  auto parallel_args = context.get_parallel_args();
  auto options = context.get_tensor_options();

  param_from_args(embedding_param_, model_args, parallel_args);
  at_weight_tensors_.resize(1);
  atb_weight_tensors_.resize(1);
  atOutTensors_.resize(1);
  dtype_ = c10::typeMetaToScalarType(options.dtype());
  at_weight_tensors_[0] = torch::zeros({1}).to(options);
}

void NpuWordEmbeddingImpl::verify_loaded_weights(
    const std::string weight_str) const {
  CHECK(at_weight_tensors_[0].sizes() != std::vector<int64_t>({1}))
      << "weight is not loaded for " << weight_str;
}

void NpuWordEmbeddingImpl::merge_loaded_weights() {
  atb_weight_tensors_[0] =
      atb_speed::Utils::AtTensor2Tensor(at_weight_tensors_[0]);
  init_layer();
}

void NpuWordEmbeddingImpl::load_state_dict(const StateDict& state_dict) {
  if (dp_size_ > 1) {
    set_weight(
        state_dict, "weight", 0, 1, dp_local_tp_rank_, dp_local_tp_size_);
  } else {
    set_weight(state_dict, "weight", 0, 1);
  }
}

int64_t NpuWordEmbeddingImpl::init_layer() {
  NpuBaseLayer::name_ = "word_embedding_layer";
  modelName_ = "llm";
  CHECK_OPERATION_STATUS_RETURN(init_node(embedding_node_, embedding_param_));
  return atb::NO_ERROR;
}

int64_t NpuWordEmbeddingImpl::init_node(
    atb_speed::Model::Node& node,
    atb_speed::common::WordEmbeddingParam& param) {
  atb::Operation* operation = nullptr;
  atb_speed::common::WordEmbedding(param, &operation);
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
  // node.outTensors.resize(1);

  node.inTensors.at(0) = &atb_weight_tensors_[0];

  node.variantPack.inTensors.reserve(node.inTensors.size());
  node.variantPack.inTensors.resize(node.inTensors.size());
  node.variantPack.outTensors.reserve(1);
  node.variantPack.outTensors.resize(1);

  return atb::NO_ERROR;
}

torch::Tensor NpuWordEmbeddingImpl::forward(const torch::Tensor& x,
                                            int nodeId) {
  atb::Status st;
  // std::cout<<"x:"<<x<<std::endl;
  build_node_variant_pack(embedding_node_, x);
  st = execute_node(embedding_node_, nodeId);
  LOG_IF(FATAL, st != 0) << modelName_
                         << "infer shape fail, error code: " << st;
  return atOutTensors_.at(0);
}

void NpuWordEmbeddingImpl::build_node_variant_pack(atb_speed::Model::Node& node,
                                                   const torch::Tensor& x) {
  if (!node.operation) {
    throw std::runtime_error(
        "node.operation is null in build_node_variant_pack");
  }

  internalTensors = atb_speed::Utils::AtTensor2Tensor(x);
  // node.outTensors[0] = &internalTensors;

  atb::SVector<atb::TensorDesc> inTensorDescs;
  inTensorDescs.reserve(node.variantPack.inTensors.size());
  inTensorDescs.resize(node.variantPack.inTensors.size());

  atb::SVector<atb::TensorDesc> outTensorDescs;

  auto output_num = node.operation->GetOutputNum();
  if (output_num <= 0) {
    throw std::runtime_error("Invalid output number: " +
                             std::to_string(output_num));
  }

  outTensorDescs.reserve(node.operation->GetOutputNum());
  outTensorDescs.resize(node.operation->GetOutputNum());

  node.variantPack.inTensors.at(0) = *node.inTensors.at(0);
  inTensorDescs.at(0) = node.inTensors.at(0)->desc;

  node.variantPack.inTensors.at(1) = internalTensors;
  inTensorDescs.at(1) = internalTensors.desc;

  atb::Status st = node.operation->InferShape(inTensorDescs, outTensorDescs);

  at::Tensor newTensor =
      atb_speed::Utils::CreateAtTensorFromTensorDesc(outTensorDescs.at(0));

  atOutTensors_.at(0) = newTensor;

  node.variantPack.outTensors.at(0) =
      atb_speed::Utils::AtTensor2Tensor(atOutTensors_.at(0));
}

}  // namespace layer
}  // namespace xllm

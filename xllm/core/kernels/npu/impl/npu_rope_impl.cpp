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

#include "npu_rope_impl.h"

#include <glog/logging.h>

namespace xllm::kernel {

void NpuRopeImpl::param_from_args(atb::infer::RopeParam& param,
                                  const ModelArgs& args) {
  param.rotaryCoeff = 2;
}

int64_t NpuRopeImpl::init_node(atb_speed::Model::Node& node,
                               atb::infer::RopeParam& param) {
  name_ = "rope";
  model_name_ = "llm";
  run_task_func_ = std::bind(&NpuRopeImpl::run_task,
                             this,
                             std::placeholders::_1,
                             std::placeholders::_2);

  atb::Operation* operation = nullptr;
  atb::Status atbStatus = atb::CreateOperation(param, &operation);
  if (atbStatus != atb::NO_ERROR) {
    return atbStatus;
  }

  node.operation.reset(operation);
  if (node.operation == nullptr) {
    LOG(ERROR) << "node.operation is null";
    return -1;
  }
  if (node.operation->GetInputNum() < 1) {
    LOG(ERROR) << "Can not resize number which is smaller than 1";
    return -1;
  }

  return atb::NO_ERROR;
}

NpuRopeImpl::NpuRopeImpl(const ModelContext& context) : NpuBaseLayer(context) {
  param_from_args(rope_param_, context.get_model_args());

  at_weight_tensors_.resize(1);
  atb_weight_tensors_.resize(1);

  auto options = context.get_tensor_options();
  dtype_ = c10::typeMetaToScalarType(options.dtype());
  at_weight_tensors_[0] = torch::zeros({1}).to(options);

  atb::Status status = init_node(rope_node_, rope_param_);
  if (status != atb::NO_ERROR) {
    LOG(ERROR) << "Failed to initialize node, status: " << status;
    throw std::runtime_error("NpuRopeImpl initialization failed with status: " +
                             std::to_string(status));
  }
}

void NpuRopeImpl::verify_loaded_weights(const std::string weight_str) const {
  // No operation needed for rope layer
}

void NpuRopeImpl::merge_loaded_weights() {
  // No operation needed for rope layer
}

void NpuRopeImpl::load_state_dict(const StateDict& state_dict) {
  // No operation needed for rope layer
}

std::vector<at::Tensor> NpuRopeImpl::forward(const torch::Tensor& q,
                                             const torch::Tensor& k,
                                             const torch::Tensor& cos_embedding,
                                             const torch::Tensor& sin_embedding,
                                             const torch::Tensor& seq_len,
                                             int nodeId) {
  atb::Status st;
  build_node_variant_pack(
      rope_node_, q, k, cos_embedding, sin_embedding, seq_len);
  st = execute_node(rope_node_, nodeId);
  LOG_IF(FATAL, st != 0) << model_name_
                         << "infer shape fail, error code: " << st;
  return at_out_tensors_;
}

void NpuRopeImpl::build_node_variant_pack(atb_speed::Model::Node& node,
                                          const torch::Tensor& q,
                                          const torch::Tensor& k,
                                          const torch::Tensor& cos_embedding,
                                          const torch::Tensor& sin_embedding,
                                          const torch::Tensor& seq_len) {
  internal_q = atb_speed::Utils::AtTensor2Tensor(q);
  internal_k = atb_speed::Utils::AtTensor2Tensor(k);
  internal_cos_embedding = atb_speed::Utils::AtTensor2Tensor(cos_embedding);
  internal_sin_embedding = atb_speed::Utils::AtTensor2Tensor(sin_embedding);
  internal_seq_len = atb_speed::Utils::AtTensor2Tensor(seq_len);

  atb::SVector<atb::Tensor> ins = {internal_q,
                                   internal_k,
                                   internal_cos_embedding,
                                   internal_sin_embedding,
                                   internal_seq_len};
  node.variantPack.inTensors = ins;

  atb::SVector<atb::TensorDesc> inTensorDescs;
  inTensorDescs.resize(node.operation->GetInputNum());
  inTensorDescs.at(0) = internal_q.desc;
  inTensorDescs.at(1) = internal_k.desc;
  inTensorDescs.at(2) = internal_cos_embedding.desc;
  inTensorDescs.at(3) = internal_sin_embedding.desc;
  inTensorDescs.at(4) = internal_seq_len.desc;

  atb::SVector<atb::TensorDesc> outTensorDescs;
  node.operation->InferShape(inTensorDescs, outTensorDescs);

  at_out_tensors_.resize(outTensorDescs.size());
  at::Tensor output_0 =
      atb_speed::Utils::CreateAtTensorFromTensorDesc(outTensorDescs.at(0));
  at_out_tensors_.at(0) = output_0;
  at::Tensor output_1 =
      atb_speed::Utils::CreateAtTensorFromTensorDesc(outTensorDescs.at(1));
  at_out_tensors_.at(1) = output_1;

  node.variantPack.outTensors = {atb_speed::Utils::AtTensor2Tensor(output_0),
                                 atb_speed::Utils::AtTensor2Tensor(output_1)};
}

}  // namespace xllm::kernel

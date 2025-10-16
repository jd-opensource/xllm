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

#include "npu_split_impl.h"

#include <glog/logging.h>

namespace xllm::kernel {

void NpuSplitImpl::param_from_args(atb::infer::SplitParam& param,
                                   const ModelArgs& args,
                                   int32_t splitDim,
                                   int32_t splitNum,
                                   atb::SVector<int32_t> splitSizes) {
  param.splitDim = splitDim;
  param.splitNum = splitNum;
  param.splitSizes = splitSizes;
}

int64_t NpuSplitImpl::init_node(atb_speed::Model::Node& node,
                                atb::infer::SplitParam& param) {
  name_ = "split";
  model_name_ = "llm";
  run_task_func_ = std::bind(&NpuSplitImpl::run_task,
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

NpuSplitImpl::NpuSplitImpl(const ModelContext& context,
                           int32_t splitDim,
                           int32_t splitNum,
                           atb::SVector<int32_t> splitSizes)
    : NpuBaseLayer(context) {
  param_from_args(
      split_param_, context.get_model_args(), splitDim, splitNum, splitSizes);

  at_weight_tensors_.resize(1);
  atb_weight_tensors_.resize(1);
  at_out_tensors_.resize(3);

  auto options = context.get_tensor_options();
  dtype_ = c10::typeMetaToScalarType(options.dtype());
  at_weight_tensors_[0] = torch::zeros({1}).to(options);

  atb::Status status = init_node(split_node_, split_param_);
  if (status != atb::NO_ERROR) {
    LOG(ERROR) << "Failed to initialize node, status: " << status;
    throw std::runtime_error(
        "NpuSplitImpl initialization failed with status: " +
        std::to_string(status));
  }
}

void NpuSplitImpl::verify_loaded_weights(const std::string weight_str) const {
  // No operation needed for split layer
}

void NpuSplitImpl::merge_loaded_weights() {
  // No operation needed for split layer
}

void NpuSplitImpl::load_state_dict(const StateDict& state_dict) {
  // No operation needed for split layer
}

std::vector<at::Tensor> NpuSplitImpl::forward(const torch::Tensor& input,
                                              int nodeId) {
  atb::Status st;
  build_node_variant_pack(split_node_, input);
  st = execute_node(split_node_, nodeId);
  LOG_IF(FATAL, st != 0) << model_name_
                         << "infer shape fail, error code: " << st;
  return at_out_tensors_;
}

void NpuSplitImpl::build_node_variant_pack(atb_speed::Model::Node& node,
                                           const torch::Tensor& input) {
  internal_input = atb_speed::Utils::AtTensor2Tensor(input);

  atb::SVector<atb::Tensor> ins = {internal_input};
  node.variantPack.inTensors = ins;

  atb::SVector<atb::TensorDesc> inTensorDescs;
  inTensorDescs.resize(node.operation->GetInputNum());
  inTensorDescs.at(0) = internal_input.desc;

  atb::SVector<atb::TensorDesc> outTensorDescs;
  node.operation->InferShape(inTensorDescs, outTensorDescs);

  at::Tensor output_0 =
      atb_speed::Utils::CreateAtTensorFromTensorDesc(outTensorDescs.at(0));
  at_out_tensors_.at(0) = output_0;
  at::Tensor output_1 =
      atb_speed::Utils::CreateAtTensorFromTensorDesc(outTensorDescs.at(1));
  at_out_tensors_.at(1) = output_1;
  at::Tensor output_2 =
      atb_speed::Utils::CreateAtTensorFromTensorDesc(outTensorDescs.at(2));
  at_out_tensors_.at(2) = output_2;

  node.variantPack.outTensors = {atb_speed::Utils::AtTensor2Tensor(output_0),
                                 atb_speed::Utils::AtTensor2Tensor(output_1),
                                 atb_speed::Utils::AtTensor2Tensor(output_2)};
}

}  // namespace xllm::kernel

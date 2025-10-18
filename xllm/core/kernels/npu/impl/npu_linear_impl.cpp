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

#include "npu_linear_impl.h"

#include <glog/logging.h>

namespace xllm::kernel {

NpuLinearImpl::NpuLinearImpl(const ModelContext& context)
    : NpuBaseLayer(context) {
  at_weight_tensors_.resize(1);
  atb_weight_tensors_.resize(1);
  at_out_tensors_.resize(1);
  dtype_ = c10::typeMetaToScalarType(context.get_tensor_options().dtype());
  at_weight_tensors_[0] = torch::zeros({1}).to(context.get_tensor_options());
  tensor_placeholder_ = torch::zeros({1}).to(context.get_tensor_options());

  atb::Status status = init_node(linear_node_);
  if (status != atb::NO_ERROR) {
    LOG(ERROR) << "Failed to initialize node, status: " << status;
    throw std::runtime_error(
        "NpuLinearImpl initialization failed with status: " +
        std::to_string(status));
  }
}

void NpuLinearImpl::verify_loaded_weights(const std::string weight_str) const {
  CHECK(at_weight_tensors_[0].sizes() != std::vector<int64_t>({1}))
      << "weight is not loaded for " << weight_str;
}

void NpuLinearImpl::merge_loaded_weights() {
  atb_weight_tensors_[0] =
      atb_speed::Utils::AtTensor2Tensor(at_weight_tensors_[0]);
}

void NpuLinearImpl::load_state_dict(const StateDict& state_dict) {
  set_weight(state_dict, "weight", 0);
}

int64_t NpuLinearImpl::init_node(atb_speed::Model::Node& node) {
  name_ = "linear";
  model_name_ = "llm";
  run_task_func_ = std::bind(&NpuLinearImpl::run_task,
                             this,
                             std::placeholders::_1,
                             std::placeholders::_2);

  atb::Operation* operation = nullptr;
  atb::infer::LinearParam linearParam;
  linearParam.transposeB = true;
  // linearParam.outDataType = ACL_BF16;
  linearParam.hasBias = false;
  atb::Status atbStatus = atb::CreateOperation(linearParam, &operation);
  if (atbStatus != atb::NO_ERROR) {
    return atbStatus;
  }

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
  ATB_SPEED_LOG_DEBUG("AddLinear");

  return atb::NO_ERROR;
}

torch::Tensor NpuLinearImpl::forward(const torch::Tensor& input, int nodeId) {
  atb::Status st;

  build_node_variant_pack(linear_node_, input);

  st = execute_node(linear_node_, nodeId);

  if (st != 0) {
    LOG(ERROR) << model_name_ << " infer shape fail, error code: " << st;
    throw std::runtime_error(
        model_name_ +
        " inference failed with error code: " + std::to_string(st));
  }

  return at_out_tensors_.at(0);
}

void NpuLinearImpl::build_node_variant_pack(atb_speed::Model::Node& node,
                                            const torch::Tensor& input) {
  internal_input = atb_speed::Utils::AtTensor2Tensor(input);

  atb::SVector<atb::Tensor> ins = {internal_input, atb_weight_tensors_[0]};
  node.variantPack.inTensors = ins;

  atb::SVector<atb::TensorDesc> inTensorDescs;
  inTensorDescs.resize(node.operation->GetInputNum());
  inTensorDescs.at(0) = internal_input.desc;
  inTensorDescs.at(1) = atb_weight_tensors_[0].desc;

  atb::SVector<atb::TensorDesc> outTensorDescs;
  node.operation->InferShape(inTensorDescs, outTensorDescs);

  at::Tensor output =
      atb_speed::Utils::CreateAtTensorFromTensorDesc(outTensorDescs.at(0));
  at_out_tensors_.at(0) = output;

  node.variantPack.outTensors = {atb_speed::Utils::AtTensor2Tensor(output)};
}

}  // namespace xllm::kernel

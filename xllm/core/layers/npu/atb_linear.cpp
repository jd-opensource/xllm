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

#include "atb_linear.h"

#include <glog/logging.h>

#include "xllm_kernels/operations/aclnn/ops/repeat_operation.h"
#include "xllm_kernels/operations/fusion/linear/linear.h"
#include "xllm_kernels/operations/fusion/linear/linear_parallel.h"
#include "xllm_kernels/operations/fusion/utils.h"

namespace xllm::hf {
std::shared_ptr<AtbLinearImpl> create_atb_linear_layer(
    const ModelContext& context) {
  return std::make_shared<AtbLinearImpl>(context);
}

AtbLinearImpl::AtbLinearImpl(const ModelContext& context) : ATBBase(context) {
  at_weight_tensors_.resize(1);
  atb_weight_tensors_.resize(1);
  at_out_tensors_.resize(1);
  dtype_ = c10::typeMetaToScalarType(context.get_tensor_options().dtype());
  at_weight_tensors_[0] = torch::zeros({1}).to(context.get_tensor_options());
  tensor_placeholder_ = torch::zeros({1}).to(context.get_tensor_options());
}

void AtbLinearImpl::verify_loaded_weights(const std::string weight_str) const {
  CHECK(at_weight_tensors_[0].sizes() != std::vector<int64_t>({1}))
      << "weight is not loaded for " << weight_str;
}

void AtbLinearImpl::merge_loaded_weights() {
  atb_weight_tensors_[0] =
      atb_speed::Utils::AtTensor2Tensor(at_weight_tensors_[0]);

  init_layer();
}

void AtbLinearImpl::load_state_dict(const StateDict& state_dict) {
  set_weight(state_dict, "weight", 0);
}

int64_t AtbLinearImpl::init_layer() {
  ATBBase::name_ = "atb_linear_layer";
  model_name_ = "Atb Linear";
  runTaskFunc_ = std::bind(&AtbLinearImpl::run_task,
                           this,
                           std::placeholders::_1,
                           std::placeholders::_2);
  CHECK_OPERATION_STATUS_RETURN(init_node(linear_node_));

  return atb::NO_ERROR;
}

int64_t AtbLinearImpl::init_node(atb_speed::Model::Node& node) {
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
  node.inTensors.resize(node.operation->GetInputNum());
  node.outTensors.resize(1);

  node.inTensors.at(1) = &atb_weight_tensors_[0];

  node.variantPack.inTensors.reserve(node.inTensors.size());
  node.variantPack.inTensors.resize(node.inTensors.size());
  node.variantPack.outTensors.reserve(1);
  node.variantPack.outTensors.resize(1);
  ATB_SPEED_LOG_DEBUG("AddLinear");

  return atb::NO_ERROR;
}

torch::Tensor AtbLinearImpl::forward(const torch::Tensor& input, int nodeId) {
  atb::Status st;

  build_node_variant_pack(linear_node_, input);
  st = execute_node(linear_node_, nodeId);
  LOG_IF(FATAL, st != 0) << model_name_
                         << "infer shape fail, error code: " << st;

  return at_out_tensors_.at(0);
}

void AtbLinearImpl::build_node_variant_pack(atb_speed::Model::Node& node,
                                            const torch::Tensor& input) {
  internal_input = atb_speed::Utils::AtTensor2Tensor(input);

  atb::SVector<atb::TensorDesc> inTensorDescs;
  inTensorDescs.reserve(node.operation->GetInputNum());
  inTensorDescs.resize(node.operation->GetInputNum());

  atb::SVector<atb::TensorDesc> outTensorDescs;
  outTensorDescs.reserve(node.operation->GetOutputNum());
  outTensorDescs.resize(node.operation->GetOutputNum());

  node.variantPack.inTensors.at(0) = internal_input;
  inTensorDescs.at(0) = internal_input.desc;
  // weight
  node.variantPack.inTensors.at(1) = *node.inTensors.at(1);
  inTensorDescs.at(1) = node.inTensors.at(1)->desc;

  outTensorDescs.at(0).format = inTensorDescs.at(0).format;
  outTensorDescs.at(0).dtype = ACL_BF16;
  outTensorDescs.at(0).shape = inTensorDescs.at(0).shape;
  auto outDimSize = outTensorDescs.at(0).shape.dimNum;
  // int nDim = param.transposeType == TransposeType::TRANSPOSE ? 0 : 1;
  int nDim = 0;
  if (inTensorDescs.at(1).shape.dimNum == 3) {  // 3: dimNum
    outTensorDescs.at(0).shape.dims[outDimSize - 1] =
        inTensorDescs.at(1).shape.dims[nDim + 1];
  } else {
    outTensorDescs.at(0).shape.dims[outDimSize - 1] =
        inTensorDescs.at(1).shape.dims[nDim];
  }

  at::Tensor output =
      atb_speed::Utils::CreateAtTensorFromTensorDesc(outTensorDescs.at(0));
  at_out_tensors_.at(0) = output;
  node.variantPack.outTensors.at(0) =
      atb_speed::Utils::AtTensor2Tensor(at_out_tensors_.at(0));
}

AtbLinear::AtbLinear(const ModelContext& context)
    : ModuleHolder(create_atb_linear_layer(context)) {}

}  // namespace xllm::hf

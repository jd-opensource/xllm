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

#include "rms_norm.h"

#include <glog/logging.h>

#include "layers/npu/attn_mask.h"

namespace xllm::hf {

std::shared_ptr<RmsNormImpl> create_rms_norm_layer(
    const ModelContext& context) {
  return std::make_shared<RmsNormImpl>(context);
}

void RmsNormImpl::param_from_args(atb::infer::RmsNormParam& param,
                                  const ModelArgs& args) {
  param.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
  param.normParam.epsilon = args.rms_norm_eps();
}

int64_t RmsNormImpl::init_node(atb_speed::Model::Node& node,
                               atb::infer::RmsNormParam& param) {
  ATBBase::name_ = "rms_norm_layer";
  model_name_ = "llm";
  runTaskFunc_ = std::bind(&RmsNormImpl::run_task,
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

RmsNormImpl::RmsNormImpl(const ModelContext& context) : ATBBase(context) {
  param_from_args(norm_param_, context.get_model_args());

  at_weight_tensors_.resize(1);
  atb_weight_tensors_.resize(1);

  auto options = context.get_tensor_options();
  dtype_ = c10::typeMetaToScalarType(options.dtype());
  at_weight_tensors_[0] = torch::zeros({1}).to(options);

  atb::Status status = init_node(norm_node_, norm_param_);
  if (status != atb::NO_ERROR) {
    LOG(ERROR) << "Failed to initialize node, status: " << status;
    throw std::runtime_error("RmsNormImpl initialization failed with status: " +
                             std::to_string(status));
  }
}

void RmsNormImpl::verify_loaded_weights(const std::string weight_str) const {
  CHECK(at_weight_tensors_[0].sizes() != std::vector<int64_t>({1}))
      << "final norm weight is not loaded for " << weight_str;
}

void RmsNormImpl::merge_loaded_weights() {
  atb_weight_tensors_[0] =
      atb_speed::Utils::AtTensor2Tensor(at_weight_tensors_[0]);
}

void RmsNormImpl::load_state_dict(const StateDict& state_dict) {
  set_weight(state_dict, "weight", 0);
  at_weight_tensors_[0] = at_weight_tensors_[0].to(dtype_);
}

torch::Tensor RmsNormImpl::forward(torch::Tensor& x, int nodeId) {
  atb::Status st;
  build_node_variant_pack(norm_node_, x);
  st = execute_node(norm_node_, nodeId);
  LOG_IF(FATAL, st != 0) << model_name_
                         << "infer shape fail, error code: " << st;
  return x;
}

void RmsNormImpl::build_node_variant_pack(atb_speed::Model::Node& node,
                                          torch::Tensor& x) {
  internal_tensors_ = atb_speed::Utils::AtTensor2Tensor(x);

  atb::SVector<atb::Tensor> ins = {internal_tensors_, atb_weight_tensors_[0]};
  atb::SVector<atb::Tensor> outs = {internal_tensors_};

  node.variantPack.inTensors = ins;
  node.variantPack.outTensors = outs;
}

RmsNorm::RmsNorm(const ModelContext& context)
    : ModuleHolder(create_rms_norm_layer(context)) {}

}  // namespace xllm::hf

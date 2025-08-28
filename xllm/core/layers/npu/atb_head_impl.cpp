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

#include "atb_head_impl.h"

#include <gflags/gflags.h>
#include <glog/logging.h>
DECLARE_string(rank_tablefile);
DECLARE_string(communication_backend);

namespace xllm::hf {

void AtbLmHeadImpl::param_from_args(atb_speed::common::LmHeadParam& param,
                                    const ModelArgs& args,
                                    const ParallelArgs& parallel_args,
                                    bool isPrefill) {
  param.unpadInputs = true;
  param.gatherAhead = isPrefill;
  param.hiddenSizePerAttentionHead = args.hidden_size() / args.n_heads();
  param.linearParallelParam.fusionLinearParam.isBF16 =
      args.dtype() == "bfloat16";
  param.linearParallelParam.unpadInputs = true;
  param.linearParallelParam.fusionLinearParam.transposeType = 1;
  if (parallel_args.world_size() > 1) {
    if (parallel_args.mapping_data().empty()) {
      if (dp_size_ > 1) {
        param.linearParallelParam.tensorParallelInfo.rank = dp_local_tp_rank_;
        param.linearParallelParam.tensorParallelInfo.worldSize =
            dp_local_tp_size_;
      } else {
        param.linearParallelParam.tensorParallelInfo.rank =
            parallel_args.rank();
        param.linearParallelParam.tensorParallelInfo.worldSize =
            parallel_args.world_size();
      }
      param.linearParallelParam.parallelType =
          atb_speed::common::COLUMN_PARALLEL;
      param.linearParallelParam.tensorParallelInfo.commDomain =
          std::to_string(dp_rank_);
      param.linearParallelParam.tensorParallelInfo.backend = "lccl";
    } else {
      param.linearParallelParam.parallelType =
          atb_speed::common::COLUMN_PARALLEL;
      atb_speed::common::ParallelInfo parallelInfo =
          parallel_args.mapping().Get(atb_speed::base::ATTN_TP);
      param.linearParallelParam.tensorParallelInfo.rank = parallelInfo.rank;
      param.linearParallelParam.tensorParallelInfo.worldSize =
          parallelInfo.rankIds.size();
      param.linearParallelParam.tensorParallelInfo.backend =
          parallelInfo.backend;
      param.linearParallelParam.tensorParallelInfo.hcommInfo =
          parallelInfo.hcclComm;
      param.linearParallelParam.tensorParallelInfo.commDomain =
          parallelInfo.commDomain;
    }
  }
}

AtbLmHeadImpl::AtbLmHeadImpl(const Context& context) : ATBBase(context) {
  param_from_args(llm_head_param_prefill_,
                  context.get_model_args(),
                  context.get_parallel_args(),
                  true);

  param_from_args(llm_head_param_decode_,
                  context.get_model_args(),
                  context.get_parallel_args(),
                  false);

  at_weight_tensors_.resize(1);
  atb_weight_tensors_.resize(1);
  atOutTensors_.resize(1);

  auto options = context.get_tensor_options();
  dtype_ = c10::typeMetaToScalarType(options.dtype());
  at_weight_tensors_[0] = torch::zeros({1}).to(options);
  prefill_tensor_storage_.resize(2);
  decode_tensor_storage_.resize(2);

  torch_placeholder_ = torch::zeros({1}).to(device_).to(dtype_);
  placeholder_ = atb_speed::Utils::AtTensor2Tensor(torch_placeholder_);
}

void AtbLmHeadImpl::verify_loaded_weights(const std::string weight_str) const {
  // std::cout<<at_weight_tensors_[0]<<std::endl;
  // std::cout<<at_weight_tensors_[0].sizes()<<std::endl;
  CHECK(at_weight_tensors_[0].sizes() != std::vector<int64_t>({1}))
      << "final llm_head weight is not loaded for " << weight_str;
}

void AtbLmHeadImpl::merge_loaded_weights() {
  atb_weight_tensors_[0] =
      atb_speed::Utils::AtTensor2Tensor(at_weight_tensors_[0]);
  init_layer();
}

void AtbLmHeadImpl::load_state_dict(const StateDict& state_dict) {
  // set_weight(state_dict, "weight", 0, 0);
  if (dp_size_ > 1) {
    set_weight(
        state_dict, "weight", 0, 0, dp_local_tp_rank_, dp_local_tp_size_);
  } else {
    set_weight(state_dict, "weight", 0, 0);
  }
}

int64_t AtbLmHeadImpl::init_layer() {
  ATBBase::name_ = "llm_head_layer";
  model_name_ = "llm";
  runTaskFunc_ = std::bind(&AtbLmHeadImpl::run_task,
                           this,
                           std::placeholders::_1,
                           std::placeholders::_2);
  CHECK_OPERATION_STATUS_RETURN(
      init_node(llm_head_node_prefill_, llm_head_param_prefill_));
  CHECK_OPERATION_STATUS_RETURN(
      init_node(llm_head_node_decode_, llm_head_param_decode_));

  return atb::NO_ERROR;
}

int64_t AtbLmHeadImpl::init_node(atb_speed::Model::Node& node,
                                 atb_speed::common::LmHeadParam& param) {
  atb::Operation* operation = nullptr;
  atb::Status atbStatus = atb_speed::common::LmHead(param, &operation);
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
  node.inTensors.resize(node.operation->GetInputNum());
  node.outTensors.resize(1);

  node.inTensors.at(1) = &atb_weight_tensors_[0];

  node.variantPack.inTensors.reserve(node.inTensors.size());
  node.variantPack.inTensors.resize(node.inTensors.size());
  node.variantPack.outTensors.reserve(1);
  node.variantPack.outTensors.resize(1);

  return atb::NO_ERROR;
}

torch::Tensor AtbLmHeadImpl::forward(const torch::Tensor& hidden_states,
                                     const torch::Tensor& seleted_idxes,
                                     atb::Context* context,
                                     AtbWorkspace& workspace,
                                     int nodeId) {
  atb::Status st;
  build_node_variant_pack(llm_head_node_prefill_, hidden_states, seleted_idxes);
  st = execute_node(llm_head_node_prefill_, context, workspace, nodeId);
  // if (is_prefill) {
  //   build_node_variant_pack(llm_head_node_prefill_,
  //   hidden_states,seleted_idxes); st = execute_node(llm_head_node_prefill_,
  //   context, workspace ,nodeId);
  // } else {
  //   build_node_variant_pack(llm_head_node_decode_,
  //   hidden_states,seleted_idxes); st = execute_node(llm_head_node_decode_,
  //   context, workspace ,nodeId);
  // }
  // c10_npu::NPUCachingAllocator::emptyCache();
  LOG_IF(FATAL, st != 0) << model_name_
                         << "execute llmhead node fail, error code: " << st;
  return atOutTensors_[0];
}

void AtbLmHeadImpl::build_node_variant_pack(
    atb_speed::Model::Node& node,
    const torch::Tensor& hidden_states,
    const torch::Tensor& seleted_idxes) {
  hidden_states_atb_ = atb_speed::Utils::AtTensor2Tensor(hidden_states);
  seleted_idxes_atb_ = atb_speed::Utils::AtTensor2Tensor(seleted_idxes);
  // node.outTensors[0] = &internalTensors;

  atb::SVector<atb::TensorDesc> inTensorDescs;
  inTensorDescs.reserve(node.variantPack.inTensors.size());
  inTensorDescs.resize(node.variantPack.inTensors.size());
  atb::SVector<atb::TensorDesc> outTensorDescs;
  outTensorDescs.reserve(node.operation->GetOutputNum());
  outTensorDescs.resize(node.operation->GetOutputNum());

  node.variantPack.inTensors.at(0) = hidden_states_atb_;
  inTensorDescs.at(0) = hidden_states_atb_.desc;

  node.variantPack.inTensors.at(1) = *node.inTensors.at(1);
  inTensorDescs.at(1) = node.inTensors.at(1)->desc;

  node.variantPack.inTensors.at(2) = placeholder_;
  inTensorDescs.at(2) = placeholder_.desc;

  node.variantPack.inTensors.at(3) = placeholder_;
  inTensorDescs.at(3) = placeholder_.desc;

  node.variantPack.inTensors.at(4) = placeholder_;
  inTensorDescs.at(4) = placeholder_.desc;

  node.variantPack.inTensors.at(5) = placeholder_;
  inTensorDescs.at(5) = placeholder_.desc;

  node.variantPack.inTensors.at(6) = placeholder_;
  inTensorDescs.at(6) = placeholder_.desc;

  node.variantPack.inTensors.at(7) = seleted_idxes_atb_;
  inTensorDescs.at(7) = seleted_idxes_atb_.desc;

  node.variantPack.inTensors.at(8) = placeholder_;
  inTensorDescs.at(8) = placeholder_.desc;

  atb::Status st = node.operation->InferShape(inTensorDescs, outTensorDescs);
  at::Tensor newTensor =
      atb_speed::Utils::CreateAtTensorFromTensorDesc(outTensorDescs.at(0));

  atOutTensors_.at(0) = newTensor;
  node.variantPack.outTensors.at(0) =
      atb_speed::Utils::AtTensor2Tensor(atOutTensors_.at(0));
}

}  // namespace xllm::hf

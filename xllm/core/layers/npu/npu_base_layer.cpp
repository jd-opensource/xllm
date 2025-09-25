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

#include "npu_base_layer.h"

#ifdef TORCH_HIGHER_THAN_PTA6
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#else
#include <torch_npu/csrc/aten/NPUNativeFunctions.h>
#include <torch_npu/csrc/framework/utils/OpPreparation.h>
#endif
#include "core/common/global_flags.h"

namespace xllm {
namespace layer {

NpuBaseLayer::NpuBaseLayer(const ModelContext& context) : BaseLayer(context) {
  context_ = const_cast<atb::Context*>(context.get_atb_context());
  work_space_ = AtbWorkspace(device_);
}

atb::Status NpuBaseLayer::execute_node(
    atb_speed::Model::Node& node,
    int node_id,
    std::vector<aclrtEvent*> event,
    std::vector<std::atomic<bool>*> event_flag) {
  // TODO（by zhangminchao1@jd.com): Stream management needs to be refactored
  // for better separation of concerns Current issues:
  // 1. ACLGraph capture requires execution on a non-default stream, so we
  // temporarily set the current stream
  // 2. After ACLGraph capture ends, the stream will be modified back to the
  // default stream
  // 3. In non-ACL graph capture mode, the context stream should be set to the
  // default stream
  // 4. The actual requirement is to separate decode node context from prefill
  // node context
  //
  // Note: The commented code below will cause runtime errors because:
  // - aclmdlRICaptureGetInfo() may fail when called at inappropriate times
  // - The capture status check logic is not robust enough for all scenarios
  // - Stream management conflicts: ATB context stream must be consistent with
  // libtorch_npu current stream.
  //   However, libtorch_npu current stream is set to default stream after
  //   capture ends, causing inconsistency between ATB context and the actual
  //   execution stream
  if (FLAGS_enable_acl_graph) {
    void* stream = c10_npu::getCurrentNPUStream(device_.index()).stream();
    context_->SetExecuteStream(stream);
  }
  // if (FLAGS_enable_acl_graph && !graph_captured_) {
  //   void* stream = c10_npu::getCurrentNPUStream(device_.index()).stream();
  //   aclmdlRICaptureStatus status;
  //   aclmdlRI modelRI;
  //   auto error = aclmdlRICaptureGetInfo(stream, &status, &modelRI);
  //   if (error != ACL_SUCCESS) {
  //     LOG(ERROR) << "aclmdlRICaptureGetInfo failed, acl error code: " <<
  //     error;
  //   }
  //   if (status == ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE) {
  //     context_->SetExecuteStream(stream);
  //     graph_captured_ = true;
  //   }
  // }
  atb::Status st =
      node.operation->Setup(node.variantPack, node.workspaceSize, context_);
  if (st != 0) {
    LOG(ERROR) << " setup layer node fail, not call execute";
    return st;
  }

  if (node.workspaceSize > 0) {
    node.workspace = work_space_.get_workspace_buffer(node.workspaceSize);
  }

  run_task_func_(name_ + std::to_string(node_id), [=]() {
    return execute_plan(
        node, name_ + std::to_string(node_id), event, event_flag);
  });

  return st;
}

atb::Status NpuBaseLayer::execute_plan(
    const atb_speed::Model::Node& node,
    const std::string& op_name,
    std::vector<aclrtEvent*> event,
    std::vector<std::atomic<bool>*> event_flag) {
  atb::Status st = node.operation->Execute(
      node.variantPack, (uint8_t*)node.workspace, node.workspaceSize, context_);
  LOG_IF(ERROR, st != 0) << name_ << " execute plan fail, error code: " << st;
  for (auto i = 0; i < event.size(); ++i) {
    if (st == 0 && event[i] != nullptr) {
      aclrtStream stream = context_->GetExecuteStream();

      aclrtEvent* aclrt_event = reinterpret_cast<aclrtEvent*>(event[i]);

      auto ret = aclrtRecordEvent(*aclrt_event, stream);
      if (ret != ACL_SUCCESS) {
        LOG(ERROR) << "Record event failed.";
        return st;
      }

      event_flag[i]->store(true, std::memory_order_release);
    }
  }

  return st;
}

void NpuBaseLayer::run_task(std::string taskName,
                            std::function<int()> task) const {
  at_npu::native::OpCommand cmd;
  cmd.Name(taskName);
  cmd.SetCustomHandler(task);
  cmd.Run();
}

atb::Tensor NpuBaseLayer::XTensor2Tensor(
    const std::shared_ptr<xllm::XTensor>& xtensor) {
  static std::map<at::ScalarType, aclDataType> dtypeMap = {
      {at::ScalarType::Bool, ACL_BOOL},
      {at::ScalarType::Byte, ACL_UINT8},
      {at::ScalarType::Char, ACL_INT8},
      {at::ScalarType::Half, ACL_FLOAT16},
      {at::ScalarType::Float, ACL_FLOAT},
      {at::ScalarType::Int, ACL_INT32},
      {at::ScalarType::Long, ACL_INT64},
      {at::ScalarType::BFloat16, ACL_BF16},
  };

  atb::Tensor tensor;
  // continuous kvcache only support ND format
  tensor.desc.format = ACL_FORMAT_ND;
  tensor.deviceData = xtensor->get_base_ptr();

  tensor.desc.shape.dimNum = 4;
  tensor.desc.shape.dims[0] = 0;
  tensor.desc.shape.dims[1] = 128;  // block_size
  tensor.desc.shape.dims[2] =
      xtensor->options().num_kv_heads();                       // num_kv_heads
  tensor.desc.shape.dims[3] = xtensor->options().head_size();  // head_size

  auto it = dtypeMap.find(xtensor->dtype());
  if (it != dtypeMap.end()) {
    tensor.desc.dtype = it->second;
  } else {
    throw std::runtime_error("XTensor2Tensor: not support dtype");
  }

  tensor.dataSize = 0;

  return tensor;
}

}  // namespace layer
}  // namespace xllm

#include "npu_base_layer.h"

#ifdef TORCH_HIGHER_THAN_PTA6
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#else
#include <torch_npu/csrc/aten/NPUNativeFunctions.h>
#include <torch_npu/csrc/framework/utils/OpPreparation.h>
#endif

namespace xllm {
namespace layer {

atb::Status NpuBaseLayer::execute_node(atb_speed::Model::Node& node,
                                       atb::Context* context,
                                       AtbWorkspace& workspace,
                                       int node_id,
                                       aclrtEvent* event,
                                       std::atomic<bool>* event_flag) {
  atb::Status st =
      node.operation->Setup(node.variantPack, node.workspaceSize, context);
  if (st != 0) {
    LOG(ERROR) << " setup layer node fail, not call execute";
    return st;
  }

  if (node.workspaceSize > 0) {
    node.workspace = workspace.get_workspace_buffer(node.workspaceSize);
  }

  run_task_func_(name_ + std::to_string(node_id), [=]() {
    return execute_plan(
        node, context, name_ + std::to_string(node_id), event, event_flag);
  });

  return st;
}

atb::Status NpuBaseLayer::execute_plan(const atb_speed::Model::Node& node,
                                       atb::Context* context,
                                       const std::string& op_name,
                                       aclrtEvent* event,
                                       std::atomic<bool>* event_flag) {
  atb::Status st = node.operation->Execute(
      node.variantPack, (uint8_t*)node.workspace, node.workspaceSize, context);
  LOG_IF(ERROR, st != 0) << name_ << " execute plan fail, error code: " << st;
  if (st == 0 && event != nullptr) {
    aclrtStream stream = context->GetExecuteStream();

    aclrtEvent* aclrt_event = reinterpret_cast<aclrtEvent*>(event);

    auto ret = aclrtRecordEvent(*aclrt_event, stream);
    if (ret != ACL_SUCCESS) {
      LOG(ERROR) << "Record event failed.";
      return st;
    }

    event_flag->store(true, std::memory_order_release);
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

}  // namespace layer
}  // namespace xllm
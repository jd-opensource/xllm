#pragma once

#include <absl/strings/match.h>
#include <torch/torch.h>
#include <torch_npu/torch_npu.h>

#include <atomic>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "atb/atb_infer.h"
#include "atb_speed/base/model.h"
#include "atb_speed/log.h"
#include "atb_speed/utils/singleton.h"
#include "atb_speed/utils/tensor_util.h"
#include "buffer/atb_workspace.h"
#include "core/layers/base_layer.h"
#include "framework/context.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_input_params.h"
#include "framework/state_dict/state_dict.h"
#include "pytorch/adapter/utils/utils.h"
#include "pytorch/adapter/workspace/workspace.h"

#ifdef TORCH_HIGHER_THAN_PTA6
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#else
#include <torch_npu/csrc/aten/NPUNativeFunctions.h>
#include <torch_npu/csrc/framework/utils/OpPreparation.h>
#endif

namespace xllm {
namespace layer {

class NpuBaseLayer : public BaseLayer {
 public:
  explicit NpuBaseLayer(const Context& context) : BaseLayer(context) {}
  ~NpuBaseLayer() = default;

  atb::Status execute_node(atb_speed::Model::Node& node,
                           atb::Context* context,
                           AtbWorkspace& workspace,
                           int nodeId = 0,
                           aclrtEvent* event = nullptr,
                           std::atomic<bool>* event_flag = nullptr);

  atb::Status execute_plan(const atb_speed::Model::Node& node,
                           atb::Context* context,
                           const std::string& op_name,
                           aclrtEvent* event,
                           std::atomic<bool>* event_flag);

  void print_atbtensor(const atb::Tensor& tensor, int i);

  virtual void run_task(std::string taskName,
                        std::function<int()> task) const override;

 protected:
  // std::vector<at::Tensor> at_weight_tensors_;
  std::vector<atb::Tensor> atb_weight_tensors_;
};

}  // namespace layer
}  // namespace xllm

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

#pragma once

#ifdef TORCH_HIGHER_THAN_PTA6
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#else
#include <torch_npu/csrc/aten/NPUNativeFunctions.h>
#include <torch_npu/csrc/framework/utils/OpPreparation.h>
#endif

#include <torch_npu/csrc/libs/init_npu.h>

#include <functional>

#include "atb/atb_infer.h"
#include "atb_base.h"
#include "framework/context.h"
#include "framework/model/model_input_params.h"
#include "framework/state_dict/state_dict.h"
#include "nlohmann/json.hpp"
#include "pytorch/adapter/utils/utils.h"
#include "xllm_kernels/core/include/atb_speed/base/hosttensor_binder.h"
#include "xllm_kernels/core/include/atb_speed/base/model.h"
#include "xllm_kernels/core/include/atb_speed/log.h"
#include "xllm_kernels/core/include/atb_speed/utils/model_factory.h"
#include "xllm_kernels/operations/fusion/linear/linear_parallel.h"

namespace xllm::hf {
// Linear layer with column parallelism.
// The linear layer is defined as Y = XA + b. A is parallelized along
// its second dimension as A = [A_1, ..., A_p].
class AtbColumnParallelLinearImpl : public torch::nn::Module, public ATBBase {
 public:
  using Task = std::function<int()>;
  using RunTaskFunc =
      std::function<void(const std::string& taskName, Task task)>;

  AtbColumnParallelLinearImpl(const Context& context);

  ~AtbColumnParallelLinearImpl() {};

  void load_state_dict(const StateDict& state_dict);

  void verify_loaded_weights(const std::string weight_str) const;

  void merge_loaded_weights();

  void param_from_args(atb_speed::common::LinearParallelParam& param,
                       const ModelArgs& args,
                       const ParallelArgs& parallel_args);

  int64_t init_layer();

  virtual torch::Tensor forward(const torch::Tensor& input,
                                atb::Context* context,
                                AtbWorkspace& workspace,
                                int nodeId);

  void build_node_variant_pack(atb_speed::Model::Node& node,
                               const torch::Tensor& input);

 protected:
  int64_t init_node(atb_speed::Model::Node& node,
                    atb_speed::common::LinearParallelParam& linearParam);

  atb_speed::Model::Node linear_node_;
  std::string model_name_;

  std::vector<at::Tensor> at_out_tensors_;
  atb::Tensor internal_input;
  torch::Tensor tensor_placeholder_;
  atb::Tensor placeholder_;

  atb_speed::common::LinearParallelParam linear_param_;
};

class AtbColumnParallelLinear
    : public torch::nn::ModuleHolder<AtbColumnParallelLinearImpl> {
 public:
  using torch::nn::ModuleHolder<AtbColumnParallelLinearImpl>::ModuleHolder;
  using Impl __attribute__((__unused__)) = AtbColumnParallelLinearImpl;

  AtbColumnParallelLinear(const Context& context);
};

std::shared_ptr<AtbColumnParallelLinearImpl>
create_atb_column_parallel_linear_layer(const Context& context);
}  // namespace xllm::hf
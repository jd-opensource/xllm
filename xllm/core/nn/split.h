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
#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_input_params.h"
#include "framework/state_dict/state_dict.h"
#include "layers/npu/atb_base.h"
#include "nlohmann/json.hpp"
#include "pytorch/adapter/utils/utils.h"
#include "xllm_kernels/core/include/atb_speed/base/hosttensor_binder.h"
#include "xllm_kernels/core/include/atb_speed/base/model.h"
#include "xllm_kernels/core/include/atb_speed/log.h"
#include "xllm_kernels/core/include/atb_speed/utils/model_factory.h"

namespace xllm::hf {

class SplitImpl : public torch::nn::Module, public ATBBase {
 public:
  using Task = std::function<int()>;
  using RunTaskFunc =
      std::function<void(const std::string& taskName, Task task)>;

  explicit SplitImpl(const ModelContext& context);

  ~SplitImpl() {};

  void load_state_dict(const StateDict& state_dict);

  void verify_loaded_weights(const std::string weight_str) const;

  void merge_loaded_weights();

  void param_from_args(atb::infer::SplitParam& param, const ModelArgs& args);

  std::vector<at::Tensor> forward(const torch::Tensor& input, int nodeId);

  void build_node_variant_pack(atb_speed::Model::Node& node,
                               const torch::Tensor& input);

 private:
  int64_t init_node(atb_speed::Model::Node& node,
                    atb::infer::SplitParam& param);

  std::vector<at::Tensor> at_out_tensors_;
  atb::Tensor internal_input;

  atb_speed::Model::Node split_node_;
  std::string model_name_;
  atb::infer::SplitParam norm_param_;
  atb::Tensor internal_tensors_;
};

class Split : public torch::nn::ModuleHolder<SplitImpl> {
 public:
  using torch::nn::ModuleHolder<SplitImpl>::ModuleHolder;
  using Impl __attribute__((__unused__)) = SplitImpl;

  Split(const ModelContext& context);
};

std::shared_ptr<SplitImpl> create_split_layer(const ModelContext& context);

}  // namespace xllm::hf

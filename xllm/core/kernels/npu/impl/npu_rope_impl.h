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
#include "layers/npu/npu_base_layer.h"
#include "nlohmann/json.hpp"
#include "pytorch/adapter/utils/utils.h"
#include "xllm_kernels/core/include/atb_speed/base/hosttensor_binder.h"
#include "xllm_kernels/core/include/atb_speed/base/model.h"
#include "xllm_kernels/core/include/atb_speed/log.h"
#include "xllm_kernels/core/include/atb_speed/utils/model_factory.h"

namespace xllm::kernel {

class NpuRopeImpl : public xllm::layer::NpuBaseLayer {
 public:
  explicit NpuRopeImpl(const ModelContext& context);

  ~NpuRopeImpl() {};

  void load_state_dict(const StateDict& state_dict);

  void verify_loaded_weights(const std::string weight_str) const;

  void merge_loaded_weights();

  std::vector<at::Tensor> forward(const torch::Tensor& q,
                                  const torch::Tensor& k,
                                  const torch::Tensor& cos_embedding,
                                  const torch::Tensor& sin_embedding,
                                  const torch::Tensor& seq_len,
                                  int nodeId);

 private:
  int64_t init_node(atb_speed::Model::Node& node, atb::infer::RopeParam& param);
  void build_node_variant_pack(atb_speed::Model::Node& node,
                               const torch::Tensor& q,
                               const torch::Tensor& k,
                               const torch::Tensor& cos_embedding,
                               const torch::Tensor& sin_embedding,
                               const torch::Tensor& seq_len);
  void param_from_args(atb::infer::RopeParam& param, const ModelArgs& args);

  std::vector<at::Tensor> at_out_tensors_;
  atb::Tensor internal_q;
  atb::Tensor internal_k;
  atb::Tensor internal_cos_embedding;
  atb::Tensor internal_sin_embedding;
  atb::Tensor internal_seq_len;

  atb_speed::Model::Node rope_node_;
  std::string model_name_;
  atb::infer::RopeParam rope_param_;
  atb::Tensor internal_tensors_;
};

}  // namespace xllm::kernel

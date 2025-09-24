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
#include "framework/model/model_input_params.h"
#include "nlohmann/json.hpp"
#include "npu_base_layer.h"
#include "pytorch/adapter/utils/utils.h"
#include "xllm_kernels/core/include/atb_speed/base/hosttensor_binder.h"
#include "xllm_kernels/core/include/atb_speed/base/model.h"
#include "xllm_kernels/core/include/atb_speed/log.h"
#include "xllm_kernels/core/include/atb_speed/utils/model_factory.h"

namespace xllm {
namespace layer {

class NpuBlockCopyImpl : public NpuBaseLayer {
 public:
  explicit NpuBlockCopyImpl(const ModelContext& context);

  ~NpuBlockCopyImpl() {};

  void load_state_dict(const StateDict& state_dict) {};

  void verify_loaded_weights(const std::string weight_str) const {};

  void merge_loaded_weights();

  int64_t init_layer();

  torch::Tensor forward(const torch::Tensor& key_cache,
                        const torch::Tensor& value_cache,
                        const torch::Tensor& src_block_ids,
                        const torch::Tensor& dst_block_ids,
                        const torch::Tensor& cum_sum,
                        int nodeId = 0);

  void build_node_variant_pack(atb_speed::Model::Node& node,
                               const torch::Tensor& key_cache,
                               const torch::Tensor& value_cache,
                               const torch::Tensor& src_block_ids,
                               const torch::Tensor& dst_block_ids,
                               const torch::Tensor& cum_sum);

 private:
  int64_t init_node(atb_speed::Model::Node& node,
                    atb::infer::BlockCopyParam& param);

  atb_speed::Model::Node node_;
  std::string model_name_;
  atb::infer::BlockCopyParam param_;
  atb::Tensor internal_key_tensors_;
  atb::Tensor internal_value_tensors_;
  atb::Tensor internal_src_block_ids_tensors_;
  atb::Tensor internal_dst_block_ids_tensors_;
  atb::Tensor internal_cum_sum_tensors_;
};

}  // namespace layer
}  // namespace xllm

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
#include "framework/model_context.h"
#include "framework/state_dict/state_dict.h"
#include "nlohmann/json.hpp"
#include "npu_base_layer.h"
#include "pytorch/adapter/utils/utils.h"
#include "xllm_kernels/core/include/atb_speed/base/hosttensor_binder.h"
#include "xllm_kernels/core/include/atb_speed/base/model.h"
#include "xllm_kernels/core/include/atb_speed/log.h"
#include "xllm_kernels/core/include/atb_speed/utils/model_factory.h"
#include "xllm_kernels/models/qwen3/layer/decoder_layer.h"

namespace xllm {
namespace layer {

class NpuQwen3DecoderLayerImpl : public NpuBaseLayer {
 public:
  explicit NpuQwen3DecoderLayerImpl(const ModelContext& context);

  ~NpuQwen3DecoderLayerImpl() {};

  virtual void load_state_dict(const StateDict& state_dict) override;

  virtual void verify_loaded_weights() const override;

  virtual void merge_loaded_weights() override;

  virtual int64_t init_layer() override;

  torch::Tensor forward(std::vector<torch::Tensor>& x,
                        std::vector<torch::Tensor>& cos_pos,
                        std::vector<torch::Tensor>& sin_pos,
                        std::vector<torch::Tensor>& attn_mask,
                        KVCache& kv_cache,
                        std::vector<ModelInputParams>& input_params,
                        std::vector<aclrtEvent*> event = {nullptr, nullptr},
                        std::vector<std::atomic<bool>*> event_flag = {nullptr,
                                                                      nullptr},
                        int node_id = 0);

 private:
  void param_from_args(atb_speed::qwen::QwenLayerParam& param,
                       const ModelArgs& args,
                       const ParallelArgs& parallel_args,
                       bool isPrefill);

  void build_node_variant_pack(atb_speed::Model::Node& node,
                               std::vector<torch::Tensor>& x,
                               std::vector<torch::Tensor>& cos_pos,
                               std::vector<torch::Tensor>& sin_pos,
                               std::vector<torch::Tensor>& attn_mask,
                               KVCache& kv_cache,
                               std::vector<ModelInputParams>& input_params,
                               bool is_prefill);

  void initialize_quantization_parameters(
      atb_speed::qwen::QwenLayerParam& param);

  int64_t init_node(atb_speed::Model::Node& node,
                    atb_speed::qwen::QwenLayerParam& param);

  int64_t init_attn_mask();

  atb_speed::Model::Node prefill_node_;
  atb_speed::Model::Node decode_node_;
  std::string model_name_;
  atb_speed::qwen::QwenLayerParam prefill_param_;
  atb_speed::qwen::QwenLayerParam decode_param_;
  atb::Tensor internal_tensors_;
  atb::Tensor internal_tensors_auxiliary;
  atb::Tensor placeholder_;
  torch::Tensor int_tensor_placeholder_;
  torch::Tensor block_tables_placeholder_;
  torch::Tensor slot_tensor_placeholder_;

  at::Tensor decode_attn_mask_;

  at::Tensor at_placeholder_;

  int device_id_;
  int32_t layer_id_;
  int rank_id_;
  std::vector<std::shared_ptr<at::Tensor>> prefill_tensor_storage_;
  std::vector<std::shared_ptr<at::Tensor>> decode_tensor_storage_;
  std::vector<std::shared_ptr<std::vector<int>>> prefill_vector_storage_;
  std::vector<std::shared_ptr<std::vector<int>>> decode_vector_storage_;
};

}  // namespace layer
}  // namespace xllm

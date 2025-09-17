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
#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_args.h"
#include "framework/model/model_input_params.h"
#include "framework/parallel_state.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"
#include "nlohmann/json.hpp"
#include "pytorch/adapter/utils/utils.h"
#include "xllm_kernels/core/include/atb_speed/base/hosttensor_binder.h"
#include "xllm_kernels/core/include/atb_speed/base/model.h"
#include "xllm_kernels/core/include/atb_speed/log.h"
#include "xllm_kernels/core/include/atb_speed/utils/model_factory.h"
#include "xllm_kernels/models/qwen3/layer/decoder_layer.h"

namespace xllm::hf {

class Qwen3DecoderImpl : public torch::nn::Module, public ATBBase {
 public:
  using Task = std::function<int()>;
  using RunTaskFunc =
      std::function<void(const std::string& taskName, Task task)>;

  explicit Qwen3DecoderImpl(const ModelContext& context);

  ~Qwen3DecoderImpl() {};

  void load_state_dict(const StateDict& state_dict);

  void verify_loaded_weights() const;

  void merge_loaded_weights();

  void param_from_args(atb_speed::qwen::QwenLayerParam& param,
                       const ModelArgs& args,
                       const ParallelArgs& parallel_args,
                       bool isPrefill);

  void initialize_quantization_parameters(
      atb_speed::qwen::QwenLayerParam& param);

  int64_t init_layer();

  torch::Tensor forward(torch::Tensor& x,
                        torch::Tensor& cos_pos,
                        torch::Tensor& sin_pos,
                        torch::Tensor& attn_mask,
                        KVCache& kv_cache,
                        ModelInputParams& input_params,
                        aclrtEvent* event = nullptr,
                        std::atomic<bool>* event_flag = nullptr,
                        int node_id = 0);

  void build_node_variant_pack(atb_speed::Model::Node& node,
                               torch::Tensor& x,
                               torch::Tensor& cos_pos,
                               torch::Tensor& sin_pos,
                               torch::Tensor& attn_mask,
                               KVCache& kv_cache,
                               ModelInputParams& input_params,
                               bool is_prefill);

 private:
  int64_t init_node(atb_speed::Model::Node& node,
                    atb_speed::qwen::QwenLayerParam& param);

  int64_t init_attn_mask();

  atb_speed::Model::Node prefill_node_;
  atb_speed::Model::Node decode_node_;
  std::string model_name_;
  atb_speed::qwen::QwenLayerParam prefill_param_;
  atb_speed::qwen::QwenLayerParam decode_param_;
  atb::Tensor internal_tensors_;
  atb::Tensor placeholder_;

  at::Tensor decode_attn_mask_;

  at::Tensor at_placeholder_;

  int device_id_;
  int rank_id_;
  std::vector<std::shared_ptr<at::Tensor>> prefill_tensor_storage_;
  std::vector<std::shared_ptr<at::Tensor>> decode_tensor_storage_;
  std::vector<std::shared_ptr<std::vector<int>>> prefill_vector_storage_;
  std::vector<std::shared_ptr<std::vector<int>>> decode_vector_storage_;
};

class Qwen3Decoder : public torch::nn::ModuleHolder<Qwen3DecoderImpl> {
 public:
  using torch::nn::ModuleHolder<Qwen3DecoderImpl>::ModuleHolder;
  using Impl __attribute__((__unused__)) = Qwen3DecoderImpl;

  Qwen3Decoder(const ModelContext& context);
};

std::shared_ptr<Qwen3DecoderImpl> create_qwen3_decode_layer(
    const ModelContext& context);

}  // namespace xllm::hf

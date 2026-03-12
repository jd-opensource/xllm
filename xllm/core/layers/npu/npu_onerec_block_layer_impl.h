/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include <torch/torch.h>

#include <cstdint>

#include "framework/model_context.h"
#include "layers/onerec_block_layer.h"

namespace xllm {
namespace layer {

class NpuOneRecBlockLayerImpl final : public OneRecBlockLayer {
 public:
  explicit NpuOneRecBlockLayerImpl(const ModelContext& context,
                                   bool is_decoder = false,
                                   int32_t layer_id = 0);

  torch::Tensor forward(torch::Tensor& hidden_states,
                        torch::Tensor& attn_mask,
                        KVCache& kv_cache,
                        ModelInputParams& input_params,
                        torch::Tensor* encoder_output = nullptr,
                        int32_t node_id = 0,
                        aclrtEvent* event = nullptr,
                        std::atomic<bool>* event_flag = nullptr) override;

 private:
  const torch::Device device_;
  bool is_decoder_ = false;
  int32_t layer_id_ = 0;
};

}  // namespace layer
}  // namespace xllm

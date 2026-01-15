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

#ifndef CORE_LAYERS_MUSA_MUSA_ATTENTION_H_
#define CORE_LAYERS_MUSA_MUSA_ATTENTION_H_

#include <torch/torch.h>

#include <cassert>
#include <cstdint>
#include <optional>

#include "MTTOplib/Attention.h"
#include "framework/state_dict/state_dict.h"
#include "framework/state_dict/utils.h"
#include "musa_layer_base.h"

namespace xllm::layer {

class MusaAttentionImpl : public MUSALayerBaseImpl {
 public:
  explicit MusaAttentionImpl(ModelArgs const& args,
                             QuantArgs const& quant_args,
                             ParallelArgs const& parallel_args,
                             torch::TensorOptions const& options);

  ~MusaAttentionImpl() {};

  torch::Tensor forward(torch::Tensor& input,
                        ForwardParams& fwd_params) override;

  void load_state_dict(StateDict const& state_dict) override;

 private:
  int32_t num_heads_;
  int32_t num_kv_heads_;
  int32_t head_dim_;
  int32_t q_size_;
  int32_t kv_size_;
  int32_t hidden_size_;
  float rms_eps;
  float scaling_;
  constexpr static int32_t weight_num_ = 7;
};
TORCH_MODULE(MusaAttention);

}  // namespace xllm::layer

#endif

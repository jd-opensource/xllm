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

#include <torch/torch.h>

#include <functional>

#include "attention.h"
#include "framework/context.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_args.h"
#include "framework/model/model_input_params.h"
#include "framework/parallel_state.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"
#include "layers/embedding.h"
#include "layers/normalization.h"
#include "qwen3_attention.h"
#include "qwen3_mlp.h"

namespace xllm::hf {

class Qwen3DecoderImpl : public torch::nn::Module {
 public:
  explicit Qwen3DecoderImpl(const Context& context);

  ~Qwen3DecoderImpl() {};

  void load_state_dict(const StateDict& state_dict);

  torch::Tensor forward(torch::Tensor& x,
                        torch::Tensor& positions,
                        torch::Tensor& residual,
                        const xllm::mlu::AttentionMetadata& attn_metadata,
                        KVCache& kv_cache,
                        const ModelInputParams& input_params);

 private:
  xllm::Qwen3Attention attention_;
  xllm::Qwen3MLP mlp_;
  RMSNorm input_norm_{nullptr};
  RMSNorm post_norm_{nullptr};

  c10::ScalarType dtype_;
  int rank_id_;
};

class Qwen3Decoder : public torch::nn::ModuleHolder<Qwen3DecoderImpl> {
 public:
  using torch::nn::ModuleHolder<Qwen3DecoderImpl>::ModuleHolder;
  using Impl __attribute__((__unused__)) = Qwen3DecoderImpl;

  Qwen3Decoder(const Context& context);
};

std::shared_ptr<Qwen3DecoderImpl> create_qwen3_decode_layer(
    const Context& context);

}  // namespace xllm::hf

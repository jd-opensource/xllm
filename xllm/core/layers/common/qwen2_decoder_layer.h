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

#if defined(USE_MLU)
#include "../mlu/attention.h"
#elif defined(USE_CUDA)
#include "../cuda/attention.h"
#endif
#include <optional>

#include "dense_mlp.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_args.h"
#include "framework/model/model_input_params.h"
#include "framework/model_context.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"
#include "layers/common/rms_norm.h"
#include "qwen2_attention.h"

namespace xllm {
namespace layer {

class Qwen2DecoderLayerImpl : public torch::nn::Module {
 public:
  explicit Qwen2DecoderLayerImpl(const ModelContext& context);

  ~Qwen2DecoderLayerImpl() {};

  void load_state_dict(const StateDict& state_dict);

  torch::Tensor forward(torch::Tensor& x,
                        std::optional<torch::Tensor>& residual,
                        torch::Tensor& positions,
                        const AttentionMetadata& attn_metadata,
                        KVCache& kv_cache,
                        const ModelInputParams& input_params);

 private:
  Qwen2Attention attention_{nullptr};
  DenseMLP mlp_{nullptr};
  RMSNorm input_norm_{nullptr};
  RMSNorm post_norm_{nullptr};

  ParallelArgs parallel_args_;
};

using Qwen3DecoderLayerImpl = Qwen2DecoderLayerImpl;

}  // namespace layer
}  // namespace xllm

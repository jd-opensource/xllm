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

#include <tuple>

#include "framework/kv_cache/kv_cache.h"
#include "layers/common/attention_metadata.h"

namespace xllm {
namespace layer {

// BaseAttentionImpl implements the standard attention computation using
// flashinfer kernels. This is the default implementation used when not in
// pure device mode.
class BaseAttentionImpl {
 public:
  BaseAttentionImpl(int num_heads,
                    int head_size,
                    float scale,
                    int num_kv_heads,
                    int sliding_window);

  std::tuple<torch::Tensor, std::optional<torch::Tensor>> forward(
      const AttentionMetadata& attn_metadata,
      torch::Tensor& query,
      torch::Tensor& key,
      torch::Tensor& value,
      torch::Tensor& output,
      KVCache& kv_cache);

 private:
  int num_heads_;
  int head_size_;
  float scale_;
  int num_kv_heads_;
  int sliding_window_;
};

}  // namespace layer
}  // namespace xllm

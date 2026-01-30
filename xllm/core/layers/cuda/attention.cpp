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

#include "attention.h"

#include "base_attention.h"
#include "core/common/rec_model_utils.h"
#include "xattention.h"

namespace xllm {
namespace layer {
AttentionImpl::AttentionImpl(int num_heads,
                             int head_size,
                             float scale,
                             int num_kv_heads,
                             int sliding_window) {
  // Select implementation based on mode. Use std::variant to manage different
  // implementations, avoiding if-else logic in forward() method.

  if (is_rec_multi_round_mode()) {
    attention_impl_ = std::make_shared<XAttentionImpl>(
        num_heads, head_size, scale, num_kv_heads, sliding_window);
  } else {
    attention_impl_ = std::make_shared<BaseAttentionImpl>(
        num_heads, head_size, scale, num_kv_heads, sliding_window);
  }
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>> AttentionImpl::forward(
    const AttentionMetadata& attn_metadata,
    torch::Tensor& query,
    torch::Tensor& key,
    torch::Tensor& value,
    KVCache& kv_cache) {
  // Create output tensor internally to unify the interface with other devices
  torch::Tensor output = torch::empty_like(query);

  // Use std::visit to dispatch to the appropriate implementation, avoiding
  // if-else logic and making the code more elegant and type-safe.
  return std::visit(
      [&](auto& impl) {
        return impl->forward(
            attn_metadata, query, key, value, output, kv_cache);
      },
      attention_impl_);
}

}  // namespace layer
}  // namespace xllm
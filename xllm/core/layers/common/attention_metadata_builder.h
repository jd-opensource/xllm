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

#include <optional>
#include <string>

namespace xllm {
struct ForwardInput;

namespace layer {

struct AttentionMetadata;

// Builder class for AttentionMetadata to avoid circular dependency.
// This class handles building AttentionMetadata without making
// attention_metadata.h depend on runtime input structures.
class AttentionMetadataBuilder {
 public:
  static AttentionMetadata build(
      const ForwardInput& forward_input,
      bool enable_mla,
      const std::optional<torch::Tensor>& attn_mask = {},
      const std::optional<torch::Device>& device = std::nullopt);

  static AttentionMetadata build(
      const ForwardInput& forward_input,
      bool enable_mla,
      const std::string& compute_dtype,
      const std::optional<torch::Tensor>& attn_mask = {},
      const std::optional<torch::Device>& device = std::nullopt);
};

}  // namespace layer
}  // namespace xllm

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

#include <memory>
#include <string>

namespace xllm::layer {
// AttentionMetadata contains batch-level information shared across all
// attention layers. It is built once at the beginning of model forward pass and
// reused by all layers. This avoids redundant computation and memory allocation
// for metadata that is identical across layers (e.g., sequence lengths, paged
// KV cache indices, plan_info). AttentionMetadata is now a member of
// AttentionParams (used for kernel calls), which also contains layer-specific
// tensors (query, key, value) that differ per layer. Use
// AttentionMetadataBuilder to build instances from ModelInputParams.
struct AttentionMetadata {
  torch::Tensor q_cu_seq_lens;
  torch::Tensor kv_cu_seq_lens;
  torch::Tensor kv_seq_lens;
  torch::Tensor q_seq_lens;
  torch::Tensor block_table;
  torch::Tensor slot_mapping;
  int64_t max_query_len;
  int64_t max_seq_len;
  std::string compute_dtype;
  bool is_prefill;
  bool is_chunked_prefill;
  bool is_dummy;
  // Whether to apply causal mask. Default: true.
  bool is_causal = true;

  // for mrope
  torch::Tensor mrope_cos;
  torch::Tensor mrope_sin;
};

}  // namespace xllm::layer

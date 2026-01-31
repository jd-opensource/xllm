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

struct PlanInfo {
  int32_t layer_id = -1;
  torch::Tensor plan_info;
  std::string uri;
};

// AttentionMetadata contains batch-level information shared across all
// attention layers. It is built once at the beginning of model forward pass and
// reused by all layers. This avoids redundant computation and memory allocation
// for metadata that is identical across layers.
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

  // for flashinfer
  // Index pointer for paged KV cache, similar to row_splits in ragged tensor.
  // paged_kv_indptr[i] is the start index of sequence i in paged_kv_indices,
  // paged_kv_indptr[i+1] is the end index (exclusive). paged_kv_indptr[0] = 0.
  // Shape: [batch_size + 1]. Type: int32.
  torch::Tensor paged_kv_indptr;
  // Page indices (block IDs) of the paged KV cache for all sequences.
  // Contains all block/page IDs used by all sequences, flattened into a 1D
  // array. Shape: [total_num_blocks]. Type: int32.
  torch::Tensor paged_kv_indices;
  // Number of valid entries in the last page of each sequence in the paged KV
  // cache. Since pages are fixed-size (block_size), the last page may be
  // partially filled. Shape: [batch_size]. Type: int32.
  torch::Tensor paged_kv_last_page_len;
  // Query/Output index pointer tensor for decode mode with tensor core.
  // Similar to row_splits in ragged tensor: cumulative sum of sequence lengths.
  // qo_indptr[i] is the start index of sequence i in the packed query/output
  // tensor, qo_indptr[i+1] is the end index (exclusive). qo_indptr[0] = 0,
  // qo_indptr[batch_size] = total_tokens. Shape: [batch_size + 1]. Type: int32.
  // Used when use_tensor_core=true. If not defined (use .defined() to check),
  // will be created internally in batch_decode.
  torch::Tensor qo_indptr;
  // FlashInfer execution plan information for attention computation.
  // Contains kernel URI and plan tensor that specifies how to execute the
  // attention kernel. Only updated at layer 0 (shared across all layers). The
  // plan_info tensor contains pre-computed execution parameters optimized for
  // the current batch configuration.
  std::shared_ptr<PlanInfo> plan_info;
  // Whether to use tensor core for decode attention computation. Default: true.
  bool use_tensor_core = true;

  // for CUDA graph - CPU tensors for plan_info update (avoid .to(CPU) during
  // graph capture) torch::Tensor q_cu_seq_lens_host;      // Prefill mode:
  // q_cu_seq_lens on CPU torch::Tensor kv_cu_seq_lens_host;    // Prefill mode:
  // kv_cu_seq_lens on CPU torch::Tensor paged_kv_indptr_host;    // Decode
  // mode: paged_kv_indptr on CPU torch::Tensor kv_seq_lens_host;        //
  // Decode mode (tensor_core) / NPU: kv_seq_lens on CPU for CUDA graph
  bool enable_cuda_graph = false;
};

}  // namespace xllm::layer

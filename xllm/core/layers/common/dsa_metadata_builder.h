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

#include <vector>

namespace xllm {
struct ModelInputParams;
struct DSACacheInfo;
struct DSAGroupInfo;

namespace layer {

struct AttentionMetadata;
struct DSAMetadata;

// Builder class for DSAMetadata.
// Builds a complete AttentionMetadata (with dsa_metadata populated) from
// ModelInputParams and model-specific data.  This replaces the need for a
// separate AttentionMetadataBuilder::build() call in DeepSeek V4.
class DSAMetadataBuilder {
 public:
  // Build a complete AttentionMetadata with DSAMetadata populated inside.
  // Internally constructs the base AttentionMetadata fields (q_cu_seq_lens,
  // block_table, slot_mapping, etc.) and the DSA-specific fields (RoPE,
  // block tables, slot mappings, sequence lengths, compressed positions).
  //
  //   params: batch-level model input params
  //   positions: token position IDs tensor
  //   dsa_cos_sin: precomputed RoPE cos/sin table (optional, can be undefined)
  //   caches_info: per-layer cache specs [layer_id][cache_idx]
  //   group_infos: per-group info [group_id]
  static AttentionMetadata build(
      const ModelInputParams& params,
      const torch::Tensor& positions,
      const torch::Tensor& dsa_cos_sin,
      const std::vector<std::vector<DSACacheInfo>>& caches_info,
      const std::vector<DSAGroupInfo>& group_infos,
      const torch::Tensor& dsa_c4_cos_sin = torch::Tensor(),
      const torch::Tensor& dsa_c128_cos_sin = torch::Tensor());

 private:
  // Build DSA-specific fields into dsa_metadata.
  static void build_dsa_fields(
      const ModelInputParams& params,
      const torch::Tensor& positions,
      const torch::Tensor& dsa_cos_sin,
      const torch::Tensor& dsa_c4_cos_sin,
      const torch::Tensor& dsa_c128_cos_sin,
      const std::vector<std::vector<DSACacheInfo>>& caches_info,
      const std::vector<DSAGroupInfo>& group_infos,
      DSAMetadata& dsa_metadata);

  // Step 1: expand block_table to slot array for one manager.
  static torch::Tensor expand_blocks_to_slots(const torch::Tensor& block_table,
                                              const DSAGroupInfo& gi,
                                              const std::vector<int>& ctx_lens,
                                              int32_t batch_size,
                                              int64_t total_tokens);

  // Compute how many slots a single seq needs for this group.
  static int64_t compute_slot_num(const DSAGroupInfo& gi, int64_t token_len);

  // Step 2: per-group processing.
  static void process_group(const torch::Tensor& raw_bt,
                            const torch::Tensor& raw_slots,
                            const DSAGroupInfo& gi,
                            const std::vector<int>& ctx_lens,
                            const std::vector<int>& q_lens,
                            int32_t batch_size,
                            int64_t total_tokens,
                            torch::Tensor& out_bt,
                            torch::Tensor& out_slots);

  static void process_token_group(const torch::Tensor& raw_bt,
                                  const torch::Tensor& raw_slots,
                                  int32_t ratio,
                                  const std::vector<int>& ctx_lens,
                                  const std::vector<int>& q_lens,
                                  int32_t batch_size,
                                  int64_t total_tokens,
                                  torch::Tensor& out_bt,
                                  torch::Tensor& out_slots);

  static void process_swa_group(const torch::Tensor& raw_bt,
                                const torch::Tensor& raw_slots,
                                int32_t block_size,
                                const std::vector<int>& ctx_lens,
                                const std::vector<int>& q_lens,
                                int32_t batch_size,
                                torch::Tensor& out_bt,
                                torch::Tensor& out_slots);

  // Build actual_seq_lengths_kv and actual_seq_lengths_query.
  static void build_seq_lengths(const ModelInputParams& params,
                                int32_t batch_size,
                                DSAMetadata& dsa_metadata);

  // Build input_positions, c4_pad_positions, c128_pad_positions.
  static void build_positions(const ModelInputParams& params,
                              int32_t batch_size,
                              int64_t total_tokens,
                              DSAMetadata& dsa_metadata);

  // Build grouped RoPE slices (c4/c128) from position IDs.
  static void build_group_cos_sin(const torch::Tensor& cos_sin_table,
                                  const torch::Tensor& pad_positions,
                                  torch::Tensor& out_cos,
                                  torch::Tensor& out_sin);
};

}  // namespace layer
}  // namespace xllm

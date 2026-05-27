/* Copyright 2025-2026 The xLLM Authors.

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

#include <acl/acl.h>
#include <torch/torch.h>
#include <torch_npu/torch_npu.h>

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wattributes"
#endif

#include "torch_npu/csrc/core/npu/NPUGraph.h"

#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#include <optional>
#include <unordered_map>
#include <vector>

#include "core/framework/model/model_args.h"
#include "core/runtime/vision_encoder_graph_adapter.h"

namespace xllm::npu {

struct EncoderGraphOutput {
  torch::Tensor hidden_states;
  std::vector<torch::Tensor> deepstack_features;
};

// Bucket-based lazy capture/replay of the vision encoder as an ACL Graph.
//
// The manager is model-agnostic: it captures and replays the encoder loop
// purely through VisionEncoderGraphAdapter, so it never depends on any concrete
// model type. A VLM registers an adapter (typically itself) via set_adapter()
// and calls replay() from its encoder forward path.
class VisionEncoderAclGraphManager final {
 public:
  VisionEncoderAclGraphManager(const ModelArgs& args,
                               const torch::Device& device,
                               const std::vector<int64_t>& budgets);

  ~VisionEncoderAclGraphManager() = default;

  // Check if we have a captured graph that fits actual_tokens
  bool can_replay(int64_t actual_tokens) const;

  // Capture the encoder loop for a given budget. Returns false on failure
  // (exception during capture); the caller falls back to eager.
  bool capture(VisionEncoderGraphAdapter* adapter,
               torch::Tensor& hidden_states,
               torch::Tensor& cos_pos,
               torch::Tensor& sin_pos,
               torch::Tensor& cu_seqlens,
               std::vector<int>& cu_seqlens_vec,
               int64_t budget);

  // Replay a captured graph (lazy-captures if needed). Returns nullopt when no
  // bucket fits, the captured graph's segment count does not match the request,
  // or lazy capture fails — caller should fall back to eager.
  std::optional<EncoderGraphOutput> replay(torch::Tensor& hidden_states,
                                           torch::Tensor& cos_pos,
                                           torch::Tensor& sin_pos,
                                           torch::Tensor& cu_seqlens,
                                           std::vector<int>& cu_seqlens_vec,
                                           int64_t actual_num_tokens);

  // Register the adapter (the model owning the encoder layers). Not owned.
  void set_adapter(VisionEncoderGraphAdapter* adapter) { adapter_ = adapter; }

 private:
  struct EncoderPersistentParam {
    // Input persistent buffers
    torch::Tensor hidden_states;  // [budget, hidden_size]
    torch::Tensor cos_pos;        // [budget, head_dim]
    torch::Tensor sin_pos;        // [budget, head_dim]
    torch::Tensor cu_seqlens;     // [num_segments], int32, stable storage
    std::vector<int> cu_seqlens_vec;

    // Output persistent buffers (written inside graph)
    torch::Tensor output_hidden;                // [budget, hidden_size]
    std::vector<torch::Tensor> deepstack_outs;  // [budget/merge_sq, d_model]
  };

  struct BucketGraph {
    c10_npu::NPUGraph graph;
    uint32_t num_tokens = 0;
    size_t num_segments = 0;  // segment count this graph was captured with
    EncoderPersistentParam param;
  };

  int64_t select_budget(int64_t actual_tokens) const;

  void copy_inputs_to_persistent(EncoderPersistentParam& param,
                                 const torch::Tensor& hidden_states,
                                 const torch::Tensor& cos_pos,
                                 const torch::Tensor& sin_pos,
                                 const torch::Tensor& cu_seqlens,
                                 const std::vector<int>& cu_seqlens_vec,
                                 int64_t actual_tokens,
                                 int64_t bucket_size);

  void initialize_persistent_param(EncoderPersistentParam& param,
                                   int64_t budget,
                                   int32_t num_deepstack,
                                   size_t num_segments);

  std::vector<int64_t> budgets_;
  torch::Device device_;
  int64_t max_budget_;
  int64_t hidden_size_;
  int64_t head_dim_;
  int64_t d_model_;
  int64_t spatial_merge_size_;
  torch::ScalarType dtype_;

  std::unordered_map<int64_t, std::unique_ptr<BucketGraph>> graphs_;

  std::optional<c10_npu::NPUStream> capture_stream_;
  c10::DeviceIndex device_index_;

  VisionEncoderGraphAdapter* adapter_ = nullptr;
};

}  // namespace xllm::npu

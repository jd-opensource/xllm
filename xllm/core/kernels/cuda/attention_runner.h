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

#include <glog/logging.h>
#include <torch/torch.h>

#include <optional>
#include <string>

#include "core/common/global_flags.h"
#include "core/kernels/cuda/cuda_ops_api.h"
#include "core/kernels/param.h"
#include "global_capture_instance.h"

namespace xllm::kernel::cuda {

// Attention replay parameters shared across all layers in a prefill batch
struct AttentionReplayParams {
  ffi::Array<int64_t> plan_info;
  torch::Tensor q_cu_seq_lens;
  torch::Tensor kv_cu_seq_lens;
  uint32_t actual_num_tokens;  // All layers share the same actual_num_tokens
};

// AttentionRunner encapsulates batch_prefill for piecewise CUDA Graph
class AttentionRunner {
 public:
  AttentionRunner() = default;

  // Piecewise mode: capture phase
  void run_capture(const std::string& uri,
                   ffi::Array<int64_t> plan_info,
                   torch::Tensor float_workspace_buffer,
                   torch::Tensor int_workspace_buffer,
                   torch::Tensor page_locked_int_workspace_buffer,
                   torch::Tensor query,  // shape: [padded_num_tokens, ...]
                   torch::Tensor key,
                   torch::Tensor value,
                   torch::Tensor q_cu_seq_lens,
                   torch::Tensor kv_cu_seq_lens,
                   int64_t window_left,
                   double sm_scale,
                   torch::Tensor output,  // shape: [padded_num_tokens, ...]
                   std::optional<torch::Tensor>& output_lse,
                   uint32_t padded_num_tokens);

  // Piecewise mode: replay phase
  void run_replay(const AttentionReplayParams& params);

 private:
  // Captured flashiner workspace buffers
  torch::Tensor float_workspace_buffer_;
  torch::Tensor int_workspace_buffer_;
  torch::Tensor page_locked_int_workspace_buffer_;

  // Captured tensors (padded shape)
  torch::Tensor query_;
  torch::Tensor key_;
  torch::Tensor value_;
  torch::Tensor output_;

  // Captured parameters
  std::string uri_;
  int64_t window_size_left_;
  double scale_;
  uint32_t padded_num_tokens_;
};

// Wrapper function for batch_prefill that conditionally uses AttentionRunner
// for piecewise CUDA Graph capture
inline void batch_prefill_with_optional_piecewise_capture(
    const std::string& uri,
    ffi::Array<int64_t> plan_info,
    torch::Tensor float_workspace_buffer,
    torch::Tensor int_workspace_buffer,
    torch::Tensor page_locked_int_workspace_buffer,
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor q_cu_seq_lens,
    torch::Tensor kv_cu_seq_lens,
    int64_t window_left,
    double sm_scale,
    torch::Tensor output,
    std::optional<torch::Tensor>& output_lse,
    bool enable_cuda_graph) {
  // This function is only called for prefill, so is_prefill is always true
  if (FLAGS_enable_graph && FLAGS_enable_prefill_piecewise_graph &&
      ::xllm::runtime::cuda::GlobalCaptureInstance::get_instance()
          .is_capturing()) {
    // Create temporary runner
    AttentionRunner runner;

    // Get padded_num_tokens from query tensor shape (query is already padded)
    uint32_t padded_num_tokens = static_cast<uint32_t>(query.size(0));

    // Run capture
    runner.run_capture(uri,
                       plan_info,
                       float_workspace_buffer,
                       int_workspace_buffer,
                       page_locked_int_workspace_buffer,
                       query,
                       key,
                       value,
                       q_cu_seq_lens,
                       kv_cu_seq_lens,
                       window_left,
                       sm_scale,
                       output,
                       output_lse,
                       padded_num_tokens);

    // Register to GlobalCaptureInstance
    ::xllm::runtime::cuda::GlobalCaptureInstance::get_instance()
        .register_attention_runner(std::move(runner));
    return;
  }
  // Non-piecewise mode: directly call batch_prefill
  batch_prefill(uri,
                plan_info,
                float_workspace_buffer,
                int_workspace_buffer,
                page_locked_int_workspace_buffer,
                query,
                key,
                value,
                q_cu_seq_lens,
                kv_cu_seq_lens,
                window_left,
                sm_scale,
                output,
                output_lse,
                enable_cuda_graph);
}

}  // namespace xllm::kernel::cuda

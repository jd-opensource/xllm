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

#include "attention_runner.h"

#include <glog/logging.h>

#include "cuda_ops_api.h"
#include "global_capture_instance.h"

namespace xllm {
namespace kernel {
namespace cuda {

void AttentionRunner::run_capture(
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
    uint32_t padded_num_tokens) {
  // Temporarily end graph capture
  ::xllm::runtime::cuda::GlobalCaptureInstance::get_instance()
      .temporarily_end_graph();

  // Save uri for replay
  uri_ = uri;

  // Save all necessary tensors and parameters
  float_workspace_buffer_ = float_workspace_buffer;
  int_workspace_buffer_ = int_workspace_buffer;
  page_locked_int_workspace_buffer_ = page_locked_int_workspace_buffer;
  query_ = query;
  key_ = key;
  value_ = value;
  output_ = output;
  window_size_left_ = window_left;
  scale_ = sm_scale;
  padded_num_tokens_ = padded_num_tokens;

  // During piecewise graph capture, we don't execute attention here.
  // The attention will be executed in the replay phase together with the
  // captured MLP graphs to ensure correct execution order.

  // Resume graph capture
  ::xllm::runtime::cuda::GlobalCaptureInstance::get_instance()
      .temporarily_begin_graph();
}

void AttentionRunner::run_replay(const AttentionReplayParams& params) {
  // Slice query and output based on actual_num_tokens
  torch::Tensor query_slice =
      query_.slice(/*dim=*/0, /*start=*/0, /*end=*/params.actual_num_tokens);
  torch::Tensor key_slice =
      key_.slice(/*dim=*/0, /*start=*/0, /*end=*/params.actual_num_tokens);
  torch::Tensor value_slice =
      value_.slice(/*dim=*/0, /*start=*/0, /*end=*/params.actual_num_tokens);
  torch::Tensor output_slice =
      output_.slice(/*dim=*/0, /*start=*/0, /*end=*/params.actual_num_tokens);

  // Execute attention with sliced tensors
  // TODO: support output_lse for replay
  std::optional<torch::Tensor> output_lse = std::nullopt;
  batch_prefill(uri_,  // Use captured uri
                params.plan_info,
                float_workspace_buffer_,
                int_workspace_buffer_,
                page_locked_int_workspace_buffer_,
                query_slice,
                key_slice,
                value_slice,
                params.q_cu_seq_lens,
                params.kv_cu_seq_lens,
                window_size_left_,
                scale_,
                output_slice,
                output_lse,
                /*enable_cuda_graph=*/false);
}

}  // namespace cuda
}  // namespace kernel
}  // namespace xllm

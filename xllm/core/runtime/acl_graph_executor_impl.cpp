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

#include "acl_graph_executor_impl.h"

#include <c10/core/Device.h>
#include <c10/core/TensorOptions.h>
#include <glog/logging.h>
#include <torch/torch.h>
#include <torch_npu/csrc/libs/init_npu.h>
#include <torch_npu/torch_npu.h>

#include "core/common/global_flags.h"
#ifdef TORCH_HIGHER_THAN_PTA6
#include <torch_npu/csrc/framework/OpCommand.h>
#else
#include <torch_npu/csrc/aten/NPUNativeFunctions.h>
#include <torch_npu/csrc/framework/utils/OpPreparation.h>
#endif
#include "core/common/global_flags.h"
#include "core/common/metrics.h"

namespace xllm {

bool AclGraph::capture(CausalLM* model,
                       const ModelArgs& args,
                       const runtime::Options& options,
                       const torch::Tensor& tokens,
                       const torch::Tensor& positions,
                       const ModelInputParams& params,
                       std::vector<KVCache>& kv_cache,
                       uint32_t bucket_size) {
  // Save bucket size for this graph instance
  batch_size_ = bucket_size;

  // Get actual batch size from params
  const uint32_t actual_batch_size = params.num_sequences;

  // Note: This implementation only supports decode phase where all sequences in
  // batch have q_len=1 It does not support chunked_prefill, so
  // num_decode_tokens equals bucket_size for buffer allocation
  const int64_t num_decode_tokens = batch_size_;

  // Create persistent tensors for this specific batch size
  // These tensors will be reused across replay calls to avoid memory allocation
  auto& tensor_options = model->options();

  // Input tensors for decode tokens
  flatten_tokens_ =
      torch::zeros({num_decode_tokens},
                   torch::dtype(torch::kInt).device(tensor_options.device()));
  flatten_positions_ =
      torch::zeros({num_decode_tokens},
                   torch::dtype(torch::kInt).device(tensor_options.device()));
  new_cache_slots_ =
      torch::zeros({num_decode_tokens},
                   torch::dtype(torch::kInt).device(tensor_options.device()));

  // Sequence length tensors
  q_seq_lens_ = torch::zeros(
      {batch_size_}, torch::dtype(torch::kInt).device(tensor_options.device()));
  kv_seq_lens_ = torch::zeros(
      {batch_size_}, torch::dtype(torch::kInt).device(tensor_options.device()));

  // Block table tensors with maximum possible size
  const auto block_size = options.block_size();
  const int64_t max_block_table_len =
      (FLAGS_max_seq_len_for_graph_mode + block_size - 1) / block_size + 1;
  block_tables_ =
      torch::zeros({batch_size_, max_block_table_len},
                   torch::dtype(torch::kInt).device(tensor_options.device()));

  // Output tensor for hidden states
  hidden_states_ =
      torch::zeros({num_decode_tokens, args.hidden_size()},
                   torch::dtype(torch::kFloat).device(tensor_options.device()));

  torch::npu::synchronize();

  // Begin graph capture using NPUGraph mempool for temporary tensor management
  // Get current NPU stream from libtorch NPU API
  aclrtStream stream =
      c10_npu::getCurrentNPUStream(tensor_options.device().index()).stream();

  // Create ModelInputParams using graph's own persistent buffers
  ModelInputParams graph_params = params;
  graph_params.kv_seq_lens = kv_seq_lens_;
  graph_params.q_seq_lens = q_seq_lens_;
  graph_params.new_cache_slots = new_cache_slots_;
  graph_params.block_tables = block_tables_;

  // Set graph_buffer if available in params
  graph_params.graph_buffer = graph_buffer_;

  // Copy input data to graph persistent buffers before capture
  copy_data_to_graph_buffer(tokens, positions, params, actual_batch_size);

  // Synchronize stream to ensure all data is copied to graph persistent buffers
  aclrtSynchronizeStream(stream);

  // Use secondary stream for graph capture to avoid blocking main stream
  bool need_restore_stream = false;
  if (c10_npu::getCurrentNPUStream(tensor_options.device().index()) ==
      c10_npu::getDefaultNPUStream(tensor_options.device().index())) {
    auto secondary_stream =
        c10_npu::getStreamFromPool(true, tensor_options.device().index());
    c10_npu::setCurrentNPUStream(secondary_stream);
    need_restore_stream = true;
  }
  LOG(INFO) << "capture begin, bucket_size: " << bucket_size
            << " actual_batch_size: " << actual_batch_size << std::endl;
  graph_.capture_begin();

  // Execute forward pass - NPUGraph mempool manages temporary tensors
  auto forward_result = model->forward(
      {flatten_tokens_}, {flatten_positions_}, kv_cache, {graph_params});

  // Store result in persistent buffer owned by NPUGraph mempool
  hidden_states_ = forward_result;
  graph_.capture_end();
  if (need_restore_stream) {
    c10_npu::setCurrentNPUStream(
        c10_npu::getDefaultNPUStream(tensor_options.device().index()));
  }

  // Synchronize and test replay to verify graph capture
  aclrtSynchronizeStream(stream);

  graph_.replay();

  // aclrtSynchronizeStream(stream);
  return true;
}

torch::Tensor AclGraph::replay(const torch::Tensor& tokens,
                               const torch::Tensor& positions,
                               const ModelInputParams& params) {
  const uint32_t actual_batch_size = params.num_sequences;
  CHECK_LE(actual_batch_size, batch_size_)
      << "batch size mismatch: expected <= " << batch_size_ << ", got "
      << actual_batch_size;

  // Copy new input data to graph persistent buffers
  copy_data_to_graph_buffer(tokens, positions, params, actual_batch_size);

  // Replay captured graph - NPUGraph mempool reuses temporary tensors
  // Get current NPU stream from libtorch NPU API
  aclrtStream stream = c10_npu::getCurrentNPUStream().stream();

  graph_.replay();

  // this is necessary to ensure the graph replay is completed
  // aclError st = aclrtSynchronizeStream(stream);
  // CHECK_EQ(st, ACL_SUCCESS)
  // << "aclrtSynchronizeStream failed, error code: " << st;

  // Return only the actual batch size portion of hidden states
  return get_hidden_states(actual_batch_size);
}

AclGraphExecutorImpl::AclGraphExecutorImpl(CausalLM* model,
                                           const ModelArgs& args,
                                           const torch::Device& device,
                                           const runtime::Options& options)
    : model_(model), args_(args), device_(device), options_(options) {
  // No pre-initialization needed, graphs will be created lazily in run() method
}

ForwardInput AclGraphExecutorImpl::prepare_inputs(Batch& batch) {
  // Prepare inputs for workers
  return batch.prepare_forward_input(options_.num_decoding_tokens(), 0, args_);
}

// Main execution method with graph optimization for decode phase
// tokens: [num_decode_tokens]
// positions: [num_decode_tokens] token pos in the sequence
// returns: [num_decode_tokens, hidden_size]
torch::Tensor AclGraphExecutorImpl::run(
    const std::vector<torch::Tensor>& tokens,
    const std::vector<torch::Tensor>& positions,
    std::vector<KVCache>& kv_caches,
    const std::vector<ModelInputParams>& params) {
  // no mirco batch in decode phase
  const torch::Tensor& tokens_tensor = tokens[0];
  const torch::Tensor& positions_tensor = positions[0];
  const ModelInputParams& params_single = params[0];

  // If not in decode phase, use eager mode directly without acl graph
  if (!params_single.batch_forward_type.is_decode()) {
    COUNTER_INC(num_model_execution_total_eager);
    return model_->forward(tokens, positions, kv_caches, params);
  }

  // Only use acl graph in decode phase for performance optimization
  // Get actual batch size from tokens shape (num_decode_tokens /
  // num_decoding_tokens)
  const uint32_t n_tokens = tokens_tensor.size(/*dim=*/0);
  const uint32_t actual_batch_size = n_tokens / options_.num_decoding_tokens();
  const uint32_t bucket_size = get_bucket_size(actual_batch_size);

  // Check if conditions are suitable for graph execution (replay or capture)
  const auto max_seq_len = FLAGS_max_seq_len_for_graph_mode > 0
                               ? FLAGS_max_seq_len_for_graph_mode
                               : args_.max_position_embeddings();
  const bool seq_len_supported = params_single.kv_max_seq_len <= max_seq_len;
  // Each sequence has the same number of decoding tokens
  const bool same_num_decoding_tokens =
      params_single.q_max_seq_len == options_.num_decoding_tokens() &&
      n_tokens == actual_batch_size * options_.num_decoding_tokens();

  // Combined condition for graph capture support
  // ACL graph executor only supports single tensor inputs (no micro-batching)
  const bool single_input =
      (tokens.size() == 1) && (positions.size() == 1) && (params.size() == 1);
  const bool capture_supported =
      single_input && seq_len_supported && same_num_decoding_tokens;

  // Early return if conditions are not suitable for graph operations
  if (!capture_supported) {
    COUNTER_INC(num_model_execution_total_eager);
    return model_->forward(tokens, positions, kv_caches, params);
  }

  // Check if captured graph exists for this bucket size
  auto it = graphs_.find(bucket_size);
  if (it != graphs_.end()) {
    // Replay the existing graph
    return it->second->replay(tokens_tensor, positions_tensor, params_single);
  }

  // Graph doesn't exist for this bucket size, try to create it lazily
  auto graph = std::make_unique<AclGraph>();
  bool capture_success = graph->capture(model_,
                                        args_,
                                        options_,
                                        tokens_tensor,
                                        positions_tensor,
                                        params_single,
                                        kv_caches,
                                        bucket_size);

  if (capture_success) {
    LOG(INFO) << "Lazy capturing ACL graph for bucket size: " << bucket_size
              << " (actual batch size: " << actual_batch_size << ") done";

    // Save the graph for future reuse
    graphs_[bucket_size] = std::move(graph);

    // Return the output from capture (no need to replay since capture
    // already executed)
    return graphs_[bucket_size]->get_hidden_states(actual_batch_size);
  }

  // Fallback to eager mode if capture fails
  LOG(ERROR) << "Failed to capture ACL graph for bucket size: " << bucket_size;
  COUNTER_INC(num_model_execution_total_eager);
  return model_->forward(tokens, positions, kv_caches, params);
}

void AclGraph::copy_data_to_graph_buffer(const torch::Tensor& tokens,
                                         const torch::Tensor& positions,
                                         const ModelInputParams& params,
                                         uint32_t actual_batch_size) {
  // Copy data from input parameters to persistent graph tensors
  // This avoids memory allocation during replay by reusing pre-allocated
  // buffers
  // Only copy the actual batch size portion to avoid processing padded data
  flatten_tokens_.slice(/*dim=*/0, /*start=*/0, /*end=*/actual_batch_size)
      .copy_(tokens, /*non_blocking=*/true);
  flatten_positions_.slice(/*dim=*/0, /*start=*/0, /*end=*/actual_batch_size)
      .copy_(positions, /*non_blocking=*/true);
  q_seq_lens_.slice(/*dim=*/0, /*start=*/0, /*end=*/actual_batch_size)
      .copy_(params.q_seq_lens, /*non_blocking=*/true);
  kv_seq_lens_.slice(/*dim=*/0, /*start=*/0, /*end=*/actual_batch_size)
      .copy_(params.kv_seq_lens, /*non_blocking=*/true);
  new_cache_slots_.slice(/*dim=*/0, /*start=*/0, /*end=*/actual_batch_size)
      .copy_(params.new_cache_slots, /*non_blocking=*/true);

  // Copy block table data with left alignment (2D matrix view)
  // Only copy the actual batch size portion to avoid processing padded data
  const int64_t actual_block_table_len = params.block_tables.size(1);

  // Copy data to the left-aligned portion of the buffer
  block_tables_.slice(/*dim=*/0, /*start=*/0, /*end=*/actual_batch_size)
      .slice(/*dim=*/1, /*start=*/0, /*end=*/actual_block_table_len)
      .copy_(params.block_tables, /*non_blocking=*/true);
}

void AclGraph::print_graph_tensors() const {
  LOG(INFO) << "graph flatten_tokens_: " << flatten_tokens_ << std::endl;
  LOG(INFO) << "graph flatten_positions_: " << flatten_positions_ << std::endl;
  LOG(INFO) << "graph new_cache_slots_: " << new_cache_slots_ << std::endl;
  LOG(INFO) << "graph q_seq_lens_: " << q_seq_lens_ << std::endl;
  LOG(INFO) << "graph kv_seq_lens_: " << kv_seq_lens_ << std::endl;
  LOG(INFO) << "graph block_tables_: " << block_tables_ << std::endl;
  LOG(INFO) << "graph hidden_states_: " << hidden_states_ << std::endl;
  LOG(INFO) << "graph graph_buffer_ defined: " << graph_buffer_.defined()
            << std::endl;
  if (graph_buffer_.defined()) {
    LOG(INFO) << "graph graph_buffer_ size: " << graph_buffer_.numel()
              << std::endl;
  }
}

uint32_t AclGraphExecutorImpl::get_bucket_size(uint32_t batch_size) const {
  if (batch_size <= 1) {
    return 1;
  } else if (batch_size <= 2) {
    return 2;
  } else if (batch_size <= 4) {
    return 4;
  } else if (batch_size <= 8) {
    return 8;
  } else {
    // For batch_size > 8, use multiples of 8
    return ((batch_size + 7) / 8) * 8;
  }
}

}  // namespace xllm

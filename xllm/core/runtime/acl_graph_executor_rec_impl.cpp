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

#include "acl_graph_executor_rec_impl.h"

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
#include "core/common/acl_graph_runtime.h"
#include "core/common/global_flags.h"
#include "core/common/metrics.h"

namespace xllm {

bool AclGraphRec::capture(CausalLM* model,
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
  const uint32_t actual_batch_size = params.num_sequences * params.beam_width;

  // Note: This implementation only supports decode phase where all sequences in
  // batch have q_len=1 It does not support chunked_prefill, so
  // num_decode_tokens equals bucket_size for buffer allocation
  const int64_t num_decode_tokens = tokens.size(0);

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

  q_seq_lens_ =
      torch::zeros(params.q_seq_lens.sizes(),
                   torch::dtype(torch::kInt).device(tensor_options.device()));
  kv_seq_lens_ =
      torch::zeros(params.kv_seq_lens.sizes(),
                   torch::dtype(torch::kInt).device(tensor_options.device()));

  // Block table tensors with maximum possible size
  const auto block_size = options.block_size();
  const int64_t max_tokens =
      FLAGS_max_decode_rounds == 0 ? FLAGS_max_token_per_req : 1;
  const int64_t max_block_table_len =
      (max_tokens + block_size - 1) / block_size + 1;
  block_tables_ =
      torch::zeros(params.block_tables.sizes(),
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

  graph_params.graph_buffer_rec = graph_buffer_;

  // Copy input data to graph persistent buffers before capture
  copy_data_to_graph_buffer(tokens, positions, params, num_decode_tokens);

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

  // Mark this thread as inside ACL graph capture so that NPU layers can align
  // ATB execute stream with the NPUGraph capture stream. The flag is reset
  // automatically when the guard goes out of scope.
  AclGraphCaptureGuard capture_guard;

  graph_.capture_begin();

  // Execute forward pass - NPUGraph mempool manages temporary tensors
  auto forward_result = model->forward(
      flatten_tokens_, flatten_positions_, kv_cache, graph_params);

  // Store result in persistent buffer owned by NPUGraph mempool
  hidden_states_ = forward_result;
  graph_.capture_end();
  // if (need_restore_stream) {
  //   c10_npu::setCurrentNPUStream(
  //       c10_npu::getDefaultNPUStream(tensor_options.device().index()));
  // }

  // Synchronize and test replay to verify graph capture
  aclrtSynchronizeStream(stream);

  graph_.replay();

  // aclrtSynchronizeStream(stream);
  return true;
}

torch::Tensor AclGraphRec::replay(const torch::Tensor& tokens,
                                  const torch::Tensor& positions,
                                  const ModelInputParams& params) {
  const uint32_t actual_batch_size = params.num_sequences * params.beam_width;
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

AclGraphRecExecutorImpl::AclGraphRecExecutorImpl(
    CausalLM* model,
    const ModelArgs& args,
    const torch::Device& device,
    const runtime::Options& options)
    : model_(model), args_(args), device_(device), options_(options) {
  // No pre-initialization needed, graphs will be created lazily in run() method
}

ForwardInput AclGraphRecExecutorImpl::prepare_inputs(Batch& batch) {
  // Prepare inputs for workers
  return batch.prepare_forward_input(options_.num_decoding_tokens(), 0, args_);
}

// Main execution method with graph optimization for decode phase
// tokens: [num_decode_tokens]
// positions: [num_decode_tokens] token pos in the sequence
// returns: [num_decode_tokens, hidden_size]
torch::Tensor AclGraphRecExecutorImpl::run(const torch::Tensor& tokens,
                                           const torch::Tensor& positions,
                                           std::vector<KVCache>& kv_caches,
                                           const ModelInputParams& params) {
  // no micro batch in decode phase
  const torch::Tensor& tokens_tensor = tokens;
  const torch::Tensor& positions_tensor = positions;
  const ModelInputParams& params_single = params;
  // Identify decode phase using q_max_seq_len for precise detection
  // Decode phase: all sequences have q_seq_len == 1 (generating one token at a
  // time) Prefill phase: sequences have q_seq_len > 1 (processing multiple
  // prompt tokens) We also check empty_kv_cache to ensure KV cache is not empty
  // (not first forward pass)
  const bool has_multi_step_meta =
      params_single.decode_q_seq_lens.defined() &&
      params_single.decode_q_seq_lens.numel() > 0 &&
      params_single.current_round > 0;
  const bool is_single_step_decode =
      (params_single.q_max_seq_len == 1) && !params_single.empty_kv_cache;
  const bool in_decoding_phase = is_single_step_decode || has_multi_step_meta;

  // If not in decode phase, use eager mode directly without acl graph
  if (!in_decoding_phase) {
    COUNTER_INC(num_model_execution_total_eager);
    return model_->forward(
        tokens_tensor, positions_tensor, kv_caches, params_single);
  }

  // Only use acl graph in decode phase for performance optimization
  // Get actual batch size from tokens shape (num_decode_tokens /
  // num_decoding_tokens)
  const uint32_t n_tokens = tokens_tensor.size(/*dim=*/0);
  const uint32_t actual_batch_size = n_tokens / options_.num_decoding_tokens();

  // VLOG debug for decode meta and batch size inference
  const uint32_t expected_tokens =
      options_.num_decoding_tokens() * params_single.num_sequences *
      std::max<uint32_t>(1, params_single.beam_width);
  const uint32_t bucket_size = get_bucket_size(actual_batch_size);

  // Check if conditions are suitable for graph execution (replay or capture)
  const auto max_seq_len = FLAGS_max_token_per_req > 0
                               ? static_cast<uint32_t>(FLAGS_max_token_per_req)
                               : args_.max_position_embeddings();
  const bool seq_len_supported = params_single.kv_max_seq_len <= max_seq_len;
  // Each sequence has the same number of decoding tokens
  const bool same_num_decoding_tokens_single =
      params_single.q_max_seq_len == options_.num_decoding_tokens() &&
      n_tokens == actual_batch_size * options_.num_decoding_tokens();

  // Multi-step: Use q_len configured in decode_q_seq_lens_vec to align
  // num_decoding_tokens
  bool same_num_decoding_tokens_multi = false;
  if (has_multi_step_meta && !params_single.decode_q_seq_lens_vec.empty()) {
    const auto step_q_len = params_single.decode_q_seq_lens_vec[0];
    same_num_decoding_tokens_multi =
        (step_q_len == options_.num_decoding_tokens()) &&
        (n_tokens == actual_batch_size * options_.num_decoding_tokens());
  }
  // n_tokens = 512, actual_batch_size = 512 ,num_decoding_tokens = 1
  const bool same_num_decoding_tokens =
      same_num_decoding_tokens_single || same_num_decoding_tokens_multi;
  // Combined condition for graph capture support
  // ExecutorImpl 层没有 micro-batch，这里始终是单一 Tensor 输入。
  const bool single_input = true;
  const bool capture_supported =
      single_input && seq_len_supported && same_num_decoding_tokens;

  // Early return if conditions are not suitable for graph operations
  if (!capture_supported) {
    COUNTER_INC(num_model_execution_total_eager);
    return model_->forward(
        tokens_tensor, positions_tensor, kv_caches, params_single);
  }

  // Check if captured graph exists for this bucket size
  auto it = graphs_.find(bucket_size);
  if (it != graphs_.end()) {
    // Replay the existing graph
    return it->second->replay(tokens_tensor, positions_tensor, params_single);
  }

  // Graph doesn't exist for this bucket size, try to create it lazily
  auto graph = std::make_unique<AclGraphRec>();
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
  return model_->forward(
      tokens_tensor, positions_tensor, kv_caches, params_single);
}

void AclGraphRec::copy_data_to_graph_buffer(const torch::Tensor& tokens,
                                            const torch::Tensor& positions,
                                            const ModelInputParams& params,
                                            uint32_t actual_batch_size) {
  // Copy data from input parameters to persistent graph tensors
  // This avoids memory allocation during replay by reusing pre-allocated
  // buffers. We avoid relying on `actual_batch_size` being perfectly aligned
  // with all tensor shapes, and instead clamp copy lengths by real sizes to
  // prevent shape/broadcast issues.

  // ---- token-level tensors ----
  const int64_t tokens_len = tokens.size(0);
  const int64_t max_tokens = flatten_tokens_.size(0);
  const int64_t copy_tokens = std::min<int64_t>(max_tokens, tokens_len);

  if (copy_tokens > 0) {
    flatten_tokens_.slice(/*dim=*/0, /*start=*/0, /*end=*/copy_tokens)
        .copy_(tokens.slice(/*dim=*/0, /*start=*/0, /*end=*/copy_tokens),
               /*non_blocking=*/true);
    flatten_positions_.slice(/*dim=*/0, /*start=*/0, /*end=*/copy_tokens)
        .copy_(positions.slice(/*dim=*/0, /*start=*/0, /*end=*/copy_tokens),
               /*non_blocking=*/true);
  }

  // ---- sequence-level tensors (batch-wise) ----
  if (params.q_seq_lens.defined() && params.q_seq_lens.numel() > 0) {
    const int64_t q_src = params.q_seq_lens.size(0);
    const int64_t q_dst = q_seq_lens_.size(0);
    const int64_t q_copy = std::min<int64_t>(q_src, q_dst);
    if (q_copy > 0) {
      q_seq_lens_.slice(/*dim=*/0, /*start=*/0, /*end=*/q_copy)
          .copy_(
              params.q_seq_lens.slice(/*dim=*/0, /*start=*/0, /*end=*/q_copy),
              /*non_blocking=*/true);
    }
  }

  if (params.kv_seq_lens.defined() && params.kv_seq_lens.numel() > 0) {
    const int64_t kv_src = params.kv_seq_lens.size(0);
    const int64_t kv_dst = kv_seq_lens_.size(0);
    const int64_t kv_copy = std::min<int64_t>(kv_src, kv_dst);
    if (kv_copy > 0) {
      kv_seq_lens_.slice(/*dim=*/0, /*start=*/0, /*end=*/kv_copy)
          .copy_(
              params.kv_seq_lens.slice(/*dim=*/0, /*start=*/0, /*end=*/kv_copy),
              /*non_blocking=*/true);
    }
  }

  // // new_cache_slots may be empty in some decode rounds, so we copy safely
  // based on the actual length.
  if (!FLAGS_enable_beam_search_kernel && params.new_cache_slots.defined() &&
      params.new_cache_slots.numel() > 0) {
    const int64_t ncs_src = params.new_cache_slots.size(0);
    const int64_t ncs_dst = new_cache_slots_.size(0);
    const int64_t ncs_copy = std::min<int64_t>(ncs_src, ncs_dst);
    if (ncs_copy > 0) {
      new_cache_slots_.slice(/*dim=*/0, /*start=*/0, /*end=*/ncs_copy)
          .copy_(params.new_cache_slots.slice(
                     /*dim=*/0, /*start=*/0, /*end=*/ncs_copy),
                 /*non_blocking=*/true);
    }
  }

  if (params.block_tables.defined() && params.block_tables.numel() > 0) {
    const int64_t src_bt_batch = params.block_tables.size(0);
    const int64_t src_bt_len = params.block_tables.size(1);
    const int64_t dst_bt_batch = block_tables_.size(0);
    const int64_t dst_bt_len = block_tables_.size(1);

    const int64_t bt_batch = std::min<int64_t>(src_bt_batch, dst_bt_batch);
    const int64_t bt_len = std::min<int64_t>(src_bt_len, dst_bt_len);

    if (bt_batch > 0 && bt_len > 0) {
      block_tables_.slice(/*dim=*/0, /*start=*/0, /*end=*/bt_batch)
          .slice(/*dim=*/1, /*start=*/0, /*end=*/bt_len)
          .copy_(params.block_tables
                     .slice(/*dim=*/0, /*start=*/0, /*end=*/bt_batch)
                     .slice(/*dim=*/1, /*start=*/0, /*end=*/bt_len),
                 /*non_blocking=*/true);
    }
  }
}

void AclGraphRec::print_graph_tensors() const {
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

uint32_t AclGraphRecExecutorImpl::get_bucket_size(uint32_t batch_size) const {
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

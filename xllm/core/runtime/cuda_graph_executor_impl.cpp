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

#include "cuda_graph_executor_impl.h"

#include <c10/core/Device.h>
#include <c10/core/TensorOptions.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <numeric>

#include "core/common/global_flags.h"
#include "core/common/metrics.h"
#include "core/platform/device.h"
#include "core/platform/stream.h"
#include "core/util/utils.h"
#include "kernels/cuda/utils.h"
#include "layers/common/attention_metadata.h"
#include "layers/common/attention_metadata_builder.h"
#include "layers/cuda/flashinfer_planinfo.h"

namespace xllm {

// CudaGraphPersistentParam implementation
CudaGraphPersistentParam::CudaGraphPersistentParam(
    const ModelArgs& args,
    const torch::Device& device,
    const runtime::Options& options)
    : args_(args), device_(device), options_(options) {
  // Use max_tokens_per_batch for first dimension size
  const int64_t max_tokens_per_batch = FLAGS_max_tokens_per_batch;
  // num_sequences
  const int64_t max_seqs_per_batch = options.max_seqs_per_batch();
  auto tensor_options = torch::TensorOptions().device(device);

  const int64_t max_seq_len = FLAGS_max_seq_len_for_graph_mode > 0
                                  ? FLAGS_max_seq_len_for_graph_mode
                                  : args_.max_position_embeddings();

  // Create persistent tensors with max_tokens_per_batch as first dimension
  persistent_tokens_ = torch::zeros({max_tokens_per_batch},
                                    torch::dtype(torch::kInt).device(device));
  persistent_positions_ = torch::zeros(
      {max_tokens_per_batch}, torch::dtype(torch::kInt).device(device));
  persistent_new_cache_slots_ = torch::zeros(
      {max_tokens_per_batch}, torch::dtype(torch::kInt).device(device));

  // q_seq_lens is q_cu_seq_lens in GPU Model.
  // kv_seq_lens is kv_cu_seq_lens in GPU Model.
  q_seq_lens_ = torch::zeros({max_seqs_per_batch + 1},
                             torch::dtype(torch::kInt).device(device));
  kv_seq_lens_ = torch::zeros({max_seqs_per_batch + 1},
                              torch::dtype(torch::kInt).device(device));

  // Block table tensors with maximum possible size
  const auto block_size = options.block_size();
  const int64_t max_block_table_len =
      (max_seq_len + block_size - 1) / block_size + 1;
  persistent_block_tables_ =
      torch::zeros({max_seqs_per_batch, max_block_table_len},
                   torch::dtype(torch::kInt).device(device));

  // Output tensor for hidden states
  torch::ScalarType dtype = util::parse_dtype(args.dtype(), device);
  if (args.dtype() == "float" || args.dtype() == "float32") {
    LOG(WARNING)
        << "Cuda graph executor init hidden_states compatible with float32 "
           "dtype: float32. This should not happen in production but for test.";
    dtype = torch::kFloat32;
  }
  hidden_states_ = torch::zeros({max_tokens_per_batch, args.hidden_size()},
                                torch::dtype(dtype).device(device));

  // FlashInfer decode mode parameters
  // paged_kv_indptr: shape [max_seqs_per_batch + 1]
  persistent_paged_kv_indptr_ = torch::zeros(
      {max_seqs_per_batch + 1}, torch::dtype(torch::kInt).device(device));

  // paged_kv_indices: maximum size based on max blocks
  // Estimate max blocks: max_seqs_per_batch * max_block_table_len
  const int64_t max_paged_kv_indices_size =
      max_seqs_per_batch * max_block_table_len;
  persistent_paged_kv_indices_ = torch::zeros(
      {max_paged_kv_indices_size}, torch::dtype(torch::kInt).device(device));

  // paged_kv_last_page_len: shape [max_seqs_per_batch]
  persistent_paged_kv_last_page_len_ = torch::zeros(
      {max_seqs_per_batch}, torch::dtype(torch::kInt).device(device));

  // For decode mode, each sequence has 1 token, so qo_indptr = [0, 1, 2, ...,
  // max_seqs_per_batch]
  persistent_decode_qo_indptr_ = torch::arange(
      0, max_seqs_per_batch + 1, torch::dtype(torch::kInt).device(device));
  // will be updated by q_cu_seq_lens, q_cu_seq_lens is the cumulative sum of
  // q_seq_lens
  persistent_chunked_prefill_qo_indptr_ = torch::zeros(
      {max_seqs_per_batch + 1}, torch::dtype(torch::kInt).device(device));
}

std::optional<ModelInputParams> CudaGraphPersistentParam::update(
    const torch::Tensor& tokens,
    const torch::Tensor& k_cache,
    const torch::Tensor& v_cache,
    const torch::Tensor& positions,
    const ModelInputParams& params,
    uint32_t padded_num_tokens,
    bool return_capture_params) {
  std::optional<ModelInputParams> params_for_capture;
  if (return_capture_params) {
    CHECK_GT(padded_num_tokens, 0)
        << "padded_num_tokens must be > 0 when return_capture_params is true";
    params_for_capture = std::make_optional<ModelInputParams>(params);
  }
  // Build attn_metadata with original model_input_params. So we can set actual
  // batch size in plan_info.
  std::shared_ptr<layer::AttentionMetadata> attn_metadata =
      std::make_shared<layer::AttentionMetadata>(
          layer::AttentionMetadataBuilder::build(params));
  attn_metadata->enable_cuda_graph = true;

  const uint32_t actual_num_tokens = tokens.size(0);
  const int64_t actual_batch_size = params.num_sequences;

  // Copy data from input parameters to persistent graph tensors
  VLOG(kGraphExecutorLogVerboseLevel)
      << "copy_ tokens: src shape=" << tokens.sizes() << ", dst slice shape=["
      << actual_num_tokens << "]";
  persistent_tokens_.slice(/*dim=*/0, /*start=*/0, /*end=*/actual_num_tokens)
      .copy_(tokens, /*non_blocking=*/true);
  VLOG(kGraphExecutorLogVerboseLevel)
      << "copy_ positions: src shape=" << positions.sizes()
      << ", dst slice shape=[" << actual_num_tokens << "]";
  persistent_positions_.slice(/*dim=*/0, /*start=*/0, /*end=*/actual_num_tokens)
      .copy_(positions, /*non_blocking=*/true);

  // q_seq_lens is q_cu_seq_lens in GPU Model.
  // kv_seq_lens is kv_cu_seq_lens in GPU Model.
  VLOG(kGraphExecutorLogVerboseLevel)
      << "copy_ q_seq_lens: src shape=" << params.q_seq_lens.sizes()
      << ", dst slice shape=[" << actual_batch_size + 1 << "]";
  q_seq_lens_.slice(/*dim=*/0, /*start=*/0, /*end=*/actual_batch_size + 1)
      .copy_(params.q_seq_lens, /*non_blocking=*/true);
  VLOG(kGraphExecutorLogVerboseLevel)
      << "copy_ kv_seq_lens: src shape=" << params.kv_seq_lens.sizes()
      << ", dst slice shape=[" << actual_batch_size + 1 << "]";
  kv_seq_lens_.slice(/*dim=*/0, /*start=*/0, /*end=*/actual_batch_size + 1)
      .copy_(params.kv_seq_lens, /*non_blocking=*/true);
  VLOG(kGraphExecutorLogVerboseLevel)
      << "copy_ new_cache_slots: src shape=" << params.new_cache_slots.sizes()
      << ", dst slice shape=[" << actual_num_tokens << "]";
  persistent_new_cache_slots_
      .slice(/*dim=*/0, /*start=*/0, /*end=*/actual_num_tokens)
      .copy_(params.new_cache_slots, /*non_blocking=*/true);

  // Copy block table data
  const int64_t actual_block_table_len = params.block_tables.size(1);
  auto slice_persistent_block_tables =
      persistent_block_tables_
          .slice(/*dim=*/0, /*start=*/0, /*end=*/actual_batch_size)
          .slice(/*dim=*/1, /*start=*/0, /*end=*/actual_block_table_len);
  VLOG(kGraphExecutorLogVerboseLevel)
      << "copy_ block_tables: src shape=" << params.block_tables.sizes()
      << ", dst slice shape=" << slice_persistent_block_tables.sizes();
  slice_persistent_block_tables.copy_(params.block_tables,
                                      /*non_blocking=*/true);

  // Update persistent embedding from input_embedding if available
  const auto& embedding = params.input_embedding;
  if (embedding.defined()) {
    const int64_t embedding_tokens = embedding.size(0);

    // Initialize persistent_embedding_ if needed and not already initialized
    if (persistent_embedding_.numel() == 0) {
      const int64_t max_tokens_per_batch = FLAGS_max_tokens_per_batch;
      const int64_t embedding_dim = embedding.size(1);
      torch::ScalarType dtype = util::parse_dtype(args_.dtype(), device_);
      persistent_embedding_ =
          torch::zeros({max_tokens_per_batch, embedding_dim},
                       torch::dtype(dtype).device(device_));
    }

    // Copy embedding data to persistent buffer
    VLOG(kGraphExecutorLogVerboseLevel)
        << "copy_ embedding: src shape=" << embedding.sizes()
        << ", dst slice shape=[" << embedding_tokens << ", "
        << embedding.size(1) << "]";
    persistent_embedding_
        .slice(/*dim=*/0, /*start=*/0, /*end=*/embedding_tokens)
        .copy_(embedding, /*non_blocking=*/true);
  }

  // FlashInfer decode parameters update (if present)
  CHECK(params.paged_kv_indptr.defined())
      << "paged_kv_indptr should not be null";
  VLOG(kGraphExecutorLogVerboseLevel)
      << "copy_ paged_kv_indptr: src shape=" << params.paged_kv_indptr.sizes()
      << ", dst slice shape=[" << (actual_batch_size + 1) << "]";
  if (VLOG_IS_ON(kGraphExecutorLogVerboseLevel)) {
    torch::Tensor paged_kv_indptr_cpu = params.paged_kv_indptr.to(torch::kCPU);
    VLOG(kGraphExecutorLogVerboseLevel)
        << "copy_ paged_kv_indptr: src values=" << paged_kv_indptr_cpu;
  }
  persistent_paged_kv_indptr_
      .slice(/*dim=*/0,
             /*start=*/0,
             /*end=*/actual_batch_size + 1)
      .copy_(params.paged_kv_indptr, /*non_blocking=*/true);

  CHECK(params.paged_kv_indices.defined())
      << "paged_kv_indices should not be null";
  const int64_t actual_indices_size = params.paged_kv_indices.size(0);
  VLOG(kGraphExecutorLogVerboseLevel)
      << "copy_ paged_kv_indices: src shape=" << params.paged_kv_indices.sizes()
      << ", dst slice shape=[" << actual_indices_size << "]";
  persistent_paged_kv_indices_
      .slice(/*dim=*/0,
             /*start=*/0,
             /*end=*/actual_indices_size)
      .copy_(params.paged_kv_indices, /*non_blocking=*/true);
  CHECK(params.paged_kv_last_page_len.defined())
      << "paged_kv_last_page_len should not be null";
  VLOG(kGraphExecutorLogVerboseLevel)
      << "copy_ paged_kv_last_page_len: src shape="
      << params.paged_kv_last_page_len.sizes() << ", dst slice shape=["
      << actual_batch_size << "]";
  persistent_paged_kv_last_page_len_
      .slice(/*dim=*/0,
             /*start=*/0,
             /*end=*/actual_batch_size)
      .copy_(params.paged_kv_last_page_len, /*non_blocking=*/true);

  // Convert cumulative lengths to individual sequence lengths using torch::diff
  // This matches the behavior in attention_metadata_builder.cpp for decode mode
  attn_metadata->kv_seq_lens =
      torch::diff(kv_seq_lens(/*actual_batch_size=*/actual_batch_size + 1));
  // Set FlashInfer decode parameters (always update, not just for capture)
  // This ensures attn_metadata points to updated persistent buffers for
  // plan_info calculation
  attn_metadata->paged_kv_indptr =
      persistent_paged_kv_indptr(actual_batch_size);
  attn_metadata->paged_kv_indices =
      persistent_paged_kv_indices(actual_indices_size);
  attn_metadata->paged_kv_last_page_len =
      persistent_paged_kv_last_page_len(actual_batch_size);
  // qo_indptr is q_cu_seq_lens in GPU Model.
  attn_metadata->qo_indptr = persistent_decode_qo_indptr(actual_batch_size);

  // Update plan_info if attn_metadata exists and enable_cuda_graph is true
  // This ensures plan_info is updated before CUDA graph capture/replay
  {
    // Get attention parameters from ModelArgs
    const int32_t head_dim = args_.head_dim();
    const int64_t n_heads = args_.n_heads();
    const int64_t n_kv_heads = args_.n_kv_heads().value_or(n_heads);
    const int64_t block_size = options_.block_size();

    // Get sliding_window from ModelArgs (default to -1 if not available)
    // Note: sliding_window in ModelArgs is the actual window size, but in
    // attention it's used as window_size_left which is typically sliding_window
    // - 1. This matches the behavior in attention.cpp where sliding_window_ is
    // initialized as sliding_window - 1 regardless of the value.
    int32_t sliding_window = args_.sliding_window();
    sliding_window =
        sliding_window - 1;  // Convert to window_size_left (always subtract 1)

    // Get dtype from k_cache
    const auto dtype = k_cache.scalar_type();

    // Determine if causal (prefill mode)
    // const bool causal =
    //     attn_metadata->is_prefill || attn_metadata->is_chunked_prefill;
    constexpr bool causal = false;

    // Determine backend
    // const std::string backend =
    //     causal ? xllm::kernel::cuda::determine_attention_backend(
    //                  /*pos_encoding_mode=*/0,
    //                  /*use_fp16_qk_reduction=*/false,
    //                  /*use_custom_mask=*/false)
    //            : "fa2";
    const static std::string backend = "fa2";

    // Update plan_info
    // Note: plan_info is only updated at layer 0, so we set layer_id to 0
    attn_metadata->plan_info->layer_id = 0;

    layer::flashinfer::update_plan_info(
        attn_metadata->plan_info,
        backend,
        *attn_metadata,
        dtype,                             // query_dtype
        dtype,                             // key_dtype
        dtype,                             // output_dtype
        head_dim,                          // head_dim_qk
        head_dim,                          // head_dim_vo
        static_cast<int32_t>(n_heads),     // num_qo_heads
        static_cast<int32_t>(n_kv_heads),  // num_kv_heads
        static_cast<int32_t>(block_size),  // block_size
        sliding_window,                    // window_size_left
        true,                              // enable_cuda_graph
        causal,                            // causal
        attn_metadata->use_tensor_core);   // use_tensor_core
  }

  // Return ModelInputParams with persistent buffer references if requested
  if (return_capture_params) {
    CHECK_GT(padded_num_tokens, 0)
        << "padded_num_tokens must be > 0 when return_capture_params is true";
    // Set persistent embedding if available
    if (params.input_embedding.defined()) {
      params_for_capture->input_embedding =
          persistent_embedding(padded_num_tokens);
    }
    params_for_capture->attn_metadata = attn_metadata;
    return params_for_capture;
  }

  return std::nullopt;
}

void CudaGraph::initialize_capture_stream(c10::DeviceIndex device_index) {
  // Get a secondary stream from high-priority pool for graph capture.
  // This is required because CUDA graphs must be captured on a non-default
  // stream. Use xllm's Device interface to get stream, but we need high
  // priority, so we use the underlying CUDA API directly (which is what Stream
  // uses internally) Note: Stream class doesn't expose a getter for CUDAStream,
  // and Device::get_stream_from_pool doesn't support high priority, so we use
  // the underlying API that Stream uses internally.
  capture_stream_ = c10::cuda::getStreamFromPool(true, device_index);
  device_index_ = device_index;
  LOG(INFO) << "Initialized capture_stream: " << capture_stream_.value()
            << ", device_index: " << device_index;
}

// CudaGraph implementation
bool CudaGraph::capture(CausalLM* model,
                        const ModelArgs& args,
                        const runtime::Options& options,
                        const torch::Tensor& tokens,
                        const torch::Tensor& positions,
                        const ModelInputParams& params,
                        std::vector<KVCache>& kv_cache,
                        uint32_t bucket_num_tokens,
                        const decltype(at::cuda::graph_pool_handle())& pool) {
  padded_num_tokens_ = bucket_num_tokens;
  const uint32_t actual_num_tokens = tokens.size(0);
  CHECK_GE(padded_num_tokens_, actual_num_tokens)
      << "bucket_num_tokens >= actual_num_tokens";

  // auto& tensor_options = model->options();

  // Update persistent parameters with input data before capture
  const torch::Tensor& k_cache = kv_cache[0].get_k_cache();
  const torch::Tensor& v_cache = kv_cache[0].get_v_cache();
  auto graph_params_opt =
      persistent_param_.update(tokens,
                               k_cache,
                               v_cache,
                               positions,
                               params,
                               padded_num_tokens_,
                               /*return_capture_params=*/true);

  // Use the returned ModelInputParams for graph capture
  CHECK(graph_params_opt.has_value())
      << "update() should return ModelInputParams when "
         "return_capture_params=true";

  LOG(INFO) << "CUDA graph capture begin, bucket_num_tokens: "
            << bucket_num_tokens
            << ", actual_num_tokens: " << actual_num_tokens;

  // Use cached capture stream for graph capture
  // capture_stream_ is initialized in constructor
  bool need_restore_stream = false;

  // Check if current stream is default stream, if so switch to capture stream
  if (c10::cuda::getCurrentCUDAStream(device_index_) ==
      c10::cuda::getDefaultCUDAStream(device_index_)) {
    c10::cuda::getCurrentCUDAStream(device_index_).synchronize();
    c10::cuda::setCurrentCUDAStream(capture_stream_.value());
    capture_stream_.value().synchronize();
    need_restore_stream = true;
  }

  // Begin graph capture (capture_mode defaults to cudaStreamCaptureModeGlobal)
  // Use shared pool passed from executor
  graph_.capture_begin(pool);

  // Execute forward pass - CUDA graph will capture this
  auto forward_result =
      model->forward(persistent_param_.persistent_tokens(padded_num_tokens_),
                     persistent_param_.persistent_positions(padded_num_tokens_),
                     kv_cache,
                     graph_params_opt.value());

  // Store result in persistent buffer
  persistent_param_.set_hidden_states(forward_result);

  // End graph capture
  graph_.capture_end();

  // Restore stream if we switched it

  if (need_restore_stream) {
    c10::cuda::setCurrentCUDAStream(
        c10::cuda::getDefaultCUDAStream(device_index_));
  }

  // Synchronize and test replay to verify graph capture
  torch::cuda::synchronize();

  graph_.replay();

  LOG(INFO) << "CUDA graph capture end, bucket_num_tokens: "
            << bucket_num_tokens;
  return true;
}

torch::Tensor CudaGraph::replay(const torch::Tensor& tokens,
                                const torch::Tensor& positions,
                                std::vector<KVCache>& kv_cache,
                                const ModelInputParams& params) {
  const uint32_t actual_num_tokens = tokens.size(0);
  CHECK_LE(actual_num_tokens, padded_num_tokens_)
      << "num_tokens mismatch: expected <= " << padded_num_tokens_ << ", got "
      << actual_num_tokens;

  // Update persistent parameters with new input data
  const torch::Tensor& k_cache = kv_cache[0].get_k_cache();
  const torch::Tensor& v_cache = kv_cache[0].get_v_cache();
  persistent_param_.update(tokens,
                           k_cache,
                           v_cache,
                           positions,
                           params,
                           padded_num_tokens_,
                           /*return_capture_params=*/false);

  // Replay captured graph
  graph_.replay();

  // Return only the actual num_tokens portion of hidden states
  return get_hidden_states(actual_num_tokens);
}

// CudaGraphExecutorImpl implementation
CudaGraphExecutorImpl::CudaGraphExecutorImpl(CausalLM* model,
                                             const ModelArgs& args,
                                             const torch::Device& device,
                                             const runtime::Options& options)
    : model_(model),
      args_(args),
      device_(device),
      options_(options),
      graph_pool_(at::cuda::graph_pool_handle()) {
  // Create single persistent parameter object shared by all CudaGraph instances
  persistent_param_ =
      std::make_unique<CudaGraphPersistentParam>(args_, device_, options_);
}

ForwardInput CudaGraphExecutorImpl::prepare_inputs(Batch& batch) {
  // Prepare inputs for workers
  return batch.prepare_forward_input(options_.num_decoding_tokens(), 0, args_);
}

torch::Tensor CudaGraphExecutorImpl::run(const torch::Tensor& tokens,
                                         const torch::Tensor& positions,
                                         std::vector<KVCache>& kv_caches,
                                         const ModelInputParams& params) {
  // Only use CUDA graph in decode phase for performance optimization
  // Identify decode phase using q_max_seq_len for precise detection
  // Decode phase: all sequences have q_seq_len == 1 (generating one token at a
  // time) Prefill phase: sequences have q_seq_len > 1 (processing multiple
  // prompt tokens) We also check empty_kv_cache to ensure KV cache is not empty
  // (not first forward pass)
  const bool in_decoding_phase = params.batch_forward_type.is_decode();

  // If not in decode phase, use eager mode directly without CUDA graph
  if (!in_decoding_phase) {
    VLOG(kGraphExecutorLogVerboseLevel)
        << "CudaGraphExecutorImpl::run() in eager mode";
    COUNTER_INC(num_model_execution_total_eager);
    return model_->forward(tokens, positions, kv_caches, params);
  }

  // Get actual num_tokens from tokens shape
  const uint32_t n_tokens = tokens.size(/*dim=*/0);
  const uint32_t bucket_num_tokens = get_bucket_num_tokens(n_tokens);

  // Check if conditions are suitable for graph execution (replay or capture)
  const auto max_seq_len = FLAGS_max_seq_len_for_graph_mode > 0
                               ? FLAGS_max_seq_len_for_graph_mode
                               : args_.max_position_embeddings();
  const bool seq_len_supported = params.kv_max_seq_len <= max_seq_len;

  // Combined condition for graph capture support
  const bool capture_supported = seq_len_supported;

  // Early return if conditions are not suitable for graph operations
  if (!capture_supported) {
    LOG(WARNING) << "Not suitable for CUDA graph operations, falling back to "
                    "eager mode.";
    COUNTER_INC(num_model_execution_total_eager);
    return model_->forward(tokens, positions, kv_caches, params);
  }

  // Check if captured graph exists for this bucket num_tokens
  auto it = graphs_.find(bucket_num_tokens);
  if (it != graphs_.end()) {
    // Replay the existing graph
    VLOG(kGraphExecutorLogVerboseLevel)
        << "CudaGraphExecutorImpl::run() in replay mode";
    return it->second->replay(tokens, positions, kv_caches, params);
  }

  // Graph doesn't exist for this bucket num_tokens, try to create it lazily
  auto graph = std::make_unique<CudaGraph>(*persistent_param_, device_.index());
  VLOG(kGraphExecutorLogVerboseLevel)
      << "CudaGraphExecutorImpl::run() in capture mode";
  bool capture_success = graph->capture(model_,
                                        args_,
                                        options_,
                                        tokens,
                                        positions,
                                        params,
                                        kv_caches,
                                        bucket_num_tokens,
                                        graph_pool_);

  if (capture_success) {
    LOG(INFO) << "Lazy capturing CUDA graph for bucket num_tokens: "
              << bucket_num_tokens << " (actual num_tokens: " << n_tokens
              << ") done";

    // Save the graph for future reuse
    graphs_[bucket_num_tokens] = std::move(graph);

    // Return the output from capture (no need to replay since capture
    // already executed)
    return graphs_[bucket_num_tokens]->get_hidden_states(n_tokens);
  }

  // Fallback to eager mode if capture fails
  LOG(ERROR) << "Failed to capture CUDA graph for bucket num_tokens: "
             << bucket_num_tokens;
  COUNTER_INC(num_model_execution_total_eager);
  return model_->forward(tokens, positions, kv_caches, params);
}

// bucket will be [1, 2, 4, 8, 16, 32, 48, 64, ..., max_seqs_per_batch]
uint32_t CudaGraphExecutorImpl::get_bucket_num_tokens(
    uint32_t num_tokens) const {
  if (FLAGS_enable_graph_no_padding) {
    return num_tokens;
  }
  if (num_tokens <= 1) {
    return 1;
  } else if (num_tokens <= 2) {
    return 2;
  } else if (num_tokens <= 4) {
    return 4;
  } else if (num_tokens <= 8) {
    return 8;
  } else {
    // For num_tokens > 8, use multiples of 16
    return ((num_tokens + 15) / 16) * 16;
  }
}

}  // namespace xllm

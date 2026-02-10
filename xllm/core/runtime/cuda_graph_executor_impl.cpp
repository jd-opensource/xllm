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
#include <c10/cuda/CUDAGuard.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <numeric>
#include <shared_mutex>

#include "core/common/global_flags.h"
#include "core/common/metrics.h"
#include "core/common/rec_model_utils.h"
#include "core/layers/common/attention_metadata.h"
#include "core/layers/common/attention_metadata_builder.h"
#include "core/layers/cuda/flashinfer_planinfo.h"
#include "core/platform/cuda/device_capture_lock.h"
#include "core/platform/device.h"
#include "core/platform/stream.h"
#include "core/util/utils.h"
#include "kernels/cuda/utils.h"

namespace xllm::runtime::cuda {

// CudaGraphPersistentParam implementation
CudaGraphPersistentParam::CudaGraphPersistentParam(
    const ModelArgs& args,
    const torch::Device& device,
    const runtime::Options& options)
    : args_(args), device_(device), options_(options) {
  // Use max_tokens_per_batch for first dimension size
  const int64_t max_tokens_per_batch = FLAGS_max_tokens_per_batch;
  // num_sequences
  int64_t max_seqs_per_batch;
  if (is_rec_multi_round_mode()) {
    // max_seqs_per_batch is the max sequence count per Batch in a scheduler
    // group.
    // When is_rec_multi_round_mode() == true, multiply by beam_width to account
    // for beam search.
    max_seqs_per_batch = options.max_seqs_per_batch() * options_.beam_width();
  } else {
    max_seqs_per_batch = options.max_seqs_per_batch();
  }
  auto tensor_options = torch::TensorOptions().device(device);

  const int64_t max_seq_len = args_.max_position_embeddings();

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
  // aux_hidden_states will be lazily initialized when needed
}

void CudaGraphPersistentParam::set_aux_hidden_states(
    const torch::Tensor& value) {
  if (!value.defined()) {
    return;
  }
  const uint32_t result_tokens = value.size(0);
  if (aux_hidden_states_.numel() == 0) {
    // Lazy initialization: create aux_hidden_states tensor if not already
    // created
    const int64_t max_tokens_per_batch = FLAGS_max_tokens_per_batch;
    auto shape = value.sizes().vec();
    shape[0] = max_tokens_per_batch;
    torch::ScalarType dtype = util::parse_dtype(args_.dtype(), device_);
    if (args_.dtype() == "float" || args_.dtype() == "float32") {
      dtype = torch::kFloat32;
    }
    aux_hidden_states_ =
        torch::zeros(shape, torch::dtype(dtype).device(device_));
  }
  // Slice to match the actual shape
  auto slice =
      aux_hidden_states_.slice(/*dim=*/0, /*start=*/0, /*end=*/result_tokens);
  // Reshape slice if needed to match value shape
  if (slice.sizes() == value.sizes()) {
    slice.copy_(value, /*non_blocking=*/true);
  }
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
  CHECK(attn_metadata) << "attn_metadata should not be null";
  attn_metadata->enable_cuda_graph = true;

  const uint32_t actual_num_tokens = tokens.size(0);
  const int64_t actual_batch_size = params.num_sequences;

  // Copy data from input parameters to persistent graph tensors
  VLOG(kGraphExecutorLogVerboseLevel)
      << "copy_ tokens: src shape=" << tokens.sizes() << ", dst slice shape=["
      << actual_num_tokens << "]";
  persistent_tokens_.slice(/*dim=*/0, /*start=*/0, /*end=*/actual_num_tokens)
      .copy_(tokens, /*non_blocking=*/true);

  // Zero out padding region for tokens to avoid stale data
  // This is needed for both capture and replay when using padded tensors
  if (padded_num_tokens > actual_num_tokens) {
    VLOG(kGraphExecutorLogVerboseLevel)
        << "fill_ tokens padding: [" << actual_num_tokens << ", "
        << padded_num_tokens << "] with 0";
    persistent_tokens_
        .slice(
            /*dim=*/0, /*start=*/actual_num_tokens, /*end=*/padded_num_tokens)
        .fill_(0);
  }

  VLOG(kGraphExecutorLogVerboseLevel)
      << "copy_ positions: src shape=" << positions.sizes()
      << ", dst slice shape=[" << actual_num_tokens << "]";
  persistent_positions_.slice(/*dim=*/0, /*start=*/0, /*end=*/actual_num_tokens)
      .copy_(positions, /*non_blocking=*/true);

  if (!is_rec_multi_round_mode()) {
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
    if (padded_num_tokens > actual_num_tokens) {
      persistent_new_cache_slots_
          .slice(/*dim=*/0,
                 /*start=*/actual_num_tokens,
                 /*end=*/padded_num_tokens)
          .fill_(0);
    }

    // Keep metadata tensors pointing to persistent buffers used by graph
    // capture/replay so their addresses are stable and shapes match padded
    // tensors in capture path.
    attn_metadata->q_cu_seq_lens = q_seq_lens(/*actual_batch_size=*/
                                              actual_batch_size + 1);
    attn_metadata->kv_cu_seq_lens = kv_seq_lens(/*actual_batch_size=*/
                                                actual_batch_size + 1);
    const uint32_t slot_mapping_tokens =
        padded_num_tokens > 0 ? padded_num_tokens : actual_num_tokens;
    attn_metadata->slot_mapping =
        persistent_new_cache_slots(slot_mapping_tokens);
  }

  // Copy block table data
  const int64_t actual_block_table_len = params.block_tables.size(1);
  torch::Tensor slice_persistent_block_tables =
      persistent_block_tables_
          .slice(/*dim=*/0, /*start=*/0, /*end=*/actual_batch_size)
          .slice(/*dim=*/1, /*start=*/0, /*end=*/actual_block_table_len);

  VLOG(kGraphExecutorLogVerboseLevel)
      << "copy_ block_tables: src shape=" << params.block_tables.sizes()
      << ", dst slice shape=" << slice_persistent_block_tables.sizes();
  slice_persistent_block_tables.copy_(params.block_tables,
                                      /*non_blocking=*/true);
  if (!attn_metadata->is_prefill || FLAGS_enable_mla) {
    attn_metadata->block_table = slice_persistent_block_tables;
  }

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
  // Match FlashInfer's CUDAGraph wrapper behavior: always pass the full
  // pre-allocated indices buffer and use indptr to delimit valid range.
  // This keeps kernel arguments stable across replays.
  attn_metadata->paged_kv_indices = persistent_paged_kv_indices_;
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
    const bool causal =
        attn_metadata->is_prefill || attn_metadata->is_chunked_prefill;

    // Determine backend
    const std::string backend = xllm::kernel::cuda::determine_attention_backend(
        /*pos_encoding_mode=*/0,
        /*use_fp16_qk_reduction=*/false,
        /*use_custom_mask=*/false,
        /*causal=*/causal);

    // Update plan_info
    // Note: plan_info is only updated at layer 0, so we set layer_id to 0
    attn_metadata->plan_info->layer_id = 0;
    CHECK_EQ(dtype, torch::ScalarType::BFloat16)
        << "only support bf16 kvcache for now";
    bool use_tensor_core =
        xllm::kernel::cuda::should_use_tensor_core(dtype, n_heads, n_kv_heads);

    VLOG(kGraphExecutorLogVerboseLevel)
        << "CudaGraphPersistentParam::update() calling update_plan_info: "
        << "is_prefill=" << attn_metadata->is_prefill
        << ", is_chunked_prefill=" << attn_metadata->is_chunked_prefill
        << ", causal=" << causal << ", backend=" << backend
        << ", enable_cuda_graph=" << attn_metadata->enable_cuda_graph;

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
        /*enable_cuda_graph=*/true,
        causal,          // causal
        use_tensor_core  // use_tensor_core for decode planning
    );

    VLOG(kGraphExecutorLogVerboseLevel)
        << "CudaGraphPersistentParam::update() plan_info updated: uri="
        << attn_metadata->plan_info->uri << ", plan_info.defined="
        << attn_metadata->plan_info->plan_info.defined() << ", plan_info.size="
        << (attn_metadata->plan_info->plan_info.defined()
                ? attn_metadata->plan_info->plan_info.size()
                : 0);
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

// CudaGraph implementation
bool CudaGraph::capture(CausalLM* model,
                        const ModelArgs& args,
                        const runtime::Options& options,
                        const torch::Tensor& tokens,
                        const torch::Tensor& positions,
                        const ModelInputParams& params,
                        std::vector<KVCache>& kv_cache,
                        uint32_t bucket_num_tokens,
                        const at::cuda::MempoolId_t& pool) {
  padded_num_tokens_ = bucket_num_tokens;
  const uint32_t actual_num_tokens = tokens.size(0);
  CHECK_GE(padded_num_tokens_, actual_num_tokens)
      << "bucket_num_tokens >= actual_num_tokens";

  // Guard CUDA graph capture region with a device-level exclusive lock to
  // prevent conflicting GPU work from other streams (e.g., prepare streams) on
  // the same device when using cudaStreamCaptureModeGlobal. Capture requires
  // exclusive access, so we use write lock.
  std::optional<std::unique_lock<std::shared_mutex>> capture_lock_guard;
  if (FLAGS_enable_graph) {
    auto& capture_lock =
        ::xllm::cuda::DeviceCaptureLock::get_instance().get_write_lock(
            device_index_);
    capture_lock_guard.emplace(capture_lock);
  }
  // Use the returned ModelInputParams for graph capture
  // Always use capture stream for plan/update + capture + forward.
  at::cuda::CUDAStream original_stream =
      at::cuda::getCurrentCUDAStream(device_index_);
  at::cuda::CUDAStream capture_stream = capture_stream_;
  if (original_stream != capture_stream) {
    original_stream.synchronize();
    capture_stream.synchronize();
  }
  std::optional<at::cuda::CUDAStreamGuard> stream_guard;
  stream_guard.emplace(capture_stream);

  // auto& tensor_options = model->options();

  // Update persistent parameters with input data before capture (includes
  // FlashInfer plan/update).
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
            << bucket_num_tokens << ", actual_num_tokens: " << actual_num_tokens
            << ", is_piecewise: " << is_piecewise_;

  if (is_piecewise_) {
    // Piecewise capture mode (for prefill)
    // Warmup: execute forward once without capture to initialize cuBLAS handles
    // and other CUDA resources. This is necessary because these resources
    // cannot be created during CUDA graph capture mode.
    model->forward(persistent_param_.persistent_tokens(padded_num_tokens_),
                   persistent_param_.persistent_positions(padded_num_tokens_),
                   kv_cache,
                   graph_params_opt.value());

    // Begin piecewise capture via GlobalCaptureInstance
    GlobalCaptureInstance::get_instance().begin_capture(pool);

    // Execute forward pass - attention operations will be captured separately
    auto forward_result = model->forward(
        persistent_param_.persistent_tokens(padded_num_tokens_),
        persistent_param_.persistent_positions(padded_num_tokens_),
        kv_cache,
        graph_params_opt.value());

    // Store result in persistent buffer
    persistent_param_.set_hidden_states(forward_result.hidden_states);
    // Only capture aux_hidden_states when enable_graph_aux_hidden_states is on
    // (e.g. main worker in EAGLE-3); draft worker has this option false.
    if (options.enable_graph_aux_hidden_states() &&
        forward_result.aux_hidden_states.defined()) {
      persistent_param_.set_aux_hidden_states(forward_result.aux_hidden_states);
    }
    VLOG(kGraphExecutorLogVerboseLevel)
        << "Piecewise capture forward_result shape: "
        << forward_result.hidden_states.sizes();

    // End capture and get piecewise graphs
    auto piecewise_graphs = GlobalCaptureInstance::get_instance().end_capture();

    if (!piecewise_graphs || piecewise_graphs->empty()) {
      LOG(WARNING) << "Failed to capture piecewise graph: no graphs captured";
      return false;
    }

    // Move piecewise graphs to member
    piecewise_graph_ = std::move(*piecewise_graphs);

    LOG(INFO) << "Piecewise graph capture end, bucket_num_tokens: "
              << bucket_num_tokens
              << ", num_graphs: " << piecewise_graph_.size()
              << ", num_runners: " << piecewise_graph_.num_runners();
  } else {
    // Normal capture mode (for decode)
    // Begin graph capture (capture_mode defaults to
    // cudaStreamCaptureModeGlobal)
    // graph_.capture_begin(pool);
    graph_.capture_begin(pool, cudaStreamCaptureModeThreadLocal);

    // Execute forward pass - CUDA graph will capture this
    auto forward_result = model->forward(
        persistent_param_.persistent_tokens(padded_num_tokens_),
        persistent_param_.persistent_positions(padded_num_tokens_),
        kv_cache,
        graph_params_opt.value());

    // Store result in persistent buffer
    persistent_param_.set_hidden_states(forward_result.hidden_states);
    if (options.enable_graph_aux_hidden_states() &&
        forward_result.aux_hidden_states.defined()) {
      persistent_param_.set_aux_hidden_states(forward_result.aux_hidden_states);
    }

    // End graph capture
    graph_.capture_end();
  }

  // Synchronize to ensure graph capture is completed.
  capture_stream.synchronize();

  // Explicitly restore stream after capture before replay logic.
  stream_guard.reset();

  // Replay is unified in CudaGraphExecutorImpl::run() after capture success
  // for both prefill and decode.

  LOG(INFO) << "CUDA graph capture end, bucket_num_tokens: "
            << bucket_num_tokens;
  return true;
}

ModelOutput CudaGraph::replay(const torch::Tensor& tokens,
                              const torch::Tensor& positions,
                              std::vector<KVCache>& kv_cache,
                              const ModelInputParams& params) {
  const uint32_t actual_num_tokens = tokens.size(0);
  CHECK_LE(actual_num_tokens, padded_num_tokens_)
      << "num_tokens mismatch: expected <= " << padded_num_tokens_ << ", got "
      << actual_num_tokens;

  // Guard CUDA graph replay with a device-level shared lock to allow multiple
  // replay operations to run concurrently while preventing conflicts with
  // capture operations. Replay can share the lock with other replay/prepare
  // operations.
  std::optional<std::shared_lock<std::shared_mutex>> replay_lock_guard;
  if (FLAGS_enable_graph) {
    auto& replay_lock =
        ::xllm::cuda::DeviceCaptureLock::get_instance().get_read_lock(
            device_index_);
    replay_lock_guard.emplace(replay_lock);
  }

  // Update persistent parameters with new input data
  const torch::Tensor& k_cache = kv_cache[0].get_k_cache();
  const torch::Tensor& v_cache = kv_cache[0].get_v_cache();

  if (is_piecewise_) {
    // Piecewise replay mode (for prefill)
    // Need to get updated params with attn_metadata for attention replay
    auto updated_params_opt =
        persistent_param_.update(tokens,
                                 k_cache,
                                 v_cache,
                                 positions,
                                 params,
                                 padded_num_tokens_,
                                 /*return_capture_params=*/true);
    CHECK(updated_params_opt.has_value())
        << "update() should return ModelInputParams for piecewise replay";

    const auto& updated_params = updated_params_opt.value();
    CHECK(piecewise_graph_.num_runners() > 0)
        << "Piecewise graph must have attention runners";
    CHECK(updated_params.attn_metadata)
        << "attn_metadata is required for piecewise replay";
    CHECK(updated_params.attn_metadata->plan_info)
        << "plan_info is required for piecewise replay";

    VLOG(kGraphExecutorLogVerboseLevel)
        << "CudaGraph::replay() piecewise replay with uri="
        << updated_params.attn_metadata->plan_info->uri
        << ", plan_info.defined="
        << updated_params.attn_metadata->plan_info->plan_info.defined();

    // Build AttentionReplayParams from updated attn_metadata
    ::xllm::kernel::cuda::AttentionReplayParams replay_params;
    replay_params.actual_num_tokens = actual_num_tokens;
    replay_params.plan_info =
        updated_params.attn_metadata->plan_info->plan_info;
    replay_params.q_cu_seq_lens = updated_params.attn_metadata->q_cu_seq_lens;
    replay_params.kv_cu_seq_lens = updated_params.attn_metadata->kv_cu_seq_lens;

    // Replay piecewise graphs and attention runners
    piecewise_graph_.replay(replay_params);
  } else {
    // Normal replay mode (for decode)
    persistent_param_.update(tokens,
                             k_cache,
                             v_cache,
                             positions,
                             params,
                             padded_num_tokens_,
                             /*return_capture_params=*/false);
    graph_.replay();
  }

  // Return the actual num_tokens portion of ModelOutput
  // Note: aux_hidden_states handling is done in CudaGraphExecutorImpl::run()
  // since replay() doesn't have access to options
  return ModelOutput(get_hidden_states(actual_num_tokens));
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
      enable_prefill_piecewise_graph_(FLAGS_enable_prefill_piecewise_graph) {
  max_tokens_for_graph_mode_ = FLAGS_max_tokens_for_graph_mode;
  if (max_tokens_for_graph_mode_ < options_.max_seqs_per_batch()) {
    max_tokens_for_graph_mode_ = options_.max_seqs_per_batch();
  }
  // Keep one pool per executor instance so all captured graphs can reuse it,
  // while avoiding cross-instance stale-handle reuse.
  graph_pool_ = at::cuda::graph_pool_handle();
  // Create single persistent parameter object shared by all CudaGraph instances
  persistent_param_ =
      std::make_unique<CudaGraphPersistentParam>(args_, device_, options_);
}

// Static method to get graph memory pool for current thread
// Each thread gets its own graph memory pool, similar to
// GlobalCaptureInstance::get_instance()
at::cuda::MempoolId_t CudaGraphExecutorImpl::get_mem_pool() {
  // Use thread_local to ensure each thread has its own graph pool
  // This follows the same pattern as GlobalCaptureInstance::get_instance()
  // but provides per-thread instances instead of a global singleton
  thread_local at::cuda::MempoolId_t thread_graph_pool =
      at::cuda::graph_pool_handle();

  // Thread-local counter to log initialization only once per thread
  thread_local bool initialized = false;
  if (!initialized) {
    LOG(INFO) << "Initialized graph_pool for thread: "
              << std::this_thread::get_id();
    initialized = true;
  }

  return thread_graph_pool;
}

// Static method to get CUDA capture stream for current thread
// Each thread gets its own high-priority capture stream
c10::cuda::CUDAStream CudaGraphExecutorImpl::get_capture_stream(
    c10::DeviceIndex device_index) {
  // Use thread_local to ensure each thread has its own capture stream
  // This is required because CUDA graphs must be captured on a non-default
  // stream. We use high-priority streams for better performance.
  thread_local c10::cuda::CUDAStream thread_capture_stream =
      c10::cuda::getStreamFromPool(/*isHighPriority=*/true, device_index);

  // Thread-local counter to log initialization only once per thread
  thread_local bool initialized = false;
  if (!initialized) {
    LOG(INFO) << "Initialized capture_stream for thread: "
              << std::this_thread::get_id()
              << ", stream: " << thread_capture_stream
              << ", device_index: " << device_index;
    initialized = true;
  }

  return thread_capture_stream;
}

ForwardInput CudaGraphExecutorImpl::prepare_inputs(Batch& batch) {
  // Prepare inputs for workers
  return batch.prepare_forward_input(options_.num_decoding_tokens(), 0, args_);
}

ModelOutput CudaGraphExecutorImpl::attach_aux_hidden_states_if_needed(
    const torch::Tensor& hidden_states,
    uint32_t n_tokens) const {
  if (options_.enable_graph_aux_hidden_states()) {
    auto aux_hidden_states = persistent_param_->aux_hidden_states(n_tokens);
    if (aux_hidden_states.defined() && aux_hidden_states.numel() > 0) {
      return ModelOutput(hidden_states, torch::Tensor(), aux_hidden_states);
    }
  }
  return ModelOutput(hidden_states);
}

ModelOutput CudaGraphExecutorImpl::run(const torch::Tensor& tokens,
                                       const torch::Tensor& positions,
                                       std::vector<KVCache>& kv_caches,
                                       const ModelInputParams& params) {
  const bool is_prefill = params.batch_forward_type.is_prefill();
  const bool is_decode = params.batch_forward_type.is_decode();

  // Get actual num_tokens from tokens shape
  const uint32_t n_tokens = tokens.size(/*dim=*/0);
  const uint32_t bucket_num_tokens =
      get_bucket_num_tokens(n_tokens, is_prefill);

  // Prefill phase with piecewise graph
  if (is_prefill && enable_prefill_piecewise_graph_) {
    // Check if token count is within limit
    const bool graph_mode_supported = n_tokens <= max_tokens_for_graph_mode_;

    if (!graph_mode_supported) {
      VLOG(kGraphExecutorLogVerboseLevel)
          << "Token count " << n_tokens
          << " exceeds max_tokens_for_graph_mode ("
          << max_tokens_for_graph_mode_ << "), falling back to eager mode";
      COUNTER_INC(num_model_execution_total_eager);
      return model_->forward(tokens, positions, kv_caches, params);
    }

    // Check if piecewise graph exists for this bucket
    auto it = prefill_graphs_.find(bucket_num_tokens);
    if (it != prefill_graphs_.end()) {
      // Replay existing piecewise graph
      VLOG(kGraphExecutorLogVerboseLevel)
          << "CudaGraphExecutorImpl::run() in prefill piecewise replay mode";
      auto result = it->second->replay(tokens, positions, kv_caches, params);
      return attach_aux_hidden_states_if_needed(result.hidden_states, n_tokens);
    }

    // Graph doesn't exist, try to create it lazily with piecewise capture
    auto graph =
        std::make_unique<CudaGraph>(*persistent_param_,
                                    device_.index(),
                                    get_capture_stream(device_.index()),
                                    /*is_piecewise=*/true);
    VLOG(kGraphExecutorLogVerboseLevel)
        << "CudaGraphExecutorImpl::run() in prefill piecewise capture mode";
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
      LOG(INFO) << "Lazy capturing piecewise CUDA graph for bucket num_tokens: "
                << bucket_num_tokens << " (actual num_tokens: " << n_tokens
                << ") done";

      // Save the graph for future reuse
      prefill_graphs_[bucket_num_tokens] = std::move(graph);

      // Run replay after capture so first request uses same execution path as
      // subsequent requests.
      auto result = prefill_graphs_[bucket_num_tokens]->replay(
          tokens, positions, kv_caches, params);
      return attach_aux_hidden_states_if_needed(result.hidden_states, n_tokens);
    }

    // Fallback to eager mode if capture fails
    LOG(WARNING)
        << "Failed to capture piecewise graph, falling back to eager mode";
    COUNTER_INC(num_model_execution_total_eager);
    return model_->forward(tokens, positions, kv_caches, params);
  }

  // Prefill without piecewise graph: use eager mode
  if (is_prefill) {
    COUNTER_INC(num_model_execution_total_eager);
    return model_->forward(tokens, positions, kv_caches, params);
  }

  // Decode phase with full graph
  if (is_decode) {
    // Check if conditions are suitable for graph execution (replay or capture)
    const auto max_seq_len = args_.max_position_embeddings();
    const bool seq_len_supported = params.kv_max_seq_len <= max_seq_len;

    // Early return if conditions are not suitable for graph operations
    if (!seq_len_supported) {
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
          << "CudaGraphExecutorImpl::run() in decode replay mode";
      auto result = it->second->replay(tokens, positions, kv_caches, params);
      return attach_aux_hidden_states_if_needed(result.hidden_states, n_tokens);
    }

    // Graph doesn't exist for this bucket num_tokens, try to create it lazily
    auto graph =
        std::make_unique<CudaGraph>(*persistent_param_,
                                    device_.index(),
                                    get_capture_stream(device_.index()));
    VLOG(kGraphExecutorLogVerboseLevel)
        << "CudaGraphExecutorImpl::run() in decode capture mode";
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

      // Run replay after capture so first request uses same execution path as
      // subsequent requests.
      auto result = graphs_[bucket_num_tokens]->replay(
          tokens, positions, kv_caches, params);
      return attach_aux_hidden_states_if_needed(result.hidden_states, n_tokens);
    }

    // Fallback to eager mode if capture fails
    LOG(ERROR) << "Failed to capture CUDA graph for bucket num_tokens: "
               << bucket_num_tokens;
    COUNTER_INC(num_model_execution_total_eager);
    return model_->forward(tokens, positions, kv_caches, params);
  }
  // Fallback to eager mode if capture fails
  LOG(ERROR) << "Failed to capture CUDA graph for bucket num_tokens: "
             << bucket_num_tokens;
  COUNTER_INC(num_model_execution_total_eager);
  return model_->forward(tokens, positions, kv_caches, params);
}

// bucket will be [1, 2, 4, 8, 16, 32, 48, 64, ..., max_seqs_per_batch]
uint32_t CudaGraphExecutorImpl::get_bucket_num_tokens(uint32_t num_tokens,
                                                      bool is_prefill) const {
  // no_padding only works for decode, prefill requires padding for graph reuse
  if (FLAGS_enable_graph_mode_decode_no_padding && !is_prefill) {
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

}  // namespace xllm::runtime::cuda

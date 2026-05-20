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

#include <algorithm>
#include <numeric>

#include "core/common/global_flags.h"
#include "core/framework/config/execution_config.h"
#include "core/framework/config/speculative_config.h"
#ifdef TORCH_HIGHER_THAN_PTA6
#include <torch_npu/csrc/framework/OpCommand.h>
#else
#include <torch_npu/csrc/aten/NPUNativeFunctions.h>
#include <torch_npu/csrc/framework/utils/OpPreparation.h>
#endif
#include "core/common/global_flags.h"
#include "core/common/metrics.h"
#include "core/util/utils.h"
#include "platform/npu/device_capture_lock.h"

// ATB includes
#include <atb/atb_infer.h>
#include <atb/context.h>
#include <atb/operation.h>
#include <customize/custom_paged_attention_function.h>
#include <customize/customize_op_params.h>

#include "pytorch/adapter/utils/utils.h"

namespace xllm::npu {

namespace {
constexpr int32_t kPaddingSeqLen = 0;
constexpr int32_t kPaddingLinearStateId = 0;

std::pair<torch::Tensor, torch::Tensor> find_attention_plan_kv_cache(
    const std::vector<KVCache>& kv_caches) {
  for (const auto& cache : kv_caches) {
    auto k_cache = cache.get_k_cache();
    auto v_cache = cache.get_v_cache();
    if (k_cache.defined() && v_cache.defined() && k_cache.numel() > 0 &&
        v_cache.numel() > 0) {
      return {std::move(k_cache), std::move(v_cache)};
    }
  }
  return {torch::Tensor(), torch::Tensor()};
}

int64_t get_decode_graph_capacity(const runtime::Options& options) {
  CHECK_GT(options.num_decoding_tokens(), 0)
      << "num_decoding_tokens must be > 0 for graph capacity";
  if (::xllm::SpeculativeConfig::get_instance().enable_atb_spec_kernel()) {
    return options.max_seqs_per_batch();
  }
  if (options.enable_speculative_decode() && !options.is_draft_engine()) {
    return options.max_seqs_per_batch() * options.num_decoding_tokens();
  }
  return options.max_seqs_per_batch();
}

int64_t infer_actual_batch_size(const ModelInputParams& params) {
  if (params.meta.num_sequences > 0) {
    return params.meta.num_sequences;
  }
  if (!params.attention.host.kv_seq_lens.empty()) {
    return static_cast<int64_t>(params.attention.host.kv_seq_lens.size());
  }
  if (!params.attention.host.q_seq_lens.empty()) {
    return static_cast<int64_t>(params.attention.host.q_seq_lens.size());
  }
  if (params.attention.device.kv_seq_lens.defined() &&
      params.attention.device.kv_seq_lens.dim() >= 1) {
    return params.attention.device.kv_seq_lens.size(0);
  }
  if (params.attention.device.q_seq_lens.defined() &&
      params.attention.device.q_seq_lens.dim() >= 1) {
    return params.attention.device.q_seq_lens.size(0);
  }
  if (params.attention.device.block_tables.defined() &&
      params.attention.device.block_tables.dim() >= 2) {
    return params.attention.device.block_tables.size(0);
  }
  for (const auto& block_table : params.multi_block_tables) {
    if (block_table.defined() && block_table.dim() >= 2) {
      return block_table.size(0);
    }
  }
  return 0;
}

}  // namespace

// GraphPersistentParam implementation
GraphPersistentParam::GraphPersistentParam(const ModelArgs& args,
                                           const torch::Device& device,
                                           const runtime::Options& options,
                                           bool need_update_attn_mask)
    : args_(args),
      device_(device),
      options_(options),
      context_for_plan_(nullptr),
      custom_pa_op_for_plan_(nullptr),
      stream_for_plan_(nullptr),
      need_update_attn_mask_(need_update_attn_mask) {
  // Determine whether attention plan needs to be updated based on model type
  // Future logic can be extended here for more complex model-specific behavior
  need_update_attention_plan_ = (args.model_type() != "deepseek_v32" &&
                                 args.model_type() != "deepseek_v4" &&
                                 args.model_type() != "glm_moe_dsa");

  // Check if mRoPE is used (for VLM models like qwen2-vl)
  use_mrope_ = !args.rope_scaling_mrope_section().empty();

  const int64_t max_tokens_per_batch = options.max_tokens_per_batch();
  // Graph-mode token capacity is narrower than max_tokens_per_batch: ACL graph
  // only serves decode / spec-verify batches, so the relevant row upper bound
  // comes from decode graph capacity instead.
  const int64_t max_graph_tokens = get_decode_graph_capacity(options);
  // num_sequences
  const int64_t max_seqs_per_batch = get_decode_graph_capacity(options);
  auto tensor_options = torch::TensorOptions().device(device);

  const int64_t max_seq_len = args_.max_position_embeddings();

  // Create persistent tensors with max_tokens_per_batch as first dimension
  persistent_tokens_ = torch::zeros({max_tokens_per_batch},
                                    torch::dtype(torch::kInt).device(device));
  if (args.rope_scaling_mrope_section().empty()) {
    persistent_positions_ = torch::zeros(
        {max_tokens_per_batch}, torch::dtype(torch::kInt).device(device));
  } else {
    persistent_positions_ = torch::zeros(
        {3, max_tokens_per_batch}, torch::dtype(torch::kInt).device(device));
    use_mrope_ = true;
  }
  persistent_new_cache_slots_ = torch::zeros(
      {max_tokens_per_batch}, torch::dtype(torch::kInt).device(device));
  persistent_linear_state_indices_ = torch::zeros(
      {max_seqs_per_batch}, torch::dtype(torch::kInt).device(device));

  // Sequence length tensors with max_seqs_per_batch
  q_seq_lens_ = torch::zeros({max_seqs_per_batch},
                             torch::dtype(torch::kInt).device(device));
  kv_seq_lens_ = torch::zeros({max_seqs_per_batch},
                              torch::dtype(torch::kInt).device(device));

  // Block table tensors with maximum possible size
  const auto block_size = options.block_size();
  const int64_t max_block_table_len =
      (max_seq_len + block_size - 1) / block_size + 1;
  persistent_block_tables_ =
      torch::zeros({max_seqs_per_batch, max_block_table_len},
                   torch::dtype(torch::kInt).device(device));

  // Output tensor for hidden states
  torch::Dtype dtype = util::parse_dtype(args.dtype(), device);
  if (args.dtype() == "float" || args.dtype() == "float32") {
    LOG(WARNING)
        << "Acl graph executor init hidden_states compatible with float32 "
           "dtype: float32. This should not happen in production but for test.";
    dtype = torch::kFloat32;
  }
  hidden_states_ = torch::zeros({max_tokens_per_batch, args.hidden_size()},
                                torch::dtype(dtype).device(device));

  // Initialize persistent_mask_ only for model types that need to update an
  // explicit attention mask in graph mode. Unlike generic token buffers, the
  // mask is only consumed by decode / spec-verify graphs, so size it by graph
  // token capacity instead of the much larger max_tokens_per_batch prefill
  // budget.
  if (need_update_attn_mask_) {
    persistent_mask_ = torch::zeros({max_graph_tokens, max_seq_len},
                                    torch::dtype(dtype).device(device));
    persistent_mask_zero_template_ = torch::zeros(
        {max_graph_tokens, max_seq_len}, torch::dtype(dtype).device(device));
    const float mask_fill_value = (dtype == torch::kFloat16)
                                      ? -std::numeric_limits<float>::infinity()
                                      : -9984.0f;
    persistent_mask_fill_template_ =
        torch::full({max_graph_tokens, max_seq_len},
                    mask_fill_value,
                    torch::dtype(dtype).device(device));
  }

  q_cu_seq_lens_default_ = torch::zeros(
      {max_seqs_per_batch + 1}, torch::dtype(torch::kInt).device(device));
  q_cu_seq_lens_ = torch::zeros({max_seqs_per_batch + 1},
                                torch::dtype(torch::kInt).device(device));

  // Do not need to create ATB context and custom paged attention operation
  if (args_.head_dim() == 0) {
    return;
  }

  if (!need_update_attention_plan_) {
    return;
  }

  initialize_paged_attention_plan_context(device);
}

GraphPersistentParam::~GraphPersistentParam() {
  if (custom_pa_op_for_plan_ != nullptr) {
    atb::DestroyOperation(custom_pa_op_for_plan_);
    custom_pa_op_for_plan_ = nullptr;
  }
  if (stream_for_plan_ != nullptr) {
    aclrtDestroyStream(stream_for_plan_);
    stream_for_plan_ = nullptr;
  }
  if (context_for_plan_ != nullptr) {
    atb::DestroyContext(context_for_plan_);
    context_for_plan_ = nullptr;
  }
}

void GraphPersistentParam::set_aux_hidden_states(const torch::Tensor& value) {
  if (!value.defined()) {
    return;
  }
  const uint32_t result_tokens = value.size(0);
  if (aux_hidden_states_.numel() == 0) {
    // Lazy initialization: create aux_hidden_states tensor if not already
    // created
    const int64_t max_tokens_per_batch = options_.max_tokens_per_batch();
    auto shape = value.sizes().vec();
    shape[0] = max_tokens_per_batch;
    torch::Dtype dtype = util::parse_dtype(args_.dtype(), device_);
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

namespace {
void zero_tensor_tail(torch::Tensor& tensor,
                      int64_t start,
                      int64_t end,
                      int64_t dim = 0) {
  if (start >= end) {
    return;
  }
  tensor.slice(/*dim=*/dim, /*start=*/start, /*end=*/end).zero_();
}
}  // namespace

std::optional<ModelInputParams> GraphPersistentParam::update(
    const torch::Tensor& tokens,
    const torch::Tensor& k_cache,
    const torch::Tensor& v_cache,
    const torch::Tensor& positions,
    const ModelInputParams& params,
    uint32_t padded_num_tokens,
    bool return_capture_params) {
  CHECK_GT(padded_num_tokens, 0)
      << "padded_num_tokens must be > 0 when return_capture_params is true";
  const uint32_t actual_num_tokens = tokens.size(0);
  int64_t actual_batch_size = infer_actual_batch_size(params);
  if (params.meta.batch_forward_type.is_decode()) {
    const int64_t decode_tokens =
        std::max<int64_t>(options_.num_decoding_tokens(), 1);
    actual_batch_size = actual_num_tokens / decode_tokens;
  }

  // Copy data from input parameters to persistent graph tensors
  if (actual_num_tokens > 0) {
    persistent_tokens_.slice(/*dim=*/0, /*start=*/0, /*end=*/actual_num_tokens)
        .copy_(tokens, /*non_blocking=*/true);
  }
  // mRoPE positions have shape [3, num_tokens], slice on dim 1
  if (actual_num_tokens > 0) {
    if (use_mrope_) {
      persistent_positions_
          .slice(/*dim=*/1, /*start=*/0, /*end=*/actual_num_tokens)
          .copy_(positions, /*non_blocking=*/true);
    } else {
      persistent_positions_
          .slice(/*dim=*/0, /*start=*/0, /*end=*/actual_num_tokens)
          .copy_(positions, /*non_blocking=*/true);
    }
  }
  int64_t q_copy_len = 0;
  if (actual_batch_size > 0 && params.attention.device.q_seq_lens.defined() &&
      params.attention.device.q_seq_lens.dim() >= 1 &&
      params.attention.device.q_seq_lens.numel() > 0) {
    q_copy_len = std::min<int64_t>(actual_batch_size,
                                   params.attention.device.q_seq_lens.size(0));
    if (q_copy_len > 0) {
      q_seq_lens_.slice(/*dim=*/0, /*start=*/0, /*end=*/q_copy_len)
          .copy_(params.attention.device.q_seq_lens.slice(/*dim=*/0,
                                                          /*start=*/0,
                                                          /*end=*/q_copy_len),
                 /*non_blocking=*/true);
    }
  }
  int64_t kv_copy_len = 0;
  if (actual_batch_size > 0 && params.attention.device.kv_seq_lens.defined() &&
      params.attention.device.kv_seq_lens.dim() >= 1 &&
      params.attention.device.kv_seq_lens.numel() > 0) {
    kv_copy_len = std::min<int64_t>(
        actual_batch_size, params.attention.device.kv_seq_lens.size(0));
    if (kv_copy_len > 0) {
      kv_seq_lens_.slice(/*dim=*/0, /*start=*/0, /*end=*/kv_copy_len)
          .copy_(params.attention.device.kv_seq_lens.slice(/*dim=*/0,
                                                           /*start=*/0,
                                                           /*end=*/kv_copy_len),
                 /*non_blocking=*/true);
    }
  }
  // Keep padded decode slots valid for empty/local-short DP shards.
  // These tensors are consumed by ATB setup alongside *_seq_lens_vec.
  const int64_t padded_batch_size = static_cast<int64_t>(padded_num_tokens);
  if (q_copy_len < padded_batch_size) {
    q_seq_lens_
        .slice(/*dim=*/0,
               /*start=*/q_copy_len,
               /*end=*/padded_batch_size)
        .fill_(1);
  }
  if (kv_copy_len < padded_batch_size) {
    kv_seq_lens_
        .slice(/*dim=*/0,
               /*start=*/kv_copy_len,
               /*end=*/padded_batch_size)
        .fill_(1);
  }

  if (actual_num_tokens > 0 &&
      params.attention.device.new_cache_slots.defined() &&
      params.attention.device.new_cache_slots.dim() >= 1 &&
      params.attention.device.new_cache_slots.numel() > 0) {
    const int64_t slot_copy_len =
        std::min<int64_t>(static_cast<int64_t>(actual_num_tokens),
                          params.attention.device.new_cache_slots.size(0));
    if (slot_copy_len > 0) {
      persistent_new_cache_slots_
          .slice(/*dim=*/0, /*start=*/0, /*end=*/slot_copy_len)
          .copy_(params.attention.device.new_cache_slots.slice(
                     /*dim=*/0, /*start=*/0, /*end=*/slot_copy_len),
                 /*non_blocking=*/true);
    }
  }
  if (actual_num_tokens < padded_num_tokens) {
    zero_tensor_tail(persistent_new_cache_slots_,
                     actual_num_tokens,
                     static_cast<int64_t>(padded_num_tokens));
  }
  if (!params.embedding.linear_state_ids.empty()) {
    const int64_t linear_copy_len = std::min<int64_t>(
        actual_batch_size,
        static_cast<int64_t>(params.embedding.linear_state_ids.size()));
    if (linear_copy_len > 0) {
      if (params.embedding.linear_state_indices.defined()) {
        persistent_linear_state_indices_
            .slice(/*dim=*/0, /*start=*/0, /*end=*/linear_copy_len)
            .copy_(params.embedding.linear_state_indices.slice(
                       /*dim=*/0, /*start=*/0, /*end=*/linear_copy_len),
                   /*non_blocking=*/true);
      } else {
        persistent_linear_state_indices_
            .slice(/*dim=*/0, /*start=*/0, /*end=*/linear_copy_len)
            .copy_(torch::tensor(params.embedding.linear_state_ids, torch::kInt)
                       .to(device_)
                       .slice(/*dim=*/0, /*start=*/0, /*end=*/linear_copy_len),
                   /*non_blocking=*/true);
      }
    }
  }

  // Copy only active block table data. Rows/columns observed by the graph are
  // overwritten below or explicitly zeroed in the active bucket slice.
  int64_t block_rows_to_copy = 0;
  int64_t actual_block_table_len = 0;
  if (actual_batch_size > 0 && params.attention.device.block_tables.defined() &&
      params.attention.device.block_tables.dim() >= 2 &&
      params.attention.device.block_tables.numel() > 0) {
    block_rows_to_copy = std::min<int64_t>(
        actual_batch_size, params.attention.device.block_tables.size(0));
    actual_block_table_len = params.attention.device.block_tables.size(1);
    if (block_rows_to_copy > 0 && actual_block_table_len > 0) {
      auto slice_persistent_block_tables =
          persistent_block_tables_
              .slice(/*dim=*/0, /*start=*/0, /*end=*/block_rows_to_copy)
              .slice(/*dim=*/1, /*start=*/0, /*end=*/actual_block_table_len);
      slice_persistent_block_tables.copy_(
          params.attention.device.block_tables.slice(
              /*dim=*/0, /*start=*/0, /*end=*/block_rows_to_copy),
          /*non_blocking=*/true);
    }
  }
  if (block_rows_to_copy > 0 &&
      actual_block_table_len < persistent_block_tables_.size(1)) {
    persistent_block_tables_
        .slice(/*dim=*/0, /*start=*/0, /*end=*/block_rows_to_copy)
        .slice(/*dim=*/1,
               /*start=*/actual_block_table_len,
               /*end=*/persistent_block_tables_.size(1))
        .zero_();
  }
  if (actual_batch_size < padded_batch_size) {
    zero_tensor_tail(
        persistent_block_tables_, actual_batch_size, padded_batch_size);
  }

  // Update persistent embedding from input_embedding if available
  const auto& embedding = params.embedding.input_embedding;
  if (embedding.defined()) {
    const int64_t embedding_tokens = embedding.size(0);

    // Initialize persistent_embedding_ if needed and not already initialized
    if (persistent_embedding_.numel() == 0) {
      const int64_t max_tokens_per_batch = options_.max_tokens_per_batch();
      const int64_t embedding_dim = embedding.size(1);
      torch::Dtype dtype = util::parse_dtype(args_.dtype(), device_);
      persistent_embedding_ =
          torch::zeros({max_tokens_per_batch, embedding_dim},
                       torch::dtype(dtype).device(device_));
    }

    // Copy embedding data to persistent buffer
    persistent_embedding_
        .slice(/*dim=*/0, /*start=*/0, /*end=*/embedding_tokens)
        .copy_(embedding, /*non_blocking=*/true);
  }
  if (q_cu_seq_lens_default_.defined() &&
      q_cu_seq_lens_default_.sizes() == q_cu_seq_lens_.sizes()) {
    q_cu_seq_lens_.copy_(q_cu_seq_lens_default_, /*non_blocking=*/true);
  }
  const bool has_q_cu = params.attention.device.q_cu_seq_lens.defined() &&
                        params.attention.device.q_cu_seq_lens.dim() >= 1;
  const int64_t q_cu_size =
      (has_q_cu && params.attention.device.q_cu_seq_lens.numel() > 0)
          ? params.attention.device.q_cu_seq_lens.size(0)
          : 0;
  const int64_t q_cu_copy_len = std::min<int64_t>(actual_batch_size, q_cu_size);
  if (q_cu_copy_len > 0) {
    q_cu_seq_lens_.slice(/*dim=*/0, /*start=*/0, /*end=*/q_cu_copy_len)
        .copy_(params.attention.device.q_cu_seq_lens.slice(
                   /*dim=*/0, /*start=*/0, /*end=*/q_cu_copy_len),
               /*non_blocking=*/true);
  }
  if (padded_batch_size > q_cu_copy_len) {
    auto tail_q_seq_lens = q_seq_lens_.slice(/*dim=*/0,
                                             /*start=*/q_cu_copy_len,
                                             /*end=*/padded_batch_size);
    auto tail_cu = torch::cumsum(tail_q_seq_lens, /*dim=*/0);
    if (q_cu_copy_len > 0) {
      auto last_prefix = q_cu_seq_lens_.slice(/*dim=*/0,
                                              /*start=*/q_cu_copy_len - 1,
                                              /*end=*/q_cu_copy_len);
      tail_cu = tail_cu + last_prefix;
    }
    q_cu_seq_lens_
        .slice(/*dim=*/0, /*start=*/q_cu_copy_len, /*end=*/padded_batch_size)
        .copy_(tail_cu, /*non_blocking=*/true);
  }

  // Update attention mask only if needed
  if (need_update_attn_mask_) {
    update_attention_mask(params);
  }

  if (uses_paged_attention_tiling()) {
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();

    if (k_cache.defined() && v_cache.defined() && k_cache.numel() > 0 &&
        v_cache.numel() > 0) {
      plan_paged_attention_tiling(
          tokens, k_cache, v_cache, persistent_block_tables_, params, stream);
    }
  }

  // Return ModelInputParams with persistent buffer references if requested
  if (return_capture_params) {
    std::optional<ModelInputParams> params_for_capture =
        std::make_optional<ModelInputParams>(params);
    // Set persistent buffers in params_for_capture
    params_for_capture->attention.device.kv_seq_lens =
        kv_seq_lens(padded_num_tokens);
    params_for_capture->attention.device.q_seq_lens =
        q_seq_lens(padded_num_tokens);
    params_for_capture->meta.actual_num_sequences =
        static_cast<int32_t>(actual_batch_size);
    params_for_capture->attention.host.kv_seq_lens.resize(padded_num_tokens);
    params_for_capture->attention.host.q_seq_lens.resize(padded_num_tokens);
    // Copy actual values from original params
    for (int64_t i = 0; i < actual_batch_size; i++) {
      params_for_capture->attention.host.kv_seq_lens[i] =
          params.attention.host.kv_seq_lens[i];
      params_for_capture->attention.host.q_seq_lens[i] =
          params.attention.host.q_seq_lens[i];
    }
    // Fill padded positions with default values
    for (int64_t i = actual_batch_size; i < padded_num_tokens; i++) {
      params_for_capture->attention.host.kv_seq_lens[i] = kPaddingSeqLen;
      params_for_capture->attention.host.q_seq_lens[i] = kPaddingSeqLen;
    }
    params_for_capture->meta.num_sequences = padded_num_tokens;
    params_for_capture->meta.batch_forward_type = BatchForwardType::DECODE;
    params_for_capture->enable_graph = true;
    if (params_for_capture->parallel.dp_global_token_nums.size() > 1) {
      params_for_capture->parallel.dp_global_token_nums = std::vector<int32_t>(
          params_for_capture->parallel.dp_global_token_nums.size(),
          static_cast<int32_t>(padded_num_tokens));
    }
    params_for_capture->attention.device.new_cache_slots =
        persistent_new_cache_slots(padded_num_tokens);
    params_for_capture->attention.device.block_tables =
        persistent_block_tables(padded_num_tokens);
    if (!params.embedding.linear_state_ids.empty()) {
      params_for_capture->embedding.linear_state_ids =
          params.embedding.linear_state_ids;
      params_for_capture->embedding.linear_state_ids.resize(
          padded_num_tokens, kPaddingLinearStateId);
      params_for_capture->embedding.linear_state_indices =
          persistent_linear_state_indices(padded_num_tokens);
    }

    // Only set attn_mask if need_update_attn_mask_ is true
    if (need_update_attn_mask_) {
      params_for_capture->graph.attn_mask = persistent_mask(padded_num_tokens);
    }
    if (uses_paged_attention_tiling()) {
      params_for_capture->graph.tiling_data = tiling_data();
    } else {
      params_for_capture->graph.tiling_data = torch::Tensor();
    }
    // Set persistent embedding if available
    if (params.embedding.input_embedding.defined()) {
      params_for_capture->embedding.input_embedding =
          persistent_embedding(padded_num_tokens);
    }
    // Set q_cu_seq_lens if available
    if (params.attention.device.q_cu_seq_lens.defined()) {
      params_for_capture->attention.device.q_cu_seq_lens =
          q_cu_seq_lens_.slice(/*dim=*/0,
                               /*start=*/0,
                               /*end=*/padded_batch_size);
    }

    return params_for_capture;
  }
  return std::nullopt;
}

void GraphPersistentParam::initialize_paged_attention_plan_context(
    const torch::Device& device) {
  // max paged attention tiling buffer size is 1024 * 256
  constexpr int64_t tiling_buffer_size = 1024 * 256;
  tiling_data_ = torch::zeros({tiling_buffer_size},
                              torch::dtype(torch::kInt32).device(device));

  // Initialize ATB context for paged attention plan
  atb::Status status = atb::customize::CreatePlanContext(&context_for_plan_);
  CHECK_EQ(status, atb::NO_ERROR)
      << "Failed to create ATB context for paged attention plan";

  // Create stream for paged attention plan
  aclError acl_status = aclrtCreateStream(&stream_for_plan_);
  CHECK_EQ(acl_status, ACL_SUCCESS)
      << "Failed to create ACL stream for paged attention plan";
  context_for_plan_->SetExecuteStream(stream_for_plan_);

  // Set launch mode to GRAPH_LAUNCH_MODE
  status = context_for_plan_->SetLaunchMode(atb::LaunchMode::GRAPH_LAUNCH_MODE);
  CHECK_EQ(status, atb::NO_ERROR)
      << "Failed to set launch mode to GRAPH_LAUNCH_MODE";

  // Create custom paged attention operation
  const int dp_local_tp_size = options_.world_size() / options_.dp_size();

  // Cache headNum and head_dim as member variables
  num_head_ = static_cast<int32_t>(args_.n_heads() / dp_local_tp_size);
  head_dim_ = static_cast<int32_t>(args_.head_dim());

  atb::customize::CustomPagedAttentionParam paOpParam;
  // default mask type is UNDEFINED, which means no mask is needed
  if (need_update_attn_mask_) {
    paOpParam.maskType =
        atb::infer::PagedAttentionParam::MaskType::MASK_TYPE_NORM;
  }
  paOpParam.headNum = num_head_;

  std::optional<long int> optionalValue = args_.n_kv_heads();
  paOpParam.kvHeadNum =
      std::max(1,
               static_cast<int32_t>(optionalValue.value_or(args_.n_heads())) /
                   dp_local_tp_size);

  const float head_dim_float = static_cast<float>(head_dim_);
  paOpParam.qkScale = 1.0f / std::sqrt(head_dim_float);

  const bool isBF16 = args_.dtype() == "bfloat16";
  if (isBF16) {
    paOpParam.outDataType = ACL_BF16;
  } else {
    paOpParam.outDataType = ACL_FLOAT16;
  }

  status = atb::CreateOperation(paOpParam, &custom_pa_op_for_plan_);
  CHECK_EQ(status, atb::NO_ERROR)
      << "Failed to create custom paged attention operation";
  CHECK_NE(custom_pa_op_for_plan_, nullptr) << "custom_pa_op_for_plan_ is null";
}

constexpr uint32_t TILING_PARA_SIZE = 17;
constexpr uint32_t TILING_HEAD_SIZE = 44;

namespace {
void parse_pa_host_tiling_buffer(const uint32_t* hostTilingBuffer,
                                 uint64_t tilingBufferSize) {
  VLOG(kGraphExecutorLogVerboseLevel)
      << "hostTilingBuffer.tilingBuffer: " << (void*)hostTilingBuffer;
  VLOG(kGraphExecutorLogVerboseLevel)
      << "hostTilingBuffer.tilingBufferSize: " << tilingBufferSize;
  if (hostTilingBuffer == nullptr || tilingBufferSize == 0) {
    VLOG(kGraphExecutorLogVerboseLevel) << "Invalid host tiling buffer!";
    return;
  }

  uint32_t tilingParamSize = tilingBufferSize / sizeof(uint32_t);
  VLOG(kGraphExecutorLogVerboseLevel)
      << "Total tiling param elements: " << tilingParamSize;

  // Parse header fields (TILING_HEAD_SIZE = 44)
  VLOG(kGraphExecutorLogVerboseLevel) << "\n=== Tiling Header Fields ===";
  VLOG(kGraphExecutorLogVerboseLevel)
      << "TILING_BATCH(tiling_head[0]): " << hostTilingBuffer[0];
  VLOG(kGraphExecutorLogVerboseLevel)
      << "TILING_NUMHEADS(tiling_head[1]): " << hostTilingBuffer[1];
  VLOG(kGraphExecutorLogVerboseLevel)
      << "TILING_HEADDIM(tiling_head[2]): " << hostTilingBuffer[2];
  VLOG(kGraphExecutorLogVerboseLevel)
      << "TILING_NUMBLOKS(tiling_head[3]): " << hostTilingBuffer[3];
  VLOG(kGraphExecutorLogVerboseLevel)
      << "TILING_BLOCKSIZE(tiling_head[4]): " << hostTilingBuffer[4];
  VLOG(kGraphExecutorLogVerboseLevel)
      << "TILING_MAXBLOCKS(tiling_head[5]): " << hostTilingBuffer[5];
  VLOG(kGraphExecutorLogVerboseLevel)
      << "TILING_TOR(tiling_head[6]): " << hostTilingBuffer[6];
  VLOG(kGraphExecutorLogVerboseLevel)
      << "TILING_KVHEADS(tiling_head[7]): " << hostTilingBuffer[7];
  VLOG(kGraphExecutorLogVerboseLevel)
      << "TILING_FORMER_BATCH(tiling_head[8]): " << hostTilingBuffer[8];
  VLOG(kGraphExecutorLogVerboseLevel)
      << "TILING_FORMER_HEAD(tiling_head[9]): " << hostTilingBuffer[9];
  VLOG(kGraphExecutorLogVerboseLevel)
      << "TILING_TAIL_BATCH(tiling_head[10]): " << hostTilingBuffer[10];
  VLOG(kGraphExecutorLogVerboseLevel)
      << "TILING_TAIL_HEAD(tiling_head[11]): " << hostTilingBuffer[11];
  VLOG(kGraphExecutorLogVerboseLevel)
      << "TILING_HEADNUM_MOVE(tiling_head[12]): " << hostTilingBuffer[12];
  VLOG(kGraphExecutorLogVerboseLevel)
      << "TILING_MASK_MAX_LEN(tiling_head[13]): " << hostTilingBuffer[13];
  VLOG(kGraphExecutorLogVerboseLevel)
      << "TILING_BATCH_STRIDE(tiling_head[14]): " << hostTilingBuffer[14];
  VLOG(kGraphExecutorLogVerboseLevel)
      << "TILING_HEAD_STRIDE(tiling_head[15]): " << hostTilingBuffer[15];
  VLOG(kGraphExecutorLogVerboseLevel)
      << "TILING_KEY(tiling_head[16]): " << hostTilingBuffer[16];
  VLOG(kGraphExecutorLogVerboseLevel)
      << "TILING_HEADSIZE(tiling_head[17]): " << hostTilingBuffer[17];
  VLOG(kGraphExecutorLogVerboseLevel)
      << "TILING_PARASIZE(tiling_head[18]): " << hostTilingBuffer[18];
  VLOG(kGraphExecutorLogVerboseLevel)
      << "TILING_GROUPNUM(tiling_head[19]): " << hostTilingBuffer[19];
  VLOG(kGraphExecutorLogVerboseLevel)
      << "TILING_FORMER_GROUP_MOVE(tiling_head[20]): " << hostTilingBuffer[20];
  VLOG(kGraphExecutorLogVerboseLevel)
      << "TILING_TAIL_GROUP_MOVE(tiling_head[21]): " << hostTilingBuffer[21];
  VLOG(kGraphExecutorLogVerboseLevel)
      << "TILING_MAX_KVSEQLEN(tiling_head[22]): " << hostTilingBuffer[22];
  VLOG(kGraphExecutorLogVerboseLevel)
      << "TILING_KVSPLIT(tiling_head[23]): " << hostTilingBuffer[23];
  VLOG(kGraphExecutorLogVerboseLevel)
      << "TILING_KVCORENUM(tiling_head[24]): " << hostTilingBuffer[24];
  VLOG(kGraphExecutorLogVerboseLevel)
      << "TILING_BLOCKSIZE_CALC(tiling_head[25]): " << hostTilingBuffer[25];
  VLOG(kGraphExecutorLogVerboseLevel)
      << "TILING_TOTAL_BLOCK_NUM(tiling_head[26]): " << hostTilingBuffer[26];
  VLOG(kGraphExecutorLogVerboseLevel)
      << "TILING_PREFILL_BS(tiling_head[27]): " << hostTilingBuffer[27];
  VLOG(kGraphExecutorLogVerboseLevel)
      << "TILING_DECODER_BS(tiling_head[28]): " << hostTilingBuffer[28];
  VLOG(kGraphExecutorLogVerboseLevel)
      << "TILING_HEADDIM_V(tiling_head[29]): " << hostTilingBuffer[29];
  VLOG(kGraphExecutorLogVerboseLevel)
      << "TILING_MODCOEF(tiling_head[30]): " << hostTilingBuffer[30];
  VLOG(kGraphExecutorLogVerboseLevel)
      << "TILING_DIVCOEF(tiling_head[31]): " << hostTilingBuffer[31];
  VLOG(kGraphExecutorLogVerboseLevel)
      << "TILING_QHEADORIGINAL(tiling_head[32]): " << hostTilingBuffer[32];
  VLOG(kGraphExecutorLogVerboseLevel)
      << "TILING_COMPRESSHEAD(tiling_head[33]): " << hostTilingBuffer[33];
  VLOG(kGraphExecutorLogVerboseLevel)
      << "TILING_QUANTYPE(tiling_head[34]): " << hostTilingBuffer[34];
  VLOG(kGraphExecutorLogVerboseLevel)
      << "TILING_DATA_SHAPE_TYPE(tiling_head[35]): " << hostTilingBuffer[35];
  VLOG(kGraphExecutorLogVerboseLevel)
      << "TILING_SCALETYPE(tiling_head[36]): " << hostTilingBuffer[36];
  VLOG(kGraphExecutorLogVerboseLevel)
      << "TILING_MASK_TYPE_ND(tiling_head[37]): " << hostTilingBuffer[37];
  VLOG(kGraphExecutorLogVerboseLevel)
      << "TILING_HEADDIM_K_SPLIT(tiling_head[38]): " << hostTilingBuffer[38];
  VLOG(kGraphExecutorLogVerboseLevel)
      << "TILING_HEADDIM_V_SPLIT(tiling_head[39]): " << hostTilingBuffer[39];
  VLOG(kGraphExecutorLogVerboseLevel)
      << "TILING_HEADDIM_V_SPLIT_VECTOR_FORMER(tiling_head[40]): "
      << hostTilingBuffer[40];
  VLOG(kGraphExecutorLogVerboseLevel)
      << "TILING_HEADDIM_V_SPLIT_VECTOR_TAIL(tiling_head[41]): "
      << hostTilingBuffer[41];
  VLOG(kGraphExecutorLogVerboseLevel)
      << "TILING_MTP_HEAD_SPLIT_SIZE(tiling_head[42]): "
      << hostTilingBuffer[42];
  VLOG(kGraphExecutorLogVerboseLevel)
      << "TILING_MTP_HEAD_SPLIT_NUM(tiling_head[43]): " << hostTilingBuffer[43];

  // Parse batch parameters
  if (tilingParamSize > TILING_HEAD_SIZE) {
    uint32_t batchCount = hostTilingBuffer[0];
    VLOG(kGraphExecutorLogVerboseLevel) << "\n=== Batch Parameters ===";
    VLOG(kGraphExecutorLogVerboseLevel) << "Number of batches: " << batchCount;
    batchCount = std::min(batchCount, 20u);

    for (uint32_t batchIdx = 0; batchIdx < batchCount; ++batchIdx) {
      uint32_t offset = TILING_HEAD_SIZE + batchIdx * TILING_PARA_SIZE;
      if (offset + TILING_PARA_SIZE <= tilingParamSize) {
        VLOG(kGraphExecutorLogVerboseLevel)
            << "\n--- Batch " << batchIdx << " ---";
        VLOG(kGraphExecutorLogVerboseLevel)
            << "  qSeqLen(batch_tiling_param[0]): "
            << hostTilingBuffer[offset + 0];
        VLOG(kGraphExecutorLogVerboseLevel)
            << "  kvSeqLen(batch_tiling_param[1]): "
            << hostTilingBuffer[offset + 1];
        VLOG(kGraphExecutorLogVerboseLevel)
            << "  qSBlockTile(batch_tiling_param[2]): "
            << hostTilingBuffer[offset + 2];
        VLOG(kGraphExecutorLogVerboseLevel)
            << "  blockSize(batch_tiling_param[3]): "
            << hostTilingBuffer[offset + 3];
        VLOG(kGraphExecutorLogVerboseLevel)
            << "  addrQSeqOffset[high](batch_tiling_param[4]): "
            << hostTilingBuffer[offset + 4];
        VLOG(kGraphExecutorLogVerboseLevel)
            << "  addrQSeqOffset[low](batch_tiling_param[5]): "
            << hostTilingBuffer[offset + 5];
        VLOG(kGraphExecutorLogVerboseLevel)
            << "  addrOSeqOffset[high](batch_tiling_param[6]): "
            << hostTilingBuffer[offset + 6];
        VLOG(kGraphExecutorLogVerboseLevel)
            << "  addrOSeqOffset[low](batch_tiling_param[7]): "
            << hostTilingBuffer[offset + 7];
        VLOG(kGraphExecutorLogVerboseLevel)
            << "  seqIdx(batch_tiling_param[8]): "
            << hostTilingBuffer[offset + 8];
        VLOG(kGraphExecutorLogVerboseLevel)
            << "  totalQBlkNum(batch_tiling_param[9]): "
            << hostTilingBuffer[offset + 9];
        VLOG(kGraphExecutorLogVerboseLevel)
            << "  maskOffset[high](batch_tiling_param[10]): "
            << hostTilingBuffer[offset + 10];
        VLOG(kGraphExecutorLogVerboseLevel)
            << "  addrLSeqOffset[high](batch_tiling_param[11]): "
            << hostTilingBuffer[offset + 11];
        VLOG(kGraphExecutorLogVerboseLevel)
            << "  addrLSeqOffset[low](batch_tiling_param[12]): "
            << hostTilingBuffer[offset + 12];
        VLOG(kGraphExecutorLogVerboseLevel)
            << "  maskOffset[low](batch_tiling_param[14]): "
            << hostTilingBuffer[offset + 14];
        VLOG(kGraphExecutorLogVerboseLevel)
            << "  addrOFdSeqOffset[high](batch_tiling_param[15]): "
            << hostTilingBuffer[offset + 15];
        VLOG(kGraphExecutorLogVerboseLevel)
            << "  addrOFdSeqOffset[low](batch_tiling_param[16]): "
            << hostTilingBuffer[offset + 16];
      }
    }
  }

  VLOG(kGraphExecutorLogVerboseLevel) << "\n=== End of Tiling Buffer Parse ===";
}
}  // namespace

void GraphPersistentParam::plan_paged_attention_tiling(
    const torch::Tensor& tokens,
    const torch::Tensor& k_cache,
    const torch::Tensor& v_cache,
    const torch::Tensor& block_tables,
    const ModelInputParams& input_params,
    aclrtStream stream) {
  // Convert torch tensors to atb tensors
  atb::Tensor atb_k_cache = atb_speed::Utils::AtTensor2Tensor(k_cache);
  atb::Tensor atb_v_cache = atb_speed::Utils::AtTensor2Tensor(v_cache);
  atb::Tensor atb_block_tables =
      atb_speed::Utils::AtTensor2Tensor(block_tables);
  // Get context_lens from input_params.attention.device.kv_seq_lens
  atb::Tensor atb_context_lens = atb_speed::Utils::AtTensor2Tensor(
      input_params.attention.device.kv_seq_lens);
  atb_context_lens.hostData =
      const_cast<int32_t*>(input_params.attention.host.kv_seq_lens.data());
  atb::Tensor atb_tiling_data = atb_speed::Utils::AtTensor2Tensor(tiling_data_);

  atb_tiling_data.desc.dtype = ACL_UINT32;

  // Construct query atb tensor from tokens: shape [num_tokens, headNum,
  // head_dim] Get number of tokens from tokens tensor
  const int64_t num_tokens = tokens.size(0);

  // Create query atb tensor with only desc (no actual data needed)
  atb::Tensor atb_query;
  // TODO: support quant dtype
  atb_query.desc.dtype = (args_.dtype() == "bfloat16") ? ACL_BF16 : ACL_FLOAT16;
  atb_query.desc.format = ACL_FORMAT_ND;
  atb_query.desc.shape.dimNum = 3;
  atb_query.desc.shape.dims[0] = num_tokens;
  atb_query.desc.shape.dims[1] = num_head_;
  atb_query.desc.shape.dims[2] = head_dim_;
  atb_query.deviceData = atb_k_cache.deviceData;
  atb_query.hostData = nullptr;
  // Calculate dataSize: num_tokens * headNum * head_dim * sizeof(dtype)
  // ACL_FLOAT16 and ACL_BF16 both use 2 bytes per element
  const uint64_t element_size =
      (atb_query.desc.dtype == ACL_BF16 || atb_query.desc.dtype == ACL_FLOAT16)
          ? 2
          : 1;
  atb_query.dataSize = static_cast<uint64_t>(num_tokens) *
                       static_cast<uint64_t>(num_head_) *
                       static_cast<uint64_t>(head_dim_) * element_size;

  atb::VariantPack custom_variantPack;
  // Conditionally include mask based on need_update_attn_mask_
  if (need_update_attn_mask_) {
    atb::Tensor atb_mask = atb_speed::Utils::AtTensor2Tensor(persistent_mask_);
    custom_variantPack.inTensors = {atb_query,
                                    atb_k_cache,
                                    atb_v_cache,
                                    atb_block_tables,
                                    atb_context_lens,
                                    atb_mask,
                                    atb_tiling_data};
  } else {
    // Skip mask when not needed
    custom_variantPack.inTensors = {atb_query,
                                    atb_k_cache,
                                    atb_v_cache,
                                    atb_block_tables,
                                    atb_context_lens,
                                    atb_tiling_data};
  }
  custom_variantPack.outTensors.push_back(atb_query);

  uint64_t custom_workspace_size = 0;
  atb::Status status = custom_pa_op_for_plan_->Setup(
      custom_variantPack, custom_workspace_size, context_for_plan_);
  CHECK_EQ(status, atb::NO_ERROR)
      << "Failed to setup custom paged attention operation for tiling";

  atb::customize::TilingBufferInfo tiling_buffer_info =
      atb::customize::GetHostTilingBufferFromCustomPagedAttentionOperation(
          custom_pa_op_for_plan_);

  CHECK_NE(tiling_buffer_info.tilingBuffer, nullptr)
      << "Tiling buffer is null after setup";
  CHECK_GT(tiling_buffer_info.tilingBufferSize, 0)
      << "Tiling buffer size is zero";

  if (VLOG_IS_ON(kGraphExecutorLogVerboseLevel)) {
    parse_pa_host_tiling_buffer((uint32_t*)tiling_buffer_info.tilingBuffer,
                                tiling_buffer_info.tilingBufferSize);
  }
  aclError acl_status =
      aclrtMemcpyAsync(tiling_data_.data_ptr(),
                       tiling_data_.numel() * sizeof(uint32_t),
                       tiling_buffer_info.tilingBuffer,
                       tiling_buffer_info.tilingBufferSize,
                       ACL_MEMCPY_HOST_TO_DEVICE,
                       stream);
  CHECK_EQ(acl_status, ACL_SUCCESS) << "Failed to copy tiling buffer to device";
}

void GraphPersistentParam::update_attention_mask(
    const ModelInputParams& input_params) {
  // update persistent_mask_ in-place
  const int64_t batch_size = input_params.attention.device.kv_seq_lens.size(0);
  const int64_t max_seq_len = input_params.meta.kv_max_seq_len > 0
                                  ? input_params.meta.kv_max_seq_len
                                  : args_.max_position_embeddings();

  // persistent_mask_ is already initialized in constructor
  // Check if size is sufficient
  CHECK_LE(max_seq_len, persistent_mask_.size(1))
      << "max_seq_len (" << max_seq_len << ") exceeds max_seq_len ("
      << persistent_mask_.size(1) << ")";

  // Check if q_max_seq_len > 1 (prefill mode, not decode mode)
  bool chunked_prefill = input_params.meta.q_max_seq_len > 1;

  // Calculate num_tokens: in chunked mode, sum of all q_len; in decode mode,
  // batch_size
  int64_t num_tokens = batch_size;  // Default for decode mode
  if (chunked_prefill) {
    CHECK_EQ(input_params.attention.host.q_seq_lens.size(), batch_size)
        << "q_seq_lens_vec size ("
        << input_params.attention.host.q_seq_lens.size() << ") != batch_size ("
        << batch_size << ")";
    num_tokens = std::accumulate(
        input_params.attention.host.q_seq_lens.begin(),
        input_params.attention.host.q_seq_lens.begin() + batch_size,
        int64_t(0));
  }

  // Check if num_tokens is within bounds
  CHECK_LE(num_tokens, persistent_mask_.size(0))
      << "num_tokens (" << num_tokens
      << ") exceeds graph attention-mask capacity (" << persistent_mask_.size(0)
      << ")";

  // Get slice for actual num_tokens (compatible with both chunked and
  // non-chunked)
  auto mask_slice =
      persistent_mask_.slice(/*dim=*/0, /*start=*/0, /*end=*/num_tokens)
          .slice(/*dim=*/1, /*start=*/0, /*end=*/max_seq_len);

  CHECK(persistent_mask_zero_template_.defined() &&
        persistent_mask_fill_template_.defined())
      << "persistent mask templates must be initialized";

  if (chunked_prefill) {
    // Generate mask considering both q_seq_lens and kv_seq_lens
    // For each sequence, generate mask with shape [q_len, kv_len]
    // mask_slice is [num_tokens, max_seq_len], where num_tokens = sum of all
    // q_len

    // Check if kv_seq_lens_vec is available
    CHECK_EQ(input_params.attention.host.kv_seq_lens.size(), batch_size)
        << "kv_seq_lens_vec size ("
        << input_params.attention.host.kv_seq_lens.size() << ") != batch_size ("
        << batch_size << ")";

    int64_t offset = 0;
    for (int64_t i = 0; i < batch_size; i++) {
      const int32_t q_len = input_params.attention.host.q_seq_lens[i];
      const int32_t kv_len = input_params.attention.host.kv_seq_lens[i];

      // For chunked mode, slice out q_len rows for this sequence
      // mask_slice is [num_tokens, max_seq_len]
      // Get [q_len, kv_len] slice from mask_slice[offset:offset+q_len, :kv_len]
      auto seq_mask_slice =
          mask_slice.slice(/*dim=*/0, /*start=*/offset, /*end=*/offset + q_len)
              .slice(
                  /*dim=*/1, /*start=*/0, /*end=*/kv_len);  // [q_len, kv_len]
      auto seq_zero_slice = persistent_mask_zero_template_
                                .slice(/*dim=*/0,
                                       /*start=*/offset,
                                       /*end=*/offset + q_len)
                                .slice(/*dim=*/1, /*start=*/0, /*end=*/kv_len);
      auto seq_fill_slice = persistent_mask_fill_template_
                                .slice(/*dim=*/0,
                                       /*start=*/offset,
                                       /*end=*/offset + q_len)
                                .slice(/*dim=*/1, /*start=*/0, /*end=*/kv_len);

      // Generate mask for this sequence: [q_len, kv_len]
      int diagonal = kv_len - q_len;
      auto int_options =
          torch::TensorOptions().dtype(torch::kInt32).device(device_);
      auto row = torch::arange(q_len, int_options).unsqueeze(1);
      auto col = torch::arange(kv_len, int_options).unsqueeze(0);
      auto bias = col > (row + diagonal);  // True positions need to be masked
      seq_mask_slice.copy_(torch::where(bias, seq_fill_slice, seq_zero_slice),
                           /*non_blocking=*/true);

      // Update offset for next sequence
      offset += q_len;
    }
  } else {
    // Original logic: only consider kv_seq_lens (decode mode, q_len = 1 for
    // all)
    auto int_options =
        torch::TensorOptions().dtype(torch::kInt32).device(device_);
    auto positions = torch::arange(max_seq_len, int_options)
                         .unsqueeze(0)
                         .expand({batch_size, max_seq_len});

    auto context_lens_expanded =
        input_params.attention.device.kv_seq_lens.to(torch::kInt32)
            .unsqueeze(1)
            .expand({batch_size, max_seq_len});

    auto mask_condition = positions >= context_lens_expanded;
    auto zero_slice = persistent_mask_zero_template_
                          .slice(/*dim=*/0, /*start=*/0, /*end=*/num_tokens)
                          .slice(/*dim=*/1, /*start=*/0, /*end=*/max_seq_len);
    auto fill_slice = persistent_mask_fill_template_
                          .slice(/*dim=*/0, /*start=*/0, /*end=*/num_tokens)
                          .slice(/*dim=*/1, /*start=*/0, /*end=*/max_seq_len);
    mask_slice.copy_(torch::where(mask_condition, fill_slice, zero_slice),
                     /*non_blocking=*/true);
  }
}

bool AclGraph::capture(CausalLM* model,
                       const ModelArgs& args,
                       const runtime::Options& options,
                       const torch::Tensor& tokens,
                       const torch::Tensor& positions,
                       const ModelInputParams& params,
                       std::vector<KVCache>& kv_cache,
                       uint32_t bucket_num_tokens) {
  // Save bucket num_tokens for this graph instance
  num_tokens_ = bucket_num_tokens;

  // Get actual num_tokens from tokens tensor
  // const uint32_t actual_num_tokens = tokens.size(0);

  auto& tensor_options = model->options();

  torch::npu::synchronize();

  // Begin graph capture using NPUGraph mempool for temporary tensor management
  // Get current NPU stream from libtorch NPU API
  aclrtStream stream =
      c10_npu::getCurrentNPUStream(tensor_options.device().index()).stream();

  // For hybrid models (e.g., qwen3_next with mixed GDN/full_attention layers),
  // we need to find the first Full Attention layer to get the correct kv_cache.
  // GDN layers have empty key_cache_/value_cache_ while Full Attention layers
  // have valid kv caches. Using layer 0's cache directly would be incorrect
  // if layer 0 is a GDN layer.
  auto [k_cache, v_cache] = find_attention_plan_kv_cache(kv_cache);
  const uint32_t actual_num_tokens = tokens.size(0);
  CHECK_GE(num_tokens_, actual_num_tokens)
      << "num_tokens_ >= actual_num_tokens";
  auto graph_params = persistent_param_.update(tokens,
                                               k_cache,
                                               v_cache,
                                               positions,
                                               params,
                                               num_tokens_,
                                               /*return_capture_params=*/true);

  // Use the returned ModelInputParams for graph capture
  CHECK(graph_params.has_value())
      << "update() should return ModelInputParams when "
         "return_capture_params=true";

  if (model->requires_graph_forward_metadata()) {
    if (!model_graph_metadata_state_) {
      model_graph_metadata_state_ =
          model->create_graph_forward_metadata_state();
    }
    model->prepare_graph_forward_metadata(
        model_graph_metadata_state_.get(),
        persistent_param_.persistent_positions(num_tokens_),
        graph_params.value());
  }

  // Synchronize stream to ensure all data is copied to graph persistent buffers
  aclrtSynchronizeStream(stream);

  // Acquire device-level lock to prevent prepare_work_before_execute from
  // executing simultaneously, which would trigger synchronous operations
  // that conflict with capture mode
  auto device_idx = tensor_options.device().index();

  bool need_restore_stream = false;
  graph_stream_ = stream;

  // capture lock scope
  {
    auto& capture_lock =
        ::xllm::npu::DeviceCaptureLock::get_instance().get_lock(device_idx);
    std::lock_guard<std::mutex> lock_guard(capture_lock);

    if (c10_npu::getCurrentNPUStream(device_idx) ==
        c10_npu::getDefaultNPUStream(device_idx)) {
      c10_npu::setCurrentNPUStream(capture_stream_.value());
      aclrtSynchronizeStream(capture_stream_.value().stream());
      graph_stream_ = capture_stream_.value().stream();
      need_restore_stream = true;
    }
    LOG(INFO) << "capture begin, bucket_num_tokens: " << bucket_num_tokens
              << ", actual_num_tokens: " << actual_num_tokens;

    // no mempool id, will create a new one; capture mode is thread local, allow
    // other threads to execute synchronous operations
    graph_.capture_begin(
        {0, 0}, aclmdlRICaptureMode::ACL_MODEL_RI_CAPTURE_MODE_THREAD_LOCAL);
    // Execute forward pass - NPUGraph mempool manages temporary tensors
    auto forward_result =
        model->forward({persistent_param_.persistent_tokens(num_tokens_)},
                       {persistent_param_.persistent_positions(num_tokens_)},
                       kv_cache,
                       {graph_params.value()});

    // Store result in persistent buffer owned by NPUGraph mempool
    persistent_param_.set_hidden_states(forward_result.hidden_states);
    if (options.enable_graph_aux_hidden_states() &&
        forward_result.aux_hidden_states.defined()) {
      persistent_param_.set_aux_hidden_states(forward_result.aux_hidden_states);
    }
    graph_.capture_end();
    // Lock is automatically released here when lock goes out of scope
    if (need_restore_stream) {
      c10_npu::setCurrentNPUStream(
          c10_npu::getDefaultNPUStream(tensor_options.device().index()));
    }
  }
  // Synchronize and test replay to verify graph capture
  aclrtSynchronizeStream(graph_stream_);
  aclrtSynchronizeStream(stream);

  graph_.replay();

  make_current_stream_wait_for_graph(stream);
  return true;
}

AclGraph::~AclGraph() {
  if (graph_stream_ != nullptr) {
    aclrtSynchronizeStream(graph_stream_);
  } else if (capture_stream_.has_value()) {
    aclrtSynchronizeStream(capture_stream_.value().stream());
  }
  if (replay_done_event_ != nullptr) {
    aclrtDestroyEvent(replay_done_event_);
    replay_done_event_ = nullptr;
  }
}

void AclGraph::initialize_capture_stream(c10::DeviceIndex device_index) {
  // Get a secondary stream from high-priority pool for graph capture.
  // This is required because NPUGraph::capture_begin() enforces that capture
  // must be performed on a non-default stream (see
  // torch_npu/csrc/core/npu/NPUGraph.cpp:159).
  capture_stream_ = c10_npu::getStreamFromPool(true, device_index);
  device_index_ = device_index;
  CHECK_EQ(aclrtCreateEventWithFlag(&replay_done_event_, ACL_EVENT_SYNC),
           ACL_SUCCESS)
      << "Failed to create ACL graph replay completion event";
  LOG(INFO) << "Initialized capture_stream: " << capture_stream_.value()
            << ", id: " << capture_stream_.value().id()
            << ", device_index: " << device_index;
}

void AclGraph::make_current_stream_wait_for_graph(aclrtStream current_stream) {
  CHECK_NE(graph_stream_, nullptr) << "graph_stream is not initialized";
  CHECK_NE(replay_done_event_, nullptr)
      << "replay_done_event is not initialized";
  CHECK_EQ(aclrtRecordEvent(replay_done_event_, graph_stream_), ACL_SUCCESS)
      << "aclrtRecordEvent(replay_done_event) failed";
  if (current_stream != graph_stream_) {
    CHECK_EQ(aclrtStreamWaitEvent(current_stream, replay_done_event_),
             ACL_SUCCESS)
        << "aclrtStreamWaitEvent(current_stream, replay_done_event) failed";
  }
}

ModelOutput AclGraph::replay(CausalLM* model,
                             const ModelArgs& args,
                             const torch::Tensor& tokens,
                             const torch::Tensor& positions,
                             std::vector<KVCache>& kv_cache,
                             const ModelInputParams& params) {
  const uint32_t actual_num_tokens = tokens.size(0);
  CHECK_LE(actual_num_tokens, num_tokens_)
      << "num_tokens mismatch: expected <= " << num_tokens_ << ", got "
      << actual_num_tokens;

  // Update persistent parameters with new input data
  // Note: tiling_data is updated in update() if needed - for hybrid models
  // (e.g., qwen3_next with mixed GDN/attention layers), tiling should only
  // be updated when Full Attention layers are involved, which is determined
  // by k_cache being valid and non-empty
  auto [k_cache, v_cache] = find_attention_plan_kv_cache(kv_cache);
  auto graph_params =
      persistent_param_.update(tokens,
                               k_cache,
                               v_cache,
                               positions,
                               params,
                               num_tokens_,
                               model->requires_graph_forward_metadata());
  if (model->requires_graph_forward_metadata()) {
    CHECK(graph_params.has_value())
        << "ACL graph replay requires persistent params for graph metadata";
    CHECK(model_graph_metadata_state_)
        << "ACL graph metadata state must be initialized during capture";
    model->prepare_graph_forward_metadata(
        model_graph_metadata_state_.get(),
        persistent_param_.persistent_positions(num_tokens_),
        graph_params.value());
  }

  // Replay captured graph - NPUGraph mempool reuses temporary tensors
  // Get current NPU stream from libtorch NPU API
  aclrtStream stream = c10_npu::getCurrentNPUStream().stream();

  graph_.replay();

  // NPUGraph replays on its capture stream. Add a device-side dependency so
  // the current/default stream only observes completed outputs.
  make_current_stream_wait_for_graph(stream);

  // Return the actual num_tokens portion of ModelOutput
  // Note: aux_hidden_states handling is done in AclGraphExecutorImpl::run()
  // since replay() doesn't have access to options
  auto hidden_states = get_hidden_states(actual_num_tokens);
  return ModelOutput(hidden_states);
}

AclGraphExecutorImpl::AclGraphExecutorImpl(CausalLM* model,
                                           const ModelArgs& args,
                                           const torch::Device& device,
                                           const runtime::Options& options)
    : model_(model), args_(args), device_(device), options_(options) {
  // Create single persistent parameter object shared by all AclGraph instances
  persistent_param_ =
      std::make_unique<GraphPersistentParam>(args_, device_, options_);
}

ForwardInput AclGraphExecutorImpl::prepare_inputs(Batch& batch) {
  // Prepare inputs for workers
  return batch.prepare_forward_input(
      options_.num_decoding_tokens(), 0, args_, options_.cp_size());
}

// Main execution method with graph optimization for decode phase
// tokens: [num_decode_tokens]
// positions: [num_decode_tokens] token pos in the sequence
// returns: [num_decode_tokens, hidden_size]
ModelOutput AclGraphExecutorImpl::run(const torch::Tensor& tokens,
                                      const torch::Tensor& positions,
                                      std::vector<KVCache>& kv_caches,
                                      const ModelInputParams& params) {
  // no mirco batch in decode phase
  const torch::Tensor& tokens_tensor = tokens;
  const torch::Tensor& positions_tensor = positions;
  const ModelInputParams& params_single = params;
  const bool in_decoding_phase =
      params_single.meta.batch_forward_type.is_decode();
  VLOG(50) << "in_decoding_phase: " << in_decoding_phase
           << " q_max_seq_len: " << params_single.meta.q_max_seq_len
           << " n_layers: " << args_.n_layers();
  // If not in decode phase, use eager mode directly without acl graph
  // TODO: fix mtp model support.
  if (!in_decoding_phase || args_.n_layers() == 1) {
    VLOG(kGraphExecutorLogVerboseLevel)
        << "AclGraphExecutorImpl::run() in eager mode";
    COUNTER_INC(num_model_execution_total_eager);
    return model_->forward(tokens, positions, kv_caches, params);
  }

  // Only use acl graph in decode phase for performance optimization
  // For DP, decode graph bucket should be based on global max tokens across dp
  // groups; local shard can be empty on some ranks.
  uint32_t graph_num_tokens = tokens_tensor.size(/*dim=*/0);
  if (params_single.parallel.dp_global_token_nums.size() > 1) {
    graph_num_tokens = util::max(params_single.parallel.dp_global_token_nums);
  }
  // Keep actual n_tokens for replay output slicing.
  const uint32_t n_tokens = tokens_tensor.size(/*dim=*/0);

  const uint32_t bucket_num_tokens = get_bucket_num_tokens(graph_num_tokens);

  // Check if conditions are suitable for graph execution (replay or capture)
  const auto max_seq_len = args_.max_position_embeddings();
  const bool seq_len_supported =
      params_single.meta.kv_max_seq_len <= max_seq_len;

  // Combined condition for graph capture support
  // ACL graph executor only supports single tensor inputs (no micro-batching)
  const bool capture_supported = seq_len_supported;

  // Early return if conditions are not suitable for graph operations
  if (!capture_supported) {
    LOG_FIRST_N(WARNING, 1)
        << "Falling back to eager mode because kv_max_seq_len ("
        << params_single.meta.kv_max_seq_len << ") > max_seq_len ("
        << max_seq_len << "). This message is logged only once. "
        << "Monitor counter 'num_model_execution_total_eager' for frequency.";
    COUNTER_INC(num_model_execution_total_eager);
    return model_->forward(tokens, positions, kv_caches, params);
  }

  // Check if captured graph exists for this bucket num_tokens
  auto it = graphs_.find(bucket_num_tokens);
  if (it != graphs_.end()) {
    // Replay the existing graph
    VLOG(kGraphExecutorLogVerboseLevel)
        << "AclGraphExecutorImpl::run() in replay mode";
    auto result = it->second->replay(model_,
                                     args_,
                                     tokens_tensor,
                                     positions_tensor,
                                     kv_caches,
                                     params_single);
    // Handle aux_hidden_states based on options
    if (options_.enable_graph_aux_hidden_states()) {
      auto aux_hidden_states = persistent_param_->aux_hidden_states(n_tokens);
      if (aux_hidden_states.defined() && aux_hidden_states.numel() > 0) {
        return ModelOutput(
            result.hidden_states, torch::Tensor(), aux_hidden_states);
      }
    }
    return result;
  }

  // Graph doesn't exist for this bucket num_tokens, try to create it lazily
  auto graph = std::make_unique<AclGraph>(*persistent_param_, device_.index());
  VLOG(kGraphExecutorLogVerboseLevel)
      << "AclGraphExecutorImpl::run() in capture mode";
  bool capture_success = graph->capture(model_,
                                        args_,
                                        options_,
                                        tokens_tensor,
                                        positions_tensor,
                                        params_single,
                                        kv_caches,
                                        bucket_num_tokens);

  if (capture_success) {
    LOG(INFO) << "Lazy capturing ACL graph for bucket num_tokens: "
              << bucket_num_tokens << " (actual num_tokens: " << n_tokens
              << ") done";

    // Save the graph for future reuse
    graphs_[bucket_num_tokens] = std::move(graph);

    // Return the output from capture (no need to replay since capture
    // already executed)
    auto hidden_states =
        graphs_[bucket_num_tokens]->get_hidden_states(n_tokens);
    if (options_.enable_graph_aux_hidden_states()) {
      auto aux_hidden_states = persistent_param_->aux_hidden_states(n_tokens);
      if (aux_hidden_states.defined() && aux_hidden_states.numel() > 0) {
        return ModelOutput(hidden_states, torch::Tensor(), aux_hidden_states);
      }
    }
    return ModelOutput(hidden_states);
  }

  // Fallback to eager mode if capture fails
  LOG(ERROR) << "Failed to capture ACL graph for bucket num_tokens: "
             << bucket_num_tokens;
  COUNTER_INC(num_model_execution_total_eager);
  return model_->forward(tokens, positions, kv_caches, params);
}

void AclGraph::print_graph_tensors() const {
  VLOG(kGraphExecutorLogVerboseLevel)
      << "graph persistent_tokens_: " << persistent_param_.persistent_tokens();
  VLOG(kGraphExecutorLogVerboseLevel)
      << "graph persistent_positions_: "
      << persistent_param_.persistent_positions();
  VLOG(kGraphExecutorLogVerboseLevel)
      << "graph persistent_new_cache_slots_: "
      << persistent_param_.persistent_new_cache_slots();
  VLOG(kGraphExecutorLogVerboseLevel)
      << "graph q_seq_lens_: " << persistent_param_.q_seq_lens();
  VLOG(kGraphExecutorLogVerboseLevel)
      << "graph kv_seq_lens_: " << persistent_param_.kv_seq_lens();
  VLOG(kGraphExecutorLogVerboseLevel)
      << "graph persistent_block_tables_: "
      << persistent_param_.persistent_block_tables();
  VLOG(kGraphExecutorLogVerboseLevel)
      << "graph hidden_states_: " << persistent_param_.hidden_states();
}

// bucket will be [1, 2, 4, 8, 16, 32, 48, 64, ..., max_seqs_per_batch]
uint32_t AclGraphExecutorImpl::get_bucket_num_tokens(
    uint32_t num_tokens) const {
  if (::xllm::ExecutionConfig::get_instance()
          .enable_graph_mode_decode_no_padding()) {
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
    // For num_tokens > 16, use multiples of 16
    return ((num_tokens + 15) / 16) * 16;
  }
}

}  // namespace xllm::npu

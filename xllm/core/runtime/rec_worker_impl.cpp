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

#include "rec_worker_impl.h"

#include <glog/logging.h>

#include <algorithm>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <vector>

#include "common/device_monitor.h"
#include "common/metrics.h"
#include "common/rec_model_utils.h"
#include "common/types.h"
#include "framework/model/model_input_params.h"
#if defined(USE_CUDA)
#include "kernels/cuda/cuda_ops_api.h"
#include "kernels/cuda/xattention/xattention_ops_api.h"
#include "layers/cuda/flashinfer_workspace.h"
#include "platform/cuda/device_capture_lock.h"
#endif
#if defined(USE_NPU)
#include "platform/npu/device_capture_lock.h"
#endif
#include "framework/model_loader.h"
#include "framework/sampling/rec_sampler.h"
#include "models/model_registry.h"
#include "util/env_var.h"
#include "util/timer.h"

namespace xllm {

// ============================================================
// RecWorkerImpl Implementation (base)
// ============================================================

void RecWorkerImpl::RecWorkPipeline::prepare_work_before_execute(
    const ForwardInput& inputs,
    ForwardInput& processed_inputs) {
#if defined(USE_NPU)
  // Without device_capture_lock, ACL graph capture will be interrupted by the
  // synchronization H2D of data update streams asynchronously scheduled by
  // other threads, even if the capture and synchronization streams are not the
  // same, and even if capture_mode is set to
  // ACL_MODEL_RI_CAPTURE_MODE_THREAD_LOCAL.
  // The possible reason is that ACL graph capture may use additional auxiliary
  // streams, and these auxiliary streams might be the same as the
  // asynchronously scheduled data update streams.

  std::optional<std::unique_lock<std::mutex>> lock_guard;
  if (FLAGS_enable_graph) {
    auto& capture_lock =
        ::xllm::npu::DeviceCaptureLock::get_instance().get_lock(
            runtime_.worker.device().index());
    lock_guard.emplace(capture_lock);
  }
#endif
  processed_inputs =
      inputs.to(runtime_.worker.device(), runtime_.worker.dtype());
  auto& input_params = processed_inputs.input_params;
#if defined(USE_NPU)
  if (input_params.swap_blocks.size() > 0 && !FLAGS_enable_block_copy_kernel) {
    auto& swap_blocks = input_params.swap_blocks;

    // collect src and dst indices
    std::vector<int64_t> src_indices, dst_indices;
    src_indices.reserve(swap_blocks.size());
    dst_indices.reserve(swap_blocks.size());

    for (const auto& block : swap_blocks) {
      src_indices.push_back(block.src_block_id);
      dst_indices.push_back(block.dst_block_id);
    }

    // batch select keys and values
    auto src_tensor = torch::tensor(
        src_indices,
        torch::dtype(torch::kLong).device(runtime_.worker.device_));
    auto dst_tensor = torch::tensor(
        dst_indices,
        torch::dtype(torch::kLong).device(runtime_.worker.device_));
    const int64_t num_layers = runtime_.context->get_model_args().n_layers();
    for (int layer_id = 0; layer_id < num_layers; layer_id++) {
      runtime_.worker.kv_caches_[layer_id].swap_blocks(src_tensor, dst_tensor);
    }
  }
  if (FLAGS_enable_mla &&
      input_params.batch_forward_type.is_chunked_prefill()) {
    runtime_.worker.prepare_mla_prefixcache_inputs(input_params);
  }

  if (!runtime_.context->get_parallel_args().mapping_data().empty() &&
      (runtime_.context->get_parallel_args().dp_size() > 1 ||
       runtime_.context->get_parallel_args().ep_size() > 1)) {
    torch::Tensor token_size_per_dp_group =
        torch::tensor(processed_inputs.input_params.dp_global_token_nums,
                      torch::TensorOptions()
                          .device(torch::kCPU)
                          .dtype(torch::kInt32)
                          .pinned_memory(true));
    bool is_prefill =
        processed_inputs.input_params.batch_forward_type.is_prefill();
    DpEpPadding dp_ep_padding(
        token_size_per_dp_group,
        runtime_.context->get_model_args().num_experts_per_tok(),
        runtime_.context->get_parallel_args().mapping_data(),
        runtime_.worker.device(),
        runtime_.worker.dtype(),
        is_prefill);
    processed_inputs.input_params.dp_ep_padding_data = dp_ep_padding.build();
  }
#endif
}

ForwardInput RecWorkerImpl::RecWorkPipeline::prepare_inputs(Batch& batch) {
  return runtime_.worker.WorkerImpl::prepare_inputs(batch);
}

std::optional<ForwardOutput> RecWorkerImpl::RecWorkPipeline::step(
    const ForwardInput& input) {
  Timer timer;
  auto& sampling_params = input.sampling_params;

  std::vector<folly::SemiFuture<bool>> futures;

  if (runtime_.worker.options_.kv_cache_transfer_mode() == "PUSH" &&
      !input.transfer_kv_infos.empty()) {
#if defined(USE_NPU)
    std::shared_ptr<NPULayerSynchronizerImpl> layer_synchronizer =
        std::make_shared<NPULayerSynchronizerImpl>(
            runtime_.context->get_model_args().n_layers());
    const_cast<ModelInputParams*>(&(input.input_params))->layer_synchronizer =
        layer_synchronizer;

    futures.emplace_back(
        runtime_.worker.kv_cache_transfer_->push_kv_blocks_async(
            input.transfer_kv_infos,
            runtime_.context->get_parallel_args(),
            layer_synchronizer,
            runtime_.worker.is_spec_draft_));
#endif
  }

  if (FLAGS_enable_eplb) {
    runtime_.eplb_executor->eplb_execute(input.eplb_info);
  }

  // temporarily use [0], will be adapted in next pr
  // call model executor forward to get hidden states
  auto model_output = runtime_.executor->forward(input.token_ids,
                                                 input.positions,
                                                 runtime_.worker.kv_caches_,
                                                 input.input_params);
  if (!model_output.hidden_states.defined()) {
    return std::nullopt;
  }

  torch::Tensor logits;
  if (sampling_params.selected_token_idxes.defined()) {
    logits = runtime_.model->logits(model_output.hidden_states,
                                    sampling_params.selected_token_idxes);
  }

  ForwardOutput output;
  if (FLAGS_enable_eplb) {
    output.expert_load_data = runtime_.expert_load_data;
    output.prepared_layer_id = runtime_.eplb_executor->get_ready_layer_id();
    if (output.prepared_layer_id != -1) {
      runtime_.eplb_executor->reset_ready_layer_id();
    }
  }

  if (!runtime_.worker.driver_ && !runtime_.worker.dp_driver_ &&
      !runtime_.worker.options_.enable_speculative_decode()) {
    auto ret = runtime_.stream->synchronize();
    // in p-d disaggregation scene, all micro batches should be in same
    // prefill/decode stage, so, to judge transfer_kv_infos.empty,
    if (runtime_.worker.options_.kv_cache_transfer_mode() == "PUSH" &&
        !input.transfer_kv_infos.empty()) {
      auto results =
          folly::collectAll(futures).within(std::chrono::seconds(60)).get();
      for (const auto& result : results) {
        // TODO: Add error handling
        if (!result.value()) {
          LOG(ERROR) << "kv_cache_transfer_ failed";
          break;
        }
      }
    }
    if (FLAGS_enable_eplb) {
      return output;
    }
    return std::nullopt;
  }

  // driver prepare model output
  SampleOutput sample_output;
  if (sampling_params.selected_token_idxes.defined()) {
    sample_output = runtime_.worker.sampler_->forward(logits, sampling_params);
    output.logits = logits;

    // beam search kernel
    BeamSearchOutput beam_search_output;
    if (sampling_params.use_beam_search && input.acc_logprob.defined() &&
        input.acc_logprob.numel() > 0) {
      beam_search_output =
          runtime_.worker.beam_searcher_->forward(input.acc_logprob,
                                                  sample_output.top_tokens,
                                                  sample_output.top_logprobs);
    }

    // set sample output to output
    output.sample_output = sample_output;
    // carry over the sampling params
    output.do_sample = sampling_params.do_sample;
    output.logprobs = sampling_params.logprobs;
    output.max_top_logprobs = sampling_params.max_top_logprobs;
    // set beam search output to output
    output.beam_search_output = beam_search_output;
  }

  if (runtime_.worker.options_.enable_speculative_decode()) {
    if (!input.input_params.batch_forward_type.is_decode() &&
        !runtime_.worker.is_spec_draft_) {
      output.sample_output.embeddings = model_output.hidden_states;
    } else if (sampling_params.selected_token_idxes.defined()) {
      auto embeddings = model_output.hidden_states.index_select(
          /*dim=*/0, sampling_params.selected_token_idxes);
      output.sample_output.embeddings = embeddings;
    }
  }

  auto ret = runtime_.stream->synchronize();

  if (runtime_.worker.options_.kv_cache_transfer_mode() == "PUSH" &&
      !input.transfer_kv_infos.empty()) {
    auto results =
        folly::collectAll(futures).within(std::chrono::seconds(60)).get();
    for (const auto& result : results) {
      // TODO: Add error handling
      if (!result.value()) {
        LOG(ERROR) << "kv_cache_transfer_ failed";
        break;
      }
    }
  }

  COUNTER_ADD(execution_latency_seconds_model, timer.elapsed_seconds());
  DeviceMonitor::get_instance().update_active_activation_memory(
      runtime_.worker.device_.index());

  return output;
}

void RecWorkerImpl::LlmRecWorkPipeline::prepare_work_before_execute(
    const ForwardInput& inputs,
    ForwardInput& processed_inputs) {
  RecWorkPipeline::prepare_work_before_execute(inputs, processed_inputs);

  runtime_.worker.prepare_multi_modal_data(processed_inputs);
}

// ============================================================
// OneRecWorkPipeline Implementation
// ============================================================

ForwardInput RecWorkerImpl::OneRecWorkPipeline::prepare_inputs(Batch& batch) {
  ThreadPool* thread_pool =
      runtime_.worker.input_builder_thread_pool_
          ? runtime_.worker.input_builder_thread_pool_.get()
          : nullptr;

  return batch.prepare_rec_forward_input(
      runtime_.worker.options_.num_decoding_tokens(),
      /*min_decoding_batch_size=*/0,
      runtime_.context->get_model_args(),
      thread_pool);
}

std::optional<ForwardOutput> RecWorkerImpl::OneRecWorkPipeline::step(
    const ForwardInput& input) {
  Timer timer;
  runtime_.worker.device_.set_device();

  const auto& sampling_params = input.sampling_params;
  const auto& input_params = input.input_params;

  const auto* onerec_params = input_params.onerec_params();
  CHECK(onerec_params != nullptr) << "OneRec requires rec_params.";

  const OneRecModelInputParams& rec_params = *onerec_params;

  torch::Tensor hidden_states;
  if (rec_params.rec_stage == OneRecModelInputParams::RecStage::PREFILL) {
    if (!rec_params.is_first_prefill) {
      ModelInputParams decoder_params = input_params;
      decoder_params.mutable_onerec_params().is_encoder_forward = false;
      auto model_output = runtime_.executor->forward(input.token_ids,
                                                     input.positions,
                                                     runtime_.worker.kv_caches_,
                                                     decoder_params);
      hidden_states = model_output.hidden_states;
    } else {
      const bool has_sparse_embedding =
          rec_params.encoder_sparse_embedding.defined();
      const bool has_encoder_tokens = rec_params.encoder_token_ids.defined() &&
                                      rec_params.encoder_positions.defined();

      if (!has_sparse_embedding && !has_encoder_tokens) {
        LOG(ERROR) << "OneRec first prefill requires encoder inputs.";
        return std::nullopt;
      }

      ModelInputParams encoder_params = input_params;
      auto& mutable_onerec_params = encoder_params.mutable_onerec_params();
      mutable_onerec_params.is_encoder_forward = true;

      torch::Tensor encoder_tokens;
      if (has_sparse_embedding) {
        mutable_onerec_params.is_hybrid_mode = true;
        encoder_tokens = rec_params.encoder_sparse_embedding;
      } else {
        encoder_tokens = rec_params.encoder_token_ids;
      }

      runtime_.executor->forward(encoder_tokens,
                                 rec_params.encoder_positions,
                                 runtime_.worker.kv_caches_,
                                 encoder_params);

      ModelInputParams decoder_params = input_params;
      decoder_params.mutable_onerec_params().is_encoder_forward = false;
      auto model_output = runtime_.executor->forward(input.token_ids,
                                                     input.positions,
                                                     runtime_.worker.kv_caches_,
                                                     decoder_params);
      hidden_states = model_output.hidden_states;
    }
  } else {
    ModelInputParams decoder_params = input_params;
    decoder_params.mutable_onerec_params().is_encoder_forward = false;
    auto model_output = runtime_.executor->forward(input.token_ids,
                                                   input.positions,
                                                   runtime_.worker.kv_caches_,
                                                   decoder_params);
    hidden_states = model_output.hidden_states;
  }

  if (!hidden_states.defined()) {
    return std::nullopt;
  }

  if (!runtime_.worker.driver_ && !runtime_.worker.dp_driver_ &&
      !runtime_.worker.options_.enable_speculative_decode()) {
    runtime_.stream->synchronize();
    COUNTER_ADD(execution_latency_seconds_model, timer.elapsed_seconds());
    DeviceMonitor::get_instance().update_active_activation_memory(
        runtime_.worker.device_.index());
    return std::nullopt;
  }

  torch::Tensor logits;
  if (sampling_params.selected_token_idxes.defined()) {
    logits = runtime_.model->logits(hidden_states,
                                    sampling_params.selected_token_idxes);
  }

  ForwardOutput output;

  if (sampling_params.selected_token_idxes.defined()) {
    auto sample_output =
        runtime_.worker.sampler_->forward(logits, sampling_params);
    output.logits = logits;
    output.sample_output = sample_output;
    output.do_sample = sampling_params.do_sample;
    output.logprobs = sampling_params.logprobs;
    output.max_top_logprobs = sampling_params.max_top_logprobs;
  }

  runtime_.stream->synchronize();
  COUNTER_ADD(execution_latency_seconds_model, timer.elapsed_seconds());
  DeviceMonitor::get_instance().update_active_activation_memory(
      runtime_.worker.device_.index());

  return output;
}

// ============================================================
// LlmRecWithMmDataWorkPipeline Implementation (qwen3 with embedding)
// ============================================================

void RecWorkerImpl::LlmRecWithMmDataWorkPipeline::prepare_work_before_execute(
    const ForwardInput& inputs,
    ForwardInput& processed_inputs) {
  RecWorkPipeline::prepare_work_before_execute(inputs, processed_inputs);

  if (!inputs.input_params.mm_data.valid()) {
    return;
  }

  torch::Tensor input_embedding;
  torch::Tensor input_tokens_tensor;
  torch::Tensor input_indices_tensor;

  const auto& mm_data = inputs.input_params.mm_data;
  const auto& processed_mm_data = processed_inputs.input_params.mm_data;

  if (auto res = processed_mm_data.get<torch::Tensor>(LLM_REC_INPUT_TOKENS)) {
    input_tokens_tensor = res.value();
  }

  // Input indices are generated on host side.
  if (auto res = mm_data.get<torch::Tensor>(LLM_REC_INPUT_INDICES)) {
    input_indices_tensor = res.value();
  }

  if (auto res =
          processed_mm_data.get<torch::Tensor>(LLM_REC_INPUT_EMBEDDING)) {
    input_embedding = res.value();
  }

  if (input_embedding.defined()) {
    input_embedding = input_embedding.to(runtime_.worker.dtype());
  }

  if (input_indices_tensor.defined()) {
    CHECK(input_tokens_tensor.defined())
        << "LLM_REC_INPUT_TOKENS is required when LLM_REC_INPUT_INDICES is "
           "set.";

#if defined(USE_NPU)
    layer::NpuWordEmbedding npu_word_embedding =
        runtime_.worker.get_npu_word_embedding();
    torch::Tensor input_tokens_embedding =
        npu_word_embedding(input_tokens_tensor, 0);
#else
    layer::WordEmbedding word_embedding = runtime_.worker.get_word_embedding();
    torch::Tensor input_tokens_embedding =
        word_embedding->forward(input_tokens_tensor);
#endif

    if (input_embedding.defined()) {
      torch::Tensor input_indices_cpu =
          input_indices_tensor.to(torch::kCPU).to(torch::kInt64).contiguous();
      const auto* input_indices_ptr = input_indices_cpu.data_ptr<int64_t>();
      std::vector<int64_t> input_indices(
          input_indices_ptr, input_indices_ptr + input_indices_cpu.numel());

      processed_inputs.input_params.input_embedding =
          runtime_.worker.merge_embeddings_by_indices(
              input_tokens_embedding, input_embedding, input_indices);
    } else {
      processed_inputs.input_params.input_embedding = input_tokens_embedding;
    }
  } else if (input_embedding.defined()) {
    processed_inputs.input_params.input_embedding = input_embedding;
  }
}

// ============================================================
// LlmRecMultiRoundPipeline Implementation (qwen3 with embedding)
// ============================================================

RecWorkerImpl::LlmRecMultiRoundPipeline::LlmRecMultiRoundPipeline(
    RecPipelineRuntime& runtime)
    : RecWorkPipeline(runtime),
      rec_sampler_(std::make_unique<RecSampler>(
          RecPipelineType::kLlmRecMultiRoundPipeline)) {
  max_seqs_per_batch_ = runtime_.worker.options_.max_seqs_per_batch();
  max_tokens_per_batch_ = runtime_.worker.options_.max_tokens_per_batch();
  max_token_per_req_ = max_seqs_per_batch_ > 0
                           ? (max_tokens_per_batch_ / max_seqs_per_batch_)
                           : 0;
  beam_width_ = runtime_.worker.options_.beam_width();

  full_kv_cache_offsets_ = std::make_unique<FullKvCacheOffsets>(this);
  allocate_kv_caches_related();
}

ForwardInput RecWorkerImpl::LlmRecMultiRoundPipeline::prepare_inputs(
    Batch& batch) {
  ThreadPool* thread_pool =
      runtime_.worker.input_builder_thread_pool_
          ? runtime_.worker.input_builder_thread_pool_.get()
          : nullptr;

  return batch.prepare_rec_forward_input(
      runtime_.worker.options_.num_decoding_tokens(),
      /*min_decoding_batch_size=*/0,
      runtime_.context->get_model_args(),
      thread_pool);
}

void RecWorkerImpl::LlmRecMultiRoundPipeline::prepare_work_before_execute(
    const ForwardInput& inputs,
    ForwardInput& processed_inputs) {
  RecWorkPipeline::prepare_work_before_execute(inputs, processed_inputs);

  runtime_.worker.prepare_multi_modal_data(processed_inputs);

#if defined(USE_NPU) || defined(USE_CUDA)
  prepare_kv_caches_related_for_input(inputs, processed_inputs);
#endif
}

void RecWorkerImpl::LlmRecMultiRoundPipeline::allocate_kv_caches_related() {
  auto dtype = runtime_.worker.dtype();
  auto device = runtime_.worker.device();
  auto kv_cache_options = torch::TensorOptions().dtype(dtype).device(device);
  auto int_options = torch::TensorOptions().dtype(torch::kInt32).device(device);
  int32_t num_layers = runtime_.context->get_model_args().n_layers();

  int32_t full_kv_len =
      max_tokens_per_batch_ + max_seqs_per_batch_ * beam_width_ *
                                  (get_rec_multi_round_decode_rounds() - 1);
  int64_t num_kv_heads =
      runtime_.context->get_model_args().n_kv_heads().value_or(
          runtime_.context->get_model_args().n_heads());
  int64_t head_dim = runtime_.context->get_model_args().head_dim();

  cached_full_k_caches_.resize(num_layers);
  cached_full_v_caches_.resize(num_layers);

  for (int32_t layer_id = 0; layer_id < num_layers; ++layer_id) {
    auto target_layer_full_k_cache =
        torch::zeros({full_kv_len, num_kv_heads, head_dim}, kv_cache_options);
    auto target_layer_full_v_cache =
        torch::zeros({full_kv_len, num_kv_heads, head_dim}, kv_cache_options);

    cached_full_k_caches_[layer_id] = target_layer_full_k_cache;
    cached_full_v_caches_[layer_id] = target_layer_full_v_cache;
  }

  cached_naive_block_table_ =
      torch::arange(max_seqs_per_batch_ * beam_width_, int_options)
          .unsqueeze(1);
  cached_current_round_tensor_ = torch::zeros({1}, int_options);
}

void RecWorkerImpl::LlmRecMultiRoundPipeline::
    prepare_kv_caches_related_for_input(const ForwardInput& inputs,
                                        ForwardInput& processed_inputs) {
  auto device = runtime_.worker.device();
  auto int_options = torch::TensorOptions().dtype(torch::kInt32).device(device);
  auto& input_params = processed_inputs.input_params;
  auto& llm_rec_params = input_params.mutable_llmrec_params();

  const auto* step_meta = inputs.step_meta();
  CHECK(step_meta != nullptr)
      << "step_meta is required for rec multi-round mode";
  int32_t batch_size = step_meta->batch_size;
  int32_t beam_width = step_meta->beam_width;
  int32_t total_round = step_meta->total_round;
  llm_rec_params.batch_size = batch_size;
  llm_rec_params.beam_width = beam_width;
  const auto& shape = step_meta->full_kv_shape;
  CHECK(shape.size() == 3) << "the dims of full_kv_shape should be three.";
  int32_t full_kv_len = shape[0];
  int64_t num_kv_heads = shape[1];
  int64_t head_dim = shape[2];
  int32_t num_layers = runtime_.context->get_model_args().n_layers();
  int32_t max_decode_step = total_round - 1;
  int32_t unshared_offset = max_tokens_per_batch_;

  if (!cached_full_k_caches_.empty() && cached_full_k_caches_[0].defined()) {
    llm_rec_params.full_k_caches.reserve(num_layers);
    llm_rec_params.full_v_caches.reserve(num_layers);
    llm_rec_params.unshared_k_caches.reserve(num_layers);
    llm_rec_params.unshared_v_caches.reserve(num_layers);

    for (int32_t layer_id = 0; layer_id < num_layers; ++layer_id) {
      auto layer_full_k_cache = cached_full_k_caches_[layer_id];
      auto layer_full_v_cache = cached_full_v_caches_[layer_id];

      auto layer_unshared_k_cache =
          layer_full_k_cache.slice(0, unshared_offset, full_kv_len);
      auto layer_unshared_v_cache =
          layer_full_v_cache.slice(0, unshared_offset, full_kv_len);

      layer_unshared_k_cache =
          layer_unshared_k_cache
              .view({static_cast<int64_t>(max_seqs_per_batch_),
                     static_cast<int64_t>(beam_width_),
                     static_cast<int64_t>(max_decode_step),
                     num_kv_heads,
                     head_dim})
              .slice(0, 0, batch_size);
      layer_unshared_v_cache =
          layer_unshared_v_cache
              .view({static_cast<int64_t>(max_seqs_per_batch_),
                     static_cast<int64_t>(beam_width_),
                     static_cast<int64_t>(max_decode_step),
                     num_kv_heads,
                     head_dim})
              .slice(0, 0, batch_size);

      llm_rec_params.full_k_caches.emplace_back(layer_full_k_cache);
      llm_rec_params.full_v_caches.emplace_back(layer_full_v_cache);
      llm_rec_params.unshared_k_caches.emplace_back(layer_unshared_k_cache);
      llm_rec_params.unshared_v_caches.emplace_back(layer_unshared_v_cache);
    }
  }

  input_params.block_tables =
      cached_naive_block_table_.slice(0, 0, batch_size * beam_width);

  const auto& decode_positions = step_meta->decode_positions_vec;
  llm_rec_params.decode_positions_tensor_list.clear();
  if (!decode_positions.empty() && beam_width > 0 && total_round > 1) {
    const int32_t num_sequences = static_cast<int32_t>(decode_positions.size());
    for (int32_t round_idx = 0; round_idx < total_round - 1; ++round_idx) {
      std::vector<int32_t> position_buffer;
      position_buffer.reserve(static_cast<size_t>(num_sequences * beam_width));
      for (int32_t seq_idx = 0; seq_idx < num_sequences; ++seq_idx) {
        const int32_t base_position = decode_positions[seq_idx] + round_idx;
        for (int32_t beam_idx = 0; beam_idx < beam_width; ++beam_idx) {
          position_buffer.push_back(base_position);
        }
      }
      llm_rec_params.decode_positions_tensor_list.push_back(
          torch::tensor(position_buffer, int_options));
    }
  }
}

std::optional<ForwardOutput> RecWorkerImpl::LlmRecMultiRoundPipeline::step(
    const ForwardInput& input) {
  Timer timer;
  auto device = runtime_.worker.device_;
  device.set_device();

  ForwardInput& mutable_input = const_cast<ForwardInput&>(input);

  const auto* step_meta = mutable_input.step_meta();
  CHECK(step_meta != nullptr)
      << "step_meta is required for rec multi-round mode";
  int32_t total_rounds = step_meta->total_round;
  int32_t max_decode_step = total_rounds - 1;
  int32_t batch_size = step_meta->batch_size;
  int32_t beam_width = step_meta->beam_width;
  int32_t num_layers =
      static_cast<int32_t>(runtime_.context->get_model_args().n_layers());

  CHECK_GT(runtime_.worker.kv_caches_.size(), 0)
      << "KV caches are not initialized.";

  BeamSearchTensors beam_tensors =
      prepare_beam_search_tensors(batch_size, beam_width, total_rounds, device);

  ForwardOutput output;
  torch::Tensor logits;
  SampleOutput sample_output;
  torch::Tensor top_tokens;
  torch::Tensor top_logprobs;
  std::optional<folly::SemiFuture<NextRoundInputResults>>
      next_round_async_result;

  for (int32_t round = 0; round < total_rounds; ++round) {
    const auto& sampling_params = round > 0
                                      ? mutable_input.decoder_sampling_params
                                      : mutable_input.sampling_params;

    // Consume async result for current round and schedule next round's async
    // computation.
    prepare_round_input_and_schedule_next(mutable_input,
                                          round,
                                          total_rounds,
                                          batch_size,
                                          beam_width,
                                          max_decode_step,
                                          top_tokens,
                                          beam_tensors,
                                          next_round_async_result);

    auto model_output = runtime_.executor->forward(mutable_input.token_ids,
                                                   mutable_input.positions,
                                                   runtime_.worker.kv_caches_,
                                                   mutable_input.input_params);
    if (!model_output.hidden_states.defined()) {
      return std::nullopt;
    }
    torch::Tensor hidden_states = model_output.hidden_states;

    if (sampling_params.selected_token_idxes.defined()) {
      logits = runtime_.model->logits(hidden_states,
                                      sampling_params.selected_token_idxes);
      sample_output = rec_sampler_->forward(logits, sampling_params);
    }

    if (sample_output.top_tokens.defined() &&
        sample_output.top_logprobs.defined()) {
      int64_t top_tokens_numel = sample_output.top_tokens.numel();
      int64_t top_logprobs_numel = sample_output.top_logprobs.numel();
      CHECK_EQ(top_tokens_numel % beam_width, 0)
          << "top_tokens numel (" << top_tokens_numel
          << ") must be divisible by beam_width (" << step_meta->beam_width
          << ")";
      CHECK_EQ(top_logprobs_numel % beam_width, 0)
          << "top_logprobs numel (" << top_logprobs_numel
          << ") must be divisible by beam_width (" << step_meta->beam_width
          << ")";

      top_tokens = sample_output.top_tokens.to(torch::kInt32)
                       .reshape({-1, step_meta->beam_width});
      top_logprobs = sample_output.top_logprobs.reshape({-1, beam_width});
      execute_beam_search(
          top_tokens, top_logprobs, beam_tensors, round, batch_size);

      if (round > 0 && round < total_rounds - 1) {
        execute_cache_select(
            beam_tensors, mutable_input, round, beam_width, num_layers);
      }

      if (round == total_rounds - 1) {
        build_final_output(
            logits, sample_output, sampling_params, beam_tensors, output);
      }
    }
  }

  runtime_.stream->synchronize();

  COUNTER_ADD(execution_latency_seconds_model, timer.elapsed_seconds());
  DeviceMonitor::get_instance().update_active_activation_memory(device.index());
  return output;
}

RecWorkerImpl::LlmRecMultiRoundPipeline::BeamSearchTensors
RecWorkerImpl::LlmRecMultiRoundPipeline::prepare_beam_search_tensors(
    int32_t batch_size,
    int32_t beam_width,
    int32_t total_rounds,
    const torch::Device& device) {
  auto int_options = torch::TensorOptions().dtype(torch::kInt32).device(device);
  auto fp32_options =
      torch::TensorOptions().dtype(torch::kFloat32).device(device);

  BeamSearchTensors tensors;
  tensors.sequence_group =
      torch::zeros({batch_size, beam_width, total_rounds}, int_options);
  int64_t num_seq = batch_size * beam_width;
  tensors.acc_logprob = torch::zeros({num_seq, 1}, fp32_options);
  tensors.out_log_probs = torch::zeros({num_seq, 1}, fp32_options);
  tensors.out_token_ids = torch::zeros({num_seq, 1}, int_options);
  tensors.out_token_index = torch::zeros({num_seq, 1}, int_options);
  tensors.out_beam_count_prefix_sums = torch::zeros({num_seq, 1}, int_options);
  tensors.out_seqgroup = torch::zeros_like(tensors.sequence_group);
  return tensors;
}

void RecWorkerImpl::LlmRecMultiRoundPipeline::execute_beam_search(
    const torch::Tensor& top_tokens,
    const torch::Tensor& top_logprobs,
    BeamSearchTensors& beam_tensors,
    int32_t round,
    int32_t batch_size) {
#if defined(USE_NPU)
// TODO: implement beam search for NPU
#elif defined(USE_CUDA)
  xllm::kernel::cuda::beam_search(beam_tensors.acc_logprob,
                                  beam_tensors.sequence_group,
                                  top_tokens,
                                  top_logprobs,
                                  beam_tensors.out_log_probs,
                                  beam_tensors.out_token_ids,
                                  beam_tensors.out_token_index,
                                  beam_tensors.out_beam_count_prefix_sums,
                                  beam_tensors.out_seqgroup,
                                  batch_size,
                                  round);
#endif
  std::swap(beam_tensors.sequence_group, beam_tensors.out_seqgroup);
  std::swap(beam_tensors.acc_logprob, beam_tensors.out_log_probs);
}

void RecWorkerImpl::LlmRecMultiRoundPipeline::execute_cache_select(
    const BeamSearchTensors& beam_tensors,
    ForwardInput& input,
    int32_t round,
    int32_t beam_width,
    int32_t num_layers) {
#if defined(USE_NPU)
// TODO: implement cache select for NPU
#elif defined(USE_CUDA)
  xllm::kernel::cuda::cache_select(
      beam_tensors.out_token_index,
      input.input_params.mutable_llmrec_params().unshared_k_caches,
      input.input_params.mutable_llmrec_params().unshared_v_caches,
      input.input_params.block_tables,
      round - 1,
      beam_width,
      num_layers);
#endif
}

void RecWorkerImpl::LlmRecMultiRoundPipeline::build_final_output(
    const torch::Tensor& logits,
    const SampleOutput& sample_output,
    const SamplingParameters& sampling_params,
    const BeamSearchTensors& beam_tensors,
    ForwardOutput& output) {
  output.logits = logits;
  output.sample_output = sample_output;
  output.do_sample = sampling_params.do_sample;
  output.logprobs = sampling_params.logprobs;
  output.max_top_logprobs = sampling_params.max_top_logprobs;
  output.beam_search_output.src_seq_idxes =
      beam_tensors.out_token_index.reshape({-1});
  output.beam_search_output.out_tokens =
      beam_tensors.out_token_ids.reshape({-1});
  output.beam_search_output.out_logprobs =
      beam_tensors.acc_logprob.reshape({-1});
  output.beam_sequence_group = beam_tensors.sequence_group;
}

void RecWorkerImpl::LlmRecMultiRoundPipeline::prepare_input_for_current_round(
    ForwardInput& input,
    const NextRoundInputResults& results,
    int32_t round,
    const torch::Tensor& top_tokens,
    const BeamSearchTensors& beam_tensors) {
#if defined(USE_NPU)
// TODO: implement prepare_input_for_current_round for NPU
#elif defined(USE_CUDA)
  input.input_params.paged_kv_indices = results.paged_kv_indices;
  input.input_params.paged_kv_indptr = results.paged_kv_indptr;
  input.input_params.paged_kv_last_page_len = results.paged_kv_last_page_len;
  input.input_params.num_sequences =
      input.input_params.paged_kv_last_page_len.numel();
#endif
  // previous_step corresponds to the decode step that produced tokens for
  // this round.
  const int32_t previous_step = round - 1;
  if (previous_step == 0) {
    // First decode step uses top_tokens from prefill.
    if (top_tokens.defined()) {
      input.token_ids = top_tokens.reshape({-1});
    }
  } else if (previous_step > 0) {
    // Later steps use beam search output tokens.
    input.token_ids = beam_tensors.out_token_ids.reshape({-1});
  }

  auto& llm_rec_params = input.input_params.mutable_llmrec_params();
  if (!llm_rec_params.decode_positions_tensor_list.empty() &&
      previous_step >= 0 &&
      previous_step < static_cast<int32_t>(
                          llm_rec_params.decode_positions_tensor_list.size())) {
    input.positions =
        llm_rec_params.decode_positions_tensor_list[previous_step];
  }

  input.input_params.batch_forward_type = BatchForwardType(2);
  input.input_params.input_embedding = torch::Tensor();
  cached_current_round_tensor_.fill_(previous_step);
  llm_rec_params.current_round_tensor = cached_current_round_tensor_;
  input.input_params.attn_metadata = nullptr;
}

folly::SemiFuture<
    RecWorkerImpl::LlmRecMultiRoundPipeline::NextRoundInputResults>
RecWorkerImpl::LlmRecMultiRoundPipeline::compute_next_round_input_async(
    const torch::Tensor& kv_seq_lens,
    int32_t current_step,
    int32_t batch_size,
    int32_t beam_width,
    int32_t max_decode_step) {
  folly::Promise<NextRoundInputResults> promise;
  auto future = promise.getSemiFuture();

#if defined(USE_NPU)
// TODO: implement compute_next_round_input_async for NPU
#elif defined(USE_CUDA)
  // Capture necessary data for async computation
  auto full_kv_offsets = full_kv_cache_offsets_->full_kv_offsets;
  auto full_kv_mask = full_kv_cache_offsets_->full_kv_mask;
  auto full_kv_indices = full_kv_cache_offsets_->full_kv_indices;
  auto unshared_full_kv_offsets = full_kv_cache_offsets_->unshared_offsets;
  auto real_max_decode_step_ids = full_kv_cache_offsets_->max_decode_step_ids;
  uint32_t unshared_kv_begin_offset = max_tokens_per_batch_;

  // Launch async computation in thread pool (can overlap with GPU execution)
  threadpool_.schedule([=, this, promise = std::move(promise)]() mutable {
    auto device = runtime_.worker.device();
    auto int32_device_options =
        torch::TensorOptions().dtype(torch::kInt32).device(device);
    // Protect CUDA graph capture from conflicting GPU work submitted on
    // prepare_stream_ while capture is in progress. Use shared lock to allow
    // multiple prepare operations to run concurrently, but prevent conflicts
    // with capture operations. This mirrors the NPU DeviceCaptureLock usage in
    // WorkerImpl::prepare_work_before_execute.
    std::optional<std::shared_lock<std::shared_mutex>> lock_guard;
    if (runtime_.worker.options_.enable_graph()) {
      auto& replay_lock =
          ::xllm::cuda::DeviceCaptureLock::get_instance().get_read_lock(
              runtime_.worker.device_.index());
      lock_guard.emplace(replay_lock);
    }

    c10::StreamGuard streamGuard =
        runtime_.worker.prepare_stream_->set_stream_guard();
    auto shared_kv_offsets =
        full_kv_offsets.slice(2, 0, max_token_per_req_).slice(0, 0, batch_size);

    auto shared_kv_lens_each_batch = torch::diff(kv_seq_lens);

    auto shared_kv_lens_each_batch_broadcast =
        shared_kv_lens_each_batch.unsqueeze(1).unsqueeze(1);

    auto shared_mask =
        full_kv_mask.slice(2, 0, max_token_per_req_).slice(0, 0, batch_size);

    shared_mask.copy_(shared_kv_offsets < shared_kv_lens_each_batch_broadcast);

    auto kv_lens_batch_offsets = kv_seq_lens.slice(0, 0, -1);

    auto kv_lens_batch_offsets_broadcast =
        kv_lens_batch_offsets.unsqueeze(1).unsqueeze(1);

    auto shared_kv_indices =
        full_kv_indices.slice(2, 0, max_token_per_req_).slice(0, 0, batch_size);

    shared_kv_indices.copy_(kv_lens_batch_offsets_broadcast +
                            shared_kv_offsets);

    auto unshared_kv_offsets = unshared_full_kv_offsets.slice(0, 0, batch_size);
    int32_t unshared_kv_len = beam_width * max_decode_step;
    auto unshared_kv_indices =
        full_kv_indices
            .slice(2, max_token_per_req_, max_token_per_req_ + unshared_kv_len)
            .slice(0, 0, batch_size);
    unshared_kv_indices.copy_(unshared_kv_offsets + unshared_kv_begin_offset);

    auto unshared_mask =
        full_kv_mask
            .slice(2, max_token_per_req_, max_token_per_req_ + unshared_kv_len)
            .slice(0, 0, batch_size);
    auto real_max_decode_step_ids_slice =
        real_max_decode_step_ids.slice(0, 0, batch_size);
    unshared_mask.copy_(real_max_decode_step_ids_slice <= current_step);

    unshared_kv_len = current_step + 1;

    auto batch_beam_shared_kv_lens =
        (shared_kv_lens_each_batch.unsqueeze(1).expand({-1, beam_width}) +
         unshared_kv_len)
            .flatten();
    auto cumsum_result = torch::cumsum(batch_beam_shared_kv_lens, 0);
    auto paged_kv_indptr = torch::cat({torch::zeros({1}, int32_device_options),
                                       cumsum_result.to(int32_device_options)},
                                      0);
    auto paged_kv_indices = full_kv_indices.masked_select(full_kv_mask);
    auto paged_kv_last_page_len =
        torch::ones({batch_size * beam_width}, int32_device_options);
    runtime_.worker.prepare_stream_->synchronize();

    NextRoundInputResults results;
    results.paged_kv_indices = paged_kv_indices;
    results.paged_kv_indptr = paged_kv_indptr;
    results.paged_kv_last_page_len = paged_kv_last_page_len;
    promise.setValue(results);
  });
#endif
  return future;
}

void RecWorkerImpl::LlmRecMultiRoundPipeline::
    prepare_round_input_and_schedule_next(
        ForwardInput& input,
        int32_t round,
        int32_t total_rounds,
        int32_t batch_size,
        int32_t beam_width,
        int32_t max_decode_step,
        const torch::Tensor& top_tokens,
        const BeamSearchTensors& beam_tensors,
        std::optional<folly::SemiFuture<NextRoundInputResults>>&
            next_round_async_result) {
  // Phase A: consume async result for the current round (prepared in last
  // round).
  if (next_round_async_result.has_value()) {
    auto results = std::move(next_round_async_result.value()).get();
    prepare_input_for_current_round(
        input, results, round, top_tokens, beam_tensors);

    // Ensure this future is not consumed twice.
    next_round_async_result.reset();
  }

  // Phase B: schedule async computation for the next round, if any.
  if (round < total_rounds - 1) {
    next_round_async_result =
        compute_next_round_input_async(input.input_params.kv_seq_lens,
                                       round,
                                       batch_size,
                                       beam_width,
                                       max_decode_step);
  }
}

RecWorkerImpl::LlmRecMultiRoundPipeline::FullKvCacheOffsets::FullKvCacheOffsets(
    LlmRecMultiRoundPipeline* multi_round_pipeline) {
#if defined(USE_NPU)
// TODO: implement FullKvCacheOffsets for NPU
#elif defined(USE_CUDA)
  auto device = multi_round_pipeline->runtime().worker.device();
  auto int32_device_options =
      torch::TensorOptions().dtype(torch::kInt32).device(device);
  int32_t max_decode_step = get_rec_multi_round_decode_rounds() - 1;
  full_kv_offsets =
      torch::arange(0,
                    multi_round_pipeline->max_token_per_req_ + max_decode_step,
                    int32_device_options)
          .unsqueeze(0)
          .expand({multi_round_pipeline->max_seqs_per_batch_, -1})
          .unsqueeze(1)
          .expand({-1, multi_round_pipeline->beam_width_, -1});
  full_kv_mask =
      torch::zeros({multi_round_pipeline->max_seqs_per_batch_,
                    multi_round_pipeline->beam_width_,
                    multi_round_pipeline->max_token_per_req_ + max_decode_step},
                   int32_device_options)
          .to(torch::kBool);
  full_kv_indices = torch::zeros_like(full_kv_offsets);

  auto batch_ids =
      torch::arange(
          0, multi_round_pipeline->max_seqs_per_batch_, int32_device_options)
          .unsqueeze(1)
          .unsqueeze(2)
          .expand({-1, multi_round_pipeline->beam_width_, max_decode_step}) *
      (multi_round_pipeline->beam_width_ * max_decode_step);

  auto beams_ids =
      torch::arange(0, multi_round_pipeline->beam_width_, int32_device_options)
          .unsqueeze(0)
          .unsqueeze(2)
          .expand({multi_round_pipeline->max_seqs_per_batch_,
                   -1,
                   max_decode_step}) *
      max_decode_step;

  max_decode_step_ids = torch::arange(0, max_decode_step, int32_device_options)
                            .unsqueeze(0)
                            .unsqueeze(1)
                            .expand({multi_round_pipeline->max_seqs_per_batch_,
                                     multi_round_pipeline->beam_width_,
                                     -1});
  unshared_offsets = batch_ids + beams_ids + max_decode_step_ids;
#endif
}

// ============================================================
// RecWorkerImpl Implementation
// ============================================================

RecWorkerImpl::RecWorkerImpl(const ParallelArgs& parallel_args,
                             const torch::Device& device,
                             const runtime::Options& options)
    : LLMWorkerImpl(parallel_args, device, options) {
  if (!is_driver()) {
    return;
  }

  step_threadpool_ = std::make_unique<ThreadPool>(
      options_.rec_worker_max_concurrency(), [this]() mutable {
        device_.set_device();
#if defined(USE_CUDA)
        ::xllm::layer::flashinfer::FlashinferWorkspace::get_instance()
            .initialize(device_);
#endif
      });

  LOG(INFO) << "RecWorkerImpl constructor: "
            << options_.rec_worker_max_concurrency();
  const int64_t num_threads = std::max<int64_t>(
      1, util::get_int_env("XLLM_REC_INPUT_BUILDER_THREADS", 16));
  input_builder_thread_pool_ =
      std::make_shared<ThreadPool>(static_cast<size_t>(num_threads));
}

RecWorkerImpl::~RecWorkerImpl() {
  // Release model_, model_executor_, eplb_executor_ in destructor to avoid
  // double deletion. Ownership actually belongs to work_pipelines_[0].
  model_.release();
  model_executor_.release();

  if (FLAGS_enable_eplb) {
    eplb_executor_.release();
  }
}

bool RecWorkerImpl::init_model(const std::string& model_weights_path,
                               int32_t random_seed) {
  if (!WorkerImpl::init_model(model_weights_path, random_seed)) {
    return false;
  }

  if (FLAGS_enable_eplb) {
    work_pipelines_[0]->runtime().expert_load_data = expert_load_data_;

    for (size_t i = 1; i < work_pipelines_.size(); ++i) {
      work_pipelines_[i]->runtime().expert_load_data =
          work_pipelines_[0]->runtime().expert_load_data.clone();
    }
  }

  return true;
}

bool RecWorkerImpl::init_model(ModelContext& context) {
  CHECK(model_ == nullptr) << "Model is already initialized.";

  // Determine rec model kind and pipeline type
  const auto& model_type = context.get_model_args().model_type();
  rec_model_kind_ = get_rec_model_kind(model_type);
  CHECK(rec_model_kind_ != RecModelKind::kNone)
      << "Unsupported rec model_type: " << model_type;

  // Create concurrent pipeline (not base class pipeline)
  auto pipeline_type = get_rec_pipeline_type(rec_model_kind_);

  // Reserve space for model instances
  work_pipelines_.reserve(options_.rec_worker_max_concurrency());
  for (size_t i = 0; i < options_.rec_worker_max_concurrency(); ++i) {
    RecPipelineRuntime runtime(*this);
    auto stream = device_.get_stream_from_pool();
    runtime.stream = std::move(stream);
    auto stream_guard = runtime.stream->set_stream_guard();

    runtime.context =
        std::make_unique<ModelContext>(context.get_parallel_args(),
                                       context.get_model_args(),
                                       context.get_quant_args(),
                                       context.get_tensor_options());

    runtime.model = create_llm_model(*runtime.context.get());
    CHECK(runtime.model != nullptr) << "Failed to create model instance " << i;

    runtime.executor =
        std::make_unique<Executor>(runtime.model.get(),
                                   runtime.context->get_model_args(),
                                   runtime.worker.device(),
                                   runtime.worker.options_);

    if (FLAGS_enable_eplb) {
      runtime.eplb_executor = std::make_unique<EplbExecutor>(
          runtime.model.get(), runtime.worker.device());
    }

    work_pipelines_.emplace_back(create_pipeline(pipeline_type, runtime));
    index_queue_.enqueue(i);
  }

  model_.reset(work_pipelines_[0]->runtime().model.get());
  model_executor_.reset(work_pipelines_[0]->runtime().executor.get());

  // Complete other initialization (EPLB, BeamSearcher, etc.)
  if (FLAGS_enable_beam_search_kernel) {
    beam_searcher_ = std::make_unique<BeamSearcher>();
  }

  if (FLAGS_enable_eplb) {
    eplb_executor_.reset(work_pipelines_[0]->runtime().eplb_executor.get());
  }

  LOG(INFO) << "Created " << work_pipelines_.size()
            << " pipelines for concurrent execution";
  return true;
}

void RecWorkerImpl::load_model(std::unique_ptr<ModelLoader> loader) {
  CHECK(!work_pipelines_.empty())
      << "Model instances are not initialized. Call init_model() first.";

  // Save model weights path to create new loaders for other instances
  std::string model_weights_path = loader->model_weights_path();

  // Load weights for the first model instance (using the original loader)
  work_pipelines_[0]->runtime().model->load_model(std::move(loader));
  LOG(INFO) << "Loaded weights for model instance 0";

  // Create new loaders and load weights for other model instances
  for (size_t i = 1; i < work_pipelines_.size(); ++i) {
    auto model_loader = ModelLoader::create(model_weights_path);
    CHECK(model_loader != nullptr)
        << "Failed to create ModelLoader for model instance " << i;
    work_pipelines_[i]->runtime().model->load_model(std::move(model_loader));
    LOG(INFO) << "Loaded weights for model instance " << i;
  }

  LOG(INFO) << "Loaded weights for all " << work_pipelines_.size() << " models";
}

ForwardInput RecWorkerImpl::prepare_inputs(Batch& batch) {
  CHECK(!work_pipelines_.empty()) << "RecWorkerImpl is not initialized.";
  return work_pipelines_[0]->prepare_inputs(batch);
}

void RecWorkerImpl::prepare_work_before_execute(
    const ForwardInput& inputs,
    ForwardInput& processed_inputs) {
  LOG(FATAL)
      << "RecWorkerImpl::prepare_work_before_execute should not be called.";
}

void RecWorkerImpl::prepare_multi_modal_data(ForwardInput& processed_inputs) {
  if (!processed_inputs.input_params.mm_data.valid()) {
    return;
  }

  torch::Tensor multi_modal_values;
  torch::Tensor multi_modal_indices;

  const auto& processed_mm_data = processed_inputs.input_params.mm_data;
  if (auto res = processed_mm_data.get<torch::Tensor>("MULTI_MODAL_VALUES")) {
    multi_modal_values = res.value();
  }

  if (auto res = processed_mm_data.get<torch::Tensor>("MULTI_MODAL_INDICES")) {
    multi_modal_indices = res.value();
  }

  if (!multi_modal_values.defined() || !multi_modal_indices.defined()) {
    return;
  }

#if defined(USE_NPU)
  layer::NpuWordEmbedding npu_word_embedding = get_npu_word_embedding();
  torch::Tensor input_tokens_embedding =
      npu_word_embedding(processed_inputs.token_ids, 0);
#else
  layer::WordEmbedding word_embedding = get_word_embedding();
  torch::Tensor input_tokens_embedding =
      word_embedding->forward(processed_inputs.token_ids);
#endif

  std::vector<torch::indexing::TensorIndex> indices = {
      torch::indexing::TensorIndex(multi_modal_indices),
      torch::indexing::Slice()};

  input_tokens_embedding.index_put_(indices, multi_modal_values);
  processed_inputs.input_params.input_embedding = input_tokens_embedding;
}

std::optional<ForwardOutput> RecWorkerImpl::step(const ForwardInput& input) {
  LOG(FATAL) << "RecWorkerImpl::step should not be called.";
  return std::nullopt;
}

folly::SemiFuture<std::optional<ForwardOutput>> RecWorkerImpl::step_async(
    const ForwardInput& input) {
  folly::Promise<std::optional<ForwardOutput>> promise;

  size_t index;
  index_queue_.wait_dequeue(index);
  auto future = promise.getSemiFuture();

  // Use schedule() to assign tasks, letting ThreadPool automatically select
  // idle threads The logic for allocating instance_id happens when the task
  // executes (see lambda below)
  step_threadpool_->schedule_with_tid(
      [this, &input, index, promise = std::move(promise)]() mutable {
        auto stream_guard =
            work_pipelines_[index]->runtime().stream->set_stream_guard();

        ForwardInput input_on_device;
        work_pipelines_[index]->prepare_work_before_execute(input,
                                                            input_on_device);

        if (hierarchy_kv_cache_transfer_ != nullptr) {
          hierarchy_kv_cache_transfer_->set_layer_synchronizer(
              input_on_device.input_params);
        }

        const auto output = work_pipelines_[index]->step(input_on_device);
        promise.setValue(output);

        index_queue_.enqueue(index);
      },
      index);

  return future;
}

// ============================================================
// RecWorkerImpl pipeline factory (static method)
// ============================================================

std::unique_ptr<RecWorkerImpl::RecWorkPipeline> RecWorkerImpl::create_pipeline(
    RecPipelineType type,
    RecPipelineRuntime& runtime) {
  switch (type) {
    case RecPipelineType::kLlmRecDefault:
      return std::make_unique<LlmRecWorkPipeline>(runtime);
    case RecPipelineType::kOneRecDefault:
      return std::make_unique<OneRecWorkPipeline>(runtime);
    case RecPipelineType::kLlmRecMultiRoundPipeline:
      return std::make_unique<LlmRecMultiRoundPipeline>(runtime);
    default:
      LOG(FATAL) << "Unknown RecWorkerImpl pipeline type: "
                 << static_cast<int>(type);
      return nullptr;
  }
}

torch::Tensor RecWorkerImpl::merge_embeddings_by_indices(
    const torch::Tensor& input_tokens_embedding,
    const torch::Tensor& input_embedding,
    const std::vector<int64_t>& input_indices) {
  CHECK_EQ(input_embedding.dim(), 2);
  CHECK_EQ(input_tokens_embedding.dim(), 2);
  CHECK_EQ(input_tokens_embedding.size(1), input_embedding.size(1));
  CHECK_EQ(input_tokens_embedding.dtype(), input_embedding.dtype());
  CHECK_EQ(input_tokens_embedding.device(), input_embedding.device());

  const int64_t total_rows =
      input_tokens_embedding.size(0) + input_embedding.size(0);
  const int64_t cols = input_embedding.size(1);

  torch::Device device = input_embedding.device();
  torch::Tensor merged = torch::empty(
      {total_rows, cols}, torch::dtype(input_embedding.dtype()).device(device));

  std::vector<int64_t> input_embedding_indices;
  for (int64_t i = 0; i < total_rows; ++i) {
    if (std::find(input_indices.begin(), input_indices.end(), i) ==
        input_indices.end()) {
      input_embedding_indices.push_back(i);
    }
  }

  CHECK_EQ(input_embedding_indices.size(), input_embedding.size(0));

  torch::Tensor input_embedding_indices_tensor =
      torch::tensor(input_embedding_indices, torch::kInt64).to(device);
  merged.index_put_({input_embedding_indices_tensor, torch::indexing::Ellipsis},
                    input_embedding);

  torch::Tensor input_indices_tensor =
      torch::tensor(input_indices, torch::kInt64).to(device);
  merged.index_put_({input_indices_tensor, torch::indexing::Ellipsis},
                    input_tokens_embedding);

  return merged;
}

}  // namespace xllm

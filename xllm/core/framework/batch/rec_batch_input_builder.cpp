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

#include "rec_batch_input_builder.h"

#include <algorithm>
#include <cstring>
#include <future>
#include <numeric>
#include <thread>
#include <vector>

#include "framework/model/model_args.h"
#include "framework/model/model_input_params.h"
#include "framework/request/sequence.h"
#include "framework/sampling/sampling_params.h"
#include "util/tensor_helper.h"
#include "util/threadpool.h"
#include "util/utils.h"

namespace xllm {

// Static member definition
RecBatchInputBuilder::HighPerformanceCache RecBatchInputBuilder::perf_cache_;

RecBatchInputBuilder::RecBatchInputBuilder(
    const std::vector<std::unique_ptr<SequencesGroup>>& sequence_groups,
    const std::vector<uint32_t>& allowed_max_tokens,
    const std::vector<torch::Tensor>& input_embeddings_vec,
    const std::vector<MMData>& mm_data_vec,
    const std::vector<CacheBlockInfo>* copy_in_cache_block_infos,
    const std::vector<CacheBlockInfo>* copy_out_cache_block_infos,
    std::vector<CacheBlockInfo>* swap_cache_block_infos,
    const ModelArgs* args,
    ThreadPool* thread_pool)
    : BatchInputBuilder(  // extract_sequences_from_groups(sequence_groups),
          {},
          allowed_max_tokens,
          input_embeddings_vec,
          mm_data_vec,
          copy_in_cache_block_infos,
          copy_out_cache_block_infos,
          swap_cache_block_infos,
          args,
          thread_pool),
      sequence_groups_(sequence_groups) {
  // Reset high performance cache
  perf_cache_.memory_pool.reset();
}

std::vector<Sequence*> RecBatchInputBuilder::extract_sequences_from_groups(
    const std::vector<std::unique_ptr<SequencesGroup>>& sequence_groups) {
  std::vector<Sequence*> sequences;
  for (const auto& group : sequence_groups) {
    for (const auto& seq : group->sequences()) {
      sequences.push_back(seq.get());
    }
  }
  return sequences;
}

ForwardInput RecBatchInputBuilder::build_rec_forward_input(
    uint32_t num_decoding_tokens,
    uint32_t min_decoding_batch_size) {
  // ========== Global constant cache ==========
  static const std::vector<int32_t> FIXED_POSITIONS = {0};
  static const torch::Tensor FIXED_ENCODER_POSITIONS =
      torch::tensor({0}, torch::kInt);

  // ========== Fast sequence information extraction ==========
  const int32_t num_sequences =
      !sequence_groups_.empty()
          ? std::accumulate(sequence_groups_.begin(),
                            sequence_groups_.end(),
                            0,
                            [](int sum, const auto& group) {
                              return sum + group->sequences().size();
                            })
          : 0;
  const int32_t THREADPOOL_THRESHOLD = 16;
  if (UNLIKELY(num_sequences == 0)) {
    return ForwardInput{};
  }

  // Get basic information of first sequence - optimize pointer access
  Sequence* first_sequence = nullptr;
  if (!sequence_groups_.empty() && !sequence_groups_[0]->sequences().empty()) {
    first_sequence = sequence_groups_[0]->sequences()[0].get();
  }

  if (!first_sequence) {
    return ForwardInput{};
  }

  const uint32_t seq_len = first_sequence->num_tokens();
  const uint32_t num_decoder_embeddings =
      first_sequence->num_decoder_embeddings();
  const uint32_t n_prompt_tokens = first_sequence->num_prompt_tokens();
  const bool is_first_prefill = (first_sequence->num_generated_tokens() == 0);
  // const uint64_t model_version = first_sequence->get_model_version();

  // ========== High-performance encoder tokens construction ==========
  auto buildEncoderTokensOptimized = [&]() -> const std::vector<int32_t>& {
    auto& cache_data = perf_cache_.cache_data;

    // encoder doesn't use cache key, because encoder doesn't use encoder_tokens
    // in non-first prefill scenarios, only uses encoder_seq_len
    if (!is_first_prefill) {
      return cache_data.encoder_tokens;
    }

    // Optimization: Use SIMD-friendly memory access patterns
    cache_data.encoder_tokens.clear();
    cache_data.encoder_seq_lens.clear();

    // Optimization for scenarios where sequences have different lengths across
    // sequence groups Pre-calculate total token count to avoid multiple memory
    // reallocations
    int32_t total_tokens = 0;
    for (const auto& group_ptr : sequence_groups_) {
      if (!group_ptr->sequences().empty()) {
        // Sequences within group have same length, only need to get first
        // sequence's length
        const int32_t group_encoder_seq_len =
            group_ptr->sequences()[0]->encoder_tokens().size();
        total_tokens += group_encoder_seq_len * group_ptr->sequences().size();
      }
    }

    cache_data.encoder_tokens.reserve(total_tokens);
    cache_data.encoder_seq_lens.resize(num_sequences);
    cache_data.encoder_sparse_embeddings.clear();
    cache_data.encoder_sparse_embeddings.reserve(num_sequences);
    cache_data.decoder_context_embeddings.clear();
    cache_data.decoder_context_embeddings.reserve(num_sequences);

    // Process by groups in batch
    int32_t global_seq_idx = 0;
    for (const auto& group_ptr : sequence_groups_) {
      const auto& group = *group_ptr;
      const int32_t group_size = group.sequences().size();

      if (group_size == 0) continue;

      const int32_t group_encoder_seq_len =
          group.sequences()[0]->encoder_seq_len();

      // Batch set same values
      std::fill_n(&cache_data.encoder_seq_lens[global_seq_idx],
                  group_size,
                  group_encoder_seq_len);

      // Batch copy tokens by sequence and collect sparse_embedding
      for (const auto& sequence : group.sequences()) {
        const auto& encoder_tokens = sequence->encoder_tokens();
        const int32_t* src_ptr = encoder_tokens.data();
        const int32_t group_encoder_seq_len = encoder_tokens.size();

        // Use efficient batch insertion
        if (group_encoder_seq_len > 0) {
          cache_data.encoder_tokens.insert(cache_data.encoder_tokens.end(),
                                           src_ptr,
                                           src_ptr + group_encoder_seq_len);
        }
        // Collect sparse_embedding
        auto mm_data = sequence->get_mm_data();
        auto sparse_embedding_optional =
            mm_data.get<torch::Tensor>(Sequence::ENCODER_SPARSE_EMBEDDING_NAME);
        if (sparse_embedding_optional.has_value()) {
          cache_data.encoder_sparse_embeddings.push_back(
              sparse_embedding_optional.value());
        }

        auto decoder_context_embedding_optional = mm_data.get<torch::Tensor>(
            Sequence::DECODER_CONTEXT_EMBEDDING_NAME);
        if (decoder_context_embedding_optional.has_value()) {
          cache_data.decoder_context_embeddings.push_back(
              decoder_context_embedding_optional.value());
        }
      }

      global_seq_idx += group_size;
    }

    return cache_data.encoder_tokens;
  };

  // ========== High-performance decoder data construction ==========
  auto buildDecoderDataOptimized = [&]() {
    // Pre-allocate all containers to avoid dynamic expansion
    const size_t total_tokens = num_sequences * seq_len;
    std::vector<int32_t> flatten_tokens_vec;
    flatten_tokens_vec.reserve(total_tokens);
    std::vector<const RequestSamplingParam*> sampling_params;
    sampling_params.reserve(num_sequences);
    std::vector<int32_t> selected_token_idxes;
    selected_token_idxes.reserve(num_sequences);
    std::vector<int32_t> sample_idxes;
    sample_idxes.reserve(num_sequences);
    std::vector<std::vector<int32_t>> generated_tokens;
    generated_tokens.reserve(num_sequences);

    // Multi-threading optimization: Use parallel processing when sequence count
    // exceeds threshold and thread pool is available
    ThreadPool* threadpool = thread_pool_;
    if (num_sequences >= THREADPOOL_THRESHOLD && threadpool != nullptr) {
      // Thread-safe result containers
      std::vector<std::vector<int32_t>> thread_flatten_tokens(num_sequences);
      std::vector<const RequestSamplingParam*> thread_sampling_params(
          num_sequences);
      std::vector<int32_t> thread_selected_token_idxes(num_sequences);
      std::vector<int32_t> thread_sample_idxes(num_sequences);
      std::vector<std::vector<int32_t>> thread_generated_tokens(num_sequences);

      // Calculate thread allocation
      const size_t num_threads =
          std::min(static_cast<size_t>(num_sequences), static_cast<size_t>(16));
      const size_t sequences_per_thread =
          (num_sequences + num_threads - 1) / num_threads;

      std::vector<std::future<void>> futures;
      std::vector<std::shared_ptr<std::promise<void>>> promises;
      futures.reserve(num_threads);
      promises.reserve(num_threads);

      // Parallel processing function
      auto process_sequences_range = [&](size_t start_idx, size_t end_idx) {
        for (size_t i = start_idx;
             i < end_idx && i < static_cast<size_t>(num_sequences);
             ++i) {
          const Sequence* sequence = nullptr;
          // Get sequence from sequence_groups
          size_t seq_idx = 0;
          for (const auto& group : sequence_groups_) {
            if (seq_idx + group->sequences().size() > i) {
              sequence = group->sequences()[i - seq_idx].get();
              break;
            }
            seq_idx += group->sequences().size();
          }

          if (!sequence) continue;

          const auto& token_ids = sequence->tokens();

          // Build generated tokens
          auto& cur_generated_tokens = thread_generated_tokens[i];
          cur_generated_tokens.reserve(seq_len - n_prompt_tokens);
          for (uint32_t j = n_prompt_tokens; j < seq_len; ++j) {
            cur_generated_tokens.push_back(token_ids[j]);
          }

          // Build flatten tokens
          auto& cur_flatten_tokens = thread_flatten_tokens[i];
          cur_flatten_tokens.reserve(seq_len);
          cur_flatten_tokens.insert(cur_flatten_tokens.end(),
                                    token_ids.begin(),
                                    token_ids.begin() + seq_len);

          // Set sampling parameters
          thread_sampling_params[i] = sequence->sampling_param();
          thread_sample_idxes[i] = static_cast<int32_t>(i);
        }
      };

      // Launch parallel tasks
      for (size_t thread_idx = 0; thread_idx < num_threads; ++thread_idx) {
        size_t start_idx = thread_idx * sequences_per_thread;
        size_t end_idx = std::min(start_idx + sequences_per_thread,
                                  static_cast<size_t>(num_sequences));

        if (start_idx >= static_cast<size_t>(num_sequences)) break;

        auto promise = std::make_shared<std::promise<void>>();
        futures.push_back(promise->get_future());
        promises.push_back(promise);

        threadpool->schedule(
            [process_sequences_range, start_idx, end_idx, promise]() mutable {
              try {
                process_sequences_range(start_idx, end_idx);
                promise->set_value();
              } catch (...) {
                promise->set_exception(std::current_exception());
              }
            });
      }

      // Wait for all tasks to complete
      for (auto& future : futures) {
        future.get();
      }

      // Merge results
      size_t start_idx = 0;
      size_t total_tokens = seq_len + num_decoder_embeddings;
      for (int32_t i = 0; i < num_sequences; ++i) {
        flatten_tokens_vec.insert(flatten_tokens_vec.end(),
                                  thread_flatten_tokens[i].begin(),
                                  thread_flatten_tokens[i].end());
        selected_token_idxes.push_back(
            static_cast<int32_t>(start_idx + total_tokens - 1));
        start_idx += total_tokens;
        sampling_params.push_back(thread_sampling_params[i]);
        sample_idxes.push_back(thread_sample_idxes[i]);
        generated_tokens.push_back(std::move(thread_generated_tokens[i]));
      }
    } else {
      // Original single-thread processing logic
      size_t start_idx = 0;
      size_t total_tokens = seq_len + num_decoder_embeddings;
      size_t seq_idx = 0;
      for (const auto& group : sequence_groups_) {
        for (const auto& sequence : group->sequences()) {
          const auto& token_ids = sequence->tokens();

          // Optimize generated tokens construction
          auto& cur_generated_tokens = generated_tokens.emplace_back();
          cur_generated_tokens.reserve(seq_len - n_prompt_tokens);
          for (uint32_t j = n_prompt_tokens; j < seq_len; ++j) {
            cur_generated_tokens.push_back(token_ids[j]);
          }
          // Optimize token processing - batch operations
          flatten_tokens_vec.insert(flatten_tokens_vec.end(),
                                    token_ids.begin(),
                                    token_ids.begin() + seq_len);

          // Simplify sampling parameter processing
          selected_token_idxes.push_back(
              static_cast<int32_t>(start_idx + total_tokens - 1));
          start_idx += total_tokens;
          sampling_params.push_back(sequence->sampling_param());
          sample_idxes.push_back(seq_idx);
          seq_idx++;
        }
      }
    }

    return std::make_tuple(std::move(flatten_tokens_vec),
                           std::move(sampling_params),
                           std::move(selected_token_idxes),
                           std::move(sample_idxes),
                           std::move(generated_tokens));
  };

  // ========== Comprehensive parallel execution of optimized data construction
  // ========== Use thread pool to execute all independent data construction
  // tasks in parallel
  std::future<const std::vector<int32_t>&> encoder_future;
  std::future<std::tuple<std::vector<int32_t>,
                         std::vector<const RequestSamplingParam*>,
                         std::vector<int32_t>,
                         std::vector<int32_t>,
                         std::vector<std::vector<int32_t>>>>
      decoder_future;

  // Declare variables to store results
  const std::vector<int32_t>* encoder_tokens_ptr = nullptr;
  std::vector<int32_t> flatten_tokens_vec;
  std::vector<const RequestSamplingParam*> sampling_params;
  std::vector<int32_t> selected_token_idxes;
  std::vector<int32_t> sample_idxes;
  std::vector<std::vector<int32_t>> generated_tokens;
  if (thread_pool_ && num_sequences >= THREADPOOL_THRESHOLD) {
    // Use ThreadPool's schedule method to execute independent tasks in parallel
    // buildDecoderDataOptimized handles multi-threading internally, no external
    // parallel calls

    // Task 1: buildEncoderTokensOptimized
    std::promise<const std::vector<int32_t>*> encoder_promise;
    auto encoder_future = encoder_promise.get_future();
    thread_pool_->schedule([&, promise = std::move(encoder_promise)]() mutable {
      const auto& result = buildEncoderTokensOptimized();
      promise.set_value(&result);
    });
    // Wait for encoder to complete
    encoder_tokens_ptr = encoder_future.get();
    // Task 2: buildDecoderDataOptimized executes directly, handles
    // multi-threading internally
    std::tie(flatten_tokens_vec,
             sampling_params,
             selected_token_idxes,
             sample_idxes,
             generated_tokens) = buildDecoderDataOptimized();
  } else {
    // Single-thread execution (original logic)
    encoder_tokens_ptr = &buildEncoderTokensOptimized();
    std::tie(flatten_tokens_vec,
             sampling_params,
             selected_token_idxes,
             sample_idxes,
             generated_tokens) = buildDecoderDataOptimized();
  }

  const auto& encoder_tokens = *encoder_tokens_ptr;

  // ========== High-performance ForwardInput construction ==========
  ForwardInput forward_input;
  auto& input_params = forward_input.input_params;
  auto& cache_data = perf_cache_.cache_data;

  // Initialize key fields for asynchronous tasks
  const int64_t bs = sequence_groups_.size();
  const int64_t group_width =
      sequence_groups_.empty() ? 1 : sequence_groups_[0]->sequences().size();

  std::vector<std::future<void>> decoder_embedding_futures;
  torch::Tensor result_embedding;

  // ========== Parallel tensor construction tasks ==========
  if (thread_pool_ && num_sequences >= THREADPOOL_THRESHOLD) {
    // Only use parallelization for time-consuming tasks (token_ids and
    // encoder_token_ids)
    std::promise<torch::Tensor> token_ids_promise;
    std::promise<torch::Tensor> encoder_token_ids_promise;

    auto token_ids_future = token_ids_promise.get_future();
    // auto encoder_token_ids_future = encoder_token_ids_promise.get_future();

    // Task 1: Build token_ids tensor -
    // Optimization: Use torch::empty+std::memcpy instead of
    // torch::from_blob().clone()
    thread_pool_->schedule([&flatten_tokens_vec,
                            promise = std::move(token_ids_promise)]() mutable {
      try {
        // Optimization: Pre-allocate memory and use std::memcpy to avoid clone
        // operations
        auto tensor =
            torch::empty({static_cast<int64_t>(flatten_tokens_vec.size())},
                         torch::TensorOptions()
                             .dtype(torch::kInt)
                             .device(torch::kCPU)
                             .pinned_memory(true));
        std::memcpy(tensor.data_ptr<int>(),
                    flatten_tokens_vec.data(),
                    flatten_tokens_vec.size() * sizeof(int));
        promise.set_value(std::move(tensor));
      } catch (...) {
        promise.set_exception(std::current_exception());
      }
    });

    // Task 2: Build encoder_token_ids tensor (if needed) -
    // Optimization: Use torch::empty+std::memcpy instead of
    // torch::from_blob().clone()
    /*
    thread_pool_->schedule(
        [&encoder_tokens,
         promise = std::move(encoder_token_ids_promise)]() mutable {
          try {
            torch::Tensor tensor;
            if (!encoder_tokens.empty()) {
              // Optimization: Pre-allocate memory and use std::memcpy to avoid
              // clone operations
              tensor =
                  torch::empty({static_cast<int64_t>(encoder_tokens.size())},
                               torch::TensorOptions()
                                   .dtype(torch::kInt)
                                   .device(torch::kCPU)
                                   .pinned_memory(true));
              std::memcpy(tensor.data_ptr<int>(),
                          encoder_tokens.data(),
                          encoder_tokens.size() * sizeof(int));
            }
            promise.set_value(std::move(tensor));
          } catch (...) {
            promise.set_exception(std::current_exception());
          }
        });
    */
    if (!perf_cache_.cache_data.decoder_context_embeddings.empty()) {
      // Task 3: Synchronously process decoder_embedding, inner group dimension
      // parallelization optimization

      // Optimization: Directly get shape information from first embedding to
      // avoid torch::cat
      auto first_embedding =
          perf_cache_.cache_data.decoder_context_embeddings[0];
      auto original_shape = first_embedding.sizes();
      int64_t context_len = original_shape[0];
      int64_t hidden_size = original_shape[1];

      // Create tensor on pinned memory
      auto options = torch::TensorOptions()
                         .dtype(first_embedding.dtype())
                         .device(first_embedding.device())
                         .pinned_memory(true)
                         .memory_format(torch::MemoryFormat::Contiguous);

      // Calculate total sequence length, pre-allocate context_len + seq_len
      int64_t total_seq_len = context_len + seq_len;

      auto combined_embedding =
          torch::empty({bs, group_width, total_seq_len, hidden_size}, options);

      // High-performance optimization: group dimension segmented
      // parallelization
      void* dst_data = combined_embedding.data_ptr();

      // Get element size (supports float, bfloat16 and other types)
      const size_t element_size = first_embedding.element_size();
      const size_t context_size = context_len * hidden_size * element_size;
      const size_t group_stride = total_seq_len * hidden_size * element_size;
      const size_t batch_stride =
          group_width * total_seq_len * hidden_size * element_size;

      // Parallelization strategy: segment by group dimension, consistent with
      // thread calculations elsewhere
      const size_t num_threads =
          std::min(static_cast<size_t>(group_width), static_cast<size_t>(16));
      const size_t groups_per_thread =
          (group_width + num_threads - 1) / num_threads;

      for (size_t thread_idx = 0; thread_idx < num_threads; ++thread_idx) {
        size_t start_group = thread_idx * groups_per_thread;
        size_t end_group = std::min(start_group + groups_per_thread,
                                    static_cast<size_t>(group_width));

        if (start_group >= static_cast<size_t>(group_width)) break;

        std::promise<void> promise;
        decoder_embedding_futures.push_back(promise.get_future());

        thread_pool_->schedule(
            [start_group,
             end_group,
             bs,
             dst_data,
             context_len,
             hidden_size,
             element_size,
             batch_stride,
             group_stride,
             context_size,
             embeddings = perf_cache_.cache_data.decoder_context_embeddings,
             dst_tensor = combined_embedding,
             promise = std::move(promise)]() mutable {
              // Copy context_embedding for specified group range of each batch
              for (int64_t b = 0; b < bs; ++b) {
                // Optimization: Access corresponding batch embedding directly
                // through index
                const void* batch_src = embeddings[b].data_ptr();
                auto* batch_dst =
                    static_cast<char*>(dst_data) + b * batch_stride;

                for (size_t g = start_group; g < end_group; ++g) {
                  std::memcpy(
                      batch_dst + g * group_stride, batch_src, context_size);
                }
              }
              promise.set_value();
            });
      }

      result_embedding = combined_embedding;
    }

    // Task 4: Build sequence length vector - changed to serial execution (very
    // time-consuming, ~0.001785ms)
    std::vector<int32_t> cu_seq_lens, q_cu_seq_lens;
#if defined(USE_NPU)
    // use all prefill;
    cu_seq_lens.assign(num_sequences, seq_len + num_decoder_embeddings);
    q_cu_seq_lens.assign(num_sequences, seq_len + num_decoder_embeddings);
#else
    cu_seq_lens.reserve(num_sequences + 1);
    q_cu_seq_lens.reserve(num_sequences + 1);
    cu_seq_lens.push_back(0);
    q_cu_seq_lens.push_back(0);

    for (int32_t i = 0; i < num_sequences; ++i) {
      cu_seq_lens.push_back(cu_seq_lens.back() + seq_len +
                            num_decoder_embeddings);
      q_cu_seq_lens.push_back(q_cu_seq_lens.back() + seq_len +
                              num_decoder_embeddings);
    }
#endif

    // Task 5: Build encoder_seq_lens_tensor - changed to serial execution (less
    // time-consuming)
    torch::Tensor encoder_seq_lens_tensor;
    if (!cache_data.encoder_seq_lens.empty()) {
      // Optimization: Pre-allocate memory and use std::memcpy to avoid clone
      // operations
      encoder_seq_lens_tensor = torch::empty(
          {static_cast<int64_t>(cache_data.encoder_seq_lens.size())},
          torch::TensorOptions()
              .dtype(torch::kInt)
              .device(torch::kCPU)
              .pinned_memory(true));
      std::memcpy(encoder_seq_lens_tensor.data_ptr<int>(),
                  cache_data.encoder_seq_lens.data(),
                  cache_data.encoder_seq_lens.size() * sizeof(int));
    }

    // Set basic parameters simultaneously (not dependent on asynchronous tasks)
    input_params.num_sequences = num_sequences;
    input_params.empty_kv_cache = true;
    input_params.global_empty_kv_cache = true;
    input_params.kv_max_seq_len = seq_len + num_decoder_embeddings;
    input_params.q_max_seq_len = seq_len + num_decoder_embeddings;
    forward_input.positions = perf_cache_.fixed_positions_tensor;

    // Wait and collect results
    forward_input.token_ids = token_ids_future.get();
    // auto encoder_token_ids = encoder_token_ids_future.get();

    // seq_lens has been changed to serial execution, use the constructed
    // variable directly

    // Optimization: Use torch::empty+std::memcpy instead of
    // torch::from_blob().clone()
    input_params.kv_seq_lens =
        torch::empty({static_cast<int64_t>(cu_seq_lens.size())},
                     torch::TensorOptions()
                         .dtype(torch::kInt)
                         .device(torch::kCPU)
                         .pinned_memory(true));
    std::memcpy(input_params.kv_seq_lens.data_ptr<int>(),
                cu_seq_lens.data(),
                cu_seq_lens.size() * sizeof(int));

    input_params.q_seq_lens =
        torch::empty({static_cast<int64_t>(q_cu_seq_lens.size())},
                     torch::TensorOptions()
                         .dtype(torch::kInt)
                         .device(torch::kCPU)
                         .pinned_memory(true));
    std::memcpy(input_params.q_seq_lens.data_ptr<int>(),
                q_cu_seq_lens.data(),
                q_cu_seq_lens.size() * sizeof(int));
    input_params.kv_seq_lens_vec = std::move(cu_seq_lens);
    input_params.q_seq_lens_vec = std::move(q_cu_seq_lens);

    // encoder_seq_lens_tensor has been changed to serial execution, use the
    // constructed variable directly
    if (encoder_seq_lens_tensor.defined()) {
      // Set RecModelInputParams encoder data
      if (!input_params.rec_params.has_value()) {
        input_params.rec_params = RecModelInputParams{};
      }
      input_params.rec_params->encoder_seq_lens_tensor =
          std::move(encoder_seq_lens_tensor);
      input_params.rec_params->encoder_seq_lens = cache_data.encoder_seq_lens;
    }
    input_params.rec_params->encoder_positions =
        perf_cache_.fixed_encoder_positions_tensor;
  } else {
    // Single-threaded execution (original logic)
    // Optimization: Use torch::empty+std::memcpy instead of
    // torch::from_blob().clone()
    forward_input.token_ids =
        torch::empty({static_cast<int64_t>(flatten_tokens_vec.size())},
                     torch::TensorOptions()
                         .dtype(torch::kInt)
                         .device(torch::kCPU)
                         .pinned_memory(true));
    std::memcpy(forward_input.token_ids.data_ptr<int>(),
                flatten_tokens_vec.data(),
                flatten_tokens_vec.size() * sizeof(int));
    forward_input.positions = perf_cache_.fixed_positions_tensor;

    if (!encoder_tokens.empty()) {
      // Set RecModelInputParams encoder data
      if (!input_params.rec_params.has_value()) {
        input_params.rec_params = RecModelInputParams{};
      }

      // Optimization: Use torch::empty+std::memcpy instead of
      // torch::from_blob().clone()
      input_params.rec_params->encoder_token_ids =
          torch::empty({static_cast<int64_t>(encoder_tokens.size())},
                       torch::TensorOptions()
                           .dtype(torch::kInt)
                           .device(torch::kCPU)
                           .pinned_memory(true));
      std::memcpy(input_params.rec_params->encoder_token_ids.data_ptr<int>(),
                  encoder_tokens.data(),
                  encoder_tokens.size() * sizeof(int));
    }
    input_params.rec_params->encoder_positions =
        perf_cache_.fixed_encoder_positions_tensor;
    // Pre-allocate and batch fill
    std::vector<int32_t> cu_seq_lens, q_cu_seq_lens;
#if defined(USE_NPU)
    // use all prefill;
    cu_seq_lens.assign(num_sequences, seq_len + num_decoder_embeddings);
    q_cu_seq_lens.assign(num_sequences, seq_len + num_decoder_embeddings);
#else
    cu_seq_lens.reserve(num_sequences + 1);
    q_cu_seq_lens.reserve(num_sequences + 1);
    cu_seq_lens.push_back(0);
    q_cu_seq_lens.push_back(0);

    for (int32_t i = 0; i < num_sequences; ++i) {
      cu_seq_lens.push_back(cu_seq_lens.back() + seq_len +
                            num_decoder_embeddings);
      q_cu_seq_lens.push_back(q_cu_seq_lens.back() + seq_len +
                              num_decoder_embeddings);
    }
#endif

    input_params.num_sequences = num_sequences;
    input_params.empty_kv_cache = true;
    input_params.global_empty_kv_cache = true;
    input_params.kv_max_seq_len = seq_len + num_decoder_embeddings;
    input_params.q_max_seq_len = seq_len + num_decoder_embeddings;

    // Optimization: Use torch::empty+std::memcpy instead of
    // torch::from_blob().clone()
    input_params.kv_seq_lens =
        torch::empty({static_cast<int64_t>(cu_seq_lens.size())},
                     torch::TensorOptions()
                         .dtype(torch::kInt)
                         .device(torch::kCPU)
                         .pinned_memory(true));
    std::memcpy(input_params.kv_seq_lens.data_ptr<int>(),
                cu_seq_lens.data(),
                cu_seq_lens.size() * sizeof(int));

    input_params.q_seq_lens =
        torch::empty({static_cast<int64_t>(q_cu_seq_lens.size())},
                     torch::TensorOptions()
                         .dtype(torch::kInt)
                         .device(torch::kCPU)
                         .pinned_memory(true));
    std::memcpy(input_params.q_seq_lens.data_ptr<int>(),
                q_cu_seq_lens.data(),
                q_cu_seq_lens.size() * sizeof(int));

    input_params.kv_seq_lens_vec = std::move(cu_seq_lens);
    input_params.q_seq_lens_vec = std::move(q_cu_seq_lens);

    if (!cache_data.encoder_seq_lens.empty()) {
      // Set RecModelInputParams encoder data
      if (!input_params.rec_params.has_value()) {
        input_params.rec_params = RecModelInputParams{};
      }

      input_params.rec_params->encoder_seq_lens = cache_data.encoder_seq_lens;

      // Optimization: Use torch::empty+std::memcpy instead of
      // torch::from_blob().clone()
      input_params.rec_params->encoder_seq_lens_tensor = torch::empty(
          {static_cast<int64_t>(cache_data.encoder_seq_lens.size())},
          torch::TensorOptions()
              .dtype(torch::kInt)
              .device(torch::kCPU)
              .pinned_memory(true));
      std::memcpy(
          input_params.rec_params->encoder_seq_lens_tensor.data_ptr<int>(),
          cache_data.encoder_seq_lens.data(),
          cache_data.encoder_seq_lens.size() * sizeof(int));
    }
  }

  // ========== Parallel processing of independent code blocks ==========
  if (thread_pool_ && num_sequences >= THREADPOOL_THRESHOLD) {
    // Define promise/future for parallel tasks
    std::promise<void> block_tables_promise;
    auto block_tables_future = block_tables_promise.get_future();

    // Task 1: Empty block tables processing - use thread pool (relatively
    // time-consuming)
    thread_pool_->schedule([&input_params,
                            num_sequences,
                            &perf_cache_,
                            &block_tables_promise]() mutable {
      try {
        std::vector<std::vector<int32_t>> empty_block_tables(num_sequences);
        util::pad_2d_vector(empty_block_tables, 0);
        // Optimization: Use create_2d_tensor_optimized, has special
        // optimization for all-zero matrices
        input_params.block_tables =
            create_2d_tensor(empty_block_tables, torch::kInt);

        std::vector<int32_t> paged_kv_indptr(num_sequences + 1, 0);
        // Optimization: Use torch::empty+std::memcpy instead of
        // torch::from_blob().clone()
        input_params.new_cache_slots =
            torch::empty({static_cast<int64_t>(paged_kv_indptr.size())},
                         torch::TensorOptions()
                             .dtype(torch::kInt)
                             .device(torch::kCPU)
                             .pinned_memory(true));
        std::memcpy(input_params.new_cache_slots.data_ptr<int>(),
                    paged_kv_indptr.data(),
                    paged_kv_indptr.size() * sizeof(int));

        block_tables_promise.set_value();
      } catch (...) {
        block_tables_promise.set_exception(std::current_exception());
      }
    });

    // Optimization: Merge small tasks into sequential execution to reduce
    // thread switching overhead Cross-attention parameter construction - use
    // placeholder
    if (!input_params.rec_params.has_value()) {
      input_params.rec_params = RecModelInputParams{};
    }
    input_params.rec_params->cross_attn_kv_cu_seq_lens =
        torch::zeros({1}, torch::kInt);
    input_params.rec_params->cross_attn_kv_cu_seq_lens_vec = {0};
    input_params.rec_params->cross_attn_block_tables =
        torch::zeros({1, 1}, torch::kInt);

    // Sampling parameter processing
    if (!selected_token_idxes.empty()) {
      forward_input.sampling_params.init(sampling_params,
                                         selected_token_idxes,
                                         sample_idxes,
                                         std::vector<std::vector<int64_t>>{},
                                         std::vector<std::vector<int32_t>>{},
                                         std::vector<int32_t>{});
    }

    // First prefill processing - use placeholder
    if (is_first_prefill) {
      // Use placeholder instead of complex cross_attn_new_cache_slots
      // construction
      input_params.rec_params->cross_attn_new_cache_slots =
          torch::zeros({1}, torch::kInt);
    }

    // Wait for parallel tasks to complete (only block_tables uses thread pool)
    block_tables_future.wait();
  } else {
    // ========== Non-parallel case: sequential processing ==========
    // Optimize empty block tables processing
    std::vector<std::vector<int32_t>> empty_block_tables(num_sequences);
    util::pad_2d_vector(empty_block_tables, 0);
    // Optimization: Use create_2d_tensor_optimized, has special optimization
    // for all-zero matrices
    input_params.block_tables =
        create_2d_tensor(empty_block_tables, torch::kInt);

    std::vector<int32_t> paged_kv_indptr(num_sequences + 1, 0);
    // Optimization: Use torch::empty+std::memcpy instead of
    // torch::from_blob().clone()
    input_params.new_cache_slots =
        torch::empty({static_cast<int64_t>(paged_kv_indptr.size())},
                     torch::TensorOptions()
                         .dtype(torch::kInt)
                         .device(torch::kCPU)
                         .pinned_memory(true));
    std::memcpy(input_params.new_cache_slots.data_ptr<int>(),
                paged_kv_indptr.data(),
                paged_kv_indptr.size() * sizeof(int));

    // ========== Cross-attention parameter construction (using placeholder)
    // ========== Use placeholder tensor instead of actual data
    if (!input_params.rec_params.has_value()) {
      input_params.rec_params = RecModelInputParams{};
    }
    input_params.rec_params->cross_attn_kv_cu_seq_lens =
        torch::zeros({1}, torch::kInt);
    input_params.rec_params->cross_attn_kv_cu_seq_lens_vec = {0};

    // Use placeholder tensor instead of actual data
    input_params.rec_params->cross_attn_block_tables =
        torch::zeros({1, 1}, torch::kInt);

    // ========== Optimize sampling parameter processing ==========
    if (!selected_token_idxes.empty()) {
      forward_input.sampling_params.init(sampling_params,
                                         selected_token_idxes,
                                         sample_idxes,
                                         std::vector<std::vector<int64_t>>{},
                                         std::vector<std::vector<int32_t>>{},
                                         std::vector<int32_t>{});
    }

    // ========== First prefill processing (using placeholder) ==========
    if (is_first_prefill) {
      // Use placeholder tensor instead of actual data
      input_params.rec_params->cross_attn_new_cache_slots =
          torch::zeros({1}, torch::kInt);
    }
  }

  // ========== Common parameter settings ==========
  // Batch set other parameters
  input_params.embedding_ids.assign(num_sequences, 0);

#if defined(USE_NPU)
  auto prefill_indices = util::find_ones_indices(input_params.q_seq_lens_vec);
  input_params.decode_seq_range =
      std::make_pair(0, static_cast<int>(flatten_tokens_vec.size()));
#else
  input_params.decode_seq_range = {
      0, static_cast<int32_t>(flatten_tokens_vec.size())};
#endif

  // Rec model parameters
  if (!input_params.rec_params.has_value()) {
    input_params.rec_params = RecModelInputParams{};
  }

  input_params.rec_params->rec_stage = RecModelInputParams::RecStage::PREFILL;
  input_params.rec_params->is_hybrid_mode = false;
  input_params.rec_params->has_encoder_output = true;
  input_params.rec_params->is_first_prefill = is_first_prefill;
  input_params.rec_params->bs = bs;
  input_params.rec_params->group_width = group_width;
  input_params.rec_params->seq_len = seq_len;
  input_params.rec_params->encoder_max_seq_len =
      cache_data.encoder_seq_lens.empty()
          ? 0
          : *std::max_element(cache_data.encoder_seq_lens.begin(),
                              cache_data.encoder_seq_lens.end());

  input_params.rec_params->generated_tokens = std::move(generated_tokens);

  // Process sparse_embedding: Efficiently concatenate from cache_data
  if (!perf_cache_.cache_data.encoder_sparse_embeddings.empty()) {
    // Use torch::cat for efficient concatenation, concatenate along dim=0
    input_params.rec_params->encoder_sparse_embedding =
        torch::cat(perf_cache_.cache_data.encoder_sparse_embeddings, /*dim=*/0);
  }

  if (!perf_cache_.cache_data.decoder_context_embeddings.empty()) {
    // Get group_width
    int64_t group_width = input_params.rec_params->group_width;
    if (group_width == 1 && seq_len == 0) {
      // Optimization: When bs==1, directly use the first embedding to avoid
      // unnecessary torch::cat
      if (bs == 1) {
        input_params.rec_params->decoder_context_embedding =
            perf_cache_.cache_data.decoder_context_embeddings[0];
      } else {
        // Use torch::cat for efficient concatenation, concatenate along dim=0
        auto original_context_embedding = torch::cat(
            perf_cache_.cache_data.decoder_context_embeddings, /*dim=*/0);
        input_params.rec_params->decoder_context_embedding =
            original_context_embedding;
      }
    } else if (group_width == 1 && seq_len > 0) {
      // Handle the scenario where group_width==1 and seq_len>0
      // Get information from the first embedding
      const auto& first_embedding =
          perf_cache_.cache_data.decoder_context_embeddings[0];
      auto original_shape = first_embedding.sizes();
      int64_t context_len = original_shape[0];
      int64_t hidden_size = original_shape[1];
      int64_t total_seq_len = context_len + seq_len;

      // Allocate a tensor of shape {bs, 1, total_seq_len, hidden_size},
      // optimized with pinned memory
      auto options = torch::TensorOptions()
                         .dtype(first_embedding.dtype())
                         .device(first_embedding.device())
                         .pinned_memory(true)
                         .memory_format(torch::MemoryFormat::Contiguous);
      auto combined_embedding =
          torch::empty({bs, 1, total_seq_len, hidden_size}, options);

      // Single-threaded copy of context_len portion of data
      void* dst_data = combined_embedding.data_ptr();
      const size_t element_size = first_embedding.element_size();
      const size_t context_size = context_len * hidden_size * element_size;
      const size_t batch_stride = total_seq_len * hidden_size * element_size;

      // Copy context_embedding for each batch
      for (int64_t b = 0; b < bs; ++b) {
        const void* batch_src =
            perf_cache_.cache_data.decoder_context_embeddings[b].data_ptr();
        auto* batch_dst = static_cast<char*>(dst_data) + b * batch_stride;
        std::memcpy(batch_dst, batch_src, context_size);
      }
      input_params.rec_params->decoder_context_embedding = combined_embedding;
    } else {
      for (auto& future : decoder_embedding_futures) {
        future.get();
      }
      input_params.rec_params->decoder_context_embedding =
          std::move(result_embedding);
    }
  }

  return forward_input;
}

}  // namespace xllm
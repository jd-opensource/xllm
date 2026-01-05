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

#include "speculative_worker_impl.h"

#include "common/global_flags.h"
#include "common/metrics.h"
#include "framework/request/mm_data.h"
#include "util/env_var.h"
#include "util/pretty_print.h"
#include "util/slice.h"
#include "util/timer.h"
#include "util/utils.h"

namespace xllm {
constexpr uint64_t MBUF_SIZE = 128 * 1024 * 1024;

namespace {
#define TENSOR_REPEAT(tensor_, repeats)                                       \
  do {                                                                        \
    tensor_ = tensor_.defined()                                               \
                  ? tensor_.repeat_interleave(/*repeats=*/repeats, /*dim=*/0) \
                  : tensor_;                                                  \
  } while (0)

std::vector<int32_t> kv_cache_slots(int32_t pos_start,
                                    int32_t offset,
                                    const Slice<int32_t>& block_table_slice,
                                    int32_t block_size) {
  std::vector<int32_t> slots;
  slots.reserve(offset);
  for (int32_t i = pos_start; i < pos_start + offset; ++i) {
    const int32_t block_id = block_table_slice[i / block_size];
    const int32_t block_offset = i % block_size;
    slots.push_back(block_id * block_size + block_offset);
  }
  return slots;
}

int32_t kv_cache_slot_id(int32_t position,
                         const Slice<int32_t>& block_table_slice,
                         int32_t block_size) {
  const int32_t block_id = block_table_slice[position / block_size];
  const int32_t block_offset = position % block_size;
  return block_id * block_size + block_offset;
}

// Convert tensor to int64 for MLU platform (temp workaround)
// MLU will support int32 for masked_scatter in the future
torch::Tensor ensure_int64_for_certain_platform(torch::Tensor tensor) {
#if defined(USE_MLU)
  return tensor.to(torch::kInt64);
#else
  return tensor;
#endif
}

// Push cumulative sum to vector (used for cumulative format)
void push_cumsum(std::vector<int32_t>& vec, int32_t len) {
  if (vec.empty()) {
    vec.emplace_back(0);
  }
  vec.emplace_back(vec.back() + len);
}

// Calculate actual kv_len based on platform type
// For NPU: direct format - returns kv_seq_lens_slice[seq_id] + offset
// For MLU/CUDA: cumulative format - returns the actual length increment
int32_t calculate_kv_len(const Slice<int32_t>& kv_seq_lens_slice,
                         int32_t seq_id,
                         int32_t offset) {
#if defined(USE_NPU)
  return kv_seq_lens_slice[seq_id] + offset;
#elif defined(USE_MLU) || defined(USE_CUDA)
  return kv_seq_lens_slice[seq_id + 1] - kv_seq_lens_slice[seq_id] + offset;
#endif
}

// Append sequence length to vector based on platform type
// For NPU: directly add the len value
// For MLU/CUDA: add using cumulative format
void append_seq_len(std::vector<int32_t>& vec, int32_t len) {
#if defined(USE_NPU)
  vec.emplace_back(len);
#elif defined(USE_MLU) || defined(USE_CUDA)
  push_cumsum(vec, len);
#endif
}

// Update kv_seq_lens_vec and kv_max_seq_len
void update_kv_seq_lens_and_max(std::vector<int32_t>& kv_seq_lens_vec,
                                int32_t kv_len,
                                int32_t& kv_max_seq_len) {
  // Update max (same logic for both platforms)
  if (kv_len > kv_max_seq_len) {
    kv_max_seq_len = kv_len;
  }
  // Update kv_seq_lens_vec
  append_seq_len(kv_seq_lens_vec, kv_len);
}

// Batch expansion strategy for validation
void batch_expansion_process_seq_lens(
    std::vector<int32_t>& kv_seq_lens_vec,
    std::vector<int32_t>& q_seq_lens_vec,
    std::vector<std::vector<int32_t>>& block_tables_vec,
    int32_t& kv_max_seq_len,
    const Slice<int32_t>& kv_seq_lens_slice,
    const Slice<int32_t>& block_table_slice,
    int32_t seq_id,
    int32_t position_offset,
    int32_t num_val_tokens) {
  for (int32_t offset = position_offset;
       offset < num_val_tokens + position_offset;
       ++offset) {
    // Calculate kv length and update kv_seq_lens_vec and kv_max_seq_len
    int32_t kv_len = calculate_kv_len(kv_seq_lens_slice, seq_id, offset);
    update_kv_seq_lens_and_max(kv_seq_lens_vec, kv_len, kv_max_seq_len);
    // Append sequence length of 1 to q_seq_lens_vec
    //  for batch expansion strategy for validation
    append_seq_len(q_seq_lens_vec, 1);
    // Append block table to block_tables_vec
    block_tables_vec.emplace_back(block_table_slice);
  }
}

// Update kv_seq_lens_vec based on platform type
// For NPU: directly add kv_seq_lens_slice[seq_id] + offset
// For others: build cumulative format
// Also updates kv_max_seq_len to track the maximum sequence length
void update_kv_seq_lens_vec(std::vector<int32_t>& kv_seq_lens_vec,
                            const Slice<int32_t>& kv_seq_lens_slice,
                            int32_t seq_id,
                            int32_t offset,
                            int32_t& kv_max_seq_len) {
  int32_t kv_len = calculate_kv_len(kv_seq_lens_slice, seq_id, offset);
  update_kv_seq_lens_and_max(kv_seq_lens_vec, kv_len, kv_max_seq_len);
}

}  // namespace

SpeculativeWorkerImpl::SpeculativeWorkerImpl(const ParallelArgs& parallel_args,
                                             const torch::Device& device,
                                             const runtime::Options& options)
    : WorkerImpl(parallel_args, device, options) {
  auto runtime_options = options;
  runtime_options.enable_schedule_overlap(false);
  impl_ =
      std::make_unique<LLMWorkerImpl>(parallel_args, device, runtime_options);
  // here we specify num speculative tokens to 0 to pass the indication of
  //  draft model to worker when enable_speculative_decode.
  // NOTE: If you want to modify this part, make sure you also check the usage
  // of
  //  num_speculative_tokens in draft model.
  runtime_options.num_decoding_tokens(1).num_speculative_tokens(0);
  draft_impl_ =
      std::make_unique<LLMWorkerImpl>(parallel_args, device, runtime_options);

  // performance debug for fixing the speculative acceptance rate
  // NOTE: This is for performance debugging only, it will
  // influence the model accuracy and should not be used in production.
  std::optional<double> fixed_acceptance_rate =
      util::get_fix_speculative_acceptance_rate();
  if (fixed_acceptance_rate.has_value()) {
    rate_controller_ = std::make_shared<RejectionSamplerRateController>(
        *fixed_acceptance_rate);
  }
}

bool SpeculativeWorkerImpl::init_model(const std::string& model_weights_path,
                                       int32_t random_seed) {
  // initialize model
  bool result = true;
  if (impl_->get_status() == WorkerImpl::Status::UNINITIALIZED) {
    result = impl_->WorkerImpl::init_model(model_weights_path, random_seed);
    if (result) {
      dtype_ = impl_->dtype();
      embedding_size_ = impl_->hidden_size();
    }
  } else {
    CHECK_EQ(draft_impl_->get_status(), WorkerImpl::Status::UNINITIALIZED);
    result =
        draft_impl_->WorkerImpl::init_model(model_weights_path, random_seed);
  }

  if (draft_impl_->get_status() == WorkerImpl::Status::LOADED) {
    // Deepseek MTP
#if defined(USE_NPU)
    if (FLAGS_npu_kernel_backend != "TORCH") {
      auto head = impl_->get_npu_lm_head();
      draft_impl_->set_npu_lm_head(head);
      auto word_embedding = impl_->get_npu_word_embedding();
      draft_impl_->set_npu_word_embedding(word_embedding);
    } else {
      // TODO: Support TORCH backend via torch_npu encapsulation in the future.
      // Currently, it is explicitly disabled.
      LOG(FATAL)
          << "SpeculativeWorkerImpl::init_model not support TORCH backend";
    }
#else
    auto head = impl_->get_lm_head();
    draft_impl_->set_lm_head(head);
    auto word_embedding = impl_->get_word_embedding();
    draft_impl_->set_word_embedding(word_embedding);
#endif
  }
  return result;
}

bool SpeculativeWorkerImpl::allocate_kv_cache(
    const std::vector<std::vector<int64_t>>& kv_cache_shape) {
  // init embedding cache, using total number of blocks
  if (impl_->get_status() == WorkerImpl::Status::LOADED) {
    embedding_allocator_ = std::make_shared<EmbeddingAllocator>(
        kv_cache_shape[0][0], embedding_size_, dtype_);
  }

  if (impl_->get_status() == WorkerImpl::Status::LOADED) {
    return impl_->allocate_kv_cache(kv_cache_shape);
  } else {
    CHECK_EQ(draft_impl_->get_status(), WorkerImpl::Status::LOADED);
    return draft_impl_->allocate_kv_cache(kv_cache_shape);
  }
}

#if defined(USE_NPU)
bool SpeculativeWorkerImpl::allocate_kv_cache_with_transfer(
    const uint64_t kv_cache_size,
    const std::vector<std::vector<int64_t>>& kv_cache_shape) {
  if (impl_->get_status() == WorkerImpl::Status::LOADED) {
    kv_cache_transfer_ =
        std::make_shared<SpecKVCacheTransfer>(options_.device_ip().value(),
                                              options_.transfer_listen_port(),
                                              options_.instance_role());

    int32_t device_id = device_.index();
    kv_cache_transfer_->initialize(device_id);
    impl_->allocate_kv_cache_with_transfer(kv_cache_transfer_, kv_cache_shape);
  } else {
    CHECK_EQ(draft_impl_->get_status(), WorkerImpl::Status::LOADED);
    draft_impl_->allocate_kv_cache_with_transfer(kv_cache_transfer_,
                                                 kv_cache_shape);
    embedding_allocator_ = std::make_shared<EmbeddingAllocator>(
        kv_cache_shape[0][0], embedding_size_, dtype_);
  }
  return true;
}
#endif

std::optional<ForwardOutput> SpeculativeWorkerImpl::step(
    const ForwardInput& input) {
  if (input.token_ids.numel() == 0) {
    return step_empty(input);
  }

  if (!input.input_params.batch_forward_type.is_decode()) {
    return step_prefill(input);
  } else {
    return step_decode(input);
  }
}

std::optional<ForwardOutput> SpeculativeWorkerImpl::step_empty(
    const ForwardInput& input) {
  if (!input.input_params.batch_forward_type.is_decode()) {
    auto output = impl_->step(input);
    auto draft_output = draft_impl_->step(input);
    return output;
  } else {
    for (size_t i = 0; i < options_.num_speculative_tokens(); ++i) {
      auto draft_future = draft_impl_->step_async(input);
      ForwardOutput draft_output = std::move(draft_future).get().value();
    }

    ForwardInput new_input = input;
    for (auto& it : new_input.input_params.dp_global_token_nums) {
      it *= options_.num_speculative_tokens() + 1;
    }

    auto future = impl_->step_async(new_input);
    ForwardOutput output = std::move(future).get().value();
    return output;
  }
}

std::optional<ForwardOutput> SpeculativeWorkerImpl::step_prefill(
    const ForwardInput& input) {
  Timer timer;
  // run the target model to get first token and hidden states
  auto future = impl_->step_async(input);
  // MTP (Eagle Medusa) which depend on hidden states need this step
  // The others speculative model use inputs directly
  // ForwardInput prefill_inputs;
  ForwardInput prefill_input;
  prepare_prefill_inputs(input, prefill_input);
  ForwardOutput output = std::move(future).get().value();
  COUNTER_ADD(speculative_execution_latency_seconds_target,
              timer.elapsed_seconds());

  // prepare input for draft model
  auto& embeddings = output.sample_output.embeddings;
  auto next_tokens = ensure_int64_for_certain_platform(
      safe_to(output.sample_output.next_tokens, torch::kInt));

  if (embeddings.defined()) {
    prefill_input.input_params.input_embedding = embeddings.clone();
  }
  if (next_tokens.defined()) {
    auto& token_ids = prefill_input.token_ids;
    token_ids = ensure_int64_for_certain_platform(token_ids);
    auto mask = (token_ids == -1);
    token_ids.masked_scatter_(mask, next_tokens);
  }

  // generate kv cache for draft model
  timer.reset();
  auto draft_future = draft_impl_->step_async(prefill_input);
  ForwardOutput draft_output = std::move(draft_future).get().value();
  COUNTER_ADD(speculative_execution_latency_seconds_draft,
              timer.elapsed_seconds());

  if (input.sampling_params.selected_token_idxes.defined()) {
    embeddings = embeddings.index_select(
        /*dim=*/0, input.sampling_params.selected_token_idxes);
    CHECK_EQ(embeddings.size(0), output.sample_output.next_tokens.size(0));
    embedding_allocator_->write(input.input_params.embedding_ids, embeddings);
  }
  output.sample_output.embeddings = torch::Tensor();

  if (!enable_schedule_overlap() && !driver_ && !dp_driver_) {
    return std::nullopt;
  }
  return output;
}

void SpeculativeWorkerImpl::prepare_prefill_inputs(
    const ForwardInput& input,
    ForwardInput& prefill_input) {
  prefill_input = input.to(device_, dtype_);
  auto& input_params = prefill_input.input_params;
  auto& extra_token_ids = input_params.extra_token_ids;

  torch::Tensor token_ids = safe_to(input.token_ids, torch::kCPU);
  Slice<int32_t> tokens_ids_slice = {
      token_ids.data_ptr<int32_t>(),
      static_cast<size_t>(input.token_ids.numel())};

  int32_t start_idx = 0;
  std::vector<int32_t> new_token_ids;
  new_token_ids.reserve(input.token_ids.numel());
  for (size_t i = 0; i < input_params.num_sequences; ++i) {
    int32_t q_len = 0;
    q_len = input_params.get_q_seq_len(i);
    Slice<int32_t> tokens_ids_slice_i =
        tokens_ids_slice.slice(start_idx + 1, start_idx + q_len);
    start_idx += q_len;
    new_token_ids.insert(new_token_ids.end(),
                         tokens_ids_slice_i.begin(),
                         tokens_ids_slice_i.end());
    new_token_ids.emplace_back(extra_token_ids[i]);
  }
  prefill_input.token_ids =
      torch::tensor(new_token_ids, prefill_input.positions.options());
}

std::optional<ForwardOutput> SpeculativeWorkerImpl::step_decode(
    const ForwardInput& input) {
  // TODO : now only support Deepseek MTP
  // More work need to support n-gram and native speculative decoding.
  ForwardInput draft_input = input;
  // get embedding cache
  torch::Tensor embeddings =
      embedding_allocator_->read(draft_input.input_params.embedding_ids);
  draft_input.input_params.input_embedding = embeddings.to(device_);

  // run the draft model to get proposals
  std::vector<ForwardOutput> draft_outputs;
  ForwardInput validate_input, next_step_input;
  Timer timer;
  std::vector<folly::SemiFuture<std::optional<ForwardOutput>>> futures;
  for (size_t i = 0; i < options_.num_speculative_tokens(); ++i) {
    auto future = draft_impl_->step_async(draft_input);
    if (i == options_.num_speculative_tokens() - 1) {
      // final step, prepare validate input
      prepare_validate_inputs(input, validate_input);
    } else {
      prepare_draft_inputs(draft_input, next_step_input, 1, device_);
    }
    draft_outputs.push_back(std::move(future).get().value());
    // update input of next step
    if (i < options_.num_speculative_tokens() - 1) {
      draft_input = next_step_input;
      auto last_output = draft_outputs.back().sample_output;
      draft_input.token_ids = safe_to(last_output.next_tokens, torch::kInt);
      draft_input.input_params.input_embedding =
          last_output.embeddings.to(device_);
    }
  }
  COUNTER_ADD(speculative_execution_latency_seconds_draft,
              timer.elapsed_seconds());

  for (int i = 0; i < options_.num_speculative_tokens(); ++i) {
    ForwardOutput draft_output = draft_outputs[i];
    auto next_tokens = ensure_int64_for_certain_platform(
        safe_to(draft_output.sample_output.next_tokens, torch::kInt));
    auto& token_ids = validate_input.token_ids;
    token_ids = ensure_int64_for_certain_platform(token_ids);
    auto mask = (token_ids == -1 * (i + 1));
    token_ids.masked_scatter_(mask, next_tokens);
  }

  // run the target model to get the verification scores
  timer.reset();
  auto future = impl_->step_async(validate_input);
  ForwardOutput target_output = std::move(future).get().value();
  COUNTER_ADD(speculative_execution_latency_seconds_target,
              timer.elapsed_seconds());

  // verify the proposals with target and update the batch
  timer.reset();
  SampleOutput val_output =
      validate(input.sampling_params, draft_outputs, target_output);
  COUNTER_ADD(speculative_execution_latency_seconds_validation,
              timer.elapsed_seconds());

  // write the right cache and clear embeddings
  embedding_allocator_->write_validate(input.input_params.embedding_ids,
                                       val_output.next_tokens.to(torch::kCPU),
                                       val_output.embeddings);

  if (!enable_schedule_overlap() && !driver_ && !dp_driver_) {
    return std::nullopt;
  }
  val_output.embeddings = torch::Tensor();
  target_output.sample_output = val_output;
  return target_output;
}

void SpeculativeWorkerImpl::prepare_draft_inputs(const ForwardInput& input,
                                                 ForwardInput& draft_input,
                                                 const int64_t offset,
                                                 const torch::Device device) {
  // prepare input for MTP in decoding phase (Like Eagle).
  draft_input = input.to(device, dtype_);

  auto& input_params = draft_input.input_params;
  const int32_t num_sequences = input_params.num_sequences;
  int32_t block_size = options_.block_size();
  torch::TensorOptions int_options = input.token_ids.options();

  std::vector<int32_t> new_positions;
  new_positions.reserve(num_sequences);
  std::vector<int32_t> kv_seq_lens_vec = {};
  // slot ids for new token
  std::vector<int32_t> new_token_slot_ids;

  torch::Tensor positions = safe_to(input.positions, torch::kCPU);
  Slice<int32_t> positions_slice = {positions.data_ptr<int32_t>(),
                                    static_cast<size_t>(positions.numel())};
  Slice<int32_t> kv_seq_lens_slice = input_params.kv_seq_lens_vec;
  torch::Tensor block_tables = safe_to(input_params.block_tables, torch::kCPU);

  // Initialize kv_max_seq_len to 0
  int32_t kv_max_seq_len = 0;

  for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
    new_positions.emplace_back(positions_slice[seq_id] + offset);
    update_kv_seq_lens_vec(
        kv_seq_lens_vec, kv_seq_lens_slice, seq_id, offset, kv_max_seq_len);
    torch::Tensor block_table = block_tables[seq_id];
    Slice<int32_t> block_table_slice = {
        block_table.data_ptr<int32_t>(),
        static_cast<size_t>(block_table.numel())};
    int32_t slot_id =
        kv_cache_slot_id(new_positions.back(), block_table_slice, block_size);
    new_token_slot_ids.emplace_back(slot_id);
  }

  draft_input.positions = torch::tensor(new_positions, int_options);
  // update the input_params
  input_params.kv_max_seq_len = kv_max_seq_len;
  input_params.kv_seq_lens_vec = kv_seq_lens_vec;
  input_params.kv_seq_lens = torch::tensor(kv_seq_lens_vec, int_options);
  input_params.new_cache_slots = torch::tensor(new_token_slot_ids, int_options);
}

void SpeculativeWorkerImpl::prepare_validate_inputs(
    const ForwardInput& input,
    ForwardInput& validate_input) {
  validate_input = input.to(device_, dtype_);
  auto& input_params = validate_input.input_params;
  torch::TensorOptions int_options = validate_input.token_ids.options();

  constexpr int32_t position_offset = 1;
  const int32_t num_speculative_tokens = options_.num_speculative_tokens();
  const int32_t num_sequences = input_params.num_sequences;
  const int32_t num_val_tokens = num_speculative_tokens + 1;
  const int32_t total_num_val_tokens = num_sequences * num_val_tokens;
  const int32_t block_size = options_.block_size();

  std::vector<std::vector<int32_t>> draft_tokens;
  draft_tokens.reserve(num_speculative_tokens);
  for (int i = 0; i < num_speculative_tokens; ++i) {
    draft_tokens.emplace_back(std::vector(num_sequences, -1 * (i + 1)));
  }

  torch::Tensor token_ids = safe_to(input.token_ids, torch::kCPU);
  Slice<int32_t> tokens_ids_slice = {token_ids.data_ptr<int32_t>(),
                                     static_cast<size_t>(token_ids.numel())};
  torch::Tensor positions = safe_to(input.positions, torch::kCPU);
  Slice<int32_t> positions_slice = {positions.data_ptr<int32_t>(),
                                    static_cast<size_t>(positions.numel())};
  Slice<int32_t> kv_seq_lens_slice = input_params.kv_seq_lens_vec;
  torch::Tensor block_tables = safe_to(input_params.block_tables, torch::kCPU);

  std::vector<int32_t> new_token_ids;
  std::vector<int32_t> new_positions;
  new_token_ids.reserve(total_num_val_tokens);
  new_positions.reserve(total_num_val_tokens);
  std::vector<int32_t> kv_seq_lens_vec = {};
  std::vector<int32_t> q_seq_lens_vec = {};
  std::vector<int32_t> new_token_slot_ids;
  std::vector<std::vector<int32_t>> block_tables_vec;

  int32_t kv_max_seq_len = 0;
  for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
    new_token_ids.emplace_back(tokens_ids_slice[seq_id]);
    new_positions.emplace_back(positions_slice[seq_id] + position_offset);
    for (int32_t j = 0; j < num_speculative_tokens; ++j) {
      new_token_ids.emplace_back(draft_tokens[j][seq_id]);
      new_positions.emplace_back(positions_slice[seq_id] + j + 1 +
                                 position_offset);
    }

    torch::Tensor block_table = block_tables[seq_id];
    Slice<int32_t> block_table_slice = {
        block_table.data_ptr<int32_t>(),
        static_cast<size_t>(block_table.numel())};

    // process kv length and q length
    if (FLAGS_enable_atb_spec_kernel) {
      // expand the num of decode tokens for each batch in the batch for
      // validation
      kv_seq_lens_vec.emplace_back(kv_seq_lens_slice[seq_id] +
                                   num_speculative_tokens + position_offset);
      q_seq_lens_vec.emplace_back(num_val_tokens);
      // update max for NPU: direct format, compare with new value
      if (kv_seq_lens_vec.back() > kv_max_seq_len) {
        kv_max_seq_len = kv_seq_lens_vec.back();
      }
    } else {
      // expand the batch sizes for validation
      //  and update max for MLU/CUDA: cumulative format, compare with new value
      batch_expansion_process_seq_lens(kv_seq_lens_vec,
                                       q_seq_lens_vec,
                                       block_tables_vec,
                                       kv_max_seq_len,
                                       kv_seq_lens_slice,
                                       block_table_slice,
                                       seq_id,
                                       position_offset,
                                       num_val_tokens);
    }

    // process slot id
    int32_t start_position = positions_slice[seq_id] + position_offset;
    auto slot_ids = kv_cache_slots(start_position,
                                   num_val_tokens,
                                   block_table_slice,
                                   options_.block_size());
    new_token_slot_ids.insert(
        new_token_slot_ids.end(), slot_ids.begin(), slot_ids.end());
  }

  validate_input.token_ids = torch::tensor(new_token_ids, int_options);
  validate_input.positions = torch::tensor(new_positions, int_options);
  // update the input_params
  if (!FLAGS_enable_atb_spec_kernel) {
    input_params.num_sequences = total_num_val_tokens;
    input_params.q_max_seq_len = 1;
    input_params.batch_forward_type = BatchForwardType::DECODE;
  } else {
    input_params.q_max_seq_len = num_val_tokens;
    input_params.batch_forward_type = BatchForwardType::CHUNKED_PREFILL;
  }
  input_params.q_seq_lens_vec = std::move(q_seq_lens_vec);
  input_params.q_seq_lens =
      torch::tensor(input_params.q_seq_lens_vec, int_options);
  input_params.kv_max_seq_len = kv_max_seq_len;
  input_params.kv_seq_lens_vec = std::move(kv_seq_lens_vec);
  input_params.kv_seq_lens =
      torch::tensor(input_params.kv_seq_lens_vec, int_options);
  input_params.new_cache_slots = torch::tensor(new_token_slot_ids, int_options);
  if (!FLAGS_enable_atb_spec_kernel) {
    util::pad_2d_vector(block_tables_vec, /*pad_value=*/0);
    input_params.block_tables =
        create_2d_tensor(block_tables_vec, torch::kInt).to(device_);
  }

  // update the sampling_params
  update_sampling_params(
      validate_input.sampling_params, num_val_tokens, total_num_val_tokens);
}

SampleOutput SpeculativeWorkerImpl::validate(
    const SamplingParameters& sampling_params,
    const std::vector<ForwardOutput>& draft_outputs,
    const ForwardOutput& target_output) {
  const int32_t num_target_tokens =
      target_output.sample_output.next_tokens.numel();
  const int32_t num_val_tokens = options_.num_speculative_tokens() + 1;
  CHECK_EQ(num_target_tokens % num_val_tokens, 0);
  const int32_t batch_size = num_target_tokens / num_val_tokens;
  const int32_t vocab_size = target_output.logits.size(/*dim=*/-1);

  using torch::indexing::None;
  using ISlice = torch::indexing::Slice;
  auto bonus_token_ids =
      target_output.sample_output.next_tokens
          .index({"...", ISlice(num_val_tokens - 1, None, num_val_tokens)})
          .view({-1, 1});

  // [batch_size, n_speculative_tokens, vocab_size]
  auto target_logits =
      target_output.logits.view({batch_size, num_val_tokens, vocab_size});

  // prepare input for rejection sampling
  std::vector<torch::Tensor> draft_token_ids_vec;
  std::vector<torch::Tensor> draft_probs_vec;
  for (const auto& draft_output : draft_outputs) {
    auto draft_token_ids =
        draft_output.sample_output.next_tokens.view({batch_size, 1});
    auto draft_probs =
        draft_output.sample_output.probs.view({{batch_size, 1, vocab_size}});
    draft_token_ids_vec.push_back(draft_token_ids);
    draft_probs_vec.push_back(draft_probs);
  }

  // concatenate the draft token ids and probs along the last dimension
  const auto draft_token_ids =
      torch::cat(draft_token_ids_vec, /*dim=*/1).to(bonus_token_ids);
  const auto draft_probs =
      torch::cat(draft_probs_vec, /*dim=*/1).to(target_logits.device());

  auto rejection_sampler =
      std::make_unique<RejectionSampler>(sampling_params.do_sample,
                                         sampling_params.all_random_sample,
                                         sampling_params.all_greedy_sample,
                                         target_output.logprobs,
                                         target_output.max_top_logprobs,
                                         rate_controller_);

  // get the accepted tokens
  SampleOutput sample_output =
      rejection_sampler->forward(draft_token_ids,
                                 draft_probs,
                                 target_logits,
                                 bonus_token_ids,
                                 /*mask_out_rejected_tokens=*/true);

  // process embedding
  auto embeddings = target_output.sample_output.embeddings;
  sample_output.embeddings =
      embeddings.view({batch_size, num_val_tokens, embeddings.size(-1)});

  // metrics
  torch::Tensor mask = (sample_output.next_tokens == -1).to(torch::kInt64);
  size_t count = mask.sum().item<int64_t>();
  size_t num_draft_tokens = num_target_tokens - batch_size;
  COUNTER_ADD(speculative_num_draft_tokens_total, num_draft_tokens);
  COUNTER_ADD(speculative_num_accepted_tokens_total, num_draft_tokens - count);

  return sample_output;
}

ForwardInput SpeculativeWorkerImpl::update_input_by_last_step_output(
    ForwardInput& inputs) {
  // only process decode batch, so prepare draft input here.
  ForwardInput& new_inputs = inputs;

  auto& input_params = new_inputs.input_params;
  const int32_t num_sequences = input_params.num_sequences;
  int32_t block_size = options_.block_size();

  torch::Tensor token_ids = safe_to(inputs.token_ids, torch::kCPU);
  Slice<int32_t> tokens_ids_slice = {token_ids.data_ptr<int32_t>(),
                                     static_cast<size_t>(token_ids.numel())};
  torch::Tensor positions = safe_to(inputs.positions, torch::kCPU);
  Slice<int32_t> positions_slice = {positions.data_ptr<int32_t>(),
                                    static_cast<size_t>(positions.numel())};
  // Get the tokens generated in the last step (flattened for easier indexing)
  torch::Tensor last_token_ids = safe_to(
      last_step_output_.sample_output.next_tokens.flatten(), torch::kCPU);
  Slice<int64_t> last_tokens_ids_slice = {
      last_token_ids.data_ptr<int64_t>(),
      static_cast<size_t>(last_token_ids.numel())};

  // Determine how many tokens were decoded in the last step
  // If the output is 2D, it means multiple tokens were generated per sequence
  int32_t last_step_decode_num = 1;
  if (last_step_output_.sample_output.next_tokens.dim() == 2) {
    last_step_decode_num = last_step_output_.sample_output.next_tokens.size(1);
  }

  Slice<int32_t> kv_seq_lens_slice = input_params.kv_seq_lens_vec;
  torch::Tensor block_tables = safe_to(input_params.block_tables, torch::kCPU);

  std::vector<int32_t> new_token_ids;
  std::vector<int32_t> new_positions;
  std::vector<int32_t> kv_seq_lens_vec = {};
  std::vector<int32_t> new_token_slot_ids;
  new_token_ids.reserve(num_sequences);
  new_positions.reserve(num_sequences);
  kv_seq_lens_vec.reserve(num_sequences);
  new_token_slot_ids.reserve(num_sequences);

  // Initialize kv_max_seq_len to 0
  int32_t kv_max_seq_len = 0;

  // Process each sequence to get the correct token ID and position for the next
  // step
  for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
    int32_t postion_offset = 0;
    int32_t last_step_token_id = 0;

    // If the token ID is non-negative, it's a direct token ID (not a
    // placeholder)
    if (tokens_ids_slice[seq_id] >= 0) {
      last_step_token_id = tokens_ids_slice[seq_id];
    } else {
      // Negative token IDs are placeholders that need to be resolved from
      // last_step_output_ The absolute value minus 1 gives the index into the
      // last step's output
      int32_t last_step_index = -1 * tokens_ids_slice[seq_id] - 1;
      last_step_index = last_step_index * last_step_decode_num;
      postion_offset = -1;
      for (int i = 0; i < last_step_decode_num; ++i) {
        int32_t token_id = last_tokens_ids_slice[last_step_index + i];
        if (token_id >= 0) {
          last_step_token_id = token_id;
          postion_offset += 1;
        }
      }
    }

    new_token_ids.emplace_back(last_step_token_id);
    new_positions.emplace_back(positions_slice[seq_id] + postion_offset);
    update_kv_seq_lens_vec(kv_seq_lens_vec,
                           kv_seq_lens_slice,
                           seq_id,
                           postion_offset,
                           kv_max_seq_len);

    // Calculate the new cache slot ID based on the position offset
    // This handles cases where we need to move to a different block
    torch::Tensor block_table = block_tables[seq_id];
    Slice<int32_t> block_table_slice = {
        block_table.data_ptr<int32_t>(),
        static_cast<size_t>(block_table.numel())};
    int32_t slot_id =
        kv_cache_slot_id(new_positions.back(), block_table_slice, block_size);
    new_token_slot_ids.emplace_back(slot_id);
  }

  // Create new tensors with updated values
  torch::TensorOptions int_options = inputs.token_ids.options();
  new_inputs.token_ids = torch::tensor(new_token_ids, int_options);
  new_inputs.positions = torch::tensor(new_positions, int_options);
  // update the input_params
  input_params.kv_max_seq_len = kv_max_seq_len;
  input_params.kv_seq_lens_vec = std::move(kv_seq_lens_vec);
  input_params.kv_seq_lens =
      torch::tensor(input_params.kv_seq_lens_vec, int_options);
  input_params.new_cache_slots = torch::tensor(new_token_slot_ids, int_options);

  return new_inputs.to(device_, dtype_);
}

void SpeculativeWorkerImpl::update_sampling_params(
    SamplingParameters& sampling_params,
    const int32_t num_val_tokens,
    const int32_t total_num_val_tokens) {
  std::vector<int32_t> selected_token_idxes_vec;
  selected_token_idxes_vec.reserve(total_num_val_tokens);
  for (int32_t i = 0; i < total_num_val_tokens; i++) {
    selected_token_idxes_vec.emplace_back(i);
  }
  torch::Tensor selected_token_idxes = torch::tensor(selected_token_idxes_vec);

  // sample_idxes equals to selected_token_idxes since only process decode batch
  sampling_params.selected_token_idxes = selected_token_idxes.to(device_);
  sampling_params.sample_idxes = selected_token_idxes.to(device_);

  TENSOR_REPEAT(sampling_params.frequency_penalties, num_val_tokens);
  TENSOR_REPEAT(sampling_params.presence_penalties, num_val_tokens);
  TENSOR_REPEAT(sampling_params.repetition_penalties, num_val_tokens);
  TENSOR_REPEAT(sampling_params.temperatures, num_val_tokens);
  TENSOR_REPEAT(sampling_params.top_p, num_val_tokens);
  TENSOR_REPEAT(sampling_params.top_k, num_val_tokens);
  TENSOR_REPEAT(sampling_params.unique_token_ids, num_val_tokens);
  TENSOR_REPEAT(sampling_params.unique_token_counts, num_val_tokens);
  TENSOR_REPEAT(sampling_params.unique_token_ids_lens, num_val_tokens);
  TENSOR_REPEAT(sampling_params.do_sample, num_val_tokens);
}

void SpeculativeWorkerImpl::prepare_work_before_execute(
    const ForwardInput& input,
    ForwardInput& processed_input) {
  WorkerImpl::prepare_work_before_execute(input, processed_input);
  if (input.input_params.batch_forward_type.is_decode() &&
      enable_schedule_overlap()) {
    processed_input.token_ids = safe_to(processed_input.token_ids, torch::kCPU);
    processed_input.positions = safe_to(processed_input.positions, torch::kCPU);
  }
}
}  // namespace xllm

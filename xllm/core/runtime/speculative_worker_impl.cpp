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
#include "framework/sampling/rejection_sampler.h"
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

inline bool check_is_prefill(const std::vector<int>& q_seq_lens_vec) {
  for (auto q_len : q_seq_lens_vec) {
    if (q_len > 1) {
      return true;
    }
  }
  return false;
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
  runtime_options.num_decoding_tokens(1).num_speculative_tokens(0);
  draft_impl_ =
      std::make_unique<LLMWorkerImpl>(parallel_args, device, runtime_options);
}

bool SpeculativeWorkerImpl::init_model(const std::string& model_weights_path) {
  // initialize model
  bool result = true;
  if (impl_->get_status() == WorkerImpl::Status::UNINITIALIZED) {
    result = impl_->WorkerImpl::init_model(model_weights_path);
    if (result) {
      dtype_ = impl_->dtype();
      embedding_size_ = impl_->hidden_size();
    }
  } else {
    CHECK_EQ(draft_impl_->get_status(), WorkerImpl::Status::UNINITIALIZED);
    result = draft_impl_->WorkerImpl::init_model(model_weights_path);
  }

  if (draft_impl_->get_status() == WorkerImpl::Status::LOADED) {
    // Deepseek MTP
    auto head = impl_->get_lm_head();
    draft_impl_->set_lm_head(head);
    auto word_embedding = impl_->get_word_embedding();
    draft_impl_->set_word_embedding(word_embedding);
  }
  return result;
}

bool SpeculativeWorkerImpl::allocate_kv_cache(
    const std::vector<std::vector<int64_t>>& kv_cache_shape) {
  // init embedding cache, using total number of blocks
  if (impl_->get_status() == WorkerImpl::Status::LOADED) {
    token_allocator_ = std::make_shared<TokenCacheAllocator>(
        kv_cache_shape[0][0], options_.num_speculative_tokens());
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
  }
  return true;
}
#endif

std::optional<ForwardOutput> SpeculativeWorkerImpl::step(
    const BatchedForwardInputs& inputs) {
  // all micro batches in multi stream parallel share the same
  // prefill/decode stage, use inputs[0] here
  CHECK_EQ(inputs.micro_inputs.size(), 1);
  auto& input = inputs.micro_inputs[0];
  if (input.token_ids.numel() == 0) {
    return step_empty(const_cast<ForwardInput&>(input));
  }

  // TODO: support data parallel case
  if (check_is_prefill(input.input_params.q_seq_lens_vec)) {
    return step_prefill(input);
  } else {
    return step_decode(input);
  }
}

std::optional<ForwardOutput> SpeculativeWorkerImpl::step_empty(
    ForwardInput& input) {
  int32_t draft_step = options_.num_speculative_tokens();
  if (!check_is_prefill(input.input_params.q_seq_lens_vec)) {
    for (auto& it : input.input_params.dp_global_token_nums) {
      it *= options_.num_speculative_tokens() + 1;
    }
  } else {
    draft_step = 1;
  }
  auto future = get_output_async(impl_.get(), input);
  ForwardOutput output = std::move(future).get().value();

  for (size_t i = 0; i < draft_step; ++i) {
    auto draft_future = get_output_async(draft_impl_.get(), input);
    ForwardOutput draft_output = std::move(draft_future).get().value();
  }
  return output;
}

std::optional<ForwardOutput> SpeculativeWorkerImpl::step_prefill(
    const ForwardInput& input) {
  Timer timer;
  // run the target model to get first token and hidden states
  auto future = get_output_async(impl_.get(), input);
  ForwardInput draft_input, next_step_input;
  prepare_first_prefill_input(input, draft_input);
  ForwardOutput output = std::move(future).get().value();
  COUNTER_ADD(speculative_execution_latency_seconds_target,
              timer.elapsed_seconds());

  // prepare first step input for draft model
  auto& embeddings = output.sample_output.embeddings;
  auto next_tokens = safe_to(output.sample_output.next_tokens, torch::kInt);
  if (embeddings.defined()) {
    draft_input.input_params.mm_data =
        MMData(MMType::EMBEDDING, {{"embedding", embeddings}});
  }
  if (next_tokens.defined()) {
    auto& token_ids = draft_input.token_ids;
    auto mask = (token_ids == -1);
    token_ids.masked_scatter_(mask, next_tokens);
  }

  // generate draft tokens
  timer.reset();
  auto draft_future = get_output_async(draft_impl_.get(), draft_input);
  ForwardOutput draft_output = std::move(draft_future).get().value();
  COUNTER_ADD(speculative_execution_latency_seconds_draft,
              timer.elapsed_seconds());

  // cache draft tokens
  if (input.sampling_params.selected_token_idxes.defined()) {
    std::vector<ForwardOutput> draft_outputs = {draft_output};
    token_allocator_->write(input.input_params.embedding_ids, draft_outputs);
  }

  if (!enable_schedule_overlap() && !driver_ && !dp_driver_) {
    return std::nullopt;
  }

  output.sample_output.embeddings = torch::Tensor();
  return output;
}

std::optional<ForwardOutput> SpeculativeWorkerImpl::step_decode(
    const ForwardInput& input) {
  // TODO: More work need to support n-gram and native speculative decoding.
  Timer timer;
  // read draft tokens
  std::vector<std::vector<int32_t>> draft_tokens =
      token_allocator_->read(input.input_params.embedding_ids);

  // prepare validate input of target model
  ForwardInput validate_input;
  prepare_validate_input(input, validate_input, draft_tokens);

  // run the target model to get the verification scores
  timer.reset();
  auto future = get_output_async(impl_.get(), validate_input);
  ForwardOutput target_output = std::move(future).get().value();
  COUNTER_ADD(speculative_execution_latency_seconds_target,
              timer.elapsed_seconds());

  // verify the proposals with target
  timer.reset();
  SampleOutput val_output =
      validate(input.sampling_params, draft_tokens, target_output);
  COUNTER_ADD(speculative_execution_latency_seconds_validation,
              timer.elapsed_seconds());

  // prepare first decode input of draft model
  ForwardInput draft_input;
  prepare_first_decode_input(validate_input, draft_input, val_output);

  // generate draft tokens
  std::vector<ForwardOutput> draft_outputs;
  ForwardInput next_step_input;
  timer.reset();
  for (size_t i = 0; i < options_.num_speculative_tokens(); ++i) {
    auto future = get_output_async(draft_impl_.get(), draft_input);
    // prepare draft input for next step
    if (i < options_.num_speculative_tokens() - 1) {
      prepare_draft_input(draft_input, next_step_input, 1);
    }
    draft_outputs.push_back(std::move(future).get().value());
    // update input of next step
    if (i < options_.num_speculative_tokens() - 1) {
      draft_input = next_step_input;
      auto& last_output = draft_outputs.back().sample_output;
      draft_input.token_ids = safe_to(last_output.next_tokens, torch::kInt);
      draft_input.input_params.mm_data =
          MMData(MMType::EMBEDDING, {{"embedding", last_output.embeddings}});
    }
  }
  COUNTER_ADD(speculative_execution_latency_seconds_draft,
              timer.elapsed_seconds());

  // cache draft tokens
  token_allocator_->write(input.input_params.embedding_ids, draft_outputs);

  if (!enable_schedule_overlap() && !driver_ && !dp_driver_) {
    return std::nullopt;
  }

  val_output.embeddings = torch::Tensor();
  target_output.sample_output = val_output;
  return target_output;
}

void SpeculativeWorkerImpl::prepare_first_prefill_input(
    const ForwardInput& input,
    ForwardInput& draft_input) {
  draft_input = input.to(device_, dtype_);
  auto& input_params = draft_input.input_params;
  auto& extra_token_ids = input_params.extra_token_ids;

  torch::Tensor token_ids = safe_to(input.token_ids, torch::kCPU);
  Slice<int32_t> tokens_ids_slice = {token_ids.data_ptr<int32_t>(),
                                     input.token_ids.numel()};

  int32_t start_idx = 0;
  std::vector<int32_t> new_token_ids;
  new_token_ids.reserve(input.token_ids.numel());
  for (size_t seq_id = 0; seq_id < input_params.num_sequences; ++seq_id) {
    int32_t q_len = 0;
    q_len = input_params.q_seq_lens_vec[seq_id];
    Slice<int32_t> tokens_ids_slice_i =
        tokens_ids_slice.slice(start_idx + 1, start_idx + q_len);
    start_idx += q_len;
    new_token_ids.insert(new_token_ids.end(),
                         tokens_ids_slice_i.begin(),
                         tokens_ids_slice_i.end());
    new_token_ids.emplace_back(extra_token_ids[seq_id]);
  }
  draft_input.token_ids =
      torch::tensor(new_token_ids, draft_input.positions.options());
}

void SpeculativeWorkerImpl::prepare_first_decode_input(
    const ForwardInput& input,
    ForwardInput& draft_input,
    const SampleOutput val_output) {
  // prepare first step input for MTP in decoding phase (Like Eagle).
  draft_input = input.to(device_, dtype_);
  torch::TensorOptions int_options = draft_input.positions.options();

  auto& input_params = draft_input.input_params;
  int32_t val_step = 1;
  if (!FLAGS_enable_atb_spec_kernel) {
    val_step = options_.num_speculative_tokens() + 1;
    input_params.num_sequences = input_params.num_sequences / val_step;
  }
  const int32_t num_sequences = input_params.num_sequences;

  std::vector<int32_t> new_tokens;
  std::vector<int32_t> new_positions;
  std::vector<int32_t> kv_seq_lens_vec = {};
  std::vector<int32_t> q_seq_lens_vec = {};
  std::vector<int32_t> new_token_slot_ids;
  std::vector<int32_t> selected_token_idxes_vec;
  selected_token_idxes_vec.reserve(num_sequences);
  std::vector<torch::Tensor> new_embeddings;
  new_embeddings.reserve(num_sequences);

  auto& embeddings = val_output.embeddings;
  auto val_token_ids = safe_to(val_output.next_tokens, torch::kCPU);
  torch::Tensor positions = safe_to(input.positions, torch::kCPU);
  Slice<int32_t> positions_slice = {positions.data_ptr<int32_t>(),
                                    positions.numel()};
  torch::Tensor kv_seq_lens = safe_to(input_params.kv_seq_lens, torch::kCPU);
  Slice<int32_t> kv_seq_lens_slice = {kv_seq_lens.data_ptr<int32_t>(),
                                      kv_seq_lens.numel()};
  torch::Tensor block_tables = safe_to(input_params.block_tables, torch::kCPU);

  for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
    int32_t start_offset = seq_id * val_step;
    torch::Tensor val_tokens = val_token_ids[seq_id];
    Slice<int64_t> val_tokens_slice = {val_tokens.data_ptr<int64_t>(),
                                       val_tokens.numel()};
    torch::Tensor block_table = block_tables[start_offset];
    Slice<int32_t> block_table_slice = {block_table.data_ptr<int32_t>(),
                                        block_table.numel()};

    int32_t num_val_tokens = 0;
    for (; num_val_tokens < val_tokens_slice.size(); ++num_val_tokens) {
      if (val_tokens_slice[num_val_tokens] > 0) {
        new_tokens.emplace_back(val_tokens_slice[num_val_tokens]);
        new_positions.emplace_back(
            positions_slice[start_offset + num_val_tokens]);
        new_embeddings.emplace_back(embeddings[start_offset + num_val_tokens]);
      } else {
        break;
      }
    }
    selected_token_idxes_vec.emplace_back(new_positions.size() - 1);
    q_seq_lens_vec.emplace_back(num_val_tokens);
    kv_seq_lens_vec.emplace_back(kv_seq_lens_slice[start_offset] +
                                 num_val_tokens - 1);

    auto slot_ids = kv_cache_slots(positions_slice[start_offset],
                                   num_val_tokens,
                                   block_table_slice,
                                   options_.block_size());
    new_token_slot_ids.insert(
        new_token_slot_ids.end(), slot_ids.begin(), slot_ids.end());
  }
  draft_input.token_ids = torch::tensor(new_tokens, int_options);
  draft_input.positions = torch::tensor(new_positions, int_options);
  input_params.decode_seq_range = {-1, -1};
  input_params.q_max_seq_len =
      *std::max_element(q_seq_lens_vec.begin(), q_seq_lens_vec.end());
  input_params.q_seq_lens_vec = std::move(q_seq_lens_vec);
  input_params.q_seq_lens =
      torch::tensor(input_params.q_seq_lens_vec, int_options);
  input_params.kv_max_seq_len =
      *std::max_element(kv_seq_lens_vec.begin(), kv_seq_lens_vec.end());
  input_params.kv_seq_lens_vec = std::move(kv_seq_lens_vec);
  input_params.kv_seq_lens =
      torch::tensor(input_params.kv_seq_lens_vec, int_options);
  input_params.new_cache_slots = torch::tensor(new_token_slot_ids, int_options);
  input_params.mm_data =
      MMData(MMType::EMBEDDING,
             {{"embedding", torch::stack(new_embeddings).to(device_)}});

  auto& sampling_params = draft_input.sampling_params;
  sampling_params.selected_token_idxes =
      torch::tensor(selected_token_idxes_vec, int_options);
  sampling_params.sample_idxes =
      sampling_params.selected_token_idxes.to(device_);
  sampling_params.do_sample =
      torch::zeros({selected_token_idxes_vec.size()}, int_options);
  sampling_params.all_random_sample = true;
}

void SpeculativeWorkerImpl::prepare_draft_input(const ForwardInput& input,
                                                ForwardInput& draft_input,
                                                const int64_t offset) {
  // prepare next step draft's input for MTP(Like Eagle).
  draft_input = input.to(device_, dtype_);

  auto& input_params = draft_input.input_params;
  const int32_t num_sequences = input_params.num_sequences;
  torch::Tensor positions = safe_to(input.positions, torch::kCPU);
  Slice<int32_t> positions_slice = {positions.data_ptr<int32_t>(),
                                    positions.numel()};

  std::vector<int32_t> new_positions;
  new_positions.reserve(num_sequences);
  std::vector<int32_t> kv_seq_lens_vec = {};
  kv_seq_lens_vec.reserve(num_sequences);
  std::vector<int32_t> q_seq_lens_vec = {};
  q_seq_lens_vec.reserve(num_sequences);
  std::vector<int32_t> new_token_slot_ids;
  new_token_slot_ids.reserve(num_sequences);
  std::vector<int32_t> selected_token_idxes_vec;
  selected_token_idxes_vec.reserve(num_sequences);

  torch::Tensor kv_seq_lens = safe_to(input_params.kv_seq_lens, torch::kCPU);
  Slice<int32_t> kv_seq_lens_slice = {kv_seq_lens.data_ptr<int32_t>(),
                                      kv_seq_lens.numel()};
  torch::Tensor q_seq_lens = safe_to(input_params.q_seq_lens, torch::kCPU);
  Slice<int32_t> q_seq_lens_slice = {q_seq_lens.data_ptr<int32_t>(),
                                     q_seq_lens.numel()};
  torch::Tensor block_tables = safe_to(input_params.block_tables, torch::kCPU);

  int32_t start_idx = -1;
  for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
    start_idx += q_seq_lens_slice[seq_id];
    new_positions.emplace_back(positions_slice[start_idx] + offset);
    selected_token_idxes_vec.emplace_back(new_positions.size() - 1);
    q_seq_lens_vec.emplace_back(1);
    kv_seq_lens_vec.emplace_back(kv_seq_lens_slice[seq_id] + offset);
    torch::Tensor block_table = block_tables[seq_id];
    Slice<int32_t> block_table_slice = {block_table.data_ptr<int32_t>(),
                                        block_table.numel()};
    auto slot_id = kv_cache_slot_id(
        new_positions.back(), block_table_slice, options_.block_size());
    new_token_slot_ids.emplace_back(slot_id);
  }

  torch::TensorOptions int_options = draft_input.token_ids.options();
  draft_input.positions = torch::tensor(new_positions, int_options);
  input_params.q_max_seq_len = 1;
  input_params.q_seq_lens_vec = std::move(q_seq_lens_vec);
  input_params.q_seq_lens =
      torch::tensor(input_params.q_seq_lens_vec, int_options);
  input_params.kv_max_seq_len = input_params.kv_max_seq_len + offset;
  input_params.kv_seq_lens_vec = std::move(kv_seq_lens_vec);
  input_params.kv_seq_lens =
      torch::tensor(input_params.kv_seq_lens_vec, int_options);
  input_params.decode_seq_range = {-1, -1};
  input_params.new_cache_slots = torch::tensor(new_token_slot_ids, int_options);

  auto& sampling_params = draft_input.sampling_params;
  auto selected_token_idxes =
      torch::tensor(selected_token_idxes_vec, int_options);
  sampling_params.selected_token_idxes = selected_token_idxes;
  sampling_params.sample_idxes = selected_token_idxes;
  sampling_params.do_sample =
      torch::zeros({selected_token_idxes_vec.size()}, int_options);
  sampling_params.all_random_sample = true;
}

void SpeculativeWorkerImpl::prepare_validate_input(
    const ForwardInput& input,
    ForwardInput& validate_input,
    std::vector<std::vector<int32_t>>& draft_tokens) {
  validate_input = input.to(device_, dtype_);
  torch::TensorOptions int_options = validate_input.token_ids.options();

  auto& input_params = validate_input.input_params;
  const int32_t num_speculative_tokens = options_.num_speculative_tokens();
  const int32_t num_sequences = input_params.num_sequences;
  const int32_t num_val_tokens = num_speculative_tokens + 1;
  const int32_t total_num_val_tokens = num_sequences * num_val_tokens;

  torch::Tensor token_ids = safe_to(input.token_ids, torch::kCPU);
  Slice<int32_t> tokens_ids_slice = {token_ids.data_ptr<int32_t>(),
                                     token_ids.numel()};
  torch::Tensor positions = safe_to(input.positions, torch::kCPU);
  Slice<int32_t> positions_slice = {positions.data_ptr<int32_t>(),
                                    positions.numel()};

  // get last step output token ids
  torch::Tensor last_token_ids;
  if (enable_schedule_overlap()) {
    last_token_ids =
        safe_to(last_step_output_.sample_output.next_tokens, torch::kCPU);
    CHECK_EQ(num_sequences, last_token_ids.size(0));
  }

  std::vector<int32_t> new_token_ids;
  std::vector<int32_t> new_positions;
  new_token_ids.reserve(total_num_val_tokens);
  new_positions.reserve(total_num_val_tokens);
  std::vector<int32_t> kv_seq_lens_vec = {};
  std::vector<int32_t> q_seq_lens_vec = {};
  std::vector<int32_t> new_token_slot_ids;
  new_token_slot_ids.reserve(total_num_val_tokens);
  std::vector<std::vector<int32_t>> block_tables_vec;

  torch::Tensor kv_seq_lens = safe_to(input_params.kv_seq_lens, torch::kCPU);
  Slice<int32_t> kv_seq_lens_slice = {kv_seq_lens.data_ptr<int32_t>(),
                                      kv_seq_lens.numel()};
  torch::Tensor block_tables = safe_to(input_params.block_tables, torch::kCPU);

  for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
    // get right token id and position
    int32_t postion_offset = 0;
    int32_t last_step_token_id = 0;
    if (tokens_ids_slice[seq_id] >= 0) {
      last_step_token_id = tokens_ids_slice[seq_id];
    } else {
      // Only schedule overlap have this branch.
      int32_t last_step_index = -1 * tokens_ids_slice[seq_id] - 1;
      torch::Tensor last_tokens = last_token_ids[last_step_index];
      Slice<int64_t> last_tokens_slice = {last_tokens.data_ptr<int64_t>(),
                                          last_tokens.numel()};
      postion_offset = -1;
      for (int i = 0; i < last_tokens_slice.size(); ++i) {
        if (last_tokens_slice[i] >= 0) {
          last_step_token_id = last_tokens_slice[i];
          postion_offset += 1;
        }
      }
    }

    new_token_ids.emplace_back(last_step_token_id);
    new_positions.emplace_back(positions_slice[seq_id] + postion_offset);
    for (int32_t j = 0; j < num_speculative_tokens; ++j) {
      new_token_ids.emplace_back(draft_tokens[seq_id][j]);
      new_positions.emplace_back(positions_slice[seq_id] + postion_offset + j +
                                 1);
    }

    torch::Tensor block_table = block_tables[seq_id];
    Slice<int32_t> block_table_slice = {block_table.data_ptr<int32_t>(),
                                        block_table.numel()};
    // process kv length and q length
    if (FLAGS_enable_atb_spec_kernel) {
      kv_seq_lens_vec.emplace_back(kv_seq_lens_slice[seq_id] +
                                   num_speculative_tokens + postion_offset);
      q_seq_lens_vec.emplace_back(num_val_tokens);
    } else {
      for (int32_t i = postion_offset; i < num_val_tokens + postion_offset;
           ++i) {
        q_seq_lens_vec.emplace_back(1);
        kv_seq_lens_vec.emplace_back(kv_seq_lens_slice[seq_id] + i);
        block_tables_vec.emplace_back(block_table_slice);
      }
    }

    // process slot id
    int32_t start_position = positions_slice[seq_id] + postion_offset;
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
    input_params.decode_seq_range = {0, total_num_val_tokens - 1};
  } else {
    input_params.q_max_seq_len = num_val_tokens;
    input_params.decode_seq_range = {-1, -1};
  }
  input_params.q_seq_lens_vec = std::move(q_seq_lens_vec);
  input_params.q_seq_lens =
      torch::tensor(input_params.q_seq_lens_vec, int_options);
  input_params.kv_max_seq_len =
      *std::max_element(kv_seq_lens_vec.begin(), kv_seq_lens_vec.end());
  input_params.kv_seq_lens_vec = std::move(kv_seq_lens_vec);
  input_params.kv_seq_lens =
      torch::tensor(input_params.kv_seq_lens_vec, int_options);
  input_params.new_cache_slots = torch::tensor(new_token_slot_ids, int_options);
  if (!FLAGS_enable_atb_spec_kernel) {
    util::pad_2d_vector(block_tables_vec, /*pad_value=*/0);
    input_params.block_tables =
        create_2d_tensor(block_tables_vec, torch::kInt).to(device_);
  }
  for (auto& it : input_params.dp_global_token_nums) {
    it *= num_val_tokens;
  }

  // update the sampling_params
  std::vector<int32_t> selected_token_idxes_vec;
  selected_token_idxes_vec.resize(total_num_val_tokens);
  for (int32_t i = 0; i < total_num_val_tokens; ++i) {
    selected_token_idxes_vec[i] = i;
  }
  torch::Tensor selected_token_idxes =
      torch::tensor(selected_token_idxes_vec, int_options);
  auto& sampling_params = validate_input.sampling_params;
  sampling_params.selected_token_idxes = selected_token_idxes;
  sampling_params.sample_idxes = selected_token_idxes;
  sampling_params.do_sample =
      torch::zeros({selected_token_idxes_vec.size()}, int_options);
  sampling_params.all_random_sample = true;
}

SampleOutput SpeculativeWorkerImpl::validate(
    const SamplingParameters& sampling_params,
    std::vector<std::vector<int32_t>>& draft_tokens,
    const ForwardOutput& target_output) {
  const int32_t num_target_tokens =
      target_output.sample_output.next_tokens.numel();
  const int32_t num_speculative_tokens = options_.num_speculative_tokens();
  const int32_t num_val_tokens = num_speculative_tokens + 1;
  CHECK_EQ(num_target_tokens % num_val_tokens, 0);
  const int32_t batch_size = draft_tokens.size();
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

  // prepare the draft token ids
  auto int_options = target_output.sample_output.next_tokens.options();
  auto draft_token_ids = create_2d_tensor(draft_tokens, torch::kInt64);
  draft_token_ids = safe_to(draft_token_ids, device_);
  auto draft_probs =
      torch::empty({batch_size, num_speculative_tokens, vocab_size},
                   target_output.logits.options());

  auto rejection_sampler =
      std::make_unique<RejectionSampler>(sampling_params.do_sample,
                                         sampling_params.all_random_sample,
                                         sampling_params.all_greedy_sample,
                                         target_output.logprobs,
                                         target_output.max_top_logprobs);

  // get the accepted tokens
  SampleOutput sample_output =
      rejection_sampler->forward(draft_token_ids,
                                 draft_probs,
                                 target_logits,
                                 bonus_token_ids,
                                 /*mask_out_rejected_tokens=*/true);

  // process embedding
  sample_output.embeddings = target_output.sample_output.embeddings;

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
  // when schedule overlap is enabled, do nothing for decode batch.
  return inputs;
}

void SpeculativeWorkerImpl::prepare_work_before_execute(
    const BatchedForwardInputs& inputs,
    BatchedForwardInputs& processed_inputs) {
  if (enable_schedule_overlap() &&
      !check_is_prefill(inputs.micro_inputs[0].input_params.q_seq_lens_vec)) {
    // when schedule overlap is enabled, do nothing for decode batch.
    processed_inputs = inputs;
  } else {
    WorkerImpl::prepare_work_before_execute(inputs, processed_inputs);
  }
}
}  // namespace xllm

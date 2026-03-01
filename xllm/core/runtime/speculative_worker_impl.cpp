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
#include "util/env_var.h"
#include "util/slice.h"
#include "util/timer.h"
#include "util/utils.h"

namespace xllm {

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
  CHECK_GE(offset, 0) << "invalid offset=" << offset;
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
#elif defined(USE_MLU) || defined(USE_CUDA) || defined(USE_ILU) || \
    defined(USE_MUSA)
  return kv_seq_lens_slice[seq_id + 1] - kv_seq_lens_slice[seq_id] + offset;
#endif
}

// Append sequence length to vector based on platform type
// For NPU: directly add the len value
// For MLU/CUDA: add using cumulative format
void append_seq_len(std::vector<int32_t>& vec, int32_t len) {
#if defined(USE_NPU)
  vec.emplace_back(len);
#elif defined(USE_MLU) || defined(USE_CUDA) || defined(USE_MUSA)
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

SpeculativeWorkerImpl::SpeculativeWorkerImpl(
    const ParallelArgs& parallel_args,
    const torch::Device& device,
    const runtime::Options& options,
    const runtime::Options& target_options)
    : WorkerImpl(parallel_args, device, options) {
  impl_ =
      std::make_unique<LLMWorkerImpl>(parallel_args, device, target_options);

  std::optional<double> fixed_acceptance_rate =
      util::get_fix_speculative_acceptance_rate();
  if (fixed_acceptance_rate.has_value()) {
    rate_controller_ = std::make_shared<RejectionSamplerRateController>(
        *fixed_acceptance_rate);
  }
}

bool SpeculativeWorkerImpl::init_model(const std::string& model_weights_path,
                                       int32_t random_seed) {
  // Base: only load target model
  bool result = true;
  if (impl_->get_status() == WorkerImpl::Status::UNINITIALIZED) {
    result = impl_->WorkerImpl::init_model(model_weights_path, random_seed);
    if (result) {
      dtype_ = impl_->dtype();
      embedding_size_ = impl_->hidden_size();
    }
  }
  enable_fused_kernel_ =
      impl_->get_optimization_config().enable_fused_spec_kernel;
  return result;
}

bool SpeculativeWorkerImpl::allocate_kv_cache(
    const std::vector<std::vector<int64_t>>& kv_cache_shape) {
  return impl_->allocate_kv_cache(kv_cache_shape);
}

#if defined(USE_NPU)
bool SpeculativeWorkerImpl::allocate_kv_cache_with_transfer(
    const uint64_t kv_cache_size,
    const std::vector<std::vector<int64_t>>& kv_cache_shape) {
  return impl_->allocate_kv_cache_with_transfer(kv_cache_size, kv_cache_shape);
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
    new_token_slot_ids.emplace_back(
        kv_cache_slot_id(new_positions.back(), block_table_slice, block_size));
  }

  CHECK_EQ(new_token_slot_ids.size(), new_token_ids.size())
      << "step-update kv slots/tokens mismatch";
  CHECK_EQ(new_positions.size(), new_token_ids.size())
      << "step-update positions/tokens mismatch";

  // Create new tensors with updated values
  torch::TensorOptions int_options = inputs.token_ids.options();
  new_inputs.token_ids = torch::tensor(new_token_ids, int_options);
  new_inputs.positions = torch::tensor(new_positions, int_options);
  // update the input_params
  input_params.kv_max_seq_len = kv_max_seq_len;
  input_params.kv_seq_lens_vec = std::move(kv_seq_lens_vec);
  input_params.kv_seq_lens =
      torch::tensor(input_params.kv_seq_lens_vec, int_options);
  // deepseek 3.2
  std::vector<int32_t> q_cu_seq_lens_vec;
  q_cu_seq_lens_vec.reserve(input_params.num_sequences);
  int32_t cum_seq_len = 0;
  for (int32_t i = 0; i < input_params.num_sequences; ++i) {
    cum_seq_len += input_params.get_q_seq_len(i);
    q_cu_seq_lens_vec.push_back(cum_seq_len);
  }
  input_params.q_cu_seq_lens = torch::tensor(q_cu_seq_lens_vec, torch::kInt);
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
    int32_t kv_len = calculate_kv_len(kv_seq_lens_slice, seq_id, /*offset=*/0);
    CHECK_EQ(start_position, kv_len)
        << "validate position/kv_len mismatch, seq_id=" << seq_id
        << ", start_position=" << start_position << ", kv_len=" << kv_len;
    auto slot_ids = kv_cache_slots(start_position,
                                   num_val_tokens,
                                   block_table_slice,
                                   options_.block_size());
    new_token_slot_ids.insert(
        new_token_slot_ids.end(), slot_ids.begin(), slot_ids.end());
  }

  CHECK_EQ(new_token_slot_ids.size(), new_token_ids.size())
      << "validate kv slots/tokens mismatch";
  CHECK_EQ(new_positions.size(), new_token_ids.size())
      << "validate positions/tokens mismatch";

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
  // deepseek 3.2
  std::vector<int32_t> q_cu_seq_lens_vec;
  q_cu_seq_lens_vec.reserve(input_params.num_sequences);
  int32_t cum_seq_len = 0;
  for (int32_t i = 0; i < input_params.num_sequences; ++i) {
    cum_seq_len += input_params.get_q_seq_len(i);
    q_cu_seq_lens_vec.push_back(cum_seq_len);
  }
  input_params.q_cu_seq_lens = torch::tensor(q_cu_seq_lens_vec, torch::kInt);
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

  // update dp_global_token_nums for dp/ep parallel
  for (auto& it : input_params.dp_global_token_nums) {
    it *= num_val_tokens;
  }
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

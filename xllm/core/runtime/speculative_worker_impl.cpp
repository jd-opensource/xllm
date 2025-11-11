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

int32_t get_new_token_slot_id(const int32_t cur_token_slot_id,
                              const int32_t block_size,
                              const int32_t offset,
                              const Slice<int32_t>& block_table_slice) {
  if (offset == 0) {
    return cur_token_slot_id;
  }
  int32_t cur_block_offset = cur_token_slot_id % block_size;
  int32_t cur_block_id = cur_token_slot_id / block_size;
  int32_t new_block_offset = cur_block_offset + offset;

  if (new_block_offset >= 0 && new_block_offset < block_size) {
    return cur_token_slot_id + offset;
  }
  // get right block id
  int32_t new_block_id = -1;
  if (new_block_offset < 0) {
    for (size_t i = 0; i < block_table_slice.size(); ++i) {
      if (cur_block_id == block_table_slice[i]) {
        if (i == 0) {
          LOG(FATAL) << "Find the first block, should never happen.";
        } else {
          new_block_id = block_table_slice[i - 1];
        }
      }
    }
  } else if (new_block_offset >= block_size) {
    for (size_t i = 0; i < block_table_slice.size(); ++i) {
      if (cur_block_id == block_table_slice[i]) {
        if (i >= block_table_slice.size() - 1) {
          LOG(FATAL) << "Find the last block, should never happen.";
        } else {
          new_block_id = block_table_slice[i + 1];
        }
      }
    }
  }
  if (new_block_id == -1) {
    LOG(FATAL) << "Fail to find block id in block table.";
  }
  new_block_offset = (new_block_offset + block_size) % block_size;
  return new_block_id * block_size + new_block_offset;
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
#if defined(USE_NPU)
  if (draft_impl_->get_status() == WorkerImpl::Status::LOADED) {
    // Deepseek MTP
    auto head = impl_->get_lm_head();
    draft_impl_->set_lm_head(head);
    auto word_embedding = impl_->get_word_embedding();
    draft_impl_->set_word_embedding(word_embedding);
  }
#endif
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
    kv_cache_transfer_->allocate_embedding(
        embedding_allocator_,
        {kv_cache_shape[0][0], embedding_size_},
        dtype_,
        device_);
  }
  return true;
}
#endif

std::optional<ForwardOutput> SpeculativeWorkerImpl::step(
    const BatchedForwardInputs& inputs) {
  // all micro batches in multi stream parallel share the same
  // prefill/decode stage, use inputs[0] here
  if (inputs.micro_inputs[0].token_ids.numel() == 0) {
    return step_empty(inputs);
  }

  // TODO: support data parallel case
  if (inputs.micro_inputs[0].input_params.q_seq_lens_vec[0] > 1) {
    return step_prefill(inputs);
  } else {
    return step_decode(inputs);
  }
}

std::optional<ForwardOutput> SpeculativeWorkerImpl::step_empty(
    const BatchedForwardInputs& inputs) {
  if (inputs.micro_inputs[0].input_params.q_seq_lens_vec[0] > 1) {
    auto output = impl_->step(inputs);
    auto draft_output = draft_impl_->step(inputs);
    return output;
  } else {
    for (size_t i = 0; i < options_.num_speculative_tokens(); ++i) {
      auto draft_future = draft_impl_->step_async(inputs);
      ForwardOutput draft_output = std::move(draft_future).get().value();
    }

    BatchedForwardInputs new_inputs = inputs;
    for (auto i = 0; i < new_inputs.micro_inputs.size(); ++i) {
      for (auto& it :
           new_inputs.micro_inputs[i].input_params.dp_global_token_nums) {
        it *= options_.num_speculative_tokens() + 1;
      }
    }
    auto future = impl_->step_async(new_inputs);
    ForwardOutput output = std::move(future).get().value();
    return output;
  }
}

std::optional<ForwardOutput> SpeculativeWorkerImpl::step_prefill(
    const BatchedForwardInputs& inputs) {
  Timer timer;
  // run the target model to get first token and hidden states
  auto future = impl_->step_async(inputs);
  // MTP (Eagle Medusa) which depend on hidden states need this step
  // The others speculative model use inputs directly
  // ForwardInput prefill_inputs;
  BatchedForwardInputs prefill_inputs;
  prepare_prefill_inputs(inputs, prefill_inputs);
  ForwardOutput output = std::move(future).get().value();
  COUNTER_ADD(speculative_execution_latency_seconds_target,
              timer.elapsed_seconds());

  // prepare input for draft model
  auto& embeddings = output.sample_output.embeddings;
  auto next_tokens = safe_to(output.sample_output.next_tokens, torch::kInt);
  auto start_idx = 0;
  auto token_start_idx = 0;
  for (auto i = 0; i < inputs.micro_inputs.size(); ++i) {
    auto offset = inputs.micro_inputs[i].input_params.num_sequences;
    auto token_offset = prefill_inputs.micro_inputs[i].token_ids.size(0);
    if (token_offset > 0) {
      prefill_inputs.micro_inputs[i].input_params.mm_data = MMData(
          MMType::EMBEDDING,
          {{"embedding", embeddings.narrow(0, token_start_idx, token_offset)}});
    }
    if (next_tokens.defined()) {
      auto& token_ids = prefill_inputs.micro_inputs[i].token_ids;
      auto mask = (token_ids == -1);
      // TODO:  support multi stream parallel case
      // token_ids.masked_scatter_(mask, next_tokens.narrow(0, start_idx,
      // offset));
      token_ids.masked_scatter_(mask, next_tokens);
    }
    start_idx += offset;
    token_start_idx += token_offset;
  }

  // generate kv cache for draft model
  timer.reset();
  auto draft_future = draft_impl_->step_async(prefill_inputs);
  ForwardOutput draft_output = std::move(draft_future).get().value();
  COUNTER_ADD(speculative_execution_latency_seconds_draft,
              timer.elapsed_seconds());

  auto concated_embedding_ids =
      inputs.micro_inputs[0].input_params.embedding_ids;
  for (auto i = 1; i < inputs.micro_inputs.size(); ++i) {
    concated_embedding_ids.insert(
        concated_embedding_ids.end(),
        inputs.micro_inputs[i].input_params.embedding_ids.begin(),
        inputs.micro_inputs[i].input_params.embedding_ids.end());
  }

  if (inputs.concated_sampling_params.selected_token_idxes.defined()) {
    embeddings = embeddings.index_select(
        /*dim=*/0, inputs.concated_sampling_params.selected_token_idxes);
    CHECK_EQ(embeddings.size(0), output.sample_output.next_tokens.size(0));
    embedding_allocator_->write(concated_embedding_ids, embeddings);
#if defined(USE_NPU)
    if (kv_cache_transfer_) {
      kv_cache_transfer_->copy_blocks(concated_embedding_ids,
                                      /*h2d*/ true);
    }
#endif
  }
  output.sample_output.embeddings = torch::Tensor();

#if defined(USE_NPU)
  for (auto i = 0; i < inputs.micro_inputs.size(); ++i) {
    if (options_.kv_cache_transfer_mode() == "PUSH" &&
        !inputs.micro_inputs[i].transfer_kv_infos.empty()) {
      auto future = kv_cache_transfer_->push_kv_blocks_async(
          inputs.micro_inputs[i].transfer_kv_infos,
          context_.get_parallel_args(),
          nullptr,
          true);
      auto out = std::move(future).get();
    }
  }
#endif
  return output;
}

void SpeculativeWorkerImpl::prepare_prefill_inputs(
    const BatchedForwardInputs& inputs,
    BatchedForwardInputs& prefill_inputs) {
  prefill_inputs.micro_inputs.reserve(inputs.micro_inputs.size());
  for (auto i = 0; i < inputs.micro_inputs.size(); ++i) {
    auto& input = inputs.micro_inputs[i];
    ForwardInput prefill_input;
    prefill_input = input.to(device_, dtype_);
    auto& input_params = prefill_input.input_params;
    auto& extra_token_ids = input_params.extra_token_ids;

    torch::Tensor token_ids = safe_to(input.token_ids, torch::kCPU);
    Slice<int32_t> tokens_ids_slice = {token_ids.data_ptr<int32_t>(),
                                       input.token_ids.numel()};

    int32_t start_idx = 0;
    std::vector<int32_t> new_token_ids;
    new_token_ids.reserve(input.token_ids.numel());
    for (size_t i = 0; i < input_params.num_sequences; ++i) {
      int32_t q_len = 0;
      q_len = input_params.q_seq_lens_vec[i];
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
    prefill_inputs.micro_inputs.push_back(std::move(prefill_input));
  }
  prefill_inputs.concated_sampling_params = inputs.concated_sampling_params;
}

std::optional<ForwardOutput> SpeculativeWorkerImpl::step_decode(
    const BatchedForwardInputs& inputs) {
  // TODO : now only support Deepseek MTP
  // More work need to support n-gram and native speculative decoding.
  // ForwardInput draft_inputs = inputs;
  BatchedForwardInputs draft_inputs = inputs;
  for (auto i = 0; i < draft_inputs.micro_inputs.size(); ++i) {
    auto& input = inputs.micro_inputs[i];
    auto& draft_input = draft_inputs.micro_inputs[i];
    // get embedding cache
#if defined(USE_NPU)
    if (kv_cache_transfer_) {
      kv_cache_transfer_->copy_blocks(input.input_params.embedding_ids,
                                      /*h2d*/ false);
    }
#endif
    torch::Tensor embeddings =
        embedding_allocator_->read(draft_input.input_params.embedding_ids);
    draft_input.input_params.mm_data =
        MMData(MMType::EMBEDDING, {{"embedding", embeddings.to(device_)}});
  }

  // run the draft model to get proposals
  std::vector<ForwardOutput> draft_outputs;
  BatchedForwardInputs validate_inputs, next_step_input;
  Timer timer;
  std::vector<folly::SemiFuture<std::optional<ForwardOutput>>> futures;
  for (size_t i = 0; i < options_.num_speculative_tokens(); ++i) {
    auto future = draft_impl_->step_async(draft_inputs);
    if (i == options_.num_speculative_tokens() - 1) {
      // final step
      prepare_validate_inputs(inputs, validate_inputs, true);
    } else {
      prepare_draft_inputs(draft_inputs, next_step_input, 1, device_);
    }
    draft_outputs.push_back(std::move(future).get().value());
    // update input of next step
    if (i < options_.num_speculative_tokens() - 1) {
      draft_inputs = next_step_input;
      auto last_output = draft_outputs.back().sample_output;
      auto start_idx = 0;
      auto token_start_idx = 0;
      for (auto i = 0; i < draft_inputs.micro_inputs.size(); ++i) {
        auto& draft_input = draft_inputs.micro_inputs[i];
        auto offset = draft_input.input_params.num_sequences;
        auto token_offset = draft_input.token_ids.size(0);
        draft_input.token_ids = safe_to(
            last_output.next_tokens.narrow(0, start_idx, offset), torch::kInt);
        if (token_offset > 0) {
          draft_input.input_params.mm_data = MMData(
              MMType::EMBEDDING,
              {{"embedding",
                last_output.embeddings.narrow(0, token_start_idx, token_offset)
                    .to(device_)}});
        }
        start_idx += offset;
        token_start_idx += token_offset;
      }
    }
  }
  COUNTER_ADD(speculative_execution_latency_seconds_draft,
              timer.elapsed_seconds());

  for (int i = 0; i < options_.num_speculative_tokens(); ++i) {
    ForwardOutput draft_output = draft_outputs[i];
    auto next_tokens =
        safe_to(draft_output.sample_output.next_tokens, torch::kInt);
    int32_t start_idx = 0;
    for (auto i = 0; i < validate_inputs.micro_inputs.size(); ++i) {
      int32_t offset = draft_inputs.micro_inputs[i].input_params.num_sequences;
      auto& validate_input = validate_inputs.micro_inputs[i];
      auto& token_ids = validate_input.token_ids;
      auto mask = (token_ids == -1 * (i + 1));
      token_ids.masked_scatter_(mask, next_tokens.narrow(0, start_idx, offset));
      start_idx += offset;
    }
  }

  // run the target model to get the verification scores
  timer.reset();
  auto future = impl_->step_async(validate_inputs);
  ForwardOutput target_output = std::move(future).get().value();
  COUNTER_ADD(speculative_execution_latency_seconds_target,
              timer.elapsed_seconds());

  // verify the proposals with target and update the batch
  timer.reset();
  SampleOutput val_output =
      validate(inputs.concated_sampling_params, draft_outputs, target_output);
  COUNTER_ADD(speculative_execution_latency_seconds_validation,
              timer.elapsed_seconds());

  for (auto i = 0; i < inputs.micro_inputs.size(); ++i) {
    auto& input = inputs.micro_inputs[i];
    // write the right cache and clear embeddings
    embedding_allocator_->write_validate(input.input_params.embedding_ids,
                                         val_output.next_tokens.to(torch::kCPU),
                                         val_output.embeddings);
#if defined(USE_NPU)
    if (kv_cache_transfer_) {
      kv_cache_transfer_->copy_blocks(input.input_params.embedding_ids,
                                      /*h2d*/ true);
    }
#endif
  }

  val_output.embeddings = torch::Tensor();

  if (!enable_schedule_overlap() && !driver_ && !dp_driver_) {
    return std::nullopt;
  }
  target_output.sample_output = val_output;
  return target_output;
}

void SpeculativeWorkerImpl::prepare_draft_inputs(
    const BatchedForwardInputs& inputs,
    BatchedForwardInputs& draft_inputs,
    const int64_t offset,
    const torch::Device device) {
  // prepare input for MTP in decoding phase (Like Eagle).
  draft_inputs.micro_inputs.reserve(inputs.micro_inputs.size());
  for (auto i = 0; i < inputs.micro_inputs.size(); ++i) {
    auto& input = inputs.micro_inputs[i];
    ForwardInput draft_input = input.to(device, dtype_);

    auto& input_params = draft_input.input_params;
    const int32_t num_sequences = input_params.num_sequences;
    torch::Tensor positions = safe_to(input.positions, torch::kCPU);
    Slice<int32_t> positions_slice = {positions.data_ptr<int32_t>(),
                                      positions.numel()};
    std::vector<int32_t> new_positions;
    new_positions.reserve(num_sequences);
    for (int32_t i = 0; i < num_sequences; ++i) {
      new_positions.emplace_back(positions_slice[i] + offset);
    }
    torch::TensorOptions int_options = input.token_ids.options();
    draft_input.positions = torch::tensor(new_positions, int_options);

    std::vector<int32_t> kv_seq_lens_vec = {};
    // slot ids for new token
    std::vector<int32_t> new_token_slot_ids;

    int32_t block_size = options_.block_size();
    torch::Tensor kv_seq_lens = safe_to(input_params.kv_seq_lens, torch::kCPU);
    Slice<int32_t> kv_seq_lens_slice = {kv_seq_lens.data_ptr<int32_t>(),
                                        kv_seq_lens.numel()};
    torch::Tensor block_tables =
        safe_to(input_params.block_tables, torch::kCPU);
    torch::Tensor new_cache_slots =
        safe_to(input_params.new_cache_slots, torch::kCPU);
    Slice<int32_t> new_cache_slots_slice = {new_cache_slots.data_ptr<int32_t>(),
                                            new_cache_slots.numel()};
    for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
      kv_seq_lens_vec.emplace_back(kv_seq_lens_slice[seq_id] + offset);
      torch::Tensor block_table = block_tables[seq_id];
      Slice<int32_t> block_table_slice = {block_table.data_ptr<int32_t>(),
                                          block_table.numel()};
      int32_t new_token_slot_id = get_new_token_slot_id(
          new_cache_slots_slice[seq_id], block_size, offset, block_table_slice);
      new_token_slot_ids.emplace_back(new_token_slot_id);
    }

    input_params.kv_max_seq_len = input_params.kv_max_seq_len + offset;
    input_params.kv_seq_lens_vec = kv_seq_lens_vec;
    input_params.kv_seq_lens = torch::tensor(kv_seq_lens_vec, int_options);
    input_params.new_cache_slots =
        torch::tensor(new_token_slot_ids, int_options);
    draft_inputs.micro_inputs.push_back(std::move(draft_input));
  }
  draft_inputs.concated_sampling_params = inputs.concated_sampling_params;
}

void SpeculativeWorkerImpl::prepare_validate_inputs(
    const BatchedForwardInputs& inputs,
    BatchedForwardInputs& validate_inputs,
    bool enable_schedule_overlap) {
  validate_inputs.micro_inputs.reserve(inputs.micro_inputs.size());
  for (auto i = 0; i < inputs.micro_inputs.size(); ++i) {
    auto& input = inputs.micro_inputs[i];

    ForwardInput validate_input = input.to(device_, dtype_);
    auto& input_params = validate_input.input_params;

    const int32_t position_offset = enable_schedule_overlap ? 1 : 0;
    const int32_t num_speculative_tokens = options_.num_speculative_tokens();
    const int32_t num_sequences = input_params.num_sequences;
    const int32_t num_val_tokens = num_speculative_tokens + 1;
    const int32_t total_num_val_tokens = num_sequences * num_val_tokens;

    std::vector<std::vector<int32_t>> draft_tokens;
    draft_tokens.reserve(num_speculative_tokens);
    for (int i = 0; i < num_speculative_tokens; ++i) {
      draft_tokens.emplace_back(std::vector(num_sequences, -1 * (i + 1)));
    }

    torch::Tensor token_ids = safe_to(input.token_ids, torch::kCPU);
    Slice<int32_t> tokens_ids_slice = {token_ids.data_ptr<int32_t>(),
                                       token_ids.numel()};
    torch::Tensor positions = safe_to(input.positions, torch::kCPU);
    Slice<int32_t> positions_slice = {positions.data_ptr<int32_t>(),
                                      positions.numel()};

    std::vector<int32_t> new_token_ids;
    std::vector<int32_t> new_positions;
    new_token_ids.reserve(total_num_val_tokens);
    new_positions.reserve(total_num_val_tokens);
    for (int32_t i = 0; i < num_sequences; ++i) {
      new_token_ids.emplace_back(tokens_ids_slice[i]);
      new_positions.emplace_back(positions_slice[i] + position_offset);
      for (int32_t j = 0; j < num_speculative_tokens; ++j) {
        new_token_ids.emplace_back(draft_tokens[j][i]);
        new_positions.emplace_back(positions_slice[i] + j + 1 +
                                   position_offset);
      }
    }
    torch::TensorOptions int_options = input.token_ids.options();
    validate_input.token_ids = torch::tensor(new_token_ids, int_options);
    validate_input.positions = torch::tensor(new_positions, int_options);

    // update the input_params
    input_params.num_sequences = total_num_val_tokens;
    input_params.kv_max_seq_len =
        input_params.kv_max_seq_len + num_speculative_tokens + position_offset;
    for (auto& it : input_params.dp_global_token_nums) {
      it *= num_val_tokens;
    }

    std::vector<int32_t> kv_seq_lens_vec = {};
    std::vector<int32_t> q_seq_lens_vec = {};
    // slot ids for new token
    std::vector<int32_t> new_token_slot_ids;
    std::vector<std::vector<int32_t>> block_tables_vec;

    int32_t block_size = options_.block_size();
    torch::Tensor kv_seq_lens = safe_to(input_params.kv_seq_lens, torch::kCPU);
    Slice<int32_t> kv_seq_lens_slice = {kv_seq_lens.data_ptr<int32_t>(),
                                        kv_seq_lens.numel()};
    torch::Tensor block_tables =
        safe_to(input_params.block_tables, torch::kCPU);
    torch::Tensor new_cache_slots =
        safe_to(input_params.new_cache_slots, torch::kCPU);
    Slice<int32_t> new_cache_slots_slice = {new_cache_slots.data_ptr<int32_t>(),
                                            new_cache_slots.numel()};
    for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
      int32_t cur_token_slot_id = new_cache_slots_slice[seq_id];
      torch::Tensor block_table = block_tables[seq_id];
      Slice<int32_t> block_table_slice = {block_table.data_ptr<int32_t>(),
                                          block_table.numel()};

      // process kv length and q length
      if (FLAGS_enable_atb_spec_kernel) {
        kv_seq_lens_vec.emplace_back(kv_seq_lens_slice[seq_id] +
                                     num_speculative_tokens + position_offset);
        q_seq_lens_vec.emplace_back(num_val_tokens);
      } else {
        for (int32_t token_id = position_offset;
             token_id < num_val_tokens + position_offset;
             ++token_id) {
          q_seq_lens_vec.emplace_back(1);
          kv_seq_lens_vec.emplace_back(kv_seq_lens_slice[seq_id] + token_id);
          // repeat block table
          block_tables_vec.emplace_back(block_table_slice);
        }
      }

      // process position related params
      for (int32_t token_id = position_offset;
           token_id < num_val_tokens + position_offset;
           ++token_id) {
        int32_t new_token_slot_id = get_new_token_slot_id(
            cur_token_slot_id, block_size, token_id, block_table_slice);
        new_token_slot_ids.emplace_back(new_token_slot_id);
      }
    }

    input_params.kv_seq_lens_vec = kv_seq_lens_vec;
    input_params.kv_seq_lens = torch::tensor(kv_seq_lens_vec, int_options);
    input_params.q_seq_lens_vec = q_seq_lens_vec;
    input_params.q_seq_lens = torch::tensor(q_seq_lens_vec, int_options);
    input_params.new_cache_slots =
        torch::tensor(new_token_slot_ids, int_options);
    if (!FLAGS_enable_atb_spec_kernel) {
      util::pad_2d_vector(block_tables_vec, /*pad_value=*/0);
      input_params.block_tables =
          create_2d_tensor(block_tables_vec, torch::kInt).to(device_);
    }
    input_params.decode_seq_range.second = input_params.num_sequences - 1;

    // update the sampling_params
    update_sampling_params(
        validate_input.sampling_params, num_val_tokens, total_num_val_tokens);
    validate_inputs.micro_inputs.push_back(std::move(validate_input));
  }

  validate_inputs.concated_sampling_params =
      validate_inputs.micro_inputs[0].sampling_params;
  for (auto i = 1; i < validate_inputs.micro_inputs.size(); ++i) {
    validate_inputs.concated_sampling_params.concat(
        validate_inputs.micro_inputs[i].sampling_params);
  }
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
                                         target_output.max_top_logprobs);

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

  torch::Tensor token_ids = safe_to(inputs.token_ids, torch::kCPU);
  Slice<int32_t> tokens_ids_slice = {token_ids.data_ptr<int32_t>(),
                                     token_ids.numel()};
  torch::Tensor positions = safe_to(inputs.positions, torch::kCPU);
  Slice<int32_t> positions_slice = {positions.data_ptr<int32_t>(),
                                    positions.numel()};
  torch::Tensor last_token_ids = safe_to(
      last_step_output_.sample_output.next_tokens.flatten(), torch::kCPU);
  Slice<int64_t> last_tokens_ids_slice = {last_token_ids.data_ptr<int64_t>(),
                                          last_token_ids.numel()};

  int32_t last_step_decode_num = 1;
  if (last_step_output_.sample_output.next_tokens.dim() == 2) {
    last_step_decode_num = last_step_output_.sample_output.next_tokens.size(1);
  }
  int32_t block_size = options_.block_size();
  torch::Tensor kv_seq_lens = safe_to(input_params.kv_seq_lens, torch::kCPU);
  Slice<int32_t> kv_seq_lens_slice = {kv_seq_lens.data_ptr<int32_t>(),
                                      kv_seq_lens.numel()};
  torch::Tensor block_tables = safe_to(input_params.block_tables, torch::kCPU);
  torch::Tensor new_cache_slots =
      safe_to(input_params.new_cache_slots, torch::kCPU);
  Slice<int32_t> new_cache_slots_slice = {new_cache_slots.data_ptr<int32_t>(),
                                          new_cache_slots.numel()};

  const int32_t num_sequences = inputs.input_params.num_sequences;
  std::vector<int32_t> new_token_ids;
  std::vector<int32_t> new_positions;
  new_token_ids.reserve(num_sequences);
  new_positions.reserve(num_sequences);

  // update the input_params
  input_params.kv_max_seq_len =
      input_params.kv_max_seq_len + last_step_decode_num - 1;

  std::vector<int32_t> kv_seq_lens_vec = {};
  std::vector<int32_t> new_token_slot_ids;
  new_token_slot_ids.reserve(num_sequences);

  // get right token id and position
  for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
    int32_t postion_offset = 0;
    int32_t last_step_token_id = 0;
    if (tokens_ids_slice[seq_id] >= 0) {
      last_step_token_id = tokens_ids_slice[seq_id];
    } else {
      int32_t last_step_index = -1 * tokens_ids_slice[seq_id] - 1;
      last_step_index = last_step_index * last_step_decode_num;
      last_step_token_id = last_tokens_ids_slice[last_step_index];
      for (int i = 1; i < last_step_decode_num; ++i) {
        int32_t token_id = last_tokens_ids_slice[last_step_index + i];
        if (token_id >= 0) {
          last_step_token_id = token_id;
          postion_offset += 1;
        }
      }
    }

    new_token_ids.push_back(last_step_token_id);
    if (postion_offset == 0) {
      new_positions.emplace_back(positions_slice[seq_id]);
      kv_seq_lens_vec.emplace_back(kv_seq_lens_slice[seq_id]);
      new_token_slot_ids.emplace_back(new_cache_slots_slice[seq_id]);
      continue;
    }
    new_positions.emplace_back(positions_slice[seq_id] + postion_offset);
    kv_seq_lens_vec.emplace_back(kv_seq_lens_slice[seq_id] + postion_offset);

    torch::Tensor block_table = block_tables[seq_id];
    Slice<int32_t> block_table_slice = {block_table.data_ptr<int32_t>(),
                                        block_table.numel()};
    int32_t new_token_slot_id =
        get_new_token_slot_id(new_cache_slots_slice[seq_id],
                              block_size,
                              postion_offset,
                              block_table_slice);
    new_token_slot_ids.emplace_back(new_token_slot_id);
  }

  torch::TensorOptions int_options = inputs.token_ids.options();
  new_inputs.token_ids = torch::tensor(new_token_ids, int_options);
  new_inputs.positions = torch::tensor(new_positions, int_options);

  input_params.kv_seq_lens_vec = kv_seq_lens_vec;
  input_params.kv_seq_lens = torch::tensor(kv_seq_lens_vec, int_options);
  input_params.new_cache_slots = torch::tensor(new_token_slot_ids, int_options);

  return new_inputs.to(device_, dtype_);
}

void SpeculativeWorkerImpl::update_sampling_params(
    SamplingParameters& sampling_params,
    const int32_t num_val_tokens,
    const int32_t total_num_val_tokens) {
  // sleceted tokens to return logits, including generated tokens and last
  // prompt token
  std::vector<int32_t> selected_token_idxes_vec;
  selected_token_idxes_vec.reserve(total_num_val_tokens);
  for (int32_t i = 0; i < total_num_val_tokens; i++) {
    selected_token_idxes_vec.emplace_back(i);
  }
  torch::Tensor selected_token_idxes = torch::tensor(selected_token_idxes_vec);

  sampling_params.selected_token_idxes = selected_token_idxes.to(device_);
  // assume sample_idxes equals to selected_token_idxes
  // when disable chunked prefill
  sampling_params.sample_idxes = selected_token_idxes.to(device_);

  TENSOR_REPEAT(sampling_params.frequency_penalties, num_val_tokens);
  TENSOR_REPEAT(sampling_params.presence_penalties, num_val_tokens);
  TENSOR_REPEAT(sampling_params.repetition_penalties, num_val_tokens);
  TENSOR_REPEAT(sampling_params.top_p, num_val_tokens);
  TENSOR_REPEAT(sampling_params.top_k, num_val_tokens);
  TENSOR_REPEAT(sampling_params.do_sample, num_val_tokens);
}

void SpeculativeWorkerImpl::prepare_work_before_execute(
    const BatchedForwardInputs& inputs,
    BatchedForwardInputs& processed_inputs) {
  if (inputs.micro_inputs[0].input_params.q_seq_lens_vec[0] > 1) {
    WorkerImpl::prepare_work_before_execute(inputs, processed_inputs);
  } else {
    if (enable_schedule_overlap()) {
      prepare_draft_inputs(inputs, processed_inputs, -1, torch::kCPU);
    } else {
      prepare_draft_inputs(inputs, processed_inputs, -1, device_);
    }
  }
}
}  // namespace xllm

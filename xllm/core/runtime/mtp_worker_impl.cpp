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

#include "mtp_worker_impl.h"

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

int32_t kv_cache_slot_id(int32_t position,
                         const Slice<int32_t>& block_table_slice,
                         int32_t block_size) {
  const int32_t block_id = block_table_slice[position / block_size];
  const int32_t block_offset = position % block_size;
  return block_id * block_size + block_offset;
}

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

void push_cumsum(std::vector<int32_t>& vec, int32_t len) {
  if (vec.empty()) {
    vec.emplace_back(0);
  }
  vec.emplace_back(vec.back() + len);
}

void append_seq_len(std::vector<int32_t>& vec, int32_t len) {
#if defined(USE_NPU)
  vec.emplace_back(len);
#elif defined(USE_MLU) || defined(USE_CUDA)
  push_cumsum(vec, len);
#endif
}

void update_kv_seq_lens_and_max(std::vector<int32_t>& kv_seq_lens_vec,
                                int32_t kv_len,
                                int32_t& kv_max_seq_len) {
  if (kv_len > kv_max_seq_len) {
    kv_max_seq_len = kv_len;
  }
  append_seq_len(kv_seq_lens_vec, kv_len);
}

void update_kv_seq_lens_vec(std::vector<int32_t>& kv_seq_lens_vec,
                            const Slice<int32_t>& kv_seq_lens_slice,
                            int32_t seq_id,
                            int32_t offset,
                            int32_t& kv_max_seq_len) {
  int32_t kv_len = calculate_kv_len(kv_seq_lens_slice, seq_id, offset);
  update_kv_seq_lens_and_max(kv_seq_lens_vec, kv_len, kv_max_seq_len);
}

}  // namespace

namespace {
runtime::Options MTPTargetOptions(const runtime::Options& options) {
  auto opts = options;
  opts.enable_schedule_overlap(false);
  return opts;
}

runtime::Options MTPDraftOptions(const runtime::Options& options) {
  auto opts = options;
  opts.enable_schedule_overlap(false)
      .num_decoding_tokens(1)
      .num_speculative_tokens(0);
  return opts;
}
}  // namespace

MTPWorkerImpl::MTPWorkerImpl(const ParallelArgs& parallel_args,
                             const torch::Device& device,
                             const runtime::Options& options)
    : MTPWorkerImpl(parallel_args,
                    device,
                    options,
                    MTPTargetOptions(options),
                    MTPDraftOptions(options),
                    FLAGS_enable_opt_validate_probs) {}

MTPWorkerImpl::MTPWorkerImpl(const ParallelArgs& parallel_args,
                             const torch::Device& device,
                             const runtime::Options& options,
                             const runtime::Options& target_options,
                             const runtime::Options& draft_options,
                             bool enable_opt_validate_probs)
    : SpeculativeWorkerImpl(parallel_args, device, options, target_options),
      enable_opt_validate_probs_(enable_opt_validate_probs) {
  draft_impl_ =
      std::make_unique<LLMWorkerImpl>(parallel_args, device, draft_options);
}

bool MTPWorkerImpl::init_model(const std::string& model_weights_path,
                               int32_t random_seed) {
  // Load target model via base class
  bool result = true;
  if (impl_->get_status() == WorkerImpl::Status::UNINITIALIZED) {
    result = SpeculativeWorkerImpl::init_model(model_weights_path, random_seed);
  } else {
    CHECK_EQ(draft_impl_->get_status(), WorkerImpl::Status::UNINITIALIZED);
    result =
        draft_impl_->WorkerImpl::init_model(model_weights_path, random_seed);
  }

  if (draft_impl_ != nullptr &&
      draft_impl_->get_status() == WorkerImpl::Status::LOADED) {
    // Share lm_head and word_embedding between target and draft models
#if defined(USE_NPU)
    if (FLAGS_npu_kernel_backend != "TORCH") {
      auto head = impl_->get_npu_lm_head();
      draft_impl_->set_npu_lm_head(head);
      auto word_embedding = impl_->get_npu_word_embedding();
      draft_impl_->set_npu_word_embedding(word_embedding);
    } else {
      auto head = impl_->get_lm_head();
      draft_impl_->set_lm_head(head);
      auto word_embedding = impl_->get_word_embedding();
      draft_impl_->set_word_embedding(word_embedding);
    }
#else
    auto head = impl_->get_lm_head();
    draft_impl_->set_lm_head(head);
    auto word_embedding = impl_->get_word_embedding();
    draft_impl_->set_word_embedding(word_embedding);
#endif
    // Sync context_ from impl_ for WorkerImpl::prepare_work_before_execute
    context_ = impl_->context_;
  }
  return result;
}

int64_t MTPWorkerImpl::get_embedding_placeholder_size() {
  return static_cast<int64_t>(embedding_size_);
}

bool MTPWorkerImpl::allocate_kv_cache(
    const std::vector<std::vector<int64_t>>& kv_cache_shape) {
  const int64_t num_blocks = kv_cache_shape[0][0];
  // init_model() must run first so dtype_/embedding_size_ are initialized.
  embedding_cache_ = std::make_shared<EmbeddingCache>(num_blocks);
  if (embedding_cache_) {
    int64_t size = get_embedding_placeholder_size();
    if (size > 0) {
      embedding_cache_->set_placeholder(
          torch::zeros({size}, torch::dtype(dtype_).device(torch::kCPU)));
    }
  }
  CHECK(impl_ != nullptr);
  CHECK(draft_impl_ != nullptr);

  bool target_allocated = true;
  const auto target_status = impl_->get_status();
  if (target_status == WorkerImpl::Status::LOADED) {
    target_allocated = impl_->allocate_kv_cache(kv_cache_shape);
  } else {
    CHECK_EQ(target_status, WorkerImpl::Status::READY);
  }

  bool draft_allocated = true;
  const auto draft_status = draft_impl_->get_status();
  if (draft_status == WorkerImpl::Status::LOADED) {
    draft_allocated = draft_impl_->allocate_kv_cache(kv_cache_shape);
  } else {
    CHECK_EQ(draft_status, WorkerImpl::Status::READY);
  }

  return target_allocated && draft_allocated;
}

#if defined(USE_NPU)
bool MTPWorkerImpl::allocate_kv_cache_with_transfer(
    const uint64_t kv_cache_size,
    const std::vector<std::vector<int64_t>>& kv_cache_shape) {
  CHECK(impl_ != nullptr);
  CHECK(draft_impl_ != nullptr);

  if (kv_cache_transfer_ == nullptr) {
    kv_cache_transfer_ = std::make_shared<SpecKVCacheTransfer>(
        options_.device_ip().value(),
        options_.transfer_listen_port(),
        options_.instance_role(),
        context_.get_model_args().model_type());

    int32_t device_id = device_.index();
    kv_cache_transfer_->initialize(device_id);
  }

  bool target_allocated = true;
  const auto target_status = impl_->get_status();
  if (target_status == WorkerImpl::Status::LOADED) {
    target_allocated = impl_->allocate_kv_cache_with_transfer(
        kv_cache_transfer_, kv_cache_shape);
  } else {
    CHECK_EQ(target_status, WorkerImpl::Status::READY);
  }

  bool draft_allocated = true;
  const auto draft_status = draft_impl_->get_status();
  if (draft_status == WorkerImpl::Status::LOADED) {
    draft_allocated = draft_impl_->allocate_kv_cache_with_transfer(
        kv_cache_transfer_, kv_cache_shape);
  } else {
    CHECK_EQ(draft_status, WorkerImpl::Status::READY);
  }

  embedding_cache_ = std::make_shared<EmbeddingCache>(kv_cache_shape[0][0]);
  if (embedding_cache_) {
    int64_t size = get_embedding_placeholder_size();
    if (size > 0) {
      embedding_cache_->set_placeholder(
          torch::zeros({size}, torch::dtype(dtype_).device(device_)));
    }
  }
  return target_allocated && draft_allocated;
}
#endif

std::optional<ForwardOutput> MTPWorkerImpl::step_empty(
    const ForwardInput& input) {
  if (!input.input_params.batch_forward_type.is_decode()) {
    auto output = impl_->step(input);
    auto draft_output = draft_impl_->step(input);
    (void)draft_output;
    output->sample_output.embeddings = torch::Tensor();
    return output;
  } else {
    for (size_t i = 0; i < options_.num_speculative_tokens(); ++i) {
      auto draft_future = draft_impl_->step_async(input);
      ForwardOutput draft_output = std::move(draft_future).get().value();
      (void)draft_output;
    }

    ForwardInput new_input = input;
    for (auto& it : new_input.input_params.dp_global_token_nums) {
      it *= options_.num_speculative_tokens() + 1;
    }

    auto future = impl_->step_async(new_input);
    ForwardOutput output = std::move(future).get().value();
    output.sample_output.embeddings = torch::Tensor();
    return output;
  }
}

std::optional<ForwardOutput> MTPWorkerImpl::step_prefill(
    const ForwardInput& input) {
  Timer timer;
  // run the target model to get first token and hidden states
  auto future = impl_->step_async(input);
  ForwardOutput output = std::move(future).get().value();
  COUNTER_ADD(speculative_execution_latency_seconds_target,
              timer.elapsed_seconds());

  // MTP path that depends on hidden states.
  ForwardInput prefill_input;
  prepare_prefill_inputs(input, prefill_input);

  // prepare input for draft model
  auto& embeddings = output.sample_output.embeddings;
  auto next_tokens = safe_to(output.sample_output.next_tokens, torch::kInt);

  if (embeddings.defined()) {
    prefill_input.input_params.input_embedding = embeddings.clone();
  }
  if (next_tokens.defined()) {
    auto& token_ids = prefill_input.token_ids;
    auto mask = (token_ids == -1);
    token_ids.masked_scatter_(mask, next_tokens);
  }

  // generate kv cache for draft model
  timer.reset();
  auto draft_future = draft_impl_->step_async(prefill_input);
  ForwardOutput draft_output = std::move(draft_future).get().value();
  (void)draft_output;
  COUNTER_ADD(speculative_execution_latency_seconds_draft,
              timer.elapsed_seconds());

  if (input.sampling_params.selected_token_idxes.defined()) {
    embeddings = embeddings.index_select(
        /*dim=*/0, input.sampling_params.selected_token_idxes);
    CHECK_EQ(embeddings.size(0), output.sample_output.next_tokens.size(0));
    embedding_cache_->write(input.input_params.embedding_ids, embeddings);
  }
  output.sample_output.embeddings = torch::Tensor();

  if (!enable_schedule_overlap() && !driver_ && !dp_driver_) {
    return std::nullopt;
  }
  return output;
}

void MTPWorkerImpl::prepare_prefill_inputs(const ForwardInput& input,
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
    int32_t q_len = input_params.get_q_seq_len(i);
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
  std::vector<int32_t> q_cu_seq_lens_vec;
  q_cu_seq_lens_vec.reserve(input_params.num_sequences);
  int32_t cum_seq_len = 0;
  for (int32_t i = 0; i < input_params.num_sequences; ++i) {
    cum_seq_len += input_params.get_q_seq_len(i);
    q_cu_seq_lens_vec.push_back(cum_seq_len);
  }
  input_params.q_cu_seq_lens = torch::tensor(q_cu_seq_lens_vec, torch::kInt);
}

std::optional<ForwardOutput> MTPWorkerImpl::step_decode(
    const ForwardInput& input) {
  ForwardInput draft_input = input;
  auto cache_items =
      embedding_cache_->read_for_decode(draft_input.input_params.embedding_ids);
  std::vector<torch::Tensor> embedding_rows;
  embedding_rows.reserve(cache_items.size());
  for (const auto& item : cache_items) {
    embedding_rows.emplace_back(item.embedding);
  }
  draft_input.input_params.input_embedding =
      torch::stack(embedding_rows).to(device_);

  // run the draft model to get proposals
  std::vector<ForwardOutput> draft_outputs;
  ForwardInput validate_input, next_step_input;
  Timer timer;
  for (size_t i = 0; i < options_.num_speculative_tokens(); ++i) {
    std::optional<ForwardInput> first_step_input_holder;
    const ForwardInput* run_input = &draft_input;
    if (i == 0) {
      first_step_input_holder.emplace();
      prepare_first_decode_inputs(
          draft_input, cache_items, first_step_input_holder.value());
      run_input = &first_step_input_holder.value();
    }
    auto future = draft_impl_->step_async(*run_input);

    // Overlap next-step input preparation with async draft forward.
    if (i == options_.num_speculative_tokens() - 1) {
      prepare_validate_inputs(input, validate_input);
    } else {
      prepare_draft_inputs(draft_input, next_step_input, 1, device_);
    }
    std::optional<ForwardOutput> draft_output_opt = std::move(future).get();
    CHECK(draft_output_opt.has_value())
        << "draft output is empty in speculative step";

    draft_outputs.push_back(std::move(draft_output_opt.value()));
    auto& last_output = draft_outputs.back().sample_output;
    if (last_output.probs.defined() && enable_opt_validate_probs_) {
      auto selected_probs =
          last_output.probs
              .gather(
                  /*dim=*/-1, last_output.next_tokens.unsqueeze(-1))
              .squeeze(-1);
      last_output.probs = selected_probs;  // [batch_size]
    }
    if (i < options_.num_speculative_tokens() - 1) {
      draft_input = next_step_input;
      draft_input.token_ids = safe_to(last_output.next_tokens, torch::kInt);
      draft_input.input_params.input_embedding =
          last_output.embeddings.to(device_);
    }
  }
  COUNTER_ADD(speculative_execution_latency_seconds_draft,
              timer.elapsed_seconds());

  for (int i = 0; i < options_.num_speculative_tokens(); ++i) {
    ForwardOutput draft_output = draft_outputs[i];
    auto next_tokens =
        safe_to(draft_output.sample_output.next_tokens, torch::kInt);
    auto& token_ids = validate_input.token_ids;
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
  val_output.next_tokens = val_output.next_tokens.to(torch::kCPU);
  embedding_cache_->write_validate(input.input_params.embedding_ids,
                                   val_output.next_tokens,
                                   val_output.embeddings,
                                   options_.num_speculative_tokens());

  if (!enable_schedule_overlap() && !driver_ && !dp_driver_) {
    return std::nullopt;
  }
  val_output.embeddings = torch::Tensor();
  target_output.sample_output = val_output;
  return target_output;
}

void MTPWorkerImpl::prepare_draft_inputs(const ForwardInput& input,
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
    // slot ids for new token
    int32_t slot_id =
        kv_cache_slot_id(new_positions.back(), block_table_slice, block_size);
    new_token_slot_ids.emplace_back(slot_id);
  }

  CHECK_EQ(new_token_slot_ids.size(), new_positions.size())
      << "draft kv slots/positions mismatch";

  draft_input.positions = torch::tensor(new_positions, int_options);
  // update the input_params
  input_params.kv_max_seq_len = kv_max_seq_len;
  input_params.kv_seq_lens_vec = kv_seq_lens_vec;
  input_params.kv_seq_lens = torch::tensor(kv_seq_lens_vec, int_options);
  input_params.new_cache_slots = torch::tensor(new_token_slot_ids, int_options);
  // deepseek 3.2
  std::vector<int32_t> q_cu_seq_lens_vec;
  q_cu_seq_lens_vec.reserve(input_params.num_sequences);
  int32_t cum_seq_len = 0;
  for (int32_t i = 0; i < input_params.num_sequences; ++i) {
    cum_seq_len += input_params.get_q_seq_len(i);
    q_cu_seq_lens_vec.push_back(cum_seq_len);
  }
  input_params.q_cu_seq_lens = torch::tensor(q_cu_seq_lens_vec, torch::kInt);
}

void MTPWorkerImpl::prepare_first_decode_inputs(
    const ForwardInput& base_input,
    const std::vector<EmbeddingCache::DecodeState>& cache_items,
    ForwardInput& first_input) {
  first_input = base_input;
  auto& input_params = first_input.input_params;
  const int32_t num_sequences = input_params.num_sequences;
  CHECK_EQ(cache_items.size(), static_cast<size_t>(num_sequences))
      << "cache items size mismatch";

  std::vector<int32_t> select_row_idx(num_sequences, 0);
  bool has_fix = false;
  std::vector<uint8_t> fix_mask(num_sequences, 0);
  for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
    const auto& item = cache_items[seq_id];
    bool can_fix = item.need_first_decode_fix && item.prev_token_id >= 0 &&
                   item.last_token_id >= 0 && item.prev_embedding.defined() &&
                   item.embedding.defined();
    if (can_fix) {
      fix_mask[seq_id] = 1;
      has_fix = true;
    }
  }
  if (!has_fix) {
    return;
  }

  const int32_t block_size = options_.block_size();
  torch::TensorOptions int_options = base_input.token_ids.options();
  torch::Tensor token_ids = safe_to(base_input.token_ids, torch::kCPU);
  Slice<int32_t> token_ids_slice = {token_ids.data_ptr<int32_t>(),
                                    static_cast<size_t>(token_ids.numel())};
  torch::Tensor positions = safe_to(base_input.positions, torch::kCPU);
  Slice<int32_t> positions_slice = {positions.data_ptr<int32_t>(),
                                    static_cast<size_t>(positions.numel())};
  Slice<int32_t> kv_seq_lens_slice = input_params.kv_seq_lens_vec;
  torch::Tensor block_tables = safe_to(input_params.block_tables, torch::kCPU);

  std::vector<int32_t> expanded_token_ids;
  std::vector<int32_t> expanded_positions;
  std::vector<int32_t> kv_seq_lens_vec = {};
  std::vector<int32_t> q_seq_lens_vec = {};
  std::vector<int32_t> new_token_slot_ids;
  std::vector<std::vector<int32_t>> block_tables_vec;
  std::vector<torch::Tensor> expanded_embeddings;
  std::vector<int32_t> expanded_embedding_ids;
  std::vector<int32_t> expanded_extra_token_ids;
  const bool has_embedding_ids =
      static_cast<int32_t>(input_params.embedding_ids.size()) == num_sequences;
  const bool has_extra_token_ids =
      static_cast<int32_t>(input_params.extra_token_ids.size()) ==
      num_sequences;
  expanded_token_ids.reserve(num_sequences * 2);
  expanded_positions.reserve(num_sequences * 2);
  new_token_slot_ids.reserve(num_sequences * 2);
  kv_seq_lens_vec.reserve(num_sequences * 2);
  q_seq_lens_vec.reserve(num_sequences * 2);
  block_tables_vec.reserve(num_sequences * 2);
  expanded_embeddings.reserve(num_sequences * 2);
  if (has_embedding_ids) {
    expanded_embedding_ids.reserve(num_sequences * 2);
  }
  if (has_extra_token_ids) {
    expanded_extra_token_ids.reserve(num_sequences * 2);
  }

  int32_t kv_max_seq_len = 0;
  for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
    torch::Tensor block_table = block_tables[seq_id];
    Slice<int32_t> block_table_slice = {
        block_table.data_ptr<int32_t>(),
        static_cast<size_t>(block_table.numel())};
    std::vector<int32_t> block_table_vec(block_table_slice.begin(),
                                         block_table_slice.end());

    auto add_row = [&](int32_t token_id,
                       int32_t position_offset,
                       const torch::Tensor& embedding) {
      const int32_t new_position = positions_slice[seq_id] + position_offset;
      CHECK_GE(new_position, 0) << "invalid decode position after fix";
      expanded_token_ids.emplace_back(token_id);
      expanded_positions.emplace_back(new_position);
      int32_t kv_len =
          calculate_kv_len(kv_seq_lens_slice, seq_id, position_offset);
      update_kv_seq_lens_and_max(kv_seq_lens_vec, kv_len, kv_max_seq_len);
      new_token_slot_ids.emplace_back(
          kv_cache_slot_id(new_position, block_table_slice, block_size));
      append_seq_len(q_seq_lens_vec, 1);
      block_tables_vec.emplace_back(block_table_vec);
      expanded_embeddings.emplace_back(embedding);
      if (has_embedding_ids) {
        expanded_embedding_ids.emplace_back(input_params.embedding_ids[seq_id]);
      }
      if (has_extra_token_ids) {
        expanded_extra_token_ids.emplace_back(
            input_params.extra_token_ids[seq_id]);
      }
    };

    if (fix_mask[seq_id] == 1) {
      const auto& item = cache_items[seq_id];
      add_row(item.prev_token_id, /*position_offset=*/-1, item.prev_embedding);
      add_row(item.last_token_id, /*position_offset=*/0, item.embedding);
      select_row_idx[seq_id] =
          static_cast<int32_t>(expanded_token_ids.size()) - 1;
    } else {
      add_row(token_ids_slice[seq_id],
              /*position_offset=*/0,
              cache_items[seq_id].embedding);
      select_row_idx[seq_id] =
          static_cast<int32_t>(expanded_token_ids.size()) - 1;
    }
  }

  CHECK_EQ(new_token_slot_ids.size(), expanded_positions.size())
      << "first decode fix slots/positions mismatch";
  CHECK_EQ(expanded_embeddings.size(), expanded_positions.size())
      << "first decode fix embeddings/positions mismatch";

  first_input.token_ids = torch::tensor(expanded_token_ids, int_options);
  first_input.positions = torch::tensor(expanded_positions, int_options);
  input_params.num_sequences = static_cast<int32_t>(expanded_positions.size());
  input_params.q_max_seq_len = 1;
  input_params.batch_forward_type = BatchForwardType::DECODE;
  input_params.q_seq_lens_vec = std::move(q_seq_lens_vec);
  input_params.q_seq_lens =
      torch::tensor(input_params.q_seq_lens_vec, int_options);
  std::vector<int32_t> q_cu_seq_lens_vec;
  q_cu_seq_lens_vec.reserve(input_params.num_sequences);
  int32_t cum_seq_len = 0;
  for (int32_t i = 0; i < input_params.num_sequences; ++i) {
    cum_seq_len += input_params.get_q_seq_len(i);
    q_cu_seq_lens_vec.emplace_back(cum_seq_len);
  }
  input_params.q_cu_seq_lens = torch::tensor(q_cu_seq_lens_vec, torch::kInt);
  input_params.kv_max_seq_len = kv_max_seq_len;
  input_params.kv_seq_lens_vec = std::move(kv_seq_lens_vec);
  input_params.kv_seq_lens =
      torch::tensor(input_params.kv_seq_lens_vec, int_options);
  input_params.new_cache_slots = torch::tensor(new_token_slot_ids, int_options);
  util::pad_2d_vector(block_tables_vec, /*pad_value=*/0);
  input_params.block_tables = create_2d_tensor(block_tables_vec, torch::kInt);
  if (has_embedding_ids) {
    input_params.embedding_ids = std::move(expanded_embedding_ids);
  }
  if (has_extra_token_ids) {
    input_params.extra_token_ids = std::move(expanded_extra_token_ids);
  }
  input_params.input_embedding = torch::stack(expanded_embeddings).to(device_);

  auto& params = first_input.sampling_params;
  torch::TensorOptions idx_options =
      torch::TensorOptions().dtype(torch::kInt).device(torch::kCPU);
  if (params.selected_token_idxes.defined()) {
    idx_options = params.selected_token_idxes.options();
  } else if (params.sample_idxes.defined()) {
    idx_options = params.sample_idxes.options();
  }

  // First-step fix only needs proposal from each sequence's "current" row.
  params.selected_token_idxes = torch::tensor(select_row_idx, idx_options);
  if (!params.sample_idxes.defined()) {
    params.sample_idxes = torch::arange(num_sequences, idx_options);
  }
}

SampleOutput MTPWorkerImpl::validate(
    const SamplingParameters& sampling_params,
    const std::vector<ForwardOutput>& draft_outputs,
    const ForwardOutput& target_output) {
  const int32_t num_target_tokens =
      target_output.sample_output.next_tokens.numel();
  const int32_t num_val_tokens = options_.num_speculative_tokens() + 1;
  CHECK_EQ(num_target_tokens % num_val_tokens, 0);
  const int32_t batch_size = num_target_tokens / num_val_tokens;
  const int32_t vocab_size = target_output.logits.size(/*dim=*/-1);

  // [batch_size, n_speculative_tokens, vocab_size]
  std::vector<torch::Tensor> draft_token_ids_vec;
  std::vector<torch::Tensor> draft_probs_vec;
  for (const auto& draft_output : draft_outputs) {
    auto draft_token_ids = draft_output.sample_output.next_tokens;
    auto draft_probs = draft_output.sample_output.probs;
    if (enable_opt_validate_probs_) {
      draft_probs = draft_probs.view({{batch_size, 1}});
    } else {
      draft_probs = draft_probs.view({{batch_size, 1, vocab_size}});
    }
    draft_token_ids = draft_token_ids.view({batch_size, 1});
    draft_token_ids_vec.push_back(draft_token_ids);
    draft_probs_vec.push_back(draft_probs);
  }

  // concatenate the draft token ids and probs along the last dimension
  const auto draft_token_ids = torch::cat(draft_token_ids_vec, /*dim=*/1);
  const auto draft_probs = torch::cat(draft_probs_vec, /*dim=*/1);
  return validate(sampling_params, draft_token_ids, draft_probs, target_output);
}

SampleOutput MTPWorkerImpl::validate(const SamplingParameters& sampling_params,
                                     const torch::Tensor& draft_token_ids,
                                     const torch::Tensor& draft_probs,
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

  auto target_logits =
      target_output.logits.view({batch_size, num_val_tokens, vocab_size});

  // prepare input for rejection sampling
  auto rejection_sampler =
      std::make_unique<RejectionSampler>(sampling_params.do_sample,
                                         sampling_params.all_random_sample,
                                         sampling_params.all_greedy_sample,
                                         target_output.logprobs,
                                         target_output.max_top_logprobs,
                                         rate_controller_,
                                         enable_fused_kernel_);

  // get the accepted tokens
  SampleOutput sample_output =
      rejection_sampler->forward(draft_token_ids.to(bonus_token_ids),
                                 draft_probs.to(target_logits.device()),
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

}  // namespace xllm

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
#include "spec_input_builder.h"
#include "util/env_var.h"
#include "util/pretty_print.h"
#include "util/slice.h"
#include "util/timer.h"
#include "util/utils.h"

namespace xllm {
constexpr uint64_t MBUF_SIZE = 128 * 1024 * 1024;

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
                               int32_t random_seed,
                               MasterStatus master_status) {
  // Load target model via base class
  bool result = true;
  if (impl_->get_status() == WorkerImpl::Status::UNINITIALIZED) {
    result = SpeculativeWorkerImpl::init_model(
        model_weights_path, random_seed, master_status);
  } else {
    CHECK_EQ(draft_impl_->get_status(), WorkerImpl::Status::UNINITIALIZED);
    result = draft_impl_->WorkerImpl::init_model(
        model_weights_path, random_seed, master_status);
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

ForwardInput MTPWorkerImpl::update_input_by_last_step_output(
    ForwardInput& inputs) {
  // MTP decode correction uses embedding cache as source of truth instead of
  // WorkerImpl::last_step_output_.
  ForwardInput& new_inputs = inputs;
  auto& input_params = new_inputs.input_params;
  const int32_t num_sequences = input_params.num_sequences;
  const int32_t block_size = options_.block_size();

  // DP empty-run may still be marked as decode in global batch type.
  // Skip correction when no local sequences are present.
  if (num_sequences == 0 || input_params.embedding_ids.empty()) {
    return inputs;
  }

  CHECK(embedding_cache_ != nullptr)
      << "embedding_cache_ must be initialized before decode correction";
  std::vector<int32_t> correction_tokens =
      embedding_cache_->read_correction_tokens(input_params.embedding_ids);
  std::vector<int32_t> position_offsets =
      embedding_cache_->read_position_offsets(input_params.embedding_ids);

  torch::Tensor token_ids = safe_to(inputs.token_ids, torch::kCPU);
  torch::Tensor positions = safe_to(inputs.positions, torch::kCPU);
  torch::Tensor block_tables = safe_to(input_params.block_tables, torch::kCPU);
  auto view = specBuilder::make_decode_cpu_view(
      token_ids, positions, block_tables, input_params.kv_seq_lens_vec);
  CHECK_EQ(correction_tokens.size(), static_cast<size_t>(num_sequences))
      << "decode correction token count mismatch";
  CHECK_EQ(position_offsets.size(), static_cast<size_t>(num_sequences))
      << "decode correction offset count mismatch";

  specBuilder::DecodeBuildBuffers buf;
  buf.out_token_ids.reserve(num_sequences);
  buf.out_positions.reserve(num_sequences);
  buf.out_kv_seq_lens.reserve(num_sequences);
  buf.out_new_cache_slots.reserve(num_sequences);

  for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
    const int32_t input_token_id = view.token_ids[seq_id];
    specBuilder::RowSpec row;
    row.seq_id = seq_id;
    if (input_token_id >= 0) {
      row.token_id = input_token_id;
      row.position_offset = 0;
    } else {
      row.token_id = correction_tokens[seq_id];
      row.position_offset = position_offsets[seq_id];
    }
    specBuilder::append_decode_row(input_params, view, row, block_size, buf);
  }

  CHECK_EQ(buf.out_new_cache_slots.size(), buf.out_token_ids.size())
      << "step-update kv slots/tokens mismatch";
  CHECK_EQ(buf.out_positions.size(), buf.out_token_ids.size())
      << "step-update positions/tokens mismatch";

  torch::TensorOptions int_options = inputs.token_ids.options();
  new_inputs.token_ids = torch::tensor(buf.out_token_ids, int_options);
  new_inputs.positions = torch::tensor(buf.out_positions, int_options);
  input_params.kv_max_seq_len = buf.kv_max_seq_len;
  input_params.kv_seq_lens_vec = std::move(buf.out_kv_seq_lens);
  input_params.kv_seq_lens =
      torch::tensor(input_params.kv_seq_lens_vec, int_options);
  input_params.q_cu_seq_lens =
      specBuilder::build_q_cu_seq_lens_tensor(input_params);
  input_params.new_cache_slots =
      torch::tensor(buf.out_new_cache_slots, int_options);

  return new_inputs.to(device_, dtype_);
}

std::optional<ForwardOutput> MTPWorkerImpl::step_empty(
    const ForwardInput& input) {
  if (!input.input_params.batch_forward_type.is_decode()) {
    auto output = impl_->step(input);
    auto draft_output = draft_impl_->step(input);
    (void)draft_output;
    output->sample_output.embeddings = torch::Tensor();
    return output;
  } else {
    for (size_t i = 1; i < options_.num_speculative_tokens(); ++i) {
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

    new_input = input;
    for (auto& it : new_input.input_params.dp_global_token_nums) {
      it *= 2;
    }
    auto draft_future = draft_impl_->step_async(new_input);
    ForwardOutput draft_output = std::move(draft_future).get().value();
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
  process_draft_sample_output(draft_output.sample_output);
  COUNTER_ADD(speculative_execution_latency_seconds_draft,
              timer.elapsed_seconds());

  if (input.sampling_params.selected_token_idxes.defined()) {
    embedding_cache_->write(input.input_params.embedding_ids,
                            draft_output.sample_output.next_tokens,
                            draft_output.sample_output.embeddings,
                            draft_output.sample_output.probs,
                            output.sample_output.next_tokens);
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
}

std::optional<ForwardOutput> MTPWorkerImpl::step_decode(
    const ForwardInput& input) {
  CHECK_GT(options_.num_speculative_tokens(), 0)
      << "num_speculative_tokens should be > 0 in MTP decode";
  if (options_.num_speculative_tokens() == 1) {
    return step_decode_single(input);
  }
  return step_decode_multi_step(input);
}

std::optional<ForwardOutput> MTPWorkerImpl::step_decode_single(
    const ForwardInput& input) {
  CHECK_EQ(options_.num_speculative_tokens(), 1)
      << "step_decode_single requires num_speculative_tokens == 1";
  std::vector<ForwardOutput> draft_outputs;
  draft_outputs.emplace_back(prepare_last_output_for_decode(input));

  ForwardInput validate_input;
  prepare_validate_inputs(input, validate_input);
  fill_validate_input_from_draft_outputs(draft_outputs, validate_input);
  return run_validate(input, draft_outputs, validate_input);
}

std::optional<ForwardOutput> MTPWorkerImpl::step_decode_multi_step(
    const ForwardInput& input) {
  CHECK_GT(options_.num_speculative_tokens(), 1)
      << "step_decode_multi_step requires num_speculative_tokens > 1";
  ForwardInput draft_input = input;
  std::vector<ForwardOutput> draft_outputs;
  draft_outputs.emplace_back(prepare_last_output_for_decode(input));

  ForwardInput validate_input, next_step_input;
  Timer timer;
  draft_input.token_ids =
      safe_to(draft_outputs.back().sample_output.next_tokens, torch::kInt);
  draft_input.input_params.input_embedding =
      draft_outputs.back().sample_output.embeddings.to(device_);

  for (size_t i = 1; i < options_.num_speculative_tokens(); ++i) {
    auto future = draft_impl_->step_async(draft_input);

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
    process_draft_sample_output(draft_outputs.back().sample_output);
    auto& last_output = draft_outputs.back().sample_output;
    if (i < options_.num_speculative_tokens() - 1) {
      draft_input = next_step_input;
      draft_input.token_ids = safe_to(last_output.next_tokens, torch::kInt);
      draft_input.input_params.input_embedding =
          last_output.embeddings.to(device_);
    }
  }
  COUNTER_ADD(speculative_execution_latency_seconds_draft,
              timer.elapsed_seconds());

  fill_validate_input_from_draft_outputs(draft_outputs, validate_input);
  return run_validate(input, draft_outputs, validate_input);
}

ForwardOutput MTPWorkerImpl::prepare_last_output_for_decode(
    const ForwardInput& input) {
  ForwardOutput last_output =
      embedding_cache_->read_for_decode(input.input_params.embedding_ids);
  last_output.sample_output.next_tokens =
      last_output.sample_output.next_tokens.to(
          torch::dtype(torch::kInt).device(device_));
  last_output.sample_output.embeddings =
      last_output.sample_output.embeddings.to(device_);
  last_output.sample_output.probs = last_output.sample_output.probs.to(device_);
  return last_output;
}

void MTPWorkerImpl::fill_validate_input_from_draft_outputs(
    const std::vector<ForwardOutput>& draft_outputs,
    ForwardInput& validate_input) {
  for (int i = 0; i < options_.num_speculative_tokens(); ++i) {
    const auto& draft_output = draft_outputs[i];
    auto next_tokens =
        safe_to(draft_output.sample_output.next_tokens, torch::kInt);
    auto& token_ids = validate_input.token_ids;
    auto mask = (token_ids == -1 * (i + 1));
    token_ids.masked_scatter_(mask, next_tokens);
  }
}

std::optional<ForwardOutput> MTPWorkerImpl::run_validate(
    const ForwardInput& input,
    const std::vector<ForwardOutput>& draft_outputs,
    ForwardInput& validate_input) {
  // run the target model to get the verification scores
  Timer timer;
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

  // update cache and clear embeddings
  timer.reset();
  run_draft_extend(input, val_output);
  COUNTER_ADD(speculative_execution_latency_seconds_draft,
              timer.elapsed_seconds());

  val_output.next_tokens = val_output.next_tokens.to(torch::kCPU);

  if (!enable_schedule_overlap() && !driver_ && !dp_driver_) {
    return std::nullopt;
  }
  val_output.embeddings = torch::Tensor();
  target_output.sample_output = val_output;
  return target_output;
}

void MTPWorkerImpl::process_draft_sample_output(SampleOutput& sample_output) {
  if (sample_output.probs.defined()) {
    CHECK(sample_output.next_tokens.defined())
        << "draft sample_output.next_tokens must be defined when probs exist";
    CHECK_EQ(sample_output.next_tokens.dim(), 1)
        << "MTP draft cache expects next_tokens [batch], got "
        << sample_output.next_tokens.sizes();
    CHECK(sample_output.probs.dim() == 1 || sample_output.probs.dim() == 2)
        << "MTP draft cache expects probs [batch] or [batch,vocab], got "
        << sample_output.probs.sizes();
    CHECK_EQ(sample_output.probs.size(0), sample_output.next_tokens.size(0))
        << "MTP draft cache probs/token batch mismatch";
    // Cache always stores selected-only draft probs [batch_size] to reduce HBM.
    sample_output.probs = specBuilder::draftProbs::compress_for_cache(
        sample_output.probs, sample_output.next_tokens);
  }
}

void MTPWorkerImpl::prepare_draft_extend_inputs(
    const ForwardInput& base_input,
    const SampleOutput& validate_output,
    ForwardInput& extend_input) {
  extend_input = base_input.to(device_, dtype_);
  auto& input_params = extend_input.input_params;
  const int32_t num_sequences = input_params.num_sequences;

  const int32_t block_size = options_.block_size();
  torch::TensorOptions int_options = extend_input.token_ids.options();
  torch::Tensor token_ids = safe_to(base_input.token_ids, torch::kCPU);
  torch::Tensor positions = safe_to(base_input.positions, torch::kCPU);
  torch::Tensor block_tables = safe_to(input_params.block_tables, torch::kCPU);
  auto view = specBuilder::make_decode_cpu_view(
      token_ids, positions, block_tables, input_params.kv_seq_lens_vec);
  torch::Tensor accepted_tokens =
      safe_to(validate_output.next_tokens, torch::kCPU);
  const auto accepted_embeddings = validate_output.embeddings;

  specBuilder::DecodeBuildBuffers buf;
  buf.out_token_ids.reserve(num_sequences * 2);
  buf.out_positions.reserve(num_sequences * 2);
  buf.out_new_cache_slots.reserve(num_sequences * 2);
  buf.out_kv_seq_lens.reserve(num_sequences * 2);
  buf.out_q_seq_lens.reserve(num_sequences * 2);
  buf.out_block_tables.reserve(num_sequences * 2);
  std::vector<torch::Tensor> expanded_embeddings;
  std::vector<int32_t> selected_row_idx;
  expanded_embeddings.reserve(num_sequences * 2);
  selected_row_idx.reserve(num_sequences);

  for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
    auto add_row = [&](int32_t token_id,
                       int32_t relative_offset,
                       const torch::Tensor& embedding) {
      specBuilder::RowSpec row;
      row.seq_id = seq_id;
      row.token_id = token_id;
      row.position_offset = relative_offset;
      row.append_q_len_one = true;
      row.append_block_table = true;
      specBuilder::append_decode_row(input_params, view, row, block_size, buf);
      if (embedding.defined()) {
        expanded_embeddings.emplace_back(embedding.to(device_));
      } else {
        expanded_embeddings.emplace_back(
            torch::zeros({get_embedding_placeholder_size()},
                         torch::dtype(dtype_).device(device_)));
      }
    };

    int32_t last_idx = -1;
    int32_t last_token_id = 0;
    for (int32_t i = 0; i < accepted_tokens.size(1); ++i) {
      int64_t token = accepted_tokens[seq_id][i].item<int64_t>();
      if (token >= 0) {
        last_idx = i;
        last_token_id = static_cast<int32_t>(token);
      }
    }
    CHECK_GE(last_idx, 0)
        << "each sequence must have at least one accepted token";

    int32_t prev_token_id = view.token_ids[seq_id];
    CHECK_GE(prev_token_id, 0);
    torch::Tensor prev_embedding = torch::Tensor();
    const int32_t prev_idx = last_idx - 1;
    if (prev_idx >= 0) {
      int64_t token = accepted_tokens[seq_id][prev_idx].item<int64_t>();
      CHECK_GE(token, 0) << "accepted tokens should be contiguous prefix";
      prev_token_id = static_cast<int32_t>(token);
      prev_embedding = accepted_embeddings[seq_id][prev_idx];
    }
    torch::Tensor last_embedding = accepted_embeddings[seq_id][last_idx];

    const int32_t prev_offset = (last_idx > 0) ? (last_idx - 1) : -1;
    const int32_t last_offset = last_idx;
    add_row(prev_token_id, prev_offset, prev_embedding);
    add_row(last_token_id, last_offset, last_embedding);
    selected_row_idx.emplace_back(2 * seq_id + 1);
  }

  CHECK_EQ(buf.out_new_cache_slots.size(), buf.out_positions.size())
      << "draft extend slots/positions mismatch";
  CHECK_EQ(expanded_embeddings.size(), buf.out_positions.size())
      << "draft extend embeddings/positions mismatch";

  extend_input.token_ids = torch::tensor(buf.out_token_ids, int_options);
  extend_input.positions = torch::tensor(buf.out_positions, int_options);
  input_params.num_sequences = static_cast<int32_t>(buf.out_positions.size());
  input_params.q_max_seq_len = 1;
  input_params.batch_forward_type = BatchForwardType::DECODE;
  input_params.q_seq_lens_vec = std::move(buf.out_q_seq_lens);
  input_params.q_seq_lens =
      torch::tensor(input_params.q_seq_lens_vec, int_options);
  input_params.q_cu_seq_lens =
      specBuilder::build_q_cu_seq_lens_tensor(input_params);
  input_params.kv_max_seq_len = buf.kv_max_seq_len;
  input_params.kv_seq_lens_vec = std::move(buf.out_kv_seq_lens);
  input_params.kv_seq_lens =
      torch::tensor(input_params.kv_seq_lens_vec, int_options);
  input_params.new_cache_slots =
      torch::tensor(buf.out_new_cache_slots, int_options);
  util::pad_2d_vector(buf.out_block_tables, /*pad_value=*/0);
  input_params.block_tables =
      create_2d_tensor(buf.out_block_tables, torch::kInt);
  input_params.input_embedding = torch::stack(expanded_embeddings).to(device_);

  // update dp_global_token_nums for dp/ep parallel
  constexpr int32_t num_extend_tokens = 2;
  for (auto& it : input_params.dp_global_token_nums) {
    it *= num_extend_tokens;
  }

  auto& params = extend_input.sampling_params;
  torch::TensorOptions idx_options = params.selected_token_idxes.options();
  params.selected_token_idxes = torch::tensor(selected_row_idx, idx_options);
  if (!params.sample_idxes.defined()) {
    params.sample_idxes = torch::arange(num_sequences, idx_options);
  }
}

void MTPWorkerImpl::run_draft_extend(const ForwardInput& input,
                                     const SampleOutput& validate_output) {
  CHECK(!input.input_params.embedding_ids.empty())
      << "draft extend requires non-empty embedding_ids";
  CHECK(validate_output.next_tokens.defined())
      << "draft extend requires validate next_tokens";
  CHECK(validate_output.embeddings.defined())
      << "draft extend requires validate embeddings";

  ForwardInput extend_input;
  prepare_draft_extend_inputs(input, validate_output, extend_input);
  auto extend_future = draft_impl_->step_async(extend_input);
  auto extend_output_opt = std::move(extend_future).get();
  CHECK(extend_output_opt.has_value()) << "draft extend output is empty";
  ForwardOutput extend_output = std::move(extend_output_opt.value());
  process_draft_sample_output(extend_output.sample_output);
  auto& sample_output = extend_output.sample_output;
  embedding_cache_->write(input.input_params.embedding_ids,
                          sample_output.next_tokens,
                          sample_output.embeddings,
                          sample_output.probs,
                          validate_output.next_tokens);
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

  torch::Tensor positions = safe_to(input.positions, torch::kCPU);
  torch::Tensor block_tables = safe_to(input_params.block_tables, torch::kCPU);
  auto view = specBuilder::make_decode_cpu_view(
      torch::Tensor(), positions, block_tables, input_params.kv_seq_lens_vec);
  specBuilder::DecodeBuildBuffers buf;
  buf.out_positions.reserve(num_sequences);
  buf.out_kv_seq_lens.reserve(num_sequences);
  buf.out_new_cache_slots.reserve(num_sequences);

  for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
    specBuilder::RowSpec row;
    row.seq_id = seq_id;
    row.position_offset = static_cast<int32_t>(offset);
    row.append_token = false;
    specBuilder::append_decode_row(input_params, view, row, block_size, buf);
  }

  CHECK_EQ(buf.out_new_cache_slots.size(), buf.out_positions.size())
      << "draft kv slots/positions mismatch";

  draft_input.positions = torch::tensor(buf.out_positions, int_options);
  // update the input_params
  input_params.kv_max_seq_len = buf.kv_max_seq_len;
  input_params.kv_seq_lens_vec = std::move(buf.out_kv_seq_lens);
  input_params.kv_seq_lens =
      torch::tensor(input_params.kv_seq_lens_vec, int_options);
  input_params.new_cache_slots =
      torch::tensor(buf.out_new_cache_slots, int_options);
  input_params.q_cu_seq_lens =
      specBuilder::build_q_cu_seq_lens_tensor(input_params);
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

  std::vector<torch::Tensor> draft_token_ids_steps;
  std::vector<torch::Tensor> draft_probs_steps;
  draft_token_ids_steps.reserve(draft_outputs.size());
  draft_probs_steps.reserve(draft_outputs.size());
  for (const auto& draft_output : draft_outputs) {
    draft_token_ids_steps.push_back(draft_output.sample_output.next_tokens);
    draft_probs_steps.push_back(draft_output.sample_output.probs);
  }

  auto [draft_token_ids, draft_probs] =
      specBuilder::draftProbs::build_validate_tensors(
          draft_token_ids_steps,
          draft_probs_steps,
          batch_size,
          vocab_size,
          enable_opt_validate_probs_);
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

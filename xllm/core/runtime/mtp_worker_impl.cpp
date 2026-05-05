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
#if defined(USE_MLU)
#include "framework/kv_cache_transfer/mooncake_kv_cache_transfer.h"
#endif
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
  opts.enable_schedule_overlap(false).is_draft_engine(false);
  return opts;
}

runtime::Options MTPDraftOptions(const runtime::Options& options) {
  auto opts = options;
  opts.enable_schedule_overlap(false)
      .is_draft_engine(true)
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

bool MTPWorkerImpl::allocate_kv_cache(const KVCacheShape& kv_cache_shape) {
  const int64_t num_blocks = kv_cache_shape.key_cache_shape()[0];
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

#if defined(USE_NPU) || defined(USE_MLU)
bool MTPWorkerImpl::allocate_kv_cache_with_transfer(
    const KVCacheShape& kv_cache_shape) {
  const int64_t num_blocks = kv_cache_shape.key_cache_shape()[0];
  CHECK(impl_ != nullptr);
  CHECK(draft_impl_ != nullptr);

  if (kv_cache_transfer_ == nullptr) {
#if defined(USE_NPU)
    kv_cache_transfer_ = std::make_shared<SpecKVCacheTransfer>(
        options_.device_ip().value(),
        options_.transfer_listen_port(),
        options_.instance_role(),
        context_.get_model_args().model_type());
#elif defined(USE_MLU)
    CHECK_EQ(FLAGS_kv_cache_transfer_type, "Mooncake")
        << "MLU MTP push only supports Mooncake KV transfer.";
    kv_cache_transfer_ = std::make_shared<MooncakeKVCacheTransferDefault>(
        device_.index(),
        options_.transfer_listen_port(),
        device_,
        context_.get_model_args().model_type());
#endif

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

  embedding_cache_ = std::make_shared<EmbeddingCache>(num_blocks);
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
  return inputs;
}

void MTPWorkerImpl::prepare_work_before_execute(const ForwardInput& input,
                                                ForwardInput& processed_input) {
  SpeculativeWorkerImpl::prepare_work_before_execute(input, processed_input);
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
    ForwardInput new_input = input;
    for (int32_t& token_num : new_input.input_params.dp_global_token_nums) {
      token_num *= 2;
    }
    auto draft_extend_future = draft_impl_->step_async(new_input);
    ForwardOutput draft_extend_output =
        std::move(draft_extend_future).get().value();
    (void)draft_extend_output;

    for (int32_t i = 1; i < options_.num_speculative_tokens(); ++i) {
      auto draft_future = draft_impl_->step_async(input);
      ForwardOutput draft_output = std::move(draft_future).get().value();
      (void)draft_output;
    }

    new_input = input;
    for (int32_t& token_num : new_input.input_params.dp_global_token_nums) {
      token_num *= options_.num_speculative_tokens() + 1;
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
  process_draft_sample_output(draft_output.sample_output);
  COUNTER_ADD(speculative_execution_latency_seconds_draft,
              timer.elapsed_seconds());

  if (input.sampling_params.selected_token_idxes.defined()) {
    embedding_cache_->write_prefill_target_context(
        input.input_params.embedding_ids,
        input.input_params.request_ids,
        output.sample_output.next_tokens,
        embeddings,
        input.sampling_params.selected_token_idxes);
  }
  output.sample_output.embeddings = torch::Tensor();

  if (!enable_schedule_overlap() && !driver_ && !dp_driver_) {
    return std::nullopt;
  }
  return output;
}

void MTPWorkerImpl::prepare_prefill_inputs(const ForwardInput& input,
                                           ForwardInput& prefill_input) {
  c10::StreamGuard stream_guard = prepare_stream_->set_stream_guard();
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
  for (int32_t i = 0; i < input_params.num_sequences; ++i) {
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
  prepare_stream_->synchronize();
}

std::optional<ForwardOutput> MTPWorkerImpl::step_decode(
    const ForwardInput& input) {
  const int32_t num_speculative_tokens = options_.num_speculative_tokens();

  std::vector<ForwardOutput> draft_outputs;
  ForwardInput current_draft_input, validate_input, next_step_input;
  Timer timer;

  // Get decode state of last step
  std::vector<EmbeddingCache::DecodeState> last_states =
      embedding_cache_->read_decode_states(input.input_params.embedding_ids,
                                           input.input_params.request_ids);
  CHECK_EQ(last_states.size(), input.input_params.embedding_ids.size())
      << "decode target state count mismatch";
  specBuilder::DecodeCpuView decode_view;
  build_decode_step_view(input, last_states, decode_view);
  prepare_draft_extend_inputs(
      input, decode_view, last_states, current_draft_input);
  draft_outputs.reserve(num_speculative_tokens);
  for (int32_t draft_idx = 0; draft_idx < num_speculative_tokens; ++draft_idx) {
    auto future = draft_impl_->step_async(current_draft_input);

    // Overlap next-step input preparation with async draft forward.
    if (draft_idx == num_speculative_tokens - 1) {
      prepare_validate_inputs(input, decode_view, validate_input);
    } else {
      prepare_draft_inputs(input, decode_view, next_step_input, draft_idx + 1);
    }

    std::optional<ForwardOutput> draft_output_opt = std::move(future).get();
    CHECK(draft_output_opt.has_value())
        << "draft output is empty in speculative step";

    draft_outputs.push_back(std::move(draft_output_opt.value()));
    process_draft_sample_output(draft_outputs.back().sample_output);
    if (draft_idx == num_speculative_tokens - 1) {
      continue;
    }

    const SampleOutput& last_output = draft_outputs.back().sample_output;
    current_draft_input = next_step_input;
    current_draft_input.token_ids =
        safe_to(last_output.next_tokens, torch::kInt);
    current_draft_input.input_params.input_embedding = last_output.embeddings;
  }
  COUNTER_ADD(speculative_execution_latency_seconds_draft,
              timer.elapsed_seconds());
  return run_validate(input, draft_outputs, validate_input);
}

void MTPWorkerImpl::fill_validate_input_from_draft_outputs(
    const std::vector<ForwardOutput>& draft_outputs,
    ForwardInput& validate_input) {
  for (int32_t i = 0; i < options_.num_speculative_tokens(); ++i) {
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
  fill_validate_input_from_draft_outputs(draft_outputs, validate_input);
  auto future = impl_->step_async(validate_input);
  ForwardOutput target_output = std::move(future).get().value();
  COUNTER_ADD(speculative_execution_latency_seconds_target,
              timer.elapsed_seconds());

  // verify the proposals with target and update the batch
  timer.reset();
  SampleOutput val_output =
      validate(input.sampling_params, draft_outputs, target_output);
  record_validate_metrics(val_output);
  COUNTER_ADD(speculative_execution_latency_seconds_validation,
              timer.elapsed_seconds());

  val_output.next_tokens = val_output.next_tokens.to(torch::kCPU);
  write_target_context_to_cache(input, val_output);

  if (!enable_schedule_overlap() && !driver_ && !dp_driver_) {
    return std::nullopt;
  }
  val_output.embeddings = torch::Tensor();
  target_output.sample_output = val_output;
  return target_output;
}

void MTPWorkerImpl::write_target_context_to_cache(
    const ForwardInput& input,
    const SampleOutput& validate_output) {
  CHECK(embedding_cache_ != nullptr)
      << "embedding_cache_ must be initialized before target cache write";
  CHECK(!input.input_params.embedding_ids.empty())
      << "target context cache write requires embedding ids";
  embedding_cache_->write_target_context(input.input_params.embedding_ids,
                                         input.input_params.request_ids,
                                         validate_output.next_tokens,
                                         validate_output.embeddings,
                                         options_.num_speculative_tokens());
}

void MTPWorkerImpl::record_validate_metrics(
    const SampleOutput& validate_output) const {
  CHECK(validate_output.next_tokens.defined())
      << "validate output tokens are undefined";
  const int32_t batch_size =
      static_cast<int32_t>(validate_output.next_tokens.size(0));
  const int32_t num_draft_tokens =
      batch_size * options_.num_speculative_tokens();
  torch::Tensor mask = (validate_output.next_tokens == -1).to(torch::kInt64);
  const int64_t rejected_count = mask.sum().item<int64_t>();
  COUNTER_ADD(speculative_num_draft_tokens_total, num_draft_tokens);
  COUNTER_ADD(speculative_num_accepted_tokens_total,
              num_draft_tokens - rejected_count);
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

void MTPWorkerImpl::build_decode_step_view(
    const ForwardInput& input,
    const std::vector<EmbeddingCache::DecodeState>& last_states,
    specBuilder::DecodeCpuView& decode_view) const {
  const int32_t num_sequences = input.input_params.num_sequences;
  CHECK_EQ(last_states.size(), static_cast<size_t>(num_sequences))
      << "decode context state count mismatch";
  const bool enable_cache_correction = enable_schedule_overlap();

  decode_view.token_ids_vec.clear();
  decode_view.positions_vec.clear();
  decode_view.kv_seq_lens_vec.clear();
  decode_view.token_ids_vec.reserve(num_sequences);
  decode_view.positions_vec.reserve(num_sequences);
#if defined(USE_NPU)
  decode_view.kv_seq_lens_vec.reserve(num_sequences);
#else
  decode_view.kv_seq_lens_vec.reserve(num_sequences + 1);
#endif

  torch::Tensor token_ids_cpu = safe_to(input.token_ids, torch::kCPU);
  torch::Tensor positions_cpu = safe_to(input.positions, torch::kCPU);
  Slice<int32_t> input_token_ids = {token_ids_cpu.data_ptr<int32_t>(),
                                    static_cast<size_t>(token_ids_cpu.numel())};
  decode_view.block_tables_cpu =
      safe_to(input.input_params.block_tables, torch::kCPU);
  Slice<int32_t> input_positions = {positions_cpu.data_ptr<int32_t>(),
                                    static_cast<size_t>(positions_cpu.numel())};

  for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
    CHECK_LT(static_cast<size_t>(seq_id), input_token_ids.size())
        << "decode context token seq_id out of range, seq_id=" << seq_id;
    CHECK_LT(static_cast<size_t>(seq_id), input_positions.size())
        << "decode context position seq_id out of range, seq_id=" << seq_id;
    const EmbeddingCache::DecodeState& state = last_states[seq_id];
    const int32_t input_token_id = input_token_ids[seq_id];
    const bool input_is_fake_token = input_token_id < 0;
    const bool use_cache_correction =
        enable_cache_correction && input_is_fake_token && state.valid;
    const bool use_fake_context =
        enable_cache_correction && input_is_fake_token && !state.valid;
    const int32_t position_offset =
        use_cache_correction ? state.position_offset : 0;
    const int32_t current_position = input_positions[seq_id] + position_offset;
    const int32_t current_kv_len = specBuilder::calc_kv_len(
        input.input_params.kv_seq_lens_vec, seq_id, position_offset);

    CHECK_EQ(current_position + 1, current_kv_len)
        << "decode context position/kv_len mismatch, seq_id=" << seq_id
        << ", current_position=" << current_position
        << ", current_kv_len=" << current_kv_len;

    decode_view.token_ids_vec.emplace_back(
        (use_cache_correction || use_fake_context) ? state.token_id
                                                   : input_token_id);
    decode_view.positions_vec.emplace_back(current_position);
    specBuilder::append_seq_len_by_layout(decode_view.kv_seq_lens_vec,
                                          current_kv_len);
  }

  decode_view.block_tables_cpu = decode_view.block_tables_cpu.contiguous();
  decode_view.block_tables_data = {
      decode_view.block_tables_cpu.data_ptr<int32_t>(),
      static_cast<size_t>(decode_view.block_tables_cpu.numel())};
  CHECK_EQ(decode_view.block_tables_cpu.dim(), 2)
      << "decode context block tables must be 2D, got "
      << decode_view.block_tables_cpu.sizes();
  decode_view.num_sequences =
      static_cast<int32_t>(decode_view.block_tables_cpu.size(0));
  decode_view.block_table_row_stride =
      static_cast<int32_t>(decode_view.block_tables_cpu.size(1));
  decode_view.token_ids = decode_view.token_ids_vec;
  decode_view.positions = decode_view.positions_vec;
  decode_view.kv_seq_lens = decode_view.kv_seq_lens_vec;
}

void MTPWorkerImpl::prepare_validate_inputs(
    const ForwardInput& input,
    const specBuilder::DecodeCpuView& decode_view,
    ForwardInput& validate_input) {
  c10::StreamGuard stream_guard = prepare_stream_->set_stream_guard();
  validate_input = input;
  auto& input_params = validate_input.input_params;
  torch::TensorOptions int_options = validate_input.token_ids.options();

  const int32_t num_speculative_tokens = options_.num_speculative_tokens();
  const int32_t num_sequences = input_params.num_sequences;
  const int32_t num_val_tokens = num_speculative_tokens + 1;
  const int32_t total_num_val_tokens = num_sequences * num_val_tokens;
  const int32_t block_size = options_.block_size();
  const specBuilder::DecodeCpuView& view = decode_view;
  specBuilder::DecodeBuildBuffers buf;
  buf.out_token_ids.reserve(total_num_val_tokens);
  buf.out_positions.reserve(total_num_val_tokens);
  buf.out_new_cache_slots.reserve(total_num_val_tokens);
  if (!FLAGS_enable_atb_spec_kernel) {
    buf.out_kv_seq_lens.reserve(total_num_val_tokens);
    buf.out_q_seq_lens.reserve(total_num_val_tokens);
    buf.out_block_tables.reserve(total_num_val_tokens);
  }

  std::vector<int32_t> atb_kv_seq_lens_vec;
  std::vector<int32_t> atb_q_seq_lens_vec;
  int32_t atb_kv_max_seq_len = 0;
  for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
    const int32_t start_position = view.positions[seq_id];
    const int32_t kv_len =
        specBuilder::calc_kv_len(view.kv_seq_lens, seq_id, /*offset=*/0);
    CHECK_EQ(start_position + 1, kv_len)
        << "validate position/kv_len mismatch, seq_id=" << seq_id
        << ", start_position=" << start_position << ", kv_len=" << kv_len;

    for (int32_t val_idx = 0; val_idx < num_val_tokens; ++val_idx) {
      specBuilder::RowSpec row;
      row.seq_id = seq_id;
      row.token_id = val_idx == 0 ? view.token_ids[seq_id] : -val_idx;
      row.position_offset = val_idx;
      row.append_kv_len = !FLAGS_enable_atb_spec_kernel;
      row.append_q_len_one = !FLAGS_enable_atb_spec_kernel;
      row.append_block_table = !FLAGS_enable_atb_spec_kernel;
      specBuilder::append_decode_row(view, row, block_size, buf);
    }

    if (FLAGS_enable_atb_spec_kernel) {
      const int32_t kv_len_after_validation = kv_len + num_speculative_tokens;
      specBuilder::update_kv_seq_lens_and_max(
          atb_kv_seq_lens_vec, kv_len_after_validation, atb_kv_max_seq_len);
      specBuilder::append_seq_len_by_layout(atb_q_seq_lens_vec, num_val_tokens);
    }
  }

  CHECK_EQ(buf.out_new_cache_slots.size(), buf.out_token_ids.size())
      << "validate kv slots/tokens mismatch";
  CHECK_EQ(buf.out_positions.size(), buf.out_token_ids.size())
      << "validate positions/tokens mismatch";

  validate_input.token_ids = torch::tensor(buf.out_token_ids, int_options);
  validate_input.positions = torch::tensor(buf.out_positions, int_options);
  if (!FLAGS_enable_atb_spec_kernel) {
    input_params.num_sequences = total_num_val_tokens;
    input_params.batch_forward_type = BatchForwardType::DECODE;
  } else {
    input_params.batch_forward_type = BatchForwardType::CHUNKED_PREFILL;
  }
  if (FLAGS_enable_atb_spec_kernel) {
    specBuilder::update_input_params(input_params,
                                     buf,
                                     int_options,
                                     num_val_tokens,
                                     std::move(atb_q_seq_lens_vec),
                                     atb_kv_max_seq_len,
                                     std::move(atb_kv_seq_lens_vec));
  } else {
    specBuilder::update_input_params(input_params,
                                     buf,
                                     int_options,
                                     1,
                                     std::move(buf.out_q_seq_lens),
                                     buf.kv_max_seq_len,
                                     std::move(buf.out_kv_seq_lens),
                                     /*update_block_tables=*/true,
                                     device_);
  }

  update_sampling_params(
      validate_input.sampling_params, num_val_tokens, total_num_val_tokens);

  for (int32_t& token_num : input_params.dp_global_token_nums) {
    token_num *= num_val_tokens;
  }

  validate_input = validate_input.to(device_, dtype_);
  prepare_stream_->synchronize();
}

void MTPWorkerImpl::prepare_draft_extend_inputs(
    const ForwardInput& base_input,
    const specBuilder::DecodeCpuView& decode_view,
    const std::vector<EmbeddingCache::DecodeState>& last_states,
    ForwardInput& extend_input) {
  extend_input = base_input;
  auto& input_params = extend_input.input_params;
  const int32_t num_sequences = input_params.num_sequences;

  const bool dp_enabled = parallel_args_.dp_size() > 1;
  CHECK_EQ(last_states.size(), static_cast<size_t>(num_sequences))
      << "draft extend state count mismatch";

  const int32_t block_size = options_.block_size();
  torch::TensorOptions int_options = extend_input.token_ids.options();
  const specBuilder::DecodeCpuView& view = decode_view;

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

  torch::Tensor placeholder = embedding_cache_->embedding_placeholder();
  CHECK(placeholder.defined())
      << "embedding placeholder must be initialized for fake draft context";
  placeholder = placeholder.to(device_);

  for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
    auto add_row = [&](int32_t token_id,
                       int32_t position_offset,
                       const torch::Tensor& embedding) {
      specBuilder::RowSpec row;
      row.seq_id = seq_id;
      row.token_id = token_id >= 0 ? token_id : 0;
      row.position_offset = position_offset;
      row.append_q_len_one = true;
      row.append_block_table = true;
      specBuilder::append_decode_row(view, row, block_size, buf);
      if (embedding.defined()) {
        expanded_embeddings.emplace_back(embedding.to(device_));
      } else {
        expanded_embeddings.emplace_back(placeholder);
      }
    };

    EmbeddingCache::DecodeState state = last_states[seq_id];
    const int32_t current_token_id = view.token_ids[seq_id];
    if (!state.valid || state.token_id != current_token_id) {
      state = EmbeddingCache::DecodeState();
      state.token_id = current_token_id >= 0 ? current_token_id : 0;
    }
    const bool use_two_rows = dp_enabled || state.all_draft_accepted;
    if (use_two_rows) {
      int32_t prev_token_id = state.prev_token_id;
      int32_t prev_position_offset = -1;
      torch::Tensor prev_embedding = state.prev_embedding;
      if (prev_token_id < 0) {
        prev_token_id = state.token_id;
        prev_embedding = torch::Tensor();
      }
      add_row(prev_token_id, prev_position_offset, prev_embedding);
    }

    selected_row_idx.emplace_back(
        static_cast<int32_t>(expanded_embeddings.size()));
    add_row(state.token_id, /*position_offset=*/0, state.embedding);
  }

  CHECK_EQ(buf.out_new_cache_slots.size(), buf.out_positions.size())
      << "draft extend slots/positions mismatch";
  CHECK_EQ(expanded_embeddings.size(), buf.out_positions.size())
      << "draft extend embeddings/positions mismatch";

  extend_input.token_ids = torch::tensor(buf.out_token_ids, int_options);
  extend_input.positions = torch::tensor(buf.out_positions, int_options);
  input_params.num_sequences = static_cast<int32_t>(buf.out_positions.size());
  input_params.batch_forward_type = BatchForwardType::DECODE;
  specBuilder::update_input_params(input_params,
                                   buf,
                                   int_options,
                                   1,
                                   std::move(buf.out_q_seq_lens),
                                   buf.kv_max_seq_len,
                                   std::move(buf.out_kv_seq_lens),
                                   /*update_block_tables=*/true);
  input_params.input_embedding = torch::stack(expanded_embeddings).to(device_);

  if (!input_params.dp_global_token_nums.empty()) {
    if (dp_enabled) {
      constexpr int32_t num_extend_tokens = 2;
      for (int32_t& token_num : input_params.dp_global_token_nums) {
        token_num *= num_extend_tokens;
      }
    } else if (input_params.dp_global_token_nums.size() == 1) {
      input_params.dp_global_token_nums[0] =
          static_cast<int32_t>(buf.out_positions.size());
    }
  }

  auto& params = extend_input.sampling_params;
  torch::TensorOptions idx_options =
      params.selected_token_idxes.defined()
          ? params.selected_token_idxes.options()
          : torch::dtype(torch::kInt).device(device_);
  params.selected_token_idxes = torch::tensor(selected_row_idx, idx_options);
  if (!params.sample_idxes.defined()) {
    params.sample_idxes = torch::arange(num_sequences, idx_options);
  }
}

void MTPWorkerImpl::prepare_draft_inputs(
    const ForwardInput& input,
    const specBuilder::DecodeCpuView& decode_view,
    ForwardInput& draft_input,
    int32_t position_offset) {
  c10::StreamGuard stream_guard = prepare_stream_->set_stream_guard();
  draft_input = input;

  auto& input_params = draft_input.input_params;
  const int32_t num_sequences = input_params.num_sequences;
  const int32_t block_size = options_.block_size();
  const specBuilder::DecodeCpuView& view = decode_view;
  specBuilder::DecodeBuildBuffers buf;
  buf.out_positions.reserve(num_sequences);
  buf.out_kv_seq_lens.reserve(num_sequences);
  buf.out_new_cache_slots.reserve(num_sequences);

  for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
    specBuilder::RowSpec row;
    row.seq_id = seq_id;
    row.position_offset = position_offset;
    row.append_token = false;
    specBuilder::append_decode_row(view, row, block_size, buf);
  }

  CHECK_EQ(buf.out_new_cache_slots.size(), buf.out_positions.size())
      << "draft kv slots/positions mismatch";

  torch::TensorOptions int_options = input.token_ids.options();
  draft_input.positions = torch::tensor(buf.out_positions, int_options);
  specBuilder::update_input_params(input_params,
                                   buf,
                                   int_options,
                                   input_params.q_max_seq_len,
                                   std::move(input_params.q_seq_lens_vec),
                                   buf.kv_max_seq_len,
                                   std::move(buf.out_kv_seq_lens));

  draft_input = draft_input.to(device_, dtype_);
  prepare_stream_->synchronize();
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

  return sample_output;
}

}  // namespace xllm

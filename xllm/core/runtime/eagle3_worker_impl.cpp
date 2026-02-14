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

#include "eagle3_worker_impl.h"

#include <glog/logging.h>

#include "common/metrics.h"
#include "framework/model_loader.h"
#include "framework/sampling/rejection_sampler.h"
#include "util/utils.h"

namespace xllm {

namespace {

runtime::Options eagle3_main_options(const runtime::Options& options) {
  auto opts = options;
  opts.enable_schedule_overlap(false).enable_graph_aux_hidden_states(true);
  return opts;
}

runtime::Options eagle3_draft_options(const runtime::Options& options) {
  auto opts = options;
  opts.enable_schedule_overlap(false)
      .num_decoding_tokens(1)
      .num_speculative_tokens(0)
      .enable_graph_aux_hidden_states(false);
  return opts;
}

}  // namespace

Eagle3WorkerImpl::Eagle3WorkerImpl(const ParallelArgs& parallel_args,
                                   const torch::Device& device,
                                   const runtime::Options& options)
    : SpeculativeWorkerImpl(parallel_args,
                            device,
                            options,
                            eagle3_main_options(options),
                            eagle3_draft_options(options)) {}

bool Eagle3WorkerImpl::init_model(const std::string& model_weights_path,
                                  int32_t random_seed) {
  // Call parent's init_model first
  bool result =
      SpeculativeWorkerImpl::init_model(model_weights_path, random_seed);

  // Load hot_token_id_ directly from state_dict (EAGLE-3 specific)
  // This should be done after draft model is loaded
  if (draft_impl_->get_status() == WorkerImpl::Status::LOADED) {
    // d2t stores diffs between draft id and target id
    // hot_token_id = d2t + arange(d2t.size(0))
    auto model_loader = ModelLoader::create(model_weights_path);
    auto& state_dicts = model_loader->get_state_dicts();
    for (const auto& state_dict : state_dicts) {
      torch::Tensor d2t_tensor = state_dict->get_tensor("d2t");
      if (d2t_tensor.defined()) {
        auto arange_tensor = torch::arange(d2t_tensor.size(0));
        hot_token_id_ = d2t_tensor + arange_tensor;
        hot_token_id_ = hot_token_id_.to(device_);
        LOG(INFO) << "Eagle3WorkerImpl: Loaded d2t tensor from state_dict, "
                     "hot_token_id size: "
                  << hot_token_id_.size(0);
        break;
      }
    }
  }

  return result;
}

int64_t Eagle3WorkerImpl::get_embedding_placeholder_size() {
  const int64_t target_hidden = context_.get_model_args().hidden_size();
  return 3 * target_hidden;
}

std::optional<ForwardOutput> Eagle3WorkerImpl::step_decode(
    const ForwardInput& input) {
  // TODO : now only support Deepseek MTP
  // More work need to support n-gram and native speculative decoding.
  ForwardInput draft_input = input;
  // get embedding cache
  torch::Tensor embeddings =
      embedding_cache_->read(draft_input.input_params.embedding_ids);
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
    auto& last_output = draft_outputs.back().sample_output;

    // Extract probability for selected draft token
    if (last_output.probs.defined()) {
      auto selected_probs =
          last_output.probs
              .gather(
                  /*dim=*/-1, last_output.next_tokens.unsqueeze(-1))
              .squeeze(-1);
      last_output.probs = selected_probs;  // [batch_size]
    }

    // EAGLE-3 specific: map draft token IDs to target token IDs using
    // hot_token_id_
    if (hot_token_id_.defined()) {
      last_output.next_tokens =
          hot_token_id_.index_select(0, last_output.next_tokens);
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
                                   val_output.embeddings);

  if (!enable_schedule_overlap() && !driver_ && !dp_driver_) {
    return std::nullopt;
  }
  val_output.embeddings = torch::Tensor();
  target_output.sample_output = val_output;
  return target_output;
}

SampleOutput Eagle3WorkerImpl::validate(
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
  // Initialize draft_probs with zeros (target vocab size)
  // Note: draft vocab size != target vocab size in EAGLE-3
  auto draft_probs =
      torch::zeros({batch_size, options_.num_speculative_tokens(), vocab_size},
                   target_logits.options());

  // prepare input for rejection sampling
  // For EAGLE-3, draft_token_ids are already mapped to target token IDs
  // via hot_token_id_ in step_decode
  // draft_output.sample_output.probs has been extracted to [batch_size] in
  // step_decode
  auto batch_indices =
      torch::arange(batch_size, torch::kInt64).to(draft_probs.device());
  std::vector<torch::Tensor> draft_token_ids_vec;
  for (size_t spec_step = 0; spec_step < draft_outputs.size(); ++spec_step) {
    const auto& draft_output = draft_outputs[spec_step];
    auto draft_token_ids = draft_output.sample_output.next_tokens;

    auto token_indices =
        torch::full(
            {batch_size}, static_cast<int64_t>(spec_step), torch::kInt64)
            .to(draft_token_ids.device());

    // draft_output.probs is already [batch_size] - probability of selected
    // token
    auto selected_probs = draft_output.sample_output.probs;
    draft_probs.index_put_({batch_indices, token_indices, draft_token_ids},
                           selected_probs);
    draft_token_ids_vec.push_back(draft_token_ids.view({batch_size, 1}));
  }

  // concatenate the draft token ids along the last dimension
  const auto draft_token_ids =
      torch::cat(draft_token_ids_vec, /*dim=*/1).to(bonus_token_ids);

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

}  // namespace xllm

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
    : MTPWorkerImpl(parallel_args,
                    device,
                    options,
                    eagle3_main_options(options),
                    eagle3_draft_options(options),
                    /*enable_opt_validate_probs=*/true) {}

bool Eagle3WorkerImpl::init_model(const std::string& model_weights_path,
                                  int32_t random_seed,
                                  int32_t master_status) {
  // Call parent's init_model first
  bool result =
      MTPWorkerImpl::init_model(model_weights_path, random_seed, master_status);

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

void Eagle3WorkerImpl::process_draft_output(ForwardOutput& draft_output) {
  auto& output = draft_output.sample_output;
  if (output.probs.defined()) {
    auto selected_probs = output.probs
                              .gather(
                                  /*dim=*/-1, output.next_tokens.unsqueeze(-1))
                              .squeeze(-1);
    output.probs = selected_probs;  // [batch_size]
  }

  // EAGLE-3 specific: map draft token IDs to target token IDs.
  if (hot_token_id_.defined()) {
    output.next_tokens = hot_token_id_.index_select(0, output.next_tokens);
  }
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

  // prepare input for rejection sampling
  // For EAGLE-3, draft_token_ids are already mapped to target token IDs
  // via hot_token_id_ in step_decode
  // draft_output.sample_output.probs has been extracted to [batch_size] in
  // step_decode
  std::vector<torch::Tensor> draft_token_ids_vec;
  std::vector<torch::Tensor> selected_draft_probs_vec;
  selected_draft_probs_vec.reserve(draft_outputs.size());
  for (size_t i = 0; i < draft_outputs.size(); ++i) {
    const auto& draft_output = draft_outputs[i];
    auto draft_token_ids = draft_output.sample_output.next_tokens;
    auto selected_probs = draft_output.sample_output.probs;
    selected_draft_probs_vec.push_back(selected_probs.view({batch_size, 1}));
    draft_token_ids_vec.push_back(draft_token_ids.view({batch_size, 1}));
  }

  // Optimized path for Eagle3:
  // keep only selected draft token probabilities [B, S].
  auto draft_probs = torch::cat(selected_draft_probs_vec, /*dim=*/1);

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

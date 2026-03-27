/* Copyright 2025 The xLLM Authors. All Rights Reserved.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include "sampler.h"

#include <glog/logging.h>
#include <torch/torch.h>

#include <utility>

#include "common/global_flags.h"
#include "logits_utils.h"
#include "sampling_params.h"

namespace xllm {

SampleOutput Sampler::forward(torch::Tensor& logits,
                              const SamplingParameters& params) const {
  SampleOutput output;
  // apply frequency and presence penalties
  if (params.frequency_penalties.defined()) {
    apply_frequency_presence_penalties(logits,
                                       params.unique_token_ids,
                                       params.unique_token_counts,
                                       params.frequency_penalties,
                                       params.presence_penalties);
  }

  // apply repetition penalties
  if (params.repetition_penalties.defined()) {
    apply_repetition_penalties(
        logits, params.unique_token_ids, params.repetition_penalties);
  }

  // apply temperatures, top-k and top-p
  apply_top_k_top_p(logits, params.temperatures, params.top_k, params.top_p);

  torch::Tensor sample_logits = logits;
  if (params.selected_token_idxes.numel() != params.sample_idxes.numel()) {
    sample_logits = logits.index_select(/*dim=*/0, params.sample_idxes);
  }

  // same batch size
  CHECK_EQ(sample_logits.size(0), params.do_sample.size(0));

  auto probs = sample_logits;
  torch::Tensor samples;
  if (params.all_random_sample) {
    // use float32 for probabilities and log probabilities
    probs =
        torch::softmax(sample_logits, /*dim=*/-1, /*dtype=*/torch::kFloat32);
    samples = random_sample(probs);
  } else if (params.all_greedy_sample) {
    samples = greedy_sample(probs);
  } else {
    // use float32 for probabilities and log probabilities
    probs =
        torch::softmax(sample_logits, /*dim=*/-1, /*dtype=*/torch::kFloat32);
    // mixed sample, sample both then choose based on do_sample
    auto random = random_sample(probs);
    auto greedy = greedy_sample(probs);
    samples = torch::where(params.do_sample, random, greedy);
  }
  output.probs = probs.to(logits.dtype());
  output.next_tokens = samples;

  const bool has_candidate_ids = has_candidate_token_ids();
  if (has_candidate_ids) {
    CHECK_EQ(sample_logits.size(-1), candidate_token_ids_tensor_.size(0))
        << "Candidate token ids size must match sampler logits last dimension.";
  }

  if (params.logprobs) {
    // log_softmax is equivalent to log(softmax) but more numerically stable
    const auto logprobs = torch::log_softmax(
        sample_logits, /*dim=*/-1, /*dtype=*/torch::kFloat32);
    // select the logprobs for each sequence
    auto selected_logprobs = logprobs.gather(/*dim=*/-1, samples.view({-1, 1}));
    selected_logprobs = selected_logprobs.view({-1});

    if (FLAGS_enable_qwen3_reranker) {
      CHECK(has_candidate_ids)
          << "Qwen3 reranker mode requires candidate token ids.";
      CHECK_EQ(sample_logits.size(-1), 2)
          << "Qwen3 reranker requires exactly two candidate logits.";

      selected_logprobs =
          logprobs.index({torch::indexing::Slice(), 1}).view({-1}).exp();
    }
    output.logprobs = selected_logprobs;

    if (params.max_top_logprobs > 0) {
      auto [values, indices] =
          logprobs.topk(params.max_top_logprobs, /*dim=*/-1);
      output.top_logprobs = values;
      output.top_tokens = indices;
    }
  }

  if (has_candidate_ids) {
    output.next_tokens = map_local_to_global_token_ids(output.next_tokens);
    if (output.top_tokens.defined()) {
      output.top_tokens = map_local_to_global_token_ids(output.top_tokens);
    }
  }

  return output;
}

void Sampler::set_candidate_token_ids(std::vector<int64_t> candidate_token_ids,
                                      std::optional<torch::Device> device) {
  if (candidate_token_ids.empty()) {
    candidate_token_ids_tensor_ = torch::Tensor();
    return;
  }

  auto options = torch::TensorOptions().dtype(torch::kLong);
  if (device.has_value()) {
    options = options.device(device.value());
  }
  candidate_token_ids_tensor_ = torch::tensor(candidate_token_ids, options);
}

torch::Tensor Sampler::map_local_to_global_token_ids(
    const torch::Tensor& token_ids) const {
  if (!token_ids.defined() || !has_candidate_token_ids()) {
    return token_ids;
  }

  auto local_token_ids = token_ids.reshape({-1});
  if (local_token_ids.scalar_type() != torch::kLong) {
    local_token_ids = local_token_ids.to(torch::kLong);
  }

  torch::Tensor mapping = candidate_token_ids_tensor_;
  if (mapping.device() != token_ids.device()) {
    mapping = mapping.to(token_ids.device(),
                         /*dtype=*/torch::kLong,
                         /*non_blocking=*/true,
                         /*copy=*/false);
  }
  return mapping.index_select(/*dim=*/0, local_token_ids).view_as(token_ids);
}

torch::Tensor Sampler::greedy_sample(const torch::Tensor& probs) {
  return probs.argmax(/*dim=*/-1);
}

torch::Tensor Sampler::random_sample(const torch::Tensor& probs) {
#if defined(USE_MLU)
  xllm::kernel::RandomSampleParams params;
  params.logits = probs;
  return xllm::kernel::random_sample(params);
#endif
  if (probs.dim() == 3) {
    auto batch_size = probs.size(0);
    auto seq_len = probs.size(1);
    auto vocab_size = probs.size(2);
    auto flat_probs = probs.reshape({-1, vocab_size});
    auto sampled =
        flat_probs.multinomial(/*num_samples=*/1, /*replacement=*/false);
    return sampled.reshape({batch_size, seq_len});
  } else {
    return probs.multinomial(/*num_samples=*/1, /*replacement=*/false)
        .flatten();
  }
}

}  // namespace xllm

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

#include "rec_sampler.h"

#include <glog/logging.h>
#include <torch/torch.h>

#include <cstdlib>
#include <tuple>

#include "common/global_flags.h"
#include "logits_utils.h"
#include "sampler.h"
#if defined(USE_CUDA)
#include "kernels/cuda/cuda_ops_api.h"
#endif

namespace xllm {
namespace {

static inline bool use_air_log_softmax_env() {
  static const bool enabled = []() {
    const char* v = std::getenv("XLLM_USE_AIR_LOG_SOFTMAX");
    if (!v) {
      return false;
    }
    char c = v[0];
    return c == '1' || c == 't' || c == 'T' || c == 'y' || c == 'Y';
  }();
  return enabled;
}

static inline bool can_use_fast_path(const SamplingParameters& params,
                                     bool enable_fast_path) {
  return params.use_beam_search && params.logprobs && enable_fast_path &&
         params.max_top_logprobs > 0 && !params.top_p.defined() &&
         !FLAGS_enable_qwen3_reranker;
}

static inline torch::Tensor log_softmax_last_dim(
    const torch::Tensor& input,
    const torch::Tensor& temperatures) {
  const bool has_temps = temperatures.defined();
#if defined(USE_CUDA)
  if (input.is_cuda() && use_air_log_softmax_env()) {
    return kernel::cuda::air_log_softmax_last_dim(input, temperatures);
  }
#endif

  if (!has_temps) {
    return torch::log_softmax(input, /*dim=*/-1, /*dtype=*/torch::kFloat32);
  }

  auto logits = input.to(torch::kFloat32);
  auto temps = temperatures.to(torch::kFloat32).to(input.device()).unsqueeze(1);
  temps = torch::where(temps == 0, torch::ones_like(temps), temps);
  logits.div_(temps);
  return torch::log_softmax(logits, /*dim=*/-1);
}

}  // namespace

RecSampler::RecSampler(RecPipelineType pipeline_type, bool enable_fast_path)
    : sampler_(std::make_unique<Sampler>()),
      strategy_(
          create_sampling_strategy(pipeline_type, *sampler_, enable_fast_path)),
      enable_fast_path_(enable_fast_path) {
  LOG(INFO) << "RecSampler initialized with Sampler delegate, fast_path="
            << enable_fast_path;
}

RecSampler::~RecSampler() = default;

SampleOutput RecSampler::forward(torch::Tensor& logits,
                                 const SamplingParameters& params) const {
  return strategy_->forward(logits, params);
}

std::unique_ptr<RecSampler::SamplingStrategy>
RecSampler::create_sampling_strategy(RecPipelineType type,
                                     const Sampler& sampler,
                                     bool enable_fast_path) {
  switch (type) {
    case RecPipelineType::kLlmRecMultiRoundPipeline:
      return std::make_unique<MultiRoundFastPathSamplingStrategy>(
          sampler, enable_fast_path);
    case RecPipelineType::kLlmRecDefault:
    case RecPipelineType::kLlmRecWithMmData:
    case RecPipelineType::kOneRecDefault:
      return std::make_unique<DefaultSamplingStrategy>(sampler);
    default:
      LOG(FATAL) << "Unknown RecPipelineType: " << static_cast<int>(type);
      __builtin_unreachable();
  }
}

RecSampler::DefaultSamplingStrategy::DefaultSamplingStrategy(
    const Sampler& sampler)
    : sampler_(sampler) {}

SampleOutput RecSampler::DefaultSamplingStrategy::forward(
    torch::Tensor& logits,
    const SamplingParameters& params) const {
  return sampler_.forward(logits, params);
}

RecSampler::MultiRoundFastPathSamplingStrategy::
    MultiRoundFastPathSamplingStrategy(const Sampler& sampler,
                                       bool enable_fast_path)
    : sampler_(sampler), enable_fast_path_(enable_fast_path) {}

SampleOutput RecSampler::MultiRoundFastPathSamplingStrategy::forward(
    torch::Tensor& logits,
    const SamplingParameters& params) const {
  const bool use_fast_path = can_use_fast_path(params, enable_fast_path_);

  if (!use_fast_path) {
    return sampler_.forward(logits, params);
  }

  LOG_FIRST_N(INFO, 1) << "RecSampler fast path activated.";

  SampleOutput output;

  if (params.frequency_penalties.defined()) {
    apply_frequency_presence_penalties(logits,
                                       params.unique_token_ids,
                                       params.unique_token_counts,
                                       params.frequency_penalties,
                                       params.presence_penalties);
  }

  if (params.repetition_penalties.defined()) {
    apply_repetition_penalties(
        logits, params.unique_token_ids, params.repetition_penalties);
  }

  torch::Tensor sample_logits = logits;
  if (params.selected_token_idxes.numel() != params.sample_idxes.numel()) {
    sample_logits = logits.index_select(/*dim=*/0, params.sample_idxes);
  }

  CHECK_EQ(sample_logits.size(0), params.do_sample.size(0));

  auto [topk_values, topk_indices] =
      sample_logits.topk(params.max_top_logprobs,
                         /*dim=*/-1,
                         /*largest=*/true,
                         /*sorted=*/FLAGS_enable_topk_sorted);
  output.top_tokens = (topk_indices.scalar_type() == torch::kLong)
                          ? topk_indices
                          : topk_indices.to(torch::kLong);

  torch::Tensor temperatures;
  if (params.temperatures.defined()) {
    temperatures = params.temperatures;
    if (params.selected_token_idxes.numel() != params.sample_idxes.numel()) {
      temperatures = temperatures.index_select(/*dim=*/0, params.sample_idxes);
    }
    temperatures = temperatures.to(torch::kFloat32);
  }

#if defined(USE_CUDA)
  if (sample_logits.is_cuda()) {
    std::tie(output.next_tokens, output.logprobs, output.top_logprobs) =
        kernel::cuda::rec_topk_postprocess(
            topk_values, output.top_tokens, temperatures);
    return output;
  }
#endif

  output.top_logprobs = log_softmax_last_dim(topk_values, temperatures);
  output.next_tokens = output.top_tokens.index({torch::indexing::Slice(), 0});
  output.logprobs = output.top_logprobs.index({torch::indexing::Slice(), 0});
  return output;
}

}  // namespace xllm

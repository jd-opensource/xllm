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

#pragma once
#include <torch/torch.h>

#include <memory>

#include "sampling_params.h"
#include "util/rec_model_utils.h"

namespace xllm {

class Sampler;

class RecSampler {
 public:
  explicit RecSampler(RecPipelineType pipeline_type, bool enable_fast_path);
  ~RecSampler();

  // logits: [batch_size, vocab_size]
  SampleOutput forward(
      torch::Tensor& logits,
      const SamplingParameters& params,
      const torch::Tensor& filter_mask = torch::Tensor()) const;

  bool fast_path_enabled() const { return enable_fast_path_; }

 private:
  class SamplingStrategy {
   public:
    virtual ~SamplingStrategy() = default;
    virtual SampleOutput forward(torch::Tensor& logits,
                                 const SamplingParameters& params,
                                 const torch::Tensor& filter_mask) const = 0;
  };

  class DefaultSamplingStrategy final : public SamplingStrategy {
   public:
    explicit DefaultSamplingStrategy(const Sampler& sampler);
    SampleOutput forward(torch::Tensor& logits,
                         const SamplingParameters& params,
                         const torch::Tensor& filter_mask) const override;

   private:
    const Sampler& sampler_;
  };

  class OneRecConstrainedSamplingStrategy final : public SamplingStrategy {
   public:
    explicit OneRecConstrainedSamplingStrategy(const Sampler& sampler);
    SampleOutput forward(torch::Tensor& logits,
                         const SamplingParameters& params,
                         const torch::Tensor& filter_mask) const override;

   private:
    const Sampler& sampler_;
  };

  class MultiRoundFastPathSamplingStrategy final : public SamplingStrategy {
   public:
    explicit MultiRoundFastPathSamplingStrategy(const Sampler& sampler,
                                                bool enable_fast_path);
    SampleOutput forward(torch::Tensor& logits,
                         const SamplingParameters& params,
                         const torch::Tensor& filter_mask) const override;

   private:
    const Sampler& sampler_;
    bool enable_fast_path_;
  };

  static std::unique_ptr<SamplingStrategy> create_sampling_strategy(
      RecPipelineType type,
      const Sampler& sampler,
      bool enable_fast_path);

  std::unique_ptr<Sampler> sampler_;
  std::unique_ptr<SamplingStrategy> strategy_;
  bool enable_fast_path_;
};

}  // namespace xllm

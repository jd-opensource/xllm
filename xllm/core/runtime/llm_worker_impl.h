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

#pragma once

#include <folly/futures/Future.h>
#include <torch/torch.h>

#include "executor.h"
#include "forward_params.h"
#include "framework/model/causal_lm.h"
#include "framework/model/embedding_lm.h"
#include "framework/model/model_args.h"
#include "framework/model/model_input_params.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"
#include "options.h"
#include "runtime/worker_impl.h"

namespace xllm {

class LLMWorkerImpl : public WorkerImpl {
 public:
  LLMWorkerImpl(const ParallelArgs& parallel_args,
                const torch::Device& device,
                const runtime::Options& options);

  ~LLMWorkerImpl() override = default;

  // initialize model, cache manager. blocking call
  bool init_model(ModelContext& context) override;

  std::optional<ForwardOutput> step(const ForwardInput& input) override;

  layer::LmHead get_lm_head() { return model_->get_lm_head(); };

  void set_lm_head(layer::LmHead& head) { model_->set_lm_head(head); };

  layer::WordEmbedding get_word_embedding() {
    return model_->get_word_embedding();
  };

  void set_word_embedding(layer::WordEmbedding& embedding) {
    model_->set_word_embedding(embedding);
  };

 protected:
  std::unique_ptr<BeamSearcher> beam_searcher_;
};

}  // namespace xllm

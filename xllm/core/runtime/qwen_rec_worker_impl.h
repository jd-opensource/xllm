/* Copyright 2025 The xLLM Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://github.com/jd-opensource/xllm/blob/main/LICENSE
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ============================================================================*/

#pragma once

#include "llm_worker_impl.h"

namespace xllm {

// Q wenRecWorkerImpl currently reuses the implementation of LLMWorkerImpl
// (including multi-round decode controlled by FLAGS_max_decode_rounds and
// the ACL graph execution path). It provides a dedicated WorkerImpl type
// for REC/Qwen scenarios so that Rec-specific behaviors can be customized
// here without affecting the generic LLM path.
class QwenRecWorkerImpl : public LLMWorkerImpl {
 public:
  QwenRecWorkerImpl(const ParallelArgs& parallel_args,
                    const torch::Device& device,
                    const runtime::Options& options);

  ~QwenRecWorkerImpl() override = default;

  // Override step: enable multi-round decode in REC scenarios when
  // FLAGS_max_decode_rounds > 0, otherwise fall back to LLMWorkerImpl's
  // single-round behavior.
  std::optional<ForwardOutput> step(const ForwardInput& input) override;

 private:
  std::optional<ForwardOutput> step_multi_round(ForwardInput input);
};

}  // namespace xllm

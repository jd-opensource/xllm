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

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "processors/multimodal_input_processor.h"
#include "processors/prompt_processor.h"

namespace xllm {

class MultimodalProcessor {
 public:
  MultimodalProcessor(std::unique_ptr<MultimodalInputProcessor> mm_processor,
                      std::unique_ptr<PromptProcessor> prompt_processor)
      : mm_processor_(std::move(mm_processor)),
        prompt_processor_(std::move(prompt_processor)) {}

  bool process_mm_input(const MMInput& inputs, MMData& data) {
    return mm_processor_->process(inputs, data);
  }

  void process_prompt(std::string& prompt, const MMData& data) {
    prompt_processor_->process(prompt, data);
  }

  void find_mm_spans(const std::vector<int32_t>& prompt_tokens, MMData& data) {
    prompt_processor_->find_mm_spans(prompt_tokens, data);
  }

 private:
  std::unique_ptr<MultimodalInputProcessor> mm_processor_;
  std::unique_ptr<PromptProcessor> prompt_processor_;
};

}  // namespace xllm

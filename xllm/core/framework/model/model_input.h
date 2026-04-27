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

#include <memory>

#include "model_input_param_groups.h"
#include "model_input_params.h"

namespace xllm {
namespace model_input {

struct ModelInput {
  std::shared_ptr<LLMModelInputParams> llm;
  std::shared_ptr<VLMModelInputParams> vlm;
  std::shared_ptr<DitModelInputParams> dit;
  std::shared_ptr<RecModelInputParams> rec;

  static ModelInput from_legacy(const xllm::ModelInputParams& params);
  void apply_to_legacy(xllm::ModelInputParams* params) const;

  bool has_llm() const { return llm != nullptr; }
  bool has_vlm() const { return vlm != nullptr; }
  bool has_dit() const { return dit != nullptr; }
  bool has_rec() const { return rec != nullptr; }
};

}  // namespace model_input
}  // namespace xllm

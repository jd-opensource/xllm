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

#include "model_input.h"

namespace xllm {

class CausalLM;
class CausalVLM;
class DiTModel;

namespace model_input {

class ModelInputFactory {
 public:
  // Generic conversion that keeps all available partitions.
  static ModelInput create_all(const xllm::ModelInputParams& params);

  // Build input payloads by model family.
  static ModelInput create_for_llm(const CausalLM& model,
                                   const xllm::ModelInputParams& params);
  static ModelInput create_for_vlm(const CausalVLM& model,
                                   const xllm::ModelInputParams& params);
  static ModelInput create_for_dit(const DiTModel& model,
                                   const xllm::ModelInputParams& params);
  static ModelInput create_for_rec(const CausalLM& model,
                                   const xllm::ModelInputParams& params);

  static void apply_to_legacy(const ModelInput& input,
                              xllm::ModelInputParams* params);
};

}  // namespace model_input
}  // namespace xllm

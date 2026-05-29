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

#include "model_input_factory.h"

#include "causal_lm.h"
#include "causal_vlm.h"
#include "dit_model.h"

namespace xllm {
namespace model_input {

ModelInput ModelInputFactory::create_all(const xllm::ModelInputParams& params) {
  return make_model_input_from_legacy(params);
}

ModelInput ModelInputFactory::create_for_llm(
    const CausalLM& model,
    const xllm::ModelInputParams& params) {
  (void)model;
  ModelInputParamBundle bundle =
      make_model_input_param_bundle_from_legacy(params);
  ModelInput input;
  input.llm = bundle.llm;
  return input;
}

ModelInput ModelInputFactory::create_for_vlm(
    const CausalVLM& model,
    const xllm::ModelInputParams& params) {
  (void)model;
  ModelInputParamBundle bundle =
      make_model_input_param_bundle_from_legacy(params);
  ModelInput input;
  input.llm = bundle.llm;
  input.vlm = bundle.vlm;
  return input;
}

ModelInput ModelInputFactory::create_for_dit(
    const DiTModel& model,
    const xllm::ModelInputParams& params) {
  (void)model;
  ModelInputParamBundle bundle =
      make_model_input_param_bundle_from_legacy(params);
  ModelInput input;
  input.dit = bundle.dit;
  return input;
}

ModelInput ModelInputFactory::create_for_rec(
    const CausalLM& model,
    const xllm::ModelInputParams& params) {
  (void)model;
  ModelInputParamBundle bundle =
      make_model_input_param_bundle_from_legacy(params);
  ModelInput input;
  input.llm = bundle.llm;
  input.rec = bundle.rec;
  return input;
}

void ModelInputFactory::apply_to_legacy(const ModelInput& input,
                                        xllm::ModelInputParams* params) {
  apply_model_input_to_legacy(input, params);
}

}  // namespace model_input
}  // namespace xllm

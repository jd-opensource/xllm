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

#include "model_input.h"

namespace xllm {
namespace model_input {

ModelInput ModelInput::from_legacy(const xllm::ModelInputParams& params) {
  ModelInput model_input;
  const ModelInputParamBundle bundle =
      ModelInputParamBundle::from_legacy(params);
  model_input.llm = bundle.llm;
  model_input.vlm = bundle.vlm;
  model_input.dit = bundle.dit;
  model_input.rec = bundle.rec;
  return model_input;
}

void ModelInput::apply_to_legacy(xllm::ModelInputParams* params) const {
  ModelInputParamBundle bundle;
  bundle.llm = llm;
  bundle.vlm = vlm;
  bundle.dit = dit;
  bundle.rec = rec;
  bundle.apply_to_legacy(params);
}

}  // namespace model_input
}  // namespace xllm

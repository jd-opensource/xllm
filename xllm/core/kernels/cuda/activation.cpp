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

#include "cuda_ops_api.h"

namespace xllm::kernel::cuda {

void act_and_mul(torch::Tensor out,
                 torch::Tensor input,
                 const std::string& act_mode) {
  if (act_mode != "silu" && act_mode != "gelu" && act_mode != "gelu_tanh") {
    LOG(FATAL) << "Unsupported act mode: " << act_mode
               << ", only support silu, gelu, gelu_tanh";
  }

  std::string uri = act_mode + "_and_mul";

  auto lib =
      torch::DynamicLibrary(path_to_uri_so_lib(uri).c_str(), nullptr, true);
  std::string schema_name = uri + "::" + uri;

  auto act_and_mul_func =
      torch::Dispatcher::singleton()
          .findSchemaOrThrow(schema_name.c_str(), "")
          .typed<void(torch::Tensor&, torch::Tensor&, bool)>();

  act_and_mul_func.call(out, input, support_pdl());
}
}  // namespace xllm::kernel::cuda
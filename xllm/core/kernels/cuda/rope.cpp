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

void apply_rope_pos_ids_cos_sin_cache(torch::Tensor& q,
                                      torch::Tensor& k,
                                      torch::Tensor& q_rope,
                                      torch::Tensor& k_rope,
                                      torch::Tensor& cos_sin_cache,
                                      torch::Tensor& pos_ids,
                                      bool interleave) {
  get_module("rope")
      ->GetFunction("apply_rope_pos_ids_cos_sin_cache")
      .value()(to_ffi_tensor(q),
               to_ffi_tensor(k),
               to_ffi_tensor(q_rope),
               to_ffi_tensor(k_rope),
               to_ffi_tensor(cos_sin_cache),
               to_ffi_tensor(pos_ids),
               interleave,
               support_pdl());
}

}  // namespace xllm::kernel::cuda
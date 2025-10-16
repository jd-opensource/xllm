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

void apply_rope_pos_ids_cos_sin_cache(torch::Tensor q,
                                      torch::Tensor k,
                                      torch::Tensor cos_sin_cache,
                                      torch::Tensor pos_ids,
                                      bool interleave) {
  const int64_t head_dim = cos_sin_cache.size(-1);
  q = q.view({q.size(0), -1, head_dim});
  k = k.view({k.size(0), -1, head_dim});

  auto lib =
      torch::DynamicLibrary(path_to_uri_so_lib("rope").c_str(), nullptr, true);
  std::string schema_name = "rope::apply_rope_pos_ids_cos_sin_cache";

  auto apply_rope_pos_ids_cos_sin_cache_func =
      torch::Dispatcher::singleton()
          .findSchemaOrThrow(schema_name.c_str(), "")
          .typed<void(torch::Tensor,
                      torch::Tensor,
                      torch::Tensor,
                      torch::Tensor,
                      torch::Tensor,
                      torch::Tensor,
                      bool)>();
  apply_rope_pos_ids_cos_sin_cache_func.call(
      q, k, q, k, cos_sin_cache, pos_ids, interleave);
}

}  // namespace xllm::kernel::cuda
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

#include "hash_util.h"

#include <MurmurHash3.h>

#include "core/common/global_flags.h"

namespace xllm {

Murmur3Key hash_tensor(const torch::Tensor& tensor) {
  Murmur3Key key;
  torch::Tensor contiguous_tensor = tensor.contiguous();
  MurmurHash3_x86_128(
      reinterpret_cast<const void*>(contiguous_tensor.data_ptr()),
      tensor.numel() * tensor.element_size(),
      FLAGS_murmur_hash3_seed,
      key.data);
  return key;
}

Murmur3Key hash_string(const std::string& str) {
  Murmur3Key key;
  MurmurHash3_x86_128(reinterpret_cast<const void*>(str.data()),
                      str.size(),
                      FLAGS_murmur_hash3_seed,
                      key.data);
  return key;
}
}  // namespace xllm
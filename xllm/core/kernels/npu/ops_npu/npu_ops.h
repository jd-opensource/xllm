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
#ifndef XLLM_NPU_OPS_H
#define XLLM_NPU_OPS_H

#include "../custom_functions_npu/atb_common.h"

using namespace std;

namespace atb {

using PagedAttentionParam = atb::infer::PagedAttentionParam;
using ReshapeAndCacheParam = atb::infer::ReshapeAndCacheParam;
using SelfAttentionParam = atb::infer::SelfAttentionParam;

void _npu_paged_attention(const at::Tensor& query,
                          const at::Tensor& key_cache,
                          const at::Tensor& value_cache,
                          int64_t num_kv_heads,
                          int64_t num_heads,
                          double scale_value,
                          const at::Tensor& block_table,
                          const at::Tensor& context_lens,
                          at::Tensor& out);

void _npu_reshape_and_cache(const at::Tensor& key,
                            const at::Tensor& value,
                            at::Tensor& key_cache,
                            at::Tensor& value_cache,
                            const at::Tensor& slot_indices);

void _npu_flash_attention(const at::Tensor& query,
                          const at::Tensor& key,
                          const at::Tensor& value,
                          const at::Tensor& mask,
                          const at::Tensor& seq_len,
                          const double scale_value,
                          const int64_t num_heads,
                          const int64_t num_kv_heads,
                          at::Tensor& out);

}  // namespace atb

#endif  // XLLM_NPU_OPS_H
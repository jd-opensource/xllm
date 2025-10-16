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

#pragma once

#include "param.h"

#if defined(USE_MLU)
#include "mlu/mlu_ops_api.h"
#elif defined(USE_CUDA)
#include "cuda/cuda_ops_api.h"
#endif

namespace xllm {
namespace kernel {

void apply_rotary(const RotaryParams& params);
void active(const ActivationParams& params);
void reshape_paged_cache(const ReshapePagedCacheParams& params);
void prefill_attention(const PrefillAttentionParams& params);
void decode_attention(const DecodeAttentionParams& params);
void fused_layernorm(const FusedLayerNormParams& params);
torch::Tensor matmul(const MatmulParams& params);
torch::Tensor fused_moe(const FusedMoEParams& params);
}  // namespace kernel
}  // namespace xllm

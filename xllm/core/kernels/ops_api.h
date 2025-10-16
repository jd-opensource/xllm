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

namespace xllm::kernel {

static const std::string kActModeSilu = "silu";
static const std::string kActModeGelu = "gelu";
static const std::string kActModeQuickGelu = "quick_gelu";
static const std::string kActModeSwish = "swish";

void apply_rotary(RotaryParams& params);

void active(ActivationParams& params);

void reshape_paged_cache(ReshapePagedCacheParams& params);

void batch_prefill(AttentionParams& params);

void batch_decode(AttentionParams& params);

void fused_layernorm(FusedLayerNormParams& params);

torch::Tensor matmul(MatmulParams& params);

torch::Tensor fused_moe(FusedMoEParams& params);

std::tuple<torch::Tensor, torch::Tensor> scaled_quantize(
    ScaledQuantizeParams& params);

torch::Tensor scaled_matmul(ScaledMatmulParams& params);

torch::Tensor apply_top_k_top_p(TopKPParams& params);

torch::Tensor random_sample(RandomSampleParams& params);

void masked_indexer_select_paged_kv(MaskedIndexerSelectPagedKVParams& params);

}  // namespace xllm::kernel

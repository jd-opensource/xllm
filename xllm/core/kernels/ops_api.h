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

void reshape_from_cache(ReshapeFromCacheParams& params);

void batch_prefill(AttentionParams& params);

void batch_decode(AttentionParams& params);

void fused_layernorm(FusedLayerNormParams& params);

torch::Tensor matmul(MatmulParams& params);

torch::Tensor group_gemm(GroupGemmParams& params);

std::tuple<torch::Tensor, torch::Tensor> moe_active_topk(
    MoeActiveTopkParams& params);

std::vector<torch::Tensor> moe_gen_idx(MoeGenIdxParams& params);

torch::Tensor moe_expand_input(MoeExpandInputParams& params);

torch::Tensor moe_combine_result(MoeCombineResultParams& params);

torch::Tensor moe_all2all_gen_send_layout(
    MoeAll2AllGenSendLayoutParams& params);

std::vector<torch::Tensor> moe_all2all_gen_gather_index(
    MoeAll2AllGenGatherIndexParams& params);

std::vector<torch::Tensor> moe_all2all_create(MoeAll2AllCreateParams& params);

void moe_all2all_init(MoeAll2AllInitParams& params);

void moe_all2all_dispatch(MoeAll2AllDispatchParams& params);

void moe_all2all_combine(MoeAll2AllCombineParams& params);

void moe_all2all_destroy(MoeAll2AllDestroyParams& params);

std::tuple<torch::Tensor, torch::Tensor> scaled_quantize(
    ScaledQuantizeParams& params);

torch::Tensor scaled_matmul(ScaledMatmulParams& params);

torch::Tensor apply_top_k_top_p(TopKPParams& params);

torch::Tensor random_sample(RandomSampleParams& params);

torch::Tensor rejection_sample(RejectionSampleParams& params);

void masked_indexer_select_paged_kv(MaskedIndexerSelectPagedKVParams& params);

void gather_split(GatherSplitParams& params);

void fused_mla_q(FusedMlaQParams& params);

void fused_mla_kv(FusedMlaKVParams& params);

void fused_indexer_q(FusedIndexerQParams& params);

void fused_indexer_k(FusedIndexerKParams& params);

std::vector<torch::Tensor> grouped_matmul(GroupedMatmulParams& params);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> moe_gating_topk_softmax(
    MoeGatingTopkSoftmaxParams& params);

torch::Tensor moe_token_unpermute(MoeTokenUnpermuteParams &params);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> moe_init_routing_v2(
    MoeInitRoutingV2Params &params);
}  // namespace xllm::kernel

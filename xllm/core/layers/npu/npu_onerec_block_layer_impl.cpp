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

#include "npu_onerec_block_layer_impl.h"

#include <glog/logging.h>
#include <mstx/ms_tools_ext.h>

#include <algorithm>
#include <cstring>
#include <set>

#include "common/global_flags.h"
namespace xllm {
namespace layer {
namespace {

// Decoder normal mode: self-attn(29) + cross-attn(28) + layer-norm(4) + mlp(18)
// = 79
static constexpr uint64_t kOneRecWeightCountPerLayer = 79;

// Decoder MoE mode weights count (exclude runtime tensors like expert_array).
static constexpr uint64_t kOneRecMoeWeightCountPerLayer = 97;

enum class OneRecBlockLayerTensorId : int32_t {
  // Self-attention layer norm
  IN_LAYER_NORM_WEIGHT = 0,
  IN_LAYER_NORM_BIAS,
  IN_INPUT_NORM_NEW_WEIGHT,
  IN_INPUT_NORM_NEW_BIAS,
  // Self-attention Q, K, V projections
  IN_Q_WEIGHT,
  IN_Q_BIAS,
  IN_Q_DEQSCALE,
  IN_Q_OFFSET,
  IN_Q_SCALE,
  IN_Q_COMPRESS_IDX,

  IN_K_WEIGHT,
  IN_K_BIAS,
  IN_K_DEQSCALE,
  IN_K_OFFSET,
  IN_K_SCALE,
  IN_K_COMPRESS_IDX,

  IN_V_WEIGHT,
  IN_V_BIAS,
  IN_V_DEQSCALE,
  IN_V_OFFSET,
  IN_V_SCALE,
  IN_V_COMPRESS_IDX,

  // Self-attention output projection
  IN_SELF_ATTN_OUT_WEIGHT,
  IN_SELF_ATTN_OUT_BIAS,
  IN_SELF_ATTN_OUT_DEQSCALE,
  IN_SELF_ATTN_OUT_OFFSET,
  IN_SELF_ATTN_OUT_SCALE,
  IN_SELF_ATTN_OUT_COMPRESS_IDX,

  // ONEREC relative attention bias (encoder only)
  IN_RELATIVE_ATTENTION_BIAS_WEIGHT,

  // Cross-attention layer norm (decoder only)
  IN_CROSS_LAYER_NORM_WEIGHT,
  IN_CROSS_LAYER_NORM_BIAS,
  IN_CROSS_LAYER_NORM_NEW_WEIGHT,
  IN_CROSS_LAYER_NORM_NEW_BIAS,

  // Cross-attention Q, K, V projections (decoder only)
  IN_CROSS_Q_WEIGHT,
  IN_CROSS_Q_BIAS,
  IN_CROSS_Q_DEQSCALE,
  IN_CROSS_Q_OFFSET,
  IN_CROSS_Q_SCALE,
  IN_CROSS_Q_COMPRESS_IDX,

  IN_CROSS_K_WEIGHT,
  IN_CROSS_K_BIAS,
  IN_CROSS_K_DEQSCALE,
  IN_CROSS_K_OFFSET,
  IN_CROSS_K_SCALE,
  IN_CROSS_K_COMPRESS_IDX,

  IN_CROSS_V_WEIGHT,
  IN_CROSS_V_BIAS,
  IN_CROSS_V_DEQSCALE,
  IN_CROSS_V_OFFSET,
  IN_CROSS_V_SCALE,
  IN_CROSS_V_COMPRESS_IDX,

  // Cross-attention output projection (decoder only)
  IN_CROSS_ATTN_OUT_WEIGHT,
  IN_CROSS_ATTN_OUT_BIAS,
  IN_CROSS_ATTN_OUT_DEQSCALE,
  IN_CROSS_ATTN_OUT_OFFSET,
  IN_CROSS_ATTN_OUT_SCALE,
  IN_CROSS_ATTN_OUT_COMPRESS_IDX,

  // Final layer norm
  IN_FINAL_LAYER_NORM_WEIGHT,
  IN_FINAL_LAYER_NORM_BIAS,
  IN_FINAL_LAYER_NORM_NEW_WEIGHT,
  IN_FINAL_LAYER_NORM_NEW_BIAS,

  // Feed-forward network (gated activation)
  IN_FFN_WI_0_WEIGHT = 61,  // wi_0 (gate projection)
  IN_FFN_WI_0_BIAS,
  IN_FFN_WI_0_DEQSCALE,
  IN_FFN_WI_0_OFFSET,
  IN_FFN_WI_0_SCALE,
  IN_FFN_WI_0_COMPRESS_IDX,

  IN_FFN_WI_1_WEIGHT,  // wi_1 (up projection)
  IN_FFN_WI_1_BIAS,
  IN_FFN_WI_1_DEQSCALE,
  IN_FFN_WI_1_OFFSET,
  IN_FFN_WI_1_SCALE,
  IN_FFN_WI_1_COMPRESS_IDX,

  IN_FFN_WO_WEIGHT,  // wo (down projection)
  IN_FFN_WO_BIAS,
  IN_FFN_WO_DEQSCALE,
  IN_FFN_WO_OFFSET,
  IN_FFN_WO_SCALE,
  IN_FFN_WO_COMPRESS_IDX,
};

constexpr int32_t IN_LAYER_NORM_WEIGHT =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_LAYER_NORM_WEIGHT);
constexpr int32_t IN_LAYER_NORM_BIAS =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_LAYER_NORM_BIAS);
constexpr int32_t IN_INPUT_NORM_NEW_WEIGHT =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_INPUT_NORM_NEW_WEIGHT);
constexpr int32_t IN_INPUT_NORM_NEW_BIAS =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_INPUT_NORM_NEW_BIAS);
constexpr int32_t IN_Q_WEIGHT =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_Q_WEIGHT);
constexpr int32_t IN_Q_BIAS =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_Q_BIAS);
constexpr int32_t IN_Q_DEQSCALE =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_Q_DEQSCALE);
constexpr int32_t IN_Q_OFFSET =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_Q_OFFSET);
constexpr int32_t IN_Q_SCALE =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_Q_SCALE);
constexpr int32_t IN_Q_COMPRESS_IDX =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_Q_COMPRESS_IDX);
constexpr int32_t IN_K_WEIGHT =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_K_WEIGHT);
constexpr int32_t IN_K_BIAS =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_K_BIAS);
constexpr int32_t IN_K_DEQSCALE =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_K_DEQSCALE);
constexpr int32_t IN_K_OFFSET =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_K_OFFSET);
constexpr int32_t IN_K_SCALE =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_K_SCALE);
constexpr int32_t IN_K_COMPRESS_IDX =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_K_COMPRESS_IDX);
constexpr int32_t IN_V_WEIGHT =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_V_WEIGHT);
constexpr int32_t IN_V_BIAS =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_V_BIAS);
constexpr int32_t IN_V_DEQSCALE =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_V_DEQSCALE);
constexpr int32_t IN_V_OFFSET =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_V_OFFSET);
constexpr int32_t IN_V_SCALE =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_V_SCALE);
constexpr int32_t IN_V_COMPRESS_IDX =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_V_COMPRESS_IDX);
constexpr int32_t IN_SELF_ATTN_OUT_WEIGHT =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_SELF_ATTN_OUT_WEIGHT);
constexpr int32_t IN_SELF_ATTN_OUT_BIAS =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_SELF_ATTN_OUT_BIAS);
constexpr int32_t IN_SELF_ATTN_OUT_DEQSCALE =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_SELF_ATTN_OUT_DEQSCALE);
constexpr int32_t IN_SELF_ATTN_OUT_OFFSET =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_SELF_ATTN_OUT_OFFSET);
constexpr int32_t IN_SELF_ATTN_OUT_SCALE =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_SELF_ATTN_OUT_SCALE);
constexpr int32_t IN_SELF_ATTN_OUT_COMPRESS_IDX = static_cast<int32_t>(
    OneRecBlockLayerTensorId::IN_SELF_ATTN_OUT_COMPRESS_IDX);
constexpr int32_t IN_RELATIVE_ATTENTION_BIAS_WEIGHT = static_cast<int32_t>(
    OneRecBlockLayerTensorId::IN_RELATIVE_ATTENTION_BIAS_WEIGHT);
constexpr int32_t IN_CROSS_LAYER_NORM_WEIGHT =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_CROSS_LAYER_NORM_WEIGHT);
constexpr int32_t IN_CROSS_LAYER_NORM_BIAS =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_CROSS_LAYER_NORM_BIAS);
constexpr int32_t IN_CROSS_LAYER_NORM_NEW_WEIGHT = static_cast<int32_t>(
    OneRecBlockLayerTensorId::IN_CROSS_LAYER_NORM_NEW_WEIGHT);
constexpr int32_t IN_CROSS_LAYER_NORM_NEW_BIAS = static_cast<int32_t>(
    OneRecBlockLayerTensorId::IN_CROSS_LAYER_NORM_NEW_BIAS);
constexpr int32_t IN_CROSS_Q_WEIGHT =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_CROSS_Q_WEIGHT);
constexpr int32_t IN_CROSS_Q_BIAS =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_CROSS_Q_BIAS);
constexpr int32_t IN_CROSS_Q_DEQSCALE =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_CROSS_Q_DEQSCALE);
constexpr int32_t IN_CROSS_Q_OFFSET =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_CROSS_Q_OFFSET);
constexpr int32_t IN_CROSS_Q_SCALE =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_CROSS_Q_SCALE);
constexpr int32_t IN_CROSS_Q_COMPRESS_IDX =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_CROSS_Q_COMPRESS_IDX);
constexpr int32_t IN_CROSS_K_WEIGHT =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_CROSS_K_WEIGHT);
constexpr int32_t IN_CROSS_K_BIAS =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_CROSS_K_BIAS);
constexpr int32_t IN_CROSS_K_DEQSCALE =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_CROSS_K_DEQSCALE);
constexpr int32_t IN_CROSS_K_OFFSET =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_CROSS_K_OFFSET);
constexpr int32_t IN_CROSS_K_SCALE =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_CROSS_K_SCALE);
constexpr int32_t IN_CROSS_K_COMPRESS_IDX =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_CROSS_K_COMPRESS_IDX);
constexpr int32_t IN_CROSS_V_WEIGHT =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_CROSS_V_WEIGHT);
constexpr int32_t IN_CROSS_V_BIAS =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_CROSS_V_BIAS);
constexpr int32_t IN_CROSS_V_DEQSCALE =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_CROSS_V_DEQSCALE);
constexpr int32_t IN_CROSS_V_OFFSET =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_CROSS_V_OFFSET);
constexpr int32_t IN_CROSS_V_SCALE =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_CROSS_V_SCALE);
constexpr int32_t IN_CROSS_V_COMPRESS_IDX =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_CROSS_V_COMPRESS_IDX);
constexpr int32_t IN_CROSS_ATTN_OUT_WEIGHT =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_CROSS_ATTN_OUT_WEIGHT);
constexpr int32_t IN_CROSS_ATTN_OUT_BIAS =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_CROSS_ATTN_OUT_BIAS);
constexpr int32_t IN_CROSS_ATTN_OUT_DEQSCALE =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_CROSS_ATTN_OUT_DEQSCALE);
constexpr int32_t IN_CROSS_ATTN_OUT_OFFSET =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_CROSS_ATTN_OUT_OFFSET);
constexpr int32_t IN_CROSS_ATTN_OUT_SCALE =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_CROSS_ATTN_OUT_SCALE);
constexpr int32_t IN_CROSS_ATTN_OUT_COMPRESS_IDX = static_cast<int32_t>(
    OneRecBlockLayerTensorId::IN_CROSS_ATTN_OUT_COMPRESS_IDX);
constexpr int32_t IN_FINAL_LAYER_NORM_WEIGHT =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_FINAL_LAYER_NORM_WEIGHT);
constexpr int32_t IN_FINAL_LAYER_NORM_BIAS =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_FINAL_LAYER_NORM_BIAS);
constexpr int32_t IN_FINAL_LAYER_NORM_NEW_WEIGHT = static_cast<int32_t>(
    OneRecBlockLayerTensorId::IN_FINAL_LAYER_NORM_NEW_WEIGHT);
constexpr int32_t IN_FINAL_LAYER_NORM_NEW_BIAS = static_cast<int32_t>(
    OneRecBlockLayerTensorId::IN_FINAL_LAYER_NORM_NEW_BIAS);
constexpr int32_t IN_FFN_WI_0_WEIGHT =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_FFN_WI_0_WEIGHT);
constexpr int32_t IN_FFN_WI_0_BIAS =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_FFN_WI_0_BIAS);
constexpr int32_t IN_FFN_WI_0_DEQSCALE =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_FFN_WI_0_DEQSCALE);
constexpr int32_t IN_FFN_WI_0_OFFSET =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_FFN_WI_0_OFFSET);
constexpr int32_t IN_FFN_WI_0_SCALE =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_FFN_WI_0_SCALE);
constexpr int32_t IN_FFN_WI_0_COMPRESS_IDX =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_FFN_WI_0_COMPRESS_IDX);
constexpr int32_t IN_FFN_WI_1_WEIGHT =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_FFN_WI_1_WEIGHT);
constexpr int32_t IN_FFN_WI_1_BIAS =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_FFN_WI_1_BIAS);
constexpr int32_t IN_FFN_WI_1_DEQSCALE =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_FFN_WI_1_DEQSCALE);
constexpr int32_t IN_FFN_WI_1_OFFSET =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_FFN_WI_1_OFFSET);
constexpr int32_t IN_FFN_WI_1_SCALE =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_FFN_WI_1_SCALE);
constexpr int32_t IN_FFN_WI_1_COMPRESS_IDX =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_FFN_WI_1_COMPRESS_IDX);
constexpr int32_t IN_FFN_WO_WEIGHT =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_FFN_WO_WEIGHT);
constexpr int32_t IN_FFN_WO_BIAS =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_FFN_WO_BIAS);
constexpr int32_t IN_FFN_WO_DEQSCALE =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_FFN_WO_DEQSCALE);
constexpr int32_t IN_FFN_WO_OFFSET =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_FFN_WO_OFFSET);
constexpr int32_t IN_FFN_WO_SCALE =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_FFN_WO_SCALE);
constexpr int32_t IN_FFN_WO_COMPRESS_IDX =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_FFN_WO_COMPRESS_IDX);

enum class OneRecMoeBlockLayerTensorId : int32_t {
  // MoE weights (only used when use_moe=true)
  IN_BLOCK_SPARSE_MOE_GATE_WEIGHT = 61,   // routing weights
  IN_BLOCK_SPARSE_MOE_GATE_BIAS = 62,     // routing bias
  IN_BLOCK_SPARSE_MOE_GATE_DESCALE,       // gate descale
  IN_BLOCK_SPARSE_MOE_GATE_OFFSET,        // gate offset
  IN_BLOCK_SPARSE_MOE_GATE_SCALE,         // gate scale
  IN_BLOCK_SPARSE_MOE_GATE_COMPRESS_IDX,  // gate compress index

  // Shared expert weights
  IN_MLP_GATEUP_WEIGHT_SHARED_EXPERT,
  IN_MLP_GATEUP_BIAS_SHARED_EXPERT,
  IN_MLP_GATEUP_DESCALE_SHARED_EXPERT,
  IN_MLP_GATEUP_OFFSET_SHARED_EXPERT,
  IN_MLP_GATEUP_SCALE_SHARED_EXPERT,
  IN_MLP_GATEUP_COMPRESS_IDX_SHARED_EXPERT,

  IN_MLP_DOWN_WEIGHT_SHARED_EXPERT,
  IN_MLP_DOWN_BIAS_SHARED_EXPERT,
  IN_MLP_DOWN_DESCALE_SHARED_EXPERT,
  IN_MLP_DOWN_OFFSET_SHARED_EXPERT,
  IN_MLP_DOWN_SCALE_SHARED_EXPERT,
  IN_MLP_DOWN_COMPRESS_IDX_SHARED_EXPERT,

  // Shared expert gate weights
  IN_SHARED_EXPERT_GATE_WEIGHT,
  IN_SHARED_EXPERT_GATE_BIAS,
  IN_SHARED_EXPERT_GATE_DESCALE,
  IN_SHARED_EXPERT_GATE_OFFSET,
  IN_SHARED_EXPERT_GATE_SCALE,
  IN_SHARED_EXPERT_GATE_COMPRESS_IDX,

  // Expert weights
  IN_MLP_GATEUP_WEIGHT_EXPERT,
  IN_MLP_GATEUP_BIAS_EXPERT,
  IN_MLP_GATEUP_DESCALE_EXPERT,
  IN_MLP_GATEUP_OFFSET_EXPERT,
  IN_MLP_GATEUP_SCALE_EXPERT,
  IN_MLP_GATEUP_COMPRESS_IDX_EXPERT,

  IN_MLP_DOWN_WEIGHT_EXPERT,
  IN_MLP_DOWN_BIAS_EXPERT,
  IN_MLP_DOWN_DESCALE_EXPERT,
  IN_MLP_DOWN_OFFSET_EXPERT,
  IN_MLP_DOWN_SCALE_EXPERT,
  IN_MLP_DOWN_COMPRESS_IDX_EXPERT = 96,

  // Runtime tensors (not part of weight tensor array)
  IN_EXPERT_ARRAY = 97,
  IN_EXPERT_GROUP = 98,
  IN_ONE_HOT = 99,
  IN_ZERO_HOT = 100,

  // Legacy aliases for backward compatibility
  IN_MOE_EXPERT_W1_WEIGHT = IN_MLP_GATEUP_WEIGHT_EXPERT,
  IN_MOE_EXPERT_W2_WEIGHT = IN_MLP_DOWN_WEIGHT_EXPERT,
  IN_MOE_EXPERT_W3_WEIGHT = IN_MLP_GATEUP_WEIGHT_EXPERT,
  IN_MOE_SHARED_W1_WEIGHT = IN_MLP_GATEUP_WEIGHT_SHARED_EXPERT,
  IN_MOE_SHARED_W2_WEIGHT = IN_MLP_DOWN_WEIGHT_SHARED_EXPERT,
};

constexpr int32_t IN_BLOCK_SPARSE_MOE_GATE_WEIGHT = static_cast<int32_t>(
    OneRecMoeBlockLayerTensorId::IN_BLOCK_SPARSE_MOE_GATE_WEIGHT);
constexpr int32_t IN_BLOCK_SPARSE_MOE_GATE_BIAS = static_cast<int32_t>(
    OneRecMoeBlockLayerTensorId::IN_BLOCK_SPARSE_MOE_GATE_BIAS);
constexpr int32_t IN_BLOCK_SPARSE_MOE_GATE_DESCALE = static_cast<int32_t>(
    OneRecMoeBlockLayerTensorId::IN_BLOCK_SPARSE_MOE_GATE_DESCALE);
constexpr int32_t IN_BLOCK_SPARSE_MOE_GATE_OFFSET = static_cast<int32_t>(
    OneRecMoeBlockLayerTensorId::IN_BLOCK_SPARSE_MOE_GATE_OFFSET);
constexpr int32_t IN_BLOCK_SPARSE_MOE_GATE_SCALE = static_cast<int32_t>(
    OneRecMoeBlockLayerTensorId::IN_BLOCK_SPARSE_MOE_GATE_SCALE);
constexpr int32_t IN_BLOCK_SPARSE_MOE_GATE_COMPRESS_IDX = static_cast<int32_t>(
    OneRecMoeBlockLayerTensorId::IN_BLOCK_SPARSE_MOE_GATE_COMPRESS_IDX);
constexpr int32_t IN_MLP_GATEUP_WEIGHT_SHARED_EXPERT = static_cast<int32_t>(
    OneRecMoeBlockLayerTensorId::IN_MLP_GATEUP_WEIGHT_SHARED_EXPERT);
constexpr int32_t IN_MLP_GATEUP_BIAS_SHARED_EXPERT = static_cast<int32_t>(
    OneRecMoeBlockLayerTensorId::IN_MLP_GATEUP_BIAS_SHARED_EXPERT);
constexpr int32_t IN_MLP_GATEUP_DESCALE_SHARED_EXPERT = static_cast<int32_t>(
    OneRecMoeBlockLayerTensorId::IN_MLP_GATEUP_DESCALE_SHARED_EXPERT);
constexpr int32_t IN_MLP_GATEUP_OFFSET_SHARED_EXPERT = static_cast<int32_t>(
    OneRecMoeBlockLayerTensorId::IN_MLP_GATEUP_OFFSET_SHARED_EXPERT);
constexpr int32_t IN_MLP_GATEUP_SCALE_SHARED_EXPERT = static_cast<int32_t>(
    OneRecMoeBlockLayerTensorId::IN_MLP_GATEUP_SCALE_SHARED_EXPERT);
constexpr int32_t IN_MLP_GATEUP_COMPRESS_IDX_SHARED_EXPERT =
    static_cast<int32_t>(
        OneRecMoeBlockLayerTensorId::IN_MLP_GATEUP_COMPRESS_IDX_SHARED_EXPERT);
constexpr int32_t IN_MLP_DOWN_WEIGHT_SHARED_EXPERT = static_cast<int32_t>(
    OneRecMoeBlockLayerTensorId::IN_MLP_DOWN_WEIGHT_SHARED_EXPERT);
constexpr int32_t IN_MLP_DOWN_BIAS_SHARED_EXPERT = static_cast<int32_t>(
    OneRecMoeBlockLayerTensorId::IN_MLP_DOWN_BIAS_SHARED_EXPERT);
constexpr int32_t IN_MLP_DOWN_DESCALE_SHARED_EXPERT = static_cast<int32_t>(
    OneRecMoeBlockLayerTensorId::IN_MLP_DOWN_DESCALE_SHARED_EXPERT);
constexpr int32_t IN_MLP_DOWN_OFFSET_SHARED_EXPERT = static_cast<int32_t>(
    OneRecMoeBlockLayerTensorId::IN_MLP_DOWN_OFFSET_SHARED_EXPERT);
constexpr int32_t IN_MLP_DOWN_SCALE_SHARED_EXPERT = static_cast<int32_t>(
    OneRecMoeBlockLayerTensorId::IN_MLP_DOWN_SCALE_SHARED_EXPERT);
constexpr int32_t IN_MLP_DOWN_COMPRESS_IDX_SHARED_EXPERT = static_cast<int32_t>(
    OneRecMoeBlockLayerTensorId::IN_MLP_DOWN_COMPRESS_IDX_SHARED_EXPERT);
constexpr int32_t IN_SHARED_EXPERT_GATE_WEIGHT = static_cast<int32_t>(
    OneRecMoeBlockLayerTensorId::IN_SHARED_EXPERT_GATE_WEIGHT);
constexpr int32_t IN_SHARED_EXPERT_GATE_BIAS = static_cast<int32_t>(
    OneRecMoeBlockLayerTensorId::IN_SHARED_EXPERT_GATE_BIAS);
constexpr int32_t IN_SHARED_EXPERT_GATE_DESCALE = static_cast<int32_t>(
    OneRecMoeBlockLayerTensorId::IN_SHARED_EXPERT_GATE_DESCALE);
constexpr int32_t IN_SHARED_EXPERT_GATE_OFFSET = static_cast<int32_t>(
    OneRecMoeBlockLayerTensorId::IN_SHARED_EXPERT_GATE_OFFSET);
constexpr int32_t IN_SHARED_EXPERT_GATE_SCALE = static_cast<int32_t>(
    OneRecMoeBlockLayerTensorId::IN_SHARED_EXPERT_GATE_SCALE);
constexpr int32_t IN_SHARED_EXPERT_GATE_COMPRESS_IDX = static_cast<int32_t>(
    OneRecMoeBlockLayerTensorId::IN_SHARED_EXPERT_GATE_COMPRESS_IDX);
constexpr int32_t IN_MLP_GATEUP_WEIGHT_EXPERT = static_cast<int32_t>(
    OneRecMoeBlockLayerTensorId::IN_MLP_GATEUP_WEIGHT_EXPERT);
constexpr int32_t IN_MLP_GATEUP_BIAS_EXPERT = static_cast<int32_t>(
    OneRecMoeBlockLayerTensorId::IN_MLP_GATEUP_BIAS_EXPERT);
constexpr int32_t IN_MLP_GATEUP_DESCALE_EXPERT = static_cast<int32_t>(
    OneRecMoeBlockLayerTensorId::IN_MLP_GATEUP_DESCALE_EXPERT);
constexpr int32_t IN_MLP_GATEUP_OFFSET_EXPERT = static_cast<int32_t>(
    OneRecMoeBlockLayerTensorId::IN_MLP_GATEUP_OFFSET_EXPERT);
constexpr int32_t IN_MLP_GATEUP_SCALE_EXPERT = static_cast<int32_t>(
    OneRecMoeBlockLayerTensorId::IN_MLP_GATEUP_SCALE_EXPERT);
constexpr int32_t IN_MLP_GATEUP_COMPRESS_IDX_EXPERT = static_cast<int32_t>(
    OneRecMoeBlockLayerTensorId::IN_MLP_GATEUP_COMPRESS_IDX_EXPERT);
constexpr int32_t IN_MLP_DOWN_WEIGHT_EXPERT = static_cast<int32_t>(
    OneRecMoeBlockLayerTensorId::IN_MLP_DOWN_WEIGHT_EXPERT);
constexpr int32_t IN_MLP_DOWN_BIAS_EXPERT =
    static_cast<int32_t>(OneRecMoeBlockLayerTensorId::IN_MLP_DOWN_BIAS_EXPERT);
constexpr int32_t IN_MLP_DOWN_DESCALE_EXPERT = static_cast<int32_t>(
    OneRecMoeBlockLayerTensorId::IN_MLP_DOWN_DESCALE_EXPERT);
constexpr int32_t IN_MLP_DOWN_OFFSET_EXPERT = static_cast<int32_t>(
    OneRecMoeBlockLayerTensorId::IN_MLP_DOWN_OFFSET_EXPERT);
constexpr int32_t IN_MLP_DOWN_SCALE_EXPERT =
    static_cast<int32_t>(OneRecMoeBlockLayerTensorId::IN_MLP_DOWN_SCALE_EXPERT);
constexpr int32_t IN_MLP_DOWN_COMPRESS_IDX_EXPERT = static_cast<int32_t>(
    OneRecMoeBlockLayerTensorId::IN_MLP_DOWN_COMPRESS_IDX_EXPERT);
constexpr int32_t IN_EXPERT_ARRAY =
    static_cast<int32_t>(OneRecMoeBlockLayerTensorId::IN_EXPERT_ARRAY);
constexpr int32_t IN_EXPERT_GROUP =
    static_cast<int32_t>(OneRecMoeBlockLayerTensorId::IN_EXPERT_GROUP);
constexpr int32_t IN_ONE_HOT =
    static_cast<int32_t>(OneRecMoeBlockLayerTensorId::IN_ONE_HOT);
constexpr int32_t IN_ZERO_HOT =
    static_cast<int32_t>(OneRecMoeBlockLayerTensorId::IN_ZERO_HOT);
constexpr int32_t IN_MOE_EXPERT_W1_WEIGHT =
    static_cast<int32_t>(OneRecMoeBlockLayerTensorId::IN_MOE_EXPERT_W1_WEIGHT);
constexpr int32_t IN_MOE_EXPERT_W2_WEIGHT =
    static_cast<int32_t>(OneRecMoeBlockLayerTensorId::IN_MOE_EXPERT_W2_WEIGHT);
constexpr int32_t IN_MOE_EXPERT_W3_WEIGHT =
    static_cast<int32_t>(OneRecMoeBlockLayerTensorId::IN_MOE_EXPERT_W3_WEIGHT);
constexpr int32_t IN_MOE_SHARED_W1_WEIGHT =
    static_cast<int32_t>(OneRecMoeBlockLayerTensorId::IN_MOE_SHARED_W1_WEIGHT);
constexpr int32_t IN_MOE_SHARED_W2_WEIGHT =
    static_cast<int32_t>(OneRecMoeBlockLayerTensorId::IN_MOE_SHARED_W2_WEIGHT);

static const std::unordered_map<std::string, int32_t>
    kOneRecEncoderWeightMapping = {
        {"layer.0.layer_norm.weight", IN_LAYER_NORM_WEIGHT},
        {"layer.0.SelfAttention.q.weight", IN_Q_WEIGHT},
        {"layer.0.SelfAttention.k.weight", IN_K_WEIGHT},
        {"layer.0.SelfAttention.v.weight", IN_V_WEIGHT},
        {"layer.0.SelfAttention.o.weight", IN_SELF_ATTN_OUT_WEIGHT},
        {"layer.0.SelfAttention.relative_attention_bias.weight",
         IN_RELATIVE_ATTENTION_BIAS_WEIGHT},
        {"layer.1.layer_norm.weight", IN_FINAL_LAYER_NORM_WEIGHT},
        {"layer.1.DenseReluDense.wi.weight", IN_FFN_WI_1_WEIGHT},
        {"layer.1.DenseReluDense.wo.weight", IN_FFN_WO_WEIGHT},
        {"layer.1.DenseReluDense.gate_proj.weight", IN_FFN_WI_0_WEIGHT},
        {"layer.1.ffn.wi.weight", IN_FFN_WI_1_WEIGHT},
        {"layer.1.ffn.wo.weight", IN_FFN_WO_WEIGHT},
        {"layer.1.ffn.gate_proj.weight", IN_FFN_WI_0_WEIGHT},
        // Alternative format
        {"0.layer_norm.weight", IN_LAYER_NORM_WEIGHT},
        {"0.SelfAttention.q.weight", IN_Q_WEIGHT},
        {"0.SelfAttention.k.weight", IN_K_WEIGHT},
        {"0.SelfAttention.v.weight", IN_V_WEIGHT},
        {"0.SelfAttention.o.weight", IN_SELF_ATTN_OUT_WEIGHT},
        {"0.SelfAttention.relative_attention_bias.weight",
         IN_RELATIVE_ATTENTION_BIAS_WEIGHT},
        {"1.layer_norm.weight", IN_FINAL_LAYER_NORM_WEIGHT},
        {"1.DenseReluDense.wi.weight", IN_FFN_WI_1_WEIGHT},
        {"1.DenseReluDense.wo.weight", IN_FFN_WO_WEIGHT},
        {"1.DenseReluDense.gate_proj.weight", IN_FFN_WI_0_WEIGHT},
        {"1.ffn.wi.weight", IN_FFN_WI_1_WEIGHT},
        {"1.ffn.wo.weight", IN_FFN_WO_WEIGHT},
        {"1.ffn.gate_proj.weight", IN_FFN_WI_0_WEIGHT},
};

static const std::unordered_map<std::string, int32_t>
    kOneRecDecoderWeightMapping = {
        {"layer.0.layer_norm.weight", IN_LAYER_NORM_WEIGHT},
        {"layer.0.SelfAttention.q.weight", IN_Q_WEIGHT},
        {"layer.0.SelfAttention.k.weight", IN_K_WEIGHT},
        {"layer.0.SelfAttention.v.weight", IN_V_WEIGHT},
        {"layer.0.SelfAttention.o.weight", IN_SELF_ATTN_OUT_WEIGHT},
        {"layer.0.SelfAttention.relative_attention_bias.weight",
         IN_RELATIVE_ATTENTION_BIAS_WEIGHT},
        {"layer.1.layer_norm.weight", IN_CROSS_LAYER_NORM_WEIGHT},
        {"layer.1.EncDecAttention.q.weight", IN_CROSS_Q_WEIGHT},
        {"layer.1.EncDecAttention.k.weight", IN_CROSS_K_WEIGHT},
        {"layer.1.EncDecAttention.v.weight", IN_CROSS_V_WEIGHT},
        {"layer.1.EncDecAttention.o.weight", IN_CROSS_ATTN_OUT_WEIGHT},
        {"layer.2.layer_norm.weight", IN_FINAL_LAYER_NORM_WEIGHT},
        {"layer.2.DenseReluDense.wi.weight", IN_FFN_WI_1_WEIGHT},
        {"layer.2.DenseReluDense.wo.weight", IN_FFN_WO_WEIGHT},
        {"layer.2.DenseReluDense.gate_proj.weight", IN_FFN_WI_0_WEIGHT},
        // Alternative format
        {"0.layer_norm.weight", IN_LAYER_NORM_WEIGHT},
        {"0.SelfAttention.q.weight", IN_Q_WEIGHT},
        {"0.SelfAttention.k.weight", IN_K_WEIGHT},
        {"0.SelfAttention.v.weight", IN_V_WEIGHT},
        {"0.SelfAttention.o.weight", IN_SELF_ATTN_OUT_WEIGHT},
        {"0.SelfAttention.relative_attention_bias.weight",
         IN_RELATIVE_ATTENTION_BIAS_WEIGHT},
        {"1.layer_norm.weight", IN_CROSS_LAYER_NORM_WEIGHT},
        {"1.EncDecAttention.q.weight", IN_CROSS_Q_WEIGHT},
        {"1.EncDecAttention.k.weight", IN_CROSS_K_WEIGHT},
        {"1.EncDecAttention.v.weight", IN_CROSS_V_WEIGHT},
        {"1.EncDecAttention.o.weight", IN_CROSS_ATTN_OUT_WEIGHT},
        {"2.layer_norm.weight", IN_FINAL_LAYER_NORM_WEIGHT},
        {"2.DenseReluDense.wi.weight", IN_FFN_WI_1_WEIGHT},
        {"2.DenseReluDense.wo.weight", IN_FFN_WO_WEIGHT},
        {"2.DenseReluDense.gate_proj.weight", IN_FFN_WI_0_WEIGHT},
        {"2.ffn.wi.weight", IN_FFN_WI_1_WEIGHT},
        {"2.ffn.wo.weight", IN_FFN_WO_WEIGHT},
        {"2.ffn.gate_proj.weight", IN_FFN_WI_0_WEIGHT},
};

static std::unordered_map<std::string, int>
get_onerec_decoder_moe_weight_mapping() {
  std::unordered_map<std::string, int> mapping = kOneRecDecoderWeightMapping;

  mapping.emplace("layer.2.ffn.gate.weight", IN_BLOCK_SPARSE_MOE_GATE_WEIGHT);
  mapping.emplace("2.ffn.gate.weight", IN_BLOCK_SPARSE_MOE_GATE_WEIGHT);

  mapping.emplace("layer.2.ffn.shared_experts.w1.weight",
                  IN_MLP_GATEUP_WEIGHT_SHARED_EXPERT);
  mapping.emplace("layer.2.ffn.shared_experts.w3.weight",
                  IN_MLP_GATEUP_WEIGHT_SHARED_EXPERT);
  mapping.emplace("layer.2.ffn.shared_experts.w2.weight",
                  IN_MLP_DOWN_WEIGHT_SHARED_EXPERT);

  mapping.emplace("layer.2.ffn.shared_expert.gate.weight",
                  IN_SHARED_EXPERT_GATE_WEIGHT);
  mapping.emplace("layer.2.ffn.shared_expert.gate.bias",
                  IN_SHARED_EXPERT_GATE_BIAS);
  mapping.emplace("layer.2.ffn.shared_expert.gate.weight_scale",
                  IN_SHARED_EXPERT_GATE_SCALE);
  mapping.emplace("layer.2.ffn.shared_expert.gate.weight_offset",
                  IN_SHARED_EXPERT_GATE_OFFSET);

  // Expert weights are handled by
  // process_expert_weights()/merge_experts_weights to avoid ambiguous suffix
  // matching and keep deterministic loading.

  return mapping;
}

static const std::unordered_map<std::string, int>
    kOneRecDecoderMoeWeightMapping = get_onerec_decoder_moe_weight_mapping();

static const std::unordered_map<int32_t, int32_t> kOneRecWeightShard = {
    {IN_Q_WEIGHT, 0},
    {IN_K_WEIGHT, 0},
    {IN_V_WEIGHT, 0},
    {IN_SELF_ATTN_OUT_WEIGHT, 1},
    {IN_CROSS_Q_WEIGHT, 0},
    {IN_CROSS_K_WEIGHT, 0},
    {IN_CROSS_V_WEIGHT, 0},
    {IN_CROSS_ATTN_OUT_WEIGHT, 1},
    {IN_FFN_WI_0_WEIGHT, 0},
    {IN_FFN_WI_1_WEIGHT, 0},
    {IN_FFN_WO_WEIGHT, 1},
    // MoE
    {IN_BLOCK_SPARSE_MOE_GATE_WEIGHT, 0},
    {IN_MLP_GATEUP_WEIGHT_EXPERT, 0},
    {IN_MLP_DOWN_WEIGHT_EXPERT, 1},
    // Shared experts
    {IN_MLP_GATEUP_WEIGHT_SHARED_EXPERT, 0},
    {IN_MLP_GATEUP_OFFSET_SHARED_EXPERT, 0},
    {IN_MLP_GATEUP_SCALE_SHARED_EXPERT, 0},
    {IN_MLP_DOWN_WEIGHT_SHARED_EXPERT, 1},
    {IN_MLP_DOWN_OFFSET_SHARED_EXPERT, 1},
    {IN_MLP_DOWN_SCALE_SHARED_EXPERT, 1},
    {IN_SHARED_EXPERT_GATE_WEIGHT, 0},
    {IN_SHARED_EXPERT_GATE_BIAS, 0},
    {IN_SHARED_EXPERT_GATE_SCALE, 0},
    {IN_SHARED_EXPERT_GATE_OFFSET, 0},
};

}  // namespace

NpuOneRecBlockLayerImpl::NpuOneRecBlockLayerImpl(const ModelContext& context,
                                                 bool is_decoder,
                                                 int32_t layer_id)
    : BaseLayer(context), is_decoder_(is_decoder), layer_id_(layer_id) {
  const auto& args = context.get_model_args();
  const auto& parallel_args = context.get_parallel_args();
  param_from_args(prefill_param_, args, parallel_args, /*is_prefill=*/true);
  param_from_args(decode_param_, args, parallel_args, /*is_prefill=*/false);

  const int weight_count = prefill_param_.use_moe
                               ? kOneRecMoeWeightCountPerLayer
                               : kOneRecWeightCountPerLayer;
  at_weight_tensors_.resize(weight_count);
  atb_weight_tensors_.resize(weight_count);

  placeholder_vec_ = {1, 1};
  dtype_ = c10::typeMetaToScalarType(context.get_tensor_options().dtype());
  device_id_ = context.get_tensor_options().device().index();

  auto placeholder_tensor = torch::empty({1, 1}, torch::kInt32).to(device_);
  placeholder_ = atb_speed::Utils::AtTensor2Tensor(placeholder_tensor);
  at_placeholder_ = torch::empty({1, args.hidden_size()}, dtype_).to(device_);

  for (int i = 0; i < weight_count; ++i) {
    at_weight_tensors_[i] =
        torch::zeros({1, args.hidden_size()}).to(context.get_tensor_options());
  }

  if (prefill_param_.use_moe) {
    auto device = context.get_tensor_options().device();
    one_hot_ = torch::tensor({1}, torch::kInt32).to(device);
    zero_hot_ = torch::tensor({0}, torch::kInt32).to(device);
    expert_group_ = torch::tensor({1}, torch::dtype(torch::kInt32)).to(device);
  }
}

void NpuOneRecBlockLayerImpl::param_from_args(
    atb_speed::onerec::BlockLayerParam& param,
    const ModelArgs& args,
    const ParallelArgs& parallel_args,
    bool is_prefill,
    const ModelInputParams* input_params) {
  (void)input_params;

  param.isFA = false;
  param.isPrefill = is_prefill;
  param.isBF16 = args.dtype() == "bfloat16";
  param.isPack = true;
  param.supportSwiGLU = true;
  param.supportLcoc = is_prefill;
  param.supportSpeculate = false;
  param.enableSplitFuse = FLAGS_enable_chunked_prefill && is_prefill;
  param.supportLora = false;
  param.loraEnableGMM = false;
  param.enableLogN = false;
  param.kvQuant = false;
  param.enableIntraLayerAddNorm = false;
  param.enableInterLayerAddNorm = false;
  param.isDecoder = is_decoder_;
  param.isOneRecEncoder = !is_decoder_;
  param.enableOneRecPrefillOnly = FLAGS_enable_rec_prefill_only;
  param.backend = FLAGS_communication_backend;
  param.rank = parallel_args.rank();
  param.worldSize = parallel_args.world_size();
  param.quantType = 0;
  param.quantGroupSize = 64;

  const int64_t args_n_heads =
      is_decoder_ ? args.decoder_n_heads() : args.n_heads();
  const int64_t args_head_dim =
      is_decoder_ ? args.decoder_head_dim() : args.head_dim();
  param.numAttentionHeadsPerRank = args_n_heads / param.worldSize;
  param.hiddenSizePerAttentionHead = args_head_dim;

  std::optional<int64_t> optional_value =
      is_decoder_ ? args.decoder_n_kv_heads().value_or(args.decoder_n_heads())
                  : args.n_kv_heads().value_or(args.n_heads());
  param.numKeyValueHeadsPerRank =
      static_cast<int>(optional_value.value()) / param.worldSize;
  param.rmsNormEps = args.rms_norm_eps();

  param.seqLen = {};
  param.tokenOffset = {};
  param.packQuantType = {1, 1};
  param.linearQuantType = {0, -1, -1, 0, 0, -1, 0};
  param.layerId = layer_id_;
  param.linearTransposeType = {1, 1, 1, 1, 1, 1, 1};

  if (param.isBF16) {
    param.linearDescs = {
        static_cast<int>(atb_speed::common::LinearDesc::BFLOAT16_DESC),
        static_cast<int>(atb_speed::common::LinearDesc::BFLOAT16_DESC),
        static_cast<int>(atb_speed::common::LinearDesc::BFLOAT16_DESC),
        static_cast<int>(atb_speed::common::LinearDesc::BFLOAT16_DESC)};
  } else {
    param.linearDescs = {
        static_cast<int>(atb_speed::common::LinearDesc::FLOAT16_DESC),
        static_cast<int>(atb_speed::common::LinearDesc::FLOAT16_DESC),
        static_cast<int>(atb_speed::common::LinearDesc::FLOAT16_DESC),
        static_cast<int>(atb_speed::common::LinearDesc::FLOAT16_DESC)};
  }

  param.use_moe = args.use_moe() && is_decoder_;
  if (param.use_moe) {
    ep_size_ = 1;
    const int32_t ep_rank = 0;
    ep_local_tp_size_ = parallel_args.world_size() / ep_size_;
    CHECK_EQ(parallel_args.world_size(), ep_size_ * ep_local_tp_size_);
    ep_local_tp_rank_ = parallel_args.rank() % ep_local_tp_size_;

    num_experts_per_partition_ = args.n_routed_experts() / ep_size_;
    start_expert_id_ = ep_rank * num_experts_per_partition_;
    end_expert_id_ = start_expert_id_ + num_experts_per_partition_ - 1;

    resize_experts_weights(num_experts_per_partition_);

    param.moe_config = std::make_unique<atb_speed::onerec::OneRecMoEConfig>();
    param.moe_config->moe_topk = args.num_experts_per_tok();
    param.moe_config->moe_num_experts = args.n_routed_experts();
    param.moe_config->moe_score_func = "softmax";
    param.moe_config->moe_route_scale = args.moe_route_scale();
    param.moe_config->moe_inter_dim = args.moe_intermediate_size();
    param.moe_config->use_bf16 = param.isBF16;
    param.moe_config->hasSharedExpertGate = false;
    param.moe_config->moe_use_shared_experts = args.moe_use_shared_experts();
    param.moe_config->moe_num_shared_experts = args.n_shared_experts();

    param.moeLinearQuantType = {atb_speed::common::LinearType::FP,
                                atb_speed::common::LinearType::FP,
                                atb_speed::common::LinearType::INVALID,
                                atb_speed::common::LinearType::FP};
  }
}

void NpuOneRecBlockLayerImpl::verify_loaded_weights(
    const std::string& prefix) const {
  const auto& weight_mapping =
      [this]() -> const std::unordered_map<std::string, int>& {
    if (prefill_param_.use_moe) {
      return kOneRecDecoderMoeWeightMapping;
    }
    return is_decoder_ ? kOneRecDecoderWeightMapping
                       : kOneRecEncoderWeightMapping;
  }();

  // verify_loaded_weights() runs before merge_loaded_weights().
  // Only allow placeholders for tensors that are intentionally absent before
  // merge in the current mode.
  std::set<int> allowed_placeholders;
  if (prefill_param_.use_moe) {
    // MoE decoder path does not consume dense FFN gate/up/down tensors.
    allowed_placeholders.insert(IN_FFN_WI_0_WEIGHT);
    allowed_placeholders.insert(IN_FFN_WI_1_WEIGHT);
    allowed_placeholders.insert(IN_FFN_WO_WEIGHT);
  }
  const bool has_shared_experts =
      prefill_param_.moe_config != nullptr &&
      prefill_param_.moe_config->moe_use_shared_experts;

  for (const auto& [name, index] : weight_mapping) {
    const auto sizes = at_weight_tensors_[index].sizes();
    const bool is_placeholder = (sizes.size() == 2 && sizes[0] == 1);
    const bool expected_placeholder = allowed_placeholders.count(index) > 0;
    const bool is_relative_bias = (index == IN_RELATIVE_ATTENTION_BIAS_WEIGHT);
    const bool is_shared_optional =
        prefill_param_.use_moe && !has_shared_experts &&
        (index == IN_MLP_GATEUP_WEIGHT_SHARED_EXPERT ||
         index == IN_MLP_DOWN_WEIGHT_SHARED_EXPERT ||
         index == IN_SHARED_EXPERT_GATE_WEIGHT ||
         index == IN_SHARED_EXPERT_GATE_BIAS ||
         index == IN_SHARED_EXPERT_GATE_OFFSET ||
         index == IN_SHARED_EXPERT_GATE_SCALE);
    if (is_placeholder && !expected_placeholder && !is_relative_bias &&
        !is_shared_optional) {
      CHECK(false) << "weight is not loaded for " << prefix << name;
    }
  }

  if (prefill_param_.use_moe) {
    CHECK(validate_decoder_moe_weights(prefix))
        << "OneRec MoE expert weights are incomplete for " << prefix;
  }
}

bool NpuOneRecBlockLayerImpl::validate_decoder_moe_weights(
    const std::string& prefix) const {
  const auto gate_it = experts_weights_.find("gate_proj.weight");
  const auto up_it = experts_weights_.find("up_proj.weight");
  const auto down_it = experts_weights_.find("down_proj.weight");
  if (gate_it == experts_weights_.end() || up_it == experts_weights_.end() ||
      down_it == experts_weights_.end()) {
    LOG(ERROR) << "Missing OneRec MoE expert tensors in " << prefix
               << " (layer " << layer_id_
               << ", gate/up/down map entry not found).";
    return false;
  }

  const auto& gate_weights = gate_it->second;
  const auto& up_weights = up_it->second;
  const auto& down_weights = down_it->second;

  if (gate_weights.size() != up_weights.size() ||
      gate_weights.size() != down_weights.size()) {
    LOG(ERROR) << "OneRec MoE expert vector size mismatch in " << prefix
               << ": gate=" << gate_weights.size()
               << ", up=" << up_weights.size()
               << ", down=" << down_weights.size() << ", layer " << layer_id_;
    return false;
  }

  for (size_t i = 0; i < gate_weights.size(); ++i) {
    const bool gate_defined = gate_weights[i].defined();
    const bool up_defined = up_weights[i].defined();
    const bool down_defined = down_weights[i].defined();
    if (gate_defined != up_defined || gate_defined != down_defined) {
      LOG(ERROR) << "OneRec MoE expert tensor mismatch in " << prefix
                 << " at local expert " << i << ": gate=" << gate_defined
                 << ", up=" << up_defined << ", down=" << down_defined
                 << ", layer " << layer_id_;
      return false;
    }
    if (!gate_defined) {
      LOG(ERROR) << "Missing OneRec MoE tensor for local expert " << i << " in "
                 << prefix << " (layer " << layer_id_ << ").";
      return false;
    }
  }
  return true;
}

void NpuOneRecBlockLayerImpl::merge_loaded_weights() {
  const bool q_loaded = !(at_weight_tensors_[IN_Q_WEIGHT].sizes().size() == 2 &&
                          at_weight_tensors_[IN_Q_WEIGHT].sizes()[0] == 1);
  const bool k_loaded = !(at_weight_tensors_[IN_K_WEIGHT].sizes().size() == 2 &&
                          at_weight_tensors_[IN_K_WEIGHT].sizes()[0] == 1);
  const bool v_loaded = !(at_weight_tensors_[IN_V_WEIGHT].sizes().size() == 2 &&
                          at_weight_tensors_[IN_V_WEIGHT].sizes()[0] == 1);
  CHECK(q_loaded && k_loaded && v_loaded)
      << "OneRec QKV weights are not properly loaded.";

  auto new_q_weight = torch::cat({at_weight_tensors_[IN_Q_WEIGHT],
                                  at_weight_tensors_[IN_K_WEIGHT],
                                  at_weight_tensors_[IN_V_WEIGHT]},
                                 0);
  at_weight_tensors_[IN_Q_WEIGHT] = new_q_weight;
  at_weight_tensors_[IN_K_WEIGHT] =
      torch::zeros({1, at_weight_tensors_[IN_Q_WEIGHT].size(1)})
          .to(device_)
          .to(dtype_);
  at_weight_tensors_[IN_V_WEIGHT] =
      torch::zeros({1, at_weight_tensors_[IN_Q_WEIGHT].size(1)})
          .to(device_)
          .to(dtype_);

  // Keep decoder cross-attention Q/K/V unpacked for current OneRec ATB
  // contract. Do not merge IN_CROSS_{Q,K,V}_WEIGHT here.

  if (!prefill_param_.use_moe) {
    const bool wi0_loaded =
        !(at_weight_tensors_[IN_FFN_WI_0_WEIGHT].sizes().size() == 2 &&
          at_weight_tensors_[IN_FFN_WI_0_WEIGHT].sizes()[0] == 1);
    const bool wi1_loaded =
        !(at_weight_tensors_[IN_FFN_WI_1_WEIGHT].sizes().size() == 2 &&
          at_weight_tensors_[IN_FFN_WI_1_WEIGHT].sizes()[0] == 1);
    CHECK(wi0_loaded && wi1_loaded)
        << "OneRec FFN gate/up weights are not properly loaded.";

    auto new_gate_up_weight =
        torch::cat({at_weight_tensors_[IN_FFN_WI_0_WEIGHT],
                    at_weight_tensors_[IN_FFN_WI_1_WEIGHT]},
                   0);
    at_weight_tensors_[IN_FFN_WI_0_WEIGHT] = new_gate_up_weight;
    at_weight_tensors_[IN_FFN_WI_1_WEIGHT] =
        torch::zeros({1, at_weight_tensors_[IN_FFN_WI_0_WEIGHT].size(1)})
            .to(device_)
            .to(dtype_);
  } else {
    merge_experts_weights();
    merge_shared_experts_weights();
  }

  const uint64_t weight_count = prefill_param_.use_moe
                                    ? kOneRecMoeWeightCountPerLayer
                                    : kOneRecWeightCountPerLayer;
  for (int i = 0; i < static_cast<int>(weight_count); ++i) {
    if (!at_weight_tensors_[i].defined()) {
      at_weight_tensors_[i] = torch::zeros(
          {1, 1}, torch::TensorOptions().device(device_).dtype(dtype_));
    }
    if (!at_weight_tensors_[i].is_contiguous()) {
      at_weight_tensors_[i] = at_weight_tensors_[i].contiguous();
    }
  }

  for (int i = 0; i < static_cast<int>(weight_count); ++i) {
    atb_weight_tensors_[i] =
        atb_speed::Utils::AtTensor2Tensor(at_weight_tensors_[i]);
  }

  LOG(INFO) << "OneRec BlockLayer merge_loaded_weights calling init_layer"
            << ", layer_role=" << (is_decoder_ ? "decoder" : "encoder")
            << ", layer_id=" << layer_id_ << ", weight_count=" << weight_count;
  const int64_t init_status = init_layer();
  LOG(INFO) << "OneRec BlockLayer merge_loaded_weights init_layer returned"
            << ", layer_role=" << (is_decoder_ ? "decoder" : "encoder")
            << ", layer_id=" << layer_id_ << ", status=" << init_status;
  CHECK_EQ(init_status, atb::NO_ERROR)
      << "OneRec BlockLayer init_layer failed, layer_role="
      << (is_decoder_ ? "decoder" : "encoder") << ", layer_id=" << layer_id_;
}

void NpuOneRecBlockLayerImpl::load_state_dict(const StateDict& state_dict) {
  const auto target_weight_dtype = [this]() -> torch::ScalarType {
    if (torch_dtype_.empty()) {
      return dtype_;
    }
    if (torch_dtype_ == "float16") {
      return torch::kFloat16;
    }
    if (torch_dtype_ == "bfloat16") {
      return torch::kBFloat16;
    }
    if (torch_dtype_ == "float32") {
      return torch::kFloat32;
    }
    if (torch_dtype_ == "float64") {
      return torch::kFloat64;
    }
    if (torch_dtype_ == "int8") {
      return torch::kInt8;
    }
    if (torch_dtype_ == "int16") {
      return torch::kInt16;
    }
    if (torch_dtype_ == "int32") {
      return torch::kInt32;
    }
    if (torch_dtype_ == "int64") {
      return torch::kInt64;
    }
    if (torch_dtype_ == "uint8") {
      return torch::kUInt8;
    }
    if (torch_dtype_ == "bool") {
      return torch::kBool;
    }
    LOG(FATAL) << "Unsupported OneRec weight dtype " << torch_dtype_
               << ", layer_id=" << layer_id_;
    return dtype_;
  };
  const auto correct_tensor_dtype = [this, &target_weight_dtype](
                                        torch::Tensor& tensor,
                                        const std::string& tensor_name) {
    if (absl::EndsWith(tensor_name, "deq_scale") &&
        torch_dtype_ == "bfloat16") {
      return;
    }
    if (tensor.dtype() != torch::kInt8 && tensor.dtype() != torch::kInt32 &&
        tensor.dtype() != torch::kInt64) {
      tensor = tensor.to(target_weight_dtype());
    }
  };
  const auto load_weight = [this, &state_dict, &correct_tensor_dtype](
                               const std::string& tensor_name,
                               int weight_position,
                               int shard_dim = -1) {
    for (const auto& [name, tensor] : state_dict) {
      if (!absl::EndsWith(name, tensor_name)) {
        continue;
      }
      torch::Tensor mutable_tensor =
          (shard_dim >= 0 && parallel_args_.world_size() > 1)
              ? state_dict.get_sharded_tensor(tensor_name,
                                              shard_dim,
                                              parallel_args_.rank(),
                                              parallel_args_.world_size())
              : tensor;
      correct_tensor_dtype(mutable_tensor, tensor_name);
      at_weight_tensors_[weight_position] = mutable_tensor.to(device_);
      return;
    }
  };

  const auto& weight_mapping =
      [this]() -> const std::unordered_map<std::string, int>& {
    if (prefill_param_.use_moe) {
      return kOneRecDecoderMoeWeightMapping;
    }
    return is_decoder_ ? kOneRecDecoderWeightMapping
                       : kOneRecEncoderWeightMapping;
  }();

  if (prefill_param_.use_moe) {
    for (auto& [key, tensors] : experts_weights_) {
      (void)key;
      for (auto& t : tensors) {
        t = torch::Tensor();
      }
    }
    shared_expert_weights_map_.clear();
    shared_expert_gate_weights_.clear();
    shared_expert_up_weights_.clear();
    shared_expert_down_weights_.clear();

    for (const auto& [state_key, tensor] : state_dict) {
      if (state_key.find(".ffn.experts.") != std::string::npos) {
        process_expert_weights(state_dict, state_key, tensor);
      }
    }

    for (const auto& [state_key, tensor] : state_dict) {
      const bool is_shared_expert =
          (state_key.find(".ffn.shared_experts.") != std::string::npos ||
           state_key.find(".ffn.shared_expert.") != std::string::npos);
      if (is_shared_expert) {
        process_shared_expert_weights(state_dict, state_key, tensor);
      }
    }
  }

  std::vector<std::pair<std::string, int>> ordered_mapping(
      weight_mapping.begin(), weight_mapping.end());
  std::sort(
      ordered_mapping.begin(),
      ordered_mapping.end(),
      [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });
  for (const auto& [name, index] : ordered_mapping) {
    const bool is_relative_bias = (index == IN_RELATIVE_ATTENTION_BIAS_WEIGHT);
    bool weight_exists = false;
    for (const auto& [state_key, tensor] : state_dict) {
      (void)tensor;
      if (absl::EndsWith(state_key, name)) {
        weight_exists = true;
        break;
      }
    }
    if (is_relative_bias && !weight_exists) {
      continue;
    }

    const auto it = kOneRecWeightShard.find(index);
    if (it != kOneRecWeightShard.end()) {
      load_weight(name, index, it->second);
    } else {
      load_weight(name, index);
    }
  }
}

int64_t NpuOneRecBlockLayerImpl::init_layer() {
  name_ =
      is_decoder_ ? "onerec_decoder_block_layer" : "onerec_encoder_block_layer";
  model_name_ = "onerec";
  CHECK_OPERATION_STATUS_RETURN(init_node(prefill_node_, prefill_param_));
  if (is_decoder_) {
    if (FLAGS_enable_rec_prefill_only) {
      LOG(INFO) << "OneRec BlockLayer init_layer skip decode node because "
                   "enable_rec_prefill_only is enabled"
                << ", layer_id=" << layer_id_;
      LOG(INFO) << "OneRec BlockLayer init_layer success"
                << ", layer_role=" << (is_decoder_ ? "decoder" : "encoder")
                << ", layer_id=" << layer_id_ << ", status=" << atb::NO_ERROR;
      return atb::NO_ERROR;
    }
    const int64_t decode_status = init_node(decode_node_, decode_param_);
    LOG(INFO) << "OneRec BlockLayer init_layer node returned"
              << ", node=decoder-decode"
              << ", layer_id=" << layer_id_ << ", status=" << decode_status;
    CHECK_OPERATION_STATUS_RETURN(decode_status);
  } else {
    LOG(INFO) << "OneRec BlockLayer init_layer skip decode node"
              << ", layer_role=" << (is_decoder_ ? "decoder" : "encoder")
              << ", layer_id=" << layer_id_;
  }
  return atb::NO_ERROR;
}

int64_t NpuOneRecBlockLayerImpl::init_attn_mask() { return atb::NO_ERROR; }

int64_t NpuOneRecBlockLayerImpl::init_node(
    atb_speed::Model::Node& node,
    atb_speed::onerec::BlockLayerParam& param) {
  atb::Operation* operation = nullptr;
  atb::Status status = atb_speed::onerec::BlockLayer(param, &operation);
  if (status != atb::NO_ERROR) {
    LOG(ERROR) << "Failed to create ONEREC BlockLayer operation, status: "
               << status;
    return status;
  }

  node.operation.reset(operation);
  if (node.operation == nullptr) {
    LOG(ERROR) << "node.operation is null after creation";
    return -1;
  }

  uint32_t input_num = node.operation->GetInputNum();
  uint32_t output_num = node.operation->GetOutputNum();

  node.inTensors.resize(input_num);
  node.outTensors.resize(output_num);

  const uint64_t weight_count = param.use_moe ? kOneRecMoeWeightCountPerLayer
                                              : kOneRecWeightCountPerLayer;
  for (size_t weight_tensor_id = 0; weight_tensor_id < weight_count;
       ++weight_tensor_id) {
    if (weight_tensor_id < input_num) {
      node.inTensors.at(weight_tensor_id) =
          &atb_weight_tensors_[weight_tensor_id];
    }
  }

  node.variantPack.inTensors.resize(input_num);
  node.variantPack.outTensors.resize(output_num);

  return atb::NO_ERROR;
}

torch::Tensor NpuOneRecBlockLayerImpl::forward(
    torch::Tensor& x,
    torch::Tensor& attn_mask,
    KVCache& kv_cache,
    ModelInputParams& input_params,
    torch::Tensor* encoder_output,
    int node_id,
    aclrtEvent* event,
    std::atomic<bool>* event_flag,
    const torch::Tensor& expert_array) {
  const auto* onerec_params = input_params.onerec_params();
  CHECK(onerec_params != nullptr) << "OneRec requires rec_params.";

  const bool is_prefill =
      onerec_params->rec_stage == OneRecModelInputParams::RecStage::PREFILL;

  atb::Status st;
  if (is_prefill) {
    if (is_decoder_) {
      if (prefill_param_.use_moe) {
        build_decoder_moe_node_variant_pack(prefill_node_,
                                            x,
                                            attn_mask,
                                            kv_cache,
                                            input_params,
                                            true,
                                            encoder_output,
                                            node_id,
                                            expert_array);
      } else {
        build_decoder_node_variant_pack(prefill_node_,
                                        x,
                                        attn_mask,
                                        kv_cache,
                                        input_params,
                                        true,
                                        encoder_output,
                                        node_id);
      }
      st = execute_node(prefill_node_, node_id, event, event_flag);
      LOG_IF(FATAL, st != 0)
          << model_name_ << " execute prefill layer fail, error code: " << st;
    } else {
      build_encoder_node_variant_pack(
          prefill_node_, x, attn_mask, input_params, true, node_id);
      st = execute_node(prefill_node_, node_id, event, event_flag);
      LOG_IF(FATAL, st != 0)
          << model_name_
          << " execute encoder prefill layer fail, error code: " << st;
    }
  } else {
    if (!is_decoder_) {
      LOG(FATAL) << model_name_ << " encoder decode stage is not supported.";
    }

    if (decode_param_.use_moe) {
      build_decoder_moe_node_variant_pack(decode_node_,
                                          x,
                                          attn_mask,
                                          kv_cache,
                                          input_params,
                                          false,
                                          encoder_output,
                                          node_id,
                                          expert_array);
    } else {
      build_decoder_node_variant_pack(decode_node_,
                                      x,
                                      attn_mask,
                                      kv_cache,
                                      input_params,
                                      false,
                                      encoder_output,
                                      node_id);
    }
    st = execute_node(decode_node_, node_id + 1000, event, event_flag);
    LOG_IF(FATAL, st != 0) << model_name_
                           << " execute decode layer fail, error code: " << st;
  }

  return at_placeholder_;
}

void NpuOneRecBlockLayerImpl::build_encoder_node_variant_pack(
    atb_speed::Model::Node& node,
    torch::Tensor& x,
    at::Tensor& attn_mask,
    ModelInputParams& input_params,
    bool is_prefill,
    int layer_id) {
  (void)is_prefill;
  (void)layer_id;

  internal_tensors_ = atb_speed::Utils::AtTensor2Tensor(x);

  for (size_t i = 0; i < kOneRecWeightCountPerLayer; ++i) {
    CHECK(node.inTensors.at(i) != nullptr)
        << model_name_ << " inTensor " << i << " is NULL";
    node.variantPack.inTensors.at(i) = *node.inTensors.at(i);
  }

  const int32_t input_tensor_idx =
      static_cast<int32_t>(kOneRecWeightCountPerLayer);
  const int32_t attention_mask_idx = input_tensor_idx + 1;
  const int32_t token_offset_idx = attention_mask_idx + 1;
  const int32_t layer_id_idx = token_offset_idx + 1;
  const int32_t seq_len_idx = layer_id_idx + 1;

  node.variantPack.inTensors.at(input_tensor_idx) = internal_tensors_;
  node.variantPack.inTensors.at(attention_mask_idx) =
      atb_speed::Utils::AtTensor2Tensor(attn_mask);

  node.variantPack.inTensors.at(token_offset_idx) = placeholder_;
  node.variantPack.inTensors.at(token_offset_idx).hostData =
      placeholder_vec_.data();
  node.variantPack.inTensors.at(layer_id_idx) = placeholder_;
  node.variantPack.inTensors.at(layer_id_idx).hostData =
      placeholder_vec_.data();

  const auto* onerec_params = input_params.onerec_params();
  if (onerec_params != nullptr &&
      onerec_params->encoder_seq_lens_tensor.defined()) {
    node.variantPack.inTensors.at(seq_len_idx) =
        atb_speed::Utils::AtTensor2Tensor(
            onerec_params->encoder_seq_lens_tensor);
    node.variantPack.inTensors.at(seq_len_idx).hostData =
        const_cast<int32_t*>(onerec_params->encoder_seq_lens.data());
  } else {
    node.variantPack.inTensors.at(seq_len_idx) = placeholder_;
    node.variantPack.inTensors.at(seq_len_idx).hostData =
        placeholder_vec_.data();
  }

  node.variantPack.outTensors.at(0) = internal_tensors_;
}

void NpuOneRecBlockLayerImpl::build_decoder_moe_node_variant_pack(
    atb_speed::Model::Node& node,
    torch::Tensor& x,
    at::Tensor& attn_mask,
    KVCache& kv_cache,
    ModelInputParams& input_params,
    bool is_prefill,
    torch::Tensor* encoder_output,
    int layer_id,
    const torch::Tensor& expert_array) {
  (void)kv_cache;
  (void)is_prefill;
  (void)layer_id;

  for (size_t i = 0; i < kOneRecMoeWeightCountPerLayer; ++i) {
    CHECK(node.inTensors.at(i) != nullptr)
        << model_name_ << " inTensor " << i << " is NULL";
    node.variantPack.inTensors.at(i) = *node.inTensors.at(i);
  }

  const int moe_tensor_start = static_cast<int>(kOneRecMoeWeightCountPerLayer);
  if (expert_array.defined()) {
    node.variantPack.inTensors.at(moe_tensor_start) =
        atb_speed::Utils::AtTensor2Tensor(expert_array);
  } else {
    node.variantPack.inTensors.at(moe_tensor_start) = placeholder_;
  }

  node.variantPack.inTensors.at(moe_tensor_start + 1) =
      expert_group_.defined() ? atb_speed::Utils::AtTensor2Tensor(expert_group_)
                              : placeholder_;
  node.variantPack.inTensors.at(moe_tensor_start + 2) =
      one_hot_.defined() ? atb_speed::Utils::AtTensor2Tensor(one_hot_)
                         : placeholder_;
  node.variantPack.inTensors.at(moe_tensor_start + 3) =
      zero_hot_.defined() ? atb_speed::Utils::AtTensor2Tensor(zero_hot_)
                          : placeholder_;

  int tensor_idx = setup_common_decoder_tensors(
      node, x, attn_mask, input_params, encoder_output, moe_tensor_start + 4);

  while (tensor_idx < static_cast<int>(node.variantPack.inTensors.size())) {
    node.variantPack.inTensors.at(tensor_idx) = placeholder_;
    node.variantPack.inTensors.at(tensor_idx).hostData =
        placeholder_vec_.data();
    ++tensor_idx;
  }
}

int NpuOneRecBlockLayerImpl::setup_common_decoder_tensors(
    atb_speed::Model::Node& node,
    torch::Tensor& x,
    at::Tensor& attn_mask,
    ModelInputParams& input_params,
    torch::Tensor* encoder_output,
    int start_tensor_idx) {
  internal_tensors_ = atb_speed::Utils::AtTensor2Tensor(x);

  int idx = start_tensor_idx;
  node.variantPack.inTensors.at(idx++) = internal_tensors_;
  node.variantPack.inTensors.at(idx++) =
      atb_speed::Utils::AtTensor2Tensor(attn_mask);

  // Token offset and layer id placeholders.
  // ATB expects hostData to be valid for these scalar inputs. Keep them as
  // placeholders but always provide hostData to avoid undefined reads.
  node.variantPack.inTensors.at(idx) = placeholder_;
  node.variantPack.inTensors.at(idx++).hostData = placeholder_vec_.data();
  node.variantPack.inTensors.at(idx) = placeholder_;
  node.variantPack.inTensors.at(idx++).hostData = placeholder_vec_.data();

  CHECK(input_params.kv_seq_lens.defined()) << "kv_seq_lens is required.";
  node.variantPack.inTensors.at(idx) =
      atb_speed::Utils::AtTensor2Tensor(input_params.kv_seq_lens);
  node.variantPack.inTensors.at(idx).hostData =
      input_params.kv_seq_lens_vec.data();
  idx++;

  node.variantPack.inTensors.at(idx) = placeholder_;
  node.variantPack.inTensors.at(idx++).hostData = placeholder_vec_.data();
  node.variantPack.inTensors.at(idx) = placeholder_;
  node.variantPack.inTensors.at(idx++).hostData = placeholder_vec_.data();

  if (!FLAGS_enable_rec_prefill_only && input_params.block_tables.defined()) {
    node.variantPack.inTensors.at(idx) =
        atb_speed::Utils::AtTensor2Tensor(input_params.block_tables);
  } else {
    node.variantPack.inTensors.at(idx) = placeholder_;
    node.variantPack.inTensors.at(idx).hostData = placeholder_vec_.data();
  }
  idx++;

  if (!FLAGS_enable_rec_prefill_only &&
      input_params.new_cache_slots.defined()) {
    node.variantPack.inTensors.at(idx) =
        atb_speed::Utils::AtTensor2Tensor(input_params.new_cache_slots);
  } else {
    node.variantPack.inTensors.at(idx) = placeholder_;
    node.variantPack.inTensors.at(idx).hostData = placeholder_vec_.data();
  }
  idx++;

  if (encoder_output != nullptr) {
    encoder_output_contiguous_ = encoder_output->is_contiguous()
                                     ? *encoder_output
                                     : encoder_output->contiguous();
    node.variantPack.inTensors.at(idx) =
        atb_speed::Utils::AtTensor2Tensor(encoder_output_contiguous_);
  } else {
    node.variantPack.inTensors.at(idx) = placeholder_;
  }
  idx++;

  for (int i = 0; i < 3; i++) {
    node.variantPack.inTensors.at(idx) = placeholder_;
    node.variantPack.inTensors.at(idx++).hostData = placeholder_vec_.data();
  }

  const auto* onerec_params = input_params.onerec_params();
  if (onerec_params != nullptr &&
      onerec_params->encoder_seq_lens_tensor.defined()) {
    node.variantPack.inTensors.at(idx) = atb_speed::Utils::AtTensor2Tensor(
        onerec_params->encoder_seq_lens_tensor);
    node.variantPack.inTensors.at(idx++).hostData =
        const_cast<int32_t*>(onerec_params->encoder_seq_lens.data());
  } else {
    node.variantPack.inTensors.at(idx) = placeholder_;
    node.variantPack.inTensors.at(idx++).hostData = placeholder_vec_.data();
  }

  node.variantPack.outTensors.at(0) = internal_tensors_;
  return idx;
}

void NpuOneRecBlockLayerImpl::build_decoder_node_variant_pack(
    atb_speed::Model::Node& node,
    torch::Tensor& x,
    at::Tensor& attn_mask,
    KVCache& kv_cache,
    ModelInputParams& input_params,
    bool is_prefill,
    torch::Tensor* encoder_output,
    int layer_id) {
  (void)kv_cache;
  (void)is_prefill;
  (void)layer_id;

  for (size_t i = 0; i < kOneRecWeightCountPerLayer; ++i) {
    CHECK(node.inTensors.at(i) != nullptr)
        << model_name_ << " inTensor " << i << " is NULL";
    node.variantPack.inTensors.at(i) = *node.inTensors.at(i);
  }

  int tensor_idx = setup_common_decoder_tensors(
      node,
      x,
      attn_mask,
      input_params,
      encoder_output,
      static_cast<int>(kOneRecWeightCountPerLayer));
  while (tensor_idx < static_cast<int>(node.variantPack.inTensors.size())) {
    node.variantPack.inTensors.at(tensor_idx) = placeholder_;
    node.variantPack.inTensors.at(tensor_idx).hostData =
        placeholder_vec_.data();
    ++tensor_idx;
  }
}

void NpuOneRecBlockLayerImpl::resize_experts_weights(
    int num_of_device_experts) {
  experts_weights_["gate_proj.weight"] =
      std::vector<torch::Tensor>(num_of_device_experts);
  experts_weights_["up_proj.weight"] =
      std::vector<torch::Tensor>(num_of_device_experts);
  experts_weights_["down_proj.weight"] =
      std::vector<torch::Tensor>(num_of_device_experts);
}

void NpuOneRecBlockLayerImpl::process_expert_weights(
    const StateDict& state_dict,
    const std::string& name,
    const torch::Tensor& tensor) {
  (void)state_dict;
  std::lock_guard<std::mutex> lock(experts_mutex_);

  int expert_id = extract_expert_index(name);
  if (expert_id < 0) {
    return;
  }

  const int local_index = expert_id % num_experts_per_partition_;
  std::string weight_suffix = extract_endswith(name);

  std::string suffix;
  if (weight_suffix == "gate_proj.weight" || weight_suffix == "w1.weight") {
    suffix = "gate_proj.weight";
  } else if (weight_suffix == "up_proj.weight" ||
             weight_suffix == "w3.weight") {
    suffix = "up_proj.weight";
  } else if (weight_suffix == "down_proj.weight" ||
             weight_suffix == "w2.weight") {
    suffix = "down_proj.weight";
  } else {
    return;
  }

  auto it = experts_weights_.find(suffix);
  if (it == experts_weights_.end() || local_index < 0 ||
      local_index >= static_cast<int>(it->second.size())) {
    LOG(ERROR) << "Invalid OneRec MoE local expert index " << local_index
               << " for " << suffix << " at layer " << layer_id_ << ".";
    return;
  }
  it->second[local_index] = tensor;
}

void NpuOneRecBlockLayerImpl::process_shared_expert_weights(
    const StateDict& state_dict,
    const std::string& name,
    const torch::Tensor& tensor) {
  (void)state_dict;
  torch::Tensor tmp_tensor = tensor.to(device_);

  std::string canonical_name;
  if (absl::StrContains(name, "gate_proj") || absl::StrContains(name, "w1")) {
    canonical_name = "gate_proj.weight";
  } else if (absl::StrContains(name, "up_proj") ||
             absl::StrContains(name, "w3")) {
    canonical_name = "up_proj.weight";
  } else if (absl::StrContains(name, "down_proj") ||
             absl::StrContains(name, "w2")) {
    canonical_name = "down_proj.weight";
  } else {
    return;
  }

  if (shared_expert_weights_map_.count(canonical_name) > 0) {
    LOG(WARNING) << "Duplicate OneRec shared expert tensor for "
                 << canonical_name << " at layer " << layer_id_
                 << ", overriding previous value.";
  }
  shared_expert_weights_map_[canonical_name] = tmp_tensor;
}

int NpuOneRecBlockLayerImpl::extract_expert_index(const std::string& name) {
  size_t experts_pos = name.find(".experts.");
  if (experts_pos == std::string::npos) {
    return -1;
  }

  size_t start_pos = experts_pos + 9;
  size_t end_pos = name.find(".", start_pos);
  if (end_pos == std::string::npos) {
    return -1;
  }

  try {
    return std::stoi(name.substr(start_pos, end_pos - start_pos));
  } catch (const std::exception&) {
    return -1;
  }
}

std::string NpuOneRecBlockLayerImpl::extract_endswith(
    const std::string& input) {
  size_t experts_pos = input.find(".experts.");
  if (experts_pos == std::string::npos) {
    return "";
  }
  size_t start_pos = experts_pos + 9;
  size_t next_dot = input.find(".", start_pos);
  if (next_dot == std::string::npos) {
    return "";
  }
  return input.substr(next_dot + 1);
}

torch::Tensor NpuOneRecBlockLayerImpl::merge_experts_weights(
    std::vector<torch::Tensor>& experts,
    bool transpose) {
  std::vector<torch::Tensor> valid;
  valid.reserve(experts.size());
  for (auto& t : experts) {
    if (t.defined()) {
      valid.push_back(t.to(device_));
    }
  }
  if (valid.empty()) {
    LOG(ERROR) << "No expert weights to merge at layer " << layer_id_ << ".";
    return torch::Tensor();
  }
  torch::Tensor merged_tensor = torch::stack(valid, 0);
  if (transpose) {
    merged_tensor = merged_tensor.transpose(1, 2);
  }
  return merged_tensor.contiguous();
}

torch::Tensor NpuOneRecBlockLayerImpl::merge_experts_weights(
    std::vector<torch::Tensor>& experts_gate,
    std::vector<torch::Tensor>& experts_up,
    bool transpose) {
  if (experts_gate.size() != experts_up.size()) {
    LOG(ERROR) << "OneRec MoE gate/up expert size mismatch: gate="
               << experts_gate.size() << ", up=" << experts_up.size()
               << ", layer " << layer_id_;
    return torch::Tensor();
  }
  for (size_t i = 0; i < experts_gate.size(); ++i) {
    const bool gate_defined = experts_gate[i].defined();
    const bool up_defined = experts_up[i].defined();
    if (gate_defined != up_defined) {
      LOG(ERROR) << "OneRec MoE gate/up tensor mismatch at local expert " << i
                 << ": gate=" << gate_defined << ", up=" << up_defined
                 << ", layer " << layer_id_;
      return torch::Tensor();
    }
    if (gate_defined) {
      experts_gate[i] = torch::cat({experts_gate[i], experts_up[i]}, 0);
    }
  }
  return merge_experts_weights(experts_gate, transpose);
}

void NpuOneRecBlockLayerImpl::merge_experts_weights() {
  if (experts_weights_.count("gate_proj.weight") == 0 ||
      experts_weights_.count("up_proj.weight") == 0 ||
      experts_weights_.count("down_proj.weight") == 0) {
    return;
  }

  auto merged_gate_up =
      merge_experts_weights(experts_weights_["gate_proj.weight"],
                            experts_weights_["up_proj.weight"],
                            /*transpose=*/false);
  CHECK(merged_gate_up.defined()) << "OneRec MoE gate/up experts merge failed.";
  at_weight_tensors_[IN_MOE_EXPERT_W1_WEIGHT] =
      at_npu::native::npu_format_cast(merged_gate_up, /*format=*/2)
          .contiguous();

  auto merged_down = merge_experts_weights(experts_weights_["down_proj.weight"],
                                           /*transpose=*/false);
  CHECK(merged_down.defined()) << "OneRec MoE down experts merge failed.";
  at_weight_tensors_[IN_MOE_EXPERT_W2_WEIGHT] =
      at_npu::native::npu_format_cast(merged_down, /*format=*/2).contiguous();
}

void NpuOneRecBlockLayerImpl::merge_shared_experts_weights() {
  shared_expert_gate_weights_.clear();
  shared_expert_up_weights_.clear();
  shared_expert_down_weights_.clear();

  if (const auto it = shared_expert_weights_map_.find("gate_proj.weight");
      it != shared_expert_weights_map_.end()) {
    shared_expert_gate_weights_.push_back(it->second);
  }
  if (const auto it = shared_expert_weights_map_.find("up_proj.weight");
      it != shared_expert_weights_map_.end()) {
    shared_expert_up_weights_.push_back(it->second);
  }
  if (const auto it = shared_expert_weights_map_.find("down_proj.weight");
      it != shared_expert_weights_map_.end()) {
    shared_expert_down_weights_.push_back(it->second);
  }

  if (shared_expert_gate_weights_.empty() &&
      shared_expert_up_weights_.empty() &&
      shared_expert_down_weights_.empty()) {
    return;
  }

  if (!shared_expert_gate_weights_.empty() &&
      !shared_expert_up_weights_.empty()) {
    auto merged_gate_up = merge_experts_weights(shared_expert_gate_weights_,
                                                shared_expert_up_weights_,
                                                /*transpose=*/false);
    CHECK(merged_gate_up.defined())
        << "OneRec shared gate/up experts merge failed at layer " << layer_id_;
    at_weight_tensors_[IN_MLP_GATEUP_WEIGHT_SHARED_EXPERT] = merged_gate_up;
  } else if (!shared_expert_gate_weights_.empty()) {
    at_weight_tensors_[IN_MLP_GATEUP_WEIGHT_SHARED_EXPERT] =
        merge_experts_weights(shared_expert_gate_weights_, false);
  }

  if (!shared_expert_down_weights_.empty()) {
    at_weight_tensors_[IN_MLP_DOWN_WEIGHT_SHARED_EXPERT] =
        merge_experts_weights(shared_expert_down_weights_, false);
  }

  shared_expert_gate_weights_.clear();
  shared_expert_up_weights_.clear();
  shared_expert_down_weights_.clear();
  shared_expert_weights_map_.clear();
}

}  // namespace layer
}  // namespace xllm

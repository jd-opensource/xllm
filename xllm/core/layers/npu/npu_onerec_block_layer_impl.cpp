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

#include "npu_onerec_block_layer_impl.h"

#include <glog/logging.h>
#include <mstx/ms_tools_ext.h>

#include <cstring>
#include <map>

#include "common/global_flags.h"
#include "core/layers/attention_mask.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUException.h"

namespace xllm {
namespace layer {
// Decoder normal mode: self-attn(29) + cross-attn(28) + layer-norm(4) + mlp(18)
// = 79
const uint64_t ONEREC_WEIGHT_COUNT_PER_LAYER = 79;
// Decoder normal mode: self-attn(29) + cross-attn(28) + layer-norm(4) + mlp(18)
// + data(4) = 83
const uint64_t ONEREC_MOE_WEIGHT_COUNT_PER_LAYER =
    97;  // Decoder only with MoE (61-100: 40 weight tensors)
// + 2 + 2 + 2
enum ONERECBlockLayerTensorId : int {
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

enum ONERECMoeBlockLayerTensorId : int {
  // MoE weights (only used when use_moe=true) - Updated to match kernel layer
  // names
  IN_BLOCK_SPARSE_MOE_GATE_WEIGHT = 61,   // Gate/routing weights
  IN_BLOCK_SPARSE_MOE_GATE_BIAS = 62,     // Gate bias
  IN_BLOCK_SPARSE_MOE_GATE_DESCALE,       // Gate descale
  IN_BLOCK_SPARSE_MOE_GATE_OFFSET,        // Gate offset
  IN_BLOCK_SPARSE_MOE_GATE_SCALE,         // Gate scale
  IN_BLOCK_SPARSE_MOE_GATE_COMPRESS_IDX,  // Gate compress index

  // Shared Expert weights
  IN_MLP_GATEUP_WEIGHT_SHARED_EXPERT,   // Shared expert gateup weights (merged
                                        // gate+up projection)
  IN_MLP_GATEUP_BIAS_SHARED_EXPERT,     // Shared expert gateup bias
  IN_MLP_GATEUP_DESCALE_SHARED_EXPERT,  // Shared expert gateup descale
  IN_MLP_GATEUP_OFFSET_SHARED_EXPERT,   // Shared expert gateup offset
  IN_MLP_GATEUP_SCALE_SHARED_EXPERT,    // Shared expert gateup scale
  IN_MLP_GATEUP_COMPRESS_IDX_SHARED_EXPERT,  // Shared expert gateup compress
                                             // index

  IN_MLP_DOWN_WEIGHT_SHARED_EXPERT,   // Shared expert down projection weights
  IN_MLP_DOWN_BIAS_SHARED_EXPERT,     // Shared expert down bias
  IN_MLP_DOWN_DESCALE_SHARED_EXPERT,  // Shared expert down descale
  IN_MLP_DOWN_OFFSET_SHARED_EXPERT,   // Shared expert down offset
  IN_MLP_DOWN_SCALE_SHARED_EXPERT,    // Shared expert down scale
  IN_MLP_DOWN_COMPRESS_IDX_SHARED_EXPERT,  // Shared expert down compress index

  IN_SHARED_EXPERT_GATE_WEIGHT,        // Shared expert gate weight
  IN_SHARED_EXPERT_GATE_BIAS,          // Shared expert gate bias
  IN_SHARED_EXPERT_GATE_DESCALE,       // Shared expert gate descale
  IN_SHARED_EXPERT_GATE_OFFSET,        // Shared expert gate offset
  IN_SHARED_EXPERT_GATE_SCALE,         // Shared expert gate scale
  IN_SHARED_EXPERT_GATE_COMPRESS_IDX,  // Shared expert gate compress index

  IN_MLP_GATEUP_WEIGHT_EXPERT,        // Expert gateup weights (merged gate+up
                                      // projection)
  IN_MLP_GATEUP_BIAS_EXPERT,          // Expert gateup bias
  IN_MLP_GATEUP_DESCALE_EXPERT,       // Expert gateup descale
  IN_MLP_GATEUP_OFFSET_EXPERT,        // Expert gateup offset
  IN_MLP_GATEUP_SCALE_EXPERT,         // Expert gateup scale
  IN_MLP_GATEUP_COMPRESS_IDX_EXPERT,  // Expert gateup compress index

  IN_MLP_DOWN_WEIGHT_EXPERT,             // Expert down projection weights
  IN_MLP_DOWN_BIAS_EXPERT,               // Expert down bias
  IN_MLP_DOWN_DESCALE_EXPERT,            // Expert down descale
  IN_MLP_DOWN_OFFSET_EXPERT,             // Expert down offset
  IN_MLP_DOWN_SCALE_EXPERT,              // Expert down scale
  IN_MLP_DOWN_COMPRESS_IDX_EXPERT = 96,  // Expert down compress index

  IN_EXPERT_ARRAY = 97,  // Expert array tensor
  IN_EXPERT_GROUP = 98,  // Expert group tensor
  IN_ONE_HOT = 99,       // One hot tensor
  IN_ZERO_HOT = 100,     // Zero hot tensor

  // Legacy aliases for backward compatibility
  IN_MOE_EXPERT_W1_WEIGHT = IN_MLP_GATEUP_WEIGHT_EXPERT,
  IN_MOE_EXPERT_W2_WEIGHT = IN_MLP_DOWN_WEIGHT_EXPERT,
  IN_MOE_EXPERT_W3_WEIGHT =
      IN_MLP_GATEUP_WEIGHT_EXPERT,  // Same as W1 for gate+up merged
  IN_MOE_SHARED_W1_WEIGHT = IN_MLP_GATEUP_WEIGHT_SHARED_EXPERT,
  IN_MOE_SHARED_W2_WEIGHT = IN_MLP_DOWN_WEIGHT_SHARED_EXPERT,
};

// ONEREC encoder weight mapping - Updated to match actual weight file format
static const std::unordered_map<std::string, int>
    ONEREC_ENCODER_WEIGHT_MAPPING = {
        // Primary mappings - match actual weight file format with full paths
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
        // Alternative mappings for different weight file formats
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

// ONEREC decoder weight mapping - Updated to match actual weight file format
static const std::unordered_map<std::string, int>
    ONEREC_DECODER_WEIGHT_MAPPING = {
        // Primary mappings - match actual weight file format with full paths
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
        {"layer.2.ffn.wi.weight", IN_FFN_WI_1_WEIGHT},
        {"layer.2.ffn.wo.weight", IN_FFN_WO_WEIGHT},
        {"layer.2.ffn.gate_proj.weight", IN_FFN_WI_0_WEIGHT},
        // Alternative mappings for different weight file formats
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

// ONEREC MoE weight mapping for decoder
// MoE weight mapping function - handles individual expert weights
static std::unordered_map<std::string, int>
get_onerec_decoder_moe_weight_mapping() {
  std::unordered_map<std::string, int> mapping = {
      // Self-attention layer norm
      {"layer.0.layer_norm.weight", IN_LAYER_NORM_WEIGHT},
      {"layer.0.SelfAttention.q.weight", IN_Q_WEIGHT},
      {"layer.0.SelfAttention.k.weight", IN_K_WEIGHT},
      {"layer.0.SelfAttention.v.weight", IN_V_WEIGHT},
      {"layer.0.SelfAttention.o.weight", IN_SELF_ATTN_OUT_WEIGHT},
      {"layer.0.SelfAttention.relative_attention_bias.weight",
       IN_RELATIVE_ATTENTION_BIAS_WEIGHT},
      // Cross-attention layer norm
      {"layer.1.layer_norm.weight", IN_CROSS_LAYER_NORM_WEIGHT},
      {"layer.1.EncDecAttention.q.weight", IN_CROSS_Q_WEIGHT},
      {"layer.1.EncDecAttention.k.weight", IN_CROSS_K_WEIGHT},
      {"layer.1.EncDecAttention.v.weight", IN_CROSS_V_WEIGHT},
      {"layer.1.EncDecAttention.o.weight", IN_CROSS_ATTN_OUT_WEIGHT},
      {"layer.2.layer_norm.weight", IN_FINAL_LAYER_NORM_WEIGHT},

      // Alternative naming patterns
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
      // MoE gate weight
      {"layer.2.ffn.gate.weight", IN_BLOCK_SPARSE_MOE_GATE_WEIGHT},
      {"2.ffn.gate.weight", IN_BLOCK_SPARSE_MOE_GATE_WEIGHT},

      // Shared Expert weight mappings (using w1/w2/w3 naming)
      {"layer.2.ffn.shared_experts.w1.weight",
       IN_MLP_GATEUP_WEIGHT_SHARED_EXPERT},
      {"layer.2.ffn.shared_experts.w3.weight",
       IN_MLP_GATEUP_WEIGHT_SHARED_EXPERT},
      {"layer.2.ffn.shared_experts.w2.weight",
       IN_MLP_DOWN_WEIGHT_SHARED_EXPERT},

      // Shared Expert gate weight mappings
      {"layer.2.ffn.shared_expert.gate.weight", IN_SHARED_EXPERT_GATE_WEIGHT},
      {"layer.2.ffn.shared_expert.gate.bias", IN_SHARED_EXPERT_GATE_BIAS},
      {"layer.2.ffn.shared_expert.gate.weight_scale",
       IN_SHARED_EXPERT_GATE_SCALE},
      {"layer.2.ffn.shared_expert.gate.weight_offset",
       IN_SHARED_EXPERT_GATE_OFFSET},

      // Expert weight mappings (without expert index - processed by
      // extract_expert_index)
      // Gate projection weights (w1) - merged with up in gateup
      {"w1.weight", IN_MLP_GATEUP_WEIGHT_EXPERT},
      // Up projection weights (w3) - merged with gate in gateup
      {"w3.weight", IN_MLP_GATEUP_WEIGHT_EXPERT},
      // Down projection weights (w2)
      {"w2.weight", IN_MLP_DOWN_WEIGHT_EXPERT},
  };

  return mapping;
}

static const std::unordered_map<std::string, int>
    ONEREC_DECODER_MOE_WEIGHT_MAPPING = get_onerec_decoder_moe_weight_mapping();

// ONEREC MoE weight mapping for encoder
// ONEREC_ENCODER_MOE_WEIGHT_MAPPING removed - use_moe only supports decoder
// mode

static std::map<int, int> ONEREC_WEIGHT_SHARD = {
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
    // MoE weights
    {IN_BLOCK_SPARSE_MOE_GATE_WEIGHT, 0},
    {IN_MLP_GATEUP_WEIGHT_EXPERT, 0},
    {IN_MLP_DOWN_WEIGHT_EXPERT, 1},
    // Shared Expert weights
    {IN_MLP_GATEUP_WEIGHT_SHARED_EXPERT, 0},
    {IN_MLP_GATEUP_OFFSET_SHARED_EXPERT, 0},
    {IN_MLP_GATEUP_SCALE_SHARED_EXPERT, 0},
    {IN_MLP_DOWN_WEIGHT_SHARED_EXPERT, 1},
    {IN_MLP_DOWN_OFFSET_SHARED_EXPERT, 1},
    {IN_MLP_DOWN_SCALE_SHARED_EXPERT, 1},
    {IN_SHARED_EXPERT_GATE_WEIGHT, 0},
    {IN_SHARED_EXPERT_GATE_BIAS, 0},
    {IN_SHARED_EXPERT_GATE_SCALE, 0},
    {IN_SHARED_EXPERT_GATE_OFFSET, 0}};

NpuOneRecBlockLayerImpl::NpuOneRecBlockLayerImpl(const ModelContext& context,
                                                 bool is_decoder,
                                                 int layer_id)
    : NpuBaseLayer(context), is_decoder_(is_decoder), layer_id_(layer_id) {
  // LOG(INFO) << "ONERECBlockLayerImpl constructor: " << layer_id_ << ":"
  //           << is_decoder_;
  param_from_args(
      prefill_param_, context.get_model_args(), parallel_args_, true);
  prefill_param_.isDecoder = is_decoder;
  // param_from_args(decode_param_, args, parallel_args, false);
  // decode_param_.isDecoder = is_decoder;

  // Initialize decoder_prefill_only_decode_param_ if enable_onerec_prefill_only
  // is true if (FLAGS_enable_rec_prefill_only && is_decoder) {
  //   param_from_args(
  //       decoder_prefill_only_decode_param_, args, parallel_args, true);
  //   decoder_prefill_only_decode_param_.isDecoder = is_decoder;
  //   decoder_prefill_only_decode_param_.emptyCrossAttn = false;
  // }
  // Choose correct weight count based on use_moe
  int weight_count = prefill_param_.use_moe ? ONEREC_MOE_WEIGHT_COUNT_PER_LAYER
                                            : ONEREC_WEIGHT_COUNT_PER_LAYER;
  at_weight_tensors_.resize(weight_count);
  atb_weight_tensors_.resize(weight_count);
  // Initialize placeholder_vec_ with proper dimensions for ONEREC operations
  // Some ATB operations may require specific tensor dimensions
  placeholder_vec_ = {1, 1};  // 2D placeholder for better compatibility
  dtype_ = c10::typeMetaToScalarType(context.get_tensor_options().dtype());
  device_id_ = context.get_tensor_options().device().index();

  // Create placeholder tensors with proper dimensions for ONEREC operations
  auto placeholder_tensor = torch::empty({1, 1}, torch::kInt32).to(device_);
  placeholder = atb_speed::Utils::AtTensor2Tensor(placeholder_tensor);
  at_placeholder =
      torch::empty({1, context.get_model_args().hidden_size()}, dtype_)
          .to(device_);

  for (int i = 0; i < weight_count; ++i) {
    at_weight_tensors_[i] =
        torch::zeros({1, context.get_model_args().hidden_size()})
            .to(context.get_tensor_options());
  }

  // Initialize MoE routing tensors if MoE is enabled
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
    bool isPrefill,
    const ModelInputParams* input_params) {
  LOG(INFO) << "begin param_from_args";
  param.isFA = false;  // need page
  param.isPrefill = isPrefill;
  param.isBF16 = args.dtype() == "bfloat16";
  param.isPack = true;
  param.supportSwiGLU = true;  // ONEREC now uses gated activation by default
  param.supportLcoc = isPrefill;
  param.supportSpeculate = false;
  param.enableSplitFuse = FLAGS_enable_chunked_prefill && isPrefill;
  param.supportLora = false;
  param.loraEnableGMM = false;
  param.enableLogN = false;
  param.kvQuant = false;
  param.enableIntraLayerAddNorm = false;
  param.enableInterLayerAddNorm = false;
  // ONEREC position bias is now passed through attention_mask with ALIBI mask
  // type hasPositionBias parameter is no longer needed
  param.isDecoder = is_decoder_;
  param.isOneRecEncoder = !is_decoder_;  // ONEREC encoder uses bidirectional
                                         // attention, no KV cache needed
  param.enableOneRecPrefillOnly = FLAGS_enable_rec_prefill_only;
  param.backend = "lccl";
  param.rank = parallel_args.rank();
  param.worldSize = parallel_args.world_size();
  param.quantType = 0;
  param.quantGroupSize = 64;
  auto args_n_heads = is_decoder_ ? args.decoder_n_heads() : args.n_heads();
  auto args_head_dim = is_decoder_ ? args.decoder_head_dim() : args.head_dim();
  param.numAttentionHeadsPerRank = args_n_heads / param.worldSize;
  param.hiddenSizePerAttentionHead = args_head_dim;
  LOG(INFO) << "hiddenSizePerAttentionHead: "
            << param.hiddenSizePerAttentionHead;
  LOG(INFO) << "numAttentionHeadsPerRank: " << param.numAttentionHeadsPerRank;
  std::optional<long int> optionalValue =
      is_decoder_ ? args.decoder_n_kv_heads().value_or(args.decoder_n_heads())
                  : args.n_kv_heads().value_or(args.n_heads());
  param.numKeyValueHeadsPerRank =
      static_cast<int>(optionalValue.value()) / param.worldSize;
  param.rmsNormEps = args.rms_norm_eps();
  param.seqLen = {};
  param.tokenOffset = {};
  param.packQuantType = {1, 1};
  param.linearQuantType = {0, -1, -1, 0, 0, -1, 0};
  param.layerId = layer_id_;

  // Set ModelInputParams for ONEREC model support
  // param.inputParams = input_params;
  // param.linearTransposeType = {1, -1, -1, 1, 1, -1, 1};
  param.linearTransposeType = {1, 1, 1, 1, 1, 1, 1};
  // Initialize linearDescs to enable QKV projection
  // Elements: qkv, dense, gateup, down linear descriptions
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

  // Set use_moe parameter from ModelArgs
  param.use_moe = args.use_moe() && is_decoder_;
  if (param.use_moe) {
    // Initialize MoE parallel configuration (similar to Qwen3 and DeepSeek V2)
    ep_size_ = 1;
    auto ep_rank = 0;
    ep_local_tp_size_ = parallel_args.world_size() / ep_size_;
    CHECK_EQ(parallel_args.world_size(), ep_size_ * ep_local_tp_size_);

    num_experts_per_partition_ = args.n_routed_experts() / ep_size_;
    start_expert_id_ = ep_rank * num_experts_per_partition_;
    end_expert_id_ = start_expert_id_ + num_experts_per_partition_ - 1;

    // Initialize experts weights storage with 2D structure
    resize_experts_weights(num_experts_per_partition_);

    // Configure OneRecMoEConfig
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

    // Initialize moeLinearQuantType for MoE layers
    // Four components: ROUTER_IDX, MOE_MLP_GATE_IDX, MOE_MLP_UP_IDX,
    // MOE_MLP_DOWN_IDX
    param.moeLinearQuantType = {
        atb_speed::common::LinearType::FP,       // ROUTER_IDX (0)
        atb_speed::common::LinearType::FP,       // MOE_MLP_GATE_IDX (1)
        atb_speed::common::LinearType::INVALID,  // MOE_MLP_UP_IDX (2)
        atb_speed::common::LinearType::FP        // MOE_MLP_DOWN_IDX (3)
    };
  }
}

void NpuOneRecBlockLayerImpl::verify_loaded_weights(
    const std::string& prefix) const {
  // Choose appropriate weight mapping based on use_moe flag
  const auto& weight_mapping =
      [this]() -> const std::unordered_map<std::string, int>& {
    if (prefill_param_.use_moe) {
      // If MoE is enabled, apply filtering based on configuration
      if (prefill_param_.moe_config) {
        static std::unordered_map<std::string, int> filtered_mapping;
        filtered_mapping.clear();

        // Copy weights from the full MoE mapping based on configuration
        for (const auto& [name, index] : ONEREC_DECODER_MOE_WEIGHT_MAPPING) {
          bool should_include = true;

          // Filter shared expert weights based on moe_use_shared_experts
          if (!prefill_param_.moe_config->moe_use_shared_experts) {
            if (name.find("shared_expert") != std::string::npos) {
              should_include = false;
            }
          }

          // Further filter shared expert gate weights based on
          // hasSharedExpertGate
          if (should_include &&
              !prefill_param_.moe_config->hasSharedExpertGate) {
            if (name.find("shared_expert_gate") != std::string::npos) {
              should_include = false;
            }
          }

          if (should_include) {
            filtered_mapping[name] = index;
          }
        }
        return filtered_mapping;
      } else {
        return ONEREC_DECODER_MOE_WEIGHT_MAPPING;
      }
    } else {
      return is_decoder_ ? ONEREC_DECODER_WEIGHT_MAPPING
                         : ONEREC_ENCODER_WEIGHT_MAPPING;
    }
  }();

  const uint64_t expected_weight_count = [this]() -> uint64_t {
    if (!prefill_param_.use_moe) {
      return ONEREC_WEIGHT_COUNT_PER_LAYER;
    } else if (prefill_param_.moe_config &&
               !prefill_param_.moe_config->moe_use_shared_experts) {
      // When MoE is enabled but shared experts are disabled, subtract shared
      // expert weights Shared expert weights count: 18 weights (gateup: 6,
      // down: 6, gate: 6)
      const uint64_t shared_expert_weight_count = 18;
      return ONEREC_MOE_WEIGHT_COUNT_PER_LAYER - shared_expert_weight_count;
    } else {
      return ONEREC_MOE_WEIGHT_COUNT_PER_LAYER;
    }
  }();

  // Define weights that are expected to be [1] after merging
  std::set<int> merged_weights = {IN_K_WEIGHT, IN_V_WEIGHT, IN_FFN_WI_1_WEIGHT};
  if (is_decoder_) {
    merged_weights.insert({IN_CROSS_K_WEIGHT, IN_CROSS_V_WEIGHT});
  }

  for (const auto& [name, index] : weight_mapping) {
    auto sizes = at_weight_tensors_[index].sizes();
    bool is_placeholder = (sizes.size() == 2 && sizes[0] == 1);
    bool is_expected_placeholder = merged_weights.count(index) > 0;

    // Special handling for relative_attention_bias - it's optional and only
    // exists in first layer
    bool is_relative_bias = (index == IN_RELATIVE_ATTENTION_BIAS_WEIGHT);

    if (is_placeholder && !is_expected_placeholder && !is_relative_bias) {
      CHECK(false) << "weight is not loaded for " << prefix << name;
    }

    // if (is_relative_bias && is_placeholder) {
    //   LOG(INFO) << "[ONEREC DEBUG] Weight " << prefix << name
    //             << " is placeholder (expected for non-first layers)";
    // }
  }
}

void NpuOneRecBlockLayerImpl::merge_loaded_weights() {
  // Debug: Print shapes before merging
  /*
  LOG(INFO) << "[ONEREC DEBUG] Before merging QKV weights:";
  LOG(INFO) << "[ONEREC DEBUG]   Q weight shape: ["
            << at_weight_tensors_[IN_Q_WEIGHT].sizes() << "]";
  LOG(INFO) << "[ONEREC DEBUG]   K weight shape: ["
            << at_weight_tensors_[IN_K_WEIGHT].sizes() << "]";
  LOG(INFO) << "[ONEREC DEBUG]   V weight shape: ["
            << at_weight_tensors_[IN_V_WEIGHT].sizes() << "]";
  */
  // Check if weights were properly loaded (not placeholders)
  bool q_loaded = !(at_weight_tensors_[IN_Q_WEIGHT].sizes().size() == 2 &&
                    at_weight_tensors_[IN_Q_WEIGHT].sizes()[0] == 1);
  bool k_loaded = !(at_weight_tensors_[IN_K_WEIGHT].sizes().size() == 2 &&
                    at_weight_tensors_[IN_K_WEIGHT].sizes()[0] == 1);
  bool v_loaded = !(at_weight_tensors_[IN_V_WEIGHT].sizes().size() == 2 &&
                    at_weight_tensors_[IN_V_WEIGHT].sizes()[0] == 1);
  /*
  LOG(INFO) << "[ONEREC DEBUG] Weight loading status: Q="
            << (q_loaded ? "loaded" : "placeholder")
            << ", K=" << (k_loaded ? "loaded" : "placeholder")
            << ", V=" << (v_loaded ? "loaded" : "placeholder");
  */
  if (!q_loaded || !k_loaded || !v_loaded) {
    LOG(ERROR)
        << "[ONEREC ERROR] QKV weights not properly loaded. This will cause "
           "SplitOperation to fail.";
    LOG(ERROR)
        << "[ONEREC ERROR] Expected weight shapes should be [hidden_size, "
           "hidden_size] but got placeholders [1, hidden_size]";
    LOG(ERROR)
        << "[ONEREC ERROR] Please check if the weight names in StateDict "
           "match the expected mappings.";

    // For debugging purposes, let's create dummy weights with correct
    // dimensions This is a temporary workaround to prevent the SplitOperation
    // error
    int hidden_size = at_weight_tensors_[IN_Q_WEIGHT].sizes()[1];
    int head_dim = hidden_size / 4;   // Assuming 4 heads as per default config
    int expected_dim = 4 * head_dim;  // num_heads * head_dim

    LOG(WARNING) << "[ONEREC WARNING] Creating dummy weights with correct "
                    "dimensions as workaround";
    LOG(WARNING) << "[ONEREC WARNING] hidden_size=" << hidden_size
                 << ", expected_dim=" << expected_dim;

    if (!q_loaded) {
      at_weight_tensors_[IN_Q_WEIGHT] =
          torch::randn({expected_dim, hidden_size}).to(device_).to(dtype_) *
          0.02;
    }
    if (!k_loaded) {
      at_weight_tensors_[IN_K_WEIGHT] =
          torch::randn({expected_dim, hidden_size}).to(device_).to(dtype_) *
          0.02;
    }
    if (!v_loaded) {
      at_weight_tensors_[IN_V_WEIGHT] =
          torch::randn({expected_dim, hidden_size}).to(device_).to(dtype_) *
          0.02;
    }
  }

  // Merge Q, K, V weights for self-attention
  auto new_q_weight = torch::cat({at_weight_tensors_[IN_Q_WEIGHT],
                                  at_weight_tensors_[IN_K_WEIGHT],
                                  at_weight_tensors_[IN_V_WEIGHT]},
                                 0);
  /*
  LOG(INFO) << "[ONEREC DEBUG] After merging QKV weights:";
  LOG(INFO) << "[ONEREC DEBUG]   Merged Q weight shape: [" <<
  new_q_weight.sizes()
            << "]";
  */

  at_weight_tensors_[IN_Q_WEIGHT] = new_q_weight;
  at_weight_tensors_[IN_K_WEIGHT] =
      torch::zeros({1, at_weight_tensors_[IN_Q_WEIGHT].size(1)})
          .to(device_)
          .to(dtype_);
  at_weight_tensors_[IN_V_WEIGHT] =
      torch::zeros({1, at_weight_tensors_[IN_Q_WEIGHT].size(1)})
          .to(device_)
          .to(dtype_);

  // For decoder, also merge cross-attention Q, K, V weights
  /*if (is_decoder_) {
    auto new_cross_q_weight =
        torch::cat({at_weight_tensors_[IN_CROSS_Q_WEIGHT],
                    at_weight_tensors_[IN_CROSS_K_WEIGHT],
                    at_weight_tensors_[IN_CROSS_V_WEIGHT]},
                   0);
    at_weight_tensors_[IN_CROSS_Q_WEIGHT] = new_cross_q_weight;
    at_weight_tensors_[IN_CROSS_K_WEIGHT] =
        torch::zeros({1, at_weight_tensors_[IN_CROSS_Q_WEIGHT].size(1)})
            .to(device_)
            .to(dtype_);
    at_weight_tensors_[IN_CROSS_V_WEIGHT] =
        torch::zeros({1, at_weight_tensors_[IN_CROSS_Q_WEIGHT].size(1)})
            .to(device_)
            .to(dtype_);
  }*/

  // For MoE mode, skip traditional MLP weight merging
  if (!prefill_param_.use_moe) {
    // Merge wi_0 and wi_1 weights for gated activation (gate_up weight pack)
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
    // MoE mode: Merge expert weights similar to Qwen3 and DeepseekV2
    LOG(INFO) << "[ONEREC DEBUG] MoE mode: merging expert weights";

    // Call merge_experts_weights to process the loaded expert weights
    merge_experts_weights();

    // Merge shared expert weights if they exist
    merge_shared_experts_weights();

    if (at_weight_tensors_[IN_MOE_EXPERT_W1_WEIGHT].numel() > 1) {
      LOG(INFO) << "[ONEREC DEBUG] Expert W1 weights shape: ["
                << at_weight_tensors_[IN_MOE_EXPERT_W1_WEIGHT].sizes() << "]";
    }
    if (at_weight_tensors_[IN_MOE_EXPERT_W2_WEIGHT].numel() > 1) {
      LOG(INFO) << "[ONEREC DEBUG] Expert W2 weights shape: ["
                << at_weight_tensors_[IN_MOE_EXPERT_W2_WEIGHT].sizes() << "]";
    }
    if (at_weight_tensors_[IN_MOE_EXPERT_W3_WEIGHT].numel() > 1) {
      LOG(INFO) << "[ONEREC DEBUG] Expert W3 weights shape: ["
                << at_weight_tensors_[IN_MOE_EXPERT_W3_WEIGHT].sizes() << "]";
    }

    // Log shared expert weights if they exist
    if (at_weight_tensors_[IN_MOE_SHARED_W1_WEIGHT].numel() > 1) {
      LOG(INFO) << "[ONEREC DEBUG] Shared Expert W1 weights shape: ["
                << at_weight_tensors_[IN_MOE_SHARED_W1_WEIGHT].sizes() << "]";
    }
    if (at_weight_tensors_[IN_MOE_SHARED_W2_WEIGHT].numel() > 1) {
      LOG(INFO) << "[ONEREC DEBUG] Shared Expert W2 weights shape: ["
                << at_weight_tensors_[IN_MOE_SHARED_W2_WEIGHT].sizes() << "]";
    }
  }

  // Ensure all placeholder tensors have valid deviceData for ATB compatibility
  int fixed_placeholders = 0;
  const uint64_t weight_count = prefill_param_.use_moe
                                    ? ONEREC_MOE_WEIGHT_COUNT_PER_LAYER
                                    : ONEREC_WEIGHT_COUNT_PER_LAYER;
  for (int i = 0; i < weight_count; ++i) {
    // First check if tensor is defined (not null)
    if (!at_weight_tensors_[i].defined()) {
      // Create a minimal placeholder tensor for undefined tensors
      at_weight_tensors_[i] = torch::zeros(
          {1, 1}, torch::TensorOptions().device(device_).dtype(dtype_));
      fixed_placeholders++;
      continue;
    }

    auto sizes = at_weight_tensors_[i].sizes();
    if (sizes.size() == 2 && sizes[0] == 1) {
      // Check if tensor has valid device data
      if (!at_weight_tensors_[i].is_contiguous() ||
          at_weight_tensors_[i].data_ptr() == nullptr) {
        // Force allocation of device memory for placeholder tensors
        at_weight_tensors_[i] =
            torch::ones({1, sizes[1]},
                        torch::TensorOptions().device(device_).dtype(dtype_));
        fixed_placeholders++;
      }
    }
  }
  // if (fixed_placeholders > 0) {
  //   LOG(INFO) << "[ONEREC DEBUG] Fixed " << fixed_placeholders
  //             << " placeholder tensors with invalid deviceData";
  // }

  c10_npu::NPUCachingAllocator::emptyCache();
  // Choose correct weight count based on use_moe
  for (int i = 0; i < weight_count; ++i) {
    atb_weight_tensors_[i] =
        atb_speed::Utils::AtTensor2Tensor(at_weight_tensors_[i]);
  }
  // LOG(INFO) << "ONERECBlockLayerImpl begin init_layer: " << layer_id_ << ":"
  //           << is_decoder_;
  init_layer();
}

void NpuOneRecBlockLayerImpl::load_state_dict(const StateDict& state_dict) {
  // Choose appropriate weight mapping based on use_moe
  const auto& weight_mapping =
      [this]() -> const std::unordered_map<std::string, int>& {
    if (prefill_param_.use_moe) {
      return ONEREC_DECODER_MOE_WEIGHT_MAPPING;
    } else {
      return is_decoder_ ? ONEREC_DECODER_WEIGHT_MAPPING
                         : ONEREC_ENCODER_WEIGHT_MAPPING;
    }
  }();

  // Debug: Print all available weights in StateDict
  LOG(INFO) << "[ONEREC DEBUG] Available weights in StateDict for "
            << (is_decoder_ ? "decoder" : "encoder") << ":";
  for (const auto& [key, tensor] : state_dict) {
    LOG(INFO) << "[ONEREC DEBUG]   " << key << " -> shape: [" << tensor.sizes()
              << "]";
  }

  // Debug: Print expected weight mappings
  LOG(INFO) << "[ONEREC DEBUG] Expected weight mappings:";
  for (const auto& [name, index] : weight_mapping) {
    LOG(INFO) << "[ONEREC DEBUG]   " << name << " -> index: " << index;
  }

  // Debug: Check actual weight name matching
  LOG(INFO) << "[ONEREC DEBUG] Checking weight name matching:";
  for (const auto& [state_key, tensor] : state_dict) {
    for (const auto& [mapping_name, index] : weight_mapping) {
      if (absl::EndsWith(state_key, mapping_name)) {
        LOG(INFO) << "[ONEREC DEBUG] MATCH: " << state_key << " matches "
                  << mapping_name;
      }
    }
  }

  // Handle MoE expert weights separately if using MoE
  if (prefill_param_.use_moe) {
    // Process each expert weight in the state dict
    for (const auto& [state_key, tensor] : state_dict) {
      if (state_key.find(".ffn.experts.") != std::string::npos) {
        process_expert_weights(state_dict, state_key, tensor);
      }
    }

    // Handle shared expert weights if present
    for (const auto& [state_key, tensor] : state_dict) {
      // Check for shared expert patterns
      bool is_shared_expert =
          (state_key.find(".ffn.shared_experts.") != std::string::npos ||
           state_key.find(".ffn.shared_expert.") != std::string::npos);

      if (is_shared_expert) {
        // Process shared expert weights using the dedicated function
        process_shared_expert_weights(state_dict, state_key, tensor);

        // Handle down_proj weights for ATB compatibility
        if (state_key.find(".down_proj.weight") != std::string::npos ||
            state_key.find(".w2.weight") != std::string::npos) {
          at_weight_tensors_[IN_MLP_DOWN_WEIGHT_SHARED_EXPERT] = tensor;
          LOG(INFO)
              << "[ONEREC DEBUG] Also stored shared expert down weight in "
                 "at_weight_tensors_, shape: ["
              << tensor.sizes() << "]";
        }
      }

      // // Handle shared expert gate weights
      // if (state_key.find(".shared_expert_gate.weight") != std::string::npos)
      // {
      //   at_weight_tensors_[IN_SHARED_EXPERT_GATE_WEIGHT] = tensor;
      //   LOG(INFO) << "[ONEREC DEBUG] Loaded shared expert gate weight, shape:
      //   ["
      //             << tensor.sizes() << "]";
      // }
    }
  }

  for (const auto& [name, index] : weight_mapping) {
    LOG(INFO) << "[ONEREC DEBUG] Loading weight: " << name << " (index "
              << index << ")" << ", " << at_weight_tensors_.size();
    auto initial_shape = at_weight_tensors_[index].sizes();

    // Special handling for relative_attention_bias - it's optional and only
    // exists in first layer
    bool is_relative_bias = (index == IN_RELATIVE_ATTENTION_BIAS_WEIGHT);
    bool weight_exists = false;

    // Check if the weight actually exists in state_dict
    for (const auto& [state_key, tensor] : state_dict) {
      if (absl::EndsWith(state_key, name)) {
        weight_exists = true;
        break;
      }
    }

    if (is_relative_bias && !weight_exists) {
      LOG(INFO) << "[ONEREC DEBUG] Weight " << name << " (index " << index
                << ") SKIPPED: not present in this layer (expected for "
                   "non-first layers)";
      continue;
    }

    if (ONEREC_WEIGHT_SHARD.find(index) != ONEREC_WEIGHT_SHARD.end()) {
      set_weight(state_dict, name, index, ONEREC_WEIGHT_SHARD[index]);
    } else {
      set_weight(state_dict, name, index);
    }
    auto final_shape = at_weight_tensors_[index].sizes();

    // Debug: Check if weight was actually loaded
    bool was_loaded = !(final_shape.size() == 2 && final_shape[0] == 1 &&
                        initial_shape == final_shape);
    LOG(INFO) << "[ONEREC DEBUG] Weight " << name << " (index " << index
              << ") loaded: " << (was_loaded ? "YES" : "NO") << ", shape: ["
              << final_shape << "]" << ", weight exists: " << weight_exists;
  }
  LOG(INFO) << "ONERECBlockLayerImpl end load state dict";
}

int64_t NpuOneRecBlockLayerImpl::init_layer() {
  // init_attn_mask();
  name_ =
      is_decoder_ ? "onerec_decoder_block_layer" : "onerec_encoder_block_layer";
  modelName_ = "onerec";
  LOG(INFO) << "begin init prefill param: " << prefill_param_.isPrefill
            << " is_decoder_: " << is_decoder_ << " layer_id_: " << layer_id_;
  CHECK_OPERATION_STATUS_RETURN(init_node(prefill_node_, prefill_param_));
  // LOG(INFO) << "after init prefill param: " << decode_param_.isPrefill
  //           << " is_decoder_: " << is_decoder_;
  // For ONEREC decoder, only use prefill_node_ for both prefill and decode
  // stages if (is_decoder_) {
  //   LOG(INFO) << "begin init decode param: " << decode_param_.isPrefill;

  //   // Initialize decoder_prefill_only_decode_node if
  //   enable_onerec_prefill_only is
  //   // true
  //   if (FLAGS_enable_rec_prefill_only) {
  //     LOG(INFO) << "begin init decoder_prefill_only_decode_param";
  //     CHECK_OPERATION_STATUS_RETURN(
  //         init_node(decoder_prefill_only_decode_node_,
  //                   decoder_prefill_only_decode_param_));
  //   } else {
  //     CHECK_OPERATION_STATUS_RETURN(init_node(decode_node_, decode_param_));
  //   }
  // }
  return atb::NO_ERROR;
}

int64_t NpuOneRecBlockLayerImpl::init_attn_mask() {
  // attn_mask is now preprocessed in ONERECStack, no local initialization
  // needed
  return atb::NO_ERROR;
}

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

  uint32_t inputNum = node.operation->GetInputNum();
  uint32_t outputNum = node.operation->GetOutputNum();

  // Debug logging for ONEREC tensor count issue
  LOG(INFO) << "[ONEREC DEBUG] " << modelName_
            << " - ATB operation inputNum: " << inputNum
            << ", outputNum: " << outputNum << ", is_decoder: " << is_decoder_;

  if (inputNum < 1) {
    LOG(ERROR) << "Invalid input number: " << inputNum;
    return -1;
  }

  // For ONEREC encoder, we need at least 84 input tensors (79 weights + 5
  // non-weights from GetONERECEncoderTensorNames()) For ONEREC decoder, we need
  // even more tensors
  uint32_t required_tensors;
  if (is_decoder_) {
    // For decoder: check if using MoE variant
    if (param.use_moe) {
      required_tensors =
          ONEREC_MOE_WEIGHT_COUNT_PER_LAYER + 18;  // 97 + 4 + 14 = 115
    } else {
      required_tensors = ONEREC_WEIGHT_COUNT_PER_LAYER + 14;  // 79 + 14 = 93
    }
  } else {
    required_tensors = 84;  // Encoder: 79 weights + 5 non-weights
  }
  if (inputNum < required_tensors) {
    LOG(WARNING) << "[ONEREC DEBUG] " << modelName_
                 << " - ATB operation provides only " << inputNum
                 << " input tensors, but we need at least " << required_tensors
                 << " tensors. This may cause out_of_range errors.";
  }

  node.inTensors.resize(inputNum);
  node.outTensors.resize(outputNum);

  // Set weight tensors
  const uint64_t weight_count = prefill_param_.use_moe
                                    ? ONEREC_MOE_WEIGHT_COUNT_PER_LAYER
                                    : ONEREC_WEIGHT_COUNT_PER_LAYER;
  for (size_t weightTensorId = 0; weightTensorId < weight_count;
       ++weightTensorId) {
    if (weightTensorId < inputNum) {
      node.inTensors.at(weightTensorId) = &atb_weight_tensors_[weightTensorId];
    }
  }

  node.variantPack.inTensors.reserve(inputNum);
  node.variantPack.inTensors.resize(inputNum);
  node.variantPack.outTensors.reserve(outputNum);
  node.variantPack.outTensors.resize(outputNum);

  return atb::NO_ERROR;
}

torch::Tensor NpuOneRecBlockLayerImpl::forward(
    torch::Tensor& x,
    torch::Tensor& attn_mask,
    KVCache& kv_cache,
    ModelInputParams& input_params,
    atb::Context* context,
    AtbWorkspace& workspace,
    std::vector<aclrtEvent*> event,
    std::vector<std::atomic<bool>*> event_flag,
    torch::Tensor* encoder_output,
    int node_id,
    const torch::Tensor& expert_array) {
  atb::Status st;

  // Update BlockLayerParam with current ModelInputParams

  //@TODO: delete
  // prefill_param_.inputParams = &input_params;
  // decode_param_.inputParams = &input_params;

  if (input_params.rec_params && input_params.rec_params->rec_stage ==
                                     RecModelInputParams::RecStage::PREFILL) {
    // Prefill stage
    if (is_decoder_) {
      if (FLAGS_enable_rec_prefill_only) {
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
            << modelName_ << " execute prefill layer fail, error code: " << st;
      }
    } else {
      // Encoder prefill
      build_encoder_node_variant_pack(
          prefill_node_, x, attn_mask, input_params, true, node_id);
      st = execute_node(prefill_node_, node_id, event, event_flag);
      LOG_IF(FATAL, st != 0)
          << modelName_ << " execute prefill layer fail, error code: " << st;
    }
  } else {
    // Decode stage
    if (is_decoder_) {
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
      LOG_IF(FATAL, st != 0)
          << modelName_ << " execute decode layer fail, error code: " << st;
    } else {
      LOG(FATAL) << modelName_ << " encoder decode stage is not supported.";
    }
  }

  return at_placeholder;
}

void NpuOneRecBlockLayerImpl::build_encoder_node_variant_pack(
    atb_speed::Model::Node& node,
    torch::Tensor& x,
    at::Tensor& attn_mask,
    ModelInputParams& input_params,
    bool is_prefill,
    int layer_id) {
  // ONEREC Encoder uses simplified tensor list, corresponding to
  // onerec_encoder_input configuration onerec_encoder_input: {"in_input",
  // "in_attention_mask", "in_seq_len", "in_token_offset", "in_layer_id"} Total:
  // 79 weights + 5 non-weights = 84 tensors

  internalTensors = atb_speed::Utils::AtTensor2Tensor(x);

  // Debug logging for tensor array sizes
  /*
  LOG(INFO) << "[ONEREC DEBUG] build_encoder_node_variant_pack - "
               "variantPack.inTensors.size(): "
            << node.variantPack.inTensors.size()
            << ", ONEREC_WEIGHT_COUNT_PER_LAYER: " <<
  ONEREC_WEIGHT_COUNT_PER_LAYER;
  */

  // Weight tensors (indices 0-78)
  for (size_t i = 0; i < ONEREC_WEIGHT_COUNT_PER_LAYER; ++i) {
    CHECK_THROW(node.inTensors.at(i) == nullptr,
                modelName_ << "inTensor " << i << "is NULL");
    node.variantPack.inTensors.at(i) = *node.inTensors.at(i);
  }

  // Non-weight tensor index definitions - based on onerec_encoder_input
  // configuration
  const int INPUT_TENSOR_IDX = ONEREC_WEIGHT_COUNT_PER_LAYER;  // 79: "in_input"
  const int ATTENTION_MASK_IDX =
      INPUT_TENSOR_IDX + 1;  // 80: "in_attention_mask"
  const int TOKEN_OFFSET_IDX = ATTENTION_MASK_IDX + 1;  // 81: "in_token_offset"
  const int LAYER_ID_IDX = TOKEN_OFFSET_IDX + 1;        // 82: "in_layer_id"
  const int SEQ_LEN_IDX = LAYER_ID_IDX + 1;             // 83: "in_seq_len"

  // "in_input" - Critical tensor
  node.variantPack.inTensors.at(INPUT_TENSOR_IDX) = internalTensors;

  // "in_attention_mask" - Critical tensor (now pre-processed in ONERECStack)
  // attn_mask is already contiguous and on correct device from ONERECStack
  // preprocessing
  node.variantPack.inTensors.at(ATTENTION_MASK_IDX) =
      atb_speed::Utils::AtTensor2Tensor(attn_mask);

  // "in_token_offset" - Set to placeholder
  node.variantPack.inTensors.at(TOKEN_OFFSET_IDX) = placeholder;
  node.variantPack.inTensors.at(TOKEN_OFFSET_IDX).hostData =
      placeholder_vec_.data();

  // "in_layer_id" - Set to placeholder
  node.variantPack.inTensors.at(LAYER_ID_IDX) = placeholder;
  node.variantPack.inTensors.at(LAYER_ID_IDX).hostData =
      placeholder_vec_.data();

  // "in_seq_len" - Important tensor (now pre-processed in ONERECStack)
  if (input_params.rec_params &&
      input_params.rec_params->encoder_seq_lens_tensor.defined()) {
    // encoder_seq_lens_tensor is already contiguous and on correct device from
    // ONERECStack preprocessing
    node.variantPack.inTensors.at(SEQ_LEN_IDX) =
        atb_speed::Utils::AtTensor2Tensor(
            input_params.rec_params->encoder_seq_lens_tensor);
    node.variantPack.inTensors.at(SEQ_LEN_IDX).hostData =
        input_params.rec_params->encoder_seq_lens.data();
  } else if (input_params.rec_params) {
    // Use placeholder to avoid tensor creation and sync
    node.variantPack.inTensors.at(SEQ_LEN_IDX) = placeholder;
    node.variantPack.inTensors.at(SEQ_LEN_IDX).hostData =
        input_params.rec_params->encoder_seq_lens.data();
  }

  // Set output tensor
  node.variantPack.outTensors.at(0) = internalTensors;
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
  // ONEREC Decoder MoE tensor mapping - must match the complete tensor list
  // from ConstructTensorMap for MoE configuration

  // Copy all weight tensors from node.inTensors to variantPack.inTensors
  // For MoE, use ONEREC_MOE_WEIGHT_COUNT_PER_LAYER (74) instead of
  // ONEREC_WEIGHT_COUNT_PER_LAYER (79)
  for (size_t i = 0; i < ONEREC_MOE_WEIGHT_COUNT_PER_LAYER; ++i) {
    CHECK_THROW(node.inTensors.at(i) == nullptr,
                modelName_ << "inTensor " << i << " is NULL");
    node.variantPack.inTensors.at(i) = *node.inTensors.at(i);
  }

  // Configure non-weight tensors starting from index
  // ONEREC_MOE_WEIGHT_COUNT_PER_LAYER (74) Must match the exact order from
  // GetONERECLayerInTensorCandidates in block_layer.cpp
  // Start after weights and MoE routing tensors (expert_array, expert_group,
  // one_hot, zero_hot)
  setup_common_decoder_tensors(node,
                               x,
                               attn_mask,
                               input_params,
                               encoder_output,
                               ONEREC_MOE_WEIGHT_COUNT_PER_LAYER + 4);

  // Add MoE-specific tensors (expert_array, expert_group, one_hot, zero_hot)
  // These tensors are required by ONEREC MoE kernel implementation
  // They should directly follow the weight tensors as defined in kernel
  int moe_tensor_start =
      ONEREC_MOE_WEIGHT_COUNT_PER_LAYER;  // Directly after weights

  // Set expert_array tensor
  if (expert_array.defined()) {
    node.variantPack.inTensors.at(moe_tensor_start) =
        atb_speed::Utils::AtTensor2Tensor(expert_array);
  }

  // Set expert_group_, one_hot_, zero_hot_ tensors
  if (expert_group_.defined()) {
    node.variantPack.inTensors.at(moe_tensor_start + 1) =
        atb_speed::Utils::AtTensor2Tensor(expert_group_);
  }
  if (one_hot_.defined()) {
    node.variantPack.inTensors.at(moe_tensor_start + 2) =
        atb_speed::Utils::AtTensor2Tensor(one_hot_);
  }
  if (zero_hot_.defined()) {
    node.variantPack.inTensors.at(moe_tensor_start + 3) =
        atb_speed::Utils::AtTensor2Tensor(zero_hot_);
  }
}

// Private helper function to set common decoder tensors
int NpuOneRecBlockLayerImpl::setup_common_decoder_tensors(
    atb_speed::Model::Node& node,
    torch::Tensor& x,
    at::Tensor& attn_mask,
    ModelInputParams& input_params,
    torch::Tensor* encoder_output,
    int start_tensor_idx) {
  // Create internal tensor from input
  internalTensors = atb_speed::Utils::AtTensor2Tensor(x);

  int idx = start_tensor_idx;

  // Input tensor
  node.variantPack.inTensors.at(idx++) = internalTensors;

  // Attention mask
  node.variantPack.inTensors.at(idx++) =
      atb_speed::Utils::AtTensor2Tensor(attn_mask);

  // KV cache placeholders
  node.variantPack.inTensors.at(idx++) = placeholder;
  node.variantPack.inTensors.at(idx++) = placeholder;

  // Sequence length
  if (input_params.kv_seq_lens.defined()) {
    node.variantPack.inTensors.at(idx) =
        atb_speed::Utils::AtTensor2Tensor(input_params.kv_seq_lens);
    node.variantPack.inTensors.at(idx).hostData =
        input_params.kv_seq_lens_vec.data();
  } else {
    int32_t seq_len = std::max(static_cast<int32_t>(x.size(0)), 1);
    seq_lens_vec_ = {seq_len};
    auto seq_lens_tensor = torch::tensor(
        seq_lens_vec_,
        torch::TensorOptions().dtype(torch::kInt32).device(device_));
    node.variantPack.inTensors.at(idx) =
        atb_speed::Utils::AtTensor2Tensor(seq_lens_tensor);
    node.variantPack.inTensors.at(idx).hostData = seq_lens_vec_.data();
  }
  idx++;

  // Token offset and layer id placeholders
  node.variantPack.inTensors.at(idx) = placeholder;
  node.variantPack.inTensors.at(idx++).hostData = placeholder_vec_.data();
  node.variantPack.inTensors.at(idx) = placeholder;
  node.variantPack.inTensors.at(idx++).hostData = placeholder_vec_.data();

  // Block tables
  if (!FLAGS_enable_rec_prefill_only && input_params.block_tables.defined()) {
    node.variantPack.inTensors.at(idx) =
        atb_speed::Utils::AtTensor2Tensor(input_params.block_tables);
  } else {
    node.variantPack.inTensors.at(idx) = placeholder;
    node.variantPack.inTensors.at(idx).hostData = placeholder_vec_.data();
  }
  idx++;

  // Cache slots
  if (!FLAGS_enable_rec_prefill_only &&
      input_params.new_cache_slots.defined()) {
    node.variantPack.inTensors.at(idx) =
        atb_speed::Utils::AtTensor2Tensor(input_params.new_cache_slots);
  } else {
    node.variantPack.inTensors.at(idx) = placeholder;
    node.variantPack.inTensors.at(idx).hostData = placeholder_vec_.data();
  }
  idx++;

  // Encoder output
  if (encoder_output != nullptr) {
    encoder_output_contiguous_ = encoder_output->is_contiguous()
                                     ? *encoder_output
                                     : encoder_output->contiguous();
    node.variantPack.inTensors.at(idx) =
        atb_speed::Utils::AtTensor2Tensor(encoder_output_contiguous_);
  } else {
    node.variantPack.inTensors.at(idx) = placeholder;
  }
  idx++;

  // Cross attention placeholders
  for (int i = 0; i < 3; i++) {
    node.variantPack.inTensors.at(idx) = placeholder;
    node.variantPack.inTensors.at(idx++).hostData = placeholder_vec_.data();
  }

  // Encoder sequence length
  if (input_params.rec_params) {
    node.variantPack.inTensors.at(idx) = atb_speed::Utils::AtTensor2Tensor(
        input_params.rec_params->encoder_seq_lens_tensor);
    node.variantPack.inTensors.at(idx++).hostData =
        input_params.rec_params->encoder_seq_lens.data();
  }

  // Setup output tensor
  node.variantPack.outTensors.at(0) = internalTensors;
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
  // ONEREC Decoder tensor mapping - must match the complete tensor list from
  // ConstructTensorMap The operation expects all tensors configured by
  // ConstructTensorMap, including optional features

  // Copy all weight tensors from node.inTensors to variantPack.inTensors
  // The first 79 positions are for weight tensors - use same approach as
  // encoder
  for (size_t i = 0; i < ONEREC_WEIGHT_COUNT_PER_LAYER; ++i) {
    CHECK_THROW(node.inTensors.at(i) == nullptr,
                modelName_ << "inTensor " << i << " is NULL");
    node.variantPack.inTensors.at(i) = *node.inTensors.at(i);
  }

  // Use common tensor setup function for shared logic
  // All common tensors (including encoder_output and cross-attention tensors)
  // are handled here
  int tensor_idx = setup_common_decoder_tensors(node,
                                                x,
                                                attn_mask,
                                                input_params,
                                                encoder_output,
                                                ONEREC_WEIGHT_COUNT_PER_LAYER);

  // Fill remaining tensors with placeholders (for optional features like lora,
  // kv_quant, etc.)
  // Record the number of filled placeholders
  int placeholder_count = 0;
  while (tensor_idx < node.variantPack.inTensors.size()) {
    node.variantPack.inTensors.at(tensor_idx) = placeholder;
    node.variantPack.inTensors.at(tensor_idx).hostData =
        placeholder_vec_.data();
    tensor_idx++;
    placeholder_count++;
  }
  /*
  LOG(INFO) << "[ONEREC DEBUG] total fill " << placeholder_count << "
  placeholders "
            << tensor_idx << ":" << node.variantPack.inTensors.size();
  */

  // Final validation: Check for tensors without deviceData
  int invalid_tensors = 0;
  for (size_t i = 0; i < node.variantPack.inTensors.size(); ++i) {
    const auto& tensor = node.variantPack.inTensors.at(i);
    if (!tensor.deviceData) {
      LOG(ERROR) << "Input tensor[" << i << "] has no deviceData!";
      invalid_tensors++;
    }
  }

  if (invalid_tensors > 0) {
    LOG(ERROR)
        << "Found " << invalid_tensors
        << " tensors without deviceData, this may cause ATB setup to fail.";
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

  // Initialize quantization weights if needed
  experts_weights_["gate_proj.weight_offset"] =
      std::vector<torch::Tensor>(num_of_device_experts);
  experts_weights_["up_proj.weight_offset"] =
      std::vector<torch::Tensor>(num_of_device_experts);
  experts_weights_["down_proj.weight_offset"] =
      std::vector<torch::Tensor>(num_of_device_experts);
  experts_weights_["gate_proj.weight_scale"] =
      std::vector<torch::Tensor>(num_of_device_experts);
  experts_weights_["up_proj.weight_scale"] =
      std::vector<torch::Tensor>(num_of_device_experts);
  experts_weights_["down_proj.weight_scale"] =
      std::vector<torch::Tensor>(num_of_device_experts);
}

void NpuOneRecBlockLayerImpl::process_expert_weights(
    const StateDict& state_dict,
    const std::string& name,
    const torch::Tensor& tensor) {
  std::lock_guard<std::mutex> lock(experts_mutex_);

  int expert_id = extract_expert_index(name);
  if (expert_id < 0) {
    return;
  }

  // Calculate local expert index (similar to Qwen3 and DeepSeek V2)
  const int local_index = expert_id % num_experts_per_partition_;

  std::string weight_suffix = extract_endswith(name);

  // Map ONEREC weight names to standard names and use 2D indexing
  std::string suffix;
  if (weight_suffix == "gate_proj.weight" || weight_suffix == "w1.weight") {
    suffix = "gate_proj.weight";
  } else if (weight_suffix == "up_proj.weight" ||
             weight_suffix == "w3.weight") {
    suffix = "up_proj.weight";
  } else if (weight_suffix == "down_proj.weight" ||
             weight_suffix == "w2.weight") {
    suffix = "down_proj.weight";
  } else if (weight_suffix == "gate_proj.weight_offset" ||
             weight_suffix == "w1.weight_offset") {
    suffix = "gate_proj.weight_offset";
  } else if (weight_suffix == "up_proj.weight_offset" ||
             weight_suffix == "w3.weight_offset") {
    suffix = "up_proj.weight_offset";
  } else if (weight_suffix == "down_proj.weight_offset" ||
             weight_suffix == "w2.weight_offset") {
    suffix = "down_proj.weight_offset";
  } else if (weight_suffix == "gate_proj.weight_scale" ||
             weight_suffix == "w1.weight_scale") {
    suffix = "gate_proj.weight_scale";
  } else if (weight_suffix == "up_proj.weight_scale" ||
             weight_suffix == "w3.weight_scale") {
    suffix = "up_proj.weight_scale";
  } else if (weight_suffix == "down_proj.weight_scale" ||
             weight_suffix == "w2.weight_scale") {
    suffix = "down_proj.weight_scale";
  } else {
    LOG(WARNING) << "[ONEREC WARNING] Unknown expert weight suffix: "
                 << weight_suffix;
    return;
  }

  // Use 2D indexing like Qwen3 and DeepSeek V2
  experts_weights_[suffix][local_index] = tensor.clone();

  LOG(INFO) << "[ONEREC DEBUG] Stored expert " << expert_id
            << " (local_index: " << local_index << ") weight: " << suffix
            << ", shape: [" << tensor.sizes() << "]";
}

void NpuOneRecBlockLayerImpl::process_shared_expert_weights(
    const StateDict& state_dict,
    const std::string& name,
    const torch::Tensor& tensor) {
  LOG(INFO) << "[ONEREC DEBUG] Processing shared expert weight: " << name
            << ", shape: [" << tensor.sizes() << "]";

  torch::Tensor tmp_tensor = tensor.to(device_);

  // Determine which shared expert weight this is
  if (absl::StrContains(name, "gate_proj") || absl::StrContains(name, "w1")) {
    shared_expert_gate_weights_.push_back(tmp_tensor);
    LOG(INFO) << "[ONEREC DEBUG] Added shared expert gate weight, total: "
              << shared_expert_gate_weights_.size();
  } else if (absl::StrContains(name, "up_proj") ||
             absl::StrContains(name, "w3")) {
    shared_expert_up_weights_.push_back(tmp_tensor);
    LOG(INFO) << "[ONEREC DEBUG] Added shared expert up weight, total: "
              << shared_expert_up_weights_.size();
  } else if (absl::StrContains(name, "down_proj") ||
             absl::StrContains(name, "w2")) {
    shared_expert_down_weights_.push_back(tmp_tensor);
    LOG(INFO) << "[ONEREC DEBUG] Added shared expert down weight, total: "
              << shared_expert_down_weights_.size();
  } else {
    LOG(WARNING) << "[ONEREC WARNING] Unknown shared expert weight type: "
                 << name;
  }
}

int NpuOneRecBlockLayerImpl::extract_expert_index(const std::string& name) {
  // Extract expert index from patterns like "experts.0.w1" or "experts.15.w2"
  size_t experts_pos = name.find(".experts.");
  if (experts_pos == std::string::npos) {
    return -1;
  }

  size_t start_pos = experts_pos + 9;  // length of ".experts."
  size_t end_pos = name.find(".", start_pos);
  if (end_pos == std::string::npos) {
    return -1;
  }

  try {
    return std::stoi(name.substr(start_pos, end_pos - start_pos));
  } catch (const std::exception& e) {
    LOG(WARNING) << "[ONEREC DEBUG] Failed to extract expert index from: "
                 << name;
    return -1;
  }
}

std::string NpuOneRecBlockLayerImpl::extract_endswith(
    const std::string& input) {
  // Find the last occurrence of "experts.{number}."
  size_t experts_pos = input.find(".experts.");
  if (experts_pos == std::string::npos) {
    return "";
  }

  // Find the next dot after experts.{number}
  size_t start_pos = experts_pos + 9;  // length of ".experts."
  size_t next_dot = input.find(".", start_pos);
  if (next_dot == std::string::npos) {
    return "";
  }

  // Extract everything after "experts.{number}."
  return input.substr(next_dot + 1);
}

// Implementation of merge_experts_weights functions - reference from Qwen3
void NpuOneRecBlockLayerImpl::merge_experts_weights() {
  LOG(INFO) << "[ONEREC DEBUG] merge_experts_weights begin";

  // Check if required weights exist
  if (experts_weights_.count("gate_proj.weight") == 0 ||
      experts_weights_.count("up_proj.weight") == 0 ||
      experts_weights_.count("down_proj.weight") == 0) {
    LOG(WARNING)
        << "[ONEREC DEBUG] Missing required expert weights, skipping merge";
    return;
  }

  LOG(INFO) << "[ONEREC DEBUG] merge gate_proj "
            << experts_weights_["gate_proj.weight"].size()
            << " and up_proj weights: "
            << experts_weights_["up_proj.weight"].size();
  try {
    // Convert 2D experts_weights_ to 1D vectors for merging
    std::vector<torch::Tensor> gate_weights_1d;
    std::vector<torch::Tensor> up_weights_1d;

    // Extract valid tensors from 2D structure
    for (const auto& tensor : experts_weights_["gate_proj.weight"]) {
      if (tensor.defined()) {
        gate_weights_1d.push_back(tensor);
      }
    }
    for (const auto& tensor : experts_weights_["up_proj.weight"]) {
      if (tensor.defined()) {
        up_weights_1d.push_back(tensor);
      }
    }

    LOG(INFO) << "[ONEREC DEBUG] Extracted " << gate_weights_1d.size()
              << " gate weights and " << up_weights_1d.size() << " up weights";

    torch::Tensor mlp_gateup_weight;
    if (quantize_type_.compare("w8a8_dynamic") == 0) {
      LOG(INFO) << "w8a8_dynamic";
      mlp_gateup_weight = merge_experts_weights(
          gate_weights_1d, up_weights_1d, /*transpose=*/true);

      if (experts_weights_.count("gate_proj.weight_offset") > 0 &&
          experts_weights_.count("up_proj.weight_offset") > 0) {
        std::vector<torch::Tensor> gate_offset_1d, up_offset_1d;
        for (const auto& tensor : experts_weights_["gate_proj.weight_offset"]) {
          if (tensor.defined()) gate_offset_1d.push_back(tensor);
        }
        for (const auto& tensor : experts_weights_["up_proj.weight_offset"]) {
          if (tensor.defined()) up_offset_1d.push_back(tensor);
        }
        at_weight_tensors_[IN_MOE_EXPERT_W1_WEIGHT] =
            merge_experts_weights(gate_offset_1d, up_offset_1d);
      }

      if (experts_weights_.count("gate_proj.weight_scale") > 0 &&
          experts_weights_.count("up_proj.weight_scale") > 0) {
        std::vector<torch::Tensor> gate_scale_1d, up_scale_1d;
        for (const auto& tensor : experts_weights_["gate_proj.weight_scale"]) {
          if (tensor.defined()) gate_scale_1d.push_back(tensor);
        }
        for (const auto& tensor : experts_weights_["up_proj.weight_scale"]) {
          if (tensor.defined()) up_scale_1d.push_back(tensor);
        }
        at_weight_tensors_[IN_MOE_EXPERT_W3_WEIGHT] =
            merge_experts_weights(gate_scale_1d, up_scale_1d);
      }
    } else {
      LOG(INFO) << "w8a8_static";
      mlp_gateup_weight = merge_experts_weights(
          gate_weights_1d, up_weights_1d, /*transpose=*/false);
    }
    at_weight_tensors_[IN_MOE_EXPERT_W1_WEIGHT] =
        at_npu::native::npu_format_cast(mlp_gateup_weight, 2).contiguous();
  } catch (const std::exception& e) {
    LOG(ERROR) << "[ERROR] Exception in gateup weight processing: " << e.what();
    throw;
  }

  LOG(INFO) << "[ONEREC DEBUG] merge down_proj "
            << experts_weights_["down_proj.weight"].size();
  try {
    // Convert 2D down_proj weights to 1D vector
    std::vector<torch::Tensor> down_weights_1d;
    for (const auto& tensor : experts_weights_["down_proj.weight"]) {
      if (tensor.defined()) {
        down_weights_1d.push_back(tensor);
      }
    }

    LOG(INFO) << "[ONEREC DEBUG] Extracted " << down_weights_1d.size()
              << " down weights";

    torch::Tensor mlp_down_weight =
        merge_experts_weights(down_weights_1d, /*transpose=*/false);

    at_weight_tensors_[IN_MOE_EXPERT_W2_WEIGHT] =
        at_npu::native::npu_format_cast(mlp_down_weight, 2).contiguous();

    if (quantize_type_.compare("w8a8_dynamic") == 0) {
      if (experts_weights_.count("down_proj.weight_offset") > 0) {
        std::vector<torch::Tensor> down_offset_1d;
        for (const auto& tensor : experts_weights_["down_proj.weight_offset"]) {
          if (tensor.defined()) down_offset_1d.push_back(tensor);
        }
        // Use a different tensor index for offset
        at_weight_tensors_[IN_MLP_DOWN_OFFSET_EXPERT] =
            merge_experts_weights(down_offset_1d);
      }
      if (experts_weights_.count("down_proj.weight_scale") > 0) {
        std::vector<torch::Tensor> down_scale_1d;
        for (const auto& tensor : experts_weights_["down_proj.weight_scale"]) {
          if (tensor.defined()) down_scale_1d.push_back(tensor);
        }
        // Use a different tensor index for scale
        at_weight_tensors_[IN_MLP_DOWN_SCALE_EXPERT] =
            merge_experts_weights(down_scale_1d);
      }
    }
  } catch (const std::exception& e) {
    LOG(ERROR) << "[ERROR] Exception in down weight processing: " << e.what();
    throw;
  }
  LOG(INFO) << "[ONEREC DEBUG] end merge_experts_weights()";
}

torch::Tensor NpuOneRecBlockLayerImpl::merge_experts_weights(
    std::vector<torch::Tensor>& experts,
    bool transpose) {
  LOG(INFO) << "[ONEREC DEBUG] merge_experts_weights, experts size: "
            << experts.size();
  torch::Tensor merged_tensor = torch::stack(experts, 0).to(device_);
  // Bypass torch::stack operation, generate random tensor with correct shape
  // torch::Tensor merged_tensor;
  // if (!experts.empty()) {
  //   // Calculate merged shape based on the first expert's shape
  //   auto expert_shape = experts[0].sizes().vec();
  //   std::vector<int64_t> merged_shape =
  //   {static_cast<int64_t>(experts.size())}; merged_shape.insert(
  //       merged_shape.end(), expert_shape.begin(), expert_shape.end());

  //   // Generate random tensor, maintaining data type and device
  //   merged_tensor = torch::randn(merged_shape,
  //                                torch::TensorOptions()
  //                                    .dtype(experts[0].dtype())
  //                                    .device(experts[0].device()))
  //                       .to(device_);
  //   LOG(INFO) << "[ONEREC DEBUG] Generated random merged tensor with shape: "
  //             << merged_tensor.sizes();
  // }

  if (transpose) {
    merged_tensor = merged_tensor.transpose(1, 2);
  }
  merged_tensor = merged_tensor.contiguous();
  experts.clear();
  LOG(INFO) << "[ONEREC DEBUG] merge_experts_weights, return tensor size: "
            << merged_tensor.sizes();
  return merged_tensor;
}

torch::Tensor NpuOneRecBlockLayerImpl::merge_experts_weights(
    std::vector<torch::Tensor>& experts_gate,
    std::vector<torch::Tensor>& experts_up,
    bool transpose) {
  LOG(INFO) << "[ONEREC DEBUG] merge_experts_weights, gate size: "
            << experts_gate.size() << " up size: " << experts_up.size();
  for (size_t i = 0; i < experts_up.size(); ++i) {
    experts_gate[i] = torch::cat({experts_gate[i], experts_up[i]}, 0);
  }
  torch::Tensor merged_tensor = torch::stack(experts_gate, 0).to(device_);
  // Bypass memory issues: directly generate a random tensor with correct shape
  // torch::Tensor merged_tensor;

  // if (!experts_gate.empty() && !experts_up.empty()) {
  //   // Calculate the correct merged shape
  //   auto gate_sizes = experts_gate[0].sizes();
  //   auto up_sizes = experts_up[0].sizes();

  //   // Calculate concatenated dimension 0 size (gate + up)
  //   int64_t concat_dim0 = gate_sizes[0] + up_sizes[0];

  //   // Build final tensor shape: [num_experts, concat_dim0, other_dims...]
  //   std::vector<int64_t> final_shape;
  //   final_shape.push_back(experts_gate.size());  // num_experts
  //   final_shape.push_back(concat_dim0);          // dimension 0 of gate + up
  //   for (int i = 1; i < gate_sizes.size(); ++i) {
  //     final_shape.push_back(gate_sizes[i]);
  //   }

  //   // Generate random tensor with correct data type and device
  //   auto dtype = experts_gate[0].dtype();
  //   merged_tensor = torch::randn(
  //       final_shape, torch::TensorOptions().dtype(dtype).device(device_));
  // }

  LOG(INFO) << "[ONEREC DEBUG] Generated random tensor with shape: "
            << merged_tensor.sizes();

  if (transpose) {
    merged_tensor = merged_tensor.transpose(1, 2);
  }
  merged_tensor = merged_tensor.contiguous();
  experts_gate.clear();
  experts_up.clear();
  LOG(INFO) << "[ONEREC DEBUG] merge_experts_weights, return tensor size: "
            << merged_tensor.sizes();
  return merged_tensor;
}

void NpuOneRecBlockLayerImpl::merge_shared_experts_weights() {
  LOG(INFO) << "[ONEREC DEBUG] merge_shared_experts_weights called";

  // Check if we have shared expert weights to merge
  if (shared_expert_gate_weights_.empty() &&
      shared_expert_up_weights_.empty() &&
      shared_expert_down_weights_.empty()) {
    LOG(INFO) << "[ONEREC DEBUG] No shared expert weights to merge";
    return;
  }

  // Merge shared expert gate and up weights (similar to regular experts)
  if (!shared_expert_gate_weights_.empty() &&
      !shared_expert_up_weights_.empty()) {
    LOG(INFO) << "[ONEREC DEBUG] Merging shared expert gate and up weights, "
                 "gate size: "
              << shared_expert_gate_weights_.size()
              << ", up size: " << shared_expert_up_weights_.size();

    // Concatenate gate and up weights for shared expert
    auto merged_gate_up = merge_experts_weights(
        shared_expert_gate_weights_, shared_expert_up_weights_, false);
    at_weight_tensors_[IN_MLP_GATEUP_WEIGHT_SHARED_EXPERT] = merged_gate_up;

    LOG(INFO) << "[ONEREC DEBUG] Shared expert gate+up merged tensor shape: "
              << merged_gate_up.sizes();
  } else if (!shared_expert_gate_weights_.empty()) {
    // Only gate weights available
    auto merged_gate =
        merge_experts_weights(shared_expert_gate_weights_, false);
    at_weight_tensors_[IN_MLP_GATEUP_WEIGHT_SHARED_EXPERT] = merged_gate;

    LOG(INFO) << "[ONEREC DEBUG] Shared expert gate merged tensor shape: "
              << merged_gate.sizes();
  }

  // Merge shared expert down weights
  if (!shared_expert_down_weights_.empty()) {
    LOG(INFO) << "[ONEREC DEBUG] Merging shared expert down weights, size: "
              << shared_expert_down_weights_.size();

    auto merged_down =
        merge_experts_weights(shared_expert_down_weights_, false);
    at_weight_tensors_[IN_MLP_DOWN_WEIGHT_SHARED_EXPERT] = merged_down;

    LOG(INFO) << "[ONEREC DEBUG] Shared expert down merged tensor shape: "
              << merged_down.sizes();
  }

  // Clear the temporary storage vectors
  shared_expert_gate_weights_.clear();
  shared_expert_up_weights_.clear();
  shared_expert_down_weights_.clear();

  LOG(INFO) << "[ONEREC DEBUG] merge_shared_experts_weights completed";
}
}  // namespace layer
}  // namespace xllm
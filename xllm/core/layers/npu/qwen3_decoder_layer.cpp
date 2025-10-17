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

#include "qwen3_decoder_layer.h"

#include <glog/logging.h>
#include <mstx/ms_tools_ext.h>

#include <map>

#include "attn_mask.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUException.h"

DECLARE_string(communication_backend);
DECLARE_bool(enable_chunked_prefill);
DECLARE_bool(enable_prefix_cache);
DECLARE_int32(block_size);

namespace xllm::hf {

enum DecoderLayerTensorId : int {
  IN_NORM_WEIGHT = 0,      // weight
  IN_NORM_BIAS = 1,        // bias
  IN_NORM_NEW_WEIGHT = 2,  // new weight
  IN_NORM_NEW_BIAS = 3,    // new bias

  IN_Q_WEIGHT = 4,    // weight
  IN_Q_BIAS = 5,      // bias
  IN_Q_DEQSCALE = 6,  // deq_scale
  IN_Q_OFFSET = 7,    // offset
  IN_Q_SCALE = 8,     // scale
  IN_Q_COMPRESS_IDX = 9,

  IN_K_WEIGHT = 10,    // weight
  IN_K_BIAS = 11,      // bias
  IN_K_DEQSCALE = 12,  // deq_scale
  IN_K_OFFSET = 13,    // offset
  IN_K_SCALE = 14,     // scale
  IN_K_COMPRESS_IDX = 15,

  IN_V_WEIGHT = 16,    // weight
  IN_V_BIAS = 17,      // bias
  IN_V_DEQSCALE = 18,  // deq_scale
  IN_V_OFFSET = 19,    // offset
  IN_V_SCALE = 20,     // scale
  IN_V_COMPRESS_IDX = 21,

  IN_ATTENTION_OUT_WEIGHT = 22,    // weight
  IN_ATTENTION_OUT_BIAS = 23,      // bias
  IN_ATTENTION_OUT_DEQSCALE = 24,  // deq_scale
  IN_ATTENTION_OUT_OFFSET = 25,    // offset
  IN_ATTENTION_OUT_SCALE = 26,     // scale
  IN_ATTENTION_OUT_COMPRESS_IDX = 27,

  IN_SELFOUT_NORM_WEIGHT = 28,      // weight
  IN_SELFOUT_NORM_BIAS = 29,        // bias
  IN_SELFOUT_NORM_NEW_WEIGHT = 30,  // new weight
  IN_SELFOUT_NORM_NEW_BIAS = 31,    // new bias

  IN_MLP_W2_WEIGHT = 32,    // weight
  IN_MLP_W2_BIAS = 33,      // bias
  IN_MLP_W2_DEQSCALE = 34,  // deq_scale
  IN_MLP_W2_OFFSET = 35,    // offset
  IN_MLP_W2_SCALE = 36,     // scale
  IN_MLP_W2_COMPRESS_IDX = 37,

  IN_MLP_W1_WEIGHT = 38,    // weight
  IN_MLP_W1_BIAS = 39,      // bias
  IN_MLP_W1_DEQSCALE = 40,  // deq_scale
  IN_MLP_W1_OFFSET = 41,    // offset
  IN_MLP_W1_SCALE = 42,     // scale
  IN_MLP_W1_COMPRESS_IDX = 43,

  IN_MLP_CPROJ_WEIGHT = 44,    // weight
  IN_MLP_CPROJ_BIAS = 45,      // bias
  IN_MLP_CPROJ_DEQSCALE = 46,  // deq_scale
  IN_MLP_CPROJ_OFFSET = 47,    // offset
  IN_MLP_CPROJ_SCALE = 48,     // scale
  IN_MLP_CPROJ_COMPRESS_IDX = 49,

  Q_NORM_WEIGHT = 50,
  K_NORM_WEIGHT = 51,
};

const uint64_t WEIGHT_COUNT_PER_LAYER = 52;

static std::vector<std::pair<int, std::string>> WEIGHT_MAPPING = {
    {IN_NORM_WEIGHT, "input_layernorm.weight"},
    {IN_Q_WEIGHT, "self_attn.q_proj.weight"},
    {IN_K_WEIGHT, "self_attn.k_proj.weight"},
    {IN_V_WEIGHT, "self_attn.v_proj.weight"},
    {IN_ATTENTION_OUT_WEIGHT, "self_attn.o_proj.weight"},
    {IN_SELFOUT_NORM_WEIGHT, "post_attention_layernorm.weight"},
    {IN_MLP_W2_WEIGHT, "mlp.gate_proj.weight"},
    {IN_MLP_W1_WEIGHT, "mlp.up_proj.weight"},
    {IN_MLP_CPROJ_WEIGHT, "mlp.down_proj.weight"},
    {Q_NORM_WEIGHT, "self_attn.q_norm.weight"},
    {K_NORM_WEIGHT, "self_attn.k_norm.weight"}};

static std::vector<std::pair<int, std::string>> WEIGHT_MAPPING_W8A8 = {
    {IN_NORM_WEIGHT, "input_layernorm.weight"},
    {IN_Q_WEIGHT, "self_attn.q_proj.weight"},
    {IN_Q_BIAS, "self_attn.q_proj.quant_bias"},
    {IN_Q_DEQSCALE, "self_attn.q_proj.deq_scale"},
    {IN_Q_OFFSET, "self_attn.q_proj.input_offset"},
    {IN_Q_SCALE, "self_attn.q_proj.input_scale"},
    {IN_K_WEIGHT, "self_attn.k_proj.weight"},
    {IN_K_BIAS, "self_attn.k_proj.quant_bias"},
    {IN_K_DEQSCALE, "self_attn.k_proj.deq_scale"},
    {IN_K_OFFSET, "self_attn.k_proj.input_offset"},
    {IN_K_SCALE, "self_attn.k_proj.input_scale"},
    {IN_V_WEIGHT, "self_attn.v_proj.weight"},
    {IN_V_BIAS, "self_attn.v_proj.quant_bias"},
    {IN_V_DEQSCALE, "self_attn.v_proj.deq_scale"},
    {IN_V_OFFSET, "self_attn.v_proj.input_offset"},
    {IN_V_SCALE, "self_attn.v_proj.input_scale"},
    {IN_ATTENTION_OUT_WEIGHT, "self_attn.o_proj.weight"},
    {IN_ATTENTION_OUT_BIAS, "self_attn.o_proj.quant_bias"},
    {IN_ATTENTION_OUT_DEQSCALE, "self_attn.o_proj.deq_scale"},
    {IN_ATTENTION_OUT_OFFSET, "self_attn.o_proj.input_offset"},
    {IN_ATTENTION_OUT_SCALE, "self_attn.o_proj.input_scale"},
    {IN_SELFOUT_NORM_WEIGHT, "post_attention_layernorm.weight"},
    {IN_MLP_W2_WEIGHT, "mlp.gate_proj.weight"},
    {IN_MLP_W2_BIAS, "mlp.gate_proj.quant_bias"},
    {IN_MLP_W2_DEQSCALE, "mlp.gate_proj.deq_scale"},
    {IN_MLP_W2_OFFSET, "mlp.gate_proj.input_offset"},
    {IN_MLP_W2_SCALE, "mlp.gate_proj.input_scale"},
    {IN_MLP_W1_WEIGHT, "mlp.up_proj.weight"},
    {IN_MLP_W1_BIAS, "mlp.up_proj.quant_bias"},
    {IN_MLP_W1_DEQSCALE, "mlp.up_proj.deq_scale"},
    {IN_MLP_W1_OFFSET, "mlp.up_proj.input_offset"},
    {IN_MLP_W1_SCALE, "mlp.up_proj.input_scale"},
    {IN_MLP_CPROJ_WEIGHT, "mlp.down_proj.weight"},
    {Q_NORM_WEIGHT, "self_attn.q_norm.weight"},
    {K_NORM_WEIGHT, "self_attn.k_norm.weight"}};

static std::map<int, int> WEIGHT_SHARD = {{IN_Q_WEIGHT, 0},
                                          {IN_K_WEIGHT, 0},
                                          {IN_V_WEIGHT, 0},
                                          {IN_ATTENTION_OUT_WEIGHT, 1},
                                          {IN_MLP_W2_WEIGHT, 0},
                                          {IN_MLP_W1_WEIGHT, 0},
                                          {IN_MLP_CPROJ_WEIGHT, 1}};

static std::map<int, int> WEIGHT_SHARD_W8A8 = {{IN_Q_WEIGHT, 0},
                                               {IN_Q_BIAS, 0},
                                               {IN_Q_DEQSCALE, 0},
                                               {IN_K_WEIGHT, 0},
                                               {IN_K_BIAS, 0},
                                               {IN_K_DEQSCALE, 0},
                                               {IN_V_WEIGHT, 0},
                                               {IN_V_BIAS, 0},
                                               {IN_V_DEQSCALE, 0},
                                               {IN_ATTENTION_OUT_WEIGHT, 1},
                                               {IN_MLP_W2_WEIGHT, 0},
                                               {IN_MLP_W2_BIAS, 0},
                                               {IN_MLP_W2_DEQSCALE, 0},
                                               {IN_MLP_W1_WEIGHT, 0},
                                               {IN_MLP_W1_BIAS, 0},
                                               {IN_MLP_W1_DEQSCALE, 0},
                                               {IN_MLP_CPROJ_WEIGHT, 1}};

std::shared_ptr<Qwen3DecoderImpl> create_qwen3_decode_layer(
    const ModelContext& context) {
  return std::make_shared<Qwen3DecoderImpl>(context);
}

void Qwen3DecoderImpl::param_from_args(atb_speed::qwen::QwenLayerParam& param,
                                       const ModelArgs& args,
                                       const ParallelArgs& parallel_args,
                                       bool isPrefill) {
  param.isFA = false;
  // Enable SwiGLU activation, as used in LLaMA
  param.enableSwiGLU = true;
  // Enable LCOC for prefill phase, similar to LLaMA
  // NOTE: Currently, single-process startup requires setting enableLcoc to
  // false, which leads to performance degradation. param.enableLcoc = false;
  // //isPrefill
  param.enableLcoc = isPrefill;
  param.rmsnormQKNorm = true;
  param.isPrefill = isPrefill;
  param.isBF16 = args.dtype() == "bfloat16";
  param.enableSplitFuse = FLAGS_enable_chunked_prefill && isPrefill;
  param.loraEnableGMM = false;

  param.linearTransposeType = {1, -1, -1, 1, 1, -1, 1};
  param.quantGroupSize = 0;
  param.normEps = args.rms_norm_eps();
  param.numAttentionHeadsPerRank = args.n_heads() / parallel_args.world_size();
  param.hiddenSizePerAttentionHead = args.head_dim();
  std::optional<long int> optionalValue = args.n_kv_heads();
  param.numKeyValueHeadsPerRank =
      static_cast<int>(optionalValue.value()) / parallel_args.world_size();
  param.backend = FLAGS_communication_backend;
  param.enableLogN = false;
  param.tensorParallelInfo = {parallel_args.rank(),
                              parallel_args.world_size(),
                              FLAGS_communication_backend};
  param.linearHasBias = {0, 0, 0, 0};
  param.useQKNorm = true;

  param.numHiddenLayers = args.n_layers();

  initialize_quantization_parameters(param);

  if (isPrefill) {
    param.enableAclnnRmsNorm = quantize_type_.empty();
    // for prefix cache without chunked prefill.
    if (FLAGS_enable_prefix_cache && !FLAGS_enable_chunked_prefill &&
        FLAGS_block_size != 128) {
      LOG(ERROR) << "try to enable prefix cache without chunked prefill but "
                    "failed, because the block_size is required to be 128.";
    }
    param.isPrefixCacheWithoutChunk = FLAGS_enable_prefix_cache &&
                                      !FLAGS_enable_chunked_prefill &&
                                      FLAGS_block_size == 128;
  }
}

void Qwen3DecoderImpl::initialize_quantization_parameters(
    atb_speed::qwen::QwenLayerParam& param) {
  if (quantize_type_.empty()) {
    param.linearDescs = {static_cast<int>(LinearTypeV2::BFLOAT16),
                         static_cast<int>(LinearTypeV2::INVALID),
                         static_cast<int>(LinearTypeV2::INVALID),
                         static_cast<int>(LinearTypeV2::BFLOAT16),
                         static_cast<int>(LinearTypeV2::BFLOAT16),
                         static_cast<int>(LinearTypeV2::INVALID),
                         static_cast<int>(LinearTypeV2::BFLOAT16)};
    param.packQuantType = {static_cast<int>(PackType::PACK_QUANT_UNDEFINED),
                           static_cast<int>(PackType::PACK_QUANT_UNDEFINED)};
    param.linearQuantType = {static_cast<int>(LinearType::INVALID),
                             static_cast<int>(LinearType::INVALID),
                             static_cast<int>(LinearType::INVALID),
                             static_cast<int>(LinearType::INVALID),
                             static_cast<int>(LinearType::INVALID),
                             static_cast<int>(LinearType::INVALID),
                             static_cast<int>(LinearType::INVALID)};
  } else {
    param.linearDescs = {static_cast<int>(LinearTypeV2::W8A8),
                         static_cast<int>(LinearTypeV2::INVALID),
                         static_cast<int>(LinearTypeV2::INVALID),
                         static_cast<int>(LinearTypeV2::W8A8),
                         static_cast<int>(LinearTypeV2::W8A8),
                         static_cast<int>(LinearTypeV2::INVALID),
                         static_cast<int>(LinearTypeV2::BFLOAT16)};
    param.packQuantType = {static_cast<int>(PackType::ALL_W8A8),
                           static_cast<int>(PackType::ALL_W8A8)};
    param.linearQuantType = {static_cast<int>(LinearType::INT),
                             static_cast<int>(LinearType::INVALID),
                             static_cast<int>(LinearType::INVALID),
                             static_cast<int>(LinearType::INT),
                             static_cast<int>(LinearType::INT),
                             static_cast<int>(LinearType::INVALID),
                             static_cast<int>(LinearType::FP)};
  }
}

Qwen3DecoderImpl::Qwen3DecoderImpl(const ModelContext& context)
    : ATBBase(context) {
  auto model_args = context.get_model_args();
  auto parallel_args = context.get_parallel_args();
  auto options = context.get_tensor_options();

  param_from_args(prefill_param_, model_args, parallel_args, true);
  param_from_args(decode_param_, model_args, parallel_args, false);
  at_weight_tensors_.resize(WEIGHT_COUNT_PER_LAYER);
  atb_weight_tensors_.resize(WEIGHT_COUNT_PER_LAYER);
  placeholder_vec_ = {1};
  dtype_ = c10::typeMetaToScalarType(options.dtype());
  rank_id_ = parallel_args.rank();
  prefill_tensor_storage_.resize(4);
  decode_tensor_storage_.resize(4);
  prefill_vector_storage_.resize(1);
  decode_vector_storage_.resize(1);
  placeholder_ = atb_speed::Utils::AtTensor2Tensor(
      torch::zeros({1}).to(device_).to(dtype_));
  at_placeholder_ = torch::zeros({1}).to(device_).to(dtype_);
  for (int i = 0; i < WEIGHT_COUNT_PER_LAYER; ++i) {
    at_weight_tensors_[i] = torch::zeros({1}).to(options);
  }
}

void Qwen3DecoderImpl::verify_loaded_weights() const {
  for (const auto& [index, name] : WEIGHT_MAPPING) {
    CHECK(at_weight_tensors_[index].sizes() != std::vector<int64_t>({1}))
        << "weight is not loaded for " << name;
  }
}

void Qwen3DecoderImpl::merge_loaded_weights() {
  if (quantize_type_.compare("w8a8") == 0) {
    at_weight_tensors_[IN_ATTENTION_OUT_DEQSCALE] =
        at_weight_tensors_[IN_ATTENTION_OUT_DEQSCALE].to(torch::kFloat32);
    at_weight_tensors_[IN_Q_DEQSCALE] =
        torch::cat({at_weight_tensors_[IN_Q_DEQSCALE],
                    at_weight_tensors_[IN_K_DEQSCALE],
                    at_weight_tensors_[IN_V_DEQSCALE]},
                   0)
            .to(torch::kFloat32);

    at_weight_tensors_[IN_Q_BIAS] = torch::cat({at_weight_tensors_[IN_Q_BIAS],
                                                at_weight_tensors_[IN_K_BIAS],
                                                at_weight_tensors_[IN_V_BIAS]},
                                               0)
                                        .to(torch::kInt32);

    for (auto idx : {IN_K_DEQSCALE,
                     IN_V_DEQSCALE,
                     IN_K_BIAS,
                     IN_V_BIAS,
                     IN_K_OFFSET,
                     IN_V_OFFSET,
                     IN_K_SCALE,
                     IN_V_SCALE}) {
      at_weight_tensors_[idx] = at_placeholder_;
    }

    at_weight_tensors_[IN_MLP_W2_BIAS] =
        torch::cat({at_weight_tensors_[IN_MLP_W2_BIAS],
                    at_weight_tensors_[IN_MLP_W1_BIAS]},
                   0);

    at_weight_tensors_[IN_MLP_W2_DEQSCALE] =
        torch::cat({at_weight_tensors_[IN_MLP_W2_DEQSCALE],
                    at_weight_tensors_[IN_MLP_W1_DEQSCALE]},
                   0)
            .to(torch::kFloat32);

    for (auto idx : {IN_MLP_W1_BIAS,
                     IN_MLP_W1_OFFSET,
                     IN_MLP_W1_SCALE,
                     IN_MLP_W1_DEQSCALE}) {
      at_weight_tensors_[idx] = at_placeholder_;
    }

    at_weight_tensors_[IN_Q_OFFSET] =
        at_weight_tensors_[IN_Q_OFFSET].to(torch::kInt8).to(device_);
    at_weight_tensors_[IN_ATTENTION_OUT_OFFSET] =
        at_weight_tensors_[IN_ATTENTION_OUT_OFFSET]
            .to(torch::kInt8)
            .to(device_);
    at_weight_tensors_[IN_MLP_W2_OFFSET] =
        at_weight_tensors_[IN_MLP_W2_OFFSET].to(torch::kInt8).to(device_);

    if (rank_id_ != 0) {
      torch::Tensor original_tensor = at_weight_tensors_[IN_ATTENTION_OUT_BIAS];
      auto shape = original_tensor.sizes();
      auto dtype = original_tensor.dtype();
      auto device = original_tensor.device();

      at_weight_tensors_[IN_ATTENTION_OUT_BIAS] = torch::zeros(
          shape, torch::TensorOptions().dtype(dtype).device(device));
    }
  }

  at_weight_tensors_[IN_Q_WEIGHT] =
      torch::cat({at_weight_tensors_[IN_Q_WEIGHT],
                  at_weight_tensors_[IN_K_WEIGHT],
                  at_weight_tensors_[IN_V_WEIGHT]},
                 0)
          .contiguous();

  at_weight_tensors_[IN_MLP_W2_WEIGHT] =
      torch::cat({at_weight_tensors_[IN_MLP_W2_WEIGHT],
                  at_weight_tensors_[IN_MLP_W1_WEIGHT]},
                 0)
          .contiguous();

  for (auto idx :
       {IN_MLP_W1_WEIGHT, IN_K_WEIGHT, IN_V_WEIGHT, IN_K_BIAS, IN_V_BIAS}) {
    at_weight_tensors_[idx] = at_placeholder_;
  }

  c10_npu::NPUCachingAllocator::emptyCache();
  for (int i = 0; i < WEIGHT_COUNT_PER_LAYER; ++i) {
    atb_weight_tensors_[i] =
        atb_speed::Utils::AtTensor2Tensor(at_weight_tensors_[i]);
  }

  init_layer();
}

void Qwen3DecoderImpl::load_state_dict(const StateDict& state_dict) {
  if (quantize_type_.compare("w8a8") == 0) {
    for (const auto& [index, name] : WEIGHT_MAPPING_W8A8) {
      if (WEIGHT_SHARD_W8A8.find(index) != WEIGHT_SHARD_W8A8.end()) {
        set_weight(state_dict, name, index, WEIGHT_SHARD_W8A8[index]);
      } else {
        set_weight(state_dict, name, index);
      }
    }
    at_weight_tensors_[IN_NORM_BIAS] =
        torch::zeros(at_weight_tensors_[IN_NORM_WEIGHT].sizes(),
                     at_weight_tensors_[IN_NORM_WEIGHT].options())
            .to(device_);

    at_weight_tensors_[IN_SELFOUT_NORM_BIAS] =
        torch::zeros(at_weight_tensors_[IN_SELFOUT_NORM_WEIGHT].sizes(),
                     at_weight_tensors_[IN_SELFOUT_NORM_WEIGHT].options())
            .to(device_);
    return;
  }

  for (const auto& [index, name] : WEIGHT_MAPPING) {
    if (WEIGHT_SHARD.find(index) != WEIGHT_SHARD.end()) {
      set_weight(state_dict, name, index, WEIGHT_SHARD[index]);
    } else {
      set_weight(state_dict, name, index);
    }
  }
}

int64_t Qwen3DecoderImpl::init_layer() {
  init_attn_mask();
  ATBBase::name_ = "qwen3_decoder_layer";
  model_name_ = "qwen3";
  CHECK_OPERATION_STATUS_RETURN(init_node(prefill_node_, prefill_param_));
  CHECK_OPERATION_STATUS_RETURN(init_node(decode_node_, decode_param_));

  return atb::NO_ERROR;
}

int64_t Qwen3DecoderImpl::init_attn_mask() {
  torch::Dtype dtype =
      prefill_param_.isBF16 ? torch::kBFloat16 : torch::kFloat16;
  // encode_attn_mask_ =
  //     AttentionMaskImpl(device_, dtype).get_attn_mask(2048, dtype, device_);
  decode_attn_mask_ = torch::zeros({1}).to(device_).to(dtype);

  return atb::NO_ERROR;
}

int64_t Qwen3DecoderImpl::init_node(atb_speed::Model::Node& node,
                                    atb_speed::qwen::QwenLayerParam& param) {
  atb::Operation* operation = nullptr;
  atb_speed::qwen::QwenDecoderLayer decoder_layer(param);
  decoder_layer.BuildGraph(&operation);
  node.operation.reset(operation);
  if (node.operation == nullptr) {
    LOG(ERROR) << "node.operation is null";
    return -1;
  }
  if (node.operation->GetInputNum() < 1) {
    LOG(ERROR) << "Can not resize number which is smaller than 1";
    return -1;
  }
  node.inTensors.resize(node.operation->GetInputNum());
  node.outTensors.resize(1);
  size_t inTensorId = 1;

  for (size_t weightTensorId = 0; weightTensorId < WEIGHT_COUNT_PER_LAYER;
       ++weightTensorId) {
    node.inTensors.at(weightTensorId) = &atb_weight_tensors_[weightTensorId];
  }

  node.variantPack.inTensors.reserve(node.inTensors.size());
  node.variantPack.inTensors.resize(node.inTensors.size());
  node.variantPack.outTensors.reserve(1);
  node.variantPack.outTensors.resize(1);

  return atb::NO_ERROR;
}

torch::Tensor Qwen3DecoderImpl::forward(torch::Tensor& x,
                                        torch::Tensor& cos_pos,
                                        torch::Tensor& sin_pos,
                                        torch::Tensor& attn_mask,
                                        KVCache& kv_cache,
                                        ModelInputParams& input_params,
                                        aclrtEvent* event,
                                        std::atomic<bool>* event_flag,
                                        int node_id) {
  // auto tensor = torch::tensor({1}).to(x.device());
  atb::Status st;
  if (input_params.prefill_indices.second !=
      input_params.q_seq_lens.size(0) - 1) {
    // if (input_params.empty_kv_cache) {
    // mstxRangeId id = mstxRangeStartA("prefill build variant", nullptr);
    build_node_variant_pack(prefill_node_,
                            x,
                            cos_pos,
                            sin_pos,
                            attn_mask,
                            kv_cache,
                            input_params,
                            true);
    // mstxRangeEnd(id);
    st = execute_node(prefill_node_, node_id, event, event_flag);
    LOG_IF(FATAL, st != 0) << model_name_
                           << "excute prefill layer fail, error code: " << st;
  } else {
    build_node_variant_pack(decode_node_,
                            x,
                            cos_pos,
                            sin_pos,
                            decode_attn_mask_,
                            kv_cache,
                            input_params,
                            false);
    st = execute_node(decode_node_, node_id + 1000, event, event_flag);
    LOG_IF(FATAL, st != 0) << model_name_
                           << "excute decode layer fail, error code: " << st;
  }

  return at_placeholder_;
}

void Qwen3DecoderImpl::build_node_variant_pack(atb_speed::Model::Node& node,
                                               torch::Tensor& x,
                                               torch::Tensor& cos_pos,
                                               torch::Tensor& sin_pos,
                                               at::Tensor& attn_mask,
                                               KVCache& kv_cache,
                                               ModelInputParams& input_params,
                                               bool is_prefill) {
  internal_tensors_ = atb_speed::Utils::AtTensor2Tensor(x);
  // std::cout<<"node.variantPack.inTensors.size:"<<node.variantPack.inTensors.size()<<std::endl;
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER) = internal_tensors_;
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 1) =
      atb_speed::Utils::AtTensor2Tensor(cos_pos);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 2) =
      atb_speed::Utils::AtTensor2Tensor(sin_pos);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 3) =
      atb_speed::Utils::AtTensor2Tensor(attn_mask);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 4) =
      atb_speed::Utils::AtTensor2Tensor(kv_cache.get_k_cache());
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 5) =
      atb_speed::Utils::AtTensor2Tensor(kv_cache.get_v_cache());
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 6) =
      atb_speed::Utils::AtTensor2Tensor(input_params.kv_seq_lens);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 6).hostData =
      input_params.kv_seq_lens_vec.data();
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 7) = placeholder_;
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 7).hostData =
      placeholder_vec_.data();
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 8) = placeholder_;
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 9) =
      atb_speed::Utils::AtTensor2Tensor(input_params.block_tables);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 10) =
      atb_speed::Utils::AtTensor2Tensor(input_params.new_cache_slots);
  if (is_prefill &&
      (FLAGS_enable_chunked_prefill || FLAGS_enable_prefix_cache)) {
    node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 11) =
        atb_speed::Utils::AtTensor2Tensor(input_params.q_seq_lens);
    node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 11).hostData =
        input_params.q_seq_lens_vec.data();
  }
  for (size_t i = 0; i < WEIGHT_COUNT_PER_LAYER; ++i) {
    CHECK_THROW(node.inTensors.at(i) == nullptr,
                model_name_ << "inTensor " << i << "is NULL");
    node.variantPack.inTensors.at(i) = *node.inTensors.at(i);
    // LOG(INFO) << model_name_ << "inTensors[" << i << "]:"
    //               << atb_speed::TensorUtil::TensorToString(
    //                      node.variantPack.inTensors.at(i));
  }

  node.variantPack.outTensors.at(0) = internal_tensors_;
}

Qwen3Decoder::Qwen3Decoder(const ModelContext& context)
    : ModuleHolder(create_qwen3_decode_layer(context)) {}

}  // namespace xllm::hf

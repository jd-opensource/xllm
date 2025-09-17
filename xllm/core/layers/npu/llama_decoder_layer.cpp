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

#include "llama_decoder_layer.h"

#include <glog/logging.h>
#include <mstx/ms_tools_ext.h>

#include <map>

#include "attn_mask.h"
#include "common/global_flags.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUException.h"

namespace xllm::hf {

const uint64_t WEIGHT_COUNT_PER_LAYER = 50;

enum DecoderLayerTensorId : int {

  IN_NORM_WEIGHT = 0,  // weight
  IN_NORM_BIAS,        // bias
  IN_NORM_NEW_WEIGHT,  // new weight
  IN_NORM_NEW_BIAS,    // new bias

  IN_Q_WEIGHT,    // weight
  IN_Q_BIAS,      // bias
  IN_Q_DEQSCALE,  // deq_scale
  IN_Q_OFFSET,    // offset
  IN_Q_SCALE,     // scale
  IN_Q_COMPRESS_IDX,

  IN_K_WEIGHT,    // weight
  IN_K_BIAS,      // bias
  IN_K_DEQSCALE,  // deq_scale
  IN_K_OFFSET,    // offset
  IN_K_SCALE,     // scale
  IN_K_COMPRESS_IDX,

  IN_V_WEIGHT,    // weight
  IN_V_BIAS,      // bias
  IN_V_DEQSCALE,  // deq_scale
  IN_V_OFFSET,    // offset
  IN_V_SCALE,     // scale
  IN_V_COMPRESS_IDX,

  IN_ATTENTION_OUT_WEIGHT,    // weight
  IN_ATTENTION_OUT_BIAS,      // bias
  IN_ATTENTION_OUT_DEQSCALE,  // deq_scale
  IN_ATTENTION_OUT_OFFSET,    // offset
  IN_ATTENTION_OUT_SCALE,     // scale
  IN_ATTENTION_OUT_COMPRESS_IDX,

  IN_SELFOUT_NORM_WEIGHT,      // weight
  IN_SELFOUT_NORM_BIAS,        // bias
  IN_SELFOUT_NORM_NEW_WEIGHT,  // new weight
  IN_SELFOUT_NORM_NEW_BIAS,    // new bias

  IN_MLP_W2_WEIGHT,    // weight
  IN_MLP_W2_BIAS,      // bias
  IN_MLP_W2_DEQSCALE,  // deq_scale
  IN_MLP_W2_OFFSET,    // offset
  IN_MLP_W2_SCALE,     // scale
  IN_MLP_W2_COMPRESS_IDX,

  IN_MLP_W1_WEIGHT,    // weight
  IN_MLP_W1_BIAS,      // bias
  IN_MLP_W1_DEQSCALE,  // deq_scale
  IN_MLP_W1_OFFSET,    // offset
  IN_MLP_W1_SCALE,     // scale
  IN_MLP_W1_COMPRESS_IDX,

  IN_MLP_CPROJ_WEIGHT,    // weight
  IN_MLP_CPROJ_BIAS,      // bias
  IN_MLP_CPROJ_DEQSCALE,  // deq_scale
  IN_MLP_CPROJ_OFFSET,    // offset
  IN_MLP_CPROJ_SCALE,     // scale
  IN_MLP_CPROJ_COMPRESS_IDX,
};

static const std::unordered_map<std::string, int> WEIGHT_MAPPING = {
    {"input_layernorm.weight", IN_NORM_WEIGHT},
    {"self_attn.q_proj.weight", IN_Q_WEIGHT},
    {"self_attn.k_proj.weight", IN_K_WEIGHT},
    {"self_attn.v_proj.weight", IN_V_WEIGHT},
    {"self_attn.o_proj.weight", IN_ATTENTION_OUT_WEIGHT},
    {"post_attention_layernorm.weight", IN_SELFOUT_NORM_WEIGHT},
    {"mlp.gate_proj.weight", IN_MLP_W2_WEIGHT},
    {"mlp.up_proj.weight", IN_MLP_W1_WEIGHT},
    {"mlp.down_proj.weight", IN_MLP_CPROJ_WEIGHT},
};

static std::map<int, int> WEIGHT_SHARD = {{IN_Q_WEIGHT, 0},
                                          {IN_K_WEIGHT, 0},
                                          {IN_V_WEIGHT, 0},
                                          {IN_ATTENTION_OUT_WEIGHT, 1},
                                          {IN_MLP_W2_WEIGHT, 0},
                                          {IN_MLP_W1_WEIGHT, 0},
                                          {IN_MLP_CPROJ_WEIGHT, 1}};

LlamaDecoderImpl::LlamaDecoderImpl(const ModelContext& context)
    : ATBBase(context) {
  param_from_args(prefill_param_,
                  context.get_model_args(),
                  context.get_parallel_args(),
                  true);
  param_from_args(decode_param_,
                  context.get_model_args(),
                  context.get_parallel_args(),
                  false);

  at_weight_tensors_.resize(WEIGHT_COUNT_PER_LAYER);
  atb_weight_tensors_.resize(WEIGHT_COUNT_PER_LAYER);
  placeholder_vec_ = {1};

  auto options = context.get_tensor_options();
  dtype_ = c10::typeMetaToScalarType(options.dtype());
  device_id_ = options.device().index();
  placeholder_ = atb_speed::Utils::AtTensor2Tensor(
      torch::zeros({1}).to(device_).to(dtype_));
  at_placeholder_ = torch::zeros({1}).to(device_).to(dtype_);
  for (int i = 0; i < WEIGHT_COUNT_PER_LAYER; ++i) {
    at_weight_tensors_[i] = torch::zeros({1}).to(options);
  }
}

// fix param
void LlamaDecoderImpl::param_from_args(atb_speed::llama::LlamaLayerParam& param,
                                       const ModelArgs& args,
                                       const ParallelArgs& parallel_args,
                                       bool isPrefill) {
  param.isFA = false;
  param.isPrefill = isPrefill;
  param.isBF16 = args.dtype() == "bfloat16";
  param.enableSwiGLU = true;
  param.enableLcoc = isPrefill;
  param.enableSpeculate = false;
  param.enableSplitFuse = FLAGS_enable_chunked_prefill && isPrefill;
  param.enableLora = false;
  param.loraEnableGMM = false;
  param.packQuantType = {1, 1};
  param.linearQuantType = {0, -1, -1, 0, 0, -1, 0};
  param.linearTransposeType = {1, -1, -1, 1, 1, -1, 1};
  param.enableKvQuant = false;
  param.quantGroupSize = 0;
  param.normEps = args.rms_norm_eps();
  param.normEps = 0.00001;
  // param.normType = 0;
  param.enableFA3 = false;
  param.worldSize = parallel_args.world_size();
  param.numAttentionHeadsPerRank = args.n_heads() / param.worldSize;
  param.hiddenSizePerAttentionHead = args.hidden_size() / args.n_heads();
  std::optional<long int> optionalValue = args.n_kv_heads();
  param.numKeyValueHeadsPerRank =
      static_cast<int>(optionalValue.value()) / param.worldSize;
  param.rank = parallel_args.rank();
  param.backend = "lccl";
  param.tensorParallelInfo = {
      parallel_args.rank(), parallel_args.world_size(), "lccl"};
  // param.enableLogN = false;
}

void LlamaDecoderImpl::verify_loaded_weights() const {
  for (const auto& [name, index] : WEIGHT_MAPPING) {
    CHECK(at_weight_tensors_[index].sizes() != std::vector<int64_t>({1}))
        << "weight is not loaded for " << name;
  }
}

void LlamaDecoderImpl::merge_loaded_weights() {
  auto new_q_weight = torch::cat({at_weight_tensors_[IN_Q_WEIGHT],
                                  at_weight_tensors_[IN_K_WEIGHT],
                                  at_weight_tensors_[IN_V_WEIGHT]},
                                 0);
  at_weight_tensors_[IN_Q_WEIGHT] = new_q_weight;

  at_weight_tensors_[IN_K_WEIGHT] = torch::zeros({1}).to(device_);
  at_weight_tensors_[IN_V_WEIGHT] = torch::zeros({1}).to(device_);

  auto new_mlp_weight = torch::cat({at_weight_tensors_[IN_MLP_W2_WEIGHT],
                                    at_weight_tensors_[IN_MLP_W1_WEIGHT]},
                                   0);
  at_weight_tensors_[IN_MLP_W2_WEIGHT] = new_mlp_weight;

  at_weight_tensors_[IN_MLP_W1_WEIGHT] = torch::zeros({1}).to(device_);

  c10_npu::NPUCachingAllocator::emptyCache();
  for (int i = 0; i < WEIGHT_COUNT_PER_LAYER; ++i) {
    atb_weight_tensors_[i] =
        atb_speed::Utils::AtTensor2Tensor(at_weight_tensors_[i]);
  }

  init_layer();
}

void LlamaDecoderImpl::load_state_dict(const StateDict& state_dict) {
  for (const auto& [name, index] : WEIGHT_MAPPING) {
    if (WEIGHT_SHARD.find(index) != WEIGHT_SHARD.end()) {
      set_weight(state_dict, name, index, WEIGHT_SHARD[index]);
    } else {
      set_weight(state_dict, name, index);
    }
  }
}

int64_t LlamaDecoderImpl::init_layer() {
  init_attn_mask();
  ATBBase::name_ = "llama_decoder_layer";
  model_name_ = "llama";
  CHECK_OPERATION_STATUS_RETURN(init_node(prefill_node_, prefill_param_));
  CHECK_OPERATION_STATUS_RETURN(init_node(decode_node_, decode_param_));

  return atb::NO_ERROR;
}

int64_t LlamaDecoderImpl::init_attn_mask() {
  torch::Dtype dtype =
      prefill_param_.isBF16 ? torch::kBFloat16 : torch::kFloat16;
  // encode_attn_mask_ =
  //     AttentionMaskImpl(device_, dtype).get_attn_mask(2048, dtype, device_);
  decode_attn_mask_ = torch::zeros({1}).to(device_).to(dtype);

  return atb::NO_ERROR;
}

int64_t LlamaDecoderImpl::init_node(atb_speed::Model::Node& node,
                                    atb_speed::llama::LlamaLayerParam& param) {
  atb::Operation* operation = nullptr;
  atb_speed::llama::LlamaDecoderLayer decoder_layer(param);
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

torch::Tensor LlamaDecoderImpl::forward(torch::Tensor& x,
                                        torch::Tensor& cos_pos,
                                        torch::Tensor& sin_pos,
                                        torch::Tensor& attn_mask,
                                        KVCache& kv_cache,
                                        ModelInputParams& input_params,
                                        int node_id) {
  atb::Status st;

  if (input_params.prefill_indices.second !=
      input_params.q_seq_lens.size(0) - 1) {
    build_node_variant_pack(prefill_node_,
                            x,
                            cos_pos,
                            sin_pos,
                            attn_mask,
                            kv_cache,
                            input_params,
                            true);
    // mstxRangeEnd(id);
    st = execute_node(prefill_node_, node_id);
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
    st = execute_node(decode_node_, node_id + 1000);
    LOG_IF(FATAL, st != 0) << model_name_
                           << "excute decode layer fail, error code: " << st;
  }

  return at_placeholder_;
}

void LlamaDecoderImpl::build_node_variant_pack(atb_speed::Model::Node& node,
                                               torch::Tensor& x,
                                               torch::Tensor& cos_pos,
                                               torch::Tensor& sin_pos,
                                               at::Tensor& attn_mask,
                                               KVCache& kv_cache,
                                               ModelInputParams& input_params,
                                               bool is_prefill) {
  internal_tensors_ = atb_speed::Utils::AtTensor2Tensor(x);
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
  if (is_prefill && FLAGS_enable_chunked_prefill) {
    node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 11) =
        atb_speed::Utils::AtTensor2Tensor(input_params.q_seq_lens);
    node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 11).hostData =
        input_params.q_seq_lens_vec.data();
  }
  for (size_t i = 0; i < WEIGHT_COUNT_PER_LAYER; ++i) {
    CHECK_THROW(node.inTensors.at(i) == nullptr,
                model_name_ << "inTensor " << i << "is NULL");
    node.variantPack.inTensors.at(i) = *node.inTensors.at(i);
  }

  node.variantPack.outTensors.at(0) = internal_tensors_;
}

LlamaDecoder::LlamaDecoder(const ModelContext& context)
    : ModuleHolder(create_llama_decode_layer(context)) {}

std::shared_ptr<LlamaDecoderImpl> create_llama_decode_layer(
    const ModelContext& context) {
  return std::make_shared<LlamaDecoderImpl>(context);
}

}  // namespace xllm::hf

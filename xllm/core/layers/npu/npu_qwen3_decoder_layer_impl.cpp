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

#include "npu_qwen3_decoder_layer_impl.h"

#include <glog/logging.h>
#include <mstx/ms_tools_ext.h>

#include <map>

#include "common/global_flags.h"

// #include "attn_mask.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUException.h"

namespace xllm {
namespace layer {

const uint64_t WEIGHT_COUNT_PER_LAYER = 56;

void Qwen3DecoderLayerImpl::param_from_args(
    atb_speed::qwen::QwenLayerParam& param,
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
  param.enableLcoc = false;
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
  param.enableIntraLayerAddNorm = true;
  param.enableInterLayerAddNorm = false;
  param.enablePreFetchWeight = FLAGS_enable_prefetch_weight;
  initialize_parallel_parameters(param, parallel_args);
  initialize_quantization_parameters(param);

  if (isPrefill) {
    param.enableAclnnRmsNorm =
        param.enableIntraLayerAddNorm && quantize_type_.empty()
            ? false
            : quantize_type_.empty();
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

void Qwen3DecoderLayerImpl::initialize_parallel_parameters(
    atb_speed::qwen::QwenLayerParam& param,
    const ParallelArgs& parallel_args) {
  param.mapping = parallel_args.mapping();
  param.tensorParallelInfo = {parallel_args.rank(),
                              parallel_args.world_size(),
                              FLAGS_communication_backend,
                              FLAGS_rank_tablefile,
                              nullptr,
                              ""};
}

void Qwen3DecoderLayerImpl::initialize_quantization_parameters(
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

Qwen3DecoderLayerImpl::Qwen3DecoderLayerImpl(const ModelContext& context)
    : BaseLayer(context) {
  auto model_args = context.get_model_args();
  auto parallel_args = context.get_parallel_args();
  auto options = context.get_tensor_options();

  param_from_args(prefill_param_, model_args, parallel_args, true);
  param_from_args(decode_param_, model_args, parallel_args, false);
  atb_weight_tensors_.resize(WEIGHT_COUNT_PER_LAYER);
  placeholder_vec_ = {1};
  dtype_ = c10::typeMetaToScalarType(options.dtype());
  prefill_tensor_storage_.resize(4);
  decode_tensor_storage_.resize(4);
  prefill_vector_storage_.resize(1);
  decode_vector_storage_.resize(1);
  placeholder_ = atb_speed::Utils::AtTensor2Tensor(
      torch::zeros({1}).to(device_).to(dtype_));
  at_placeholder_ = torch::zeros({1}).to(device_).to(dtype_);
  loader_ = std::make_unique<Qwen3DecoderManualLoader>(
      WEIGHT_COUNT_PER_LAYER,
      context,
      prefill_param_.enableIntraLayerAddNorm ||
          prefill_param_.enableInterLayerAddNorm);
}

void Qwen3DecoderLayerImpl::merge_loaded_weights() {
  loader_->merge_loaded_weights();
  auto& at_weight_tensors = loader_->get_at_weight_tensors();
  c10_npu::NPUCachingAllocator::emptyCache();
  for (int i = 0; i < WEIGHT_COUNT_PER_LAYER; ++i) {
    atb_weight_tensors_[i] =
        atb_speed::Utils::AtTensor2Tensor(at_weight_tensors[i]);
  }

  init_layer();
}

int64_t Qwen3DecoderLayerImpl::init_layer() {
  init_attn_mask();
  name_ = "qwen3_decoder_layer";
  model_name_ = "qwen3";
  CHECK_OPERATION_STATUS_RETURN(init_node(prefill_node_, prefill_param_));
  CHECK_OPERATION_STATUS_RETURN(init_node(decode_node_, decode_param_));

  return atb::NO_ERROR;
}

int64_t Qwen3DecoderLayerImpl::init_attn_mask() {
  torch::Dtype dtype =
      prefill_param_.isBF16 ? torch::kBFloat16 : torch::kFloat16;
  decode_attn_mask_ = torch::zeros({1}).to(device_).to(dtype);

  return atb::NO_ERROR;
}

int64_t Qwen3DecoderLayerImpl::init_node(
    atb_speed::Model::Node& node,
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

torch::Tensor Qwen3DecoderLayerImpl::forward(torch::Tensor& x,
                                             torch::Tensor& cos_pos,
                                             torch::Tensor& sin_pos,
                                             torch::Tensor& attn_mask,
                                             KVCache& kv_cache,
                                             ModelInputParams& input_params,
                                             aclrtEvent* event,
                                             std::atomic<bool>* event_flag,
                                             int node_id) {
  atb::Status st;
  if (!input_params.batch_forward_type.is_decode()) {
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

void Qwen3DecoderLayerImpl::build_node_variant_pack(
    atb_speed::Model::Node& node,
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
  }

  node.variantPack.outTensors.at(0) = internal_tensors_;
}

}  // namespace layer
}  // namespace xllm

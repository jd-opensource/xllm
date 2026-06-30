/* Copyright 2025-2026 The xLLM Authors.

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

#include "core/framework/config/eplb_config.h"
#include "core/framework/config/execution_config.h"
#include "core/framework/config/kernel_config.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/load_config.h"
#include "core/framework/config/parallel_config.h"
#include "core/framework/config/scheduler_config.h"
#include "core/layers/npu/loader/qwen3_decoder_loader.h"
#include "operations/fusion/mlp/mlp.h"
#include "util/rec_model_utils.h"

// #include "attn_mask.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUException.h"

namespace xllm {
namespace layer {

const uint64_t WEIGHT_COUNT_PER_LAYER = 56;
constexpr int64_t FLASH_COMM_VOLUME_THRESHOLD = 1024 * 5120;
constexpr size_t FLASH_COMM_TENSOR_COUNT = 8;
constexpr size_t FLASH_COMM_HOST_TENSOR_COUNT = 6;

void NpuQwen3DecoderLayerImpl::param_from_args(
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
  param.enableSplitFuse =
      ::xllm::SchedulerConfig::get_instance().enable_chunked_prefill() &&
      isPrefill;
  param.loraEnableGMM = false;
  param.enableXattention = is_rec_multi_round_mode();
  const auto& kernel_config = ::xllm::KernelConfig::get_instance();

  param.linearTransposeType = {static_cast<int>(TransposeType::NOT_TRANSPOSE),
                               static_cast<int>(TransposeType::INVALID),
                               static_cast<int>(TransposeType::INVALID),
                               static_cast<int>(TransposeType::NOT_TRANSPOSE),
                               static_cast<int>(TransposeType::NOT_TRANSPOSE),
                               static_cast<int>(TransposeType::INVALID),
                               static_cast<int>(TransposeType::NOT_TRANSPOSE)};
  param.quantGroupSize = 0;
  param.normEps = args.rms_norm_eps();
  param.numAttentionHeadsPerRank = args.n_heads() / parallel_args.world_size();
  param.hiddenSizePerAttentionHead = args.head_dim();
  std::optional<long int> optionalValue = args.n_kv_heads();
  param.numKeyValueHeadsPerRank =
      static_cast<int>(optionalValue.value()) / parallel_args.world_size();
  param.backend =
      ::xllm::ParallelConfig::get_instance().communication_backend();
  param.enableMC2 = kernel_config.enable_fused_mc2() > 0 &&
                    param.backend == "hccl" && quantize_type_.empty();
  if (param.enableMC2) {
    LOG(WARNING) << "currently A3 doesn't support MC2.";
  }
  param.enableLogN = false;
  param.tensorParallelInfo = {
      parallel_args.rank(),
      parallel_args.world_size(),
      ::xllm::ParallelConfig::get_instance().communication_backend()};
  param.linearHasBias = {0, 0, 0, 0};
  param.useQKNorm = true;

  param.numHiddenLayers = args.n_layers();
  param.enableFlashComm = false;
  param.enableFlashCommInterLayer = false;
  param.enableIntraLayerAddNorm = true;
  if (kernel_config.enable_interlayer_addnorm()) {
    param.enableInterLayerAddNorm = true;
  }
  param.enablePreFetchWeight =
      ::xllm::LoadConfig::get_instance().enable_prefetch_weight();
  param.enableAclGraphPagedAttention =
      ::xllm::ExecutionConfig::get_instance().enable_graph() && !isPrefill;
  if (kernel_config.enable_aclnn_matmul()) {
    param.matmulBackend = atb_speed::common::OpBackend::ACLNN;
  }
  if (kernel_config.enable_aclnn_swiglu()) {
    param.swigluBackend = atb_speed::common::OpBackend::ACLNN;
  }
  initialize_parallel_parameters(param, parallel_args);
  initialize_quantization_parameters(param);

  if (isPrefill) {
    param.enableAclnnRmsNorm =
        param.enableIntraLayerAddNorm && quantize_type_.empty()
            ? false
            : quantize_type_.empty();
    // for prefix cache without chunked prefill.
    if (::xllm::KVCacheConfig::get_instance().enable_prefix_cache() &&
        !::xllm::SchedulerConfig::get_instance().enable_chunked_prefill() &&
        ::xllm::KVCacheConfig::get_instance().block_size() != 128) {
      LOG(ERROR) << "try to enable prefix cache without chunked prefill but "
                    "failed, because the block_size is required to be 128.";
    }
    param.isPrefixCacheWithoutChunk =
        ::xllm::KVCacheConfig::get_instance().enable_prefix_cache() &&
        !::xllm::SchedulerConfig::get_instance().enable_chunked_prefill() &&
        ::xllm::KVCacheConfig::get_instance().block_size() == 128;
  }
  num_hidden_layers_ = args.n_layers();
  if (kernel_config.enable_split_rmsnorm_rope()) {
    param.enableSplitRmsNormRope = true;
  }
}

void NpuQwen3DecoderLayerImpl::initialize_parallel_parameters(
    atb_speed::qwen::QwenLayerParam& param,
    const ParallelArgs& parallel_args) {
  param.mapping = parallel_args.mapping();
  param.tensorParallelInfo = {
      parallel_args.rank(),
      parallel_args.world_size(),
      ::xllm::ParallelConfig::get_instance().communication_backend(),
      ::xllm::EPLBConfig::get_instance().rank_tablefile(),
      nullptr,
      ""};
}

void NpuQwen3DecoderLayerImpl::initialize_quantization_parameters(
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

NpuQwen3DecoderLayerImpl::NpuQwen3DecoderLayerImpl(const ModelContext& context)
    : BaseLayer(context) {
  auto model_args = context.get_model_args();
  auto parallel_args = context.get_parallel_args();
  auto options = context.get_tensor_options();

  param_from_args(prefill_param_, model_args, parallel_args, true);
  param_from_args(decode_graph_param_, model_args, parallel_args, false);
  decode_eager_param_ = decode_graph_param_;
  decode_eager_param_.enableAclGraphPagedAttention = false;
  prefill_flash_comm_param_ = prefill_param_;
  decode_graph_flash_comm_param_ = decode_graph_param_;
  decode_eager_flash_comm_param_ = decode_eager_param_;
  auto enable_flash_comm = [](atb_speed::qwen::QwenLayerParam& param) {
    param.enableFlashComm = true;
    param.enableFlashCommInterLayer = false;
    param.enableMC2 = false;
  };
  enable_flash_comm(prefill_flash_comm_param_);
  enable_flash_comm(decode_graph_flash_comm_param_);
  enable_flash_comm(decode_eager_flash_comm_param_);
  atb_weight_tensors_.resize(WEIGHT_COUNT_PER_LAYER);
  placeholder_vec_ = {1};
  dtype_ = c10::typeMetaToScalarType(options.dtype());
  prefill_tensor_storage_.resize(4);
  decode_tensor_storage_.resize(4);
  prefill_flash_comm_tensors_.resize(FLASH_COMM_TENSOR_COUNT);
  decode_flash_comm_tensors_.resize(FLASH_COMM_TENSOR_COUNT);
  prefill_flash_comm_host_.resize(FLASH_COMM_HOST_TENSOR_COUNT);
  decode_flash_comm_host_.resize(FLASH_COMM_HOST_TENSOR_COUNT);
  prefill_vector_storage_.resize(1);
  decode_vector_storage_.resize(1);
  placeholder_ = atb_speed::Utils::AtTensor2Tensor(
      torch::zeros({1}).to(device_).to(dtype_));
  at_placeholder_ = torch::zeros({1}).to(device_).to(dtype_);
  loader_ = std::make_unique<Qwen3DecoderLoader>(
      WEIGHT_COUNT_PER_LAYER,
      context,
      prefill_param_.enableIntraLayerAddNorm ||
          prefill_param_.enableInterLayerAddNorm,
      ::xllm::LoadConfig::get_instance().enable_manual_loader()
          ? LoadMode::kManual
          : LoadMode::kEager);
}

int64_t NpuQwen3DecoderLayerImpl::init_layer() {
  init_attn_mask();
  name_ = "qwen3_decoder_layer";
  model_name_ = "qwen3";

  if (quantize_type_ == "w8a8") {
    Qwen3DecoderLoader* qwen3_loader =
        dynamic_cast<Qwen3DecoderLoader*>(loader_.get());
    if (qwen3_loader && qwen3_loader->down_proj_quantized()) {
      auto update_down_proj = [](atb_speed::qwen::QwenLayerParam& p) {
        p.linearDescs[atb_speed::common::DOWN_LINEAR_INDEX] =
            static_cast<int>(LinearTypeV2::W8A8);
        p.linearQuantType[atb_speed::common::DOWN_LINEAR_INDEX] =
            static_cast<int>(LinearType::INT);
      };
      update_down_proj(prefill_param_);
      update_down_proj(decode_graph_param_);
      update_down_proj(decode_eager_param_);
      update_down_proj(prefill_flash_comm_param_);
      update_down_proj(decode_graph_flash_comm_param_);
      update_down_proj(decode_eager_flash_comm_param_);
    }
  }

  CHECK_OPERATION_STATUS_RETURN(init_node(prefill_node_, prefill_param_));
  CHECK_OPERATION_STATUS_RETURN(
      init_node(decode_graph_node_, decode_graph_param_));
  CHECK_OPERATION_STATUS_RETURN(
      init_node(decode_eager_node_, decode_eager_param_));

  const bool can_init_flash_comm =
      ::xllm::KernelConfig::get_instance().enable_qwen3_flash_comm() &&
      !::xllm::KernelConfig::get_instance().enable_interlayer_addnorm() &&
      prefill_flash_comm_param_.tensorParallelInfo.worldSize > 1 &&
      prefill_flash_comm_param_.backend == "hccl";
  if (can_init_flash_comm &&
      !::xllm::ParallelConfig::get_instance().enable_multi_stream_parallel()) {
    CHECK_OPERATION_STATUS_RETURN(
        init_node(prefill_flash_comm_node_, prefill_flash_comm_param_));
    prefill_flash_comm_initialized_ = true;
  }
  if (can_init_flash_comm) {
    CHECK_OPERATION_STATUS_RETURN(init_node(decode_graph_flash_comm_node_,
                                            decode_graph_flash_comm_param_));
    CHECK_OPERATION_STATUS_RETURN(init_node(decode_eager_flash_comm_node_,
                                            decode_eager_flash_comm_param_));
    decode_flash_comm_initialized_ = true;
  }

  return atb::NO_ERROR;
}

int64_t NpuQwen3DecoderLayerImpl::init_attn_mask() {
  torch::Dtype dtype =
      prefill_param_.isBF16 ? torch::kBFloat16 : torch::kFloat16;
  decode_attn_mask_ = torch::zeros({1}).to(device_).to(dtype);

  return atb::NO_ERROR;
}

int64_t NpuQwen3DecoderLayerImpl::init_node(
    atb_speed::Model::Node& node,
    atb_speed::qwen::QwenLayerParam& param) {
  atb::Operation* operation = nullptr;
  atb_speed::qwen::QwenDecoderLayer decoder_layer(param);
  decoder_layer.BuildGraph(&operation);
  node.operation.reset(operation);
  CHECK_NOTNULL(node.operation);
  CHECK_GT(node.operation->GetInputNum(), 0);
  node.inTensors.resize(node.operation->GetInputNum());
  node.outTensors.resize(node.operation->GetOutputNum());
  size_t inTensorId = 1;

  for (size_t weightTensorId = 0; weightTensorId < WEIGHT_COUNT_PER_LAYER;
       ++weightTensorId) {
    node.inTensors.at(weightTensorId) = &atb_weight_tensors_[weightTensorId];
  }

  node.variantPack.inTensors.resize(node.inTensors.size());
  node.variantPack.outTensors.resize(node.outTensors.size());

  return atb::NO_ERROR;
}

torch::Tensor NpuQwen3DecoderLayerImpl::forward(torch::Tensor& x,
                                                torch::Tensor& cos_pos,
                                                torch::Tensor& sin_pos,
                                                torch::Tensor& attn_mask,
                                                KVCache& kv_cache,
                                                ModelInputParams& input_params,
                                                aclrtEvent* event,
                                                std::atomic<bool>* event_flag,
                                                int node_id) {
  atb::Status st;
  if (!input_params.meta.batch_forward_type.is_decode()) {
    const bool use_flash_comm = should_enable_flash_comm(x, true);
    auto& node = use_flash_comm ? prefill_flash_comm_node_ : prefill_node_;
    build_node_variant_pack(node,
                            x,
                            cos_pos,
                            sin_pos,
                            attn_mask,
                            kv_cache,
                            input_params,
                            /*is_prefill=*/true,
                            node_id,
                            /*use_graph_decode_input=*/false,
                            use_flash_comm);
    // mstxRangeEnd(id);
    st = execute_node(node, node_id, event, event_flag);
    LOG_IF(FATAL, st != 0) << model_name_
                           << "execute prefill layer fail, error code: " << st;
  } else {
    const bool use_graph_decode_input =
        ::xllm::ExecutionConfig::get_instance().enable_graph() &&
        input_params.graph.tiling_data.defined();
    const bool use_flash_comm = should_enable_flash_comm(x, false);
    auto& decode_node = use_graph_decode_input
                            ? (use_flash_comm ? decode_graph_flash_comm_node_
                                              : decode_graph_node_)
                            : (use_flash_comm ? decode_eager_flash_comm_node_
                                              : decode_eager_node_);
    build_node_variant_pack(decode_node,
                            x,
                            cos_pos,
                            sin_pos,
                            decode_attn_mask_,
                            kv_cache,
                            input_params,
                            /*is_prefill=*/false,
                            node_id,
                            use_graph_decode_input,
                            use_flash_comm);
    st = execute_node(decode_node, node_id + 1000, event, event_flag);
    LOG_IF(FATAL, st != 0) << model_name_
                           << "execute decode layer fail, error code: " << st;
  }

  return at_placeholder_;
}

bool NpuQwen3DecoderLayerImpl::should_enable_flash_comm(
    torch::Tensor& x, bool is_prefill) const {
  const auto& kernel_config = ::xllm::KernelConfig::get_instance();
  const auto& param =
      is_prefill ? prefill_flash_comm_param_ : decode_graph_flash_comm_param_;
  if (is_prefill ? !prefill_flash_comm_initialized_
                 : !decode_flash_comm_initialized_) {
    return false;
  }
  if (!kernel_config.enable_qwen3_flash_comm() ||
      param.tensorParallelInfo.worldSize <= 1 || param.backend != "hccl") {
    return false;
  }
  if (kernel_config.enable_interlayer_addnorm()) {
    return false;
  }
  if (is_prefill &&
      ::xllm::ParallelConfig::get_instance().enable_multi_stream_parallel()) {
    return false;
  }
  if (x.dim() != 2) {
    return false;
  }
  const int64_t token_num = x.size(0);
  const int64_t hidden_size = x.size(x.dim() - 1);
  return token_num * hidden_size > FLASH_COMM_VOLUME_THRESHOLD;
}

void NpuQwen3DecoderLayerImpl::append_flash_comm_tensors(
    atb_speed::Model::Node& node,
    int& input_idx,
    int64_t token_num,
    int64_t hidden_size,
    bool is_prefill) {
  const auto& param =
      is_prefill ? prefill_flash_comm_param_ : decode_graph_flash_comm_param_;
  const int64_t world_size = param.tensorParallelInfo.worldSize;
  const int64_t rank = param.tensorParallelInfo.rank;
  const int64_t split_size = token_num / world_size;
  const int64_t remain_size = token_num % world_size;

  auto& host_storage =
      is_prefill ? prefill_flash_comm_host_ : decode_flash_comm_host_;
  auto& tensor_storage =
      is_prefill ? prefill_flash_comm_tensors_ : decode_flash_comm_tensors_;
  for (auto& values : host_storage) {
    values.assign(world_size, 0);
  }

  auto& send_counts = host_storage[0];
  auto& sdispls = host_storage[1];
  auto& send_count = host_storage[2];
  auto& recv_counts = host_storage[3];
  auto& rdispls = host_storage[4];
  auto& recv_count = host_storage[5];
  send_count.resize(1);
  recv_count.resize(1);

  int64_t send_offset = 0;
  int64_t recv_offset = 0;
  for (int64_t i = 0; i < world_size; ++i) {
    const int64_t local_tokens = split_size + (i < remain_size ? 1 : 0);
    recv_counts[i] = local_tokens;
    send_counts[i] = local_tokens * hidden_size;
    sdispls[i] = send_offset;
    rdispls[i] = recv_offset;
    send_offset += send_counts[i];
    recv_offset += recv_counts[i];
  }
  send_count[0] = recv_counts[rank];
  recv_count[0] = send_counts[rank];

  auto tensor_options =
      torch::TensorOptions().dtype(torch::kInt64).device(device_);
  for (size_t i = 0; i < FLASH_COMM_HOST_TENSOR_COUNT; ++i) {
    tensor_storage[i] = torch::tensor(host_storage[i], tensor_options);
    node.variantPack.inTensors.at(input_idx) =
        atb_speed::Utils::AtTensor2Tensor(tensor_storage[i]);
    node.variantPack.inTensors.at(input_idx).hostData =
        host_storage[i].data();
    ++input_idx;
  }

  const int64_t local_tokens = recv_counts[rank];
  tensor_storage[6] = torch::zeros({local_tokens}).to(device_).to(dtype_);
  tensor_storage[7] = torch::zeros({token_num}).to(device_).to(dtype_);
  node.variantPack.inTensors.at(input_idx++) =
      atb_speed::Utils::AtTensor2Tensor(tensor_storage[6]);
  node.variantPack.inTensors.at(input_idx++) =
      atb_speed::Utils::AtTensor2Tensor(tensor_storage[7]);
}

void NpuQwen3DecoderLayerImpl::build_node_variant_pack(
    atb_speed::Model::Node& node,
    torch::Tensor& x,
    torch::Tensor& cos_pos,
    torch::Tensor& sin_pos,
    at::Tensor& attn_mask,
    KVCache& kv_cache,
    ModelInputParams& input_params,
    bool is_prefill,
    int node_id,
    bool use_graph_decode_input,
    bool enable_flash_comm) {
  internal_tensors_ = atb_speed::Utils::AtTensor2Tensor(x);
  if (::xllm::KernelConfig::get_instance().enable_interlayer_addnorm() &&
      residual_.defined()) {
    residual_tensors_ = atb_speed::Utils::AtTensor2Tensor(residual_);
  }
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER) = internal_tensors_;
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 1) =
      atb_speed::Utils::AtTensor2Tensor(cos_pos);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 2) =
      atb_speed::Utils::AtTensor2Tensor(sin_pos);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 3) =
      atb_speed::Utils::AtTensor2Tensor(attn_mask);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 6) =
      atb_speed::Utils::AtTensor2Tensor(
          input_params.attention.device.kv_seq_lens);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 6).hostData =
      input_params.attention.host.kv_seq_lens.data();
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 7) = placeholder_;
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 7).hostData =
      placeholder_vec_.data();
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 8) = placeholder_;
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 9) =
      atb_speed::Utils::AtTensor2Tensor(
          input_params.attention.device.block_tables);

  int input_idx = WEIGHT_COUNT_PER_LAYER + 11;
  if (is_rec_multi_round_mode()) {
    const auto* llmrec = input_params.llmrec_params();

    if (is_prefill) {
      node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 4) =
          atb_speed::Utils::AtTensor2Tensor(kv_cache.get_k_cache());
      node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 5) =
          atb_speed::Utils::AtTensor2Tensor(kv_cache.get_v_cache());
    } else {
      CHECK_LT(static_cast<size_t>(node_id), llmrec->unshared_k_caches.size());
      CHECK_LT(static_cast<size_t>(node_id), llmrec->unshared_v_caches.size());
      node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 4) =
          atb_speed::Utils::AtTensor2Tensor(llmrec->unshared_k_caches[node_id]);
      node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 5) =
          atb_speed::Utils::AtTensor2Tensor(llmrec->unshared_v_caches[node_id]);
    }
    node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 10) = placeholder_;
    node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 11) =
        atb_speed::Utils::AtTensor2Tensor(llmrec->shared_k_caches[node_id]);
    node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 12) =
        atb_speed::Utils::AtTensor2Tensor(llmrec->shared_v_caches[node_id]);
    node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 13) =
        atb_speed::Utils::AtTensor2Tensor(llmrec->beam_width_tensor);
    node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 14) =
        atb_speed::Utils::AtTensor2Tensor(llmrec->current_round_tensor);
    input_idx = WEIGHT_COUNT_PER_LAYER + 15;
  } else {
    node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 4) =
        atb_speed::Utils::AtTensor2Tensor(kv_cache.get_k_cache());
    node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 5) =
        atb_speed::Utils::AtTensor2Tensor(kv_cache.get_v_cache());
    node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 10) =
        atb_speed::Utils::AtTensor2Tensor(
            input_params.attention.device.new_cache_slots);
  }

  if (is_prefill &&
      (::xllm::SchedulerConfig::get_instance().enable_chunked_prefill() ||
       ::xllm::KVCacheConfig::get_instance().enable_prefix_cache())) {
    node.variantPack.inTensors.at(input_idx++) =
        atb_speed::Utils::AtTensor2Tensor(
            input_params.attention.device.q_seq_lens);
    node.variantPack.inTensors.at(input_idx - 1).hostData =
        input_params.attention.host.q_seq_lens.data();
  }

  if (::xllm::KernelConfig::get_instance().enable_interlayer_addnorm() &&
      node_id > 0 && residual_.defined()) {
    node.variantPack.inTensors.at(input_idx++) = residual_tensors_;
  }

  if (enable_flash_comm) {
    append_flash_comm_tensors(
        node, input_idx, x.size(0), x.size(x.dim() - 1), is_prefill);
  }

  if (!is_prefill && use_graph_decode_input &&
      input_params.graph.tiling_data.defined()) {
    node.variantPack.inTensors.at(input_idx++) =
        atb_speed::Utils::AtTensor2Tensor(input_params.graph.tiling_data);
  }

  for (size_t i = 0; i < WEIGHT_COUNT_PER_LAYER; ++i) {
    CHECK_THROW(node.inTensors.at(i) == nullptr,
                model_name_ << "inTensor " << i << "is NULL");
    node.variantPack.inTensors.at(i) = *node.inTensors.at(i);
  }

  node.variantPack.outTensors.at(0) = internal_tensors_;
  if (::xllm::KernelConfig::get_instance().enable_interlayer_addnorm() &&
      (node_id < num_hidden_layers_ - 1) && residual_.defined()) {
    node.variantPack.outTensors.at(1) = residual_tensors_;
  }
}

}  // namespace layer
}  // namespace xllm

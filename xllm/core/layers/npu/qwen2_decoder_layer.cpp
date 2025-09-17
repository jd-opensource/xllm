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

#include "qwen2_decoder_layer.h"

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <mstx/ms_tools_ext.h>

#include <map>

#include "attn_mask.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUException.h"

DECLARE_bool(enable_chunked_prefill);
namespace xllm::hf {

const uint64_t WEIGHT_COUNT_PER_LAYER = 50;

static std::vector<std::pair<int, std::string>> WEIGHT_MAPPING = {
    {IN_NORM_WEIGHT, "input_layernorm.weight"},
    {IN_Q_WEIGHT, "self_attn.q_proj.weight"},
    {IN_Q_BIAS, "self_attn.q_proj.bias"},
    {IN_K_WEIGHT, "self_attn.k_proj.weight"},
    {IN_K_BIAS, "self_attn.k_proj.bias"},
    {IN_V_WEIGHT, "self_attn.v_proj.weight"},
    {IN_V_BIAS, "self_attn.v_proj.bias"},
    {IN_ATTENTION_OUT_WEIGHT, "self_attn.o_proj.weight"},
    {IN_SELFOUT_NORM_WEIGHT, "post_attention_layernorm.weight"},
    {IN_MLP_W2_WEIGHT, "mlp.gate_proj.weight"},
    {IN_MLP_W1_WEIGHT, "mlp.up_proj.weight"},
    {IN_MLP_CPROJ_WEIGHT, "mlp.down_proj.weight"}};

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
    {IN_MLP_CPROJ_WEIGHT, "mlp.down_proj.weight"}};

static std::map<int, int> WEIGHT_SHARD = {{IN_Q_WEIGHT, 0},
                                          {IN_Q_BIAS, 0},
                                          {IN_K_WEIGHT, 0},
                                          {IN_K_BIAS, 0},
                                          {IN_V_WEIGHT, 0},
                                          {IN_V_BIAS, 0},
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

std::shared_ptr<Qwen2DecoderImpl> create_qwen2_decode_layer(
    const ModelContext& context) {
  return std::make_shared<Qwen2DecoderImpl>(context);
}

void Qwen2DecoderImpl::param_from_args(
    atb_speed::qwen::DecoderLayerParam& param,
    const ModelArgs& args,
    const ParallelArgs& parallel_args,
    bool isPrefill) {
  param.isFA = false;
  param.isPrefill = isPrefill;
  param.isBF16 = args.dtype() == "bfloat16";
  // std::cout<<"param.isBF16:"<<param.isBF16<<std::endl;
  param.supportSwiGLU = true;
  param.supportLcoc = isPrefill;  // isPrefill
  param.supportSpeculate = false;
  param.enableSplitFuse = FLAGS_enable_chunked_prefill && isPrefill;
  param.supportLora = false;
  param.loraEnableGMM = false;
  param.packQuantType = {1, 1};
  param.linearQuantType = {0, -1, -1, 0, 0, -1, 0};
  param.linearTransposeType = {static_cast<int>(TransposeType::TRANSPOSE),
                               static_cast<int>(TransposeType::INVALID),
                               static_cast<int>(TransposeType::INVALID),
                               static_cast<int>(TransposeType::TRANSPOSE),
                               static_cast<int>(TransposeType::TRANSPOSE),
                               static_cast<int>(TransposeType::INVALID),
                               static_cast<int>(TransposeType::TRANSPOSE)};
  param.kvQuant = false;
  param.quantGroupSize = 0;
  param.rmsNormEps = args.rms_norm_eps();
  param.worldSize = parallel_args.world_size();
  param.numAttentionHeadsPerRank = args.n_heads() / param.worldSize;
  param.hiddenSizePerAttentionHead = args.hidden_size() / args.n_heads();
  param.enableIntraLayerAddNorm = false;
  param.enableInterLayerAddNorm = false;
  // param.numKeyValueHeadsPerRank = args.n_kv_heads();
  std::optional<long int> optionalValue = args.n_kv_heads();
  param.numKeyValueHeadsPerRank =
      static_cast<int>(optionalValue.value()) / param.worldSize;
  ;
  // param.numKeyValueHeadsPerRank = static_cast<int>(args.n_kv_heads());

  param.rank = parallel_args.rank();
  param.backend = "lccl";
  // param.backend = "lccl";
  param.enableLogN = false;
}

Qwen2DecoderImpl::Qwen2DecoderImpl(const ModelContext& context)
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
  device_id_ = options.device().index();
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

void Qwen2DecoderImpl::verify_loaded_weights() const {
  for (const auto& [index, name] : WEIGHT_MAPPING) {
    CHECK(at_weight_tensors_[index].sizes() != std::vector<int64_t>({1}))
        << "weight is not loaded for " << name;
  }
}

TransposeType Qwen2DecoderImpl::check_transpose(at::Tensor& tensor) {
  bool is_k_divisible = tensor.size(1) % 256 == 0;
  bool is_n_divisible = tensor.size(0) % 256 == 0;

  if (!is_k_divisible && is_n_divisible) {
    return TransposeType::NOT_TRANSPOSE;
  }

  return TransposeType::TRANSPOSE;
}

void Qwen2DecoderImpl::merge_loaded_weights() {
  if (quantize_type_ == "w8a8") {
    at_weight_tensors_[IN_ATTENTION_OUT_DEQSCALE] =
        at_weight_tensors_[IN_ATTENTION_OUT_DEQSCALE].to(torch::kFloat32);
    at_weight_tensors_[IN_Q_DEQSCALE] =
        torch::cat({at_weight_tensors_[IN_Q_DEQSCALE],
                    at_weight_tensors_[IN_K_DEQSCALE],
                    at_weight_tensors_[IN_V_DEQSCALE]},
                   0)
            .to(torch::kFloat32);
    at_weight_tensors_[IN_K_DEQSCALE] = torch::zeros({1}).to(device_);
    at_weight_tensors_[IN_V_DEQSCALE] = torch::zeros({1}).to(device_);
    at_weight_tensors_[IN_K_OFFSET] = torch::zeros({1}).to(device_);
    at_weight_tensors_[IN_V_OFFSET] = torch::zeros({1}).to(device_);

    at_weight_tensors_[IN_K_SCALE] = torch::zeros({1}).to(device_);
    at_weight_tensors_[IN_V_SCALE] = torch::zeros({1}).to(device_);
    at_weight_tensors_[IN_MLP_W2_BIAS] =
        torch::cat({at_weight_tensors_[IN_MLP_W2_BIAS],
                    at_weight_tensors_[IN_MLP_W1_BIAS]},
                   0);
    at_weight_tensors_[IN_MLP_W1_BIAS] = torch::zeros({1}).to(device_);
    at_weight_tensors_[IN_MLP_W2_DEQSCALE] =
        torch::cat({at_weight_tensors_[IN_MLP_W2_DEQSCALE],
                    at_weight_tensors_[IN_MLP_W1_DEQSCALE]},
                   0)
            .to(torch::kFloat32);
    at_weight_tensors_[IN_MLP_W1_DEQSCALE] = torch::zeros({1}).to(device_);

    at_weight_tensors_[IN_MLP_W1_OFFSET] = torch::zeros({1}).to(device_);
    at_weight_tensors_[IN_MLP_W1_SCALE] = torch::zeros({1}).to(device_);
    at_weight_tensors_[IN_Q_OFFSET] =
        at_weight_tensors_[IN_Q_OFFSET].to(torch::kInt8).to(device_);
    at_weight_tensors_[IN_ATTENTION_OUT_OFFSET] =
        at_weight_tensors_[IN_ATTENTION_OUT_OFFSET]
            .to(torch::kInt8)
            .to(device_);
    at_weight_tensors_[IN_MLP_W2_OFFSET] =
        at_weight_tensors_[IN_MLP_W2_OFFSET].to(torch::kInt8).to(device_);
    if (device_id_ != 0) {
      torch::Tensor original_tensor = at_weight_tensors_[IN_ATTENTION_OUT_BIAS];
      auto shape = original_tensor.sizes();
      auto dtype = original_tensor.dtype();
      auto device = original_tensor.device();

      at_weight_tensors_[IN_ATTENTION_OUT_BIAS] = torch::zeros(
          shape, torch::TensorOptions().dtype(dtype).device(device));
    }
  }

  auto new_q_weight = torch::cat({at_weight_tensors_[IN_Q_WEIGHT],
                                  at_weight_tensors_[IN_K_WEIGHT],
                                  at_weight_tensors_[IN_V_WEIGHT]},
                                 0);

  at_weight_tensors_[IN_Q_WEIGHT] = new_q_weight;

  at_weight_tensors_[IN_K_WEIGHT] = torch::zeros({1}).to(device_);
  at_weight_tensors_[IN_V_WEIGHT] = torch::zeros({1}).to(device_);

  auto new_q_bias = torch::cat({at_weight_tensors_[IN_Q_BIAS],
                                at_weight_tensors_[IN_K_BIAS],
                                at_weight_tensors_[IN_V_BIAS]},
                               0);
  at_weight_tensors_[IN_Q_BIAS] = new_q_bias;

  at_weight_tensors_[IN_K_BIAS] = torch::zeros({1}).to(device_);
  at_weight_tensors_[IN_V_BIAS] = torch::zeros({1}).to(device_);

  TransposeType transpose_type =
      check_transpose(at_weight_tensors_[IN_MLP_W2_WEIGHT]);
  int transpose_value = static_cast<int>(transpose_type);
  prefill_param_.linearTransposeType[4] = transpose_value;
  decode_param_.linearTransposeType[4] = transpose_value;
  if (transpose_type == TransposeType::TRANSPOSE) {
    auto new_mlp_weight = torch::cat({at_weight_tensors_[IN_MLP_W2_WEIGHT],
                                      at_weight_tensors_[IN_MLP_W1_WEIGHT]},
                                     0);
    at_weight_tensors_[IN_MLP_W2_WEIGHT] = new_mlp_weight.contiguous();
  } else {
    auto new_mlp_weight = torch::cat({at_weight_tensors_[IN_MLP_W2_WEIGHT],
                                      at_weight_tensors_[IN_MLP_W1_WEIGHT]},
                                     0)
                              .transpose(0, 1);
    at_weight_tensors_[IN_MLP_W2_WEIGHT] = new_mlp_weight.contiguous();
  }

  at_weight_tensors_[IN_MLP_W1_WEIGHT] = torch::zeros({1}).to(device_);

  c10_npu::NPUCachingAllocator::emptyCache();
  for (int i = 0; i < WEIGHT_COUNT_PER_LAYER; ++i) {
    atb_weight_tensors_[i] =
        atb_speed::Utils::AtTensor2Tensor(at_weight_tensors_[i]);
  }

  init_layer();
}

void Qwen2DecoderImpl::load_state_dict(const StateDict& state_dict) {
  if (quantize_type_ == "w8a8") {
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

    prefill_param_.packQuantType = {static_cast<int>(PackType::ALL_W8A8),
                                    static_cast<int>(PackType::ALL_W8A8)};
    decode_param_.packQuantType = {static_cast<int>(PackType::ALL_W8A8),
                                   static_cast<int>(PackType::ALL_W8A8)};
    prefill_param_.linearQuantType = {static_cast<int>(LinearType::INT),
                                      static_cast<int>(LinearType::INVALID),
                                      static_cast<int>(LinearType::INVALID),
                                      static_cast<int>(LinearType::INT),
                                      static_cast<int>(LinearType::INT),
                                      static_cast<int>(LinearType::INVALID),
                                      static_cast<int>(LinearType::FP)};
    decode_param_.linearQuantType = {static_cast<int>(LinearType::INT),
                                     static_cast<int>(LinearType::INVALID),
                                     static_cast<int>(LinearType::INVALID),
                                     static_cast<int>(LinearType::INT),
                                     static_cast<int>(LinearType::INT),
                                     static_cast<int>(LinearType::INVALID),
                                     static_cast<int>(LinearType::FP)};
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

int64_t Qwen2DecoderImpl::init_layer() {
  init_attn_mask();
  ATBBase::name_ = "qwen2_decoder_layer";
  model_name_ = "qwen2";
  CHECK_OPERATION_STATUS_RETURN(init_node(prefill_node_, prefill_param_));
  CHECK_OPERATION_STATUS_RETURN(init_node(decode_node_, decode_param_));

  return atb::NO_ERROR;
}

int64_t Qwen2DecoderImpl::init_attn_mask() {
  torch::Dtype dtype =
      prefill_param_.isBF16 ? torch::kBFloat16 : torch::kFloat16;
  // encode_attn_mask_ =
  //     AttentionMaskImpl(device_, dtype).get_attn_mask(2048, dtype, device_);
  decode_attn_mask_ = torch::zeros({1}).to(device_).to(dtype);

  return atb::NO_ERROR;
}

int64_t Qwen2DecoderImpl::init_node(atb_speed::Model::Node& node,
                                    atb_speed::qwen::DecoderLayerParam& param) {
  atb::Operation* operation = nullptr;
  atb_speed::qwen::DecoderLayer(param, &operation);
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

torch::Tensor Qwen2DecoderImpl::forward(torch::Tensor& x,
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

void Qwen2DecoderImpl::build_node_variant_pack(atb_speed::Model::Node& node,
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
    // LOG(INFO) << model_name_ << "inTensors[" << i << "]:"
    //               << atb_speed::TensorUtil::TensorToString(
    //                      node.variantPack.inTensors.at(i));
  }

  node.variantPack.outTensors.at(0) = internal_tensors_;
}

Qwen2Decoder::Qwen2Decoder(const ModelContext& context)
    : ModuleHolder(create_qwen2_decode_layer(context)) {}

}  // namespace xllm::hf

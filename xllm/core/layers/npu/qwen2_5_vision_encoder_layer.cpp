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

#include "qwen2_5_vision_encoder_layer.h"

#include <glog/logging.h>
#include <mstx/ms_tools_ext.h>

#include <iostream>
#include <map>

#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUException.h"

namespace xllm::hf {
const uint64_t WEIGHT_COUNT_PER_LAYER = 18;

static std::vector<std::pair<int, std::string>> WEIGHT_MAPPING = {
    {IN_INPUT_NORM_WEIGHT, "norm1.weight"},
    {IN_POST_NORM_WEIGHT, "norm2.weight"},
    {IN_QKV_WEIGHT, "qkv.weight"},
    {IN_QKV_BIAS, "qkv.bias"},
    {IN_WATTENTION_OUT_WEIGHT, "attn.proj.weight"},
    {IN_WATTENTION_OUT_BIAS, "attn.proj.bias"},
    {IN_MLP_GATE_WEIGHT, "mlp.gate_proj.weight"},
    {IN_MLP_GATE_BIAS, "mlp.gate_proj.bias"},
    {IN_MLP_UP_WEIGHT, "mlp.up_proj.weight"},
    {IN_MLP_UP_BIAS, "mlp.up_proj.bias"},
    {IN_MLP_DOWN_WEIGHT, "mlp.down_proj.weight"},
    {IN_MLP_DOWN_BIAS, "mlp.down_proj.bias"},
};

// {weight,dim}
static std::map<int, int> WEIGHT_SHARD = {
    {IN_WATTENTION_OUT_WEIGHT, 1},
    {IN_MLP_GATE_WEIGHT, 0},
    {IN_MLP_GATE_BIAS, 0},
    {IN_MLP_UP_WEIGHT, 0},
    {IN_MLP_UP_BIAS, 0},
    {IN_MLP_DOWN_WEIGHT, 1},
};

std::shared_ptr<Qwen2_5VisionEncoderImpl> create_qwen2_5_vision_encoder_layer(
    const Context& context) {
  return std::make_shared<Qwen2_5VisionEncoderImpl>(context);
}

void Qwen2_5VisionEncoderImpl::param_from_args(
    atb_speed::qwen::VisionEncoderLayerParam& param,
    const ModelArgs& args,
    const ParallelArgs& parallel_args) {
  param.isBF16 = args.dtype() == "bfloat16";
  param.rmsNormEps = args.rms_norm_eps();
  param.worldSize = parallel_args.world_size();
  param.numAttentionHeadsPerRank =
      args.mm_num_attention_heads() / param.worldSize;
  param.hiddenSizePerAttentionHead =
      args.mm_hidden_size() / args.mm_num_attention_heads();
  std::optional<long int> optionalValue = args.mm_num_attention_heads();
  param.numKeyValueHeadsPerRank =
      static_cast<int>(optionalValue.value()) / param.worldSize;
  param.rank = parallel_args.rank();
  param.backend = "lccl";
  param.enableLogN = false;
}

Qwen2_5VisionEncoderImpl::Qwen2_5VisionEncoderImpl(const Context& context)
    : ATBBase(context) {
  auto model_args = context.get_model_args();
  auto parallel_args = context.get_parallel_args();
  auto options = context.get_tensor_options();
  param_from_args(encode_param_, model_args, parallel_args);
  at_weight_tensors_.resize(WEIGHT_COUNT_PER_LAYER);
  atb_weight_tensors_.resize(WEIGHT_COUNT_PER_LAYER);
  dtype_ = c10::typeMetaToScalarType(options.dtype());
  device_id_ = options.device().index();
  placeholder_ = atb_speed::Utils::AtTensor2Tensor(
      torch::zeros({1}).to(device_).to(dtype_));
  at_placeholder_ = torch::zeros({1}).to(device_).to(dtype_);
  for (int i = 0; i < WEIGHT_COUNT_PER_LAYER; ++i) {
    at_weight_tensors_[i] = torch::zeros({1}).to(options);
  }
}

void Qwen2_5VisionEncoderImpl::verify_loaded_weights() const {
  for (const auto& [index, name] : WEIGHT_MAPPING) {
    CHECK(at_weight_tensors_[index].sizes() != std::vector<int64_t>({1}))
        << "weight is not loaded for " << name;
  }
}
void Qwen2_5VisionEncoderImpl::pad_mlp_weights() {
  torch::Tensor weight = at_weight_tensors_[IN_MLP_GATE_WEIGHT];
  torch::Tensor bias = at_weight_tensors_[IN_MLP_GATE_BIAS];

  int64_t tp_intermediate_size_half = weight.size(0) / 2;
  int64_t remainder = tp_intermediate_size_half % 32;
  int64_t tp_intermediate_size_half_pad;
  if (remainder != 0) {
    tp_intermediate_size_half_pad =
        tp_intermediate_size_half + (32 - remainder);
  } else {
    tp_intermediate_size_half_pad = tp_intermediate_size_half;
  }

  auto weight_split1 = weight.slice(0, 0, tp_intermediate_size_half);
  auto weight_split2 = weight.slice(0, tp_intermediate_size_half);
  auto bias_split1 = bias.slice(0, 0, tp_intermediate_size_half);
  auto bias_split2 = bias.slice(0, tp_intermediate_size_half);

  auto weight_split1_padded =
      pad_tensor(weight_split1, tp_intermediate_size_half_pad);
  auto weight_split2_padded =
      pad_tensor(weight_split2, tp_intermediate_size_half_pad);
  auto bias_split1_padded =
      pad_tensor(bias_split1, tp_intermediate_size_half_pad);
  auto bias_split2_padded =
      pad_tensor(bias_split2, tp_intermediate_size_half_pad);

  auto weight_padded =
      torch::cat({weight_split1_padded, weight_split2_padded}, 0);
  auto bias_padded = torch::cat({bias_split1_padded, bias_split2_padded}, 0);
  at_weight_tensors_[IN_MLP_GATE_WEIGHT] = weight_padded;
  at_weight_tensors_[IN_MLP_GATE_BIAS] = bias_padded;

  torch::Tensor down_weight = at_weight_tensors_[IN_MLP_DOWN_WEIGHT];

  auto tp_intermediate_size = down_weight.size(1);
  remainder = tp_intermediate_size % 32;
  int64_t tp_intermediate_size_pad;
  if (remainder != 0) {
    tp_intermediate_size_pad = tp_intermediate_size + (32 - remainder);
  } else {
    tp_intermediate_size_pad = tp_intermediate_size;
  }

  auto down_weight_padded =
      pad_tensor(down_weight, tp_intermediate_size_pad, 1);
  at_weight_tensors_[IN_MLP_DOWN_WEIGHT] = down_weight_padded;
}
void Qwen2_5VisionEncoderImpl::pad_qkv_weights() {
  auto qkv_proj_weight = at_weight_tensors_[IN_QKV_WEIGHT];
  auto qkv_proj_bias = at_weight_tensors_[IN_QKV_BIAS];
  int num_heads_pre_rank = encode_param_.numAttentionHeadsPerRank;
  int hidden_size = num_heads_pre_rank * 80 * encode_param_.worldSize;

  auto qkv_proj_weight_reshaped =
      qkv_proj_weight.reshape({num_heads_pre_rank, 3, 80, hidden_size});

  auto first_half = qkv_proj_weight_reshaped.slice(2, 0, 40);
  auto second_half = qkv_proj_weight_reshaped.slice(2, 40, 80);

  auto first_half_padded = torch::nn::functional::pad(
      first_half, torch::nn::functional::PadFuncOptions({0, 0, 0, 24}));
  auto second_half_padded = torch::nn::functional::pad(
      second_half, torch::nn::functional::PadFuncOptions({0, 0, 0, 24}));

  auto qkv_proj_weight_padded =
      torch::cat({first_half_padded, second_half_padded}, 2);
  auto qkv_proj_weight_final = qkv_proj_weight_padded.reshape(
      {num_heads_pre_rank * 128 * 3, hidden_size});
  qkv_proj_weight_final =
      at_npu::native::npu_format_cast(qkv_proj_weight_final, 2);

  auto qkv_proj_bias_reshaped =
      qkv_proj_bias.reshape({num_heads_pre_rank, 3, 80});
  first_half = qkv_proj_bias_reshaped.slice(2, 0, 40);
  second_half = qkv_proj_bias_reshaped.slice(2, 40, 80);

  first_half_padded = torch::nn::functional::pad(
      first_half, torch::nn::functional::PadFuncOptions({0, 24}));
  second_half_padded = torch::nn::functional::pad(
      second_half, torch::nn::functional::PadFuncOptions({0, 24}));
  auto qkv_proj_bias_padded =
      torch::cat({first_half_padded, second_half_padded}, 2);
  auto qkv_proj_bias_final =
      qkv_proj_bias_padded.reshape({num_heads_pre_rank * 128 * 3});

  at_weight_tensors_[IN_QKV_WEIGHT] = qkv_proj_weight_final;
  at_weight_tensors_[IN_QKV_BIAS] = qkv_proj_bias_final;

  auto out_proj_weight = at_weight_tensors_[IN_WATTENTION_OUT_WEIGHT];

  out_proj_weight =
      torch::nn::functional::pad(
          out_proj_weight.reshape({hidden_size, num_heads_pre_rank * 2, 40}),
          torch::nn::functional::PadFuncOptions({0, 24, 0, 0}))
          .reshape({hidden_size, num_heads_pre_rank * 128});
  at_weight_tensors_[IN_WATTENTION_OUT_WEIGHT] = out_proj_weight;
}
void Qwen2_5VisionEncoderImpl::merge_loaded_weights() {
  pad_qkv_weights();
  if (encode_param_.worldSize > 1) {
    // merge qkv weight
    auto new_qkv_weight = torch::cat({at_weight_tensors_[IN_VISION_Q_WEIGHT],
                                      at_weight_tensors_[IN_VISION_K_WEIGHT],
                                      at_weight_tensors_[IN_VISION_V_WEIGHT]},
                                     0);
    at_weight_tensors_[IN_QKV_WEIGHT] = new_qkv_weight;
    at_weight_tensors_[IN_VISION_Q_WEIGHT] = torch::zeros({1}).to(device_);
    at_weight_tensors_[IN_VISION_K_WEIGHT] = torch::zeros({1}).to(device_);
    at_weight_tensors_[IN_VISION_V_WEIGHT] = torch::zeros({1}).to(device_);

    // merge qkv bias
    auto new_qkv_bias = torch::cat({at_weight_tensors_[IN_VISION_Q_BIAS],
                                    at_weight_tensors_[IN_VISION_K_BIAS],
                                    at_weight_tensors_[IN_VISION_V_BIAS]},
                                   0);
    at_weight_tensors_[IN_QKV_BIAS] = new_qkv_bias;
    at_weight_tensors_[IN_VISION_Q_BIAS] = torch::zeros({1}).to(device_);
    at_weight_tensors_[IN_VISION_K_BIAS] = torch::zeros({1}).to(device_);
    at_weight_tensors_[IN_VISION_V_BIAS] = torch::zeros({1}).to(device_);
  }
  // merge gate up
  auto new_mlp_weight = torch::cat({at_weight_tensors_[IN_MLP_GATE_WEIGHT],
                                    at_weight_tensors_[IN_MLP_UP_WEIGHT]},
                                   0);
  at_weight_tensors_[IN_MLP_GATE_WEIGHT] = new_mlp_weight;
  auto new_mlp_bias = torch::cat({at_weight_tensors_[IN_MLP_GATE_BIAS],
                                  at_weight_tensors_[IN_MLP_UP_BIAS]},
                                 0);
  at_weight_tensors_[IN_MLP_GATE_BIAS] = new_mlp_bias;
  at_weight_tensors_[IN_MLP_UP_BIAS] = torch::zeros({1}).to(device_);
  pad_mlp_weights();
  c10_npu::NPUCachingAllocator::emptyCache();
  for (int i = 0; i < WEIGHT_COUNT_PER_LAYER; ++i) {
    atb_weight_tensors_[i] =
        atb_speed::Utils::AtTensor2Tensor(at_weight_tensors_[i]);
  }

  init_layer();
}
// tp spilt weight
void Qwen2_5VisionEncoderImpl::get_weights_col_packed_qkv() {
  int rank = encode_param_.rank;
  int worldSize = encode_param_.worldSize;
  // split qkv weight
  qkv_weight = torch::chunk(at_weight_tensors_[IN_QKV_WEIGHT], 3, 0);
  qkv_bias = torch::chunk(at_weight_tensors_[IN_QKV_BIAS], 3, 0);
  // weight
  at_weight_tensors_[IN_VISION_Q_WEIGHT] =
      (qkv_weight[0].chunk(worldSize, 0))[rank];
  at_weight_tensors_[IN_VISION_K_WEIGHT] =
      (qkv_weight[1].chunk(worldSize, 0))[rank];
  at_weight_tensors_[IN_VISION_V_WEIGHT] =
      (qkv_weight[2].chunk(worldSize, 0))[rank];
  // bias
  at_weight_tensors_[IN_VISION_Q_BIAS] =
      (qkv_bias[0].chunk(worldSize, 0))[rank];
  at_weight_tensors_[IN_VISION_K_BIAS] =
      (qkv_bias[1].chunk(worldSize, 0))[rank];
  at_weight_tensors_[IN_VISION_V_BIAS] =
      (qkv_bias[2].chunk(worldSize, 0))[rank];
}

void Qwen2_5VisionEncoderImpl::load_state_dict(const StateDict& state_dict) {
  for (const auto& [index, name] : WEIGHT_MAPPING) {
    if (WEIGHT_SHARD.find(index) != WEIGHT_SHARD.end()) {
      set_weight(state_dict, name, index, WEIGHT_SHARD[index]);
    } else {
      set_weight(state_dict, name, index);
    }
  }
  get_weights_col_packed_qkv();
}

int64_t Qwen2_5VisionEncoderImpl::init_layer() {
  ATBBase::name_ = "qwen2_5_encoder_layer";
  model_name_ = "qwen2_5_vl";
  CHECK_OPERATION_STATUS_RETURN(init_node(encode_node_, encode_param_));
  return atb::NO_ERROR;
}

int64_t Qwen2_5VisionEncoderImpl::init_node(
    atb_speed::Model::Node& node,
    atb_speed::qwen::VisionEncoderLayerParam& param) {
  atb::Operation* operation = nullptr;
  atb_speed::qwen::EncoderLayer(param, &operation);
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

torch::Tensor Qwen2_5VisionEncoderImpl::forward(torch::Tensor& x,
                                                torch::Tensor& cos_pos,
                                                torch::Tensor& sin_pos,
                                                torch::Tensor& cu_seqlen,
                                                std::vector<int>& cu_seqlen_vec,
                                                ModelInputParams& input_params,
                                                atb::Context* context,
                                                AtbWorkspace& workspace,
                                                int node_id,
                                                aclrtEvent* event,
                                                std::atomic<bool>* event_flag) {
  atb::Status st;

  build_node_variant_pack(encode_node_,
                          x,
                          cos_pos,
                          sin_pos,
                          cu_seqlen,
                          cu_seqlen_vec,
                          input_params,
                          true);
  // mstxRangeEnd(id);
  st = execute_node(encode_node_, context, workspace, node_id);
  LOG_IF(FATAL, st != 0) << model_name_
                         << "excute encode layer fail, error code: " << st;
  return x;
}

void Qwen2_5VisionEncoderImpl::build_node_variant_pack(
    atb_speed::Model::Node& node,
    torch::Tensor& x,
    torch::Tensor& cos_pos,
    torch::Tensor& sin_pos,
    torch::Tensor& cu_seqlen,
    std::vector<int>& cu_seqlen_vec,
    ModelInputParams& input_params,
    bool is_prefill) {
  internal_tensors_ = atb_speed::Utils::AtTensor2Tensor(x);

  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER) = internal_tensors_;
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 1) =
      atb_speed::Utils::AtTensor2Tensor(cos_pos);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 2) =
      atb_speed::Utils::AtTensor2Tensor(sin_pos);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 3) =
      atb_speed::Utils::AtTensor2Tensor(cu_seqlen);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 3).hostData =
      cu_seqlen_vec.data();

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

Qwen2_5VisionEncoder::Qwen2_5VisionEncoder(const Context& context)
    : ModuleHolder(create_qwen2_5_vision_encoder_layer(context)) {}
}  // namespace xllm::hf

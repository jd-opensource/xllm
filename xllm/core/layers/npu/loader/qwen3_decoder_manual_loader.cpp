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

#include "qwen3_decoder_manual_loader.h"

#include "qwen_loader_constants.h"

namespace xllm {
namespace layer {
using namespace qwen3_decoder_constants;

Qwen3DecoderManualLoader::Qwen3DecoderManualLoader(uint64_t weight_count,
                                                   const ModelContext& context,
                                                   bool enableAddNorm)
    : BaseManualLoader(weight_count, context), enableAddNorm_(enableAddNorm) {
  auto options = context.get_tensor_options();
  rank_id_ = parallel_args_.rank();

  dtype_ = c10::typeMetaToScalarType(options.dtype());
  for (int i = 0; i < weight_count; ++i) {
    at_weight_tensors_[i] = torch::zeros({1}).to(options);
  }

  at_placeholder_ = torch::zeros({1}).to(at::kCPU).to(dtype_);
}

void Qwen3DecoderManualLoader::load_state_dict(const StateDict& state_dict) {
  if (quantize_type_.compare("w8a8") == 0) {
    for (const auto& [index, name] : WEIGHT_MAPPING_W8A8) {
      if (WEIGHT_SHARD_W8A8.find(index) != WEIGHT_SHARD_W8A8.end()) {
        set_weight(state_dict, name, index, WEIGHT_SHARD_W8A8[index], true);
      } else {
        set_weight(state_dict, name, index, true);
      }
    }
    at_host_weight_tensors_[IN_NORM_BIAS] =
        torch::zeros(at_host_weight_tensors_[IN_NORM_WEIGHT].sizes(),
                     at_host_weight_tensors_[IN_NORM_WEIGHT].options());

    at_host_weight_tensors_[IN_SELFOUT_NORM_BIAS] =
        torch::zeros(at_host_weight_tensors_[IN_SELFOUT_NORM_WEIGHT].sizes(),
                     at_host_weight_tensors_[IN_SELFOUT_NORM_WEIGHT].options());
    return;
  }

  for (const auto& [index, name] : WEIGHT_MAPPING) {
    if (WEIGHT_SHARD.find(index) != WEIGHT_SHARD.end()) {
      set_weight(state_dict, name, index, WEIGHT_SHARD[index], true);
    } else {
      set_weight(state_dict, name, index, true);
    }
  }
}

bool Qwen3DecoderManualLoader::is_nz_format_tensor(int weight_index) {
  if (weight_index == IN_Q_WEIGHT || weight_index == IN_ATTENTION_OUT_WEIGHT ||
      weight_index == IN_MLP_W2_WEIGHT || weight_index == IN_MLP_CPROJ_WEIGHT) {
    return true;
  }
  return false;
}

void Qwen3DecoderManualLoader::merge_host_at_weights() {
  if (quantize_type_.compare("w8a8") == 0) {
    at_host_weight_tensors_[IN_ATTENTION_OUT_DEQSCALE] =
        at_host_weight_tensors_[IN_ATTENTION_OUT_DEQSCALE].to(torch::kFloat32);
    at_host_weight_tensors_[IN_Q_DEQSCALE] =
        torch::cat({at_host_weight_tensors_[IN_Q_DEQSCALE],
                    at_host_weight_tensors_[IN_K_DEQSCALE],
                    at_host_weight_tensors_[IN_V_DEQSCALE]},
                   0)
            .to(torch::kFloat32);

    at_host_weight_tensors_[IN_Q_BIAS] =
        torch::cat({at_host_weight_tensors_[IN_Q_BIAS],
                    at_host_weight_tensors_[IN_K_BIAS],
                    at_host_weight_tensors_[IN_V_BIAS]},
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
      at_host_weight_tensors_[idx] = at_placeholder_;
    }

    at_host_weight_tensors_[IN_MLP_W2_BIAS] =
        torch::cat({at_host_weight_tensors_[IN_MLP_W2_BIAS],
                    at_host_weight_tensors_[IN_MLP_W1_BIAS]},
                   0);

    at_host_weight_tensors_[IN_MLP_W2_DEQSCALE] =
        torch::cat({at_host_weight_tensors_[IN_MLP_W2_DEQSCALE],
                    at_host_weight_tensors_[IN_MLP_W1_DEQSCALE]},
                   0)
            .to(torch::kFloat32);

    for (auto idx : {IN_MLP_W1_BIAS,
                     IN_MLP_W1_OFFSET,
                     IN_MLP_W1_SCALE,
                     IN_MLP_W1_DEQSCALE}) {
      at_host_weight_tensors_[idx] = at_placeholder_;
    }

    at_host_weight_tensors_[IN_Q_OFFSET] =
        at_host_weight_tensors_[IN_Q_OFFSET].to(torch::kInt8);
    at_host_weight_tensors_[IN_ATTENTION_OUT_OFFSET] =
        at_host_weight_tensors_[IN_ATTENTION_OUT_OFFSET].to(torch::kInt8);
    at_host_weight_tensors_[IN_MLP_W2_OFFSET] =
        at_host_weight_tensors_[IN_MLP_W2_OFFSET].to(torch::kInt8);

    if (rank_id_ != 0) {
      torch::Tensor original_tensor =
          at_host_weight_tensors_[IN_ATTENTION_OUT_BIAS];
      auto shape = original_tensor.sizes();
      auto dtype = original_tensor.dtype();
      auto device = original_tensor.device();

      at_host_weight_tensors_[IN_ATTENTION_OUT_BIAS] = torch::zeros(
          shape, torch::TensorOptions().dtype(dtype).device(device));
    }
  }

  at_host_weight_tensors_[IN_Q_WEIGHT] =
      torch::cat({at_host_weight_tensors_[IN_Q_WEIGHT],
                  at_host_weight_tensors_[IN_K_WEIGHT],
                  at_host_weight_tensors_[IN_V_WEIGHT]},
                 0)
          .transpose(0, 1)
          .contiguous();

  at_host_weight_tensors_[IN_ATTENTION_OUT_WEIGHT] =
      at_host_weight_tensors_[IN_ATTENTION_OUT_WEIGHT]
          .transpose(0, 1)
          .contiguous();

  at_host_weight_tensors_[IN_MLP_W2_WEIGHT] =
      torch::cat({at_host_weight_tensors_[IN_MLP_W2_WEIGHT],
                  at_host_weight_tensors_[IN_MLP_W1_WEIGHT]},
                 0)
          .transpose(0, 1)
          .contiguous();

  at_host_weight_tensors_[IN_MLP_CPROJ_WEIGHT] =
      at_host_weight_tensors_[IN_MLP_CPROJ_WEIGHT].transpose(0, 1).contiguous();

  for (auto idx :
       {IN_MLP_W1_WEIGHT, IN_K_WEIGHT, IN_V_WEIGHT, IN_K_BIAS, IN_V_BIAS}) {
    at_host_weight_tensors_[idx] = at_placeholder_;
  }

  if (enableAddNorm_) {
    if (quantize_type_.compare("w8a8") == 0) {
      // quantize
      torch::ScalarType weight_fill_dtype = torch::kBFloat16;
      int64_t weight_attn_shape = at_host_weight_tensors_[IN_Q_WEIGHT].size(-1);
      int64_t weight_mlp_shape =
          at_host_weight_tensors_[IN_MLP_W2_WEIGHT].size(-1);
      at_host_weight_tensors_[IN_QKV_SCALE_FILL] =
          at_host_weight_tensors_[IN_Q_SCALE]
              .repeat(weight_attn_shape)
              .to(weight_fill_dtype);
      at_host_weight_tensors_[IN_MLP_SCALE_FILL] =
          at_host_weight_tensors_[IN_MLP_W2_SCALE]
              .repeat(weight_mlp_shape)
              .to(weight_fill_dtype);
      at_host_weight_tensors_[IN_QKV_OFFSET_FILL] =
          at_host_weight_tensors_[IN_Q_OFFSET]
              .repeat(weight_attn_shape)
              .to(weight_fill_dtype);
      at_host_weight_tensors_[IN_MLP_OFFSET_FILL] =
          at_host_weight_tensors_[IN_MLP_W2_OFFSET]
              .repeat(weight_mlp_shape)
              .to(weight_fill_dtype);
    } else {
      // bfloat16 or float16
      for (auto idx : {IN_QKV_SCALE_FILL,
                       IN_QKV_OFFSET_FILL,
                       IN_MLP_SCALE_FILL,
                       IN_MLP_OFFSET_FILL}) {
        at_host_weight_tensors_[idx] = at_placeholder_;
      }
    }
  }
}

void Qwen3DecoderManualLoader::verify_loaded_weights() const {
  for (const auto& [index, name] : WEIGHT_MAPPING) {
    CHECK(at_host_weight_tensors_[index].sizes() != std::vector<int64_t>({1}))
        << "weight is not loaded for " << name;
  }
}

}  // namespace layer
}  // namespace xllm
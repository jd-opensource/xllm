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

namespace xllm {
namespace layer {

NpuOneRecBlockLayerImpl::NpuOneRecBlockLayerImpl(const ModelContext& context,
                                                 bool is_decoder,
                                                 int32_t layer_id)
    : device_(context.get_tensor_options().device()),
      is_decoder_(is_decoder),
      layer_id_(layer_id) {}

torch::Tensor NpuOneRecBlockLayerImpl::forward(torch::Tensor& hidden_states,
                                               torch::Tensor& attn_mask,
                                               KVCache& kv_cache,
                                               ModelInputParams& input_params,
                                               torch::Tensor* encoder_output,
                                               int32_t node_id,
                                               aclrtEvent* event,
                                               std::atomic<bool>* event_flag) {
  return forward(hidden_states,
                 attn_mask,
                 kv_cache,
                 input_params,
                 encoder_output,
                 node_id,
                 event,
                 event_flag,
                 torch::Tensor());
}

torch::Tensor NpuOneRecBlockLayerImpl::forward(
    torch::Tensor& hidden_states,
    torch::Tensor& attn_mask,
    KVCache& kv_cache,
    ModelInputParams& input_params,
    torch::Tensor* encoder_output,
    int32_t node_id,
    aclrtEvent* event,
    std::atomic<bool>* event_flag,
    const torch::Tensor& expert_array) {
  (void)attn_mask;
  (void)kv_cache;
  (void)input_params;
  (void)node_id;
  (void)event;
  (void)event_flag;
  (void)expert_array;

  if (encoder_output != nullptr && encoder_output->defined() &&
      encoder_output->device() != device_) {
    *encoder_output = encoder_output->to(device_);
  }

  if (hidden_states.device() != device_) {
    hidden_states = hidden_states.to(device_);
  }

  if (!is_decoder_ && hidden_states.dim() > 1) {
    return hidden_states.contiguous();
  }

  return hidden_states;
}

void NpuOneRecBlockLayerImpl::load_state_dict(const StateDict& state_dict) {
  (void)state_dict;
}

void NpuOneRecBlockLayerImpl::verify_loaded_weights(
    const std::string& prefix) const {
  (void)prefix;
}

void NpuOneRecBlockLayerImpl::merge_loaded_weights() {}

}  // namespace layer
}  // namespace xllm

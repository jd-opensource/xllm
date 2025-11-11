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

#pragma once

#include <torch/torch.h>

#include "framework/model/model_input_params.h"
#include "framework/model_context.h"
#include "framework/state_dict/state_dict.h"
#include "npu_base_layer.h"
#include "xllm_kernels/core/include/atb_speed/base/hosttensor_binder.h"
#include "xllm_kernels/core/include/atb_speed/base/model.h"
#include "xllm_kernels/core/include/atb_speed/log.h"
#include "xllm_kernels/core/include/atb_speed/utils/model_factory.h"
#include "xllm_kernels/models/onerec/layer/block_layer.h"
#include "xllm_kernels/operations/fusion/utils.h"

namespace xllm {
namespace layer {

class NpuOneRecBlockLayerImpl : public NpuBaseLayer {
 public:
  explicit NpuOneRecBlockLayerImpl(const ModelContext& context,
                                   bool is_decoder = false,
                                   int layer_id = 0);

  ~NpuOneRecBlockLayerImpl() {};

  virtual void load_state_dict(const StateDict& state_dict) override;

  void verify_loaded_weights(const std::string& prefix) const;

  virtual void merge_loaded_weights() override;

  virtual int64_t init_layer() override;

  torch::Tensor forward(torch::Tensor& x,
                        torch::Tensor& attn_mask,
                        KVCache& kv_cache,
                        ModelInputParams& input_params,
                        atb::Context* context,
                        AtbWorkspace& workspace,
                        std::vector<aclrtEvent*> event,
                        std::vector<std::atomic<bool>*> event_flag,
                        torch::Tensor* encoder_output = nullptr,
                        int node_id = 0,
                        const torch::Tensor& expert_array = torch::Tensor());

 private:
  void param_from_args(atb_speed::onerec::BlockLayerParam& param,
                       const ModelArgs& args,
                       const ParallelArgs& parallel_args,
                       bool isPrefill,
                       const ModelInputParams* input_params = nullptr);

  void build_encoder_node_variant_pack(atb_speed::Model::Node& node,
                                       torch::Tensor& x,
                                       at::Tensor& attn_mask,
                                       ModelInputParams& input_params,
                                       bool is_prefill,
                                       int layer_id = 0);

  void build_decoder_node_variant_pack(atb_speed::Model::Node& node,
                                       torch::Tensor& x,
                                       at::Tensor& attn_mask,
                                       KVCache& kv_cache,
                                       ModelInputParams& input_params,
                                       bool is_prefill,
                                       torch::Tensor* encoder_output = nullptr,
                                       int layer_id = 0);

  void build_decoder_moe_node_variant_pack(
      atb_speed::Model::Node& node,
      torch::Tensor& x,
      at::Tensor& attn_mask,
      KVCache& kv_cache,
      ModelInputParams& input_params,
      bool is_prefill,
      torch::Tensor* encoder_output = nullptr,
      int layer_id = 0,
      const torch::Tensor& expert_array = torch::Tensor());

  int64_t init_node(atb_speed::Model::Node& node,
                    atb_speed::onerec::BlockLayerParam& param);
  int64_t init_attn_mask();
  int setup_common_decoder_tensors(atb_speed::Model::Node& node,
                                   torch::Tensor& x,
                                   at::Tensor& attn_mask,
                                   ModelInputParams& input_params,
                                   torch::Tensor* encoder_output = nullptr,
                                   int start_tensor_idx = 0);

  // Expert weights processing functions
  void resize_experts_weights(int num_of_device_experts);
  void process_expert_weights(const StateDict& state_dict,
                              const std::string& state_key,
                              const torch::Tensor& tensor);
  void process_shared_expert_weights(const StateDict& state_dict,
                                     const std::string& name,
                                     const torch::Tensor& tensor);
  void merge_experts_weights();
  void merge_shared_experts_weights();
  torch::Tensor merge_experts_weights(std::vector<torch::Tensor>& experts,
                                      bool transpose = false);
  torch::Tensor merge_experts_weights(std::vector<torch::Tensor>& experts_gate,
                                      std::vector<torch::Tensor>& experts_up,
                                      bool transpose = false);
  int extract_expert_index(const std::string& name);
  std::string extract_endswith(const std::string& input);

  atb_speed::Model::Node prefill_node_;
  atb_speed::Model::Node decode_node_;
  atb_speed::Model::Node decoder_prefill_only_decode_node_;
  std::string modelName_;
  atb_speed::onerec::BlockLayerParam prefill_param_;
  atb_speed::onerec::BlockLayerParam decode_param_;
  atb_speed::onerec::BlockLayerParam decoder_prefill_only_decode_param_;
  atb::Tensor internalTensors;
  atb::Tensor placeholder;

  at::Tensor encoder_output_contiguous_;  // Cache contiguous encoder_output to
                                          // avoid repeated contiguous() calls
  at::Tensor at_placeholder;
  std::vector<int32_t> seq_lens_vec_;     // Store sequence lengths for hostData
  std::vector<int32_t> placeholder_vec_;  // Store placeholder data for hostData
  std::vector<int32_t> encoder_seq_lens_vec_;

  int device_id_;
  bool is_decoder_;
  int layer_id_;

  // MoE expert weights storage
  std::unordered_map<std::string, std::vector<torch::Tensor>> experts_weights_;
  std::mutex experts_mutex_;
  int start_expert_id_;
  int end_expert_id_;
  int num_experts_per_partition_;
  int ep_size_;
  int ep_local_tp_rank_;
  int ep_local_tp_size_;

  // Shared expert weights storage
  std::vector<torch::Tensor> shared_expert_gate_weights_;
  std::vector<torch::Tensor> shared_expert_up_weights_;
  std::vector<torch::Tensor> shared_expert_down_weights_;

  // MoE routing tensors
  torch::Tensor expert_group_;
  torch::Tensor one_hot_;
  torch::Tensor zero_hot_;
};

}  // namespace layer
}  // namespace xllm
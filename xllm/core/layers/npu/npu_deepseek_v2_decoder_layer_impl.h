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

#include <glog/logging.h>
#include <torch/torch.h>
#include <torch_npu/torch_npu.h>

#include <nlohmann/json.hpp>

#include "framework/eplb/expert_buffer_manager.h"
#include "framework/eplb/expert_weight_buffer_shm.h"
#include "framework/model/model_input_params.h"
#include "framework/model/npu_dp_ep_padding.h"
#include "framework/model_context.h"
#include "framework/state_dict/state_dict.h"
#include "npu_base_layer.h"
#include "xllm_kernels/models/deepseekv2/layer/decoder_layer.h"

namespace xllm {
namespace layer {

class ExpertBuffer {
 public:
  torch::Tensor gateup_weight;
  torch::Tensor gateup_offset;
  torch::Tensor gateup_scale;
  torch::Tensor down_weight;
  torch::Tensor down_offset;
  torch::Tensor down_scale;

  static ExpertBuffer& Instance() {
    static ExpertBuffer instance;
    return instance;
  }

  void initialize_or_reuse(const std::vector<int64_t>& gateup_weight_shape,
                           const std::vector<int64_t>& gateup_offset_shape,
                           const std::vector<int64_t>& gateup_scale_shape,
                           const std::vector<int64_t>& down_weight_shape,
                           const std::vector<int64_t>& down_offset_shape,
                           const std::vector<int64_t>& down_scale_shape,
                           const torch::TensorOptions& weight_options,
                           const torch::TensorOptions& offset_options,
                           const torch::TensorOptions& scale_options,

                           bool force_reinit = false) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (force_reinit) {
      initialized_ = false;
    }

    if (!initialized_) {
      gateup_weight =
          torch::empty(gateup_weight_shape, weight_options).contiguous();
      gateup_offset =
          torch::empty(gateup_offset_shape, offset_options).contiguous();
      gateup_scale =
          torch::empty(gateup_scale_shape, scale_options).contiguous();
      down_weight =
          torch::empty(down_weight_shape, weight_options).contiguous();
      down_offset =
          torch::empty(down_offset_shape, offset_options).contiguous();
      down_scale = torch::empty(down_scale_shape, scale_options).contiguous();
      initialized_ = true;
    } else {
      auto validate_shape = [](const torch::Tensor& t,
                               const std::vector<int64_t>& expected) {
        TORCH_CHECK(t.sizes() == expected,
                    "Shape mismatch. Expected ",
                    expected,
                    " got ",
                    t.sizes());
      };

      validate_shape(gateup_weight, gateup_weight_shape);
      validate_shape(gateup_offset, gateup_offset_shape);
      validate_shape(down_weight, down_weight_shape);
      validate_shape(down_offset, down_offset_shape);
      // gateup_weight = at_npu::native::npu_format_cast(
      //   gateup_weight.contiguous(), 2);
      gateup_offset = gateup_offset.contiguous();
      gateup_scale = gateup_scale.contiguous();
      down_weight = down_weight.contiguous();
      down_offset = down_offset.contiguous();
      down_scale = down_scale.contiguous();
    }
  }

 private:
  std::mutex mutex_;
  bool initialized_ = false;
};

class NpuDeepseekV2DecoderLayerImpl : public NpuBaseLayer {
 public:
  explicit NpuDeepseekV2DecoderLayerImpl(const ModelContext& context,
                                         const int32_t layer_id,
                                         const float sm_scale);

  ~NpuDeepseekV2DecoderLayerImpl() {};

  virtual void load_state_dict(const StateDict& state_dict) override;

  void verify_loaded_weights(const std::string& prefix) const;

  virtual void merge_loaded_weights() override;

  void prepare_expert_weight(const std::vector<int32_t>& expert_list);

  void update_expert_weight();

  virtual int64_t init_layer() override;

  torch::Tensor forward(std::vector<torch::Tensor>& x,
                        std::vector<torch::Tensor>& cos_pos,
                        std::vector<torch::Tensor>& sin_pos,
                        std::vector<torch::Tensor>& attn_mask,
                        KVCache& kv_cache,
                        const std::vector<ModelInputParams>& input_params,
                        std::vector<aclrtEvent*> event = {nullptr, nullptr},
                        std::vector<std::atomic<bool>*> event_flag = {nullptr,
                                                                      nullptr},
                        int node_id = 0);

 private:
  struct ShardingConfig {
    bool is_sharded;
    int index;
    bool use_dp_sharding = false;
  };

  void initialize_tensors(const torch::TensorOptions& options);

  void initialize_weight_tensors(const torch::TensorOptions& options);

  void param_from_args(atb_speed::deepseekV2::DecoderLayerParam& param,
                       const ModelArgs& args,
                       const ParallelArgs& parallel_args,
                       bool is_prefill);

  void reserve_experts_weights(int num_of_device_experts);
  void initialize_device_expert_list(int numdevice, int num_layers);
  void initialize_basic_parameters(
      atb_speed::deepseekV2::DecoderLayerParam& param,
      const ModelArgs& args,
      const ParallelArgs& parallel_args,
      bool is_prefill);

  void initialize_attention_parameters(
      atb_speed::deepseekV2::DecoderLayerParam& param,
      const ModelArgs& args,
      const ParallelArgs& parallel_args);

  void initialize_mlp_parameters(
      atb_speed::deepseekV2::DecoderLayerParam& param,
      const ModelArgs& args,
      const ParallelArgs& parallel_args);

  void initialize_parallel_parameters(
      atb_speed::deepseekV2::DecoderLayerParam& param,
      const ParallelArgs& parallel_args);

  void initialize_quantization_parameters(
      atb_speed::deepseekV2::DecoderLayerParam& param);

  void initialize_kimi_k2_parameters(
      atb_speed::deepseekV2::DecoderLayerParam& param,
      const ModelArgs& args,
      bool is_prefill);

  torch::Tensor get_sharded_tensor(const StateDict& state_dict,
                                   const std::string& name,
                                   int dim);

  torch::Tensor get_sharded_tensor(const StateDict& state_dict,
                                   const std::string& name,
                                   int dim,
                                   int local_tp_rank,
                                   int local_tp_size);

  std::string extract_endswith(const std::string& input);

  std::string get_expert_shm_key(int32_t layer_id,
                                 int32_t expert_ids,
                                 const std::string& suffix);
  torch::Tensor build_expert_routing_map(std::vector<int32_t> expert_lists);
  void set_kv_weight(const StateDict& state_dict,
                     const std::string& tensor_name,
                     int weight_position,
                     int dim);

  int extract_expert_index(const std::string& name);

  void convert_descaled_weights_to_float();

  void convert_offsets_to_int8();

  torch::Tensor convert_fp16_to_int64(const torch::Tensor& fp16_tensor);

  void handle_device_specific_bias();

  void merge_shared_experts_weights();

  void merge_experts_weights();

  void squeeze_experts_weights();

  void preprocess_linear_for_rope();

  void process_expert_weights(const StateDict& state_dict,
                              const std::string& name,
                              const torch::Tensor& tensor);

  void process_shared_expert_weights(const StateDict& state_dict,
                                     const std::string& name,
                                     const torch::Tensor& tensor);

  void process_mlp_common_weights(const StateDict& state_dict,
                                  const std::string& name,
                                  const torch::Tensor& tensor);

  void process_general_weights(const StateDict& state_dict,
                               const std::string& name,
                               const torch::Tensor& tensor);

  int get_mapped_index(const std::string& name,
                       const std::unordered_map<std::string, int>& mapping);

  torch::Tensor view_tensor(torch::Tensor weight,
                            const std::string& name,
                            bool pre_view);

  torch::Tensor trans_rope_weight(torch::Tensor weight);

  torch::Tensor merge_experts_weights(std::vector<torch::Tensor>& experts,
                                      at::Device device,
                                      bool transpose = false);

  torch::Tensor merge_experts_weights(std::vector<torch::Tensor>& experts_up,
                                      std::vector<torch::Tensor>& experts_gate,
                                      at::Device device,
                                      bool transpose = false);

  void merge_and_copy_gate_up_weights(
      torch::Tensor& target_buffer,
      const std::vector<torch::Tensor>& experts_gate,
      const std::vector<torch::Tensor>& experts_up,
      bool do_transpose = false);
  void merge_and_copy_down_weights(
      torch::Tensor& target_buffer,
      const std::vector<torch::Tensor>& experts_down);

  int64_t init_node(atb_speed::Model::Node& node,
                    atb_speed::deepseekV2::DecoderLayerParam& param);

  void build_node_variant_pack(
      atb_speed::Model::Node& node,
      std::vector<torch::Tensor>& x,
      std::vector<torch::Tensor>& cos_pos,
      std::vector<torch::Tensor>& sin_pos,
      std::vector<torch::Tensor>& attn_mask,
      KVCache& kv_cache,
      const std::vector<ModelInputParams>& input_params,
      bool is_prefill);

  torch::Tensor block_tables_placeholder_;
  std::string model_name_;

  int32_t device_id_;
  int32_t layer_id_;
  int32_t num_key_value_heads_;
  int32_t qk_nope_head_dim_;
  int32_t v_head_dim_;
  int32_t kv_lora_rank_;
  int32_t qk_rope_head_dim_;

  int32_t rank_;
  int32_t first_k_dense_replace_;
  int32_t n_layers_;
  int32_t localWorldSize_;
  int32_t ep_size_;
  int32_t num_experts_;
  int32_t num_experts_per_partition_;
  int32_t ep_local_tp_size_;
  int32_t ep_local_tp_rank_;
  int32_t start_expert_id_;
  int32_t end_expert_id_;
  int32_t ep_rank_;
  int32_t redundant_experts_num_;

  int32_t dp_size_;
  int32_t dp_local_tp_size_;
  int32_t dp_rank_;
  int32_t dp_local_tp_rank_;

  float sm_scale_;
  int32_t num_speculative_tokens_ = 0;

  atb_speed::deepseekV2::DecoderLayerParam prefill_param_;
  atb_speed::deepseekV2::DecoderLayerParam decode_param_;
  atb_speed::deepseekV2::DecoderLayerParam decode_mla_param_;

  atb_speed::Model::Node prefill_node_;
  atb_speed::Model::Node decode_node_;
  atb_speed::Model::Node decode_mla_node_;

  atb::Tensor internal_tensor_;
  atb::Tensor internal_tensor_auxiliary_;

  torch::Tensor at_cumsum_;
  torch::Tensor tensor_placeholder_;
  torch::Tensor slot_tensor_placeholder_;
  torch::Tensor int_tensor_placeholder_;
  torch::Tensor decode_attn_mask_;
  torch::Tensor expert_group_;
  torch::Tensor one_hot_;
  torch::Tensor zero_hot_;
  torch::Tensor final_hidden_states_;
  torch::Tensor at_start_expert_id_;
  torch::Tensor at_in_device_expert_count_;

  std::vector<int32_t> int_placeholder_;
  std::vector<int32_t> device_expert_list_;

  std::unordered_map<std::string, torch::Tensor> shared_experts_weights_;
  std::unordered_map<std::string, std::vector<torch::Tensor>> experts_weights_;
  std::unordered_map<std::string, std::vector<torch::Tensor>>
      all_experts_weights_buffer_;

  std::mutex shared_experts_mutex_;
  std::mutex experts_mutex_;

  std::unique_ptr<ExpertBufferManager> shared_buffer_ = nullptr;
  torch::Tensor expert_routing_map_;
  torch::Tensor expert_routing_map_buffer_;
};

std::vector<torch::Tensor> get_dtp_inputs(torch::Tensor token_size_per_dp_group,
                                          int32_t dp_local_tp_size,
                                          int32_t dp_rank,
                                          int32_t dp_size,
                                          int32_t rank,
                                          at::Device device);

}  // namespace layer
}  // namespace xllm

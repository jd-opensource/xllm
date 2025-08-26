#pragma once
#ifdef TORCH_HIGHER_THAN_PTA6
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#else
#include <torch_npu/csrc/aten/NPUNativeFunctions.h>
#include <torch_npu/csrc/framework/utils/OpPreparation.h>
#endif

#include <torch_npu/csrc/libs/init_npu.h>

#include <functional>

#include "atb/atb_infer.h"
#include "framework/context.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_input_params.h"
#include "framework/state_dict/state_dict.h"
#include "nlohmann/json.hpp"
#include "npu_base_layer.h"
#include "pytorch/adapter/utils/utils.h"
#include "xllm_kernels/core/include/atb_speed/base/hosttensor_binder.h"
#include "xllm_kernels/core/include/atb_speed/base/model.h"
#include "xllm_kernels/core/include/atb_speed/log.h"
#include "xllm_kernels/core/include/atb_speed/utils/model_factory.h"
#include "xllm_kernels/models/llama/layer/decoder_layer.h"

namespace xllm {
namespace layer {

class NpuLlamaDecoderLayerImpl : public NpuBaseLayer {
 public:
  explicit NpuLlamaDecoderLayerImpl(const Context& context);

  ~NpuLlamaDecoderLayerImpl() {};

  virtual void load_state_dict(const StateDict& state_dict) override;

  virtual void verify_loaded_weights() const override;

  virtual void merge_loaded_weights() override;

  virtual int64_t init_layer() override;

  torch::Tensor forward(torch::Tensor& x,
                        torch::Tensor& cos_pos,
                        torch::Tensor& sin_pos,
                        torch::Tensor& attn_mask,
                        KVCache& kv_cache,
                        ModelInputParams& input_params,
                        atb::Context* context,
                        AtbWorkspace& workspace,
                        int node_id = 0);

 private:
  void build_node_variant_pack(atb_speed::Model::Node& node,
                               torch::Tensor& x,
                               torch::Tensor& cos_pos,
                               torch::Tensor& sin_pos,
                               torch::Tensor& attn_mask,
                               KVCache& kv_cache,
                               ModelInputParams& input_params,
                               bool is_prefill);

  int64_t init_node(atb_speed::Model::Node& node,
                    atb_speed::llama::LlamaLayerParam& param);

  int64_t init_attn_mask();

  void param_from_args(atb_speed::llama::LlamaLayerParam& param,
                       const ModelArgs& args,
                       const ParallelArgs& parallel_args,
                       bool isPrefill);

  atb_speed::Model::Node prefill_node_;
  atb_speed::Model::Node decode_node_;
  std::string model_name_;
  atb_speed::llama::LlamaLayerParam prefill_param_;
  atb_speed::llama::LlamaLayerParam decode_param_;
  atb::Tensor internal_tensors_;
  atb::Tensor placeholder_;

  // at::Tensor encode_attn_mask_;
  at::Tensor decode_attn_mask_;

  at::Tensor at_placeholder_;

  int device_id_;
};

}  // namespace layer
}  // namespace xllm

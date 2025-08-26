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
#include "atb_speed/base/hosttensor_binder.h"
#include "atb_speed/base/model.h"
#include "atb_speed/log.h"
#include "atb_speed/utils/model_factory.h"
#include "core/framework/model/model_args.h"
#include "core/framework/model/model_input_params.h"
#include "core/framework/parallel_state.h"
#include "core/framework/state_dict/state_dict.h"
#include "nlohmann/json.hpp"
#include "npu_base_layer.h"
#include "pytorch/adapter/utils/utils.h"
#include "xllm_kernels/models/qwen2_5/vision_encoder/encoder_layer.h"

namespace xllm {
namespace layer {

enum VisionEncoderLayerTensorId : int {
  IN_INPUT_NORM_WEIGHT = 0,
  IN_POST_NORM_WEIGHT,
  IN_QKV_WEIGHT,
  IN_QKV_BIAS,
  IN_WATTENTION_OUT_WEIGHT,
  IN_WATTENTION_OUT_BIAS,
  IN_MLP_GATE_WEIGHT,
  IN_MLP_GATE_BIAS,
  IN_MLP_UP_WEIGHT,
  IN_MLP_UP_BIAS,
  IN_MLP_DOWN_WEIGHT,
  IN_MLP_DOWN_BIAS,
  IN_VISION_Q_WEIGHT,
  IN_VISION_Q_BIAS,
  IN_VISION_K_WEIGHT,
  IN_VISION_K_BIAS,
  IN_VISION_V_WEIGHT,
  IN_VISION_V_BIAS
};

class NpuQwen2dot5VisionEncoderLayerImpl : public NpuBaseLayer {
 public:
  explicit NpuQwen2dot5VisionEncoderLayerImpl(const Context& context);

  ~NpuQwen2dot5VisionEncoderLayerImpl() {};

  virtual void load_state_dict(const StateDict& state_dict) override;

  virtual void verify_loaded_weights() const override;

  virtual void merge_loaded_weights() override;

  virtual int64_t init_layer() override;

  torch::Tensor forward(torch::Tensor& x,
                        torch::Tensor& cos_pos,
                        torch::Tensor& sin_pos,
                        torch::Tensor& cu_seqlen,
                        std::vector<int>& cu_seqlen_vec,
                        ModelInputParams& input_params,
                        atb::Context* context,
                        AtbWorkspace& workspace,
                        int node_id = 0,
                        aclrtEvent* event = nullptr,
                        std::atomic<bool>* event_flag = nullptr);

 private:
  void build_node_variant_pack(atb_speed::Model::Node& node,
                               torch::Tensor& x,
                               torch::Tensor& cos_pos,
                               torch::Tensor& sin_pos,
                               torch::Tensor& cu_seqlen,
                               std::vector<int>& cu_seqlen_vec,
                               ModelInputParams& input_params,
                               bool is_prefill);

  void get_weights_col_packed_qkv();

  void param_from_args(atb_speed::qwen::VisionEncoderLayerParam& param,
                       const ModelArgs& args,
                       const ParallelArgs& parallel_args);

  int64_t init_node(atb_speed::Model::Node& node,
                    atb_speed::qwen::VisionEncoderLayerParam& param);

  void pad_qkv_weights();

  void pad_mlp_weights();

  torch::Tensor pad_tensor(const torch::Tensor& tensor,
                           int64_t target_shape,
                           int64_t dim = 0) {
    int64_t pad_size = target_shape - tensor.size(dim);
    if (tensor.dim() == 1) {
      return torch::nn::functional::pad(
          tensor, torch::nn::functional::PadFuncOptions({0, pad_size}));
    } else if (tensor.dim() == 2) {
      if (1 == dim)
        return torch::nn::functional::pad(
            tensor, torch::nn::functional::PadFuncOptions({0, pad_size, 0, 0}));
      else
        return torch::nn::functional::pad(
            tensor, torch::nn::functional::PadFuncOptions({0, 0, 0, pad_size}));
    }
    return tensor;
  }

  atb_speed::Model::Node encode_node_;
  std::string model_name_;

  atb_speed::qwen::VisionEncoderLayerParam encode_param_;
  atb::Tensor internal_tensors_;
  atb::Tensor placeholder_;
  at::Tensor cu_seqlen_;
  at::Tensor at_placeholder_;
  std::vector<torch::Tensor> qkv_weight;
  std::vector<torch::Tensor> qkv_bias;
  int device_id_;
};

// class Qwen2_5VisionEncoder
//     : public torch::nn::ModuleHolder<NpuQwen2dot5VisionEncoderLayerImpl> {
//  public:
//   using
//   torch::nn::ModuleHolder<NpuQwen2dot5VisionEncoderLayerImpl>::ModuleHolder;
//   using Impl __attribute__((__unused__)) =
//   NpuQwen2dot5VisionEncoderLayerImpl;

//   Qwen2_5VisionEncoder(const Context& context);
// };

// std::shared_ptr<NpuQwen2dot5VisionEncoderLayerImpl>
// create_qwen2_5_vision_encoder_layer(
//     const Context& context);

}  // namespace layer
}  // namespace xllm

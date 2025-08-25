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

namespace xllm {
namespace layer {

class NpuRmsNormImpl : public NpuBaseLayer {
 public:
  explicit NpuRmsNormImpl(const Context& context);

  ~NpuRmsNormImpl() {};

  virtual void load_state_dict(const StateDict& state_dict) override;

  void verify_loaded_weights(const std::string weight_str) const;

  virtual void merge_loaded_weights() override;

  virtual int64_t init_layer() override;

  torch::Tensor forward(torch::Tensor& x,
                        atb::Context* context,
                        AtbWorkspace& workspace,
                        int nodeId);

 private:
  void build_node_variant_pack(atb_speed::Model::Node& node, torch::Tensor& x);

  int64_t init_node(atb_speed::Model::Node& node,
                    atb::infer::RmsNormParam& param);

  void param_from_args(atb::infer::RmsNormParam& param, const ModelArgs& args);

  atb_speed::Model::Node norm_node_;
  std::string model_name_;
  atb::infer::RmsNormParam norm_param_;
  atb::Tensor internal_tensors_;
};

}  // namespace layer
}  // namespace xllm

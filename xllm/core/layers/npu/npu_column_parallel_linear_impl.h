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
#include "framework/model/model_input_params.h"
#include "framework/state_dict/state_dict.h"
#include "nlohmann/json.hpp"
#include "npu_base_layer.h"
#include "pytorch/adapter/utils/utils.h"
#include "xllm_kernels/core/include/atb_speed/base/hosttensor_binder.h"
#include "xllm_kernels/core/include/atb_speed/base/model.h"
#include "xllm_kernels/core/include/atb_speed/log.h"
#include "xllm_kernels/core/include/atb_speed/utils/model_factory.h"
#include "xllm_kernels/operations/fusion/linear/linear_parallel.h"

namespace xllm::layer {
// Linear layer with column parallelism.
// The linear layer is defined as Y = XA + b. A is parallelized along
// its second dimension as A = [A_1, ..., A_p].
class NpuColumnParallelLinearImpl : public NpuBaseLayer {
 public:
  NpuColumnParallelLinearImpl(const Context& context);

  ~NpuColumnParallelLinearImpl() {};

  virtual void load_state_dict(const StateDict& state_dict) override;

  void verify_loaded_weights(const std::string weight_str) const;

  virtual void merge_loaded_weights() override;

  virtual int64_t init_layer() override;

  virtual torch::Tensor forward(const torch::Tensor& input,
                                atb::Context* context,
                                AtbWorkspace& workspace,
                                int nodeId);

 protected:
  void build_node_variant_pack(atb_speed::Model::Node& node,
                               const torch::Tensor& input);

  int64_t init_node(atb_speed::Model::Node& node,
                    atb_speed::common::LinearParallelParam& linearParam);

  void param_from_args(atb_speed::common::LinearParallelParam& param,
                       const ModelArgs& args,
                       const ParallelArgs& parallel_args);

  atb_speed::Model::Node linear_node_;
  std::string model_name_;

  std::vector<at::Tensor> at_out_tensors_;
  atb::Tensor internal_input;
  torch::Tensor tensor_placeholder_;
  atb::Tensor placeholder_;

  atb_speed::common::LinearParallelParam linear_param_;
};

// class AtbColumnParallelLinear
//     : public torch::nn::ModuleHolder<NpuColumnParallelLinearImpl> {
//  public:
//   using torch::nn::ModuleHolder<NpuColumnParallelLinearImpl>::ModuleHolder;
//   using Impl __attribute__((__unused__)) = NpuColumnParallelLinearImpl;

//   AtbColumnParallelLinear(const Context& context);
// };

// std::shared_ptr<NpuColumnParallelLinearImpl>
// create_atb_column_parallel_linear_layer(const Context& context);
}  // namespace xllm::layer
#pragma once
#include <torch/torch.h>

#include "framework/context.h"
#include "framework/model/model_input_params.h"
#include "framework/state_dict/state_dict.h"
#include "npu_base_layer.h"
#include "xllm_kernels/pytorch/atb_torch/core/include/base_operation.h"
#include "xllm_kernels/pytorch/atb_torch/core/include/graph_operation.h"

namespace xllm {
namespace layer {

class NpuSiglipEncoderLayerUpImpl : public NpuBaseLayer {
 public:
  NpuSiglipEncoderLayerUpImpl(const Context& context,
                              const std::string& prefix = "");

  ~NpuSiglipEncoderLayerUpImpl() {};

  virtual void load_state_dict(const StateDict& state_dict) override;

  torch::Tensor forward(torch::Tensor& x);

 private:
  void build_graph(const std::string& prefix = "");

  atb_torch::GraphOperation graph_;
  std::vector<std::shared_ptr<atb_torch::BaseOperation>> ops_;
  std::vector<torch::Tensor> weights_;

  ModelArgs model_args_;
  torch::TensorOptions options_;

  std::string prefix_;
};

class NpuSiglipEncoderLayerUp
    : public torch::nn::ModuleHolder<NpuSiglipEncoderLayerUpImpl> {
 public:
  using torch::nn::ModuleHolder<NpuSiglipEncoderLayerUpImpl>::ModuleHolder;
  using Impl __attribute__((__unused__)) = NpuSiglipEncoderLayerUpImpl;

  NpuSiglipEncoderLayerUp(const Context& context,
                          const std::string& prefix = "")
      : ModuleHolder(
            std::make_shared<NpuSiglipEncoderLayerUpImpl>(context, prefix)) {}
};

class NpuSiglipEncoderLayerDownImpl : public NpuBaseLayer {
 public:
  NpuSiglipEncoderLayerDownImpl(const Context& context,
                                const std::string& prefix = "");

  ~NpuSiglipEncoderLayerDownImpl() {};

  virtual void load_state_dict(const StateDict& state_dict) override;

  torch::Tensor forward(torch::Tensor& x, torch::Tensor& y);

 private:
  void build_graph(const std::string& prefix = "");

  std::string prefix_;

  atb_torch::GraphOperation graph_;
  std::vector<std::shared_ptr<atb_torch::BaseOperation>> ops_;
  std::vector<torch::Tensor> weights_;

  ModelArgs model_args_;
  torch::TensorOptions options_;
};

class NpuSiglipEncoderLayerDown
    : public torch::nn::ModuleHolder<NpuSiglipEncoderLayerDownImpl> {
 public:
  using torch::nn::ModuleHolder<NpuSiglipEncoderLayerDownImpl>::ModuleHolder;
  using Impl __attribute__((__unused__)) = NpuSiglipEncoderLayerDownImpl;

  NpuSiglipEncoderLayerDown(const Context& context,
                            const std::string& prefix = "")
      : ModuleHolder(
            std::make_shared<NpuSiglipEncoderLayerDownImpl>(context, prefix)) {}
};

class NpuSiglipEncoderLayerImpl : public NpuBaseLayer {
 public:
  NpuSiglipEncoderLayerImpl(const Context& context,
                            const std::string& prefix = "");

  ~NpuSiglipEncoderLayerImpl() {};

  virtual void load_state_dict(const StateDict& state_dict) override;

  void verify_loaded_weights(const std::string& weight_str) const {};

  torch::Tensor forward(torch::Tensor& x);

 private:
  std::string prefix_;

  ModelArgs model_args_;
  torch::TensorOptions options_;

  NpuSiglipEncoderLayerUp up_{nullptr};
  NpuSiglipEncoderLayerDown down_{nullptr};
};

}  // namespace layer
}  // namespace xllm

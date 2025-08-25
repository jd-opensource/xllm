#pragma once

#include <torch/torch.h>

namespace xllm {
namespace F = torch::nn::functional;
class DiTLinearImpl : public torch::nn::Module {
 public:
  torch::Tensor weight;
  torch::Tensor bias;
  DiTLinearImpl(int64_t in, int64_t out, bool with_bias = true) {
    weight = register_parameter("weight", torch::empty({out, in}));
    if (with_bias) {
      bias = register_parameter("bias", torch::empty(out));
    } else {
      bias = register_parameter("bias", {}, false);
    }
  }

  torch::Tensor forward(const torch::Tensor& x) {
    return F::linear(x, weight, bias);
  }
};
TORCH_MODULE(DiTLinear);
}  // namespace xllm
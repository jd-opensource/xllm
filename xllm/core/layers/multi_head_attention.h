#pragma once

#include <torch/torch.h>

#include "framework/context.h"
#include "framework/state_dict/state_dict.h"

namespace xllm {
namespace layer {

class MultiheadAttentionImpl : public torch::nn::Module {
 public:
  MultiheadAttentionImpl(const Context& context);

  torch::Tensor forward(torch::Tensor query,
                        torch::Tensor key,
                        torch::Tensor value,
                        torch::Tensor key_padding_mask);

  void load_state_dict(const StateDict& state_dict);

  void verify_loaded_weights(const std::string& prefix) const;

 private:
  int64_t n_head_;
  int64_t head_dim_;
  int64_t hidden_size_;
  torch::TensorOptions options_;

  torch::Tensor in_proj_weight_;
  torch::Tensor in_proj_bias_;
  torch::Tensor out_proj_weight_;
  torch::Tensor out_proj_bias_;

  bool is_in_proj_weight_loaded_;
  bool is_in_proj_bias_loaded_;
  bool is_out_proj_weight_loaded_;
  bool is_out_proj_bias_loaded_;
};

TORCH_MODULE(MultiheadAttention);

}  // namespace layer
}  // namespace xllm
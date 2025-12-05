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

#include "core/framework/state_dict/utils.h"
#include "core/layers/common/add_matmul.h"
#include "framework/parallel_state/parallel_state.h"

namespace xllm {
namespace F = torch::nn::functional;

enum class LinearType { Default, SequenceParallel };

// NOTE: The order of linear and all2all Operations depends on the
// before_attention param if before_attention is true, order is: linear->all2all
// if before_attention is false, order is: all2all->linear
struct SpOptions {
  // the num of attention heads
  int64_t head_num = 0;

  // the size of single attention head
  int64_t head_dim = 0;

  // hidden_size
  int64_t hidden_size = 0;

  // before_attention: a Bool value that indicates where to apply the all2all,
  //  According to the classic ulysses sequence parallel, we should apply
  //  all2all communication for q, k, v, text_q (optional), text_k (optional),
  //  text_v (optional), before attention operation to gather full sequence
  //  (splited_sequence * group_size) and scatter the head nums (head_nums /
  //  group_size) , and we should apply all2all communication for attn_output,
  //  text_attn_output (optional) after the attention operation to split the
  //  full sequence (full_sequence / group_size) , and gather the head nums
  //  (splited_head_num * group_size)
  bool before_attention = false;

  // the process_group for sequence parallel
  ProcessGroup* process_group = nullptr;

  SpOptions() = default;

  SpOptions(int64_t head_num,
            int64_t head_dim,
            int64_t hidden_size,
            bool before_attention,
            ProcessGroup* process_group = nullptr)
      : head_num(head_num),
        head_dim(head_dim),
        hidden_size(hidden_size),
        before_attention(before_attention),
        process_group(process_group) {}

  void valid() const {
    CHECK(head_num > 0) << "head_num should be greater than 0 to initialize "
                           "DiTParallelLinear for "
                           "linear type 'sequence_parallel' "
                        << " but got " << head_num;
    CHECK(head_dim > 0) << "head_dim should be greater than 0 to initialize "
                           "DiTParallelLinear for "
                           "linear type 'sequence_parallel' "
                        << " but got " << head_dim;
    CHECK(hidden_size > 0) << "head_size should be greater than 0 to "
                              "initialize DiTParallelLinear for "
                              "linear type 'sequence_parallel' "
                           << " but got " << hidden_size;
    CHECK(hidden_size == head_dim * head_num)
        << "hidden_size should equal to head_dim * head_num"
        << "got head_dim " << head_dim << ", head num" << head_num
        << ", hidden_size " << hidden_size;
    if (!process_group) {
      LOG(ERROR)
          << "DiTSpLinear expected to receive an initialized processgroup for"
          << "all2all communication, but got nullptr";
    }
  }
};

// TODO : Need to Implement a template funciton, but
// libtorch doesn't allow to creat module holder for
// template class.
// template <typename Linear>
class DiTParallelLinearImpl : public torch::nn::Module {
 public:
  DiTParallelLinearImpl(layer::WeightTransposeAddMatmul linear,
                        const string& module_name,
                        LinearType linear_type = LinearType::Default,
                        const SpOptions& sp_options = SpOptions())
      : sp_options_(sp_options), linear_type_(linear_type) {
    linear_ = register_module(module_name, std::move(linear));
    if (linear_type == LinearType::SequenceParallel) {
      sp_options_.valid();
    }
  }

  torch::Tensor linear_forward(const torch::Tensor& input) {
    return linear_->forward(input);
  }

  // sp_forward combines the linear operation with all2all communication,
  // output: A torch tensor with shape {batch, seq_len, hidden_size}
  torch::Tensor sp_forward(const torch::Tensor& input) {
    CHECK(input.sizes().size() == 3)
        << "Sp linear input is expected to be a tensor "
        << "with shape {batch, seq_len, hidden_size}";
    auto group_size = sp_options_.process_group->world_size();
    if (sp_options_.before_attention) {
      auto linear_output = this->linear_forward(input);
      auto all_to_all_func = parallel_state::all_to_all_4D(
          /*input=*/linear_output.view(
              {input.size(0), -1, sp_options_.head_num, sp_options_.head_dim}),
          /*scatter_dim=*/2,
          /*gather_dim=*/1,
          /*async_ops=*/false,
          sp_options_.process_group);
      auto output = all_to_all_func();
      return output.view(
          {input.size(0), -1, sp_options_.hidden_size / group_size});
    } else {
      auto all_to_all_func = parallel_state::all_to_all_4D(
          /*input=*/input.view({input.size(0),
                                -1,
                                sp_options_.head_num / group_size,
                                sp_options_.head_dim}),
          /*scatter_dim=*/1,
          /*gather_dim=*/2,
          /*async_ops=*/false,
          sp_options_.process_group);
      auto all_to_all_output = all_to_all_func();
      all_to_all_output =
          all_to_all_output.view({input.size(0), -1, sp_options_.hidden_size});
      auto output = this->linear_forward(all_to_all_output);
      return output;
    }
  }

  torch::Tensor forward(const torch::Tensor& input) {
    if (linear_type_ == LinearType::Default) {
      return this->linear_forward(input);
    } else {
      return this->sp_forward(input);
    }
  }

  void load_state_dict(const StateDict& state_dict) {
    linear_->load_state_dict(state_dict);
  }

  void verify_loaded_weights(const std::string& prefix) const {
    linear_->verify_loaded_weights(prefix);
  }

 private:
  layer::WeightTransposeAddMatmul linear_{nullptr};
  SpOptions sp_options_;
  LinearType linear_type_;
};

TORCH_MODULE(DiTParallelLinear);
}  // namespace xllm

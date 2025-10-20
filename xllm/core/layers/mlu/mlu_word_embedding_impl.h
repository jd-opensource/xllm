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

#include <cstdint>

#include "framework/parallel_state/parallel_args.h"
#include "framework/parallel_state/parallel_state.h"
#include "framework/state_dict/state_dict.h"

namespace xllm {
namespace layer {

// Embedding parallelized in the embedding dimension.
class MluWordEmbeddingImpl : public torch::nn::Module {
 public:
  MluWordEmbeddingImpl(int64_t num_embeddings,
                       int64_t embedding_dim,
                       const ParallelArgs& parallel_args,
                       const torch::TensorOptions& options)
      : parallel_args_(parallel_args) {
    rank_ = parallel_args_.tp_group_->rank();
    world_size_ = parallel_args_.tp_group_->world_size();

    CHECK(embedding_dim % world_size_ == 0)
        << "out_features " << embedding_dim << " not divisible by world_size "
        << world_size_;
    const int64_t embedding_dim_per_partition = embedding_dim / world_size_;

    // register the weight parameter
    weight_ = register_parameter(
        "weight",
        torch::empty({num_embeddings, embedding_dim_per_partition}, options),
        /*requires_grad=*/false);
  }

  // The input to the module is a list of indices, and the output is the
  // corresponding word embeddings.
  torch::Tensor forward(torch::Tensor input) {
    namespace F = torch::nn::functional;
    auto output = F::embedding(input, weight_);
    if (world_size_ > 1) {
      output = xllm::parallel_state::gather(output, parallel_args_.tp_group_);
    }
    return output;
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    const auto weight =
        state_dict.get_sharded_tensor("weight",
                                      /*dim=*/1,
                                      /*rank=*/rank_,
                                      /*world_size=*/world_size_);
    if (weight.defined()) {
      CHECK_EQ(weight_.sizes(), weight.sizes())
          << "weight size mismatch for " << name();
      weight_.copy_(weight);
      is_loaded_ = true;
    }
  }

  // whether the weight is loaded
  void verify_loaded_weights(const std::string& prefix) const {
    CHECK(is_loaded_) << "weight is not loaded for " << prefix + "weight";
  }

  void pretty_print(std::ostream& stream) const override {
    stream << name() << " " << weight_.sizes() << " " << weight_.device();
  }

  // return the weight (for testing)
  torch::Tensor weight() const { return weight_; }

 private:
  // rank of current process
  PROPERTY(int32_t, rank) = 0;

  // world size
  PROPERTY(int32_t, world_size) = 0;
  // parameter members, must be registered
  torch::Tensor weight_{nullptr};

  // whether the weight is loaded
  bool is_loaded_ = false;

  // parallel args
  ParallelArgs parallel_args_;
};

}  // namespace layer
}  // namespace xllm

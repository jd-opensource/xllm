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

#include "framework/parallel_state/parallel_args.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"
#include "framework/state_dict/utils.h"

namespace xllm {
namespace layer {

// an interface for parallel linear layer.
// all linear classes should inherit from this class and implement the forward
// function.
class ParallelLinearImpl : public torch::nn::Module {
 public:
  ~ParallelLinearImpl() override = default;

  virtual torch::Tensor forward(torch::Tensor input) = 0;

  virtual void load_state_dict(const StateDict& state_dict) = 0;

  virtual void verify_loaded_weights(const std::string& prefix = "") const = 0;

  // special load_state_dict for fused cases
  virtual void load_state_dict(const StateDict& /*state_dict*/,
                               const std::vector<std::string>& /*prefixes*/) {
    LOG(FATAL) << "not implemented";
  }
};

// Linear layer with column parallelism.
// The linear layer is defined as Y = XA + b. A is parallelized along
// its second dimension as A = [A_1, ..., A_p].
class ColumnParallelLinearImpl : public ParallelLinearImpl {
 public:
  ColumnParallelLinearImpl(int64_t in_features,
                           int64_t out_features,
                           bool bias,
                           bool gather_output,
                           const ParallelArgs& parallel_args,
                           const torch::TensorOptions& options);

  torch::Tensor forward(torch::Tensor input) override;

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) override;

  // special load_state_dict for fused cases
  void load_state_dict(const StateDict& state_dict,
                       const std::vector<std::string>& prefixes) override;

  // whether the weight is loaded
  void verify_loaded_weights(const std::string& prefix) const override {
    CHECK(weight_is_loaded_)
        << "weight is not loaded for " << prefix + "weight";
    CHECK(!bias_.defined() || bias_is_loaded_)
        << "bias is not loaded for " << prefix + "bias";
  }

  void pretty_print(std::ostream& stream) const override {
    stream << name() << " " << weight_.sizes() << " " << weight_.device();
  }

  // return the weight (for testing)
  torch::Tensor weight() const { return weight_; }

 private:
  // parameter members, must be registered
  // we allocate the transpose since linear performs XA^T.
  // A^T: [out_features_per_partition, in_features]
  DEFINE_FUSED_WEIGHT(weight);
  DEFINE_FUSED_WEIGHT(bias);

  int rank_;
  int world_size_;
  // whether to gather the output
  bool gather_output_;
  at::Device device_;
  // parallel args
  ParallelArgs parallel_args_;
};

class QKVParallelLinearImpl : public ParallelLinearImpl {
 public:
  QKVParallelLinearImpl(int64_t hidden_size,
                        int64_t num_heads,
                        int64_t num_kv_heads,
                        int64_t head_size,
                        int64_t num_kv_head_replicas,
                        bool bias,
                        bool gather_output,
                        const ParallelArgs& parallel_args,
                        const torch::TensorOptions& options);

  torch::Tensor forward(torch::Tensor input) override;

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) override;
  bool load_qkv_weight(const StateDict& state_dict, int32_t index);

  // whether the weight is loaded
  void verify_loaded_weights(const std::string& prefix) const override {
    CHECK(qkv_weight_is_loaded_)
        << "weight is not loaded for " << prefix + "weight";
    CHECK(!qkv_bias_.defined() || qkv_bias_is_loaded_)
        << "bias is not loaded for " << prefix + "bias";
  }

  void pretty_print(std::ostream& stream) const override {
    stream << name() << " " << weight().sizes() << " " << weight().device();
  }

  // return the weight (for testing)
  torch::Tensor weight() const { return qkv_weight_; }

 private:
  // parameter members, must be registered
  // we allocate the transpose since linear performs XA^T.
  // A^T: [out_features_per_partition, in_features]
  DEFINE_FUSED_WEIGHT(qkv_weight);
  DEFINE_FUSED_WEIGHT(qkv_bias);

  int rank_;
  int world_size_;
  int64_t hidden_size_;
  int64_t num_heads_;
  int64_t num_kv_heads_;
  int64_t head_size_;
  int64_t num_kv_head_replicas_;
  // whether to gather the output
  bool gather_output_;
  at::Device device_;
  // parallel args
  ParallelArgs parallel_args_;
  torch::TensorOptions options_;
};

// Linear layer with row parallelism.
//     The linear layer is defined as Y = XA + b. A is parallelized along
//     its first dimension and X along its second dimension as:
//                -   -
//               | A_1 |
//               | .   |
//           A = | .   |       X = [X_1, ..., X_p]
//               | .   |
//               | A_p |
//                -   -
class RowParallelLinearImpl : public ParallelLinearImpl {
 public:
  RowParallelLinearImpl(int64_t in_features,
                        int64_t out_features,
                        bool bias,
                        bool input_is_parallelized,
                        bool if_reduce_results,
                        const ParallelArgs& parallel_args,
                        const torch::TensorOptions& options);

  torch::Tensor forward(torch::Tensor input) override;

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) override;

  // whether the weight is loaded
  void verify_loaded_weights(const std::string& prefix = "") const override {
    CHECK(weight_is_loaded_)
        << "weight is not loaded for " << prefix + "weight";
    CHECK(!bias_.defined() || bias_is_loaded_)
        << "bias is not loaded for " << prefix + "bias";
  }

  void pretty_print(std::ostream& stream) const override {
    stream << name() << " " << weight_.sizes() << " " << weight_.device();
  }

  // return the weight (for testing)
  torch::Tensor weight() const { return weight_; }

 private:
  // parameter members, must be registered
  // we allocate the transpose since linear performs XA^T.
  // A^T: [out_features, in_features_per_partition]
  DEFINE_WEIGHT(weight);
  DEFINE_WEIGHT(bias);

  // whether the input is already parallelized
  bool input_is_parallelized_;

  // whether to reduce the results
  bool if_reduce_results_;

  // parallel args
  ParallelArgs parallel_args_;

  int rank_;
  int world_size_;
};

class ReplicatedLinearImpl : public ParallelLinearImpl {
 public:
  ReplicatedLinearImpl(int64_t in_features,
                       int64_t out_features,
                       bool bias,
                       const QuantArgs& quant_args,
                       const torch::TensorOptions& options);

  torch::Tensor forward(torch::Tensor input) override;

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) override;

  // whether the weight is loaded
  void verify_loaded_weights(const std::string& prefix = "") const override {
    CHECK(weight_is_loaded_)
        << "weight is not loaded for " << prefix + "weight";
    CHECK(!bias_.defined() || bias_is_loaded_)
        << "bias is not loaded for " << prefix + "bias";
  }

  void pretty_print(std::ostream& stream) const override {
    stream << name() << " " << weight_.sizes() << " " << weight_.device();
  }

  // return the weight (for testing)
  torch::Tensor weight() const { return weight_; }

 private:
  // parameter members, must be registered
  // we allocate the transpose since linear performs XA^T.
  // A^T: [out_features, in_features]
  DEFINE_WEIGHT(weight);
  DEFINE_WEIGHT(bias);
};

}  // namespace layer
}  // namespace xllm
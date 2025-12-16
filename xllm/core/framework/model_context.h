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

#if defined(USE_NPU)
#include <acl/acl.h>

#include "layers/npu/buffer/atb_workspace.h"
#endif

#include <memory>

#include "core/framework/model/model_args.h"
#include "core/framework/parallel_state/parallel_args.h"
#include "core/framework/quant_args.h"
#include "framework/parallel_state/parallel_args.h"

namespace xllm {

class ModelContext {
 public:
  ModelContext() : parallel_args_(1, 1, nullptr) {};

  ModelContext(const ParallelArgs& input_parallel_args,
               const ModelArgs& model_args,
               const QuantArgs& quant_args,
               const torch::TensorOptions& tensor_options);

#if defined(USE_NPU)
  ModelContext(const ParallelArgs& input_parallel_args,
               const ModelArgs& model_args,
               const QuantArgs& quant_args,
               const torch::TensorOptions& tensor_options,
               atb::Context* context);
#endif

  const ModelArgs& get_model_args() const { return model_args_; }

  const QuantArgs& get_quant_args() const { return quant_args_; }

  const ParallelArgs& get_parallel_args() const { return parallel_args_; }

  const torch::TensorOptions& get_tensor_options() const {
    return tensor_options_;
  }

#if defined(USE_NPU)
  const atb::Context* get_atb_context() const { return context_; }
  std::shared_ptr<AtbWorkspace> get_atb_workspace() const {
    return atb_workspace_;
  }
#endif

  void set_image_embedding_mode(bool image_embedding_mode) {
    model_args_.image_embedding_mode() = image_embedding_mode;
  }

 private:
  ModelArgs model_args_;
  QuantArgs quant_args_;
  ParallelArgs parallel_args_;
  torch::TensorOptions tensor_options_;

#if defined(USE_NPU)
  // used for npu atb
  atb::Context* context_;
  std::shared_ptr<AtbWorkspace> atb_workspace_;
#endif
};

}  // namespace xllm

/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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
#include <acl/acl.h>
#include <torch/torch.h>

#include <algorithm>
#include <memory>
#include <string>

#include "autoencoder_kl_qwenimage.h"
#include "core/common/global_flags.h"
#include "core/framework/dit_cache/dit_cache.h"
#include "core/framework/dit_model_loader.h"
#include "core/framework/model/model_input_params.h"
#include "core/framework/model_context.h"
#include "core/framework/request/dit_request_state.h"
#include "core/framework/state_dict/state_dict.h"
#include "core/framework/state_dict/utils.h"
#include "framework/parallel_state/parallel_state.h"
#include "models/dit/autoencoder_kl.h"
#include "models/dit/flowmatch_euler_discrete_scheduler.h"
#include "models/dit/utils/common_util.h"
#include "models/model_registry.h"
#include "processors/qwen2_vl_image_processor.h"
#include "transformer_qwen_image.h"
namespace xllm::dit::npu {
namespace qwenimage {

class QwenImagePipelineBaseImpl : public torch::nn::Module {
 protected:
  torch::Device device_ = torch::kCPU;
  torch::ScalarType dtype_;
  torch::TensorOptions options_;
  AutoencoderKLQwenImage vae_{nullptr};
  xllm::VAEImageProcessor vae_image_processor_{nullptr};
  std::unique_ptr<Qwen2VLImageProcessor> qwen_image_processor_{nullptr};
  QwenImageTransformer2DModel transformer_{nullptr};
  std::unique_ptr<Tokenizer> qwen_tokenizer_;
  std::unique_ptr<Tokenizer> tokenizer_;
  xllm::FlowMatchEulerDiscreteScheduler scheduler_{nullptr};
};
}  // namespace qwenimage
}  // namespace xllm::dit::npu

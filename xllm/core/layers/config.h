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

#include "core/framework/model_context.h"

#define UNIFY_CLASS_NAME(origin_name, target_name) \
  namespace xllm {                                 \
  namespace layer {                                \
  using target_name = origin_name;                 \
  }                                                \
  }

#define REGISTER_NOT_IMPLEMENTED_CLASS(CLS)                         \
  namespace xllm {                                                  \
  namespace layer {                                                 \
  class CLS {                                                       \
   public:                                                          \
    template <typename... Args>                                     \
    CLS(const ModelContext& context, Args&&... args) {              \
      (void)context;                                                \
      (void)sizeof...(args);                                        \
      LOG(FATAL) << "Class is not implemented in current backend."; \
    }                                                               \
  };                                                                \
  }                                                                 \
  }

#include "common/linear.h"
#include "common/qwen2_5_vision_layer.h"
#include "common/qwen2_decoder_layer.h"
#include "common/qwen3_moe_decoder_layer.h"
#include "common/rotary_embedding.h"
#include "common/word_embedding_impl.h"

#if defined(USE_MLU)
#include "mlu/deepseek_v2_decoder_layer_impl.h"
#else
REGISTER_NOT_IMPLEMENTED_CLASS(DeepseekV2DecoderLayerImpl);
#endif

UNIFY_CLASS_NAME(ColumnParallelLinearImpl, LmHeadImpl)
UNIFY_CLASS_NAME(Qwen2_VisionLayerImpl, Qwen2VisionEncoderLayerImpl)
UNIFY_CLASS_NAME(Qwen2_5_VisionLayerImpl, Qwen2dot5VisionEncoderLayerImpl)
UNIFY_CLASS_NAME(Qwen3_VisionLayerImpl, Qwen3VisionEncoderLayerImpl)

REGISTER_NOT_IMPLEMENTED_CLASS(DeepseekV32DecoderLayerImpl);
REGISTER_NOT_IMPLEMENTED_CLASS(LlamaDecoderLayerImpl);
REGISTER_NOT_IMPLEMENTED_CLASS(SiglipEncoderLayerImpl);
REGISTER_NOT_IMPLEMENTED_CLASS(Glm4DecoderLayerImpl);
REGISTER_NOT_IMPLEMENTED_CLASS(Glm4VisionEncoderLayerImpl);
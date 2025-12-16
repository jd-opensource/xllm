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

#if defined(USE_NPU)
#include "npu/npu_word_embedding_impl.h"
#else
#include "common/word_embedding_impl.h"
#endif

#if defined(USE_NPU)
#include "npu/npu_pos_embedding_impl.h"
#else
#include "common/rotary_embedding.h"
#endif

#if defined(USE_NPU)
#include "npu/npu_lm_head_impl.h"
#else
#include "common/linear.h"
UNIFY_CLASS_NAME(ColumnParallelLinearImpl, LmHeadImpl)
#endif

#if defined(USE_NPU)
#include "npu/npu_deepseek_v2_decoder_layer_impl.h"
#elif defined(USE_MLU)
#include "mlu/deepseek_v2_decoder_layer_impl.h"
#else
REGISTER_NOT_IMPLEMENTED_CLASS(DeepseekV2DecoderLayerImpl);
#endif

#if defined(USE_NPU)
#include "npu/npu_deepseek_v32_decoder_layer_impl.h"
#else
REGISTER_NOT_IMPLEMENTED_CLASS(DeepseekV32DecoderLayerImpl);
#endif

#if defined(USE_NPU)
#include "npu/npu_llama_decoder_layer_impl.h"
#else
REGISTER_NOT_IMPLEMENTED_CLASS(LlamaDecoderLayerImpl);
#endif

#if defined(USE_NPU)
#include "npu/npu_qwen2_decoder_layer_impl.h"
#else
#include "common/qwen2_decoder_layer.h"
#endif

#if defined(USE_NPU)
#include "npu/npu_qwen2_vision_encoder_layer_impl.h"
#else
#include "common/qwen2_5_vision_layer.h"
UNIFY_CLASS_NAME(Qwen2_VisionLayerImpl, Qwen2VisionEncoderLayerImpl)
#endif

#if defined(USE_NPU)
#include "npu/npu_qwen2dot5_vision_encoder_layer_impl.h"
#else
#include "common/qwen2_5_vision_layer.h"
UNIFY_CLASS_NAME(Qwen2_5_VisionLayerImpl, Qwen2dot5VisionEncoderLayerImpl)
#endif

#if defined(USE_NPU)
#include "npu/npu_qwen3_decoder_layer_impl.h"
#else
#include "common/qwen2_decoder_layer.h"
#endif

#if defined(USE_NPU)
#include "npu/npu_qwen3_moe_decoder_layer_impl.h"
#else
#include "common/qwen3_moe_decoder_layer.h"
#endif

#if defined(USE_NPU)
#include "npu/npu_qwen3_vision_encoder_layer_impl.h"
#else
#include "common/qwen2_5_vision_layer.h"
UNIFY_CLASS_NAME(Qwen3_VisionLayerImpl, Qwen3VisionEncoderLayerImpl)
#endif

#if defined(USE_NPU)
#include "npu/npu_siglip_encoder_layer_impl.h"
#else
REGISTER_NOT_IMPLEMENTED_CLASS(SiglipEncoderLayerImpl);
#endif

#if defined(USE_NPU)
#include "npu/npu_glm4_decoder_layer_impl.h"
#else
REGISTER_NOT_IMPLEMENTED_CLASS(Glm4DecoderLayerImpl);
#endif

#if defined(USE_NPU)
#include "npu/npu_glm4_vision_encoder_layer_impl.h"
namespace xllm {
namespace layer {
using Glm4VisionEncoderLayerImpl = NpuGlm4VisionEncoderLayerImpl;
}
}  // namespace xllm
#else
REGISTER_NOT_IMPLEMENTED_CLASS(Glm4VisionEncoderLayerImpl);
#endif

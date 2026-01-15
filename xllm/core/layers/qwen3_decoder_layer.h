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

#ifdef USE_MUSA
#include "musa/musa_qwen3_decoder_layer_impl.h"
#else
#include "qwen2_decoder_layer.h"
#endif

namespace xllm {
namespace layer {
#ifdef USE_MUSA
using Qwen3DecoderLayerImpl = MUSAQwen3DecoderImpl;
#else
using Qwen3DecoderLayerImpl = Qwen2DecoderLayerImpl;
#endif
TORCH_MODULE(Qwen3DecoderLayer);

}  // namespace layer
}  // namespace xllm
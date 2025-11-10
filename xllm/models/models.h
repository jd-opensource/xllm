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

#include "llm/llm_model_base.h"  // IWYU pragma: keep
#include "llm/qwen2.h"           // IWYU pragma: keep
#include "llm/qwen3.h"           // IWYU pragma: keep
#include "llm/qwen3_moe.h"       // IWYU pragma: keep

#if defined(USE_NPU)
#include "dit/pipeline_flux.h"          // IWYU pragma: keep
#include "dit/pipeline_flux_control.h"  // IWYU pragma: keep
#include "dit/pipeline_flux_fill.h"     // IWYU pragma: keep
#include "llm/deepseek_v2.h"            // IWYU pragma: keep
#include "llm/deepseek_v2_mtp.h"        // IWYU pragma: keep
#include "llm/deepseek_v3.h"            // IWYU pragma: keep
#include "llm/glm4_moe.h"               // IWYU pragma: keep
#include "llm/glm4_moe_mtp.h"           // IWYU pragma: keep
#include "llm/kimi_k2.h"                // IWYU pragma: keep
#include "llm/llama.h"                  // IWYU pragma: keep
#include "llm/llama3.h"                 // IWYU pragma: keep
#include "llm/qwen3_embedding.h"        // IWYU pragma: keep
#include "vlm/minicpmv.h"               // IWYU pragma: keep
#include "vlm/qwen2_5_vl.h"             // IWYU pragma: keep
#include "vlm/qwen3_vl.h"               // IWYU pragma: keep
#include "vlm/qwen3_vl_moe.h"           // IWYU pragma: keep
#endif

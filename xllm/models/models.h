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
#include "deepseek_v2.h"         // IWYU pragma: keep
#include "deepseek_v2_mtp.h"     // IWYU pragma: keep
#include "deepseek_v3.h"         // IWYU pragma: keep
#include "flux/pipeline_flux.h"  // IWYU pragma: keep
#include "kimi_k2.h"             // IWYU pragma: keep
#include "llama.h"               // IWYU pragma: keep
#include "llama3.h"              // IWYU pragma: keep
#include "minicpmv.h"            // IWYU pragma: keep
#include "qwen2.h"               // IWYU pragma: keep
#include "qwen2_5_vl.h"          // IWYU pragma: keep
#include "qwen3.h"               // IWYU pragma: keep
#include "qwen3_embedding.h"     // IWYU pragma: keep
#include "qwen3_moe.h"           // IWYU pragma: keep
#include "qwen_base.h"           // IWYU pragma: keep
#endif

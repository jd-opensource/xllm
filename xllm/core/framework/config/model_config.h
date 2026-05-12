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

#include <cstdint>
#include <string>

#include "core/common/macros.h"

namespace xllm {

class ModelConfig final {
 public:
  ModelConfig() = default;
  ~ModelConfig() = default;

  static ModelConfig& get_instance();

  static ModelConfig from_flags();
  static void reload_from_flags();

  PROPERTY(std::string, model_id);

  PROPERTY(std::string, model);

  PROPERTY(std::string, backend);

  PROPERTY(std::string, task) = "generate";

  PROPERTY(std::string, devices) = "npu:0";

  PROPERTY(int32_t, limit_image_per_prompt) = 4;

  PROPERTY(std::string, reasoning_parser);

  PROPERTY(std::string, tool_call_parser);

  PROPERTY(bool, enable_qwen3_reranker) = false;

  PROPERTY(bool, enable_return_mm_full_embeddings) = false;

  PROPERTY(int32_t, flashinfer_workspace_buffer_size) = 128 * 1024 * 1024;

  PROPERTY(bool, use_audio_in_video) = false;

  PROPERTY(bool, use_cpp_chat_template) = true;
};

}  // namespace xllm

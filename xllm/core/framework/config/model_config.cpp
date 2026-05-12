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

#include "core/framework/config/model_config.h"

#include "core/common/global_flags.h"

DEFINE_string(model_id, "", "hf model name.");

DEFINE_string(model, "", "Name or path of the huggingface model to use.");

DEFINE_string(
    backend,
    "",
    "Choose the backend model type. 'llm' for text-only, "
    "'vlm' for multimodal (text and images), 'dit' for diffusion models.");

DEFINE_string(task,
              "generate",
              "The task to use the model for(e.g. generate, embed, mm_embed).");

DEFINE_string(devices,
              "npu:0",
              "Devices to run the model on, e.g. npu:0, npu:0,npu:1.");

DEFINE_int32(limit_image_per_prompt,
             4,
             "Maximum number of image per prompt. Only applicable for "
             "multimodal models.");

DEFINE_string(reasoning_parser,
              "",
              "Specify the reasoning parser for handling reasoning "
              "interactions(e.g. auto, glm45, glm47, glm5, qwen3, qwen35, "
              "deepseek-r1).");

DEFINE_string(tool_call_parser,
              "",
              "Specify the parser for handling tool-call interactions(e.g. "
              "auto, qwen25, qwen3, qwen35, qwen3_coder, kimi_k2, "
              "deepseekv3, glm45, glm47, glm5).");

DEFINE_bool(enable_qwen3_reranker, false, "Whether to enable qwen3 reranker.");

DEFINE_int32(flashinfer_workspace_buffer_size,
             128 * 1024 * 1024,
             "The user reserved workspace buffer used to store intermediate "
             "attention results in split-k algorithm for flashinfer.");

DEFINE_bool(enable_return_mm_full_embeddings,
            false,
            "return vit and sequence embeddings for vlm models");

DEFINE_bool(
    use_audio_in_video,
    false,
    "Whether to decode both audio and video when the input is a video.");

// NOTE: This is an experimental flag,
//       it needs to be removed after the function is stable.
DEFINE_bool(use_cpp_chat_template,
            true,
            "Use native C++ chat template for supported models "
            "(e.g. deepseek_v32) instead of Jinja. "
            "Set to false to fallback to Jinja for debugging.");

namespace xllm {

ModelConfig ModelConfig::from_flags() {
  ModelConfig config;
  config.model_id(FLAGS_model_id)
      .model(FLAGS_model)
      .backend(FLAGS_backend)
      .task(FLAGS_task)
      .devices(FLAGS_devices)
      .limit_image_per_prompt(FLAGS_limit_image_per_prompt)
      .reasoning_parser(FLAGS_reasoning_parser)
      .tool_call_parser(FLAGS_tool_call_parser)
      .enable_qwen3_reranker(FLAGS_enable_qwen3_reranker)
      .enable_return_mm_full_embeddings(FLAGS_enable_return_mm_full_embeddings)
      .flashinfer_workspace_buffer_size(FLAGS_flashinfer_workspace_buffer_size)
      .use_audio_in_video(FLAGS_use_audio_in_video)
      .use_cpp_chat_template(FLAGS_use_cpp_chat_template);
  return config;
}

ModelConfig& ModelConfig::get_instance() {
  static ModelConfig config = ModelConfig::from_flags();
  return config;
}

void ModelConfig::reload_from_flags() {
  ModelConfig::get_instance() = ModelConfig::from_flags();
}

}  // namespace xllm

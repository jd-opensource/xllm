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

#include <glog/logging.h>

#include "core/common/global_flags.h"
#include "core/framework/config/config_json_utils.h"

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
namespace {

bool is_cpp_chat_template_supported_model(const std::string& model_type) {
  return model_type == "deepseek_v32" || model_type == "deepseek_v4";
}

}  // namespace

void ModelConfig::from_flags() {
  model_id(FLAGS_model_id)
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
}

void ModelConfig::normalize_cpp_chat_template(const std::string& model_type) {
  if (!use_cpp_chat_template()) {
    return;
  }

  if (is_cpp_chat_template_supported_model(model_type)) {
    return;
  }

  use_cpp_chat_template(false);
  LOG(WARNING) << "use_cpp_chat_template is not supported for model_type="
               << model_type << ", forcing use_cpp_chat_template=false.";
}

void ModelConfig::from_json(const JsonReader& json) {
  model_id(json.value_or<std::string>("model_id", model_id()))
      .model(json.value_or<std::string>("model", model()))
      .backend(json.value_or<std::string>("backend", backend()))
      .task(json.value_or<std::string>("task", task()))
      .devices(json.value_or<std::string>("devices", devices()))
      .limit_image_per_prompt(json.value_or<int32_t>("limit_image_per_prompt",
                                                     limit_image_per_prompt()))
      .reasoning_parser(
          json.value_or<std::string>("reasoning_parser", reasoning_parser()))
      .tool_call_parser(
          json.value_or<std::string>("tool_call_parser", tool_call_parser()))
      .enable_qwen3_reranker(
          json.value_or<bool>("enable_qwen3_reranker", enable_qwen3_reranker()))
      .enable_return_mm_full_embeddings(
          json.value_or<bool>("enable_return_mm_full_embeddings",
                              enable_return_mm_full_embeddings()))
      .flashinfer_workspace_buffer_size(
          json.value_or<int32_t>("flashinfer_workspace_buffer_size",
                                 flashinfer_workspace_buffer_size()))
      .use_audio_in_video(
          json.value_or<bool>("use_audio_in_video", use_audio_in_video()))
      .use_cpp_chat_template(json.value_or<bool>("use_cpp_chat_template",
                                                 use_cpp_chat_template()));
}

ModelConfig& ModelConfig::get_instance() {
  static ModelConfig config;
  return config;
}

void ModelConfig::initialize() {
  from_flags();
  if (!FLAGS_config_json_file.empty()) {
    from_json(config::load_json_file(FLAGS_config_json_file));
  }
}

}  // namespace xllm

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

#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <vector>

#include "core/common/message.h"
#include "core/common/types.h"
#include "framework/tokenizer/tokenizer_args.h"

namespace xllm {

class ChatTemplate {
 public:
  virtual ~ChatTemplate() = default;

  virtual std::optional<std::string> apply(
      const ChatMessages& messages) const = 0;

  virtual std::optional<std::string> apply(
      const ChatMessages& messages,
      const std::vector<xllm::JsonTool>& json_tools,
      const nlohmann::ordered_json& chat_template_kwargs) const = 0;

  static std::unique_ptr<ChatTemplate> create(
      const TokenizerArgs& tokenizer_args,
      const std::string& model_type);
};

}  // namespace xllm

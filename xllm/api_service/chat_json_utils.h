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

#include <string>
#include <utility>

#include "core/common/types.h"

namespace xllm {

// Preprocess chat JSON to normalize array content to string.
// For text-only backends (is_multimodal=false), combines text array items into
// a single string. Returns an error if non-text content is encountered.
// For multimodal backends (is_multimodal=true), leaves non-text content
// unchanged for downstream processing.
// Returns Status with processed JSON on success, or error status on failure.
std::pair<Status, std::string> preprocess_chat_json(std::string json_str,
                                                    bool is_multimodal);

}  // namespace xllm

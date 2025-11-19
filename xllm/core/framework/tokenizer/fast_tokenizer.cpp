/* Copyright 2025 The xLLM Authors. All Rights Reserved.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include "fast_tokenizer.h"

#include <glog/logging.h>

namespace xllm {

FastTokenizer::FastTokenizer(const std::string& tokenizer_json_path)
    : tokenizer_json_path_(tokenizer_json_path) {
  handle_ = tokenizers_new_from_path(tokenizer_json_path.c_str());
  CHECK(handle_ != nullptr)
      << "Failed to load tokenizer from file: " << tokenizer_json_path;
}

std::unique_ptr<Tokenizer> FastTokenizer::clone() const {
  return std::make_unique<FastTokenizer>(tokenizer_json_path_);
}

FastTokenizer::~FastTokenizer() { tokenizers_free(handle_); }

bool FastTokenizer::encode(const std::string_view& text,
                           std::vector<int32_t>* ids,
                           bool add_special_tokens) const {
  TokenizerEncodeResult result;
  tokenizers_encode(
      handle_, text.data(), text.size(), add_special_tokens, &result);

  std::vector<int32_t> ret(result.token_ids, result.token_ids + result.len);
  *ids = std::move(ret);

  return true;
}

std::string FastTokenizer::decode(const Slice<int32_t>& ids,
                                  bool skip_special_tokens) const {
  const char* data = nullptr;
  size_t len = 0;
  tokenizers_decode(handle_,
                    reinterpret_cast<const uint32_t*>(ids.data()),
                    ids.size(),
                    skip_special_tokens,
                    &data,
                    &len);
  return {data, len};
}

std::optional<int32_t> FastTokenizer::token_to_id(
    const std::string_view& token) const {
  int32_t id = -1;
  tokenizers_token_to_id(handle_, token.data(), token.size(), &id);
  return id == -1 ? std::optional<int32_t>(std::nullopt)
                  : std::optional<int32_t>(id);
}

std::string FastTokenizer::id_to_token(int32_t id) const {
  const char* data = nullptr;
  size_t len = 0;
  tokenizers_id_to_token(handle_, id, &data, &len);
  return {data, len};
}

size_t FastTokenizer::vocab_size() const {
  size_t size;
  tokenizers_get_vocab_size(handle_, &size);
  CHECK(size > 0) << "vocab_size must be greater than 0.";
  return size;
}

}  // namespace xllm

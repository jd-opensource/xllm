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

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "tokenizer.h"
#include "tokenizer_args.h"
#include "util/slice.h"

namespace xllm {

class RecTokenizer : public Tokenizer {
 public:
  RecTokenizer(const std::string_view& dir_path, const TokenizerArgs& args);

  virtual ~RecTokenizer() = default;

  bool encode(int64_t item_id, std::vector<int32_t>* token_ids) const override;

  bool decode(const Slice<int32_t>& token_ids,
              bool skip_special_tokens,
              std::vector<int64_t>* item_ids) const override;

  size_t vocab_size() const override;

  std::unique_ptr<Tokenizer> clone() const override;

 private:
  TokenizerArgs args_;

  std::string dir_path_;

  std::string model_version_;
};

}  // namespace xllm

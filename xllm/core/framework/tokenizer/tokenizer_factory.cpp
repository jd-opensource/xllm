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

#include "tokenizer_factory.h"

#include <filesystem>

namespace xllm {

std::unique_ptr<Tokenizer> TokenizerFactory::create_tokenizer(
    const std::string& model_weights_path,
    TokenizerArgs tokenizer_args,
    bool proxy) {
  const std::string tokenizer_json_path =
      model_weights_path + "/tokenizer.json";
  std::unique_ptr<Tokenizer> tokenizer;
  if (std::filesystem::exists(tokenizer_json_path)) {
    // 1. fast tokenizer
    LOG(INFO) << "Create fast tokenizer.";
    tokenizer = std::make_unique<FastTokenizer>(tokenizer_json_path);
  } else if (tokenizer_args.tokenizer_type() == "tiktoken" ||
             tokenizer_args.tokenizer_class() == "TikTokenTokenizer") {
    // 2. create tiktoken tokenizer
    LOG(INFO) << "Create Tiktoken tokenizer.";
    tokenizer =
        std::make_unique<TiktokenTokenizer>(model_weights_path, tokenizer_args);
  } else {
    // 3. create sentencepiece tokenizer
    LOG(INFO) << "Create SentencePiece tokenizer.";
    tokenizer = std::make_unique<SentencePieceTokenizer>(model_weights_path,
                                                         tokenizer_args);
  }
  if (proxy) {
    return std::make_unique<TokenizerProxy>(std::move(tokenizer));
  }
  return tokenizer;
}

}  // namespace xllm

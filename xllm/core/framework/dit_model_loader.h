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

#include <torch/torch.h>

#include <vector>

#include "core/framework/model/model_args.h"
#include "core/framework/quant_args.h"
#include "core/framework/state_dict/state_dict.h"
#include "core/framework/tokenizer/tokenizer.h"
#include "core/framework/tokenizer/tokenizer_args.h"
#include "model_loader.h"
namespace xllm {
class DiTFolderLoader {
 public:
  DiTFolderLoader(const std::string& folder_path,
                  const std::string& component_name,
                  const std::string& model_type);

  std::unique_ptr<Tokenizer> tokenizer() const;
  const ModelArgs& model_args() const { return args_; }
  const QuantArgs& quant_args() const { return quant_args_; }
  const TokenizerArgs& tokenizer_args() const { return tokenizer_args_; }
  std::vector<std::unique_ptr<StateDict>>& get_state_dicts();
  std::string model_weights_path() const { return model_weights_path_; }

 private:
  bool load_args(const std::string& model_weights_path);
  bool load_model_args(const std::string& model_weights_path);
  bool load_tokenizer_args(const std::string& model_weights_path);
  bool load_image_preprocessor_args(const std::string& model_weights_path);

  // model args
  ModelArgs args_;
  // quantization args
  QuantArgs quant_args_;
  // tokenizer args
  TokenizerArgs tokenizer_args_;

  std::string model_weights_path_;

  std::string model_type_;

  std::string component_name_;

  // sorted model weights files
  std::vector<std::string> model_weights_files_;
  // models weights tensors
  std::vector<std::unique_ptr<StateDict>> state_dicts_;
};

class DiTModelLoader {
 public:
  explicit DiTModelLoader(const std::string& model_root_path);

  std::unique_ptr<DiTFolderLoader> take_component_loader(
      const std::string& component);

  bool has_component(const std::string& component) const;
  std::string model_root_path() const { return model_root_path_; }

  std::unordered_map<std::string, ModelArgs> get_model_args() const;
  std::unordered_map<std::string, QuantArgs> get_quant_args() const;

  std::string get_torch_dtype() const;

 private:
  std::string model_root_path_;

  std::unordered_map<std::string, std::unique_ptr<DiTFolderLoader>>
      name_to_loader_;
};
}  // namespace xllm

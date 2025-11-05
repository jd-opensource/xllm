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

#include <torch/torch.h>

#include <vector>

#include "core/framework/state_dict/state_dict.h"
#include "model_loader.h"

namespace xllm {

class HFModelLoader : public ModelLoader {
 public:
  HFModelLoader(const std::string& model_weights_path);

  std::unique_ptr<Tokenizer> tokenizer() const override;

  std::vector<std::unique_ptr<StateDict>>& get_state_dicts() override;

 private:
  bool load_args(const std::string& model_weights_path);
  bool load_rec_vocab(const std::string& model_weights_path);
  bool load_model_args(const std::string& model_weights_path);
  bool load_quant_args(const std::string& model_weights_path);
  bool load_tokenizer_args(const std::string& model_weights_path);
  bool load_image_preprocessor_args(const std::string& model_weights_path);
  std::string model_weights_path() const override {
    return model_weights_path_;
  }
  std::string model_weights_path_;

  // sorted model weights files
  std::vector<std::string> model_weights_files_;

  // models weights tensors
  std::vector<std::unique_ptr<StateDict>> state_dicts_;
};
}  // namespace xllm

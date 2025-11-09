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

#include "hf_model_loader.h"

#include <absl/strings/match.h>
#include <absl/strings/str_replace.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <boost/algorithm/string.hpp>
#include <filesystem>
#include <vector>

#include "core/common/version_singleton.h"
#include "core/framework/state_dict/rec_vocab_dict.h"
#include "core/framework/tokenizer/fast_tokenizer.h"
#include "core/framework/tokenizer/rec_tokenizer.h"
#include "core/framework/tokenizer/sentencepiece_tokenizer.h"
#include "core/framework/tokenizer/tiktoken_tokenizer.h"
#include "core/framework/tokenizer/tokenizer_factory.h"
#include "core/util/json_reader.h"
#include "models/model_registry.h"

namespace xllm {

HFModelLoader::HFModelLoader(const std::string& model_weights_path)
    : model_weights_path_(model_weights_path) {
  CHECK(load_args(model_weights_path))
      << "Failed to load model args from " << model_weights_path;
  // try to load safetensors first
  for (const auto& entry :
       std::filesystem::directory_iterator(model_weights_path)) {
    // load bin or safe tensors
    if (entry.path().extension() == ".safetensors") {
      model_weights_files_.push_back(entry.path().string());
    }
  }
  CHECK(!model_weights_files_.empty())
      << "Failed to find model weights files in " << model_weights_path;
  // sort the model weights files by name
  std::sort(model_weights_files_.begin(), model_weights_files_.end());

  //@todo: 'false' will be replaced with generative recommendation judgment
  if (false) {
    CHECK(load_rec_vocab(model_weights_path))
        << "Failed to load rec content from " << model_weights_path;
  }
}

std::unique_ptr<Tokenizer> HFModelLoader::tokenizer() const {
  return TokenizerFactory::create_tokenizer(model_weights_path_,
                                            tokenizer_args_);
}

std::vector<std::unique_ptr<StateDict>>& HFModelLoader::get_state_dicts() {
  if (state_dicts_.empty()) {
    // load state dict
    state_dicts_.reserve(model_weights_files_.size());
    for (auto& model_weights_file : model_weights_files_) {
      LOG(INFO) << "Loading model weights from " << model_weights_file;
      state_dicts_.emplace_back(
          StateDictFromSafeTensor::load(model_weights_file));
    }
  }
  return state_dicts_;
}

bool HFModelLoader::load_rec_vocab(const std::string& model_weights_path) {
  if (!tokenizer_args_.vocab_file().empty()) {
    std::filesystem::path path = model_weights_path;
    std::string model_version = path.filename();
    std::string vocab_full_path =
        path.append(tokenizer_args_.vocab_file()).string();

    LOG(INFO) << "model_version:" << model_version
              << ", vocab_full_path:" << vocab_full_path;

    CHECK(nullptr != VersionSingleton<RecVocabDict>::GetInstance(model_version))
        << "Failed to get vocab dict instance";
    CHECK(VersionSingleton<RecVocabDict>::GetInstance(model_version)
              ->initialize(vocab_full_path))
        << "Failed to initialize vocab dict from " << vocab_full_path;
  } else {
    LOG(ERROR) << "vocab file is not set";
  }

  return true;
}

bool HFModelLoader::load_args(const std::string& model_weights_path) {
  if (!load_model_args(model_weights_path)) {
    LOG(ERROR) << "Failed to load model args from " << model_weights_path;
    return false;
  }

  if (!load_quant_args(model_weights_path)) {
    LOG(ERROR) << "Failed to load quant args from " << model_weights_path;
    return false;
  }

  if (!load_tokenizer_args(model_weights_path)) {
    LOG(ERROR) << "Failed to load tokenizer args from " << model_weights_path;
    return false;
  }

  if (!load_image_preprocessor_args(model_weights_path)) {
    LOG(ERROR) << "Failed to load image preprocess args from "
               << model_weights_path;
    return false;
  }

  // Some hacky logics to support loading of old models
  // always use float16 for quantization
  // TODO: support quantization for other data types
  if (!quant_args_.quant_method().empty() && args_.dtype() != "bfloat16") {
    LOG(WARNING) << "Overwriting dtype from " << args_.dtype()
                 << " to float16 for quantization";
    args_.dtype() = "bfloat16";
  }

  return true;
}

bool HFModelLoader::load_model_args(const std::string& model_weights_path) {
  JsonReader reader;
  const std::string args_file_path = model_weights_path + "/config.json";
  if (!reader.parse(args_file_path)) {
    LOG(ERROR) << "Failed to parse model args file: " << args_file_path;
    return false;
  }

  std::string model_type;
  if (auto data = reader.value<std::string>("model_type")) {
    model_type = data.value();
  } else {
    LOG(ERROR) << "Failed to find model_type in " << args_file_path;
    return false;
  }

  auto model_args_loader = ModelRegistry::get_model_args_loader(model_type);
  if (model_args_loader == nullptr) {
    LOG(ERROR) << "Failed to find model args loader for model type "
               << model_type;
    return false;
  }
  model_args_loader(reader, &args_);

  return true;
}

bool HFModelLoader::load_quant_args(const std::string& model_weights_path) {
  JsonReader reader;
  const std::string args_file_path = model_weights_path + "/config.json";
  if (!reader.parse(args_file_path)) {
    LOG(ERROR) << "Failed to parse model args file: " << args_file_path;
    return false;
  }

  if (reader.contains("quantization_config")) {
    if (auto v =
            reader.value<std::string>("quantization_config.quant_method")) {
      quant_args_.quant_method() = v.value();
    }
    if (auto v = reader.value<int64_t>("quantization_config.bits")) {
      quant_args_.bits() = v.value();
    }
    if (auto v = reader.value<int64_t>("quantization_config.group_size")) {
      quant_args_.group_size() = v.value();
    }
    if (auto v = reader.value<bool>("quantization_config.desc_act")) {
      quant_args_.desc_act() = v.value();
    }
    if (auto v = reader.value<bool>("quantization_config.sym")) {
      quant_args_.is_sym() = v.value();
    }
    if (auto v = reader.value<std::string>(
            "quantization_config.activation_scheme")) {
      std::string activation_scheme = v.value();
      if (boost::iequals(activation_scheme, "static")) {
        quant_args_.activation_dynamic() = false;
      } else if (boost::iequals(activation_scheme, "dynamic")) {
        quant_args_.activation_dynamic() = true;
      } else {
        LOG(ERROR) << "quantization_config.activation_scheme only support "
                      "dynamic and static.";
        return false;
      }
    }
    if (auto v = reader.value<std::string>("quantization_config.fmt")) {
      // TODO(liangzhiwei20): check fp8 quantization format
      quant_args_.fmt() = v.value();
    }
    if (reader.contains("quantization_config.weight_block_size")) {
      const auto& data = reader.data();
      quant_args_.weight_block_size() =
          data["quantization_config"]["weight_block_size"]
              .get<std::vector<int64_t>>();
    }
  }

  // load quantization args for npu if exists
  if (reader.contains("quantize")) {
    quant_args_.quantize_type() = reader.value_or<std::string>("quantize", "");
  }
  if (reader.contains("torch_dtype")) {
    quant_args_.torch_dtype() = reader.value_or<std::string>("torch_dtype", "");
  }

  // awq quantization args
  JsonReader awq_reader;
  const std::string quant_args_file_path =
      model_weights_path + "/quant_config.json";
  if (awq_reader.parse(quant_args_file_path)) {
    // hardcode the quant_method to awq if not exists
    quant_args_.quant_method() =
        awq_reader.value_or<std::string>("quant_method", "awq");

    if (auto v = awq_reader.value<int64_t>("w_bit")) {
      quant_args_.bits() = v.value();
    }
    if (auto v = awq_reader.value<int64_t>("q_group_size")) {
      quant_args_.group_size() = v.value();
    }
  }

  // gptq quantization args
  const std::string gptq_args_file_path =
      model_weights_path + "/quantize_config.json";
  JsonReader gptq_reader;
  if (gptq_reader.parse(gptq_args_file_path)) {
    // hardcode the quant_method to gptq if not exists
    quant_args_.quant_method() =
        gptq_reader.value_or<std::string>("quant_method", "gptq");

    if (auto v = gptq_reader.value<int64_t>("bits")) {
      quant_args_.bits() = v.value();
    }
    if (auto v = gptq_reader.value<int64_t>("group_size")) {
      quant_args_.group_size() = v.value();
    }
    if (auto v = gptq_reader.value<bool>("desc_act")) {
      quant_args_.desc_act() = v.value();
    }
    if (auto v = gptq_reader.value<bool>("sym")) {
      quant_args_.is_sym() = v.value();
    }
  }

  return true;
}

namespace {
std::optional<std::string> load_chat_template_file(const std::string& dir) {
  // chat_template.json
  const std::string chat_template_path = dir + "/chat_template.json";
  JsonReader reader;
  if (reader.parse(chat_template_path);
      auto v = reader.value<std::string>("chat_template")) {
    return v;
  }
  // chat_template.jinja
  const std::string raw_chat_template_path = dir + "/chat_template.jinja";
  std::ifstream file(raw_chat_template_path);
  if (file.is_open()) {
    std::ostringstream content;
    content << file.rdbuf();
    file.close();
    return content.str();
  }
  return std::nullopt;
}
}  // namespace

bool HFModelLoader::load_tokenizer_args(const std::string& model_weights_path) {
  // tokenizer args from tokenizer_config.json
  JsonReader tokenizer_reader;
  const std::string tokenizer_args_file_path =
      model_weights_path_ + "/tokenizer_config.json";
  if (tokenizer_reader.parse(tokenizer_args_file_path)) {
    // read chat template if exists
    if (auto v = load_chat_template_file(model_weights_path_)) {
      tokenizer_args_.chat_template() = v.value();
    } else if (auto v = tokenizer_reader.value<std::string>("chat_template")) {
      tokenizer_args_.chat_template() = v.value();
    }
    if (auto v = tokenizer_reader.value<bool>("add_bos_token")) {
      tokenizer_args_.add_bos_token() = v.value();
    }
    if (auto v = tokenizer_reader.value<bool>("add_eos_token")) {
      tokenizer_args_.add_eos_token() = v.value();
    }
    if (auto v = tokenizer_reader.value<std::string>("tokenizer_class")) {
      tokenizer_args_.tokenizer_class() = v.value();
    }
    // read bos_token
    if (auto v = tokenizer_reader.value<std::string>("bos_token.content")) {
      tokenizer_args_.bos_token() = v.value();
    } else if (auto v = tokenizer_reader.value<std::string>("bos_token")) {
      tokenizer_args_.bos_token() = v.value();
    }
    // read eos_token
    if (auto v = tokenizer_reader.value<std::string>("eos_token.content")) {
      tokenizer_args_.eos_token() = v.value();
    } else if (auto v = tokenizer_reader.value<std::string>("eos_token")) {
      tokenizer_args_.eos_token() = v.value();
    }
    // read pad_token
    if (auto v = tokenizer_reader.value<std::string>("pad_token.content")) {
      tokenizer_args_.pad_token() = v.value();
    } else if (auto v = tokenizer_reader.value<std::string>("pad_token")) {
      tokenizer_args_.pad_token() = v.value();
    }
  }

  auto tokenizer_args_loader =
      ModelRegistry::get_tokenizer_args_loader(args_.model_type());
  if (tokenizer_args_loader != nullptr) {
    if (!tokenizer_args_loader(tokenizer_reader, &tokenizer_args_)) {
      LOG(ERROR) << "Failed to load tokenizer args from "
                 << tokenizer_args_file_path;
      return false;
    }
  }

  return true;
}

bool HFModelLoader::load_image_preprocessor_args(
    const std::string& model_weights_path) {
  // image preprocessor args
  JsonReader image_preprocess_reader;
  const std::string image_preprocess_file_path =
      model_weights_path + "/preprocessor_config.json";
  if (image_preprocess_reader.parse(image_preprocess_file_path)) {
    LOG(INFO) << "Success to parse image preprocess args file: "
              << image_preprocess_file_path;
    args_.mm_image_do_center_crop() =
        image_preprocess_reader.value_or<bool>("do_center_crop", false);
    args_.mm_image_crop_height_size() =
        image_preprocess_reader.value_or<int>("crop_size.height", 335);
    args_.mm_image_crop_width_size() =
        image_preprocess_reader.value_or<int>("crop_size.width", 335);

    args_.mm_image_do_resize() =
        image_preprocess_reader.value_or<bool>("do_resize", false);
    args_.mm_image_resize_shortest_edge() =
        image_preprocess_reader.value_or<int>("size.shortest_edge", 335);
    args_.mm_image_resample() =
        image_preprocess_reader.value_or<int>("resample", 335);

    args_.mm_image_do_rescale() =
        image_preprocess_reader.value_or<bool>("do_rescale", false);
    args_.mm_image_rescale_factor() =
        image_preprocess_reader.value_or<double>("rescale_factor", 0);

    args_.mm_image_do_normalize() =
        image_preprocess_reader.value_or<bool>("do_normalize", false);

    const auto& image_prerocess_data = image_preprocess_reader.data();
    if (image_preprocess_reader.contains("image_mean")) {
      args_.mm_image_normalize_mean() =
          image_prerocess_data["image_mean"].get<std::vector<double>>();
    }

    if (image_preprocess_reader.contains("image_std")) {
      args_.mm_image_normalize_std() =
          image_prerocess_data["image_std"].get<std::vector<double>>();
    }

    if (image_preprocess_reader.contains("norm_mean")) {
      args_.mm_image_normalize_mean() =
          image_prerocess_data["norm_mean"].get<std::vector<double>>();
    }

    if (image_preprocess_reader.contains("norm_std")) {
      args_.mm_image_normalize_std() =
          image_prerocess_data["norm_std"].get<std::vector<double>>();
    }

    args_.mm_image_min_pixels() =
        image_preprocess_reader.value_or<int>("min_pixels", 0);

    args_.mm_image_max_pixels() =
        image_preprocess_reader.value_or<int>("max_pixels", 0);

    args_.mm_image_patch_size() =
        image_preprocess_reader.value_or<int>("patch_size", 0);

    args_.mm_image_temporal_patch_size() =
        image_preprocess_reader.value_or<int>("temporal_patch_size", 0);

    args_.mm_image_merge_size() =
        image_preprocess_reader.value_or<int>("merge_size", 0);

    args_.mm_image_feature_size() =
        image_preprocess_reader.value_or<int>("image_feature_size", 0);

    args_.mm_scale_resolution() =
        image_preprocess_reader.value_or<int>("scale_resolution", 0);

    args_.mm_slice_mode() =
        image_preprocess_reader.value_or<bool>("slice_mode", false);

    args_.mm_use_image_id() =
        image_preprocess_reader.value_or<bool>("use_image_id", false);
  }

  return true;
}

}  // namespace xllm

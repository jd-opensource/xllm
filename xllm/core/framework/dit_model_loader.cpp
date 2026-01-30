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

#include "dit_model_loader.h"

#include <absl/strings/match.h>
#include <absl/strings/str_replace.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <boost/algorithm/string.hpp>
#include <filesystem>
#include <vector>

#include "core/framework/tokenizer/tokenizer_factory.h"
#include "core/util/json_reader.h"
#include "models/model_registry.h"

namespace xllm {
DiTFolderLoader::DiTFolderLoader(const std::string& folder_path,
                                 const std::string& component_name,
                                 const std::string& model_type)
    : model_weights_path_(folder_path),
      component_name_(component_name),
      model_type_(model_type) {
  CHECK(load_args(folder_path))
      << "Failed to load model args from " << folder_path;
  // try to load safetensors first
  for (const auto& entry : std::filesystem::directory_iterator(folder_path)) {
    // load bin or safe tensors
    if (entry.path().extension() == ".safetensors") {
      model_weights_files_.push_back(entry.path().string());
    }
  }
  if (!model_weights_files_.empty()) {
    // sort the model weights files by name
    std::sort(model_weights_files_.begin(), model_weights_files_.end());
  }
}

std::unique_ptr<Tokenizer> DiTFolderLoader::tokenizer() const {
  return TokenizerFactory::create_tokenizer(model_weights_path_,
                                            tokenizer_args_,
                                            /*proxy*/ false);
}

std::vector<std::unique_ptr<StateDict>>& DiTFolderLoader::get_state_dicts() {
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

bool DiTFolderLoader::load_args(const std::string& model_weights_path) {
  if (!load_tokenizer_args(model_weights_path)) {
    LOG(ERROR) << "Failed to load tokenizer args from " << model_weights_path;
    return false;
  }

  if (!load_model_args(model_weights_path)) {
    LOG(ERROR) << "Failed to load model args from " << model_weights_path;
    return false;
  }

  return true;
}

bool DiTFolderLoader::load_model_args(const std::string& model_weights_path) {
  bool has_safetensors = false;
  std::filesystem::path model_dir(model_weights_path);

  if (std::filesystem::is_directory(model_dir)) {
    for (const auto& entry : std::filesystem::directory_iterator(model_dir)) {
      if (entry.path().extension() == ".safetensors") {
        has_safetensors = true;
        break;
      }
    }
  } else {
    LOG(ERROR) << "Model path is not a valid directory: " << model_weights_path;
    return false;
  }

  auto load_json_config = [&](const std::string& json_filename) -> bool {
    JsonReader reader;
    std::string json_path = model_weights_path + "/" + json_filename;

    if (!std::filesystem::exists(json_path)) {
      LOG(WARNING) << "JSON config file not found: " << json_path;
      return false;
    }

    if (!reader.parse(json_path)) {
      LOG(ERROR) << "Failed to parse JSON config: " << json_path;
      return false;
    }

    auto model_args_loader = ModelRegistry::get_model_args_loader(model_type_);
    if (model_args_loader != nullptr) {
      model_args_loader(reader, &args_);
    } else {
      LOG(WARNING) << "No args loader for model type: " << model_type_;
    }

    return true;
  };

  if (has_safetensors) {
    if (!load_json_config("config.json")) {
      LOG(ERROR) << "Failed to load required config.json for safetensors model";
      return false;
    }
  } else {
    std::filesystem::path tokenizer_config_path =
        model_dir / "tokenizer_config.json";
    if (std::filesystem::exists(tokenizer_config_path)) {
      return true;
    }
    std::vector<std::filesystem::path> json_file_paths;
    for (const auto& entry :
         std::filesystem::directory_iterator(model_weights_path)) {
      if (entry.is_regular_file() &&
          entry.path().extension().string() == ".json") {
        json_file_paths.push_back(entry.path());
      }
    }

    if (json_file_paths.empty()) {
      LOG(ERROR) << "No JSON config files found in " << model_weights_path;
      return false;
    }

    bool loaded_any = false;
    for (const auto& json_file : json_file_paths) {
      if (!load_json_config(json_file.filename().string())) {
        LOG(ERROR) << "Failed to parse JSON file: " << json_file;
        continue;
      }
      loaded_any = true;
    }

    if (!loaded_any) {
      LOG(ERROR) << "No valid JSON config files found in "
                 << model_weights_path;
      return false;
    }
  }

  return true;
}

bool DiTFolderLoader::load_tokenizer_args(
    const std::string& model_weights_path) {
  // tokenizer args from tokenizer_config.json
  JsonReader tokenizer_reader;
  const std::string tokenizer_args_file_path =
      model_weights_path_ + "/tokenizer_config.json";

  // check if tokenizer.json exists, if exists, set the tokenizer type to fast
  const std::string tokenizer_json_path =
      model_weights_path + "/tokenizer.json";
  if (std::filesystem::exists(tokenizer_json_path)) {
    tokenizer_args_.tokenizer_type() = "fast";
    tokenizer_args_.vocab_file() = tokenizer_json_path;
  }

  if (!std::filesystem::exists(tokenizer_args_file_path)) {
    return true;
  }
  if (tokenizer_reader.parse(tokenizer_args_file_path)) {
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

  return true;
}

DiTModelLoader::DiTModelLoader(const std::string& model_root_path)
    : model_root_path_(model_root_path) {
  if (!std::filesystem::exists(model_root_path_)) {
    LOG(FATAL) << "Model root path does not exist: " << model_root_path_;
  }

  std::filesystem::path root_path(model_root_path_);
  std::filesystem::path index_file = root_path / "model_index.json";
  const std::string model_index_file = index_file.string();
  if (!std::filesystem::exists(model_index_file)) {
    LOG(FATAL) << "Model index file does not exist: " << model_index_file;
  }

  JsonReader model_index_reader;
  if (!model_index_reader.parse(model_index_file)) {
    LOG(FATAL) << "Failed to parse model index file: " << model_index_file;
  }

  const nlohmann::json root_json = model_index_reader.data();
  if (!root_json.is_object()) {
    LOG(FATAL) << "DiTModelLoader: model_index.json root is not an object!";
  }

  // parse model_index.json & initialize model_loader
  for (const auto& [json_key, json_value] : root_json.items()) {
    if (!json_value.is_array() || json_value.size() != 2) {
      continue;
    }

    const std::string model_type = json_value[1].get<std::string>();
    const std::string component_name = json_key;

    std::filesystem::path component_folder_path =
        std::filesystem::path(model_root_path_) / component_name;
    const std::string component_folder = component_folder_path.string();
    if (!std::filesystem::exists(component_folder)) {
      LOG(FATAL) << "DiTModelLoader: Component folder not found! "
                 << "ComponentName=" << component_name
                 << ", Folder=" << component_folder;
      continue;
    }
    if (!std::filesystem::is_directory(component_folder)) {
      LOG(FATAL) << "DiTModelLoader: Component path is not a directory! "
                 << "ComponentName=" << component_name
                 << ", Path=" << component_folder;
      continue;
    }

    // create model loader for each Folder
    std::unique_ptr<DiTFolderLoader> loader = std::make_unique<DiTFolderLoader>(
        component_folder, component_name, model_type);
    if (!loader) {
      LOG(FATAL) << "Failed to create loader for: " << component_name;
      continue;
    }

    name_to_loader_[component_name] = std::move(loader);
  }
}

std::unique_ptr<DiTFolderLoader> DiTModelLoader::take_component_loader(
    const std::string& component) {
  auto itor = name_to_loader_.find(component);
  if (itor != name_to_loader_.end()) {
    std::unique_ptr<DiTFolderLoader> loader = std::move(itor->second);
    name_to_loader_.erase(itor);

    return loader;
  } else {
    LOG(FATAL) << "Loader not found, component: " << component;
    return nullptr;
  }
}

bool DiTModelLoader::has_component(const std::string& name) const {
  if (name_to_loader_.find(name) != name_to_loader_.end()) {
    return true;
  } else {
    return false;
  }
}

std::unordered_map<std::string, ModelArgs> DiTModelLoader::get_model_args()
    const {
  std::unordered_map<std::string, ModelArgs> map;
  for (const auto& pair : name_to_loader_) {
    map.insert({pair.first, pair.second->model_args()});
  }

  return map;
}

std::unordered_map<std::string, QuantArgs> DiTModelLoader::get_quant_args()
    const {
  std::unordered_map<std::string, QuantArgs> map;
  for (const auto& pair : name_to_loader_) {
    map.insert({pair.first, pair.second->quant_args()});
  }

  return map;
}

std::string DiTModelLoader::get_torch_dtype() const {
  std::string dtype;
  for (const auto& pair : name_to_loader_) {
    const auto& args = pair.second->model_args();

    const auto& type = args.dtype();
    if (dtype.empty() && !type.empty()) {
      dtype = type;
    } else if (!dtype.empty() && !type.empty() && dtype != type) {
      LOG(WARNING) << " dtype is not equal, dtype=" << dtype
                   << " type:" << type;
    }
  }

  return dtype;
}

}  // namespace xllm

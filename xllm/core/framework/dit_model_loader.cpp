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

#include "core/util/json_reader.h"
#include "models/model_registry.h"
namespace xllm {
DiTSubfolderLoader::DiTSubfolderLoader(const std::string& folder_path,
                                       const std::string& component_name)
    : model_weights_path_(folder_path), component_name_(component_name) {
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

std::vector<std::unique_ptr<StateDict>>& DiTSubfolderLoader::get_state_dicts() {
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

bool DiTSubfolderLoader::load_args(const std::string& model_weights_path) {
  std::filesystem::path tokenizer_config_path =
      std::filesystem::path(model_weights_path) / "tokenizer_config.json";

  if (std::filesystem::exists(tokenizer_config_path)) {
    if (!load_tokenizer_args(model_weights_path)) {
      LOG(ERROR) << "Failed to load tokenizer args from " << model_weights_path;
      return false;
    }
  } else {
    if (!load_model_args(model_weights_path)) {
      LOG(ERROR) << "Failed to load model args from " << model_weights_path;
      return false;
    }
  }
  return true;
}

void DiTSubfolderLoader::print_model_loader_info() const {
  LOG(INFO) << "Model Loader Info for component: " << component_name_;
  LOG(INFO) << "Model Weights Path: " << model_weights_path_;
  LOG(INFO) << "Model Args: " << args_;
  LOG(INFO) << "Quant Args: " << quant_args_;
  LOG(INFO) << "Tokenizer Args: " << tokenizer_args_;
}

bool DiTSubfolderLoader::load_model_args(
    const std::string& model_weights_path) {
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

    std::string model_type = component_name_;

    if (!model_type.empty()) {
      auto model_args_loader = ModelRegistry::get_model_args_loader(model_type);
      if (model_args_loader != nullptr) {
        model_args_loader(reader, &args_);
      } else {
        LOG(WARNING) << "No args loader for model type: " << model_type;
      }
    }

    return true;
  };

  if (has_safetensors) {
    if (!load_json_config("config.json")) {
      LOG(ERROR) << "Failed to load required config.json for safetensors model";
      return false;
    }
  } else {
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

bool DiTSubfolderLoader::load_tokenizer_args(
    const std::string& model_weights_path) {
  // tokenizer args from tokenizer_config.json
  JsonReader tokenizer_reader;
  const std::string tokenizer_args_file_path =
      model_weights_path_ + "/tokenizer_config.json";
  if (tokenizer_reader.parse(tokenizer_args_file_path)) {
    // read chat template if exists
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
    LOG(ERROR) << "Model root path does not exist: " << model_root_path_;
  }

  // check if model_index.json exists
  std::filesystem::path root_path(model_root_path_);
  std::filesystem::path index_file = root_path / "model_index.json";
  const std::string model_index_file = index_file.string();
  if (!std::filesystem::exists(model_index_file)) {
    LOG(ERROR) << "Model index file does not exist: " << model_index_file;
  }

  JsonReader model_index_reader;
  if (!model_index_reader.parse(model_index_file)) {
    LOG(ERROR) << "Failed to parse model index file: " << model_index_file;
  }

  LOG(INFO) << "Success to parse model index file: " << model_index_file;

  const nlohmann::json root_json = model_index_reader.data();
  if (!root_json.is_object()) {
    LOG(FATAL) << "DITModelLoader: model_index.json root is not an object!";
  }

  // parse model_index.json & initialize model_loader
  for (const auto& [json_key, json_value] : root_json.items()) {
    if (!json_value.is_array() || json_value.size() != 2) {
      LOG(WARNING) << "DITModelLoader: Invalid format for component! "
                   << "JsonKey=" << json_key
                   << ", Expected [library, class_name] array";
      continue;
    }

    const std::string component_name = json_value[1].get<std::string>();
    std::filesystem::path component_folder_path =
        std::filesystem::path(model_root_path_) / json_key;
    const std::string component_folder = component_folder_path.string();
    if (!std::filesystem::exists(component_folder)) {
      LOG(WARNING) << "DITModelLoader: Component folder not found! "
                   << "ComponentName=" << component_name
                   << ", Folder=" << component_folder;
      continue;
    }
    if (!std::filesystem::is_directory(component_folder)) {
      LOG(WARNING) << "DITModelLoader: Component path is not a directory! "
                   << "ComponentName=" << component_name
                   << ", Path=" << component_folder;
      continue;
    }

    // create model loader for each subfolder
    std::unique_ptr<DiTSubfolderLoader> sub_loader =
        std::make_unique<DiTSubfolderLoader>(component_folder, component_name);
    if (!sub_loader) {
      LOG(WARNING) << "Failed to create loader for: " << component_name;
      continue;
    }

    name_to_folder_[component_name] = component_folder;
    name_to_loader_[component_name] = std::move(sub_loader);
    sub_model_loaders_.push_back(name_to_loader_[component_name].get());
  }
}

const DiTSubfolderLoader* DiTModelLoader::get_sub_model_loader_by_name(
    const std::string& component_name) const {
  auto it = name_to_loader_.find(component_name);
  if (it == name_to_loader_.end()) {
    LOG(WARNING) << "Component not found: " << component_name;
    return nullptr;
  }
  return it->second.get();
}

const DiTSubfolderLoader* DiTModelLoader::get_sub_model_loader_by_folder(
    const std::string& component_folder) const {
  std::filesystem::path abs_folder =
      std::filesystem::absolute(component_folder);
  for (const auto& [name, folder] : name_to_folder_) {
    if (folder == abs_folder.string()) {
      return name_to_loader_.at(name).get();
    }
  }
  LOG(WARNING) << "Component folder not found: " << component_folder;
  return nullptr;
}

const std::vector<DiTSubfolderLoader*>&
DiTModelLoader::get_all_sub_model_loaders() const {
  return sub_model_loaders_;
}

std::vector<std::string> DiTModelLoader::get_all_sub_model_names() const {
  std::vector<std::string> names;
  for (const auto& [name, _] : name_to_loader_) {
    names.push_back(name);
  }
  return names;
}

bool DiTModelLoader::has_sub_model(const std::string& component_name) const {
  return name_to_loader_.count(component_name) > 0;
}
}  // namespace xllm
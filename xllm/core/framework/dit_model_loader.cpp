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

std::unique_ptr<Tokenizer> DiTFolderLoader::tokenizer() const {
  return TokenizerFactory::create_tokenizer(model_weights_path_,
                                            tokenizer_args_);
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
  debug_info();
  return true;
}

void DiTFolderLoader::debug_info() const {
  LOG(INFO) << "Model Loader Info for component: " << component_name_;
  LOG(INFO) << "Model Weights Path: " << model_weights_path_;
  LOG(INFO) << "Model Args: " << args_;
  LOG(INFO) << "Quant Args: " << quant_args_;
  LOG(INFO) << "Tokenizer Args: " << tokenizer_args_;
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
  if (!std::filesystem::exists(tokenizer_args_file_path)) {
    LOG(INFO) << "Tokenizer config file not found: "
              << tokenizer_args_file_path;
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
    LOG(FATAL) << "DiTModelLoader: model_index.json root is not an object!";
  }

  // parse model_index.json & initialize model_loader
  for (const auto& [json_key, json_value] : root_json.items()) {
    if (!json_value.is_array() || json_value.size() != 2) {
      LOG(WARNING) << "DiTModelLoader: Invalid format for component! "
                   << "JsonKey=" << json_key
                   << ", Expected [library, class_name] array";
      continue;
    }

    const std::string component_name = json_value[1].get<std::string>();
    std::filesystem::path component_folder_path =
        std::filesystem::path(model_root_path_) / json_key;
    const std::string component_folder = component_folder_path.string();
    if (!std::filesystem::exists(component_folder)) {
      LOG(WARNING) << "DiTModelLoader: Component folder not found! "
                   << "ComponentName=" << component_name
                   << ", Folder=" << component_folder;
      continue;
    }
    if (!std::filesystem::is_directory(component_folder)) {
      LOG(WARNING) << "DiTModelLoader: Component path is not a directory! "
                   << "ComponentName=" << component_name
                   << ", Path=" << component_folder;
      continue;
    }

    // create model loader for each Folder
    std::unique_ptr<DiTFolderLoader> sub_loader =
        std::make_unique<DiTFolderLoader>(component_folder, component_name);
    if (!sub_loader) {
      LOG(WARNING) << "Failed to create loader for: " << component_name;
      continue;
    }

    name_to_folder_[component_name] = component_folder;
    name_to_loader_[component_name] = std::move(sub_loader);
    component_names_.push_back(component_name);
    update_model_args(name_to_loader_[component_name]->model_args());
  }
}

void DiTModelLoader::update_model_args(const ModelArgs& sub_args) {
  // VAE
  if (!arg_status_["vae_in_channels"] && sub_args.vae_in_channels() != -1) {
    args_.vae_in_channels() = sub_args.vae_in_channels();
    arg_status_["vae_in_channels"] = true;
  }
  if (!arg_status_["vae_out_channels"] && sub_args.vae_out_channels() != -1) {
    args_.vae_out_channels() = sub_args.vae_out_channels();
    arg_status_["vae_out_channels"] = true;
  }
  if (!arg_status_["vae_down_block_types"] &&
      !sub_args.vae_down_block_types().empty()) {
    args_.vae_down_block_types() = sub_args.vae_down_block_types();
    arg_status_["vae_down_block_types"] = true;
  }
  if (!arg_status_["vae_up_block_types"] &&
      !sub_args.vae_up_block_types().empty()) {
    args_.vae_up_block_types() = sub_args.vae_up_block_types();
    arg_status_["vae_up_block_types"] = true;
  }
  if (!arg_status_["vae_block_out_channels"] &&
      !sub_args.vae_block_out_channels().empty()) {
    args_.vae_block_out_channels() = sub_args.vae_block_out_channels();
    arg_status_["vae_block_out_channels"] = true;
  }
  if (!arg_status_["vae_layers_per_block"] &&
      sub_args.vae_layers_per_block() != 1) {
    args_.vae_layers_per_block() = sub_args.vae_layers_per_block();
    arg_status_["vae_layers_per_block"] = true;
  }
  if (!arg_status_["vae_act_fn"] && !sub_args.vae_act_fn().empty()) {
    args_.vae_act_fn() = sub_args.vae_act_fn();
    arg_status_["vae_act_fn"] = true;
  }
  if (!arg_status_["vae_latent_channels"] &&
      sub_args.vae_latent_channels() != -1) {
    args_.vae_latent_channels() = sub_args.vae_latent_channels();
    arg_status_["vae_latent_channels"] = true;
  }
  if (!arg_status_["vae_norm_num_groups"] &&
      sub_args.vae_norm_num_groups() != -1) {
    args_.vae_norm_num_groups() = sub_args.vae_norm_num_groups();
    arg_status_["vae_norm_num_groups"] = true;
  }
  if (!arg_status_["vae_sample_size"] && sub_args.vae_sample_size() != -1) {
    args_.vae_sample_size() = sub_args.vae_sample_size();
    arg_status_["vae_sample_size"] = true;
  }
  if (!arg_status_["vae_scale_factor"] && sub_args.vae_scale_factor() != 0.0f) {
    args_.vae_scale_factor() = sub_args.vae_scale_factor();
    arg_status_["vae_scale_factor"] = true;
  }
  if (!arg_status_["vae_shift_factor"] && sub_args.vae_shift_factor() != 0.0f) {
    args_.vae_shift_factor() = sub_args.vae_shift_factor();
    arg_status_["vae_shift_factor"] = true;
  }
  if (!arg_status_["vae_mid_block_add_attention"]) {
    args_.vae_mid_block_add_attention() =
        sub_args.vae_mid_block_add_attention();
    arg_status_["vae_mid_block_add_attention"] = true;
  }
  if (!arg_status_["vae_force_upcast"]) {
    args_.vae_force_upcast() = sub_args.vae_force_upcast();
    arg_status_["vae_force_upcast"] = true;
  }
  if (!arg_status_["vae_use_quant_conv"]) {
    args_.vae_use_quant_conv() = sub_args.vae_use_quant_conv();
    arg_status_["vae_use_quant_conv"] = true;
  }
  if (!arg_status_["vae_use_post_quant_conv"]) {
    args_.vae_use_post_quant_conv() = sub_args.vae_use_post_quant_conv();
    arg_status_["vae_use_post_quant_conv"] = true;
  }

  // DiT
  if (!arg_status_["dit_num_layers"] && sub_args.dit_num_layers() != 0) {
    args_.dit_num_layers() = sub_args.dit_num_layers();
    arg_status_["dit_num_layers"] = true;
  }
  if (!arg_status_["dit_patch_size"] && sub_args.dit_patch_size() != 0) {
    args_.dit_patch_size() = sub_args.dit_patch_size();
    arg_status_["dit_patch_size"] = true;
  }
  if (!arg_status_["dit_in_channels"] && sub_args.dit_in_channels() != 0) {
    args_.dit_in_channels() = sub_args.dit_in_channels();
    arg_status_["dit_in_channels"] = true;
  }
  if (!arg_status_["dit_attention_head_dim"] &&
      sub_args.dit_attention_head_dim() != 0) {
    args_.dit_attention_head_dim() = sub_args.dit_attention_head_dim();
    arg_status_["dit_attention_head_dim"] = true;
  }
  if (!arg_status_["dit_num_attention_heads"] &&
      sub_args.dit_num_attention_heads() != 0) {
    args_.dit_num_attention_heads() = sub_args.dit_num_attention_heads();
    arg_status_["dit_num_attention_heads"] = true;
  }
  if (!arg_status_["dit_joint_attention_dim"] &&
      sub_args.dit_joint_attention_dim() != 0) {
    args_.dit_joint_attention_dim() = sub_args.dit_joint_attention_dim();
    arg_status_["dit_joint_attention_dim"] = true;
  }
  if (!arg_status_["dit_pooled_projection_dim"] &&
      sub_args.dit_pooled_projection_dim() != 0) {
    args_.dit_pooled_projection_dim() = sub_args.dit_pooled_projection_dim();
    arg_status_["dit_pooled_projection_dim"] = true;
  }
  if (!arg_status_["dit_guidance_embeds"]) {
    args_.dit_guidance_embeds() = sub_args.dit_guidance_embeds();
    arg_status_["dit_guidance_embeds"] = true;
  }
  if (!arg_status_["dit_axes_dims_rope"] &&
      !sub_args.dit_axes_dims_rope().empty()) {
    args_.dit_axes_dims_rope() = sub_args.dit_axes_dims_rope();
    arg_status_["dit_axes_dims_rope"] = true;
  }
  if (!arg_status_["dit_num_single_layers"] &&
      sub_args.dit_num_single_layers() != 0) {
    args_.dit_num_single_layers() = sub_args.dit_num_single_layers();
    arg_status_["dit_num_single_layers"] = true;
  }

  // T5
  if (!arg_status_["t5_vocab_size"] && sub_args.t5_vocab_size() != 0) {
    args_.t5_vocab_size() = sub_args.t5_vocab_size();
    arg_status_["t5_vocab_size"] = true;
  }
  if (!arg_status_["t5_d_model"] && sub_args.t5_d_model() != 0) {
    args_.t5_d_model() = sub_args.t5_d_model();
    arg_status_["t5_d_model"] = true;
  }
  if (!arg_status_["t5_num_layers"] && sub_args.t5_num_layers() != 0) {
    args_.t5_num_layers() = sub_args.t5_num_layers();
    arg_status_["t5_num_layers"] = true;
  }
  if (!arg_status_["t5_d_kv"] && sub_args.t5_d_kv() != 0) {
    args_.t5_d_kv() = sub_args.t5_d_kv();
    arg_status_["t5_d_kv"] = true;
  }
  if (!arg_status_["t5_num_heads"] && sub_args.t5_num_heads() != 0) {
    args_.t5_num_heads() = sub_args.t5_num_heads();
    arg_status_["t5_num_heads"] = true;
  }
  if (!arg_status_["t5_d_ff"] && sub_args.t5_d_ff() != 0) {
    args_.t5_d_ff() = sub_args.t5_d_ff();
    arg_status_["t5_d_ff"] = true;
  }
  if (!arg_status_["t5_dropout_rate"] && sub_args.t5_dropout_rate() != 0.0f) {
    args_.t5_dropout_rate() = sub_args.t5_dropout_rate();
    arg_status_["t5_dropout_rate"] = true;
  }
  if (!arg_status_["t5_dense_act_fn"] && !sub_args.t5_dense_act_fn().empty()) {
    args_.t5_dense_act_fn() = sub_args.t5_dense_act_fn();
    arg_status_["t5_dense_act_fn"] = true;
  }
  if (!arg_status_["t5_is_gated_act"]) {
    args_.t5_is_gated_act() = sub_args.t5_is_gated_act();
    arg_status_["t5_is_gated_act"] = true;
  }
  if (!arg_status_["t5_relative_attention_num_buckets"] &&
      sub_args.t5_relative_attention_num_buckets() != 0) {
    args_.t5_relative_attention_num_buckets() =
        sub_args.t5_relative_attention_num_buckets();
    arg_status_["t5_relative_attention_num_buckets"] = true;
  }
  if (!arg_status_["t5_relative_attention_max_distance"] &&
      sub_args.t5_relative_attention_max_distance() != 0) {
    args_.t5_relative_attention_max_distance() =
        sub_args.t5_relative_attention_max_distance();
    arg_status_["t5_relative_attention_max_distance"] = true;
  }
  if (!arg_status_["t5_layer_norm_epsilon"] &&
      sub_args.t5_layer_norm_epsilon() != 0.0f) {
    args_.t5_layer_norm_epsilon() = sub_args.t5_layer_norm_epsilon();
    arg_status_["t5_layer_norm_epsilon"] = true;
  }

  // Scheduler
  if (!arg_status_["scheduler_num_train_timesteps"] &&
      sub_args.scheduler_num_train_timesteps() != 0) {
    args_.scheduler_num_train_timesteps() =
        sub_args.scheduler_num_train_timesteps();
    arg_status_["scheduler_num_train_timesteps"] = true;
  }
  if (!arg_status_["scheduler_shift"] && sub_args.scheduler_shift() != 0) {
    args_.scheduler_shift() = sub_args.scheduler_shift();
    arg_status_["scheduler_shift"] = true;
  }
  if (!arg_status_["scheduler_use_dynamic_shifting"]) {
    args_.scheduler_use_dynamic_shifting() =
        sub_args.scheduler_use_dynamic_shifting();
    arg_status_["scheduler_use_dynamic_shifting"] = true;
  }
  if (!arg_status_["scheduler_base_shift"] &&
      sub_args.scheduler_base_shift() != 0.0f) {
    args_.scheduler_base_shift() = sub_args.scheduler_base_shift();
    arg_status_["scheduler_base_shift"] = true;
  }
  if (!arg_status_["scheduler_max_shift"] &&
      sub_args.scheduler_max_shift() != 0.0f) {
    args_.scheduler_max_shift() = sub_args.scheduler_max_shift();
    arg_status_["scheduler_max_shift"] = true;
  }
  if (!arg_status_["scheduler_base_image_seq_len"] &&
      sub_args.scheduler_base_image_seq_len() != 0) {
    args_.scheduler_base_image_seq_len() =
        sub_args.scheduler_base_image_seq_len();
    arg_status_["scheduler_base_image_seq_len"] = true;
  }
  if (!arg_status_["scheduler_max_image_seq_len"] &&
      sub_args.scheduler_max_image_seq_len() != 0) {
    args_.scheduler_max_image_seq_len() =
        sub_args.scheduler_max_image_seq_len();
    arg_status_["scheduler_max_image_seq_len"] = true;
  }

  // Clip
  if (!arg_status_["clip_vocab_size"] && sub_args.clip_vocab_size() != -1) {
    args_.clip_vocab_size() = sub_args.clip_vocab_size();
    arg_status_["clip_vocab_size"] = true;
  }
  if (!arg_status_["clip_hidden_size"] && sub_args.clip_hidden_size() != -1) {
    args_.clip_hidden_size() = sub_args.clip_hidden_size();
    arg_status_["clip_hidden_size"] = true;
  }
  if (!arg_status_["clip_intermediate_size"] &&
      sub_args.clip_intermediate_size() != -1) {
    args_.clip_intermediate_size() = sub_args.clip_intermediate_size();
    arg_status_["clip_intermediate_size"] = true;
  }
  if (!arg_status_["clip_projection_dim"] &&
      sub_args.clip_projection_dim() != -1) {
    args_.clip_projection_dim() = sub_args.clip_projection_dim();
    arg_status_["clip_projection_dim"] = true;
  }
  if (!arg_status_["clip_num_attention_heads"] &&
      sub_args.clip_num_attention_heads() != -1) {
    args_.clip_num_attention_heads() = sub_args.clip_num_attention_heads();
    arg_status_["clip_num_attention_heads"] = true;
  }
  if (!arg_status_["clip_num_hidden_layers"] &&
      sub_args.clip_num_hidden_layers() != -1) {
    args_.clip_num_hidden_layers() = sub_args.clip_num_hidden_layers();
    arg_status_["clip_num_hidden_layers"] = true;
  }
  if (!arg_status_["clip_layer_norm_eps"] &&
      sub_args.clip_layer_norm_eps() != -1) {
    args_.clip_layer_norm_eps() = sub_args.clip_layer_norm_eps();
    arg_status_["clip_layer_norm_eps"] = true;
  }
  if (!arg_status_["clip_hidden_act"] && !sub_args.clip_hidden_act().empty()) {
    args_.clip_hidden_act() = sub_args.clip_hidden_act();
    arg_status_["clip_hidden_act"] = true;
  }
  if (!arg_status_["clip_max_position_embeddings"] &&
      sub_args.clip_max_position_embeddings() != -1) {
    args_.clip_max_position_embeddings() =
        sub_args.clip_max_position_embeddings();
    arg_status_["clip_max_position_embeddings"] = true;
  }
  if (!arg_status_["clip_bos_token_id"] && sub_args.clip_bos_token_id() != 0) {
    args_.clip_bos_token_id() = sub_args.clip_bos_token_id();
    arg_status_["clip_bos_token_id"] = true;
  }
  if (!arg_status_["clip_eos_token_id"] && sub_args.clip_eos_token_id() != 0) {
    args_.clip_eos_token_id() = sub_args.clip_eos_token_id();
    arg_status_["clip_eos_token_id"] = true;
  }
  if (!arg_status_["clip_pad_token_id"] && sub_args.clip_pad_token_id() != 0) {
    args_.clip_pad_token_id() = sub_args.clip_pad_token_id();
    arg_status_["clip_pad_token_id"] = true;
  }
  if (!arg_status_["clip_attention_dropout"] &&
      sub_args.clip_attention_dropout() != 0.0f) {
    args_.clip_attention_dropout() = sub_args.clip_attention_dropout();
    arg_status_["clip_attention_dropout"] = true;
  }
  if (!arg_status_["clip_initializer_factor"] &&
      sub_args.clip_initializer_factor() != 0.0f) {
    args_.clip_initializer_factor() = sub_args.clip_initializer_factor();
    arg_status_["clip_initializer_factor"] = true;
  }
  if (!arg_status_["clip_initializer_range"] &&
      sub_args.clip_initializer_range() != 0.0f) {
    args_.clip_initializer_range() = sub_args.clip_initializer_range();
    arg_status_["clip_initializer_range"] = true;
  }
  if (!arg_status_["clip_head_dim"] && sub_args.clip_head_dim() != 0) {
    args_.clip_head_dim() = sub_args.clip_head_dim();
    arg_status_["clip_head_dim"] = true;
  }
}

const DiTFolderLoader* DiTModelLoader::get_sub_model_loader_by_name(
    const std::string& component_name) const {
  auto it = name_to_loader_.find(component_name);
  if (it == name_to_loader_.end()) {
    LOG(WARNING) << "Component not found: " << component_name;
    return nullptr;
  }
  return it->second.get();
}

const DiTFolderLoader* DiTModelLoader::get_sub_model_loader_by_folder(
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

std::unique_ptr<DiTFolderLoader> DiTModelLoader::take_sub_model_loader_by_name(
    const std::string& component_name) {
  auto it = name_to_loader_.find(component_name);
  if (it == name_to_loader_.end()) {
    LOG(WARNING) << "Component not found: " << component_name;
    return nullptr;
  }

  // move ownership out
  std::unique_ptr<DiTFolderLoader> taken = std::move(it->second);

  // erase from maps / lists
  name_to_loader_.erase(it);
  name_to_folder_.erase(component_name);

  // remove from component_names_
  component_names_.erase(
      std::remove(
          component_names_.begin(), component_names_.end(), component_name),
      component_names_.end());

  return taken;
}

std::unique_ptr<DiTFolderLoader>
DiTModelLoader::take_sub_model_loader_by_folder(
    const std::string& component_folder) {
  LOG(INFO) << "Taking sub model loader by folder: " << component_folder;
  std::filesystem::path target_path(component_folder);
  std::string target_folder = target_path.filename().string();
  for (auto it = name_to_folder_.begin(); it != name_to_folder_.end(); ++it) {
    std::filesystem::path current_path(it->second);
    std::string current_folder = current_path.filename().string();
    LOG(INFO) << "Checking folder: " << current_folder
              << " against: " << target_folder;
    if (current_folder == target_folder) {
      const std::string name = it->first;

      auto loader_it = name_to_loader_.find(name);
      if (loader_it == name_to_loader_.end()) {
        // inconsistency: folder exists but loader not found
        LOG(WARNING) << "Loader for folder found in name_to_folder_ but "
                        "missing in name_to_loader_: "
                     << name;
        // remove folder entry and continue/return nullptr as you prefer
        name_to_folder_.erase(it);
        component_names_.erase(
            std::remove(component_names_.begin(), component_names_.end(), name),
            component_names_.end());
        return nullptr;
      }

      std::unique_ptr<DiTFolderLoader> taken = std::move(loader_it->second);

      // erase loader and folder entries
      name_to_loader_.erase(loader_it);
      // erase by iterator `it` (valid)
      name_to_folder_.erase(it);

      // remove from component_names_
      component_names_.erase(
          std::remove(component_names_.begin(), component_names_.end(), name),
          component_names_.end());

      return taken;
    }
  }

  LOG(WARNING) << "Component folder not found: " << component_folder;
  return nullptr;
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

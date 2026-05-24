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

#include <glog/logging.h>
#include <torch/torch.h>

#include <algorithm>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "rec.pb.h"
#include "slice.h"
#include "tensor.pb.h"
#include "worker.pb.h"

// -------------------
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_set>

#include "core/framework/config/disagg_pd_config.h"
#include "core/framework/config/parallel_config.h"
#include "core/util/json_reader.h"
#include "models/model_registry.h"

namespace xllm {
namespace util {

std::pair<int, int> find_ones_indices(std::vector<int>& q_seq_lens);

template <typename T>
void pad_2d_vector(std::vector<std::vector<T>>& vec, T pad_value) {
  size_t max_col_size = 0;
  for (const auto& row : vec) {
    max_col_size = std::max(max_col_size, row.size());
  }

  for (auto& row : vec) {
    row.resize(max_col_size, pad_value);
  }
}

torch::ScalarType parse_dtype(const std::string& dtype_str,
                              const std::optional<torch::Device>& device);

std::optional<std::vector<uint32_t>> parse_batch_sizes(
    const std::string& batch_sizes_str);

template <typename T>
T sum(const std::vector<T>& vec) {
  if (vec.empty()) LOG(FATAL) << "vector is empty.";
  return std::accumulate(vec.begin(), vec.end(), T{});
}

template <typename T>
const T& min(const std::vector<T>& vec) {
  if (vec.empty()) LOG(FATAL) << "vector is empty.";
  return *std::min_element(vec.begin(), vec.end());
}

template <typename T>
const T& max(const std::vector<T>& vec) {
  if (vec.empty()) LOG(FATAL) << "vector is empty.";
  return *std::max_element(vec.begin(), vec.end());
}

template <typename T>
inline std::enable_if_t<std::is_integral_v<T>, T> ceil_div(T value, T divisor) {
  CHECK_GT(divisor, 0) << "divisor must be positive.";
  return value / divisor + static_cast<T>(value % divisor != 0);
}

static inline int64_t align_up(int64_t value, int64_t alignment) {
  if (alignment == 0) {
    return value;
  }
  return ((value + alignment - 1) / alignment) * alignment;
}

bool match_suffix(const Slice<int32_t>& data, const Slice<int32_t>& suffix);

std::vector<uint32_t> cal_vec_split_index(uint32_t vec_size, uint32_t part_num);

torch::Tensor convert_rec_tensor_to_torch(
    const proto::InferInputTensor& input_tensor);

torch::Tensor proto_to_torch(const proto::Tensor& proto_tensor);

bool torch_to_proto(const torch::Tensor& torch_tensor,
                    proto::Tensor* proto_tensor);

int32_t ceil_pow2(int32_t n);

torch::ScalarType datatype_proto_to_torch(const std::string& proto_datatype);

std::string torch_datatype_to_proto(torch::ScalarType torch_dtype);

inline const std::unordered_set<std::string>& mla_model_type_set() {
  static const std::unordered_set<std::string> kMlaModelTypeSet = {
      "deepseek_v2",
      "deepseek_v3",
      "deepseek_v32",
      "deepseek_v3_mtp",
      "deepseek_v32_mtp",
      "kimi_k2",
      "glm4_moe_lite",
      "glm_moe_dsa",  // glm5 model type
      "glm_moe_dsa_mtp",
      "joyai_llm_flash"};
  return kMlaModelTypeSet;
}

inline bool is_mla_model_type(std::string_view model_type) {
  return mla_model_type_set().contains(std::string(model_type));
}

inline std::string get_model_name(
    const std::filesystem::path& normalized_model_path) {
  std::string model_name;

  if (normalized_model_path.has_filename()) {
    model_name = normalized_model_path.filename().string();
  } else {
    model_name = normalized_model_path.parent_path().filename().string();
  }

  if (model_name.empty()) {
    LOG(FATAL) << "Cannot extract model name from path, as it appears to be a "
                  "root directory: "
               << normalized_model_path.string();
    return "";
  }

  return model_name;
}

inline std::string get_model_type(const std::filesystem::path& model_path) {
  JsonReader reader;

  // Try model_index.json first (DiT models like Cola-DLM, Flux, LongCat)
  std::filesystem::path model_index_path = model_path / "model_index.json";
  if (std::filesystem::exists(model_index_path)) {
    if (reader.parse(model_index_path)) {
      if (auto v = reader.value<std::string>("_class_name")) {
        return v.value();
      }
    }
  }

  std::filesystem::path config_json_path = model_path / "config.json";

  if (!std::filesystem::exists(config_json_path)) {
    // Auto-discovery: scan subdirectories for config.json with model_type.
    // Handles models like Cola-DLM where components are in subdirectories
    // without a root config.json or model_index.json.
    // Also handles nested layouts like Cola-DLM where the structure is:
    //   model_root/cola_dlm/cola_dit/config.json
    //   model_root/cola_dlm/cola_vae/config.json
    for (const auto& entry : std::filesystem::directory_iterator(model_path)) {
      if (!entry.is_directory()) continue;
      const std::string dir_name = entry.path().filename().string();
      if (dir_name.empty() || dir_name[0] == '.') continue;
      // Check this subdirectory for config.json
      std::filesystem::path sub_config = entry.path() / "config.json";
      if (std::filesystem::exists(sub_config)) {
        JsonReader sub_reader;
        if (sub_reader.parse(sub_config.string())) {
          if (auto v = sub_reader.value<std::string>("model_type")) {
            return v.value();
          }
        }
      }
      // If not found, check nested subdirectories (e.g. cola_dlm/cola_dit/)
      for (const auto& nested_entry :
           std::filesystem::directory_iterator(entry.path())) {
        if (!nested_entry.is_directory()) continue;
        const std::string nested_name = nested_entry.path().filename().string();
        if (nested_name.empty() || nested_name[0] == '.') continue;
        std::filesystem::path nested_config =
            nested_entry.path() / "config.json";
        if (!std::filesystem::exists(nested_config)) continue;
        JsonReader nested_reader;
        if (!nested_reader.parse(nested_config.string())) continue;
        if (auto v = nested_reader.value<std::string>("model_type")) {
          return v.value();
        }
      }
    }
    LOG(FATAL) << "Please check config.json or model_index.json file, one of "
                  "them should exist in the model path: "
               << model_path;
  }

  reader.parse(config_json_path);
  auto model_type = reader.value<std::string>("model_type");
  if (!model_type.has_value()) {
    model_type = reader.value<std::string>("model_name");
  }
  if (!model_type.has_value()) {
    LOG(FATAL) << "Please check config.json file in model path: " << model_path
               << ", it should contain model_type or model_name key.";
  }
  return model_type.value();
}

inline std::string get_model_backend(const std::filesystem::path& model_path) {
  JsonReader reader;
  std::filesystem::path model_index_json_path = model_path / "model_index.json";

  if (std::filesystem::exists(model_index_json_path)) {
    reader.parse(model_index_json_path);
    if (reader.value<std::string>("_diffusers_version").has_value()) {
      return "dit";
    }
    // DiT models that are not diffusers-based (e.g. Cola-DLM) may have
    // _class_name but no _diffusers_version. Treat them as dit backend.
    if (reader.value<std::string>("_class_name").has_value()) {
      return "dit";
    }
    LOG(FATAL) << "Please check model_index.json file in model path: "
               << model_path << ", it should contain _diffusers_version key.";
  }

  // Check if this looks like a DiT model with component subdirectories
  // (e.g. Cola-DLM with cola_dit/cola_vae subdirectories)
  for (const auto& entry : std::filesystem::directory_iterator(model_path)) {
    if (!entry.is_directory()) continue;
    const std::string dir_name = entry.path().filename().string();
    if (dir_name.empty() || dir_name[0] == '.') continue;
    std::filesystem::path sub_config = entry.path() / "config.json";
    if (std::filesystem::exists(sub_config)) {
      // Found a subdirectory with config.json - check for safetensors
      for (const auto& f : std::filesystem::directory_iterator(entry.path())) {
        if (f.path().extension() == ".safetensors") {
          return "dit";
        }
      }
    }
    // Also check nested subdirectories
    for (const auto& nested :
         std::filesystem::directory_iterator(entry.path())) {
      if (!nested.is_directory()) continue;
      std::filesystem::path nested_config = nested.path() / "config.json";
      if (std::filesystem::exists(nested_config)) {
        for (const auto& f :
             std::filesystem::directory_iterator(nested.path())) {
          if (f.path().extension() == ".safetensors") {
            return "dit";
          }
        }
      }
    }
  }

  return ModelRegistry::get_model_backend(get_model_type(model_path));
}

inline bool should_enable_mla(
    const std::filesystem::path& model_path,
    const std::optional<std::string>& backend = std::nullopt) {
  const std::string resolved_backend =
      backend.has_value() ? backend.value() : get_model_backend(model_path);
  if (resolved_backend == "dit") {
    return false;
  }
  return is_mla_model_type(get_model_type(model_path));
}

inline int32_t kv_split_size_effective(void) {
  return ParallelConfig::get_instance().kv_split_size_effective();
}

inline int32_t prefill_kv_split_size_effective(void) {
  return ParallelConfig::get_instance().prefill_kv_split_size_effective();
}

// PD KV transfer: P uses local kv_split; D uses prefill_kv_split_size only.
inline int32_t kv_split_stride_for_kv_transfer() {
  if (DisaggPDConfig::get_instance().enable_disagg_pd() &&
      DisaggPDConfig::get_instance().instance_role() == "DECODE") {
    return prefill_kv_split_size_effective();
  }
  return kv_split_size_effective();
}

inline bool enable_kvcache_split(void) { return kv_split_size_effective() > 1; }

}  // namespace util
}  // namespace xllm

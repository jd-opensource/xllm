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

#include <optional>
#include <string>

#include "nlohmann/json.hpp"
#include "util/slice.h"

namespace xllm {

class EngineType {
 public:
  enum Value : int8_t {
    LLM = 0,
    SSM = 1,
    VLM = 2,
    DIT = 3,
    INVALID = -1,
  };

  constexpr EngineType(Value v) : value_(v) {}
  EngineType(const std::string& str) {
    if (str == "LLM") {
      value_ = LLM;
    } else if (str == "SSM") {
      value_ = SSM;
    } else if (str == "VLM") {
      value_ = VLM;
    } else if (str == "DIT") {
      value_ = DIT;
    } else {
      value_ = INVALID;
    }
  }

  EngineType() = delete;

  constexpr operator Value() const { return value_; }
  explicit operator bool() = delete;

  bool operator==(EngineType rhs) const { return value_ == rhs.value_; }
  bool operator!=(EngineType rhs) const { return value_ != rhs.value_; }
  bool operator==(Value rhs) const { return value_ == rhs; }
  bool operator!=(Value rhs) const { return value_ != rhs; }

  constexpr const char* to_string() const {
    if (this->value_ == LLM) {
      return "LLM";
    } else if (this->value_ == SSM) {
      return "SSM";
    } else if (this->value_ == VLM) {
      return "VLM";
    } else if (this->value_ == DIT) {
      return "DIT";
    } else {
      return "INVALID";
    }
  }

 private:
  Value value_;
};

enum class StatusCode : uint8_t {
  OK = 0,
  // request was cancelled.
  CANCELLED = 1,
  // Unknown error.
  UNKNOWN = 2,
  // invalid argument.
  INVALID_ARGUMENT = 3,
  // timeout.
  DEADLINE_EXCEEDED = 4,
  // resource exhausted.
  RESOURCE_EXHAUSTED = 5,
};

class Status final {
 public:
  Status() = default;

  Status(StatusCode code) : code_(code) {}

  Status(StatusCode code, std::string msg)
      : code_(code), msg_(std::move(msg)) {}

  StatusCode code() const { return code_; }

  const std::string& message() const { return msg_; }

  bool ok() const { return code_ == StatusCode::OK; }

 private:
  StatusCode code_ = StatusCode::OK;
  std::string msg_;
};

class InstanceRole {
 public:
  enum RoleType : int8_t {
    DEFAULT = 0,
    PREFILL = 1,
    DECODE = 2,
    MIX = 3,
    INVALID = -1,
  };

  InstanceRole() : role_(DEFAULT) {}
  InstanceRole(const std::string& role) {
    if (role == "DEFAULT") {
      role_ = DEFAULT;
    } else if (role == "PREFILL") {
      role_ = PREFILL;
    } else if (role == "DECODE") {
      role_ = DECODE;
    } else if (role == "MIX") {
      role_ = MIX;
    } else {
      role_ = INVALID;
    }
  }

  InstanceRole(RoleType role) : role_(role) {}
  operator RoleType() const { return role_; }
  explicit operator bool() const = delete;

  bool operator==(InstanceRole rhs) const { return role_ == rhs.role_; }
  bool operator!=(InstanceRole rhs) const { return role_ != rhs.role_; }
  bool operator==(RoleType role) const { return role_ == role; }
  bool operator!=(RoleType role) const { return role_ != role; }

  std::string to_string() const {
    if (this->role_ == DEFAULT) {
      return "DEFAULT";
    } else if (this->role_ == PREFILL) {
      return "PREFILL";
    } else if (this->role_ == DECODE) {
      return "DECODE";
    } else if (this->role_ == MIX) {
      return "MIX";
    } else {
      return "INVALID";
    }
  }

 private:
  RoleType role_;
};

struct Token {
  explicit Token(int64_t id) : id(id) {}

  int64_t id = 0;
  std::optional<float> logprob;
  Slice<int64_t> top_tokens;
  Slice<float> top_logprobs;
};

struct RemoteToken {
  int64_t token_id;
  std::optional<float> token_logprob;
  std::vector<int64_t> token_top_tokens;
  std::vector<float> token_top_logprobs;
};

struct RawToken {
  int64_t id = 0;
  std::optional<float> logprob;
  std::vector<int64_t> top_tokens;
  std::vector<float> top_logprobs;
  std::vector<float> embeddings;  // hidden states
};

struct InstanceInfo {
  std::string name = "";
  std::string rpc_address = "";
  // DEFAULT/PREFILL/DECODE/MIX
  std::string type = "";
  // remote kv cache info
  std::vector<uint64_t> cluster_ids;
  std::vector<std::string> addrs;
  std::vector<int64_t> k_cache_ids;
  std::vector<int64_t> v_cache_ids;
  int32_t dp_size;
  // ttft profiling data
  std::vector<std::pair<int32_t, double>> ttft_profiling_data;
  // tpot profiling data
  std::vector<std::tuple<int32_t, int32_t, double>> tpot_profiling_data;

  nlohmann::json serialize_to_json() const {
    nlohmann::json json_val;
    json_val["name"] = name;
    json_val["rpc_address"] = rpc_address;
    if (InstanceRole(type) == InstanceRole::DEFAULT) {
      json_val["type"] = 0;
    } else if (InstanceRole(type) == InstanceRole::PREFILL) {
      json_val["type"] = 1;
    } else if (InstanceRole(type) == InstanceRole::DECODE) {
      json_val["type"] = 2;
    } else if (InstanceRole(type) == InstanceRole::MIX) {
      json_val["type"] = 3;
    } else {
      LOG(ERROR) << "Unsupported instance type: " << type;
      return json_val;
    }
    json_val["cluster_ids"] = cluster_ids;
    json_val["addrs"] = addrs;
    json_val["k_cache_ids"] = k_cache_ids;
    json_val["v_cache_ids"] = v_cache_ids;
    json_val["dp_size"] = dp_size;
    json_val["ttft_profiling_data"] = ttft_profiling_data;
    json_val["tpot_profiling_data"] = tpot_profiling_data;
    return json_val;
  }
};

struct TransferKVInfo {
  std::string request_id;
  std::vector<uint64_t> local_blocks_ids;
  std::vector<uint64_t> remote_blocks_ids;
  int32_t dp_rank;
  InstanceInfo remote_instance_info;
};

// in bytes
struct DeviceStats {
  int64_t total_memory = 0;
  int64_t weights_memory = 0;
  int64_t total_kv_cache_memory = 0;
  int64_t total_activation_memory = 0;
  int64_t active_activation_memory = 0;
  // TODO: add more device stats
};

// Function call related types
struct JsonFunction {
  std::string name;
  std::string description;
  nlohmann::json parameters;

  JsonFunction() = default;
  JsonFunction(const std::string& func_name,
               const std::string& desc,
               const nlohmann::json& params)
      : name(func_name), description(desc), parameters(params) {}
};

struct JsonTool {
  std::string type;  // "function"
  JsonFunction function;

  JsonTool() : type("function") {}
  JsonTool(const std::string& tool_type, const JsonFunction& func)
      : type(tool_type), function(func) {}
};
// Experts update the required information
struct EplbInfo {
  // Target layer ID for new expert weight pre-loading (-1 = no pending load)
  // Values >=0 indicate the layer ID that should start loading new expert
  // weights
  int32_t prepare_layer_id = -1;
  // Expert IDs requiring updates, ordered by device shard assignment
  // Contains per-device expert indices for distributed weight updates
  std::vector<int32_t> expert_ids;
  // Layer ID ready for expert weight activation (-1 = no pending update)
  // Values >=0 indicate the layer ID whose pre-loaded weights are ready for
  // deployment
  int32_t update_layer_id = -1;
};

struct MMInputData {
  std::string type = "";
  std::string text = "";
  std::string image_url = "";
  std::string video_url = "";
  std::string audio_url = "";
};

struct MMChatMessage {
  std::string role = "";
  std::vector<MMInputData> content;
};

inline constexpr int REC_TOKEN_SIZE = 3;

using RecTokenTriple = std::array<int32_t, REC_TOKEN_SIZE>;

}  // namespace xllm

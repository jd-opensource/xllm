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

#include "utils.h"

#include <cuda_runtime.h>

#include <cstdlib>

#include "core/platform/device.h"
#include "core/util/env_var.h"

namespace {
const std::unordered_map<torch::ScalarType, std::string_view>
    filename_safe_dtype_map = {
        {torch::kFloat16, "f16"},
        {torch::kBFloat16, "bf16"},
        {torch::kFloat8_e4m3fn, "e4m3"},
        {torch::kFloat8_e5m2, "e5m2"},
        {torch::kInt8, "i8"},
        {torch::kUInt8, "u8"},
        {torch::kInt32, "i32"},
        {torch::kUInt32, "u32"},
        {torch::kInt64, "i64"},
        {torch::kUInt64, "u64"},
};
}  // namespace

namespace xllm::kernel::cuda {

bool support_pdl() { return Device::is_enable_pdl(); }

std::string path_to_uri_so_lib(const std::string& uri) {
  return util::get_string_env("FLASHINFER_OPS_PATH") + "/" + uri + "/" + uri +
         ".so";
}

std::string determine_attention_backend(int64_t pos_encoding_mode,
                                        bool use_fp16_qk_reduction,
                                        bool use_custom_mask) {
  bool support_fa3_backend =
      (pos_encoding_mode == 0) && !use_fp16_qk_reduction && !use_custom_mask;

  if (Device::is_support_sm90a() && support_fa3_backend) {
    return "fa3";
  }
  return "fa2";
}

std::string get_batch_prefill_uri(const std::string& backend,
                                  torch::ScalarType dtype_q,
                                  torch::ScalarType dtype_kv,
                                  torch::ScalarType dtype_o,
                                  torch::ScalarType dtype_idx,
                                  int64_t head_dim_qk,
                                  int64_t head_dim_vo,
                                  int64_t pos_encoding_mode,
                                  bool use_sliding_window,
                                  bool use_logits_soft_cap,
                                  bool use_fp16_qk_reduction) {
  std::ostringstream oss;
  oss << "batch_prefill_with_kv_cache_"
      << "dtype_q_" << filename_safe_dtype_map.at(dtype_q) << "_"
      << "dtype_kv_" << filename_safe_dtype_map.at(dtype_kv) << "_"
      << "dtype_o_" << filename_safe_dtype_map.at(dtype_o) << "_"
      << "dtype_idx_" << filename_safe_dtype_map.at(dtype_idx) << "_"
      << "head_dim_qk_" << head_dim_qk << "_"
      << "head_dim_vo_" << head_dim_vo << "_"
      << "posenc_" << pos_encoding_mode << "_"
      << "use_swa_" << (use_sliding_window ? "True" : "False") << "_"
      << "use_logits_cap_" << (use_logits_soft_cap ? "True" : "False") << "_"
      << "f16qk_" << (use_fp16_qk_reduction ? "True" : "False");

  if (backend == "fa3") oss << "_sm90";

  return oss.str();
}

std::string get_batch_decode_uri(torch::ScalarType dtype_q,
                                 torch::ScalarType dtype_kv,
                                 torch::ScalarType dtype_o,
                                 torch::ScalarType dtype_idx,
                                 int64_t head_dim_qk,
                                 int64_t head_dim_vo,
                                 int64_t pos_encoding_mode,
                                 bool use_sliding_window,
                                 bool use_logits_soft_cap) {
  std::ostringstream oss;
  oss << "batch_decode_with_kv_cache_"
      << "dtype_q_" << filename_safe_dtype_map.at(dtype_q) << "_"
      << "dtype_kv_" << filename_safe_dtype_map.at(dtype_kv) << "_"
      << "dtype_o_" << filename_safe_dtype_map.at(dtype_o) << "_"
      << "dtype_idx_" << filename_safe_dtype_map.at(dtype_idx) << "_"
      << "head_dim_qk_" << head_dim_qk << "_"
      << "head_dim_vo_" << head_dim_vo << "_"
      << "posenc_" << pos_encoding_mode << "_"
      << "use_swa_" << (use_sliding_window ? "True" : "False") << "_"
      << "use_logits_cap_" << (use_logits_soft_cap ? "True" : "False");

  return oss.str();
}

// torch tensor is only on cpu
torch::Tensor get_cache_buffer(const int32_t seq_len,
                               const torch::Device& device) {
  static std::unordered_map<std::string, torch::Tensor> cache_buffer_map;
  int32_t seq_len_pow2 = xllm::util::ceil_pow2(seq_len);

  std::string key = std::string("range_") + std::to_string(seq_len_pow2);
  auto it = cache_buffer_map.find(key);
  if (it != cache_buffer_map.end()) {
    return it->second.slice(0, 0, seq_len);
  }

  auto options = torch::TensorOptions().dtype(torch::kInt32).device(device);
  torch::Tensor buffer = torch::arange(seq_len_pow2, options);
  cache_buffer_map.insert(std::make_pair(key, buffer));
  return buffer.slice(0, 0, seq_len);
}

}  // namespace xllm::kernel::cuda

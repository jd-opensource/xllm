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

#include "util/utils.h"

#include <absl/strings/numbers.h>
#include <absl/strings/str_split.h>

#include <algorithm>
#include <boost/algorithm/string.hpp>

namespace xllm {
namespace util {

std::pair<int, int> find_ones_indices(std::vector<int>& q_seq_lens) {
  int left = 0, right = q_seq_lens.size() - 1;
  int start_index = -1, end_index = -1;

  // Binary search for the start index of 1s
  while (left < right) {
    int mid = (left + right) / 2;
    if (q_seq_lens[mid] < 1) {
      left = mid + 1;
    } else {
      right = mid;
    }
  }
  if (q_seq_lens[left] == 1) {
    start_index = left;
  } else {
    return {start_index, end_index};  // No 1s found
  }

  left = 0;
  right = q_seq_lens.size() - 1;

  while (left < right) {
    int mid = (left + right + 1) / 2;
    if (q_seq_lens[mid] > 1) {
      right = mid - 1;
    } else {
      left = mid;
    }
  }
  if (q_seq_lens[right] == 1) {
    end_index = right;
  }
  return {start_index, end_index};
}

torch::ScalarType parse_dtype(const std::string& dtype_str,
                              const torch::Device& device) {
  if (device.is_cpu()) {
    // cpu only supports float32 for now
    return torch::kFloat32;
  }

  if (boost::iequals(dtype_str, "half") ||
      boost::iequals(dtype_str, "float16")) {
    return torch::kFloat16;
  }
  if (boost::iequals(dtype_str, "bfloat16")) {
    return torch::kBFloat16;
  }
  if ((boost::iequals(dtype_str, "float") ||
       boost::iequals(dtype_str, "float32"))) {
    return torch::kFloat16;
  }

  if (dtype_str.empty() || boost::iequals(dtype_str, "auto")) {
    return torch::kFloat16;
  }
  CHECK(false) << "Unsupported dtype: " << dtype_str << " on device " << device;
}

std::optional<std::vector<uint32_t>> parse_batch_sizes(
    const std::string& batch_sizes_str) {
  if (batch_sizes_str.empty() || batch_sizes_str == "auto") {
    return std::nullopt;
  }

  // parse devices string
  const auto size_strs = absl::StrSplit(batch_sizes_str, ',');
  // remove duplicates
  std::unordered_set<uint32_t> sizes_set;
  for (const auto& size_str : size_strs) {
    uint32_t batch_size = 0;
    if (!absl::SimpleAtoi(size_str, &batch_size)) {
      LOG(ERROR) << "Failed to parse batch size: " << size_str;
      continue;
    }
    sizes_set.emplace(batch_size);
  }
  if (sizes_set.empty()) {
    return std::nullopt;
  }
  return std::vector<uint32_t>{sizes_set.begin(), sizes_set.end()};
}

bool match_suffix(const Slice<int32_t>& data, const Slice<int32_t>& suffix) {
  if (suffix.empty()) {
    return true;
  }

  const auto data_len = data.size();
  const auto suf_len = suffix.size();
  if (data_len < suf_len) {
    return false;
  }

  const auto data_start = data.data() + (data_len - suf_len);
  const auto data_end = data.data() + data_len;
  return std::equal(data_start, data_end, suffix.data());
}

}  // namespace util
}  // namespace xllm

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

// Find the range of decode sequence indices (q_seq_lens == 1) in q_seq_lens
// Returns {start_index, end_index} of decode sequences,
// or {-1, -1} if no decode sequences found
std::pair<int, int> find_ones_indices(std::vector<int>& q_seq_lens) {
  int left = 0, right = q_seq_lens.size() - 1;
  int start_index = -1, end_index = -1;

  // Binary search for the start index of decode sequences (q_seq_lens == 1)
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
    return {start_index, end_index};  // No decode sequences found
  }

  left = 0;
  right = q_seq_lens.size() - 1;

  // Binary search for the end index of decode sequences (q_seq_lens == 1)
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

std::vector<uint32_t> cal_vec_split_index(uint32_t vec_size,
                                          uint32_t part_num) {
  std::vector<uint32_t> split_index;
  split_index.reserve(part_num + 1);
  split_index.push_back(0);

  if (part_num == 1) {
    split_index.push_back(vec_size);
  } else {
    auto base = vec_size / part_num;
    auto remainder = vec_size % part_num;
    for (auto i = 0; i < part_num; ++i) {
      split_index.push_back(split_index[i] +
                            ((i < remainder) ? (base + 1) : base));
    }
  }
  return split_index;
}

torch::Dtype convert_rec_type_to_torch(proto::DataType data_type) {
  // Future extensions go here.
  switch (data_type) {
    case proto::DataType::FLOAT:
      return torch::kFloat32;

    case proto::DataType::BFLOAT16:
      return torch::kBFloat16;

    case proto::DataType::BOOL:
      return torch::kBool;

    case proto::DataType::UINT8:
      return torch::kUInt8;

      // case proto::DataType::UINT32:
      //   return torch::kUInt32;

    case proto::DataType::INT8:
      return torch::kInt8;

    case proto::DataType::INT16:
      return torch::kInt16;

    default:
      throw std::runtime_error("Unsupported data type: " +
                               std::to_string(static_cast<int>(data_type)));
  }
}

torch::Tensor convert_rec_tensor_to_torch(
    const proto::InferInputTensor& input_tensor) {
  std::vector<int64_t> shape;
  shape.reserve(input_tensor.shape_size());
  for (int i = 0; i < input_tensor.shape_size(); ++i) {
    shape.push_back(input_tensor.shape(i));
  }

  if (!input_tensor.has_contents()) {
    throw std::runtime_error("Input tensor '" + input_tensor.name() +
                             "' has no contents");
  }

  const auto& contents = input_tensor.contents();
  torch::Dtype dtype = convert_rec_type_to_torch(input_tensor.data_type());

  switch (dtype) {
    case torch::kFloat32: {
      // Directly use protobuf's float array
      const auto& data = contents.fp32_contents();
      return torch::from_blob(
                 const_cast<float*>(data.data()),
                 shape,
                 torch::dtype(torch::kFloat32).requires_grad(false))
          .clone();  // Clone to ensure independent memory
    }
      // not support now.
      // case torch::kFloat16: {
      //   // Need type conversion (protobuf usually stores float16 as uint16)
      //   const auto& data = contents.bytes_contents();
      //   std::vector<at::Half> half_data;
      //   half_data.reserve(data.size());
      //   for (auto val : data) {
      //     half_data.push_back(static_cast<at::Half>(val));
      //   }
      //   return torch::tensor(half_data, torch::dtype(torch::kFloat16))
      //       .view(shape);
      // }

    case torch::kInt32: {
      const auto& data = contents.int_contents();
      return torch::from_blob(const_cast<int32_t*>(data.data()),
                              shape,
                              torch::dtype(torch::kInt32))
          .clone();
    }

    case torch::kInt64: {
      const auto& data = contents.int64_contents();
      return torch::from_blob(const_cast<int64_t*>(data.data()),
                              shape,
                              torch::dtype(torch::kInt64))
          .clone();
    }

    case torch::kBool: {
      const auto& data = contents.bool_contents();
      return torch::tensor(std::vector<uint8_t>(data.begin(), data.end()),
                           torch::dtype(torch::kBool))
          .view(shape);
    }

    default:
      throw std::runtime_error("Unhandled data type conversion for: " +
                               std::to_string(static_cast<int>(dtype)));
  }
}

}  // namespace util
}  // namespace xllm

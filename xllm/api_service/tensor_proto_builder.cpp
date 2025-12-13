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

#include "tensor_proto_builder.h"

namespace xllm {

TensorProtoBuilder::TensorProtoBuilder(bool use_binary_encoding)
    : use_binary_encoding_(use_binary_encoding) {};
TensorProtoBuilder::~TensorProtoBuilder() {};

bool TensorProtoBuilder::build_repeated_tensor(
    const std::vector<torch::Tensor>& in_tensors,
    google::protobuf::RepeatedPtrField<xllm::proto::Tensor>& out_tensors,
    std::string& binary_payload) {
  for (const auto& in_tensor : in_tensors) {
    CHECK(in_tensor.is_contiguous())
        << "Internal Error: only support contiguous mm_embedding";

    xllm::proto::Tensor* out_tensor = out_tensors.Add();

    if (!build_tensor(in_tensor, *out_tensor, binary_payload)) {
      return false;
    }
  }
  return true;
}

bool TensorProtoBuilder::build_tensor(const torch::Tensor& in_tensor,
                                      xllm::proto::Tensor& out_tensor,
                                      std::string& binary_payload) {
  if (use_binary_encoding_) {
    // build tensor in binary format
    out_tensor.set_datatype(
        util::torch_datatype_to_proto(in_tensor.scalar_type()));

    for (auto dim : in_tensor.sizes()) {
      out_tensor.add_shape(static_cast<int32_t>(dim));
    }

    auto numel = in_tensor.numel();
    auto byte_len = numel * in_tensor.element_size();
    size_t offset = binary_payload.size();
    auto* params = out_tensor.mutable_parameters();
    (*params)["offset"].set_int64_param(offset);
    (*params)["len"].set_int64_param(byte_len);
    (*params)["is_binary"].set_bool_param(true);
    binary_payload.append(reinterpret_cast<const char*>(in_tensor.data_ptr()),
                          byte_len);
  } else {
    // build tensor in json format
    util::torch_to_proto(in_tensor, &out_tensor);
  }
  return true;
}

};  // namespace xllm
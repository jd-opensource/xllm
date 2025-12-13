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

#include "core/common/message.h"
#include "core/common/types.h"
#include "core/util/utils.h"
#include "tensor.pb.h"

namespace xllm {
class TensorProtoBuilder {
 private:
  std::string torch_dtype_to_string(at::ScalarType dtype);

 public:
  TensorProtoBuilder(bool use_binary_encoding);
  ~TensorProtoBuilder();
  bool build_repeated_tensor(
      const std::vector<torch::Tensor>& in_tensors,
      google::protobuf::RepeatedPtrField<xllm::proto::Tensor>& out_tensors,
      std::string& binary_payload);
  bool build_tensor(const torch::Tensor& in_tensor,
                    xllm::proto::Tensor& out_tensor,
                    std::string& binary_payload);

 private:
  bool use_binary_encoding_;
};

}  // namespace xllm
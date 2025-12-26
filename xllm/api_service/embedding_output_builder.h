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
#include "core/framework//request/request_output.h"
#include "core/util/utils.h"
#include "embedding.pb.h"
#include "tensor.pb.h"

namespace xllm {
class TensorProtoBuilder {
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
  bool build_tensor(const xllm::proto::Tensor& in_tensor,
                    const std::string& binary_payload,
                    torch::Tensor& out_tensor);

 private:
  bool use_binary_encoding_;
};

class EmbeddingOutputBuilder {
 public:
  EmbeddingOutputBuilder(bool embedding_use_binary_encoding,
                         bool metadata_use_binary_encoding);
  ~EmbeddingOutputBuilder();
  bool build_repeated_embedding_output(
      const std::vector<EmbeddingOutput>& in_outputs,
      google::protobuf::RepeatedPtrField<xllm::proto::Embedding>& out_outputs,
      std::string& binary_payload);
  bool build_embedding_output(const EmbeddingOutput& in_output,
                              xllm::proto::Embedding& out_output,
                              std::string& binary_payload);
  bool build_embedding_output(const xllm::proto::Embedding& in_embedding_output,
                              std::string& binary_payload,
                              EmbeddingOutput& out_embedding_output);
  bool embedding_use_binary_encoding_;
  bool metadata_use_binary_encoding_;
};

}  // namespace xllm
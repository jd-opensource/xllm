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

#include "dit_request_params.h"

#include "butil/base64.h"
#include "core/common/instance_name.h"
#include "core/common/macros.h"
#include "core/util/uuid.h"
#include "mm_codec.h"
#include "request.h"

namespace xllm {
namespace {
thread_local ShortUUID short_uuid;

std::string generate_image_generation_request_id() {
  return "imggen-" + InstanceName::name()->get_name_hash() + "-" +
         short_uuid.random();
}

}  // namespace

static torch::ScalarType datatype_proto_to_torch(
    const std::string& proto_datatype) {
  static const std::unordered_map<std::string, torch::ScalarType> kDatatypeMap =
      {{"BOOL", torch::kBool},
       {"INT32", torch::kInt},
       {"INT64", torch::kLong},
       {"UINT32", torch::kInt32},
       {"UINT64", torch::kInt64},
       {"FP32", torch::kFloat},
       {"FP64", torch::kDouble},
       {"BYTES", torch::kByte}};

  auto iter = kDatatypeMap.find(proto_datatype);
  if (iter == kDatatypeMap.end()) {
    LOG(FATAL)
        << "Unsupported proto datatype: " << proto_datatype
        << " (supported types: BOOL/INT32/INT64/UINT32/UINT64/FP32/FP64/BYTES)";
  }
  return iter->second;
}

template <typename T>
static const void* get_data_from_contents(const proto::TensorContents& contents,
                                          const std::string& datatype) {
  if constexpr (std::is_same_v<T, bool>) {
    if (contents.bool_contents().empty()) {
      LOG(ERROR) << "TensorContents.bool_contents is empty (datatype="
                 << datatype << ")";
      return nullptr;
    }
    return contents.bool_contents().data();
  } else if constexpr (std::is_same_v<T, int32_t>) {
    if (contents.int_contents().empty()) {
      LOG(ERROR) << "TensorContents.int_contents is empty (datatype="
                 << datatype << ")";
      return nullptr;
    }
    return contents.int_contents().data();
  } else if constexpr (std::is_same_v<T, int64_t>) {
    if (contents.int64_contents().empty()) {
      LOG(ERROR) << "TensorContents.int64_contents is empty (datatype="
                 << datatype << ")";
      return nullptr;
    }
    return contents.int64_contents().data();
  } else if constexpr (std::is_same_v<T, uint32_t>) {
    if (contents.uint_contents().empty()) {
      LOG(ERROR) << "TensorContents.uint_contents is empty (datatype="
                 << datatype << ")";
      return nullptr;
    }
    return contents.uint_contents().data();
  } else if constexpr (std::is_same_v<T, uint64_t>) {
    if (contents.uint64_contents().empty()) {
      LOG(ERROR) << "TensorContents.uint64_contents is empty (datatype="
                 << datatype << ")";
      return nullptr;
    }
    return contents.uint64_contents().data();
  } else if constexpr (std::is_same_v<T, float>) {
    if (contents.fp32_contents().empty()) {
      LOG(ERROR) << "TensorContents.fp32_contents is empty (datatype="
                 << datatype << ")";
      return nullptr;
    }
    return contents.fp32_contents().data();
  } else if constexpr (std::is_same_v<T, double>) {
    if (contents.fp64_contents().empty()) {
      LOG(ERROR) << "TensorContents.fp64_contents is empty (datatype="
                 << datatype << ")";
      return nullptr;
    }
    return contents.fp64_contents().data();
  } else {
    LOG(FATAL) << "Unsupported data type for TensorContents: "
               << typeid(T).name();
    return nullptr;
  }
}

torch::Tensor proto_to_torch(const proto::Tensor& proto_tensor) {
  if (proto_tensor.datatype().empty()) {
    LOG(ERROR) << "Proto Tensor missing required field: datatype (e.g., "
                  "\"FP32\", \"INT64\")";
    return torch::Tensor();
  }
  if (proto_tensor.shape().empty()) {
    LOG(ERROR) << "Proto Tensor has empty shape (invalid tensor)";
    return torch::Tensor();
  }
  if (!proto_tensor.has_contents()) {
    LOG(ERROR)
        << "Proto Tensor missing required field: contents (TensorContents)";
    return torch::Tensor();
  }
  const auto& proto_contents = proto_tensor.contents();

  const std::string& proto_datatype = proto_tensor.datatype();
  torch::ScalarType torch_dtype = datatype_proto_to_torch(proto_datatype);
  const size_t element_size = torch::elementSize(torch_dtype);

  std::vector<int64_t> torch_shape;
  int64_t total_elements = 1;
  for (const auto& dim : proto_tensor.shape()) {
    if (dim <= 0) {
      LOG(ERROR) << "Proto Tensor has invalid dimension: " << dim
                 << " (must be positive, datatype=" << proto_datatype << ")";
      return torch::Tensor();
    }
    torch_shape.emplace_back(dim);
    total_elements *= dim;
  }
  torch::IntArrayRef tensor_shape(torch_shape);

  const void* data_ptr = nullptr;
  size_t data_count = 0;
  if (proto_datatype == "BOOL") {
    data_ptr = get_data_from_contents<bool>(proto_contents, proto_datatype);
    data_count = proto_contents.bool_contents_size();
  } else if (proto_datatype == "INT32") {
    data_ptr = get_data_from_contents<int32_t>(proto_contents, proto_datatype);
    data_count = proto_contents.int_contents_size();
  } else if (proto_datatype == "INT64") {
    data_ptr = get_data_from_contents<int64_t>(proto_contents, proto_datatype);
    data_count = proto_contents.int64_contents_size();
  } else if (proto_datatype == "UINT32") {
    data_ptr = get_data_from_contents<uint32_t>(proto_contents, proto_datatype);
    data_count = proto_contents.uint_contents_size();
  } else if (proto_datatype == "UINT64") {
    data_ptr = get_data_from_contents<uint64_t>(proto_contents, proto_datatype);
    data_count = proto_contents.uint64_contents_size();
  } else if (proto_datatype == "FP32") {
    data_ptr = get_data_from_contents<float>(proto_contents, proto_datatype);
    data_count = proto_contents.fp32_contents_size();
  } else if (proto_datatype == "FP64") {
    data_ptr = get_data_from_contents<double>(proto_contents, proto_datatype);
    data_count = proto_contents.fp64_contents_size();
  }

  if (data_ptr == nullptr) {
    LOG(ERROR) << "Failed to get data from TensorContents (datatype="
               << proto_datatype << ")";
    return torch::Tensor();
  }
  if (data_count != static_cast<size_t>(total_elements)) {
    LOG(ERROR) << "Proto Tensor data count mismatch (datatype="
               << proto_datatype << "): "
               << "expected " << total_elements
               << " elements (shape=" << tensor_shape << "), "
               << "got " << data_count << " elements";
    return torch::Tensor();
  }

  torch::Tensor tensor =
      torch::from_blob(const_cast<void*>(data_ptr), tensor_shape, torch_dtype)
          .clone();
  return tensor;
}

std::pair<int, int> splitResolution(const std::string& s) {
  size_t pos = s.find('*');
  int width = std::stoi(s.substr(0, pos));
  int height = std::stoi(s.substr(pos + 1));
  return {width, height};
}

DiTRequestParams::DiTRequestParams(const proto::ImageGenerationRequest& request,
                                   const std::string& x_rid,
                                   const std::string& x_rtime) {
  if (request.has_request_id()) {
    request_id = request.request_id();
  } else {
    request_id = generate_image_generation_request_id();
  }
  x_request_id = x_rid;
  x_request_time = x_rtime;

  model = request.model();

  // input params
  const auto& input = request.input();
  input_params.prompt = input.prompt();
  if (input.has_prompt_2()) {
    input_params.prompt_2 = input.prompt_2();
  }
  if (input.has_negative_prompt()) {
    input_params.negative_prompt = input.negative_prompt();
  }
  if (input.has_negative_prompt_2()) {
    input_params.negative_prompt_2 = input.negative_prompt_2();
  }

  if (input.has_prompt_embed()) {
    input_params.prompt_embed = proto_to_torch(input.prompt_embed());
  }
  if (input.has_pooled_prompt_embed()) {
    input_params.pooled_prompt_embed =
        proto_to_torch(input.pooled_prompt_embed());
  }
  if (input.has_negative_prompt_embed()) {
    input_params.negative_prompt_embed =
        proto_to_torch(input.negative_prompt_embed());
  }
  if (input.has_negative_pooled_prompt_embed()) {
    input_params.negative_pooled_prompt_embed =
        proto_to_torch(input.negative_pooled_prompt_embed());
  }
  if (input.has_latent()) {
    input_params.latent = proto_to_torch(input.latent());
  }
  if (input.has_masked_image_latent()) {
    input_params.masked_image_latent =
        proto_to_torch(input.masked_image_latent());
  }

  OpenCVImageDecoder decoder;
  if (input.has_image()) {
    std::string raw_bytes;
    if (!butil::Base64Decode(input.image(), &raw_bytes)) {
      LOG(ERROR) << "Base64 image decode failed";
    }
    if (!decoder.decode(raw_bytes, input_params.image)) {
      LOG(ERROR) << "Image decode failed.";
    }
  }

  if (input.has_mask_image()) {
    std::string raw_bytes;
    if (!butil::Base64Decode(input.mask_image(), &raw_bytes)) {
      LOG(ERROR) << "Base64 mask_image decode failed";
    }
    if (!decoder.decode(raw_bytes, input_params.mask_image)) {
      LOG(ERROR) << "Mask_image decode failed.";
    }
  }

  // generation params
  const auto& params = request.parameters();
  if (params.has_size()) {
    auto size = splitResolution(params.size());
    generation_params.width = size.first;
    generation_params.height = size.second;
  }
  if (params.has_num_inference_steps()) {
    generation_params.num_inference_steps = params.num_inference_steps();
  }
  if (params.has_true_cfg_scale()) {
    generation_params.true_cfg_scale = params.true_cfg_scale();
  }
  if (params.has_guidance_scale()) {
    generation_params.guidance_scale = params.guidance_scale();
  }
  if (params.has_num_images_per_prompt()) {
    generation_params.num_images_per_prompt =
        static_cast<uint32_t>(params.num_images_per_prompt());
  } else {
    generation_params.num_images_per_prompt = 1;
  }
  if (params.has_seed()) {
    generation_params.seed = params.seed();
  }
  if (params.has_max_sequence_length()) {
    generation_params.max_sequence_length = params.max_sequence_length();
  }
}

bool DiTRequestParams::verify_params(
    std::function<bool(DiTRequestOutput)> callback) const {
  if (input_params.prompt.empty()) {
    CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT, "prompt is empty");
    return false;
  }

  return true;
}

}  // namespace xllm

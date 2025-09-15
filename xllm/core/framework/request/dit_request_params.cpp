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

#include "core/common/instance_name.h"
#include "core/util/uuid.h"
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

static std::optional<torch::Tensor> proto_to_torch(
    const proto::Tensor& proto_tensor) {
  if (proto_tensor.datatype().empty()) {
    LOG(ERROR) << "Proto Tensor missing required field: datatype (e.g., "
                  "\"FP32\", \"INT64\")";
    return std::nullopt;
  }
  if (proto_tensor.shape().empty()) {
    LOG(ERROR) << "Proto Tensor has empty shape (invalid tensor)";
    return std::nullopt;
  }
  if (!proto_tensor.has_contents()) {
    LOG(ERROR)
        << "Proto Tensor missing required field: contents (TensorContents)";
    return std::nullopt;
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
      return std::nullopt;
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
    return std::nullopt;
  }
  if (data_count != static_cast<size_t>(total_elements)) {
    LOG(ERROR) << "Proto Tensor data count mismatch (datatype="
               << proto_datatype << "): "
               << "expected " << total_elements
               << " elements (shape=" << tensor_shape << "), "
               << "got " << data_count << " elements";
    return std::nullopt;
  }

  torch::Tensor tensor =
      torch::from_blob(const_cast<void*>(data_ptr), tensor_shape, torch_dtype)
          .clone();
  return tensor;
}

DiTRequestParams::DiTRequestParams(const proto::ImageGenerationRequest& request,
                                   const std::string& x_rid,
                                   const std::string& x_rtime) {
  request_id = generate_image_generation_request_id();
  x_request_id = x_rid;
  x_request_time = x_rtime;
  model = request.model();
  if (request.has_service_request_id()) {
    service_request_id = request.service_request_id();
  }
  const auto& proto_input = request.input();
  input_params.prompt = proto_input.prompt();
  if (proto_input.has_prompt_2()) {
    input_params.prompt_2 = proto_input.prompt_2();
  }
  if (proto_input.has_negative_prompt()) {
    input_params.negative_prompt = proto_input.negative_prompt();
  }
  if (proto_input.has_negative_prompt_2()) {
    input_params.negative_prompt_2 = proto_input.negative_prompt_2();
  }
  LOG(INFO) << "Prompt: " << input_params.prompt
            << "start to convert prompt_embeds";
  if (proto_input.has_prompt_embeds()) {
    const auto& proto_tensor = proto_input.prompt_embeds();
    input_params.prompt_embeds = proto_to_torch(proto_tensor);
  }
  if (proto_input.has_pooled_prompt_embeds()) {
    input_params.pooled_prompt_embeds =
        proto_to_torch(proto_input.pooled_prompt_embeds());
  }
  if (proto_input.has_negative_prompt_embeds()) {
    input_params.negative_prompt_embeds =
        proto_to_torch(proto_input.negative_prompt_embeds());
  }
  if (proto_input.has_negative_pooled_prompt_embeds()) {
    input_params.negative_pooled_prompt_embeds =
        proto_to_torch(proto_input.negative_pooled_prompt_embeds());
  }
  if (proto_input.has_latents()) {
    const auto& proto_tensor = proto_input.latents();
    input_params.latents = proto_to_torch(proto_tensor);
  }
  const auto& proto_params = request.parameters();
  if (proto_params.has_size()) {
    generation_params.size = proto_params.size();
  }
  if (proto_params.has_num_inference_steps()) {
    generation_params.num_inference_steps = proto_params.num_inference_steps();
  }
  if (proto_params.has_true_cfg_scale()) {
    generation_params.true_cfg_scale = proto_params.true_cfg_scale();
  }
  if (proto_params.has_guidance_scale()) {
    generation_params.guidance_scale = proto_params.guidance_scale();
  }
  if (proto_params.has_num_images_per_prompt()) {
    generation_params.num_images_per_prompt =
        static_cast<uint32_t>(proto_params.num_images_per_prompt());
  } else {
    generation_params.num_images_per_prompt = 1;
  }
  if (proto_params.has_seed()) {
    generation_params.seed = proto_params.seed();
  }
  if (proto_params.has_max_sequence_length()) {
    generation_params.max_sequence_length = proto_params.max_sequence_length();
  }
}

bool DiTRequestParams::verify_params(
    std::function<bool(DiTRequestOutput)> callback) const {
  return true;
}

}  // namespace xllm
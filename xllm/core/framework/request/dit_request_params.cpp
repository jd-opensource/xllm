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
#include "core/util/utils.h"
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
    input_params.prompt_embed = util::proto_to_torch(input.prompt_embed());
  }
  if (input.has_pooled_prompt_embed()) {
    input_params.pooled_prompt_embed =
        util::proto_to_torch(input.pooled_prompt_embed());
  }
  if (input.has_negative_prompt_embed()) {
    input_params.negative_prompt_embed =
        util::proto_to_torch(input.negative_prompt_embed());
  }
  if (input.has_negative_pooled_prompt_embed()) {
    input_params.negative_pooled_prompt_embed =
        util::proto_to_torch(input.negative_pooled_prompt_embed());
  }
  if (input.has_latent()) {
    input_params.latent = util::proto_to_torch(input.latent());
  }
  if (input.has_masked_image_latent()) {
    input_params.masked_image_latent =
        util::proto_to_torch(input.masked_image_latent());
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

  if (input.has_control_image()) {
    std::string raw_bytes;
    if (!butil::Base64Decode(input.control_image(), &raw_bytes)) {
      LOG(ERROR) << "Base64 control_image decode failed";
    }
    if (!decoder.decode(raw_bytes, input_params.control_image)) {
      LOG(ERROR) << "Control_image decode failed.";
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

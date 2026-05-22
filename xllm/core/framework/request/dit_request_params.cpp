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
#include "core/framework/config/dit_config.h"
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

std::string generate_audio_generation_request_id() {
  return "audiogen-" + InstanceName::name()->get_name_hash() + "-" +
         short_uuid.random();
}

// Decode a base64-encoded WAV/audio blob into a float32 CPU tensor of shape
// (1, num_samples) at the given sample rate (mono).
// Returns true on success and writes to `out`; logs ERROR and returns false
// on any decoding failure.
bool decode_prompt_audio(const std::string& b64_audio,
                         int64_t target_sample_rate,
                         torch::Tensor& out) {
  std::string raw_bytes;
  if (!butil::Base64Decode(b64_audio, &raw_bytes)) {
    LOG(ERROR) << "Base64 prompt_audio decode failed";
    return false;
  }
  FFmpegAudioDecoder audio_decoder;
  AudioMetadata audio_meta;
  torch::Tensor audio_tensor;
  if (!audio_decoder.decode(
          raw_bytes, audio_tensor, audio_meta, target_sample_rate)) {
    LOG(ERROR) << "prompt_audio WAV decode failed";
    return false;
  }
  // Ensure shape (1, num_samples): FFmpegAudioDecoder may return
  // (num_samples,) or (num_channels, num_samples).
  if (audio_tensor.dim() == 1) {
    audio_tensor = audio_tensor.unsqueeze(0);
  } else if (audio_tensor.dim() == 2 && audio_tensor.size(0) > 1) {
    audio_tensor = audio_tensor.mean(0, /*keepdim=*/true);
  }
  out = audio_tensor.to(torch::kFloat32);
  return true;
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

  input_params.images.reserve(input.images().size());
  for (const auto& image : input.images()) {
    std::string binary;
    if (!butil::Base64Decode(image, &binary)) {
      LOG(ERROR) << "Base64 image decode failed";
      continue;
    }
    torch::Tensor tensor;
    if (!decoder.decode(binary, tensor)) {
      LOG(ERROR) << "Image decode failed.";
      continue;
    }
    input_params.images.emplace_back(std::move(tensor));
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
  if (params.has_enable_cfg_renorm()) {
    generation_params.enable_cfg_renorm = params.enable_cfg_renorm();
  }
  if (params.has_cfg_renorm_min()) {
    generation_params.cfg_renorm_min = params.cfg_renorm_min();
  }
}

DiTRequestParams::DiTRequestParams(const proto::AudioGenerationRequest& request,
                                   const std::string& x_rid,
                                   const std::string& x_rtime) {
  if (request.has_request_id()) {
    request_id = request.request_id();
  } else {
    request_id = generate_audio_generation_request_id();
  }
  x_request_id = x_rid;
  x_request_time = x_rtime;
  model = request.model();

  // generation params — must be parsed before input to make audio_sampling_rate
  // available for prompt audio decoding.
  const auto& params = request.parameters();
  if (params.has_seed()) {
    generation_params.seed = params.seed();
  }
  if (params.has_max_sequence_length()) {
    generation_params.max_sequence_length = params.max_sequence_length();
  }
  if (params.has_guidance_scale()) {
    generation_params.guidance_scale = params.guidance_scale();
  }
  if (params.has_audio_duration_frames()) {
    generation_params.audio_duration_frames = params.audio_duration_frames();
  }
  if (params.has_audio_steps()) {
    generation_params.audio_steps = params.audio_steps();
  }
  if (params.has_audio_guidance_method()) {
    generation_params.audio_guidance_method = params.audio_guidance_method();
  }
  if (params.has_sampling_rate()) {
    generation_params.audio_sampling_rate = params.sampling_rate();
  }

  // input params — decode prompt audio using the model's sampling rate.
  const auto& input = request.input();
  input_params.prompt = input.prompt();

  if (input.has_prompt_audio()) {
    decode_prompt_audio(input.prompt_audio(),
                        generation_params.audio_sampling_rate,
                        input_params.prompt_audio);
  }
  if (input.has_prompt_text()) {
    input_params.audio_prompt_text = input.prompt_text();
  }
}

bool DiTRequestParams::verify_params(
    std::function<bool(DiTRequestOutput)> callback) const {
  if (input_params.prompt.empty() && !input_params.prompt_embed.defined()) {
    CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT, "prompt is empty");
    return false;
  }

  if (generation_params.width < 0 || generation_params.height < 0) {
    CALLBACK_WITH_ERROR(
        StatusCode::INVALID_ARGUMENT,
        "Invalid image dimensions: width and height must be non-negative.");
    return false;
  }

  // Check if the image area exceeds the maximum allowed area.
  if (::xllm::DiTConfig::get_instance().dit_generation_image_area_max() > 0) {
    int64_t area = static_cast<int64_t>(generation_params.width) *
                   static_cast<int64_t>(generation_params.height);
    if (area >
        ::xllm::DiTConfig::get_instance().dit_generation_image_area_max()) {
      CALLBACK_WITH_ERROR(
          StatusCode::INVALID_ARGUMENT,
          "Requested image area (" + std::to_string(area) +
              ") exceeds the maximum allowed area (" +
              std::to_string(::xllm::DiTConfig::get_instance()
                                 .dit_generation_image_area_max()) +
              ").");
      return false;
    }
  }

  return true;
}

}  // namespace xllm

#pragma once
#include <cstdint>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <vector>

#include "tensor.pb.h"

namespace xllm {
struct GenerationParams {
  std::optional<std::string> size;

  int32_t width = 512;

  int32_t height = 512;

  std::optional<int32_t> num_inference_steps;

  std::optional<float> true_cfg_scale;

  std::optional<float> guidance_scale;

  std::optional<uint32_t> num_images_per_prompt = 1;

  std::optional<int64_t> seed;

  std::optional<int32_t> max_sequence_length;
};

struct InputParams {
  std::string prompt;

  std::optional<std::string> prompt_2;

  std::optional<std::string> negative_prompt;

  std::optional<std::string> negative_prompt_2;

  // std::optional<std::string> ip_adapter_image;

  // std::optional<std::string> negative_ip_adapter_image;

  std::optional<proto::Tensor> prompt_embeds;

  std::optional<proto::Tensor> pooled_prompt_embeds;

  // std::optional<std::vector<std::vector<std::vector<float>>>>
  //     ip_adapter_image_embeds;

  std::optional<proto::Tensor> negative_prompt_embeds;

  std::optional<proto::Tensor> negative_pooled_prompt_embeds;

  // std::optional<std::vector<std::vector<std::vector<float>>>>
  //     negative_ip_adapter_image_embeds;

  std::optional<proto::Tensor> latents;
};
}  // namespace xllm
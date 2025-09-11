#pragma once
#include <torch/torch.h>

#include <cstdint>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <vector>

#include "dit_request_output.h"
#include "image_generation.pb.h"
#include "request.h"
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

  std::optional<torch::Tensor> prompt_embeds;

  std::optional<torch::Tensor> pooled_prompt_embeds;

  // std::optional<std::vector<std::vector<std::vector<float>>>>
  //     ip_adapter_image_embeds;

  std::optional<torch::Tensor> negative_prompt_embeds;

  std::optional<torch::Tensor> negative_pooled_prompt_embeds;

  // std::optional<std::vector<std::vector<std::vector<float>>>>
  //     negative_ip_adapter_image_embeds;

  std::optional<torch::Tensor> latents;
};

struct DiTRequestParams {
  DiTRequestParams() = default;
  DiTRequestParams(const proto::ImageGenerationRequest& request,
                   const std::string& x_rid,
                   const std::string& x_rtime);

  bool verify_params(DiTOutputCallback callback) const;

  // request id
  std::string request_id;
  std::string service_request_id = "";
  std::string x_request_id;
  std::string x_request_time;

  std::string model;

  bool offline = false;

  int32_t slo_ms = 0;

  RequestPriority priority = RequestPriority::NORMAL;

  InputParams input_params;
  // Mandatory: Generation control parameters (encapsulates all fields related
  // to "image generation process")
  GenerationParams generation_params;
};

}  // namespace xllm
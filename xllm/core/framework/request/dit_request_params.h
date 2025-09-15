#pragma once
#include <torch/torch.h>

#include <cstdint>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <vector>

#include "dit_request_output.h"
#include "dit_request_state.h"
#include "image_generation.pb.h"
#include "request.h"
#include "tensor.pb.h"
namespace xllm {

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
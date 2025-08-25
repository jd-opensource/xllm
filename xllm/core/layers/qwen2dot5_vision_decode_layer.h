#pragma once

// #ifdef WITH_ASCEND
#include "npu/npu_qwen2dot5_vision_encoder_layer_impl.h"
// #endif

#include "pytorch/adapter/utils/utils.h"

namespace xllm {
namespace layer {
// #ifdef WITH_ASCEND
class Qwen2dot5VisionEncoderLayer
    : public torch::nn::ModuleHolder<NpuQwen2dot5VisionEncoderLayerImpl> {
 public:
  using torch::nn::ModuleHolder<
      NpuQwen2dot5VisionEncoderLayerImpl>::ModuleHolder;
  using Impl __attribute__((__unused__)) = NpuQwen2dot5VisionEncoderLayerImpl;

  Qwen2dot5VisionEncoderLayer(const Context& context)
      : ModuleHolder(
            std::make_shared<NpuQwen2dot5VisionEncoderLayerImpl>(context)) {}
};
// #endif

}  // namespace layer
}  // namespace xllm

#pragma once

// #ifdef WITH_ASCEND
#include "npu/npu_siglip_encoder_layer_impl.h"
// #endif

#include "pytorch/adapter/utils/utils.h"

namespace xllm {
namespace layer {
// #ifdef WITH_ASCEND
class SiglipEncoderLayer
    : public torch::nn::ModuleHolder<NpuSiglipEncoderLayerImpl> {
 public:
  using torch::nn::ModuleHolder<NpuSiglipEncoderLayerImpl>::ModuleHolder;
  using Impl __attribute__((__unused__)) = NpuSiglipEncoderLayerImpl;

  SiglipEncoderLayer(const Context& context, const std::string& prefix = "")
      : ModuleHolder(
            std::make_shared<NpuSiglipEncoderLayerImpl>(context, prefix)) {}
};
// #endif

}  // namespace layer
}  // namespace xllm

#pragma once

// #ifdef WITH_ASCEND
#include "npu/npu_qwen2_decoder_layer_impl.h"
// #endif

#include "pytorch/adapter/utils/utils.h"

namespace xllm {
namespace layer {
// #ifdef WITH_ASCEND
class Qwen2DecoderLayer
    : public torch::nn::ModuleHolder<NpuQwen2DecoderLayerImpl> {
 public:
  using torch::nn::ModuleHolder<NpuQwen2DecoderLayerImpl>::ModuleHolder;
  using Impl __attribute__((__unused__)) = NpuQwen2DecoderLayerImpl;

  Qwen2DecoderLayer(const Context& context)
      : ModuleHolder(std::make_shared<NpuQwen2DecoderLayerImpl>(context)) {}
};
// #endif

}  // namespace layer
}  // namespace xllm

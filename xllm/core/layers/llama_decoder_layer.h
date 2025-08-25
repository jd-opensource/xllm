#pragma once

// #ifdef WITH_ASCEND
#include "npu/npu_llama_decoder_layer_impl.h"
// #endif

#include "pytorch/adapter/utils/utils.h"

namespace xllm {
namespace layer {
// #ifdef WITH_ASCEND
class LlamaDecoderLayer
    : public torch::nn::ModuleHolder<NpuLlamaDecoderLayerImpl> {
 public:
  using torch::nn::ModuleHolder<NpuLlamaDecoderLayerImpl>::ModuleHolder;
  using Impl __attribute__((__unused__)) = NpuLlamaDecoderLayerImpl;

  LlamaDecoderLayer(const Context& context)
      : ModuleHolder(std::make_shared<NpuLlamaDecoderLayerImpl>(context)) {}
};
// #endif

}  // namespace layer
}  // namespace xllm

#pragma once

// #ifdef WITH_ASCEND
#include "npu/npu_qwen3_decoder_layer_impl.h"
// #endif

#include "pytorch/adapter/utils/utils.h"

namespace xllm {
namespace layer {
// #ifdef WITH_ASCEND
class Qwen3DecoderLayer
    : public torch::nn::ModuleHolder<NpuQwen3DecoderLayerImpl> {
 public:
  using torch::nn::ModuleHolder<NpuQwen3DecoderLayerImpl>::ModuleHolder;
  using Impl __attribute__((__unused__)) = NpuQwen3DecoderLayerImpl;

  Qwen3DecoderLayer(const Context& context)
      : ModuleHolder(std::make_shared<NpuQwen3DecoderLayerImpl>(context)) {}
};
// #endif

}  // namespace layer
}  // namespace xllm

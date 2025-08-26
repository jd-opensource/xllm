#pragma once

// #ifdef WITH_ASCEND
#include "npu/npu_qwen3_moe_decoder_layer_impl.h"
// #endif

#include "pytorch/adapter/utils/utils.h"

namespace xllm {
namespace layer {
// #ifdef WITH_ASCEND
class Qwen3MoeDecoderLayer
    : public torch::nn::ModuleHolder<NpuQwen3MoeDecoderLayerImpl> {
 public:
  using torch::nn::ModuleHolder<NpuQwen3MoeDecoderLayerImpl>::ModuleHolder;
  using Impl __attribute__((__unused__)) = NpuQwen3MoeDecoderLayerImpl;

  Qwen3MoeDecoderLayer(const Context& context, int32_t layer_id)
      : Qwen3MoeDecoderLayer(
            std::make_shared<NpuQwen3MoeDecoderLayerImpl>(context, layer_id)) {}
};
// #endif

}  // namespace layer
}  // namespace xllm

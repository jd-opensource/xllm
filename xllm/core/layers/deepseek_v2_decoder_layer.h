#pragma once

// #ifdef WITH_ASCEND
#include "npu/npu_deepseek_v2_decoder_layer_impl.h"
// #endif

#include "pytorch/adapter/utils/utils.h"

namespace xllm {
namespace layer {
// #ifdef WITH_ASCEND
class DeepseekV2DecoderLayer
    : public torch::nn::ModuleHolder<NpuDeepseekV2DecoderLayerImpl> {
 public:
  using torch::nn::ModuleHolder<NpuDeepseekV2DecoderLayerImpl>::ModuleHolder;
  using Impl __attribute__((__unused__)) = NpuDeepseekV2DecoderLayerImpl;

  DeepseekV2DecoderLayer(const Context& context,
                         const int32_t layer_id,
                         const float sm_scale)
      : ModuleHolder(
            std::make_shared<NpuDeepseekV2DecoderLayerImpl>(context,
                                                            layer_id,
                                                            sm_scale)) {}
};
// #endif

}  // namespace layer
}  // namespace xllm

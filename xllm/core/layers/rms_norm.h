#pragma once
#ifdef TORCH_HIGHER_THAN_PTA6
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#else
#include <torch_npu/csrc/aten/NPUNativeFunctions.h>
#include <torch_npu/csrc/framework/utils/OpPreparation.h>
#endif

#include <torch_npu/csrc/libs/init_npu.h>

#include <functional>

#include "atb/atb_infer.h"
#include "atb_speed/base/hosttensor_binder.h"
#include "atb_speed/base/model.h"
#include "atb_speed/log.h"
#include "atb_speed/utils/model_factory.h"
#include "framework/context.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_input_params.h"
#include "framework/state_dict/state_dict.h"
#include "nlohmann/json.hpp"
#include "npu/npu_rms_norm_impl.h"
#include "pytorch/adapter/utils/utils.h"

namespace xllm::layer {

class RmsNorm : public torch::nn::ModuleHolder<NpuRmsNormImpl> {
 public:
  using torch::nn::ModuleHolder<NpuRmsNormImpl>::ModuleHolder;
  using Impl __attribute__((__unused__)) = NpuRmsNormImpl;

  RmsNorm(const Context& context)
      : ModuleHolder(std::make_shared<NpuRmsNormImpl>(context)) {}
};

}  // namespace xllm::layer

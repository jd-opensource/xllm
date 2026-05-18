/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "core/framework/config/kernel_config.h"

#include "core/common/global_flags.h"

#if defined(USE_NPU)
DEFINE_bool(enable_customize_mla_kernel, false, "enable customize mla kernel");

DEFINE_string(npu_kernel_backend,
              "AUTO",
              "NPU kernel backend. Supported options: AUTO, ATB, TORCH.");

DEFINE_bool(enable_intralayer_addnorm,
            false,
            "enable fused intralayer addnorm ops.");
#endif

namespace xllm {

KernelConfig KernelConfig::from_flags() {
  KernelConfig config;
#if defined(USE_NPU)
  config.enable_customize_mla_kernel(FLAGS_enable_customize_mla_kernel)
      .npu_kernel_backend(FLAGS_npu_kernel_backend)
      .enable_intralayer_addnorm(FLAGS_enable_intralayer_addnorm);
#endif
  return config;
}

KernelConfig& KernelConfig::get_instance() {
  static KernelConfig config = KernelConfig::from_flags();
  return config;
}

void KernelConfig::reload_from_flags() {
  KernelConfig::get_instance() = KernelConfig::from_flags();
}

}  // namespace xllm

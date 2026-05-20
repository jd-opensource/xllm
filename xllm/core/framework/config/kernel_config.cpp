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
#include "core/framework/config/config_json_utils.h"

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

void KernelConfig::from_flags() {
#if defined(USE_NPU)
  enable_customize_mla_kernel(FLAGS_enable_customize_mla_kernel)
      .npu_kernel_backend(FLAGS_npu_kernel_backend)
      .enable_intralayer_addnorm(FLAGS_enable_intralayer_addnorm);
#endif
}

void KernelConfig::from_json(const JsonReader& json) {
#if defined(USE_NPU)
  enable_customize_mla_kernel(
      json.value_or<bool>("enable_customize_mla_kernel",
                          enable_customize_mla_kernel()))
      .npu_kernel_backend(json.value_or<std::string>("npu_kernel_backend",
                                                     npu_kernel_backend()))
      .enable_intralayer_addnorm(json.value_or<bool>(
          "enable_intralayer_addnorm", enable_intralayer_addnorm()));
#endif
}

KernelConfig& KernelConfig::get_instance() {
  static KernelConfig config;
  return config;
}

void KernelConfig::initialize() {
  from_flags();
  if (!FLAGS_config_json_file.empty()) {
    from_json(config::load_json_file(FLAGS_config_json_file));
  }
}

}  // namespace xllm

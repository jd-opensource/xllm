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

#include "utils.h"

#include <musa_runtime.h>

#include <cstdlib>

#include "core/util/env_var.h"

namespace {
const std::unordered_map<torch::ScalarType, std::string_view>
    filename_safe_dtype_map = {
        {torch::kFloat16, "f16"},
        {torch::kBFloat16, "bf16"},
        {torch::kFloat8_e4m3fn, "e4m3"},
        {torch::kFloat8_e5m2, "e5m2"},
        {torch::kInt8, "i8"},
        {torch::kUInt8, "u8"},
        {torch::kInt32, "i32"},
        {torch::kUInt32, "u32"},
        {torch::kInt64, "i64"},
        {torch::kUInt64, "u64"},
};
}  // namespace

namespace xllm::kernel::musa {

// Whether to enable Programmatic Dependent Launch (PDL). See
// https://docs.nvidia.com/musa/musa-c-programming-guide/#programmatic-dependent-launch-and-synchronization
// Only supported for >= sm90, and currently only for FA2, CUDA core, and
// trtllm-gen decode.
bool support_pdl() { return false; }

std::string path_to_uri_so_lib(const std::string& uri) {
  return util::get_string_env("FLASHINFER_OPS_PATH") + "/" + uri + "/" + uri +
         ".so";
}

}  // namespace xllm::kernel::musa

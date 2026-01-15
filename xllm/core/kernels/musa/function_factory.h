/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

#pragma once

#include <ATen/DynamicLibrary.h>

#include "core/util/env_var.h"
#include "utils.h"

namespace xllm::kernel::musa {

namespace {
using ACT_AND_MUL_FUNC_TYPE =
    torch::TypedOperatorHandle<void(torch::Tensor&, torch::Tensor&, bool)>;
using MATE_FUNC_TYPE = torch::TypedOperatorHandle<
    std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>(
        at::Tensor,
        at::Tensor,
        at::Tensor,
        std::optional<at::Tensor>,
        std::optional<at::Tensor>,
        std::optional<at::Tensor>,
        std::optional<at::Tensor>,
        std::optional<at::Tensor>,
        std::optional<at::Tensor>,
        std::optional<at::Tensor>,
        std::optional<at::Tensor>,
        std::optional<at::Tensor>,
        std::optional<long>,
        std::optional<long>,
        std::optional<at::Tensor>,
        std::optional<at::Tensor>,
        std::optional<at::Tensor>,
        std::optional<at::Tensor>,
        std::optional<at::Tensor>,
        std::optional<at::Tensor>,
        std::optional<at::Tensor>,
        std::optional<at::Tensor>,
        std::optional<at::Tensor>,
        std::optional<double>,
        bool,
        long,
        long,
        long,
        double,
        bool,
        std::optional<at::Tensor>,
        long,
        std::optional<bool>,
        long)>;
using RMSNORM_FUNC_TYPE = torch::TypedOperatorHandle<
    void(torch::Tensor&, torch::Tensor&, torch::Tensor&, double, bool)>;
using ROPE_FUNC_TYPE = torch::TypedOperatorHandle<void(torch::Tensor,
                                                       torch::Tensor,
                                                       torch::Tensor,
                                                       torch::Tensor,
                                                       torch::Tensor,
                                                       torch::Tensor,
                                                       bool)>;
}  // namespace

class FunctionFactory {
 public:
  static FunctionFactory& get_instance() {
    static FunctionFactory instance;
    return instance;
  }

  ACT_AND_MUL_FUNC_TYPE act_and_mul(const std::string& uri) {
    static std::optional<ACT_AND_MUL_FUNC_TYPE> f;
    static std::unique_ptr<torch::DynamicLibrary> lib;
    if (f.has_value()) {
      return f.value();
    }

    static std::once_flag flag;
    std::call_once(flag, [&uri]() {
      lib = std::make_unique<torch::DynamicLibrary>(
          path_to_uri_so_lib(uri).c_str(), nullptr, true);
      std::string schema_name = uri + "::" + uri;
      f = torch::Dispatcher::singleton()
              .findSchemaOrThrow(schema_name.c_str(), "")
              .typed<void(torch::Tensor&, torch::Tensor&, bool)>();
    });

    return f.value();
  }

  MATE_FUNC_TYPE mate_func() {
    std::string uri = util::get_string_env("MATE_OPS_PATH");
    static std::optional<MATE_FUNC_TYPE> f;
    static std::unique_ptr<torch::DynamicLibrary> lib;
    if (f.has_value()) {
      return f.value();
    }

    static std::once_flag flag;
    std::call_once(flag, [&uri]() {
      lib = std::make_unique<torch::DynamicLibrary>(uri.c_str(), nullptr, true);
      std::string plan_schema_name = "mate::fmha_fwd";
      f = torch::Dispatcher::singleton()
              .findSchemaOrThrow(plan_schema_name.c_str(), "")
              .typed<std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>(
                  at::Tensor,
                  at::Tensor,
                  at::Tensor,
                  std::optional<at::Tensor>,
                  std::optional<at::Tensor>,
                  std::optional<at::Tensor>,
                  std::optional<at::Tensor>,
                  std::optional<at::Tensor>,
                  std::optional<at::Tensor>,
                  std::optional<at::Tensor>,
                  std::optional<at::Tensor>,
                  std::optional<at::Tensor>,
                  std::optional<long>,
                  std::optional<long>,
                  std::optional<at::Tensor>,
                  std::optional<at::Tensor>,
                  std::optional<at::Tensor>,
                  std::optional<at::Tensor>,
                  std::optional<at::Tensor>,
                  std::optional<at::Tensor>,
                  std::optional<at::Tensor>,
                  std::optional<at::Tensor>,
                  std::optional<at::Tensor>,
                  std::optional<double>,
                  bool,
                  long,
                  long,
                  long,
                  double,
                  bool,
                  std::optional<at::Tensor>,
                  long,
                  std::optional<bool>,
                  long)>();
    });

    return f.value();
  }

  RMSNORM_FUNC_TYPE rmsnorm_func(const std::string& uri) {
    static std::optional<RMSNORM_FUNC_TYPE> f;
    static std::unique_ptr<torch::DynamicLibrary> lib;
    if (f.has_value()) {
      return f.value();
    }

    static std::once_flag flag;
    std::call_once(flag, [&uri]() {
      lib = std::make_unique<torch::DynamicLibrary>(
          path_to_uri_so_lib(uri).c_str(), nullptr, true);
      std::string schema_name = "norm::rmsnorm";
      f = torch::Dispatcher::singleton()
              .findSchemaOrThrow(schema_name.c_str(), "")
              .typed<void(torch::Tensor&,
                          torch::Tensor&,
                          torch::Tensor&,
                          double,
                          bool)>();
    });

    return f.value();
  }

  ROPE_FUNC_TYPE rope_func(const std::string& uri) {
    static std::optional<ROPE_FUNC_TYPE> f;
    static std::unique_ptr<torch::DynamicLibrary> lib;
    if (f.has_value()) {
      return f.value();
    }

    static std::once_flag flag;
    std::call_once(flag, [&uri]() {
      lib = std::make_unique<torch::DynamicLibrary>(
          path_to_uri_so_lib(uri).c_str(), nullptr, true);
      std::string schema_name = "rope::apply_rope_pos_ids_cos_sin_cache";
      f = torch::Dispatcher::singleton()
              .findSchemaOrThrow(schema_name.c_str(), "")
              .typed<void(torch::Tensor,
                          torch::Tensor,
                          torch::Tensor,
                          torch::Tensor,
                          torch::Tensor,
                          torch::Tensor,
                          bool)>();
    });

    return f.value();
  }
};

}  // namespace xllm::kernel::musa

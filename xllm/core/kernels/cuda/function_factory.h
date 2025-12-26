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

#include "utils.h"

namespace xllm::kernel::cuda {

namespace {
using ACT_AND_MUL_FUNC_TYPE =
    torch::TypedOperatorHandle<void(torch::Tensor&, torch::Tensor&, bool)>;
using DECODE_PLAN_FUNC_TYPE =
    torch::TypedOperatorHandle<torch::Tensor(torch::Tensor,
                                             torch::Tensor,
                                             torch::Tensor,
                                             torch::Tensor,
                                             int64_t,
                                             int64_t,
                                             int64_t,
                                             int64_t,
                                             bool,
                                             int64_t,
                                             double,
                                             int64_t,
                                             int64_t,
                                             torch::Tensor,
                                             torch::Tensor)>;
using DECODE_RUN_FUNC_TYPE =
    torch::TypedOperatorHandle<void(torch::Tensor,
                                    torch::Tensor,
                                    torch::Tensor,
                                    torch::Tensor,
                                    torch::Tensor,
                                    torch::Tensor,
                                    torch::Tensor,
                                    torch::Tensor,
                                    torch::Tensor,
                                    torch::Tensor,
                                    std::optional<torch::Tensor>,
                                    int64_t,
                                    int64_t,
                                    bool,
                                    std::optional<torch::Tensor>,
                                    double,
                                    double,
                                    double,
                                    double)>;
using FA2_PREFILL_PLAN_FUNC_TYPE =
    torch::TypedOperatorHandle<torch::Tensor(torch::Tensor,
                                             torch::Tensor,
                                             torch::Tensor,
                                             torch::Tensor,
                                             torch::Tensor,
                                             torch::Tensor,
                                             int64_t,
                                             int64_t,
                                             int64_t,
                                             int64_t,
                                             int64_t,
                                             bool,
                                             int64_t,
                                             int64_t,
                                             bool)>;
using FA3_PREFILL_PLAN_FUNC_TYPE =
    torch::TypedOperatorHandle<torch::Tensor(torch::Tensor,
                                             torch::Tensor,
                                             torch::Tensor,
                                             torch::Tensor,
                                             torch::Tensor,
                                             torch::Tensor,
                                             int64_t,
                                             int64_t,
                                             int64_t,
                                             int64_t,
                                             int64_t,
                                             bool,
                                             int64_t,
                                             int64_t,
                                             bool)>;
using FA2_PREFILL_RAGGED_RUN_FUNC_TYPE =
    torch::TypedOperatorHandle<void(torch::Tensor,
                                    torch::Tensor,
                                    torch::Tensor,
                                    torch::Tensor,
                                    torch::Tensor,
                                    torch::Tensor,
                                    torch::Tensor,
                                    torch::Tensor,
                                    torch::Tensor,
                                    std::optional<torch::Tensor>,
                                    int64_t,
                                    int64_t,
                                    int64_t,
                                    bool,
                                    std::optional<torch::Tensor>,
                                    std::optional<torch::Tensor>,
                                    std::optional<torch::Tensor>,
                                    std::optional<torch::Tensor>,
                                    std::optional<torch::Tensor>,
                                    std::optional<torch::Tensor>,
                                    double,
                                    double,
                                    double,
                                    double,
                                    int64_t)>;
using FA3_PREFILL_RAGGED_RUN_FUNC_TYPE =
    torch::TypedOperatorHandle<void(torch::Tensor,
                                    torch::Tensor,
                                    torch::Tensor,
                                    torch::Tensor,
                                    torch::Tensor,
                                    torch::Tensor,
                                    torch::Tensor,
                                    torch::Tensor,
                                    torch::Tensor,
                                    std::optional<torch::Tensor>,
                                    int64_t,
                                    int64_t,
                                    int64_t,
                                    bool,
                                    std::optional<torch::Tensor>,
                                    std::optional<torch::Tensor>,
                                    std::optional<torch::Tensor>,
                                    double,
                                    double,
                                    int64_t)>;
using FA2_PREFILL_PAGED_RUN_FUNC_TYPE =
    torch::TypedOperatorHandle<void(torch::Tensor,
                                    torch::Tensor,
                                    torch::Tensor,
                                    torch::Tensor,
                                    torch::Tensor,
                                    torch::Tensor,
                                    torch::Tensor,
                                    torch::Tensor,
                                    torch::Tensor,
                                    torch::Tensor,
                                    torch::Tensor,
                                    std::optional<torch::Tensor>,
                                    int64_t,
                                    int64_t,
                                    int64_t,
                                    bool,
                                    std::optional<torch::Tensor>,
                                    std::optional<torch::Tensor>,
                                    std::optional<torch::Tensor>,
                                    std::optional<torch::Tensor>,
                                    std::optional<torch::Tensor>,
                                    std::optional<torch::Tensor>,
                                    double,
                                    double,
                                    double,
                                    double,
                                    int64_t)>;
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

  DECODE_PLAN_FUNC_TYPE decode_plan_func(const std::string& uri) {
    static std::optional<DECODE_PLAN_FUNC_TYPE> f;
    static std::unique_ptr<torch::DynamicLibrary> lib;
    if (f.has_value()) {
      return f.value();
    }

    static std::once_flag flag;
    std::call_once(flag, [&uri]() {
      lib = std::make_unique<torch::DynamicLibrary>(
          path_to_uri_so_lib(uri).c_str(), nullptr, true);
      std::string plan_schema_name = uri + "::plan";
      f = torch::Dispatcher::singleton()
              .findSchemaOrThrow(plan_schema_name.c_str(), "")
              .typed<torch::Tensor(torch::Tensor,
                                   torch::Tensor,
                                   torch::Tensor,
                                   torch::Tensor,
                                   int64_t,
                                   int64_t,
                                   int64_t,
                                   int64_t,
                                   bool,
                                   int64_t,
                                   double,
                                   int64_t,
                                   int64_t,
                                   torch::Tensor,
                                   torch::Tensor)>();
    });

    return f.value();
  }

  DECODE_RUN_FUNC_TYPE decode_run_func(const std::string& uri) {
    static std::optional<DECODE_RUN_FUNC_TYPE> f;
    static std::unique_ptr<torch::DynamicLibrary> lib;
    if (f.has_value()) {
      return f.value();
    }

    static std::once_flag flag;
    std::call_once(flag, [&uri]() {
      lib = std::make_unique<torch::DynamicLibrary>(
          path_to_uri_so_lib(uri).c_str(), nullptr, true);
      std::string run_schema_name = uri + "::run";
      f = torch::Dispatcher::singleton()
              .findSchemaOrThrow(run_schema_name.c_str(), "")
              .typed<void(torch::Tensor,
                          torch::Tensor,
                          torch::Tensor,
                          torch::Tensor,
                          torch::Tensor,
                          torch::Tensor,
                          torch::Tensor,
                          torch::Tensor,
                          torch::Tensor,
                          torch::Tensor,
                          std::optional<torch::Tensor>,
                          int64_t,
                          int64_t,
                          bool,
                          std::optional<torch::Tensor>,
                          double,
                          double,
                          double,
                          double)>();
    });

    return f.value();
  }

  FA2_PREFILL_PLAN_FUNC_TYPE fa2_prefill_plan_func(const std::string& uri) {
    static std::optional<FA2_PREFILL_PLAN_FUNC_TYPE> f;
    static std::unique_ptr<torch::DynamicLibrary> lib;
    if (f.has_value()) {
      return f.value();
    }

    static std::once_flag flag;
    std::call_once(flag, [&uri]() {
      lib = std::make_unique<torch::DynamicLibrary>(
          path_to_uri_so_lib(uri).c_str(), nullptr, true);
      std::string plan_schema_name = uri + "::plan";
      f = torch::Dispatcher::singleton()
              .findSchemaOrThrow(plan_schema_name.c_str(), "")
              .typed<torch::Tensor(torch::Tensor,
                                   torch::Tensor,
                                   torch::Tensor,
                                   torch::Tensor,
                                   torch::Tensor,
                                   torch::Tensor,
                                   int64_t,
                                   int64_t,
                                   int64_t,
                                   int64_t,
                                   int64_t,
                                   bool,
                                   int64_t,
                                   int64_t,
                                   bool)>();
    });

    return f.value();
  }

  FA3_PREFILL_PLAN_FUNC_TYPE fa3_prefill_plan_func(const std::string& uri) {
    static std::optional<FA3_PREFILL_PLAN_FUNC_TYPE> f;
    static std::unique_ptr<torch::DynamicLibrary> lib;
    if (f.has_value()) {
      return f.value();
    }

    static std::once_flag flag;
    std::call_once(flag, [&uri]() {
      lib = std::make_unique<torch::DynamicLibrary>(
          path_to_uri_so_lib(uri).c_str(), nullptr, true);
      std::string plan_schema_name = uri + "::plan";
      f = torch::Dispatcher::singleton()
              .findSchemaOrThrow(plan_schema_name.c_str(), "")
              .typed<torch::Tensor(torch::Tensor,
                                   torch::Tensor,
                                   torch::Tensor,
                                   torch::Tensor,
                                   torch::Tensor,
                                   torch::Tensor,
                                   int64_t,
                                   int64_t,
                                   int64_t,
                                   int64_t,
                                   int64_t,
                                   bool,
                                   int64_t,
                                   int64_t,
                                   bool)>();
    });

    return f.value();
  }

  FA2_PREFILL_RAGGED_RUN_FUNC_TYPE fa2_prefill_ragged_run_func(
      const std::string& uri) {
    static std::optional<FA2_PREFILL_RAGGED_RUN_FUNC_TYPE> f;
    static std::unique_ptr<torch::DynamicLibrary> lib;
    if (f.has_value()) {
      return f.value();
    }

    static std::once_flag flag;
    std::call_once(flag, [&uri]() {
      lib = std::make_unique<torch::DynamicLibrary>(
          path_to_uri_so_lib(uri).c_str(), nullptr, true);
      std::string run_schema_name = uri + "::ragged_run";
      f = torch::Dispatcher::singleton()
              .findSchemaOrThrow(run_schema_name.c_str(), "")
              .typed<void(torch::Tensor,
                          torch::Tensor,
                          torch::Tensor,
                          torch::Tensor,
                          torch::Tensor,
                          torch::Tensor,
                          torch::Tensor,
                          torch::Tensor,
                          torch::Tensor,
                          std::optional<torch::Tensor>,
                          int64_t,
                          int64_t,
                          int64_t,
                          bool,
                          std::optional<torch::Tensor>,
                          std::optional<torch::Tensor>,
                          std::optional<torch::Tensor>,
                          std::optional<torch::Tensor>,
                          std::optional<torch::Tensor>,
                          std::optional<torch::Tensor>,
                          double,
                          double,
                          double,
                          double,
                          int64_t)>();
    });

    return f.value();
  }

  FA3_PREFILL_RAGGED_RUN_FUNC_TYPE fa3_prefill_ragged_run_func(
      const std::string& uri) {
    static std::optional<FA3_PREFILL_RAGGED_RUN_FUNC_TYPE> f;
    static std::unique_ptr<torch::DynamicLibrary> lib;
    if (f.has_value()) {
      return f.value();
    }

    static std::once_flag flag;
    std::call_once(flag, [&uri]() {
      lib = std::make_unique<torch::DynamicLibrary>(
          path_to_uri_so_lib(uri).c_str(), nullptr, true);
      std::string run_schema_name = uri + "::ragged_run";
      f = torch::Dispatcher::singleton()
              .findSchemaOrThrow(run_schema_name.c_str(), "")
              .typed<void(torch::Tensor,
                          torch::Tensor,
                          torch::Tensor,
                          torch::Tensor,
                          torch::Tensor,
                          torch::Tensor,
                          torch::Tensor,
                          torch::Tensor,
                          torch::Tensor,
                          std::optional<torch::Tensor>,
                          int64_t,
                          int64_t,
                          int64_t,
                          bool,
                          std::optional<torch::Tensor>,
                          std::optional<torch::Tensor>,
                          std::optional<torch::Tensor>,
                          double,
                          double,
                          int64_t)>();
    });

    return f.value();
  }

  FA2_PREFILL_PAGED_RUN_FUNC_TYPE fa2_prefill_paged_run_func(
      const std::string& uri) {
    static std::optional<FA2_PREFILL_PAGED_RUN_FUNC_TYPE> f;
    static std::unique_ptr<torch::DynamicLibrary> lib;
    if (f.has_value()) {
      return f.value();
    }

    static std::once_flag flag;
    std::call_once(flag, [&uri]() {
      lib = std::make_unique<torch::DynamicLibrary>(
          path_to_uri_so_lib(uri).c_str(), nullptr, true);
      std::string run_schema_name = uri + "::paged_run";
      f = torch::Dispatcher::singleton()
              .findSchemaOrThrow(run_schema_name.c_str(), "")
              .typed<void(torch::Tensor,
                          torch::Tensor,
                          torch::Tensor,
                          torch::Tensor,
                          torch::Tensor,
                          torch::Tensor,
                          torch::Tensor,
                          torch::Tensor,
                          torch::Tensor,
                          torch::Tensor,
                          torch::Tensor,
                          std::optional<torch::Tensor>,
                          int64_t,
                          int64_t,
                          int64_t,
                          bool,
                          std::optional<torch::Tensor>,
                          std::optional<torch::Tensor>,
                          std::optional<torch::Tensor>,
                          std::optional<torch::Tensor>,
                          std::optional<torch::Tensor>,
                          std::optional<torch::Tensor>,
                          double,
                          double,
                          double,
                          double,
                          int64_t)>();
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

}  // namespace xllm::kernel::cuda

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

#include "executor_impl_factory.h"

#include "platform/device.h"
#include "runtime/base_executor_impl.h"
#if defined(USE_NPU)
#include "runtime/acl_graph_executor_impl.h"
#elif defined(USE_MLU)
#include "runtime/mlu_graph_executor_impl.h"
#endif

namespace xllm {

ExecutorImplFactory& ExecutorImplFactory::get_instance() {
  static ExecutorImplFactory instance;
  return instance;
}

bool ExecutorImplFactory::register_creator(const std::string& name,
                                           Creator creator) {
  auto [it, inserted] = creators_.emplace(name, std::move(creator));
  return inserted;
}

std::unique_ptr<ExecutorImpl> ExecutorImplFactory::create_executor_impl(
    CausalLM* model,
    const ModelArgs& args,
    const torch::Device& device,
    const runtime::Options& options) {
  std::string backend = "base";
  if (FLAGS_enable_graph) {
    backend = Device::type_str();
    LOG(INFO) << "Creating Graph Executor for " << backend << " device";
  }

  auto it = creators_.find(backend);
  if (it == creators_.end()) {
    throw std::runtime_error("No valid graph backend found: " + backend);
  }

  return it->second(model, args, device, options);
}

}  // namespace xllm

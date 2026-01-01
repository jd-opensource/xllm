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

#include "shm_channel.h"

#include "common/global_flags.h"
#include "util/net.h"

namespace xllm {

ShmChannel::ShmChannel(int dp_group,
                       int rank,
                       bool is_driver,
                       const runtime::Options& options)
    : enable_shm_(options.enable_shm()) {
  bool is_creator;

  std::string name_prefix =
      "xllm_" + net::extract_port(options.master_node_addr().value_or(""));
  if (is_driver) {
    auto name = ForwardSharedMemoryManager::create_unique_name(
        name_prefix, dp_group, FORWARD_RAW_INPUT_TYPE, rank);
    input_shm_manager_ = std::make_unique<ForwardSharedMemoryManager>(
        name, PB_INPUT_SHM_SIZE, is_creator, FORWARD_RAW_INPUT_TYPE);
    LOG(INFO) << "Create input shared memory manager with name: " << name;
  }

  auto name = ForwardSharedMemoryManager::create_unique_name(
      name_prefix, dp_group, FORWARD_RAW_OUTPUT_TYPE, rank);
  output_shm_manager_ = std::make_unique<ForwardSharedMemoryManager>(
      name, PB_OUTPUT_SHM_SIZE, is_creator, FORWARD_RAW_OUTPUT_TYPE);
  LOG(INFO) << "Create output shared memory manager with name: " << name;
}

bool ShmChannel::execute_model_with_shm(const RawForwardInput& input,
                                        RawForwardOutput& raw_output) {
  // write to shared memory, then wait output.
  if (input_shm_manager_) {
    int use_shm_ret = input_shm_manager_->raw_input_write(input);
    if (use_shm_ret < 0) {
      // fallback
      enable_shm_ = false;
      LOG(ERROR)
          << "RemoteWorker SharedMemoryManager write failed, fallback to brpc.";
      return false;
    }
  }
  output_shm_manager_->raw_output_read(raw_output);
  return true;
}

void ShmChannel::execute_model_async(
    const RawForwardInput& input,
    folly::Promise<std::optional<RawForwardOutput>>& promise) {
  if (enable_shm_) {
    // write to shared memory, then wait output.
    RawForwardOutput raw_output;
    bool shm_success = execute_model_with_shm(input, raw_output);
    if (shm_success) {
      promise.setValue(raw_output);
      return;
    }
  }
  execute_model_with_brpc(input, promise);
}
}  // namespace xllm

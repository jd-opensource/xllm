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

#include <gflags/gflags.h>

#include <memory>

#include "common/macros.h"
#include "engine.h"
#include "framework/batch/batch.h"
#include "framework/block/block_manager_pool.h"
#include "framework/quant_args.h"
#include "framework/tokenizer/tokenizer.h"
#include "framework/tokenizer/tokenizer_args.h"
#include "worker.h"

namespace xllm {

class RecEngine : public Engine {
 public:
  // create an engine with the given devices
  RecEngine(const runtime::Options& options);

  virtual ~RecEngine() = default;

  ForwardOutput step(std::vector<Batch>& batch) override;

  const runtime::Options& options() const { return options_; }

  bool init() override;

  void update_last_step_result(std::vector<Batch>& batch) override;

  // return the active activation memory
  std::vector<int64_t> get_active_activation_memory() const override;

 private:
  bool init_model();
  Engine::KVCacheCapacity estimate_kv_cache_capacity();
  bool allocate_kv_cache(const Engine::KVCacheCapacity& kv_cache_cap);

  // Helper methods for rec-specific execution
  ForwardOutput get_model_output(const ForwardInput& model_inputs);

 private:
  // options
  runtime::Options options_;

  // dtype
  torch::ScalarType dtype_;

  // quantization args
  QuantArgs quant_args_;

  // a list of process groups, with each process group handling a single device
  std::vector<std::unique_ptr<ProcessGroup>> process_groups_;

  // a list of workers, with each worker handling a partial of model
  std::vector<std::unique_ptr<Worker>> workers_;

  // config for kv cache
  int64_t n_local_kv_heads_ = 0;
  int64_t head_dim_ = 0;
};

}  // namespace xllm
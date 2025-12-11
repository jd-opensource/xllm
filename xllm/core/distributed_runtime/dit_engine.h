/* Copyright 2025 The xLLM Authors. All Rights Reserved.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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
#include "framework/batch/dit_batch.h"
#include "framework/parallel_state/process_group.h"
#include "framework/quant_args.h"
#include "runtime/dit_worker.h"

namespace xllm {

class DiTEngine {
 public:
  DiTEngine(const runtime::Options& options);

  ~DiTEngine() = default;

  DiTForwardOutput step(std::vector<DiTBatch>& batch);

  const runtime::Options& options() const { return options_; }

  bool init();

  // return the active activation memory
  std::vector<int64_t> get_active_activation_memory() const;

 private:
  bool init_model();
  // options
  runtime::Options options_;

  // a list of process groups, with each process group handling a single device
  std::vector<std::unique_ptr<ProcessGroup>> process_groups_;

  // a list of workers, with each worker handling a partial of model
  std::vector<std::unique_ptr<DiTWorker>> workers_;
};

}  // namespace xllm
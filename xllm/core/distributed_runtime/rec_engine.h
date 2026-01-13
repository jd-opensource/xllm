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
#include "common/rec_model_utils.h"
#include "distributed_runtime/dist_manager.h"
#include "engine.h"
#include "framework/batch/batch.h"
#include "framework/block/block_manager_pool.h"
#include "framework/quant_args.h"
#include "framework/tokenizer/tokenizer.h"
#include "framework/tokenizer/tokenizer_args.h"
#include "runtime/worker.h"
#include "util/threadpool.h"

namespace xllm {

class RecEngine : public Engine {
 public:
  RecEngine(const runtime::Options& options,
            std::shared_ptr<DistManager> dist_manager = nullptr);

  virtual ~RecEngine() = default;

  ForwardOutput step(std::vector<Batch>& batch) override;

  const runtime::Options& options() const { return options_; }

  bool init() override;

  void update_last_step_result(std::vector<Batch>& batch) override;

  std::vector<int64_t> get_active_activation_memory() const override;

 private:
  // ============================================================
  // RecEnginePipeline: Abstract base class for rec engine execution
  // ============================================================
  class RecEnginePipeline {
   public:
    explicit RecEnginePipeline(RecEngine& engine) : engine_(engine) {}
    virtual ~RecEnginePipeline() = default;

    // Initialization
    virtual void setup_workers() = 0;
    virtual void process_group_test() = 0;
    virtual bool init_model_workers(const std::string& model_path) = 0;

    // KV Cache
    virtual int64_t estimate_min_available_memory() = 0;
    virtual bool allocate_kv_cache(
        const std::vector<std::vector<int64_t>>& kv_cache_shape) = 0;

    // Execution
    virtual ForwardOutput step(std::vector<Batch>& batches) = 0;

    // Misc
    virtual std::vector<int64_t> get_active_activation_memory() const = 0;
    virtual size_t num_workers() const = 0;

   protected:
    RecEngine& engine_;
  };

  // ============================================================
  // LlmRecEnginePipeline: kLlmRec (qwen2/qwen3) via DistManager
  // ============================================================
  class LlmRecEnginePipeline final : public RecEnginePipeline {
   public:
    explicit LlmRecEnginePipeline(RecEngine& engine);

    void setup_workers() override;
    void process_group_test() override;
    bool init_model_workers(const std::string& model_path) override;
    int64_t estimate_min_available_memory() override;
    bool allocate_kv_cache(
        const std::vector<std::vector<int64_t>>& kv_cache_shape) override;
    ForwardOutput step(std::vector<Batch>& batches) override;
    std::vector<int64_t> get_active_activation_memory() const override;
    size_t num_workers() const override;

   private:
    std::vector<RawForwardInput> prepare_inputs(std::vector<Batch>& batch);

    // Get max tokens from batch for dynamic step control
    size_t get_max_steps_from_batch(std::vector<Batch>& batches) const;
  };

  // ============================================================
  // OneRecEnginePipeline: kOneRec via local Worker
  // ============================================================
  class OneRecEnginePipeline final : public RecEnginePipeline {
   public:
    explicit OneRecEnginePipeline(RecEngine& engine);

    void setup_workers() override;
    void process_group_test() override;
    bool init_model_workers(const std::string& model_path) override;
    int64_t estimate_min_available_memory() override;
    bool allocate_kv_cache(
        const std::vector<std::vector<int64_t>>& kv_cache_shape) override;
    ForwardOutput step(std::vector<Batch>& batches) override;
    std::vector<int64_t> get_active_activation_memory() const override;
    size_t num_workers() const override;

   private:
    ForwardOutput get_model_output(const ForwardInput& model_inputs);
  };

  // ============================================================
  // PureDeviceEnginePipeline: kLlmRecPureDevicePipeline via local Worker
  // For multi-round pure device inference (multi-round logic in worker)
  // ============================================================
  class PureDeviceEnginePipeline final : public RecEnginePipeline {
   public:
    explicit PureDeviceEnginePipeline(RecEngine& engine);

    void setup_workers() override;
    void process_group_test() override;
    bool init_model_workers(const std::string& model_path) override;
    int64_t estimate_min_available_memory() override;
    bool allocate_kv_cache(
        const std::vector<std::vector<int64_t>>& kv_cache_shape) override;
    ForwardOutput step(std::vector<Batch>& batches) override;
    std::vector<int64_t> get_active_activation_memory() const override;
    size_t num_workers() const override;

   private:
    ForwardOutput get_model_output(const ForwardInput& model_inputs);
  };

  // Factory method to create pipeline (can access private classes)
  static std::unique_ptr<RecEnginePipeline> create_pipeline(
      RecPipelineType type,
      RecEngine& engine);

  // ============================================================
  // Private methods
  // ============================================================
  bool init_model();
  Engine::KVCacheCapacity estimate_kv_cache_capacity();
  bool allocate_kv_cache(const Engine::KVCacheCapacity& kv_cache_cap);

  // ============================================================
  // Member variables
  // ============================================================
  runtime::Options options_;
  torch::ScalarType dtype_;
  QuantArgs quant_args_;

  // Pipeline
  std::unique_ptr<RecEnginePipeline> pipeline_;
  RecModelKind rec_model_kind_ = RecModelKind::kNone;

  // Shared by both pipelines
  std::shared_ptr<DistManager> dist_manager_;
  std::unique_ptr<ThreadPool> threadpool_;

  // LlmRec specific (managed by LlmRecEnginePipeline)
  std::vector<std::shared_ptr<WorkerClient>> worker_clients_;
  size_t worker_clients_num_ = 0;
  int32_t dp_size_ = 1;
  int32_t dp_local_tp_size_ = 1;

  // OneRec specific (managed by OneRecEnginePipeline)
  std::vector<std::unique_ptr<ProcessGroup>> process_groups_;
  std::vector<std::unique_ptr<Worker>> workers_;

  // KV cache config
  int64_t n_local_kv_heads_ = 0;
  int64_t head_dim_ = 0;
};

}  // namespace xllm

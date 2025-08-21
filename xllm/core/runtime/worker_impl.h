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

#include <folly/futures/Future.h>
#include <torch/torch.h>

#include <memory>

#include "common/types.h"
#include "executor.h"
#include "forward_params.h"
#include "framework/context.h"
#if defined(USE_NPU)
#include "framework/kv_cache/hccl_kv_cache_transfer.h"
#include "framework/kv_cache/kv_cache_store.h"
#include "framework/kv_cache/llm_data_dist_transfer.h"
#endif
#include "framework/eplb/eplb_executor.h"
#include "framework/model/causal_lm.h"
#include "framework/model/embedding_lm.h"
#include "framework/model/model_input_params.h"
#include "framework/page/xtensor.h"
#include "framework/sampling/sampler.h"
#include "framework/state_dict/state_dict.h"
#include "memory"
#include "options.h"
#include "util/threadpool.h"

namespace xllm {

class WorkerImpl {
 public:
  enum Status : int8_t {
    UNINITIALIZED = 0,
    LOADED,
    READY,
  };

  WorkerImpl(const ParallelArgs& parallel_args,
             const torch::Device& device,
             const runtime::Options& options);

  virtual ~WorkerImpl();

  // initialize model, cache manager. blocking call
  virtual bool init_model(torch::ScalarType dtype,
                          const ModelArgs& args,
                          const QuantArgs& quant_args) = 0;

  virtual bool init_model(const std::string& model_weights_path);

  virtual void load_model(std::unique_ptr<ModelLoader> loader);

  virtual std::tuple<int64_t, int64_t> estimate_kv_cache_capacity();

  // allocate kv cache. blocking call
  virtual bool allocate_kv_cache(
      const std::vector<std::vector<int64_t>>& kv_cache_shape);

  virtual bool allocate_host_kv_cache(
      const std::vector<std::vector<int64_t>>& kv_cache_shape);

  virtual bool allocate_kv_cache_with_transfer(
      uint64_t kv_cache_size,
      const std::vector<std::vector<int64_t>>& kv_cache_shape);

#if defined(USE_NPU)
  virtual bool allocate_kv_cache_with_transfer(
      std::shared_ptr<KVCacheTransfer> kv_cache_transfer,
      const std::vector<std::vector<int64_t>>& kv_cache_shape);
#endif

  virtual bool allocate_continuous_kv_cache(
      const std::vector<XTensor::Options>& options);

  virtual void get_device_info(std::string& device_ip, uint16_t& port);

  virtual void get_cache_info(uint64_t& cluster_id,
                              std::string& addr,
                              int64_t& k_cache_id,
                              int64_t& v_cache_id);

  virtual bool link_cluster(const std::vector<uint64_t>& cluster_ids,
                            const std::vector<std::string>& addrs,
                            const std::vector<std::string>& device_ips,
                            const std::vector<uint16_t>& ports);

  virtual bool unlink_cluster(const std::vector<uint64_t>& cluster_ids,
                              const std::vector<std::string>& addrs,
                              const std::vector<std::string>& device_ips,
                              const std::vector<uint16_t>& ports);

  // prepare input for execution
  virtual ForwardInput prepare_inputs(Batch& batch);

  // prepare work before model execution
  virtual void prepare_work_before_execute(const ForwardInput& inputs,
                                           ForwardInput& processed_inputs);

  virtual std::optional<ForwardOutput> step(const ForwardInput& inputs) = 0;

  virtual void process_group_test();

  virtual ForwardInput update_input_by_last_step_output(ForwardInput& inputs);

  // initialize model, cache manager. async call
  virtual folly::SemiFuture<bool> init_model_async(
      const std::string& model_weights_path);

  virtual folly::SemiFuture<std::tuple<int64_t, int64_t>>
  estimate_kv_cache_capacity_async();

  // initialize kv cache. async call
  virtual folly::SemiFuture<bool> allocate_kv_cache_async(
      const std::vector<std::vector<int64_t>>& kv_cache_shape);

  virtual folly::SemiFuture<bool> allocate_kv_cache_with_transfer_async(
      uint64_t kv_cache_size,
      const std::vector<std::vector<int64_t>>& kv_cache_shape);

  virtual folly::SemiFuture<bool> allocate_continuous_kv_cache_async(
      const std::vector<XTensor::Options>& options);

  virtual folly::SemiFuture<bool> pull_kv_blocks_async(
      uint64_t src_cluster_id,
      const std::string& src_addr,
      int64_t src_k_cache_id,
      int64_t src_v_cache_id,
      const std::vector<uint64_t>& src_blocks,
      const std::vector<uint64_t>& dst_blocks);

  virtual folly::SemiFuture<uint32_t> load_kv_blocks_from_store_async(
      const std::vector<CacheBlockInfo>& cache_block_info);

  virtual uint32_t offload_kv_blocks_to_store(
      const std::vector<CacheBlockInfo>& cache_block_info);

  virtual folly::SemiFuture<uint32_t> offload_kv_blocks_to_store_async(
      const std::vector<CacheBlockInfo>& cache_block_info);

  // Run the model on the given input. async call
  // the future returns a successfull status with no meaningful value
  virtual folly::SemiFuture<std::optional<ForwardOutput>> step_async(
      const ForwardInput& inputs);

  virtual folly::SemiFuture<folly::Unit> process_group_test_async();

  const torch::Device& device() const { return device_; }

  torch::ScalarType dtype() const { return dtype_; }

  int32_t hidden_size() const {
    return context_.get_model_args().hidden_size();
  }

  bool enable_schedule_overlap() const {
    return options_.enable_schedule_overlap_;
  }

  virtual ForwardOutput get_last_step_result();

  bool is_driver() const { return driver_ || dp_driver_; }

  int64_t get_active_activation_memory();

  Status get_status() const { return status_; }

  folly::SemiFuture<bool> copy_out_blocks_async(ModelInputParams& input_params);

 private:
  void update_last_step_output(const std::optional<ForwardOutput>& output);

 protected:
  // runtime options
  runtime::Options options_;

  // whether the worker is a driver, who takes care of the sampling
  bool driver_ = false;
  bool dp_driver_ = false;

  // working thread
  // make sure only 1 thread in the pool
  // if enable_schedule_overlap, two step tasks might be dispatched to
  // the task queue, step need to be executed one-by-one
  ThreadPool threadpool_;

  // general working thread
  // do some overlap work with model execute
  ThreadPool general_threadpool_;

  // dtype of the model
  torch::ScalarType dtype_;

  // device to run the model on
  torch::Device device_;

  // model context, includes model args, parallel args and date type etc.
  mutable Context context_;

  // kv caches
  std::vector<xllm::KVCache> kv_caches_;
  std::vector<xllm::KVCache> host_kv_caches_;

  // causal LM model
  std::unique_ptr<CausalLM> model_;

  std::unique_ptr<Executor> model_executor_;

  std::unique_ptr<Sampler> sampler_;

  std::unique_ptr<EplbExecutor> eplb_executor_;

  // params for enable_schedule_overlap case
  // an output to store the result of last step
  ForwardOutput last_step_output_;
  bool last_step_output_valid_ = false;
  std::mutex mtx_;
  std::condition_variable cv_;
  bool is_recorded_ = false;

  InstanceRole instance_role_ = InstanceRole::DEFAULT;

#if defined(USE_NPU)
  std::shared_ptr<KVCacheTransfer> kv_cache_transfer_;
#endif

  std::shared_ptr<KVCacheStore> kv_cache_store_;

  // a walkaround to avoid compilation conflict involved by
  // c10_npu::NPUStream related files.
#if defined(USE_NPU)
  struct NPUStreamHelper;
  std::unique_ptr<NPUStreamHelper> npu_stream_helper_;
  std::unique_ptr<NPUStreamHelper> extra_stream_helper_;
#endif

  bool is_spec_draft_ = false;

  Status status_ = Status::UNINITIALIZED;

  torch::Tensor expert_load_data_;
};

}  // namespace xllm

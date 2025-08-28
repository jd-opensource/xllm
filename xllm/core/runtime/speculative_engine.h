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

#include "common/macros.h"
#include "framework/batch/batch.h"
#include "framework/block/block_manager_pool.h"
#include "framework/model/model_args.h"
#include "framework/tokenizer/tokenizer.h"
#include "framework/tokenizer/tokenizer_args.h"
#include "runtime/engine.h"
#include "runtime/llm_engine.h"

namespace xllm {

class SpeculativeEngine : public Engine {
 public:
  // create an engine with the given devices
  SpeculativeEngine(const runtime::Options& options);

  virtual ~SpeculativeEngine() = default;

  bool init() override;

  // step the engine forward
  ForwardOutput step(std::vector<Batch>& batch) override;

  const Tokenizer* tokenizer() const override { return engine_->tokenizer(); }

  BlockManagerPool* block_manager_pool() const override {
    return engine_->block_manager_pool();
  }

  const ModelArgs& model_args() const override { return model_args_; }

  const TokenizerArgs& tokenizer_args() const override {
    return engine_->tokenizer_args();
  }

  void update_last_step_result(std::vector<Batch>& batch) override;

  // return the active activation memory
  std::vector<int64_t> get_active_activation_memory() const override;

  // P/D
  bool pull_kv_blocks(const int32_t src_dp_size,
                      const int32_t src_dp_rank,
                      const std::vector<uint64_t>& src_cluster_ids,
                      const std::vector<std::string>& src_addrs,
                      const std::vector<int64_t>& src_k_cache_ids,
                      const std::vector<int64_t>& src_v_cache_ids,
                      const std::vector<uint64_t>& src_blocks,
                      const int32_t dst_dp_rank,
                      const std::vector<uint64_t>& dst_blocks) override;

  void get_device_info(std::vector<std::string>& device_ips,
                       std::vector<uint16_t>& ports) override;

  void get_cache_info(std::vector<uint64_t>& cluster_ids,
                      std::vector<std::string>& addrs,
                      std::vector<int64_t>& k_cache_ids,
                      std::vector<int64_t>& v_cache_ids) override;

  bool link_cluster(const std::vector<uint64_t>& cluster_ids,
                    const std::vector<std::string>& addrs,
                    const std::vector<std::string>& device_ips,
                    const std::vector<uint16_t>& ports,
                    const int32_t src_dp_size) override;

  bool unlink_cluster(const std::vector<uint64_t>& cluster_ids,
                      const std::vector<std::string>& addrs,
                      const std::vector<std::string>& device_ips,
                      const std::vector<uint16_t>& ports,
                      const int32_t dp_size) override;

 private:
  bool init_model();

  bool allocate_kv_cache();

  int64_t calculate_kv_cache(int64_t cache_size_in_bytes,
                             int64_t target_slot_size,
                             int64_t draft_slot_size) const;

  // dtype
  torch::ScalarType dtype_;

  // options
  const runtime::Options options_;

  // engine
  std::unique_ptr<LLMEngine> engine_;

  // draft engine
  std::unique_ptr<LLMEngine> draft_engine_;

  // whether target and draft engine are sharing the same device
  bool share_device_ = false;

  ModelArgs model_args_;

  std::shared_ptr<DistManager> dist_manager_ = nullptr;
};

}  // namespace xllm

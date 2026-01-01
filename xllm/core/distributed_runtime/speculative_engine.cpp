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

#include "speculative_engine.h"

#include <gflags/gflags_declare.h>
#include <glog/logging.h>

#include <memory>

#include "common/metrics.h"
#include "llm_engine.h"
#include "runtime/forward_params.h"
#include "util/timer.h"
#include "util/utils.h"

namespace xllm {

SpeculativeEngine::SpeculativeEngine(const runtime::Options& options)
    : options_(options) {
  CHECK_GT(options.num_speculative_tokens(), 0)
      << "speculative tokens should not be zero";

  runtime::Options dist_options = options;
  dist_options.num_decoding_tokens(options.num_speculative_tokens() + 1);
  dist_options.enable_speculative_decode(true);
  dist_manager_ = std::make_shared<DistManager>(dist_options);

  runtime::Options engine_options = options_;
  engine_options.num_decoding_tokens(options.num_speculative_tokens() + 1);
  engine_ = std::make_unique<LLMEngine>(engine_options, dist_manager_);

  // draft engine
  engine_options.model_path(options_.draft_model_path().value_or(""))
      .devices(options.draft_devices())
      .num_decoding_tokens(1)
      .is_draft_engine(true);
  draft_engine_ = std::make_unique<LLMEngine>(engine_options, dist_manager_);

  // check if llm and ssm are using the same device
  for (const auto& target : options.devices()) {
    for (const auto& draft : options.draft_devices()) {
      if (target == draft) {
        share_device_ = true;
        break;
      } else {
        LOG(FATAL) << "Current only support target and draft engine using the "
                      "same devices";
      }
    }
  }
}

bool SpeculativeEngine::init() {
  if (!init_model()) {
    return false;
  }

  if (!allocate_kv_cache()) {
    return false;
  }

  return true;
}

bool SpeculativeEngine::init_model() {
  if (!engine_->init_model()) {
    return false;
  }
  if (!draft_engine_->init_model()) {
    return false;
  }

  // check if the tokenizers are compatible
  const auto* draft_tokenizer = draft_engine_->tokenizer();
  const auto* target_tokenizer = engine_->tokenizer();
  if (draft_tokenizer->vocab_size() != target_tokenizer->vocab_size()) {
    LOG(ERROR) << "draft and target tokenizers have different vocab sizes, "
                  "draft vocab_size: "
               << draft_tokenizer->vocab_size()
               << ", target vocab_size: " << target_tokenizer->vocab_size();
    return false;
  }

  const std::string test_text = "hello from xllm!";
  std::vector<int32_t> draft_token_ids;
  std::vector<int32_t> target_token_ids;
  if (!draft_tokenizer->encode(test_text, &draft_token_ids) ||
      !target_tokenizer->encode(test_text, &target_token_ids)) {
    if (draft_token_ids != target_token_ids) {
      LOG(ERROR) << "draft and target tokenizers are not compatible";
      return false;
    }
  }

  // check if the max context length are the same
  model_args_ = engine_->model_args();
  const auto& draft_model_args = draft_engine_->model_args();
  if (model_args_.max_position_embeddings() !=
      draft_model_args.max_position_embeddings()) {
    LOG(WARNING) << "draft and target models have different max context "
                    "lengths, draft max_position_embeddings: "
                 << draft_model_args.max_position_embeddings()
                 << ", target max_position_embeddings: "
                 << model_args_.max_position_embeddings()
                 << ", using the minimum between them";
    model_args_.max_position_embeddings() =
        std::min(model_args_.max_position_embeddings(),
                 draft_model_args.max_position_embeddings());
  }
  dtype_ = util::parse_dtype(model_args_.dtype(), options_.devices()[0]);
  return true;
}

bool SpeculativeEngine::allocate_kv_cache() {
  Engine::KVCacheCapacity target_kv_cache_cap =
      engine_->estimate_kv_cache_capacity();
  Engine::KVCacheCapacity draft_kv_cache_cap =
      draft_engine_->estimate_kv_cache_capacity();
  const int64_t kv_cache_size =
      std::min(target_kv_cache_cap.cache_size_in_bytes,
               draft_kv_cache_cap.cache_size_in_bytes);

  int64_t n_blocks = 0;
  // check if llm and ssm are using same device
  if (share_device_) {
    // on the same device, use the smaller kv cache size
    n_blocks = calculate_kv_cache(
        kv_cache_size,
        target_kv_cache_cap.n_layers * target_kv_cache_cap.slot_size,
        draft_kv_cache_cap.n_layers * draft_kv_cache_cap.slot_size);
  } else {
    // on different devices, use the smaller number of blocks
    n_blocks =
        std::min(target_kv_cache_cap.n_blocks, draft_kv_cache_cap.n_blocks);
  }
  CHECK_GT(n_blocks, 0) << "no memory for kv cache";

  // allocate kv cache
  target_kv_cache_cap.n_blocks = n_blocks;
  target_kv_cache_cap.cache_size_in_bytes = kv_cache_size;
  draft_kv_cache_cap.n_blocks = n_blocks;
  draft_kv_cache_cap.cache_size_in_bytes = kv_cache_size;
  return engine_->allocate_kv_cache(target_kv_cache_cap) &&
         draft_engine_->allocate_kv_cache(draft_kv_cache_cap);
}

// TODO: support dp batches later
ForwardOutput SpeculativeEngine::step(std::vector<Batch>& batches) {
  return engine_->step(batches);
}

int64_t SpeculativeEngine::calculate_kv_cache(int64_t cache_size_in_bytes,
                                              int64_t target_size,
                                              int64_t draft_size) const {
  CHECK_GT(cache_size_in_bytes, 0) << "no memory for kv cache";
  const int32_t block_size = options_.block_size();

  // compute the number of blocks
  const int64_t block_size_in_bytes = block_size * (target_size + draft_size);
  return cache_size_in_bytes / block_size_in_bytes;
}

void SpeculativeEngine::update_last_step_result(std::vector<Batch>& batch) {
  engine_->update_last_step_result(batch);
}

std::vector<int64_t> SpeculativeEngine::get_active_activation_memory() const {
  return engine_->get_active_activation_memory();
}

bool SpeculativeEngine::pull_kv_blocks(
    const int32_t src_dp_size,
    const int32_t src_dp_rank,
    const std::vector<uint64_t>& src_cluster_ids,
    const std::vector<std::string>& src_addrs,
    const std::vector<int64_t>& src_k_cache_ids,
    const std::vector<int64_t>& src_v_cache_ids,
    const std::vector<uint64_t>& src_blocks,
    const int32_t dst_dp_rank,
    const std::vector<uint64_t>& dst_blocks) {
  return engine_->pull_kv_blocks(src_dp_size,
                                 src_dp_rank,
                                 src_cluster_ids,
                                 src_addrs,
                                 src_k_cache_ids,
                                 src_v_cache_ids,
                                 src_blocks,
                                 dst_dp_rank,
                                 dst_blocks);
};

void SpeculativeEngine::get_device_info(std::vector<std::string>& device_ips,
                                        std::vector<uint16_t>& ports) {
  engine_->get_device_info(device_ips, ports);
};

void SpeculativeEngine::get_cache_info(std::vector<uint64_t>& cluster_ids,
                                       std::vector<std::string>& addrs,
                                       std::vector<int64_t>& k_cache_ids,
                                       std::vector<int64_t>& v_cache_ids) {
  engine_->get_cache_info(cluster_ids, addrs, k_cache_ids, v_cache_ids);
};

bool SpeculativeEngine::link_cluster(const std::vector<uint64_t>& cluster_ids,
                                     const std::vector<std::string>& addrs,
                                     const std::vector<std::string>& device_ips,
                                     const std::vector<uint16_t>& ports,
                                     const int32_t src_dp_size) {
  return engine_->link_cluster(
      cluster_ids, addrs, device_ips, ports, src_dp_size);
};

bool SpeculativeEngine::unlink_cluster(
    const std::vector<uint64_t>& cluster_ids,
    const std::vector<std::string>& addrs,
    const std::vector<std::string>& device_ips,
    const std::vector<uint16_t>& ports,
    const int32_t dp_size) {
  return engine_->unlink_cluster(
      cluster_ids, addrs, device_ips, ports, dp_size);
};
}  // namespace xllm

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

#include <memory>
#include <vector>

#include "common/macros.h"
#include "common/types.h"
#include "framework/block/block_manager_pool.h"
#include "framework/request/request.h"
#include "framework/request/sequence.h"
#include "runtime/engine.h"
#include "runtime/xservice_client.h"
#include "time_predictor.h"

namespace xllm {
class Engine;
class ProfileManager {
 public:
  struct Options {
    PROPERTY(bool, enable_schedule_overlap) = false;

    PROPERTY(int32_t, dp_size) = 1;
    // config for profile
    PROPERTY(bool, enable_profile_step_time) = false;

    PROPERTY(bool, enable_profile_token_budget) = false;

    PROPERTY(int32_t, profile_max_prompt_length) = 2048;

    PROPERTY(bool, enable_profile_kv_blocks) = true;

    PROPERTY(int32_t,
             max_tokens_per_batch) = std::numeric_limits<int32_t>::max();

    PROPERTY(int32_t, max_global_ttft_ms) = std::numeric_limits<int32_t>::max();

    PROPERTY(int32_t, max_global_tpot_ms) = std::numeric_limits<int32_t>::max();
  };
  ProfileManager(Engine* engine, const Options& options);

  int32_t get_token_budget();
  // for single sequence
  int32_t predict_step_time(int32_t length,
                            int32_t prefix_length = 0,
                            bool if_need_add_constant_term = true);

  int32_t predict_step_time(Sequence* sequence,
                            bool if_need_add_constant_term = true);
  // for single batch or sequences
  int32_t predict_step_time(int32_t length,
                            int32_t prefix_length,
                            int32_t batch_size);

  int32_t predict_step_time(std::vector<Sequence*>& sequences);
  // Generate a request of token_length and prefix_length, finally
  // executing and returning the inference time.
  int32_t run_request(int32_t token_length,
                      int32_t prefix_length,
                      int32_t vocab_size,
                      int32_t batch_size = 1,
                      int32_t extra_token_length = 0);

  void train_time_predictor(
      std::vector<std::tuple<int32_t, int32_t, int32_t>> time_profiling_data);

  void train_time_predictor(
      std::vector<std::pair<int32_t, int32_t>> time_profiling_data);

  TimePredictor* get_time_predictor() { return time_predictor_.get(); }

 private:
  void dump_step_time_profile_to_file(
      const std::vector<std::pair<int32_t, int32_t>>& time_profiling_data);

  void dump_step_time_profile_to_file(
      const std::vector<std::tuple<int32_t, int32_t, int32_t>>&
          time_profiling_data);

  std::shared_ptr<Request> generate_single_request(int32_t token_length,
                                                   int32_t prefix_length,
                                                   int32_t vocab_size);

  std::string generate_filename(const std::string& file_suffix);

  void profile_step_time(bool if_dump_to_file);

  void eval_sequence_latency_prediction();

  void eval_batch_latency_prediction();

  void profile_token_budget();

  bool check_if_satisfy_slo(int32_t num_tokens, int32_t tpot_slo_ms);

  int32_t binary_search_max_tokens(int32_t tpot_slo_ms,
                                   int32_t lower_bound,
                                   int32_t upper_bound);

  std::unique_ptr<TimePredictor> time_predictor_;

  const Options options_;

  Engine* engine_;

  BlockManagerPool* block_manager_pool_;

  int32_t profile_length_step_ = 256;

  int32_t profile_count_per_step_ = 3;

  int32_t profile_token_budget_ = std::numeric_limits<int32_t>::max();
};

}  // namespace xllm
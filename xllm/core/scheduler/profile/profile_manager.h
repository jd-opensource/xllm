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

    PROPERTY(int32_t, profile_max_prompt_length) = 2048;

    PROPERTY(bool, if_profile_kv_blocks) = true;
  };
  ProfileManager(Engine* engine, const Options& options);

  int32_t get_token_budget(int32_t tpot_slo_ms);

  int32_t predict_step_time(int32_t length, int32_t prefix_length);

  int32_t predict_step_time(Sequence* sequence);

  int32_t predict_step_time(std::vector<Sequence*>& sequences);
  // Generate a request of token_length and prefix_length, finally
  // executing and returning the inference time.
  int32_t run_request(int32_t token_length,
                      int32_t prefix_length,
                      int32_t vocab_size);

  void train_time_predictor(
      std::vector<std::tuple<int32_t, int32_t, int32_t>> time_profiling_data);

  void train_time_predictor(
      std::vector<std::pair<int32_t, int32_t>> time_profiling_data);

 private:
  void dump_step_time_profile_to_file(
      const std::vector<std::pair<int32_t, int32_t>>& time_profiling_data);

  void dump_step_time_profile_to_file(
      const std::vector<std::tuple<int32_t, int32_t, int32_t>>&
          time_profiling_data);

  std::string generate_filename(const std::string& file_suffix);

  void profile_step_time(bool if_dump_to_file);

  void profile_token_budget(int32_t tpot_slo_ms);

  std::unique_ptr<TimePredictor> time_predictor_;

  const Options options_;

  Engine* engine_;

  BlockManagerPool* block_manager_pool_;
};

}  // namespace xllm
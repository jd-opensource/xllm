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

#include "profile_manager.h"

#include <absl/time/time.h>
#include <glog/logging.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <random>
#include <sstream>

#include "framework/batch/batch_factory.h"
#include "framework/request/request_state.h"

namespace xllm {

ProfileManager::ProfileManager(Engine* engine, const Options& options)
    : options_(options), engine_(engine) {
  CHECK(engine_ != nullptr);
  block_manager_pool_ = engine_->block_manager_pool();
  CHECK(block_manager_pool_ != nullptr);
  time_predictor_ =
      std::make_unique<TimePredictor>(options.if_profile_kv_blocks());
  if (options.enable_profile_step_time()) {
    LOG(INFO) << "Starting profiliing step time.";
    profile_step_time(true);
  }
  // more profile here, such as token_budget profile and decode length
  // prediction.
}

std::string ProfileManager::generate_filename(const std::string& file_suffix) {
  auto now = std::chrono::system_clock::now();
  auto in_time_t = std::chrono::system_clock::to_time_t(now);

  std::stringstream ss;
  ss << std::put_time(std::localtime(&in_time_t), "%Y%m%d_%H%M%S");

  std::string filename;
  filename = ss.str() + "_" + file_suffix + ".txt";

  return filename;
}

void ProfileManager::dump_step_time_profile_to_file(
    const std::vector<std::pair<int32_t, int32_t>>& time_profiling_data) {
  std::string filename = generate_filename("profile_step_time");
  std::ofstream outfile(filename);
  if (!outfile.is_open()) {
    LOG(FATAL) << "Could not open file " << filename << " for writing.";
    return;
  }
  // write data
  for (const auto& data : time_profiling_data) {
    outfile << data.first << "," << data.second << std::endl;
  }
  outfile.close();
  LOG(INFO) << "Profile data saved to: " << filename;
}

void ProfileManager::dump_step_time_profile_to_file(
    const std::vector<std::tuple<int32_t, int32_t, int32_t>>&
        time_profiling_data) {
  std::string filename = generate_filename("profile_step_time");
  std::ofstream outfile(filename);
  if (!outfile.is_open()) {
    LOG(FATAL) << "Could not open file " << filename << " for writing.";
    return;
  }
  // write data
  for (const auto& data : time_profiling_data) {
    outfile << std::get<0>(data) << "," << std::get<1>(data) << ","
            << std::get<2>(data) << std::endl;
  }
  outfile.close();
  LOG(INFO) << "Profile data saved to: " << filename;
}

void ProfileManager::profile_step_time(bool if_dump_to_file) {
  // get the maximum prefill token length
  auto& model_args = engine_->model_args();
  int32_t max_context_len = model_args.max_position_embeddings();
  int32_t vocab_size = model_args.vocab_size();

  // TODO: support length for decode request profile
  int32_t profile_length_step = 128;
  int32_t profile_max_prompt_length =
      std::min(max_context_len, options_.profile_max_prompt_length());
  int32_t profile_count_per_step = 3;
  auto block_size = block_manager_pool_->options().block_size();
  bool if_profile_kv_blocks = options_.if_profile_kv_blocks();

  // warm up
  run_request(profile_max_prompt_length, 0, vocab_size);

  if (options_.if_profile_kv_blocks()) {
    // starting from max_context_len, dividing the token length by 2 in
    // each loop iteration
    // consider to generate kv blocks for prompt
    std::vector<std::tuple<int32_t, int32_t, int32_t>> time_profiling_data;
    for (int32_t token_length = profile_max_prompt_length; token_length > 1;
         token_length >>= 1) {
      // increase prefix length according to block size
      auto block_step = (profile_length_step + block_size - 1) / block_size;
      for (int32_t prefix_length = 0;
           prefix_length < token_length - 1 + (block_step * block_size);
           prefix_length += (block_step * block_size)) {
        if (prefix_length > token_length - 1) {
          // avoid kv_cache_token_num == token_length
          prefix_length = token_length - 1;
        }
        float latency_mean = 0;

        for (int32_t k = 0; k < profile_count_per_step; k++) {
          latency_mean += run_request(token_length, prefix_length, vocab_size);
        }
        latency_mean /= profile_count_per_step;
        // use token_length and prefix_length to predict
        time_profiling_data.emplace_back(
            token_length, prefix_length, static_cast<int32_t>(latency_mean));
      }
    }
    if (if_dump_to_file) {
      dump_step_time_profile_to_file(time_profiling_data);
    }
    train_time_predictor(time_profiling_data);
  } else {
    // not consider kv cache
    std::vector<std::pair<int32_t, int32_t>> time_profiling_data;
    for (int32_t token_length = profile_max_prompt_length; token_length > 1;
         token_length >>= 1) {
      float latency_mean = 0;
      for (int32_t k = 0; k < profile_count_per_step; k++) {
        latency_mean += run_request(token_length, 0, vocab_size);
      }
      latency_mean /= profile_count_per_step;
      time_profiling_data.emplace_back(token_length,
                                       static_cast<int32_t>(latency_mean));
    }
    if (if_dump_to_file) {
      dump_step_time_profile_to_file(time_profiling_data);
    }
    train_time_predictor(time_profiling_data);
  }
}

void ProfileManager::train_time_predictor(
    std::vector<std::tuple<int32_t, int32_t, int32_t>> time_profiling_data) {
  time_predictor_->fit(time_profiling_data);
}
void ProfileManager::train_time_predictor(
    std::vector<std::pair<int32_t, int32_t>> time_profiling_data) {
  time_predictor_->fit(time_profiling_data);
}

int32_t ProfileManager::predict_step_time(int32_t length,
                                          int32_t prefix_length) {
  return time_predictor_->predict_time(length, prefix_length);
}

int32_t ProfileManager::predict_step_time(Sequence* sequence) {
  auto length = sequence->num_tokens();
  auto prefix_length = sequence->kv_state().kv_cache_tokens_num();
  int32_t latency = predict_step_time(length, prefix_length);
  return latency;
}

int32_t ProfileManager::predict_step_time(std::vector<Sequence*>& sequences) {
  // TODO: OPTIMIZE for multi-node, dp_size > 1
  int32_t total_latency = 0;
  for (auto* sequence : sequences) {
    total_latency += predict_step_time(sequence);
  }
  return total_latency;
}

void ProfileManager::profile_token_budget(int32_t tpot_slo_ms) {
  // TODO: support adaptively token budget
  // use token budget means defaultly ignoring prefix cache and decode request's
  // kv cache load overhead
  ;
}
int32_t ProfileManager::get_token_budget(int32_t tpot_slo_ms) {}

// collect the latency of each step
int32_t ProfileManager::run_request(int32_t token_length,
                                    int32_t prefix_length,
                                    int32_t vocab_size) {
  // generate random token ids and request
  std::random_device rd;
  std::mt19937_64 gen(rd());
  // generate token id within the range [0, vocab_size - 1]
  std::uniform_int_distribution<int32_t> dis(0, vocab_size - 1);
  std::vector<int32_t> token_ids(token_length);
  std::generate(token_ids.begin(), token_ids.end(), [&]() { return dis(gen); });

  // generate request
  RequestState req_state(token_ids);
  Request request(/*request_id=*/"",
                  /*x_request_id=*/"",
                  /*x_request_time=*/"",
                  req_state);

  // TODO: better disable prefix cache
  // pre-allocate for sequence to get initail kv cache
  if (prefix_length > 0) {
    if (!block_manager_pool_->allocate(request.sequences()[0].get(),
                                       prefix_length)) {
      LOG(FATAL) << "Profiling TTFT failed! Not enough blocks, token length : "
                 << prefix_length;
    }
    request.sequences()[0]->kv_state().incr_kv_cache_tokens_num(prefix_length);
  }

  // allocate blocks
  if (!block_manager_pool_->allocate(request.sequences()[0].get())) {
    LOG(FATAL) << "Profiling TTFT failed! Not enough blocks, token length : "
               << token_length;
  }
  // build batch
  auto batches =
      BatchFactory::get_instance(options_.dp_size())
          ->create_batches(
              {request.sequences()[0].get()}, {token_length}, nullptr, nullptr);

  absl::Time start_time = absl::Now();
  engine_->step(batches);
  if (options_.enable_schedule_overlap()) {
    engine_->update_last_step_result(batches);
  }
  const int32_t latency = absl::ToInt64Milliseconds(absl::Now() - start_time);
  block_manager_pool_->deallocate(&request);

  return latency;
}

}  // namespace xllm

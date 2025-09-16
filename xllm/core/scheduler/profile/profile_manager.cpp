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
      std::make_unique<TimePredictor>(options.enable_profile_kv_blocks());
  if (options.enable_profile_step_time()) {
    LOG(INFO) << "Starting profiliing step time.";
    profile_step_time(true);
    eval_sequence_latency_prediction();
    eval_batch_latency_prediction();
  }
  if (options.enable_profile_token_budget()) {
    LOG(INFO) << "Starting profiliing token budget.";
    profile_token_budget();
  }
  // more profile here, such as token_budget profile and decode length
  // prediction.
}

// currently only for test debug
void ProfileManager::eval_sequence_latency_prediction() {
  auto& model_args = engine_->model_args();
  int32_t vocab_size = model_args.vocab_size();
  std::vector<int32_t> pred_vec;
  std::vector<int32_t> target_vec;
  int32_t token_step = 80;
  int32_t prefix_step = 400;
  int32_t upper_bound = 4000;

  LOG(INFO) << "Starting testing sequence latency prediction.";
  for (int32_t token_length = token_step; token_length < upper_bound;
       token_length += token_step) {
    for (int32_t prefix_length = 0; prefix_length < token_length;
         prefix_length += prefix_step) {
      target_vec.emplace_back(
          run_request(token_length, prefix_length, vocab_size));
      pred_vec.emplace_back(predict_step_time(token_length, prefix_length));
    }
  }

  // print
  for (const auto& element : pred_vec) {
    std::cout << element << " ";
  }
  std::cout << std::endl;
  for (const auto& element : target_vec) {
    std::cout << element << " ";
  }
  std::cout << std::endl;

  double sum_error = 0.0;
  for (size_t i = 0; i < pred_vec.size(); ++i) {
    sum_error += std::abs(pred_vec[i] - target_vec[i]);
  }
  double mae = sum_error / pred_vec.size();

  LOG(INFO) << "Mean Absolute Error (MAE) of latency prediction: " << mae;
}
void ProfileManager::eval_batch_latency_prediction() {
  auto& model_args = engine_->model_args();
  int32_t vocab_size = model_args.vocab_size();
  std::vector<int32_t> pred_vec;
  std::vector<int32_t> target_vec;
  int32_t token_step = 400;
  int32_t prefix_step = 400;
  int32_t upper_bound = 4000;
  int32_t batch_size = 10;

  LOG(INFO) << "Starting testing batch latency prediction.";
  for (int32_t token_length = token_step; token_length < upper_bound;
       token_length += token_step) {
    for (int32_t prefix_length = 0; prefix_length < token_length;
         prefix_length += prefix_step) {
      target_vec.emplace_back(
          run_request(token_length, prefix_length, vocab_size, batch_size));
      pred_vec.emplace_back(
          predict_step_time(token_length, prefix_length, batch_size));
    }
  }

  // print
  for (const auto& element : pred_vec) {
    std::cout << element << " ";
  }
  std::cout << std::endl;
  for (const auto& element : target_vec) {
    std::cout << element << " ";
  }
  std::cout << std::endl;

  double sum_error = 0.0;
  for (size_t i = 0; i < pred_vec.size(); ++i) {
    sum_error += std::abs(pred_vec[i] - target_vec[i]);
  }
  double mae = sum_error / pred_vec.size();

  LOG(INFO) << "Mean Absolute Error (MAE) of latency prediction: " << mae;
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
  int32_t profile_max_prompt_length =
      std::min(max_context_len, options_.profile_max_prompt_length());
  auto block_size = block_manager_pool_->options().block_size();
  bool enable_profile_kv_blocks = options_.enable_profile_kv_blocks();

  // warm up
  run_request(profile_max_prompt_length, 0, vocab_size);

  if (options_.enable_profile_kv_blocks()) {
    // starting from max_context_len, dividing the token length by 2 in
    // each loop iteration
    // consider to generate kv blocks for prompt
    std::vector<std::tuple<int32_t, int32_t, int32_t>> time_profiling_data;
    for (int32_t token_length = profile_max_prompt_length; token_length > 1;
         token_length >>= 1) {
      // increase prefix length according to block size
      auto block_step = (profile_length_step_ + block_size - 1) / block_size;
      for (int32_t prefix_length = 0;
           prefix_length < token_length - 1 + (block_step * block_size);
           prefix_length += (block_step * block_size)) {
        if (prefix_length > token_length - 1) {
          // avoid kv_cache_token_num == token_length
          prefix_length = token_length - 1;
        }
        float latency_mean = 0;

        for (int32_t k = 0; k < profile_count_per_step_; k++) {
          latency_mean += run_request(token_length, prefix_length, vocab_size);
        }
        latency_mean /= profile_count_per_step_;
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
      for (int32_t k = 0; k < profile_count_per_step_; k++) {
        latency_mean += run_request(token_length, 0, vocab_size);
      }
      latency_mean /= profile_count_per_step_;
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
// for single sequence
int32_t ProfileManager::predict_step_time(int32_t length,
                                          int32_t prefix_length,
                                          bool if_need_add_constant_term) {
  return time_predictor_->predict_time(
      length, prefix_length, if_need_add_constant_term);
}

int32_t ProfileManager::predict_step_time(Sequence* sequence,
                                          bool if_need_add_constant_term) {
  auto length = sequence->num_tokens();
  auto prefix_length = sequence->kv_state().kv_cache_tokens_num();
  int32_t latency =
      predict_step_time(length, prefix_length, if_need_add_constant_term);
  return latency;
}
// for single batch or sequences
// seq in batch with the same token and prefix length
int32_t ProfileManager::predict_step_time(int32_t length,
                                          int32_t prefix_length,
                                          int32_t batch_size) {
  // int32_t total_latency = time_predictor_->get_constant_overhead();
  // for (int32_t i = 0; i < batch_size; i++) {
  //   // predict for each sequence
  //   total_latency += predict_step_time(length, prefix_length, false);
  // }
  // return total_latency;
  int32_t total_latency = 0;
  for (int32_t i = 0; i < batch_size; i++) {
    total_latency += predict_step_time(length, prefix_length, true);
  }
  return total_latency;
}

int32_t ProfileManager::predict_step_time(std::vector<Sequence*>& sequences) {
  // TODO: OPTIMIZE for multi-node, dp_size > 1
  int32_t total_latency = time_predictor_->get_constant_overhead();
  for (auto* sequence : sequences) {
    total_latency += predict_step_time(sequence, false);
  }
  return total_latency;
}

void ProfileManager::profile_token_budget() {
  // use token budget means defaultly ignoring prefix cache and decode request's
  // kv cache load overhead
  profile_token_budget_ = binary_search_max_tokens(
      options_.max_global_tpot_ms(), 1, options_.max_tokens_per_batch());
  LOG(INFO) << "Profile token budget: " << profile_token_budget_
            << "for TPOT SLO: " << options_.max_global_tpot_ms();
}

bool ProfileManager::check_if_satisfy_slo(int32_t num_tokens,
                                          int32_t tpot_slo_ms) {
  auto& model_args = engine_->model_args();
  int32_t vocab_size = model_args.vocab_size();
  int32_t prompt_tokens_per_batch = 1024;

  auto batch_size = num_tokens / prompt_tokens_per_batch;
  int32_t extra_token_length = num_tokens % prompt_tokens_per_batch;
  int32_t batch_latency = 0;
  for (int32_t k = 0; k < profile_count_per_step_; k++) {
    batch_latency += run_request(
        prompt_tokens_per_batch, 0, vocab_size, batch_size, extra_token_length);
  }
  batch_latency /= profile_count_per_step_;
  if (batch_latency <= tpot_slo_ms) {
    return true;
  } else {
    return false;
  }
}

int32_t ProfileManager::binary_search_max_tokens(int32_t tpot_slo_ms,
                                                 int32_t lower_bound,
                                                 int32_t upper_bound) {
  int32_t left = lower_bound;
  int32_t right = upper_bound;
  // [left, right)
  while (left < right) {
    int32_t mid = left + (right - left) / 2;
    if (check_if_satisfy_slo(mid, tpot_slo_ms)) {
      left = mid + 1;
    } else {
      right = mid;
    }
  }
  return left - 1;
}

int32_t ProfileManager::get_token_budget() { return profile_token_budget_; }

std::shared_ptr<Request> ProfileManager::generate_single_request(
    int32_t token_length,
    int32_t prefix_length,
    int32_t vocab_size) {
  std::random_device rd;
  std::mt19937_64 gen(rd());
  std::uniform_int_distribution<int32_t> dis(0, vocab_size - 1);
  std::vector<int32_t> token_ids(token_length);
  std::generate(token_ids.begin(), token_ids.end(), [&]() { return dis(gen); });

  // 直接构造 shared_ptr<Request>
  RequestState req_state(token_ids);
  auto request = std::make_shared<Request>(
      /*request_id=*/"",
      /*x_request_id=*/"",
      /*x_request_time=*/"",
      req_state);

  // TODO: better disable prefix cache
  if (prefix_length > 0) {
    if (!block_manager_pool_->allocate(request->sequences()[0].get(),
                                       prefix_length)) {
      LOG(FATAL) << "Profiling time failed! Not enough blocks, prefix length : "
                 << prefix_length;
    }
    request->sequences()[0]->kv_state().incr_kv_cache_tokens_num(prefix_length);
  }

  if (!block_manager_pool_->allocate(request->sequences()[0].get())) {
    LOG(FATAL) << "Profiling time failed! Not enough blocks, token length : "
               << token_length;
  }

  return request;
}

// collect the latency of each step
int32_t ProfileManager::run_request(int32_t token_length,
                                    int32_t prefix_length,
                                    int32_t vocab_size,
                                    int32_t batch_size,
                                    int32_t extra_token_length) {
  CHECK(token_length >= prefix_length);
  std::vector<Sequence*> sequences;
  std::vector<size_t> sequences_budget;
  std::vector<std::shared_ptr<Request>> requests;

  // batch sequences with the same kv cahce and token length
  for (int32_t i = 0; i < batch_size; i++) {
    // generate random token ids and request
    std::shared_ptr<Request> request =
        generate_single_request(token_length, prefix_length, vocab_size);
    requests.emplace_back(request);
    sequences.emplace_back(request->sequences()[0].get());
    sequences_budget.emplace_back(token_length - prefix_length);
  }
  // maybe another sequence for extra token length (< token_length) for token
  // budget profiling
  if (extra_token_length > 0) {
    std::shared_ptr<Request> request =
        generate_single_request(token_length, prefix_length, vocab_size);
    requests.emplace_back(request);
    sequences.emplace_back(request->sequences()[0].get());
    sequences_budget.emplace_back(token_length - prefix_length);
  }
  // build batch
  auto batches =
      BatchFactory::get_instance(options_.dp_size())
          ->create_batches(
              requests, sequences, sequences_budget, nullptr, nullptr);

  absl::Time start_time = absl::Now();
  engine_->step(batches);
  if (options_.enable_schedule_overlap()) {
    engine_->update_last_step_result(batches);
  }
  const int32_t latency = absl::ToInt64Milliseconds(absl::Now() - start_time);
  for (auto& request : requests) {
    block_manager_pool_->deallocate(request.get());
  }

  return latency;
}

}  // namespace xllm

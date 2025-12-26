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

#include "common/global_flags.h"
#include "framework/batch/batch_factory.h"
#include "framework/request/request_state.h"

namespace xllm {

ProfileManager::ProfileManager(Engine* engine, const Options& options)
    : options_(options), engine_(engine) {
  CHECK(engine_ != nullptr);
  block_manager_pool_ = engine_->block_manager_pool();
  CHECK(block_manager_pool_ != nullptr);
  prefill_time_predictor_ = std::make_unique<TimePredictor>(
      options.enable_profile_kv_blocks(), true /*is_prefill*/);
  decode_time_predictor_ = std::make_unique<TimePredictor>(
      options.enable_profile_kv_blocks(), false /*is_prefill*/);
  if (options.enable_profile_step_time()) {
    LOG(INFO) << "Starting profiliing step time.";
    profile_step_time(true);
    // test accuracy
    eval_sequence_latency_prediction();
    eval_batch_latency_prediction("only_prefill");
    eval_batch_latency_prediction("only_decode");
    eval_batch_latency_prediction("mix");
  }
  if (options.enable_profile_token_budget()) {
    LOG(INFO) << "Starting profiliing token budget.";
    profile_token_budget();
  }
  // more profile here, such as token_budget profile and decode length
  // prediction.

#if defined(USE_NPU)
  // Warmup ACL graph executor if enabled
  if (FLAGS_enable_graph) {
    LOG(INFO) << "Starting ACL graph warmup.";
    warmup_for_acl_graph();
  }
#endif
}

// --------------------- for test only ---------------------------
void ProfileManager::eval_sequence_latency_prediction() {
  std::vector<double> pred_vec;
  std::vector<double> target_vec;
  int32_t token_step = 500;
  int32_t prefix_step = 500;
  int32_t upper_bound = 4000;

  LOG(INFO) << "Starting testing sequence latency prediction";
  for (int32_t token_length = token_step; token_length < upper_bound;
       token_length += token_step) {
    for (int32_t prefix_length = 0; prefix_length < token_length;
         prefix_length += prefix_step) {
      target_vec.emplace_back(run_request(token_length, prefix_length));
      pred_vec.emplace_back(predict_step_time(token_length, prefix_length));
    }
  }

  // print for debug
  for (const auto& element : pred_vec) {
    std::cout << static_cast<int32_t>(element) << " ";
  }
  std::cout << std::endl;
  for (const auto& element : target_vec) {
    std::cout << static_cast<int32_t>(element) << " ";
  }
  std::cout << std::endl;

  double sum_error = 0.0;
  double sum_percentage_error = 0.0;

  for (size_t i = 0; i < pred_vec.size(); ++i) {
    double error = std::abs(pred_vec[i] - target_vec[i]);
    sum_error += error;
    sum_percentage_error += error / std::abs(target_vec[i]);
  }
  double mae = sum_error / pred_vec.size();
  double mape = (sum_percentage_error / pred_vec.size()) * 100.0;

  LOG(INFO) << "Mean Absolute Error (MAE) of latency prediction: " << mae
            << " ms";
  LOG(INFO) << "Mean Absolute Percentage Error (MAPE) of latency prediction: "
            << mape << " %";
}
void ProfileManager::eval_batch_latency_prediction(const std::string mode) {
  std::vector<double> pred_vec;
  std::vector<double> target_vec;

  LOG(INFO) << "Starting testing batch latency prediction for " << mode;
  if (mode == "only_prefill") {
    int32_t max_batch_size = 10;
    int32_t token_step = 500;
    int32_t prefix_step = 500;
    int32_t upper_bound = 4000;
    for (int32_t token_length = token_step; token_length < upper_bound;
         token_length += token_step) {
      for (int32_t prefix_length = 0; prefix_length < token_length;
           prefix_length += prefix_step) {
        target_vec.emplace_back(
            run_request(token_length, prefix_length, max_batch_size));
        pred_vec.emplace_back(
            predict_step_time(token_length, prefix_length, max_batch_size));
      }
    }
  }
  if (mode == "only_decode") {
    int32_t max_batch_size = 200;
    int32_t token_length = 500;
    for (int32_t batch_size = 1; batch_size < max_batch_size; batch_size++) {
      target_vec.emplace_back(
          run_request(token_length, token_length - 1, batch_size));
      pred_vec.emplace_back(
          predict_step_time(token_length, token_length - 1, batch_size));
    }
  }
  if (mode == "mix") {
    if (!FLAGS_enable_chunked_prefill) {
      LOG(WARNING) << "When chunked prefill is disabled, mixed prefill and "
                      "decode scenarios will not be tested.";
      return;
    }
    int32_t max_batch_size = 100;
    int32_t max_prefill_cnt = 5;
    int32_t token_length = 500;
    for (int32_t batch_size = 50; batch_size <= max_batch_size;
         batch_size += 10) {
      for (int32_t prefill_cnt = 0; prefill_cnt <= max_prefill_cnt;
           prefill_cnt++) {
        std::vector<int32_t> token_length_vec;
        std::vector<int32_t> prefix_length_vec;
        token_length_vec.insert(
            token_length_vec.end(), prefill_cnt, token_length);
        prefix_length_vec.insert(prefix_length_vec.end(), prefill_cnt, 0);
        // token_length_vec.insert(token_length_vec.end(), batch_size/5,
        // token_length); prefix_length_vec.insert(prefix_length_vec.end(),
        // batch_size/5, token_length-1);
        token_length_vec.insert(
            token_length_vec.end(), batch_size - prefill_cnt, token_length);
        prefix_length_vec.insert(prefix_length_vec.end(),
                                 batch_size - prefill_cnt,
                                 token_length - 1);
        target_vec.emplace_back(
            run_request(token_length_vec, prefix_length_vec));
        pred_vec.emplace_back(
            predict_step_time(token_length_vec, prefix_length_vec));
      }
    }
  }

  // print for debug
  for (const auto& element : pred_vec) {
    std::cout << static_cast<int32_t>(element) << " ";
  }
  std::cout << std::endl;
  for (const auto& element : target_vec) {
    std::cout << static_cast<int32_t>(element) << " ";
  }
  std::cout << std::endl;

  double sum_error = 0.0;
  double sum_percentage_error = 0.0;

  for (size_t i = 0; i < pred_vec.size(); ++i) {
    double error = std::abs(pred_vec[i] - target_vec[i]);
    sum_error += error;
    sum_percentage_error += error / std::abs(target_vec[i]);
  }
  double mae = sum_error / pred_vec.size();
  double mape = (sum_percentage_error / pred_vec.size()) * 100.0;

  LOG(INFO) << "Mean Absolute Error (MAE) of latency prediction: " << mae
            << " ms";
  LOG(INFO) << "Mean Absolute Percentage Error (MAPE) of latency prediction: "
            << mape << " %";
}
// -------------------------------------------------------------

// ---------------------- dump to file-----------------------
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
    const std::vector<std::pair<int32_t, double>>& time_profiling_data,
    bool is_prefill) {
  std::string filename = is_prefill
                             ? generate_filename("profile_prefill_step_time")
                             : generate_filename("profile_decode_step_time");
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
    const std::vector<std::tuple<int32_t, int32_t, double>>&
        time_profiling_data,
    bool is_prefill) {
  std::string filename = is_prefill
                             ? generate_filename("profile_prefill_step_time")
                             : generate_filename("profile_decode_step_time");
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
// -------------------------------------------------------------

void ProfileManager::profile_step_time(bool if_dump_to_file) {
  // get the maximum prefill token length
  auto& model_args = engine_->model_args();
  int32_t max_context_len = model_args.max_position_embeddings();

  // TODO: support length for decode request profile
  int32_t profile_max_prompt_length =
      std::min(max_context_len, options_.profile_max_prompt_length());
  auto block_size = block_manager_pool_->options().block_size();
  bool enable_profile_kv_blocks = options_.enable_profile_kv_blocks();

  // warm up
  run_request(profile_max_prompt_length, 0);

  // prefill time profile
  if (options_.enable_profile_kv_blocks()) {
    // starting from max_context_len, dividing the token length by 2 in
    // each loop iteration
    // consider to generate kv blocks for prompt
    std::vector<std::tuple<int32_t, int32_t, double>> time_profiling_data;
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
        double latency_mean = 0;

        for (int32_t k = 0; k < profile_count_per_step_; k++) {
          latency_mean += run_request(token_length, prefix_length);
        }
        latency_mean /= profile_count_per_step_;
        // use token_length and prefix_length to predict
        time_profiling_data.emplace_back(
            token_length, prefix_length, latency_mean);
      }
    }
    if (if_dump_to_file) {
      dump_step_time_profile_to_file(time_profiling_data, true /*is_prefill*/);
    }
    train_prefill_time_predictor(time_profiling_data);
  } else {
    // not consider kv cache
    std::vector<std::pair<int32_t, double>> time_profiling_data;
    for (int32_t token_length = profile_max_prompt_length; token_length > 1;
         token_length >>= 1) {
      double latency_mean = 0;
      for (int32_t k = 0; k < profile_count_per_step_; k++) {
        latency_mean += run_request(token_length, 0);
      }
      latency_mean /= profile_count_per_step_;
      time_profiling_data.emplace_back(token_length, latency_mean);
    }
    if (if_dump_to_file) {
      dump_step_time_profile_to_file(time_profiling_data, true /*is_prefill*/);
    }
    train_prefill_time_predictor(time_profiling_data);
  }

  // decode time profile
  std::vector<std::tuple<int32_t, int32_t, double>> time_profiling_data;
  int32_t max_batch_size = 50;
  // for (int32_t token_length = profile_max_prompt_length; token_length >
  // 1;token_length >>= 1)
  for (int32_t token_length = 2; token_length < profile_max_prompt_length;
       token_length += profile_length_step_) {
    for (int32_t batch_size = 1; batch_size < max_batch_size; batch_size += 2) {
      double latency_mean = 0;
      for (int32_t k = 0; k < profile_count_per_step_; k++) {
        latency_mean += run_request(token_length, token_length - 1, batch_size);
      }
      latency_mean /= profile_count_per_step_;
      time_profiling_data.emplace_back(token_length, batch_size, latency_mean);
    }
  }
  if (if_dump_to_file) {
    dump_step_time_profile_to_file(time_profiling_data, false /*is_prefill*/);
  }
  train_decode_time_predictor(time_profiling_data);
}

void ProfileManager::train_prefill_time_predictor(
    std::vector<std::tuple<int32_t, int32_t, double>> time_profiling_data) {
  prefill_time_predictor_->fit_for_prefill(time_profiling_data);
}
void ProfileManager::train_prefill_time_predictor(
    std::vector<std::pair<int32_t, double>> time_profiling_data) {
  prefill_time_predictor_->fit_for_prefill(time_profiling_data);
}
void ProfileManager::train_decode_time_predictor(
    std::vector<std::tuple<int32_t, int32_t, double>> time_profiling_data) {
  decode_time_predictor_->fit_for_decode(time_profiling_data);
}

// ----------------------predict step time-----------------------

double ProfileManager::get_constant_overhead() {
  if (prefill_time_predictor_->is_trained() &&
      decode_time_predictor_->is_trained()) {
    return (prefill_time_predictor_->get_constant_overhead() +
            decode_time_predictor_->get_constant_overhead()) /
           2;
  } else if (prefill_time_predictor_->is_trained()) {
    return prefill_time_predictor_->get_constant_overhead();
  } else if (decode_time_predictor_->is_trained()) {
    return decode_time_predictor_->get_constant_overhead();
  }
  return 0.0;
}

// for single sequence
double ProfileManager::predict_step_time(int32_t length,
                                         int32_t prefix_length,
                                         bool if_need_add_constant_term,
                                         bool force_use_prefill_predictor) {
  CHECK(length > prefix_length);
  if (force_use_prefill_predictor) {
    return prefill_time_predictor_->predict_time(
        length, prefix_length, if_need_add_constant_term);
  }
  if (length - 1 == prefix_length) {
    return decode_time_predictor_->predict_time(
        length, prefix_length, if_need_add_constant_term);
  } else {
    return prefill_time_predictor_->predict_time(
        length, prefix_length, if_need_add_constant_term);
  }
}

double ProfileManager::predict_step_time(Sequence* sequence,
                                         bool if_need_add_constant_term,
                                         bool force_use_prefill_predictor) {
  auto length = sequence->num_tokens();
  auto prefix_length = sequence->kv_state().kv_cache_tokens_num();
  double latency = predict_step_time(length,
                                     prefix_length,
                                     if_need_add_constant_term,
                                     force_use_prefill_predictor);
  return latency;
}
// for single batch or sequences
double ProfileManager::predict_step_time(
    const std::vector<int32_t>& length_vec,
    const std::vector<int32_t>& prefix_length_vec) {
  CHECK(length_vec.size() == prefix_length_vec.size());
  double total_latency = get_constant_overhead();
  for (int32_t i = 0; i < length_vec.size(); i++) {
    // predict for each sequence
    int32_t length = length_vec[i];
    int32_t prefix_length = prefix_length_vec[i];
    total_latency += predict_step_time(length, prefix_length, false);
  }
  return total_latency;
}

// for seq in batch with the same token and prefix length
double ProfileManager::predict_step_time(int32_t length,
                                         int32_t prefix_length,
                                         int32_t batch_size) {
  double total_latency = get_constant_overhead();
  for (int32_t i = 0; i < batch_size; i++) {
    // predict for each sequence
    total_latency += predict_step_time(length, prefix_length, false);
  }
  return total_latency;
}
// ---------------------------------------------

// ----------------------for profile token budget-----------------------
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
  int32_t prompt_tokens_per_batch = 1024;

  auto batch_size = num_tokens / prompt_tokens_per_batch;
  int32_t extra_token_length = num_tokens % prompt_tokens_per_batch;
  double batch_latency = 0;
  for (int32_t k = 0; k < profile_count_per_step_; k++) {
    batch_latency +=
        run_request(prompt_tokens_per_batch, 0, batch_size, extra_token_length);
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

// ---------------------------------------------

std::shared_ptr<Request> ProfileManager::generate_single_request(
    int32_t token_length,
    int32_t prefix_length) {
  auto& model_args = engine_->model_args();
  int32_t vocab_size = model_args.vocab_size();
  int32_t eos_token_id = model_args.eos_token_id();

  std::random_device rd;
  std::mt19937_64 gen(rd());

  // If req_state does not initialize the stopchecker, default eos_token_id = 0,
  // need to skip it
  std::uniform_int_distribution<int32_t> dis(1, vocab_size - 2);

  std::vector<int32_t> token_ids(token_length);
  std::generate(token_ids.begin(), token_ids.end(), [&]() {
    int32_t token = dis(gen);
    return token == eos_token_id ? token + 1 : token;  // skip eos
  });

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
double ProfileManager::run_request(int32_t token_length,
                                   int32_t prefix_length,
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
        generate_single_request(token_length, prefix_length);
    requests.emplace_back(request);
    sequences.emplace_back(request->sequences()[0].get());
    sequences_budget.emplace_back(token_length - prefix_length);
  }
  // maybe another sequence for extra token length (< token_length) for token
  // budget profiling
  if (extra_token_length > 0) {
    std::shared_ptr<Request> request =
        generate_single_request(token_length, prefix_length);
    requests.emplace_back(request);
    sequences.emplace_back(request->sequences()[0].get());
    sequences_budget.emplace_back(token_length - prefix_length);
  }
  // build batch
  auto batches = BatchFactory::get_instance(options_.dp_size())
                     ->create_batches(requests, sequences, sequences_budget);

  absl::Time start_time = absl::Now();
  engine_->step(batches);
  if (options_.enable_schedule_overlap()) {
    engine_->update_last_step_result(batches);
  }
  double latency = absl::ToDoubleMilliseconds(absl::Now() - start_time);
  for (auto& request : requests) {
    block_manager_pool_->deallocate(request.get());
  }

  return latency;
}

// currently for test only
double ProfileManager::run_request(
    const std::vector<int32_t>& token_length_vec,
    const std::vector<int32_t>& prefix_length_vec) {
  CHECK(token_length_vec.size() == prefix_length_vec.size());
  std::vector<Sequence*> sequences;
  std::vector<size_t> sequences_budget;
  std::vector<std::shared_ptr<Request>> requests;

  // batch sequences with the same kv cahce and token length
  for (int32_t i = 0; i < token_length_vec.size(); i++) {
    // generate random token ids and request
    int32_t token_length = token_length_vec[i];
    int32_t prefix_length = prefix_length_vec[i];

    std::shared_ptr<Request> request =
        generate_single_request(token_length, prefix_length);
    requests.emplace_back(request);
    sequences.emplace_back(request->sequences()[0].get());
    sequences_budget.emplace_back(token_length - prefix_length);
  }
  // build batch
  auto batches =
      BatchFactory::get_instance(options_.dp_size())
          ->create_batches(requests, sequences, sequences_budget, nullptr);

  absl::Time start_time = absl::Now();
  engine_->step(batches);
  if (options_.enable_schedule_overlap()) {
    engine_->update_last_step_result(batches);
  }
  double latency = absl::ToDoubleMilliseconds(absl::Now() - start_time);
  for (auto& request : requests) {
    block_manager_pool_->deallocate(request.get());
  }

  return latency;
}

// Generate a batch of decode requests and execute it, then return the step
// latency.
double ProfileManager::profile_decode_step_time(int32_t token_length,
                                                int32_t batch_size,
                                                int32_t min_context_len,
                                                int32_t max_context_len) {
  double total_latency = 0;
  for (int32_t i = 0; i < profile_count_per_step_; ++i) {
    std::vector<int32_t> token_length_vec;
    std::vector<int32_t> prefix_length_vec;
    generate_random_decode_batch(batch_size * token_length,
                                 batch_size,
                                 min_context_len,
                                 max_context_len,
                                 token_length_vec,
                                 prefix_length_vec);
    double latency = run_request(token_length_vec, prefix_length_vec);
    total_latency += latency;
  }
  return total_latency / profile_count_per_step_;
}

// Generate a batch of random decode requests with an average length of
// token_length.
void ProfileManager::generate_random_decode_batch(
    int32_t total_length,
    int32_t batch_size,
    int32_t min_context_len,
    int32_t max_context_len,
    std::vector<int32_t>& token_length_vec,
    std::vector<int32_t>& prefix_length_vec) {
  CHECK(total_length >= batch_size * min_context_len);
  CHECK(total_length <= batch_size * max_context_len);

  token_length_vec.resize(batch_size, min_context_len);
  prefix_length_vec.resize(batch_size, min_context_len - 1);
  int remain = total_length - batch_size * min_context_len;

  std::random_device rd;
  std::mt19937_64 gen(rd());

  for (int i = 0; i < batch_size; ++i) {
    if (remain == 0) break;

    int max = remain > (max_context_len - min_context_len)
                  ? (max_context_len - min_context_len)
                  : remain;

    std::uniform_int_distribution<int> dis(0, max);
    int add = dis(gen);
    token_length_vec[i] += add;
    prefix_length_vec[i] += add;
    remain -= add;
  }

  int idx = 0;
  while (remain > 0) {
    if (token_length_vec[idx % batch_size] < max_context_len) {
      token_length_vec[idx % batch_size] += 1;
      prefix_length_vec[idx % batch_size] += 1;
      --remain;
    }
    ++idx;
  }
}

void ProfileManager::warmup_for_acl_graph() {
  LOG(INFO) << "Starting ACL graph warmup with prefill and decode requests...";

  auto& model_args = engine_->model_args();
  int32_t max_context_len = model_args.max_position_embeddings();

  // Warmup parameters - align with bucket logic
  // Prefill: align max_tokens_per_batch to bucket
  int32_t prefill_tokens =
      std::min(FLAGS_max_tokens_per_batch, max_context_len);

  std::vector<int32_t> decode_seq_lens = {16};

  // Generate decode_batch_sizes aligned with bucket logic
  // For decode: n_tokens = batch_size * num_decoding_tokens (usually
  // num_decoding_tokens = 1) So batch_size directly corresponds to n_tokens
  // bucket values Bucket values: 1, 2, 4, 8, 16, then 32, 48, 64, ...
  // (multiples of 16)
  std::vector<int32_t> decode_batch_sizes = {1, 2, 4, 8, 16};
  int32_t max_seqs_per_batch = FLAGS_max_seqs_per_batch;
  // From 32 onwards, use multiples of 16 (bucket alignment)
  for (int32_t batch_size = 32; batch_size <= max_seqs_per_batch;
       batch_size += 16) {
    decode_batch_sizes.push_back(batch_size);
  }
  // Ensure max_seqs_per_batch is included if not already added
  if (decode_batch_sizes.back() != max_seqs_per_batch) {
    decode_batch_sizes.push_back(max_seqs_per_batch);
  }

  // Limit decode seq_lens to max_context_len
  for (auto& seq_len : decode_seq_lens) {
    if (seq_len > max_context_len) {
      seq_len = max_context_len;
    }
  }

  // ========== Warmup Prefill Request ==========
  LOG(INFO) << "Warming up prefill request: tokens=" << prefill_tokens;
  try {
    // Prefill: prefix_length = 0 (empty KV cache), batch_size = 10,
    // sequence_length = prefill_tokens / 10
    double latency = run_request(prefill_tokens, 0, 1);
    LOG(INFO) << "Prefill warmup completed: tokens=" << prefill_tokens
              << ", latency=" << latency << " ms";
  } catch (const std::exception& e) {
    LOG(WARNING) << "Prefill warmup failed: tokens=" << prefill_tokens
                 << ", error: " << e.what();
  }

  // ========== Warmup Decode Requests ==========
  // confict with async_schedule, so skip for now

  LOG(INFO) << "ACL graph warmup completed";
}

}  // namespace xllm

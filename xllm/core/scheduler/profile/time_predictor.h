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

#include <Eigen/Dense>

namespace xllm {

// Predictor for predicting time based on input length
class TimePredictor final {
 public:
  explicit TimePredictor(bool if_profile_prefix);

  void fit(const std::vector<std::pair<int32_t, int32_t>>& time_profiling_data);

  void fit(const std::vector<std::tuple<int32_t, int32_t, int32_t>>&
               time_profiling_data);

  ~TimePredictor() = default;

  int32_t predict_time(int32_t length,
                       int32_t prefix_length = 0,
                       bool if_need_add_constant_term = true);

  int32_t get_constant_overhead();

 private:
  Eigen::VectorXd coefficients_;
  bool if_profile_prefix_ = false;
};

}  // namespace xllm

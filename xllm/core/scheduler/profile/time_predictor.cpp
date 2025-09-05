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

#include "time_predictor.h"

#include <glog/logging.h>

#include <vector>

namespace xllm {

TimePredictor::TimePredictor(bool if_profile_prefix)
    : if_profile_prefix_(if_profile_prefix) {
  if (!if_profile_prefix) {
    coefficients_ = Eigen::VectorXd::Zero(3);
  } else {
    coefficients_ = Eigen::VectorXd::Zero(5);
  }
}

void TimePredictor::fit(
    const std::vector<std::pair<int32_t, int32_t>>& time_profiling_data) {
  // construct Vandermonde matrix
  int32_t m = time_profiling_data.size();
  int32_t n = 3;  // use 2-degree polynomial
  Eigen::MatrixXd matrix(m, n);
  for (int32_t i = 0; i < m; ++i) {
    for (int32_t j = 0; j < n; ++j) {
      matrix(i, j) = std::pow(time_profiling_data[i].first, j);
    }
  }
  // construct target vector
  Eigen::VectorXd target(m);
  for (int32_t i = 0; i < m; ++i) {
    target(i) = time_profiling_data[i].second;
  }
  // get coefficients
  coefficients_ = matrix.colPivHouseholderQr().solve(target);

  // Output equation
  std::stringstream equation;
  equation << "Fitted equation: time = ";
  for (int32_t i = 0; i < coefficients_.size(); ++i) {
    if (i > 0) equation << " + ";
    equation << coefficients_(i);
    if (i == 1)
      equation << " * x";
    else if (i > 1)
      equation << " * x^" << i;
  }
  LOG(INFO) << equation.str();
}

void TimePredictor::fit(
    const std::vector<std::tuple<int32_t, int32_t, int32_t>>&
        time_profiling_data) {
  int32_t m = time_profiling_data.size();
  int32_t n = 5;
  Eigen::MatrixXd matrix(m, n);

  for (int32_t i = 0; i < m; ++i) {
    int32_t token_length = std::get<0>(time_profiling_data[i]);
    int32_t prefix_length = std::get<1>(time_profiling_data[i]);
    int32_t diff = token_length - prefix_length;

    matrix(i, 0) = diff * diff;           // (token_length-prefix_length)^2
    matrix(i, 1) = diff;                  // (token_length-prefix_length)
    matrix(i, 2) = diff * prefix_length;  // prefix_length
    matrix(i, 3) = prefix_length;         // (token_length-prefix_length)
    matrix(i, 4) = 1.0;                   // constant
  }
  // construct target vector
  Eigen::VectorXd target(m);
  for (int32_t i = 0; i < m; ++i) {
    target(i) = std::get<2>(time_profiling_data[i]);
  }
  // get coefficients
  coefficients_ = matrix.colPivHouseholderQr().solve(target);
  // output equation
  LOG(INFO) << "Fitted equation: time = " << coefficients_(0) << " * diff^2 + "
            << coefficients_(1) << " * diff + " << coefficients_(2)
            << " * (diff * prefix_length) + " << coefficients_(3)
            << " * prefix_length + " << coefficients_(4);
}

int32_t TimePredictor::predict_time(int32_t length, int32_t prefix_length) {
  double result = 0.0;
  if (!if_profile_prefix_) {
    // use prefix-free profile
    int32_t effective_length = length - prefix_length;
    double power = 1.0;
    for (int32_t i = 0; i < coefficients_.size(); ++i) {
      result += coefficients_(i) * power;
      power *= effective_length;
    }

  } else {
    // prefix profile
    int32_t diff = length - prefix_length;
    result = coefficients_(0) * diff * diff + coefficients_(1) * diff +
             coefficients_(2) * diff * prefix_length +
             coefficients_(3) * prefix_length + coefficients_(4);
  }
  if (result < 0) {
    LOG(WARNING) << "Negative time prediction: " << result;
    result = 0;
  }
  return static_cast<int32_t>(result);
}

}  // namespace xllm
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

#include <folly/Function.h>

#include <functional>
#include <future>
#include <vector>

#include "common/macros.h"
#include "common/options.h"
#include "common/rate_limiter.h"
#include "common/types.h"
#include "engine.h"
#include "framework/request/request_params.h"
namespace xllm {

class Master {
 public:
  explicit Master(const Options& options, EngineType type);
  virtual ~Master() = default;
  virtual void run() = 0;
  virtual const Options& options() const { return options_; }

  virtual void get_cache_info(std::vector<uint64_t>& cluster_ids,
                              std::vector<std::string>& addrs,
                              std::vector<int64_t>& k_cache_ids,
                              std::vector<int64_t>& v_cache_ids) {}

  virtual bool link_cluster(const std::vector<uint64_t>& cluster_ids,
                            const std::vector<std::string>& addrs,
                            const std::vector<std::string>& device_ips,
                            const std::vector<uint16_t>& ports,
                            const int32_t dp_size) {
    return false;
  }

  virtual bool unlink_cluster(const std::vector<uint64_t>& cluster_ids,
                              const std::vector<std::string>& addrs,
                              const std::vector<std::string>& device_ips,
                              const std::vector<uint16_t>& ports,
                              const int32_t dp_size) {
    return false;
  }

  RateLimiter* get_rate_limiter() { return &rate_limiter_; }

 protected:
  Options options_;
  std::unique_ptr<Engine> engine_;
  RateLimiter rate_limiter_;
};

std::unique_ptr<Master> create_master(const std::string& backend,
                                      const Options& options);

}  // namespace xllm

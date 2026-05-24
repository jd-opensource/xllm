/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include "parallel_layer_loader.h"

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "core/common/global_flags.h"
#include "util/blocking_counter.h"
#include "util/threadpool.h"

namespace xllm::npu::model {

namespace {

using Clock = std::chrono::steady_clock;
using Ms = std::chrono::duration<double, std::milli>;

inline double to_ms(Clock::time_point a, Clock::time_point b) {
  return Ms(b - a).count();
}

// Compose per-layer aggregated stats for logging.
struct LayerStats {
  double total_ms = 0.0;
  double max_ms = 0.0;
  double min_ms = 0.0;
  double avg_ms = 0.0;
};

LayerStats aggregate(const std::vector<double>& xs) {
  LayerStats s;
  if (xs.empty()) {
    return s;
  }
  s.total_ms = 0.0;
  s.max_ms = xs.front();
  s.min_ms = xs.front();
  for (double x : xs) {
    s.total_ms += x;
    s.max_ms = std::max(s.max_ms, x);
    s.min_ms = std::min(s.min_ms, x);
  }
  s.avg_ms = s.total_ms / static_cast<double>(xs.size());
  return s;
}

// `num_cores` derived from std::thread::hardware_concurrency, with a sane
// fallback. We intentionally divide by 4 in auto mode so the loader does
// not monopolise the box during model startup on shared hosts.
int hardware_concurrency_safe() {
  unsigned n = std::thread::hardware_concurrency();
  if (n == 0) {
    return 8;  // conservative fallback
  }
  return static_cast<int>(n);
}

}  // namespace

int resolve_load_parallelism(int num_layers) {
  const int requested = FLAGS_weight_load_parallelism;
  if (requested == -1) {
    return 0;  // explicit serial
  }
  if (num_layers <= 0) {
    return 0;
  }
  int parallelism = 0;
  if (requested == 0) {
    const int by_cores = hardware_concurrency_safe() / 4;
    const int by_layers = num_layers / 4;
    parallelism = std::min({by_cores, by_layers, 8});
  } else {
    parallelism = requested;
  }
  parallelism = std::max(parallelism, 1);
  parallelism = std::min(parallelism, num_layers);
  parallelism = std::min(parallelism, 32);
  return parallelism;
}

bool should_run_parallel(int num_layers,
                         int parallelism,
                         const std::function<bool(int)>& per_layer_supports) {
  if (parallelism <= 0 || num_layers <= 0) {
    return false;
  }
  for (int i = 0; i < num_layers; ++i) {
    if (!per_layer_supports(i)) {
      return false;
    }
  }
  return true;
}

void parallel_run_per_layer(int num_layers,
                            int parallelism,
                            const std::function<void(int)>& task,
                            const std::string& tag) {
  if (num_layers <= 0) {
    return;
  }

  const auto t_start = Clock::now();
  std::vector<double> per_layer(num_layers, 0.0);

  if (parallelism <= 0) {
    LOG(INFO) << "[weight_load][" << tag
              << "] dispatch path=serial num_layers=" << num_layers;
    for (int i = 0; i < num_layers; ++i) {
      const auto t0 = Clock::now();
      task(i);
      per_layer[i] = to_ms(t0, Clock::now());
    }
  } else {
    LOG(INFO) << "[weight_load][" << tag
              << "] dispatch path=parallel num_layers=" << num_layers
              << " parallelism=" << parallelism
              << " gflag=" << FLAGS_weight_load_parallelism;
    const size_t pool_size = static_cast<size_t>(parallelism);
    auto pool = std::make_unique<xllm::ThreadPool>(pool_size);
    xllm::BlockingCounter counter(num_layers);
    for (int i = 0; i < num_layers; ++i) {
      pool->schedule([i, &task, &per_layer, &counter]() {
        const auto t0 = Clock::now();
        task(i);
        per_layer[i] = to_ms(t0, Clock::now());
        counter.decrement_count();
      });
    }
    counter.wait();
  }

  const auto t_end = Clock::now();
  const double total_ms = to_ms(t_start, t_end);
  const auto stats = aggregate(per_layer);
  LOG(INFO) << "[weight_load][" << tag << "] aggregate total=" << total_ms
            << "ms layer_sum=" << stats.total_ms << "ms max=" << stats.max_ms
            << "ms min=" << stats.min_ms << "ms avg=" << stats.avg_ms
            << "ms effective_parallelism="
            << (total_ms > 0.0 ? stats.total_ms / total_ms : 0.0);
}

void parallel_prepare_serial_finalize(int num_layers,
                                      int parallelism,
                                      const std::function<void(int)>& phase1,
                                      const std::function<void(int)>& phase2,
                                      const std::string& tag) {
  if (num_layers <= 0) {
    return;
  }

  const auto t_total_start = Clock::now();

  if (parallelism <= 0) {
    LOG(INFO) << "[weight_load][" << tag
              << "] dispatch path=serial num_layers=" << num_layers;
    std::vector<double> p1(num_layers, 0.0);
    std::vector<double> p2(num_layers, 0.0);
    for (int i = 0; i < num_layers; ++i) {
      const auto a = Clock::now();
      phase1(i);
      const auto b = Clock::now();
      phase2(i);
      const auto c = Clock::now();
      p1[i] = to_ms(a, b);
      p2[i] = to_ms(b, c);
    }
    const auto t_total_end = Clock::now();
    const double total_ms = to_ms(t_total_start, t_total_end);
    const auto s1 = aggregate(p1);
    const auto s2 = aggregate(p2);
    LOG(INFO) << "[weight_load][" << tag
              << "] phase1_prepare total=" << s1.total_ms
              << "ms max=" << s1.max_ms << "ms min=" << s1.min_ms
              << "ms avg=" << s1.avg_ms;
    LOG(INFO) << "[weight_load][" << tag
              << "] phase2_finalize total=" << s2.total_ms
              << "ms max=" << s2.max_ms << "ms min=" << s2.min_ms
              << "ms avg=" << s2.avg_ms;
    LOG(INFO) << "[weight_load][" << tag << "] aggregate total=" << total_ms
              << "ms effective_parallelism=1.00";
    return;
  }

  LOG(INFO) << "[weight_load][" << tag
            << "] dispatch path=parallel num_layers=" << num_layers
            << " parallelism=" << parallelism
            << " gflag=" << FLAGS_weight_load_parallelism;

  const size_t pool_size = static_cast<size_t>(parallelism);
  auto pool = std::make_unique<xllm::ThreadPool>(pool_size);

  // Per-layer phase 1 ready-flags + condition variable so the main thread
  // can sleep until layer i's prepare result is ready.
  std::mutex mu;
  std::condition_variable cv;
  std::vector<int> ready(num_layers, 0);
  std::vector<double> p1(num_layers, 0.0);
  std::vector<double> p2(num_layers, 0.0);
  std::vector<double> wait_ms(num_layers, 0.0);

  for (int i = 0; i < num_layers; ++i) {
    pool->schedule([i, &phase1, &p1, &mu, &cv, &ready]() {
      const auto a = Clock::now();
      phase1(i);
      const auto b = Clock::now();
      const double dt = to_ms(a, b);
      {
        std::lock_guard<std::mutex> lock(mu);
        p1[i] = dt;
        ready[i] = 1;
      }
      cv.notify_all();
    });
  }

  for (int i = 0; i < num_layers; ++i) {
    const auto wait_start = Clock::now();
    {
      std::unique_lock<std::mutex> lock(mu);
      cv.wait(lock, [&ready, i]() { return ready[i] != 0; });
    }
    const auto phase2_start = Clock::now();
    wait_ms[i] = to_ms(wait_start, phase2_start);
    phase2(i);
    p2[i] = to_ms(phase2_start, Clock::now());
  }

  const auto t_total_end = Clock::now();
  const double total_ms = to_ms(t_total_start, t_total_end);
  const auto s1 = aggregate(p1);
  const auto s2 = aggregate(p2);
  const auto sw = aggregate(wait_ms);

  LOG(INFO) << "[weight_load][" << tag
            << "] phase1_prepare layer_sum=" << s1.total_ms
            << "ms max=" << s1.max_ms << "ms min=" << s1.min_ms
            << "ms avg=" << s1.avg_ms;
  LOG(INFO) << "[weight_load][" << tag
            << "] phase2_finalize layer_sum=" << s2.total_ms
            << "ms max=" << s2.max_ms << "ms min=" << s2.min_ms
            << "ms avg=" << s2.avg_ms;
  LOG(INFO) << "[weight_load][" << tag
            << "] main_wait_for_phase1 total=" << sw.total_ms
            << "ms max=" << sw.max_ms << "ms avg=" << sw.avg_ms;
  LOG(INFO) << "[weight_load][" << tag << "] aggregate total=" << total_ms
            << "ms phase1_layer_sum=" << s1.total_ms
            << "ms phase2_layer_sum=" << s2.total_ms
            << "ms effective_parallelism="
            << (total_ms > 0.0 ? s1.total_ms / total_ms : 0.0);
}

}  // namespace xllm::npu::model

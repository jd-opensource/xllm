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

#include "scheduler/profile/graph_warmup.h"

#include <glog/logging.h>

#include <iomanip>
#include <sstream>

namespace xllm {
namespace {

constexpr int32_t kGraphWarmupBarWidth = 20;

}  // namespace

std::vector<int32_t> graph_warmup_buckets(int32_t max_seqs_per_batch) {
  CHECK_GT(max_seqs_per_batch, 0);

  std::vector<int32_t> buckets;
  const std::vector<int32_t> small_buckets = {1, 2, 4, 8, 16};
  for (int32_t bucket : small_buckets) {
    if (bucket <= max_seqs_per_batch) {
      buckets.push_back(bucket);
    }
  }

  for (int32_t bucket = 32; bucket <= max_seqs_per_batch; bucket += 16) {
    buckets.push_back(bucket);
  }

  if (buckets.back() != max_seqs_per_batch) {
    buckets.push_back(max_seqs_per_batch);
  }

  return buckets;
}

bool skip_graph_bucket(int32_t bucket, int32_t dp_size) {
  CHECK_GT(bucket, 0);
  CHECK_GT(dp_size, 0);
  return bucket < dp_size;
}

std::vector<int32_t> graph_decode_buckets(int32_t max_seqs_per_batch,
                                          int32_t dp_size) {
  std::vector<int32_t> buckets = graph_warmup_buckets(max_seqs_per_batch);
  std::vector<int32_t> decode_buckets;
  decode_buckets.reserve(buckets.size());
  for (int32_t bucket : buckets) {
    if (!skip_graph_bucket(bucket, dp_size)) {
      decode_buckets.push_back(bucket);
    }
  }

  return decode_buckets;
}

std::string graph_warmup_progress(int32_t completed,
                                  int32_t total,
                                  int32_t bucket,
                                  double latency_ms) {
  CHECK_GT(total, 0);
  CHECK_GE(completed, 0);
  CHECK_LE(completed, total);
  CHECK_GT(bucket, 0);
  CHECK_GE(latency_ms, 0.0);

  const int32_t filled = static_cast<int32_t>(
      (static_cast<int64_t>(completed) * kGraphWarmupBarWidth + total / 2) /
      total);

  std::string bar;
  bar.reserve(kGraphWarmupBarWidth);
  bar.append(static_cast<size_t>(filled), '#');
  bar.append(static_cast<size_t>(kGraphWarmupBarWidth - filled), '-');

  const double percent =
      static_cast<double>(completed) * 100.0 / static_cast<double>(total);

  std::ostringstream oss;
  oss << "Graph warmup progress: [" << bar << "] " << completed << "/" << total
      << " " << std::fixed << std::setprecision(1) << percent
      << "%, bucket=" << bucket << ", latency=" << std::setprecision(2)
      << latency_ms << " ms";
  return oss.str();
}

}  // namespace xllm

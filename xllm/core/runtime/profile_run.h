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

#pragma once

#include <cstdint>
#include <vector>

#include "runtime/options.h"

namespace xllm {
struct ModelArgs;
struct ForwardInput;
namespace runtime {

struct PeakMem {
  int64_t alloc_bytes = 0;
  int64_t cache_bytes = 0;
};

struct ProfileMem {
  int64_t total_bytes = 0;
  int64_t weight_bytes = 0;
  int64_t runtime_peak_bytes = 0;
  int64_t tmp_kv_bytes = 0;
  int64_t free_bytes = 0;
  bool ok = false;
};

struct ProfilePlan {
  int32_t num_tokens = 0;
  int32_t num_seqs = 0;
  std::vector<int32_t> seq_lens;
  std::vector<std::vector<int32_t>> block_tables;
  int64_t tmp_kv_bytes = 0;
};

bool use_profile_run(const Options& opt, bool is_mlu_build);

int32_t pick_profile_tokens(const Options& opt);

int64_t calc_runtime_peak(const PeakMem& base, const PeakMem& peak);

int64_t calc_safe_kv_bytes(const ProfileMem& mem, const Options& opt);

int64_t calc_safe_kv_bytes(const std::vector<ProfileMem>& worker_mems,
                           const Options& opt);

ProfilePlan build_profile_plan(const ModelArgs& args,
                               const Options& opt,
                               int32_t block_size,
                               bool is_mla,
                               bool is_mlu_build);

ForwardInput build_profile_input(const ModelArgs& args,
                                 const Options& opt,
                                 const ProfilePlan& plan,
                                 bool is_mlu_build);

}  // namespace runtime
}  // namespace xllm

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

namespace xllm {

std::vector<int32_t> graph_warmup_buckets(int32_t max_seqs_per_batch);

bool skip_graph_bucket(int32_t bucket, int32_t dp_size);

std::vector<int32_t> graph_decode_buckets(int32_t max_seqs_per_batch,
                                          int32_t dp_size);

}  // namespace xllm

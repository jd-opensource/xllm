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

#include <torch/torch.h>

#include <optional>
#include <string>
#include <vector>

#include "core/util/hash_util.h"

namespace xllm {

struct MMItemState {
  struct TokenPos {
    int32_t offset = 0;
    int32_t length = 0;
  };

  struct ScheduleData {
    Murmur3Key key;
    int32_t start_pos = 0;
    int32_t end_pos = -1;
  };

  const TokenPos& token_pos() const { return token_pos_; }
  TokenPos& mutable_token_pos() { return token_pos_; }

  const torch::Tensor& mm_token_mask() const { return mm_token_mask_; }
  torch::Tensor& mutable_mm_token_mask() { return mm_token_mask_; }

  const ScheduleData& schedule_data() const { return schedule_data_; }
  ScheduleData& mutable_schedule_data() { return schedule_data_; }

  bool is_scheduling() const;

 private:
  TokenPos token_pos_;
  torch::Tensor mm_token_mask_;
  ScheduleData schedule_data_;
};

}  // namespace xllm

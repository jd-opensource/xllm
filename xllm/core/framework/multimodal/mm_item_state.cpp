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

#include "mm_item_state.h"

namespace xllm {

bool MMItemState::prefix_cached() const {
  return prefix_cache_.cached_token_num > 0;
}

bool MMItemState::prefix_complete_cached() const {
  if (!prefix_cached()) {
    return false;
  }

  return prefix_cache_.cached_token_num == token_pos_.length;
}

}  // namespace xllm

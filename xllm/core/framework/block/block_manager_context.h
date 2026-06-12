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

#include "framework/block/cache_group.h"
#include "util/slice.h"

namespace xllm {

class Sequence;
class KVCacheState;
class PrefixHashState;

// Explicit binding between sequence metadata and the target block state of
// one manager call. A Sequence holds both device and host block states, so
// the callee must never pick kv_state()/host_kv_state() implicitly.
struct BlockManagerContext {
  Sequence* sequence = nullptr;
  // Device context points to sequence->kv_state(), host context to
  // sequence->host_kv_state(). deallocate(context) releases only this state.
  KVCacheState* kv_state = nullptr;
  CacheStorageRole role = CacheStorageRole::DEVICE;

  // Chosen only by the device BlockManagerPool. The host pool dispatches with
  // the same rank and must not select a DP rank on its own.
  int32_t device_dp_rank = -1;

  // Inputs the composite needs to insert completed blocks into its prefix
  // caches from inside allocate/deallocate. The committed-token boundary is
  // read internally from kv_state->kv_cache_tokens_num(); `tokens` is the
  // sequence's full token view and `hash_state` its incremental prefix-hash
  // chain. CONTRACT: a call with hash_state == nullptr or empty tokens
  // performs no prefix-cache insert -- this is how the preempt path
  // (deallocate_without_cache) releases blocks without caching them.
  Slice<int32_t> tokens;
  PrefixHashState* hash_state = nullptr;
};

}  // namespace xllm

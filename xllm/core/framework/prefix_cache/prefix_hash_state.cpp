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

#include "prefix_hash_state.h"

#include <glog/logging.h>

#include <memory>

namespace xllm {

PrefixHashState::PrefixHashState(BlockHasherType hasher_type)
    : hasher_type_(hasher_type) {}

size_t PrefixHashState::group_index(PrefixCacheGroup group) {
  switch (group) {
    case PrefixCacheGroup::C1:
      return 0;
    case PrefixCacheGroup::SINGLE:
      return 1;
    case PrefixCacheGroup::C4:
      return 2;
    case PrefixCacheGroup::C128:
      return 3;
    default:
      LOG(FATAL) << "PrefixHashState: invalid prefix cache group "
                 << static_cast<int>(
                        static_cast<PrefixCacheGroup::Value>(group));
      return 0;
  }
}

PrefixHashState::GroupChain& PrefixHashState::chain_for(PrefixCacheGroup group,
                                                        uint32_t stride) {
  CHECK_GT(stride, 0u) << "PrefixHashState: stride must be positive";
  GroupChain& chain = chains_[group_index(group)];
  if (chain.stride == 0) {
    chain.stride = stride;
  } else {
    CHECK_EQ(chain.stride, stride)
        << "PrefixHashState: stride must stay constant for group "
        << group.to_string();
  }
  return chain;
}

void PrefixHashState::ensure(PrefixCacheGroup group,
                             uint32_t stride,
                             const Slice<int32_t>& tokens,
                             size_t token_end,
                             const MMData& mm_data) {
  GroupChain& chain = chain_for(group, stride);

  const size_t covered_blocks = chain.keys.size();
  const size_t covered_end = covered_blocks * stride;

  // Only whole strides that both fall within `token_end` and are actually
  // backed by `tokens` can be hashed; everything else is left uncovered.
  const size_t aligned_target = (token_end / stride) * stride;
  const size_t available = (tokens.size() / stride) * stride;
  const size_t limit = std::min(aligned_target, available);
  if (covered_end >= limit) {
    return;
  }

  // Resume the chain at `covered_end`; the MM hasher recovers its item cursor
  // from this start index, the TEXT hasher ignores it.
  auto hasher = BlockHasher::create(
      hasher_type_, mm_data, static_cast<int32_t>(covered_end));

  // `running` carries the previous block key forward; aliasing it as both the
  // pre-hash input and the output buffer is safe (xxh3_128bits_hash copies the
  // pre-hash before writing) and matches PrefixCache::match / insert exactly.
  XXH3Key running = covered_blocks == 0 ? XXH3Key{} : chain.keys.back();

  chain.keys.reserve(limit / stride);
  for (size_t i = covered_end; i < limit; i += stride) {
    const uint8_t* pre_hash_value = (i == 0) ? nullptr : running.data;
    hasher->compute(tokens,
                    static_cast<int32_t>(i),
                    static_cast<int32_t>(i + stride),
                    pre_hash_value,
                    running);
    chain.keys.push_back(running);
  }
}

const XXH3Key& PrefixHashState::key(PrefixCacheGroup group,
                                    size_t token_end) const {
  const GroupChain& chain = chains_[group_index(group)];
  CHECK_GT(chain.stride, 0u)
      << "PrefixHashState: key() before any ensure() for group "
      << group.to_string();
  CHECK_EQ(token_end % chain.stride, 0u)
      << "PrefixHashState: token_end must align to stride";
  const size_t idx = token_end / chain.stride;
  CHECK_GE(idx, 1u) << "PrefixHashState: token_end must be a positive multiple "
                       "of stride";
  CHECK_LE(idx, chain.keys.size())
      << "PrefixHashState: token_end not covered; call ensure() first";
  return chain.keys[idx - 1];
}

bool PrefixHashState::covers(PrefixCacheGroup group, size_t token_end) const {
  if (token_end == 0) {
    return true;
  }
  const GroupChain& chain = chains_[group_index(group)];
  if (chain.stride == 0) {
    return false;
  }
  return token_end % chain.stride == 0 &&
         token_end / chain.stride <= chain.keys.size();
}

size_t PrefixHashState::covered_tokens(PrefixCacheGroup group) const {
  const GroupChain& chain = chains_[group_index(group)];
  return chain.keys.size() * static_cast<size_t>(chain.stride);
}

void PrefixHashState::truncate(size_t token_pos) {
  for (GroupChain& chain : chains_) {
    if (chain.stride == 0) {
      continue;
    }
    const size_t keep = token_pos / chain.stride;
    if (keep < chain.keys.size()) {
      chain.keys.resize(keep);
    }
  }
}

void PrefixHashState::reset() {
  for (GroupChain& chain : chains_) {
    chain = GroupChain{};
  }
}

}  // namespace xllm

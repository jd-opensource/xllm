#pragma once
#include <string>

#include "prefix_cache.h"

namespace xllm {

std::unique_ptr<PrefixCache> create_prefix_cache(
    const int32_t block_size,
    const bool& enable_cache_upload = false);

}  // namespace xllm

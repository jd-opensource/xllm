#include <absl/strings/numbers.h>
#include <absl/strings/str_split.h>

#include "prefix_cache.h"
#include "prefix_cache_with_upload.h"

namespace xllm {

std::unique_ptr<PrefixCache> create_prefix_cache(
    int32_t block_size,
    const bool& enable_cache_upload) {
  if (enable_cache_upload) {
    return std::make_unique<PrefixCacheWithUpload>(block_size);
  }
  return std::make_unique<PrefixCache>(block_size);
}

}  // namespace xllm

#include "prefix_cache_factory.h"

#include <absl/strings/numbers.h>
#include <absl/strings/str_split.h>

#include "mm_prefix_cache.h"
#include "prefix_cache_with_upload.h"

namespace xllm {

std::unique_ptr<PrefixCache> create_prefix_cache(PrefixCache::Options options) {
  int32_t block_size = options.block_size();
  if (options.enable_cache_upload()) {
    return std::make_unique<PrefixCacheWithUpload>(block_size);
  } else if (options.enable_mm_prefix_cache()) {
    return std::make_unique<MMPrefixCache>(block_size);
  }
  return std::make_unique<PrefixCache>(block_size);
}

}  // namespace xllm

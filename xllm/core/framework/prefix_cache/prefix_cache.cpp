#include "prefix_cache.h"

#include <absl/strings/numbers.h>
#include <absl/strings/str_split.h>

#include "prefix_cache_hash.h"

namespace xllm {

std::unique_ptr<PrefixCache> CreatePrefixCachePolicy(
    int32_t block_size,
    const std::string& policy,
    const bool& enbale_service_routing) {
  std::vector<absl::string_view> subs = absl::StrSplit(policy, ':');
  CHECK(subs.size() > 0) << " Prefix cache, input param invalid."
                         << " policy:" << policy;

  return std::make_unique<PrefixCacheHash>(block_size);
}

}  // namespace xllm

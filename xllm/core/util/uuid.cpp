#include "uuid.h"

#include <absl/random/distributions.h>

namespace xllm {

std::string ShortUUID::random(size_t len) {
  if (len == 0) {
    len = 22;
  }

  std::string uuid(len, ' ');
  for (size_t i = 0; i < len; i++) {
    const size_t rand = absl::Uniform<size_t>(
        absl::IntervalClosedOpen, gen_, 0, alphabet_.size());
    uuid[i] = alphabet_[rand];
  }
  return uuid;
}

}  // namespace xllm
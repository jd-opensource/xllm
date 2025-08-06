#pragma once
#include <absl/random/random.h>

#include <string>

namespace xllm {

class ShortUUID {
 public:
  ShortUUID() = default;

  std::string random(size_t len = 0);

 private:
  std::string alphabet_ =
      "23456789ABCDEFGHJKLMNPQRSTUVWXYZ"
      "abcdefghijkmnopqrstuvwxyz";
  absl::BitGen gen_;
};

std::string generate_uuid(size_t len = 22);

std::string generate_uuid(size_t len = 22);

}  // namespace xllm
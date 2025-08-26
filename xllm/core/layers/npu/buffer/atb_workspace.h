#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <memory>

#include "atb_buffer.h"

namespace xllm {

class AtbWorkspace {
 public:
  AtbWorkspace() = default;

  AtbWorkspace(at::Device device);

  ~AtbWorkspace();

  AtbWorkspace(const AtbWorkspace&) = delete;

  AtbWorkspace& operator=(const AtbWorkspace&) = delete;

  AtbWorkspace(AtbWorkspace&&) = default;

  AtbWorkspace& operator=(AtbWorkspace&&) = default;

  void* get_workspace_buffer(uint64_t bufferSize);

 private:
  static std::map<int32_t, std::unique_ptr<AtbBuffer>> buffer_map_;
};

}  // namespace xllm

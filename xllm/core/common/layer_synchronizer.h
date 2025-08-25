#pragma once

#if defined(USE_NPU)
#include <acl/acl.h>
#endif

#include <atomic>
#include <vector>

namespace xllm {

#if defined(USE_NPU)
class NPULayerSynchronizerImpl {
 public:
  NPULayerSynchronizerImpl(const int64_t num_layers);
  virtual ~NPULayerSynchronizerImpl();

  aclrtEvent* get_event(const int64_t layer_index);
  std::atomic<bool>* get_event_flag(const int64_t layer_index);
  bool synchronize_layer(const int64_t layer_index);

 private:
  std::vector<aclrtEvent> events_;
  std::vector<std::atomic<bool>> event_record_flags_;
};
#endif
}  // namespace xllm

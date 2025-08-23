#include "common/layer_synchronizer.h"

#include <glog/logging.h>

namespace xllm {

#if defined(USE_NPU)
NPULayerSynchronizerImpl::NPULayerSynchronizerImpl(const int64_t num_layers)
    : events_(num_layers, nullptr), event_record_flags_(num_layers) {
  uint32_t flags = ACL_EVENT_SYNC;
  for (int64_t i = 0; i < num_layers; ++i) {
    auto ret = aclrtCreateEventWithFlag(&events_[i], flags);
    CHECK(ret == ACL_SUCCESS) << "Create event failed.";
  }
}

NPULayerSynchronizerImpl::~NPULayerSynchronizerImpl() {
  for (int64_t i = 0; i < events_.size(); ++i) {
    aclrtDestroyEvent(events_[i]);
  }
}

aclrtEvent* NPULayerSynchronizerImpl::get_event(const int64_t layer_index) {
  return &events_[layer_index];
}

std::atomic<bool>* NPULayerSynchronizerImpl::get_event_flag(
    const int64_t layer_index) {
  return &event_record_flags_[layer_index];
}

bool NPULayerSynchronizerImpl::synchronize_layer(const int64_t layer_index) {
  while (!event_record_flags_[layer_index].load(std::memory_order_acquire));
  auto ret = aclrtSynchronizeEvent(events_[layer_index]);
  if (ret != ACL_SUCCESS) {
    LOG(ERROR) << "Synchronize event failed.";
    return false;
  }
  return true;
}
#endif

}  // namespace xllm

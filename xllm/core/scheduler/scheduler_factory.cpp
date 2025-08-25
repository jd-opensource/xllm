#include "scheduler/scheduler_factory.h"

#include "scheduler/chunked_prefill_scheduler.h"
#include "scheduler/continuous_scheduler.h"
#include "scheduler/disagg_pd_scheduler.h"
#include "scheduler/zero_eviction_scheduler.h"

DEFINE_bool(use_zero_evict,
            false,
            "Use ZeroEvictionScheduler but ContinuousScheduler.");
namespace xllm {

std::unique_ptr<ContinuousScheduler> create_continuous_scheduler(
    Engine* engine,
    ContinuousScheduler::Options options) {
  if (options.enable_disagg_pd()) {
    return std::make_unique<DisaggPDScheduler>(engine, options);
  }

  if (options.enable_chunked_prefill()) {
    return std::make_unique<ChunkedPrefillScheduler>(engine, options);
  }

  if (FLAGS_use_zero_evict) {
    return std::make_unique<ZeroEvictionScheduler>(engine, options);
  }

  return std::make_unique<ContinuousScheduler>(engine, options);
}

}  // namespace xllm

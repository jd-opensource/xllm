#include "priority_comparator.h"

#include "glog/logging.h"

namespace xllm {

// implement operator()
bool FCFSComparator::operator()(const std::shared_ptr<Request>& a,
                                const std::shared_ptr<Request>& b) const {
  return a->created_time() > b->created_time();
}

bool StrictPriorityComparator::operator()(
    const std::shared_ptr<Request>& a,
    const std::shared_ptr<Request>& b) const {
  auto priority_a = a->priority();
  auto priority_b = b->priority();
  if (priority_a != priority_b) {
    return priority_a > priority_b;  // HIGH(1) < NORMAL(2) < LOW(3)
  }
  return a->created_time() > b->created_time();
}

bool DeadlineComparator::operator()(const std::shared_ptr<Request>& a,
                                    const std::shared_ptr<Request>& b) const {
  return a->slo_ms() - a->elapsed_seconds() * 1000 >
         b->slo_ms() - b->elapsed_seconds() * 1000;
}

std::function<bool(const std::shared_ptr<Request>&,
                   const std::shared_ptr<Request>&)>
create_comparator(const std::string& priority_strategy) {
  if (priority_strategy == "FCFS") {
    return [](const std::shared_ptr<Request>& a,
              const std::shared_ptr<Request>& b) {
      return FCFSComparator()(a, b);
    };
  } else if (priority_strategy == "priority") {
    return [](const std::shared_ptr<Request>& a,
              const std::shared_ptr<Request>& b) {
      return StrictPriorityComparator()(a, b);
    };
  } else if (priority_strategy == "deadline") {
    return [](const std::shared_ptr<Request>& a,
              const std::shared_ptr<Request>& b) {
      return DeadlineComparator()(a, b);
    };
  } else {
    LOG(FATAL) << "Unknown strategy: " << priority_strategy;
    return nullptr;
  }
}

}  // namespace xllm
/* Copyright 2025 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

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
  int32_t remain_time_a = a->get_remaining_time();
  int32_t remain_time_b = b->get_remaining_time();

  return remain_time_a > remain_time_b;
}

bool DensityComparator::operator()(const std::shared_ptr<Request>& a,
                                   const std::shared_ptr<Request>& b) const {
  auto& sequence_a = a->sequences()[0];
  auto& sequence_b = b->sequences()[0];

  const double epsilon = 1e-9;  // Set an appropriate tolerance value
  double density_a, density_b;

  if (sequence_a->stage() == SequenceStage::DECODE) {
    density_a =
        a->tpot_priority_weight() * 1.0 / sequence_a->estimated_latency();
    density_b =
        b->tpot_priority_weight() * 1.0 / sequence_b->estimated_latency();
  } else {
    density_a =
        a->ttft_priority_weight() * 1.0 / sequence_a->estimated_latency();
    density_b =
        b->ttft_priority_weight() * 1.0 / sequence_b->estimated_latency();
  }
  // Compare using tolerance (epsilon)
  if (std::abs(density_a - density_b) < epsilon) {
    // If densities are very close, use a stable fallback criterion (e.g.,
    // pointer address or creation time)
    return a->created_time() > b->created_time();
  }
  // For sorting, '<' puts smaller first; for priority_queue, '<' puts larger
  // first.
  return density_a < density_b;
}

bool SJFComparator::operator()(const std::shared_ptr<Request>& a,
                               const std::shared_ptr<Request>& b) const {
  auto& sequence_a = a->sequences()[0];
  auto& sequence_b = b->sequences()[0];

  const double epsilon = 1e-9;  // Set an appropriate tolerance value
  double density_a, density_b;

  density_a = 1.0 / sequence_a->estimated_latency();
  density_b = 1.0 / sequence_b->estimated_latency();
  // Compare using tolerance (epsilon)
  if (std::abs(density_a - density_b) < epsilon) {
    // If densities are very close, use a stable fallback criterion (e.g.,
    // pointer address or creation time)
    return a->created_time() > b->created_time();
  }
  // For sorting, '<' puts smaller first; for priority_queue, '<' puts larger
  // first.
  return density_a < density_b;
}

bool DecodeDeadlineComparator::operator()(
    const std::shared_ptr<Request>& a,
    const std::shared_ptr<Request>& b) const {
  auto& sequence_a = a->sequences()[0];
  auto& sequence_b = b->sequences()[0];

  if (sequence_a->stage() == sequence_b->stage()) {
    return DeadlineComparator()(a, b);
  }

  return sequence_a->stage() < sequence_b->stage();
}

bool DecodeDensityComparator::operator()(
    const std::shared_ptr<Request>& a,
    const std::shared_ptr<Request>& b) const {
  auto& sequence_a = a->sequences()[0];
  auto& sequence_b = b->sequences()[0];

  if (sequence_a->stage() == sequence_b->stage()) {
    return DensityComparator()(a, b);
  }

  return sequence_a->stage() < sequence_b->stage();
}

bool UrgencyDensityComparator::operator()(
    const std::shared_ptr<Request>& a,
    const std::shared_ptr<Request>& b) const {
  if (a->urgency() == b->urgency()) {
    if (a->urgency() == Urgency::URGENT) {
      return DensityComparator()(a, b);
    }
    if (a->urgency() == Urgency::NORMAL) {
      return DeadlineComparator()(a, b);
    }
    if (a->urgency() == Urgency::TIMEOUT) {
      return DensityComparator()(a, b);
    }
    return FCFSComparator()(a, b);
  }
  return a->urgency() < b->urgency();
}

bool UrgencyPriorityComparator::operator()(
    const std::shared_ptr<Request>& a,
    const std::shared_ptr<Request>& b) const {
  if (a->urgency() == b->urgency()) {
    if (a->urgency() == Urgency::URGENT) {
      return StrictPriorityComparator()(a, b);
    }
    if (a->urgency() == Urgency::NORMAL) {
      return DeadlineComparator()(a, b);
    }
    if (a->urgency() == Urgency::TIMEOUT) {
      return StrictPriorityComparator()(a, b);
    }
    return FCFSComparator()(a, b);
  }
  return a->urgency() < b->urgency();
}

bool DecodeUrgencyDensityComparator::operator()(
    const std::shared_ptr<Request>& a,
    const std::shared_ptr<Request>& b) const {
  auto& sequence_a = a->sequences()[0];
  auto& sequence_b = b->sequences()[0];

  if (sequence_a->stage() != SequenceStage::DECODE &&
      sequence_b->stage() != SequenceStage::DECODE) {
    return UrgencyDensityComparator()(a, b);
  } else {
    return sequence_a->stage() < sequence_b->stage();
  }
}

// reverse = false for priority_queue comparator (default)
// reverse = true for sorting comparator
std::function<bool(const std::shared_ptr<Request>&,
                   const std::shared_ptr<Request>&)>
create_comparator(const std::string& priority_strategy, bool reverse) {
  if (priority_strategy == "fcfs") {
    return [reverse](const std::shared_ptr<Request>& a,
                     const std::shared_ptr<Request>& b) {
      return FCFSComparator()(a, b) ^ reverse;
    };
  } else if (priority_strategy == "priority") {
    return [reverse](const std::shared_ptr<Request>& a,
                     const std::shared_ptr<Request>& b) {
      return StrictPriorityComparator()(a, b) ^ reverse;
    };
  } else if (priority_strategy == "deadline") {
    return [reverse](const std::shared_ptr<Request>& a,
                     const std::shared_ptr<Request>& b) {
      return DeadlineComparator()(a, b) ^ reverse;
    };
  } else if (priority_strategy == "sjf") {
    return [reverse](const std::shared_ptr<Request>& a,
                     const std::shared_ptr<Request>& b) {
      return SJFComparator()(a, b) ^ reverse;
    };
  } else if (priority_strategy == "decode_density") {
    return [reverse](const std::shared_ptr<Request>& a,
                     const std::shared_ptr<Request>& b) {
      return DecodeDensityComparator()(a, b) ^ reverse;
    };
  } else if (priority_strategy == "density") {
    return [reverse](const std::shared_ptr<Request>& a,
                     const std::shared_ptr<Request>& b) {
      return DensityComparator()(a, b) ^ reverse;
    };
  } else if (priority_strategy == "urgency_density") {
    return [reverse](const std::shared_ptr<Request>& a,
                     const std::shared_ptr<Request>& b) {
      return UrgencyDensityComparator()(a, b) ^ reverse;
    };
  } else if (priority_strategy == "decode_urgency_density") {
    return [reverse](const std::shared_ptr<Request>& a,
                     const std::shared_ptr<Request>& b) {
      return DecodeUrgencyDensityComparator()(a, b) ^ reverse;
    };
  } else if (priority_strategy == "urgency_priority") {
    return [reverse](const std::shared_ptr<Request>& a,
                     const std::shared_ptr<Request>& b) {
      return UrgencyPriorityComparator()(a, b) ^ reverse;
    };
  } else if (priority_strategy == "decode_deadline") {
    return [reverse](const std::shared_ptr<Request>& a,
                     const std::shared_ptr<Request>& b) {
      return DecodeDeadlineComparator()(a, b) ^ reverse;
    };
  } else {
    LOG(FATAL) << "Unknown strategy: " << priority_strategy;
    return nullptr;
  }
}

}  // namespace xllm
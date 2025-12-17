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

// standard FCFS strategy
bool FCFSComparator::operator()(const std::shared_ptr<Request>& a,
                                const std::shared_ptr<Request>& b) const {
  return a->created_time() > b->created_time();
}

// use request priority. if same, use created time
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

// deadline-first strategy
bool DeadlineComparator::operator()(const std::shared_ptr<Request>& a,
                                    const std::shared_ptr<Request>& b) const {
  int32_t remain_time_a = a->get_remaining_time();
  int32_t remain_time_b = b->get_remaining_time();

  return remain_time_a > remain_time_b;
}

// density-first strategy. denisty = weight / latency
bool DensityComparator::operator()(const std::shared_ptr<Request>& a,
                                   const std::shared_ptr<Request>& b) const {
  auto& sequence_a = a->sequences()[0];
  auto& sequence_b = b->sequences()[0];

  const double epsilon =
      std::numeric_limits<double>::epsilon();  // Set an appropriate tolerance
                                               // value
  double density_a, density_b;

  if (sequence_a->stage() == SequenceStage::DECODE) {
    density_a = static_cast<double>(a->tpot_priority_weight()) /
                sequence_a->estimated_latency();
    density_b = static_cast<double>(b->tpot_priority_weight()) /
                sequence_b->estimated_latency();
  } else {
    density_a = static_cast<double>(a->ttft_priority_weight()) /
                sequence_a->estimated_latency();
    density_b = static_cast<double>(b->ttft_priority_weight()) /
                sequence_b->estimated_latency();
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

// shortest-job-first
bool SJFComparator::operator()(const std::shared_ptr<Request>& a,
                               const std::shared_ptr<Request>& b) const {
  auto& sequence_a = a->sequences()[0];
  auto& sequence_b = b->sequences()[0];

  const double epsilon =
      std::numeric_limits<double>::epsilon();  // Set an appropriate tolerance
                                               // value
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

// decode-first, then deadline-first
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

// decode-first, then density-first
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

// Sort first by urgency, then sort URGENT requests in
// DensityComparator and sort NORMAL requests in DeadlineComparator.
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

// Sort first by urgency, then sort URGENT requests in
// StrictPriorityComparator and sort NORMAL requests in DeadlineComparator.
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

// decode-first, then use UrgencyDensityComparator.
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

// is_reversed = false for priority_queue comparator (default)
// is_reversed = true for sorting comparator
std::function<bool(const std::shared_ptr<Request>&,
                   const std::shared_ptr<Request>&)>
create_comparator(const std::string& priority_strategy, bool is_reversed) {
  if (priority_strategy == "fcfs") {
    return [is_reversed](const std::shared_ptr<Request>& a,
                         const std::shared_ptr<Request>& b) {
      return FCFSComparator()(a, b) ^ is_reversed;
    };
  } else if (priority_strategy == "priority") {
    return [is_reversed](const std::shared_ptr<Request>& a,
                         const std::shared_ptr<Request>& b) {
      return StrictPriorityComparator()(a, b) ^ is_reversed;
    };
  } else if (priority_strategy == "deadline") {
    return [is_reversed](const std::shared_ptr<Request>& a,
                         const std::shared_ptr<Request>& b) {
      return DeadlineComparator()(a, b) ^ is_reversed;
    };
  } else if (priority_strategy == "sjf") {
    return [is_reversed](const std::shared_ptr<Request>& a,
                         const std::shared_ptr<Request>& b) {
      return SJFComparator()(a, b) ^ is_reversed;
    };
  } else if (priority_strategy == "decode_density") {
    return [is_reversed](const std::shared_ptr<Request>& a,
                         const std::shared_ptr<Request>& b) {
      return DecodeDensityComparator()(a, b) ^ is_reversed;
    };
  } else if (priority_strategy == "density") {
    return [is_reversed](const std::shared_ptr<Request>& a,
                         const std::shared_ptr<Request>& b) {
      return DensityComparator()(a, b) ^ is_reversed;
    };
  } else if (priority_strategy == "urgency_density") {
    return [is_reversed](const std::shared_ptr<Request>& a,
                         const std::shared_ptr<Request>& b) {
      return UrgencyDensityComparator()(a, b) ^ is_reversed;
    };
  } else if (priority_strategy == "decode_urgency_density") {
    return [is_reversed](const std::shared_ptr<Request>& a,
                         const std::shared_ptr<Request>& b) {
      return DecodeUrgencyDensityComparator()(a, b) ^ is_reversed;
    };
  } else if (priority_strategy == "urgency_priority") {
    return [is_reversed](const std::shared_ptr<Request>& a,
                         const std::shared_ptr<Request>& b) {
      return UrgencyPriorityComparator()(a, b) ^ is_reversed;
    };
  } else if (priority_strategy == "decode_deadline") {
    return [is_reversed](const std::shared_ptr<Request>& a,
                         const std::shared_ptr<Request>& b) {
      return DecodeDeadlineComparator()(a, b) ^ is_reversed;
    };
  } else {
    LOG(FATAL) << "Unknown strategy: " << priority_strategy;
    return nullptr;
  }
}

}  // namespace xllm
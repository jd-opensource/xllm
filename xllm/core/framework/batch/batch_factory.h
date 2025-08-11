#pragma once

#include "common/metrics.h"
#include "framework/batch/batch.h"

namespace xllm {

class BatchFactory {
 public:
  static BatchFactory* get_instance(int32_t dp_size) {
    static BatchFactory instance(dp_size);
    return &instance;
  }

  std::vector<Batch> create_batches(
      const std::vector<Sequence*>& running_sequences,
      const std::vector<size_t>& running_sequences_budgets,
      const std::vector<Sequence*>& copy_out_sequences = {});

 private:
  BatchFactory(int32_t dp_size) : dp_size_(dp_size) {}
  ~BatchFactory() = default;

  DISALLOW_COPY_AND_ASSIGN(BatchFactory);

 private:
  int32_t dp_size_;
};
}  // namespace xllm

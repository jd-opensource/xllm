#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <memory>

#include "common/macros.h"
#include "framework/model/causal_lm.h"
#include "framework/model/model_input_params.h"
#include "runtime/forward_params.h"

namespace xllm {

class EplbExecutor final {
 public:
  using Callback = std::function<void(int32_t)>;
  EplbExecutor(CausalLM* model);

  virtual ~EplbExecutor();
  void reset_ready_layer_id();
  int32_t get_ready_layer_id() const;
  void eplb_execute(const EplbInfo& eplb_info);

 private:
  struct Task {
    int32_t layer_id;
    std::vector<int32_t> expert_ids;
    Callback callback;
  };

  void eplb_worker_loop();
  void prepare_expert_weight_async(int32_t layer_id,
                                   const std::vector<int32_t>& expert_ids,
                                   Callback callback = nullptr);
  CausalLM* model_;
  std::thread eplb_worker_;
  std::queue<Task> tasks_;
  std::mutex queue_mutex_;
  std::condition_variable condition_;
  bool stop_ = false;

  mutable std::mutex ready_mutex_;
  int32_t ready_layer_id_ = -1;
  struct EplbStream;
  std::unique_ptr<EplbStream> eplb_stream_;
};

}  // namespace xllm

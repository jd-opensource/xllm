#include "eplb_executor.h"

#include <c10/core/Device.h>
#include <c10/core/TensorOptions.h>
#include <glog/logging.h>
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/core/npu/NPUFunctions.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include <torch_npu/torch_npu.h>

#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>

#include "runtime/forward_params.h"

namespace xllm {

struct EplbExecutor::EplbStream {
  c10_npu::NPUStream eplb_stream;
  EplbStream() : eplb_stream(c10_npu::getNPUStreamFromPool()) {}
};
EplbExecutor::EplbExecutor(CausalLM* model)
    : model_(model), eplb_worker_(&EplbExecutor::eplb_worker_loop, this) {
  eplb_stream_ = std::make_unique<EplbStream>();
}

EplbExecutor::~EplbExecutor() {
  {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    stop_ = true;
  }
  condition_.notify_one();
  if (eplb_worker_.joinable()) {
    eplb_worker_.join();
  }
}

void EplbExecutor::eplb_execute(const EplbInfo& eplb_info) {
  if (eplb_info.update_layer_id != -1) {
    model_->update_expert_weight(eplb_info.update_layer_id);
  };
  if (eplb_info.prepare_layer_id != -1) {
    prepare_expert_weight_async(
        eplb_info.prepare_layer_id,
        eplb_info.expert_ids,
        [eplb_info](int32_t id) {
          LOG(INFO) << "prepare expert weight complete, layer: "
                    << eplb_info.prepare_layer_id << std::endl;
        });
  };
}

void EplbExecutor::prepare_expert_weight_async(
    int32_t layer_id,
    const std::vector<int32_t>& expert_ids,
    Callback callback) {
  {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    tasks_.emplace(Task{layer_id, expert_ids, callback});
  }
  condition_.notify_one();
}

int32_t EplbExecutor::get_ready_layer_id() const {
  std::lock_guard<std::mutex> lock(ready_mutex_);
  return ready_layer_id_;
}

void EplbExecutor::reset_ready_layer_id() {
  std::lock_guard<std::mutex> lock(ready_mutex_);
  ready_layer_id_ = -1;
}

void EplbExecutor::eplb_worker_loop() {
  while (true) {
    Task task;
    {
      std::unique_lock<std::mutex> lock(queue_mutex_);
      condition_.wait(lock, [this] { return !tasks_.empty() || stop_; });
      if (stop_) return;
      task = std::move(tasks_.front());
      tasks_.pop();
    }
    auto prepare_start = std::chrono::high_resolution_clock::now();

    c10::StreamGuard streamGuard(eplb_stream_->eplb_stream.unwrap());
    model_->prepare_expert_weight(task.layer_id, task.expert_ids);
    aclrtSynchronizeStream(eplb_stream_->eplb_stream.stream());
    auto prepare_end = std::chrono::high_resolution_clock::now();
    auto prepare_duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(prepare_end -
                                                              prepare_start)
            .count();
    LOG(INFO) << "prepare_expert_weight | layer=" << task.layer_id
              << " | experts=" << task.expert_ids.size()
              << " | duration=" << prepare_duration << "ms";
    {
      std::lock_guard<std::mutex> lock(ready_mutex_);
      ready_layer_id_ = task.layer_id;
    }
    if (task.callback) {
      task.callback(task.layer_id);
    }
  }
}
}  // namespace xllm

// eplb_manager.h
#pragma once

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>

#include "eplb_executor.h"
#include "eplb_policy.h"
#include "framework/model/model_input_params.h"
namespace xllm {

class EplbManager {
 public:
  // Initialize with model dimensions:
  // - layer_num: Total layers in the model
  // - device_num: Parallel devices in cluster
  // - experts_num: Experts per model layer
  EplbManager(int32_t layer_num, int32_t device_num, int32_t experts_num);

  ~EplbManager();

  // Feed new expert workload data for load balancing
  // Input tensors should have shape [layer_num, experts_num]
  void update_expert_load(const std::vector<torch::Tensor> expert_load);

  // Fetch current coordination instructions for expert updates
  // Returns struct containing layer preparation/activation commands
  EplbInfo get_eplb_info();

  // Mark specified layers as prepared (call after async loading completes)
  // expert_layer_ids: Prepared layer IDs per device
  void set_prepared_layer_ids(const std::vector<int32_t>& expert_layer_ids);

 private:
  // Thread functions
  void rebalance_experts_loop();
  void eplb_manager_loop();
  size_t find_next_true(const std::vector<bool>& vec, size_t start_pos);
  // Shared data with mutex protection
  struct ThreadSafeData {
    std::mutex mtx;
    std::condition_variable data_cv;
    std::condition_variable state_cv;
    bool stop = false;

    // Expert load tracking
    torch::Tensor expert_load;
    torch::Tensor expert_distribution;
    std::vector<bool> enable_update_vec;
    std::queue<std::vector<torch::Tensor>> expert_load_queue;

    // Layer state tracking
    int32_t to_be_prepared = -1;
    std::vector<int32_t> prepared_layer_id;
    int32_t ready_layer_id = -1;
    int32_t preparing_layer_id = -1;
  };

  // Components
  std::unique_ptr<EplbPolicy> eplb_policy_ = nullptr;
  ThreadSafeData state_;

  // Constants
  const int32_t layer_num_;
  const int32_t device_num_;
  const int32_t experts_num_;
  const int32_t device_experts_num_;

  // Threads
  std::thread rebalance_thread_;
  std::thread manager_thread_;

  // Internal functions
  void aggregate_multi_layer_expert_loads(
      torch::Tensor& expert_load,
      torch::Tensor& expert_ids_list,
      std::vector<torch::Tensor>& expert_loads_list);
};

}  // namespace xllm
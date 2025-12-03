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

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <sys/wait.h>
#include <torch/torch.h>
#include <unistd.h>

#include <cstring>
#include <memory>
#include <vector>

#include "framework/parallel_state/parallel_args.h"
#include "layers/common/deep_ep.h"
#include "platform/device.h"
#include "tests_utils.h"

#if defined(USE_MLU)
#include "framework/parallel_state/mlu_process_group.h"
#elif defined(USE_CUDA)
#include "framework/parallel_state/cuda_process_group.h"
#endif

namespace xllm {
namespace layer {
namespace test {

// Helper function to create ProcessGroup for multi-device testing
std::unique_ptr<xllm::ProcessGroup> create_test_process_group(
    int rank,
    int world_size,
    int port,
    const std::string& host,
    const torch::Device& device) {
  return xllm::create_process_group(static_cast<int32_t>(rank),
                                    static_cast<int32_t>(world_size),
                                    static_cast<int32_t>(world_size),
                                    static_cast<int32_t>(port),
                                    false,
                                    host,
                                    "deep_ep_test_group",
                                    device);
}

// Test parameters structure to pass to child processes
struct TestParams {
  int32_t rank;
  int32_t world_size;
  int32_t port;
  std::string host;
  int32_t device_index;
  int64_t dispatch_token_size;
  int64_t combine_token_size;
  int64_t max_num_tokens_per_rank;
  int64_t num_global_experts;
  bool test_dispatch;
  bool test_combine;
};

// Child process test function
int run_deep_ep_test_child(const TestParams& params) {
  try {
    // Set device
    torch::Device device(Device::type_torch(), params.device_index);
    xllm::Device xllm_device(device);
    xllm_device.set_device();

    // Create ProcessGroup
    auto process_group = create_test_process_group(
        params.rank, params.world_size, params.port, params.host, device);

    if (!process_group) {
      LOG(ERROR) << "Rank " << params.rank << ": Failed to create ProcessGroup";
      return 1;
    }

    // Create ParallelArgs
    ParallelArgs parallel_args(
        params.rank, params.world_size, process_group.get());
    parallel_args.moe_ep_group_ = process_group.get();

    // Create tensor options
    auto options = torch::TensorOptions()
                       .dtype(torch::kFloat32)
                       .device(device)
                       .requires_grad(false);

    // Create DeepEP instance
    DeepEPImpl deep_ep(params.dispatch_token_size,
                       params.combine_token_size,
                       params.max_num_tokens_per_rank,
                       params.num_global_experts,
                       parallel_args,
                       options);

    LOG(INFO) << "Rank " << params.rank << ": DeepEP created successfully";

    // Test dispatch operation
    if (params.test_dispatch) {
      LOG(INFO) << "Rank " << params.rank << ": Testing dispatch operation";

      // Create test data: each rank sends different tokens
      const int64_t token_num = 4;
      const int64_t token_byte = params.dispatch_token_size;
      const int64_t hidden_dim = token_byte / sizeof(float);

      // Create send_layout: rank 0 sends to rank 1, rank 1 sends to rank 0
      torch::Tensor send_layout = torch::zeros(
          {params.world_size, 2},
          torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
      torch::Tensor recv_layout = torch::zeros(
          {params.world_size, 2},
          torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));

      if (params.rank == 0) {
        // Rank 0 sends all tokens to rank 1
        send_layout[1][0] = 0;          // offset
        send_layout[1][1] = token_num;  // count
        // Rank 0 receives from rank 1
        recv_layout[1][0] = 0;
        recv_layout[1][1] = token_num;
      } else {
        // Rank 1 sends all tokens to rank 0
        send_layout[0][0] = 0;
        send_layout[0][1] = token_num;
        // Rank 1 receives from rank 0
        recv_layout[0][0] = 0;
        recv_layout[0][1] = token_num;
      }

      // Create send_token_num and recv_token_num
      torch::Tensor send_token_num = torch::zeros(
          {params.num_global_experts},
          torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
      torch::Tensor recv_token_num = torch::zeros(
          {params.num_global_experts},
          torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));

      // Create send tokens: each rank sends tokens with its rank value
      torch::Tensor send_token = torch::full(
          {token_num, hidden_dim}, static_cast<float>(params.rank), options);

      // Create recv token buffer
      torch::Tensor recv_token = torch::zeros({token_num, hidden_dim}, options);

      // Move layouts to device
      send_layout = send_layout.to(device);
      recv_layout = recv_layout.to(device);
      send_token_num = send_token_num.to(device);
      recv_token_num = recv_token_num.to(device);

      // Perform dispatch
      deep_ep.dispatch(token_byte,
                       token_num,
                       send_layout,
                       send_token_num,
                       recv_layout,
                       recv_token_num,
                       send_token,
                       recv_token);

      // Synchronize device
      xllm_device.synchronize_default_stream();

      // Verify received tokens: rank 0 should receive rank 1's tokens
      // and vice versa
      float expected_value = static_cast<float>(1 - params.rank);
      torch::Tensor recv_token_cpu = recv_token.to(torch::kCPU);
      auto recv_values = recv_token_cpu.accessor<float, 2>();
      bool all_match = true;
      for (int64_t i = 0; i < token_num; ++i) {
        for (int64_t j = 0; j < hidden_dim; ++j) {
          if (std::abs(recv_values[i][j] - expected_value) > 1e-5) {
            all_match = false;
            LOG(ERROR) << "Rank " << params.rank << ": Mismatch at [" << i
                       << "][" << j << "]: expected=" << expected_value
                       << ", got=" << recv_values[i][j];
          }
        }
      }

      if (!all_match) {
        LOG(ERROR) << "Rank " << params.rank
                   << ": Dispatch test failed - received values don't match";
        return 1;
      }

      LOG(INFO) << "Rank " << params.rank << ": Dispatch test passed";
    }

    // Test combine operation
    if (params.test_combine) {
      LOG(INFO) << "Rank " << params.rank << ": Testing combine operation";

      const int64_t token_num = 4;
      const int64_t token_byte = params.combine_token_size;
      const int64_t hidden_dim = token_byte / sizeof(float);

      // Create send_src_layout and send_dst_layout
      torch::Tensor send_src_layout = torch::zeros(
          {params.world_size, 2},
          torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
      torch::Tensor send_dst_layout = torch::zeros(
          {params.world_size, 2},
          torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));

      if (params.rank == 0) {
        // Rank 0 sends to rank 1
        send_src_layout[1][0] = 0;
        send_src_layout[1][1] = token_num;
        // Rank 0 receives from rank 1
        send_dst_layout[1][0] = 0;
        send_dst_layout[1][1] = token_num;
      } else {
        // Rank 1 sends to rank 0
        send_src_layout[0][0] = 0;
        send_src_layout[0][1] = token_num;
        // Rank 1 receives from rank 0
        send_dst_layout[0][0] = 0;
        send_dst_layout[0][1] = token_num;
      }

      // Create send tokens: each rank sends tokens with its rank value + 10
      torch::Tensor send_token =
          torch::full({token_num, hidden_dim},
                      static_cast<float>(params.rank + 10),
                      options);

      // Create recv token buffer
      torch::Tensor recv_token = torch::zeros({token_num, hidden_dim}, options);

      // Move layouts to device
      send_src_layout = send_src_layout.to(device);
      send_dst_layout = send_dst_layout.to(device);

      // Perform combine
      deep_ep.combine(token_byte,
                      token_num,
                      send_src_layout,
                      send_dst_layout,
                      send_token,
                      recv_token);

      // Synchronize device
      xllm_device.synchronize_default_stream();

      // Verify received tokens
      float expected_value = static_cast<float>(1 - params.rank + 10);
      torch::Tensor recv_token_cpu = recv_token.to(torch::kCPU);
      auto recv_values = recv_token_cpu.accessor<float, 2>();
      bool all_match = true;
      for (int64_t i = 0; i < token_num; ++i) {
        for (int64_t j = 0; j < hidden_dim; ++j) {
          if (std::abs(recv_values[i][j] - expected_value) > 1e-5) {
            all_match = false;
            LOG(ERROR) << "Rank " << params.rank << ": Mismatch at [" << i
                       << "][" << j << "]: expected=" << expected_value
                       << ", got=" << recv_values[i][j];
          }
        }
      }

      if (!all_match) {
        LOG(ERROR) << "Rank " << params.rank
                   << ": Combine test failed - received values don't match";
        return 1;
      }

      LOG(INFO) << "Rank " << params.rank << ": Combine test passed";
    }

    LOG(INFO) << "Rank " << params.rank << ": All tests passed";
    return 0;

  } catch (const std::exception& e) {
    LOG(ERROR) << "Rank " << params.rank << ": Exception: " << e.what();
    return 1;
  }
}

// Multi-process test fixture
class DeepEPMultiDeviceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Initialize test parameters
    world_size_ = 2;
    port_ = 29500;
    host_ = "127.0.0.1";
    dispatch_token_size_ = 128;
    combine_token_size_ = 128;
    max_num_tokens_per_rank_ = 64;
    num_global_experts_ = 2;
  }

  void TearDown() override {
    // Clean up if needed
  }

  // Run test with multiple processes
  void RunMultiProcessTest(bool test_dispatch, bool test_combine) {
    std::vector<pid_t> child_pids;
    std::vector<int> child_statuses(world_size_);

    // Fork child processes
    for (int32_t rank = 0; rank < world_size_; ++rank) {
      pid_t pid = fork();
      if (pid == 0) {
        // Child process
        TestParams params;
        params.rank = rank;
        params.world_size = world_size_;
        params.port = port_;
        params.host = host_;
        params.device_index = rank % Device::device_count();
        params.dispatch_token_size = dispatch_token_size_;
        params.combine_token_size = combine_token_size_;
        params.max_num_tokens_per_rank = max_num_tokens_per_rank_;
        params.num_global_experts = num_global_experts_;
        params.test_dispatch = test_dispatch;
        params.test_combine = test_combine;

        int exit_code = run_deep_ep_test_child(params);
        _exit(exit_code);
      } else if (pid > 0) {
        // Parent process
        child_pids.push_back(pid);
      } else {
        // Fork failed
        LOG(FATAL) << "Failed to fork child process for rank " << rank;
      }
    }

    // Wait for all child processes to complete
    bool all_passed = true;
    for (size_t i = 0; i < child_pids.size(); ++i) {
      int status;
      pid_t waited_pid = waitpid(child_pids[i], &status, 0);
      if (waited_pid == child_pids[i]) {
        if (WIFEXITED(status)) {
          child_statuses[i] = WEXITSTATUS(status);
          if (child_statuses[i] != 0) {
            all_passed = false;
            LOG(ERROR) << "Child process for rank " << i << " exited with code "
                       << child_statuses[i];
          }
        } else {
          all_passed = false;
          LOG(ERROR) << "Child process for rank " << i
                     << " did not exit normally";
        }
      } else {
        all_passed = false;
        LOG(ERROR) << "Failed to wait for child process for rank " << i;
      }
    }

    ASSERT_TRUE(all_passed) << "One or more child processes failed";
  }

  int32_t world_size_;
  int32_t port_;
  std::string host_;
  int64_t dispatch_token_size_;
  int64_t combine_token_size_;
  int64_t max_num_tokens_per_rank_;
  int64_t num_global_experts_;
};

TEST_F(DeepEPMultiDeviceTest, DispatchTest) {
  RunMultiProcessTest(true, false);
}

TEST_F(DeepEPMultiDeviceTest, CombineTest) { RunMultiProcessTest(false, true); }

TEST_F(DeepEPMultiDeviceTest, DispatchAndCombineTest) {
  RunMultiProcessTest(true, true);
}

}  // namespace test
}  // namespace layer
}  // namespace xllm

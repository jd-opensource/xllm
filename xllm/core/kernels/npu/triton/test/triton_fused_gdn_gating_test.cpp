#include <acl/acl.h>
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <torch_npu/torch_npu.h>

#include "../csrc/torch_api/npu_triton_fused_gdn_gating.h"
#include "kernel_loader.h"
#include "test_utils.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"

namespace triton {
namespace test {

constexpr float kTolerance = 1e-3f;
constexpr int32_t kDeviceId = 0;
constexpr int32_t g_num_v_heads = 32;  // qwen3-next

std::pair<torch::Tensor, torch::Tensor> torch_fused_gdn_gating(
    const torch::Tensor& A_log,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& dt_bias,
    float beta = 1.0f,
    float threshold = 20.0f) {
  auto A_log_float = A_log.to(torch::kFloat32);
  auto a_float = a.to(torch::kFloat32);
  auto dt_bias_broadcast = dt_bias.unsqueeze(0).expand_as(a_float);
  auto a_plus_bias = a_float + dt_bias_broadcast;

  auto softplus_nn = torch::nn::Softplus(
      torch::nn::SoftplusOptions().beta(beta).threshold(threshold));
  auto softplus_res = softplus_nn(a_plus_bias);

  auto A_log_exp_broadcast =
      A_log_float.exp().unsqueeze(0).expand_as(softplus_res);

  auto g = -A_log_exp_broadcast * softplus_res;
  g = g.unsqueeze(0);

  auto beta_output = b.sigmoid().unsqueeze(0);

  return std::make_pair(g, beta_output);
}

class TritonFusedGdnGatingTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // 检查 NPU 可用性
    try {
      torch::zeros({1}, torch::TensorOptions().device("npu:0"));
      tensor_options_ =
          torch::TensorOptions().dtype(torch::kFloat16).device("npu:0");
      npu_available_ = true;
      std::cout << "Using NPU device" << std::endl;
    } catch (...) {
      tensor_options_ =
          torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCPU);
      npu_available_ = false;
      std::cout << "Using CPU device (NPU unavailable)." << std::endl;
      return;
    }

    // 固定测试参数
    kernel_name_ = "fused_gdn_gating_head8_kernel";
    binary_filename_ = "fused_gdn_gating_head8_kernel.npubin";

    torch::manual_seed(42);
    torch_npu::init_npu(device_str_);

    binary_path_ = triton::test::GetKernelBinaryPath(binary_filename_);
    std::cout << "kernel binary absolute path is " << binary_path_ << std::endl;
    auto handle = KernelLoader::getKernel(kernel_name_);
    if (!handle.isValid()) {
      handle = KernelLoader::loadKernel(kernel_name_, binary_path_);
    }
    ASSERT_TRUE(handle.isValid()) << "Failed to load Kernel: " << kernel_name_
                                  << " from " << binary_path_;
  }

  void TearDown() override {
    if (npu_available_) {
      try {
        torch_npu::finalize_npu();
      } catch (...) {
        // NPU cleanup failures are usually not critical
      }
    }
  }

  torch::TensorOptions tensor_options_;
  bool npu_available_ = false;
  std::string device_str_ = "npu:" + std::to_string(kDeviceId);
  std::string binary_filename_;
  std::string kernel_name_;
  std::string binary_path_;
};

TEST_F(TritonFusedGdnGatingTest, Head8KernelTest) {
  if (!npu_available_) {
    GTEST_SKIP() << "NPU device not available";
  }

  auto device = at::Device(device_str_);
  constexpr int64_t num_tokens = 3;
  constexpr int64_t tp = 4;
  constexpr int64_t num_heads = g_num_v_heads / tp;
  constexpr float beta = 1.0f;
  constexpr float threshold = 20.0f;

  // Create input Tensor
  auto A_log = torch::randn(
      {num_heads},
      torch::TensorOptions().dtype(torch::kFloat32).device(device));
  auto a = torch::randn(
      {num_tokens, num_heads},
      torch::TensorOptions().dtype(torch::kFloat16).device(device));
  auto b = torch::randn(
      {num_tokens, num_heads},
      torch::TensorOptions().dtype(torch::kFloat16).device(device));
  auto dt_bias = torch::randn(
      {num_heads},
      torch::TensorOptions().dtype(torch::kFloat32).device(device));

  // CPU reference
  auto A_log_cpu = A_log.cpu();
  auto a_cpu = a.cpu();
  auto b_cpu = b.cpu();
  auto dt_bias_cpu = dt_bias.cpu();
  auto [torch_g, torch_beta] = torch_fused_gdn_gating(
      A_log_cpu, a_cpu, b_cpu, dt_bias_cpu, beta, threshold);

  // NPU execution
  auto npu_stream = c10_npu::getCurrentNPUStream(0);
  auto [triton_g, triton_beta] =
      triton::npu::npu_fused_gdn_gating(A_log, a, b, dt_bias, beta, threshold);
  aclrtSynchronizeStream(npu_stream.stream());

  // Compare results
  auto triton_g_cpu = triton_g.cpu();
  auto triton_beta_cpu = triton_beta.cpu();

  // Compare g
  auto g_diff = torch::abs(torch_g - triton_g_cpu);
  float g_max_diff = torch::max(g_diff).item<float>();
  EXPECT_LT(g_max_diff, kTolerance) << "g: max diff (" << g_max_diff
                                    << ") > tolerance (" << kTolerance << ")";

  // Compare beta
  auto beta_diff = torch::abs(torch_beta - triton_beta_cpu);
  float beta_max_diff = torch::max(beta_diff).item<float>();
  EXPECT_LT(beta_max_diff, kTolerance)
      << "beta: max diff (" << beta_max_diff << ") > tolerance (" << kTolerance
      << ")";
}

}  // namespace test
}  // namespace triton

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);

  bool npu_available = false;
  std::string device_str = "npu:" + std::to_string(triton::test::kDeviceId);
  try {
    // 注意：这里必须使用 torch::device (而非字符串 "npu:0")
    auto test_tensor =
        torch::zeros({1}, torch::TensorOptions().device(device_str));
    npu_available = true;
  } catch (...) {
    npu_available = false;
  }

  if (!npu_available) {
    std::cout << "NPU device not available, skipping all tests." << std::endl;
    return 0;  // Exit with success code, all tests skipped
  }

  int result = RUN_ALL_TESTS();

  // 使用标准 exit (避免 _exit 可能导致的清理问题)
  return result;
}

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

#include "tests_utils.h"

#include "core/platform/device.h"

namespace xllm {
namespace layer {
namespace test {

torch::Tensor CreateOnesTensor(const std::vector<int64_t>& shape,
                               const torch::TensorOptions& options) {
  return torch::ones(shape, options);
}

torch::Tensor CreateFullTensor(const std::vector<int64_t>& shape,
                               float value,
                               const torch::TensorOptions& options) {
  return torch::full(shape, value, options);
}

// Supports both 2D and 3D input shapes
torch::Tensor CreateCustomInput(const std::vector<int64_t>& shape,
                                const std::vector<float>& values,
                                const torch::TensorOptions& options) {
  // Only support 2D or 3D
  CHECK(shape.size() == 2 || shape.size() == 3) << "Shape must be 2D or 3D";
  int64_t numel = 1;
  for (auto d : shape) numel *= d;
  CHECK_EQ(values.size(), numel) << "Values size must match tensor size";

  // Create tensor from values directly with given shape
  auto tensor =
      torch::from_blob(const_cast<float*>(values.data()), shape, torch::kFloat)
          .clone()
          .to(options);
  return tensor;
}

torch::Tensor CreateCustomResidual(const std::vector<int64_t>& shape,
                                   const std::vector<float>& values,
                                   const torch::TensorOptions& options) {
  return CreateCustomInput(shape, values, options);
}

void VerifyTensorClose(const torch::Tensor& actual,
                       const torch::Tensor& expected,
                       double rtol,
                       double atol) {
  ASSERT_TRUE(actual.sizes() == expected.sizes())
      << "Tensor shapes don't match: actual=" << actual.sizes()
      << ", expected=" << expected.sizes();

  auto diff = torch::abs(actual - expected);
  auto max_diff = torch::max(diff).item<float>();
  auto mean_diff = torch::mean(diff).item<float>();

  LOG(INFO) << "Max difference: " << max_diff;
  LOG(INFO) << "Mean difference: " << mean_diff;

  ASSERT_TRUE(torch::allclose(actual, expected, rtol, atol))
      << "Tensors are not close enough. Max diff: " << max_diff;
}

void VerifyPrecision(const torch::Tensor& actual_output,
                     const std::vector<float>& expected_values,
                     double rtol,
                     double atol) {
  ASSERT_FALSE(expected_values.empty())
      << "Expected output not set. Call SetExpectedOutput() first.";

  // Support both 2D and 3D outputs
  std::vector<int64_t> output_shape(actual_output.sizes().begin(),
                                    actual_output.sizes().end());
  int64_t numel = actual_output.numel();

  ASSERT_TRUE(output_shape.size() == 2 || output_shape.size() == 3)
      << "Output tensor must be 2D or 3D";
  ASSERT_EQ(expected_values.size(), numel)
      << "Expected output size mismatch: expected " << expected_values.size()
      << ", actual tensor numel " << numel;

  // Create expected tensor from values and shape
  auto expected_tensor =
      CreateCustomInput(output_shape, expected_values, actual_output.options());

  LOG(INFO) << "Verifying precision with rtol=" << rtol << ", atol=" << atol;
  VerifyTensorClose(actual_output, expected_tensor, rtol, atol);
}

ModelArgs CreateDefaultModelArgs() {
  ModelArgs model_args;
  model_args.hidden_size() = 7168;
  model_args.intermediate_size() = 18432;
  model_args.hidden_act() = "silu";
  return model_args;
}

QuantArgs CreateDefaultQuantArgs() {
  QuantArgs quant_args;
  quant_args.quant_method() = "smoothquant";
  quant_args.bits() = 8;
  quant_args.activation_dynamic() = true;
  return quant_args;
}

torch::TensorOptions CreateDefaultTensorOptions() {
  return torch::TensorOptions()
      .dtype(torch::kBFloat16)
      .device(Device::type_torch(), 0)
      .requires_grad(false);
}

ParallelArgs CreateDefaultParallelArgs(
    std::unique_ptr<xllm::ProcessGroup>& mock_process_group) {
  // Create mock ProcessGroup for MLU testing
  mock_process_group = std::make_unique<MockProcessGroup>(
      torch::Device(Device::type_torch(), 0));

  // Initialize ParallelArgs with mock ProcessGroup
  ParallelArgs parallel_args(0, 1, mock_process_group.get());

  // Set tp_group_ for MLU environment
  parallel_args.tp_group_ = mock_process_group.get();

  return parallel_args;
}

}  // namespace test
}  // namespace layer
}  // namespace xllm

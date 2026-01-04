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

// Supports both 2D and 3D input shapes
torch::Tensor create_custom_input(const std::vector<int64_t>& shape,
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

void verify_tensor_close(const torch::Tensor& actual,
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

void verify_precision(const torch::Tensor& actual_output,
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
  auto expected_tensor = create_custom_input(
      output_shape, expected_values, actual_output.options());

  LOG(INFO) << "Verifying precision with rtol=" << rtol << ", atol=" << atol;
  verify_tensor_close(actual_output, expected_tensor, rtol, atol);
}

ModelArgs create_default_model_args() {
  ModelArgs model_args;
  model_args.hidden_size() = 7168;
  model_args.intermediate_size() = 18432;
  model_args.hidden_act() = "silu";
  return model_args;
}

QuantArgs create_default_quant_args() {
  QuantArgs quant_args;
  quant_args.quant_method() = "smoothquant";
  quant_args.bits() = 8;
  quant_args.activation_dynamic() = true;
  return quant_args;
}

ParallelArgs create_default_parallel_args(
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

// helper function for seeded_tensor
// Calculate number of elements from shape
inline int64_t numel_from_shape(torch::IntArrayRef shape) {
  return std::accumulate(
      shape.begin(), shape.end(), (int64_t)1, std::multiplies<int64_t>());
}

// helper function for seeded_tensor
// FNV-1a 64-bit hash function
inline uint64_t fnv1a64(const std::string& s) {
  uint64_t h = 0xcbf29ce484222325ULL;
  for (unsigned char c : s) {
    h ^= c;
    h *= 0x100000001b3ULL;
  }
  return h;
}

// helper struct for seeded_tensor
// SplitMix64 pseudo-random number generator
struct SplitMix64 {
  uint64_t state;
  explicit SplitMix64(uint64_t seed) : state(seed) {}
  inline uint64_t next_u64() {
    state += 0x9E3779B97F4A7C15ULL;
    uint64_t z = state;
    z ^= (z >> 30);
    z *= 0xBF58476D1CE4E5B9ULL;
    z ^= (z >> 27);
    z *= 0x94D049BB133111EBULL;
    z ^= (z >> 31);
    return z;
  }
};

// Generate tensor consistent with Python version
torch::Tensor seeded_tensor(const std::string& key,
                            torch::IntArrayRef shape,
                            torch::ScalarType dtype,
                            torch::Device device) {
  const int64_t N = numel_from_shape(shape);
  // Generate u64 stream
  SplitMix64 rng(fnv1a64(key));
  std::vector<uint64_t> buf;
  buf.reserve(N);
  for (int64_t i = 0; i < N; ++i) buf.push_back(rng.next_u64());

  // Map and build CPU tensor according to dtype
  torch::Tensor out_cpu;

  if (torch::isFloatingType(dtype)) {
    // Floating point: use high 53 bit -> [0,1)
    std::vector<double> vals;
    vals.reserve(N);
    const double inv_2_53 = 1.0 / static_cast<double>(1ULL << 53);
    for (uint64_t u : buf)
      vals.push_back(static_cast<double>(u >> 11) * inv_2_53);
    out_cpu =
        torch::from_blob(
            vals.data(), {N}, torch::TensorOptions().dtype(torch::kDouble))
            .clone()
            .to(dtype);
  } else if (dtype == torch::kBool) {
    std::vector<uint8_t> vals;
    vals.reserve(N);
    for (uint64_t u : buf) vals.push_back(static_cast<uint8_t>(u & 1ULL));
    out_cpu = torch::from_blob(
                  vals.data(), {N}, torch::TensorOptions().dtype(torch::kBool))
                  .clone();
  } else if (torch::isIntegralType(dtype, /*includeBool=*/false)) {
    // Integer: min + (u % span), use __int128 to handle (cover int64)
    auto map_mod_span = [&](auto tag) -> torch::Tensor {
      using T = decltype(tag);
      std::vector<T> vals;
      vals.reserve(N);
      const __int128 minv =
          static_cast<__int128>(std::numeric_limits<T>::min());
      const __int128 maxv =
          static_cast<__int128>(std::numeric_limits<T>::max());
      const unsigned __int128 span =
          static_cast<unsigned __int128>(maxv - minv) + 1U;
      for (uint64_t u : buf) {
        T x = static_cast<T>(
            minv +
            static_cast<__int128>((static_cast<unsigned __int128>(u) % span)));
        vals.push_back(x);
      }
      return torch::from_blob(
                 vals.data(), {N}, torch::TensorOptions().dtype(dtype))
          .clone();
    };

    // handle aliases in a single point for each type.
    switch (dtype) {
      case torch::kUInt8:  // alias for torch::kByte
        out_cpu = map_mod_span(uint8_t{});
        break;
      case torch::kInt8:  // alias for torch::kChar
        out_cpu = map_mod_span(int8_t{});
        break;
      case torch::kInt16:  // alias for torch::kShort
        out_cpu = map_mod_span(int16_t{});
        break;
      case torch::kInt32:  // alias for torch::kInt
        out_cpu = map_mod_span(int32_t{});
        break;
      case torch::kInt64:  // alias for torch::kLong
        out_cpu = map_mod_span(int64_t{});
        break;
      default:
        LOG(FATAL) << "Unsupported integer dtype: " << dtype;
    }
  } else {
    LOG(FATAL) << "Unsupported dtype for seeded_tensor";
  }

  // Shape & device
  out_cpu = out_cpu.view(shape).contiguous();
  if (device != c10::Device(c10::kCPU)) {
    return out_cpu.to(device);
  }
  return out_cpu;
}

}  // namespace test
}  // namespace layer
}  // namespace xllm

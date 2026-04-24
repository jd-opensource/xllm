/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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
#include <torch/torch.h>

#include <cmath>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "framework/state_dict/state_dict.h"
#include "layers/common/rotary_embedding.h"
#include "layers/mlu/deepseek_v4/compressor.h"
#include "platform/device.h"

namespace xllm {
namespace layer {

namespace {

double find_correction_dim(double num_rotations,
                           int64_t dim,
                           double base,
                           int64_t max_seq_len) {
  const double pi = std::acos(-1.0);
  return dim * std::log(max_seq_len / (num_rotations * 2.0 * pi)) /
         (2.0 * std::log(base));
}

std::pair<int64_t, int64_t> find_correction_range(double low_rot,
                                                  double high_rot,
                                                  int64_t dim,
                                                  double base,
                                                  int64_t max_seq_len) {
  const int64_t low = std::max(
      static_cast<int64_t>(
          std::floor(find_correction_dim(low_rot, dim, base, max_seq_len))),
      int64_t{0});
  const int64_t high = std::min(
      static_cast<int64_t>(
          std::ceil(find_correction_dim(high_rot, dim, base, max_seq_len))),
      dim - 1);
  return {low, high};
}

torch::Tensor linear_ramp_factor(int64_t min, int64_t max, int64_t dim) {
  const int64_t max_adj = (min == max) ? max + 1 : max;
  torch::Tensor linear_func =
      (torch::arange(dim, torch::kFloat32) - min) / (max_adj - min);
  return torch::clamp(linear_func, 0.0, 1.0);
}

torch::Tensor precompute_freqs_cis(int64_t dim,
                                   int64_t seqlen,
                                   int64_t original_seq_len,
                                   double base,
                                   double factor,
                                   double beta_fast,
                                   double beta_slow,
                                   torch::Device device,
                                   torch::ScalarType dtype) {
  CHECK(dim % 2 == 0) << "precompute_freqs_cis: dim must be even";

  torch::Tensor freqs =
      1.0 / torch::pow(base, torch::arange(0, dim, 2, torch::kFloat32) / dim);
  if (original_seq_len > 0) {
    auto [low, high] = find_correction_range(
        beta_fast, beta_slow, dim, base, original_seq_len);
    torch::Tensor smooth = 1.0 - linear_ramp_factor(low, high, dim / 2);
    freqs = freqs / factor * (1.0 - smooth) + freqs * smooth;
  }

  torch::Tensor t = torch::arange(seqlen, torch::kFloat32);
  torch::Tensor freqs_outer = torch::outer(t, freqs);
  torch::Tensor ones = torch::ones_like(freqs_outer);
  torch::Tensor freqs_cis = torch::polar(ones, freqs_outer);
  return freqs_cis.to(device, dtype);
}

void expect_tensor_stats(const torch::Tensor& tensor,
                         double expected_min,
                         double expected_max,
                         double expected_sum) {
  EXPECT_TRUE(tensor.defined() && tensor.numel() > 0)
      << "Tensor must be defined and non-empty";
  torch::Tensor flat = tensor.flatten().to(torch::kFloat32);
  if (tensor.device() != torch::kCPU) {
    flat = flat.cpu();
  }

  double actual_min = torch::min(flat).item<double>();
  double actual_max = torch::max(flat).item<double>();
  double actual_sum = torch::sum(flat).item<double>();
  const double rtol = 1e-2;
  const double atol = 1e-5;

  auto within_tol = [rtol, atol](double actual, double expected) {
    double tol = atol + rtol * std::abs(expected);
    return std::abs(actual - expected) <= tol;
  };

  EXPECT_TRUE(within_tol(actual_min, expected_min))
      << "min mismatch: actual=" << actual_min << ", expected=" << expected_min;
  EXPECT_TRUE(within_tol(actual_max, expected_max))
      << "max mismatch: actual=" << actual_max << ", expected=" << expected_max;
  EXPECT_TRUE(within_tol(actual_sum, expected_sum))
      << "sum mismatch: actual=" << actual_sum << ", expected=" << expected_sum;
}

}  // namespace

class CompressorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    options_ = torch::TensorOptions()
                   .dtype(torch::kBFloat16)
                   .device(Device::type_torch(), 0)
                   .requires_grad(false);
    options_fp32_ = torch::TensorOptions()
                        .dtype(torch::kFloat32)
                        .device(Device::type_torch(), 0)
                        .requires_grad(false);
  }

  // Creates a rotary embedding for the Compressor constructor.
  // Note: rotary_emb_ is not used when rotate=False, but we need a valid
  // reference.
  std::shared_ptr<RotaryEmbeddingBase> create_rotary_embedding(
      int64_t rope_head_dim,
      int64_t max_seq_len,
      int64_t rope_theta) {
    return std::make_shared<DeepseekScalingRotaryEmbeddingImpl>(
        /*head_size=*/rope_head_dim,
        /*rotary_dim=*/rope_head_dim,
        /*max_position_embeddings=*/max_seq_len,
        /*rope_scaling_original_max_position_embeddings=*/max_seq_len,
        /*rope_theta=*/rope_theta,
        /*interleaved=*/false,
        /*scaling_factor=*/1.0f,
        /*extrapolation_factor=*/1.0f,
        /*attn_factor=*/1.0f,
        /*beta_fast=*/32.0f,
        /*beta_slow=*/1.0f,
        /*mscale=*/1.0f,
        /*mscale_all_dim=*/0.0f,
        options_);
  }

  Compressor create_compressor(
      int64_t dim,
      int64_t head_dim,
      int64_t rope_head_dim,
      int64_t compress_ratio,
      int64_t cached_state_num,
      double norm_eps,
      bool rotate,
      std::shared_ptr<RotaryEmbeddingBase> rotary_emb) {
    CompressorImpl impl(dim,
                        head_dim,
                        rope_head_dim,
                        compress_ratio,
                        cached_state_num,
                        norm_eps,
                        rotate,
                        rotary_emb,
                        options_);
    return Compressor(impl);
  }

  // Creates weight dictionary for Compressor.
  // Matches the vLLM MLU test initialization:
  //   - ape: [compress_ratio, coeff * head_dim] with ape_val
  //   - wkv.weight: [coeff * head_dim, dim] with wkv_val
  //   - wgate.weight: [coeff * head_dim, dim] with wgate_val
  //   - norm.weight: [head_dim] with norm_val
  std::unordered_map<std::string, torch::Tensor> create_compressor_weights(
      int64_t dim,
      int64_t head_dim,
      int64_t compress_ratio,
      int64_t coeff,
      float ape_val,
      float wkv_val,
      float wgate_val,
      float norm_val) {
    std::unordered_map<std::string, torch::Tensor> weight_dict;
    int64_t internal_dim = coeff * head_dim;

    weight_dict["ape"] =
        torch::full({compress_ratio, internal_dim}, ape_val, options_fp32_);
    weight_dict["wkv.weight"] =
        torch::full({internal_dim, dim}, wkv_val, options_fp32_);
    weight_dict["wgate.weight"] =
        torch::full({internal_dim, dim}, wgate_val, options_fp32_);
    weight_dict["norm.weight"] =
        torch::full({head_dim}, norm_val, options_fp32_);

    LOG(INFO) << "Compressor weights created:";
    LOG(INFO) << "  ape shape: " << weight_dict["ape"].sizes();
    LOG(INFO) << "  wkv.weight shape: " << weight_dict["wkv.weight"].sizes();
    LOG(INFO) << "  wgate.weight shape: "
              << weight_dict["wgate.weight"].sizes();
    LOG(INFO) << "  norm.weight shape: " << weight_dict["norm.weight"].sizes();

    return weight_dict;
  }

  void synchronize_device() {
    xllm::Device device(options_.device());
    device.synchronize_default_stream();
  }

  void verify_first_n_elements(const torch::Tensor& output,
                               const std::vector<float>& expected_values,
                               float rtol = 1e-2) {
    auto output_fp32 = output.flatten().to(torch::kFloat32);
    int64_t n = static_cast<int64_t>(expected_values.size());

    LOG(INFO) << "  First " << n << " elements:";
    for (int64_t i = 0; i < n; ++i) {
      float actual = output_fp32[i].item<float>();
      LOG(INFO) << "    [" << i << "]: " << actual
                << " (expected: " << expected_values[i] << ")";
      EXPECT_NEAR(actual,
                  expected_values[i],
                  std::abs(expected_values[i]) * rtol + 1e-6)
          << "Element [" << i << "] mismatch";
    }
  }

  torch::TensorOptions options_;
  torch::TensorOptions options_fp32_;
};

// Test Compressor forward pass with rotate=False (prefill phase)
// This matches the vLLM MLU test configuration.
TEST_F(CompressorTest, PrefillForwardRotateFalseTest) {
  const int64_t dim = 4096;
  const int64_t head_dim = 512;
  const int64_t rope_head_dim = 128;
  const int64_t compress_ratio = 4;
  const int64_t cached_state_num = 4;
  const double norm_eps = 1e-6;
  const bool rotate = false;
  const int64_t batch_size = 2;
  const int64_t seq_len = 16;
  const int64_t max_seq_len = 8192;
  const int64_t rope_theta = 10000;
  const int64_t num_blocks = 100;
  const int64_t window_offset = 0;
  const int64_t coeff = 2;  // 1 + overlap; overlap = (compress_ratio == 4)

  const float ape_val = 0.1f;
  const float wkv_val = 0.15f;
  const float wgate_val = 0.3f;
  const float norm_val = 0.15f;

  LOG(INFO) << "Testing Compressor forward pass (rotate=False, prefill)";
  LOG(INFO) << "  dim: " << dim << ", head_dim: " << head_dim;
  LOG(INFO) << "  compress_ratio: " << compress_ratio;
  LOG(INFO) << "  batch_size: " << batch_size << ", seq_len: " << seq_len;

  // Create rotary embedding (not used when rotate=False, but required)
  auto rotary_emb =
      create_rotary_embedding(rope_head_dim, max_seq_len, rope_theta);

  // Create and configure Compressor
  auto compressor = create_compressor(dim,
                                      head_dim,
                                      rope_head_dim,
                                      compress_ratio,
                                      cached_state_num,
                                      norm_eps,
                                      rotate,
                                      rotary_emb);

  // Load weights
  auto weight_dict = create_compressor_weights(dim,
                                               head_dim,
                                               compress_ratio,
                                               coeff,
                                               ape_val,
                                               wkv_val,
                                               wgate_val,
                                               norm_val);
  compressor->load_state_dict(StateDict(weight_dict));

  // Precompute freqs_cis for RoPE
  auto freqs_cis = precompute_freqs_cis(rope_head_dim,
                                        max_seq_len,
                                        max_seq_len,
                                        static_cast<double>(rope_theta),
                                        1.0,
                                        32.0,
                                        1.0,
                                        options_.device(),
                                        torch::kComplexFloat);
  LOG(INFO) << "  freqs_cis shape: " << freqs_cis.sizes();

  // Prepare input tensors
  auto x = torch::full({batch_size * seq_len, dim}, 0.5f, options_);
  LOG(INFO) << "  x shape: " << x.sizes() << ", dtype: " << x.dtype();

  auto options_int32 = torch::TensorOptions()
                           .dtype(torch::kInt)
                           .device(Device::type_torch(), 0)
                           .requires_grad(false);

  auto positions = torch::arange(seq_len, options_int32).repeat({batch_size});
  LOG(INFO) << "  positions shape: " << positions.sizes();

  auto seq_lens = torch::full({batch_size}, seq_len, options_int32);
  LOG(INFO) << "  seq_lens: " << seq_lens;

  std::vector<int32_t> q_cu_seq_lens_vec = {
      0, static_cast<int32_t>(seq_len), static_cast<int32_t>(2 * seq_len)};
  auto q_cu_seq_lens = torch::tensor(q_cu_seq_lens_vec, options_int32);
  LOG(INFO) << "  q_cu_seq_lens: " << q_cu_seq_lens;

  auto block_tables = torch::arange(num_blocks, options_int32)
                          .unsqueeze(0)
                          .repeat({batch_size, 1});
  LOG(INFO) << "  block_tables shape: " << block_tables.sizes();

  std::vector<int64_t> batch_to_kv_state = {0, 1};

  auto kv_cache = torch::zeros({num_blocks, 1, 1, head_dim}, options_fp32_)
                      .to(options_.device());
  LOG(INFO) << "  kv_cache shape: " << kv_cache.sizes();

  // Run forward pass
  LOG(INFO) << "Running Compressor forward...";
  auto [compress_kvs, compress_lens] = compressor->forward(x,
                                                           positions,
                                                           block_tables,
                                                           q_cu_seq_lens,
                                                           seq_lens,
                                                           batch_to_kv_state,
                                                           kv_cache,
                                                           window_offset,
                                                           freqs_cis);

  synchronize_device();

  // Verify results
  LOG(INFO) << "Output results:";
  LOG(INFO) << "  compress_kvs count: " << compress_kvs.size();
  LOG(INFO) << "  compress_lens count: " << compress_lens.size();

  CHECK_EQ(compress_kvs.size(), batch_size) << "compress_kvs size mismatch";
  CHECK_EQ(compress_lens.size(), batch_size) << "compress_lens size mismatch";

  // Expected values from vLLM MLU: compress_kvs[i].shape: [4, 1, 512]
  const float expected_sum = 291.3675;
  const float expected_min = -0.211914f;
  const float expected_max = 0.212891f;
  const int64_t expected_compress_len = 4;

  for (int64_t i = 0; i < batch_size; ++i) {
    LOG(INFO) << "Batch " << i << ":";
    LOG(INFO) << "  compress_kvs[" << i
              << "].shape: " << compress_kvs[i].sizes();
    LOG(INFO) << "  compress_lens[" << i << "]: " << compress_lens[i];

    CHECK_EQ(compress_kvs[i].sizes().size(), 3)
        << "compress_kvs should be 3D tensor";
    CHECK_EQ(compress_kvs[i].size(0), expected_compress_len)
        << "compress_kvs first dim should be " << expected_compress_len;
    CHECK_EQ(compress_kvs[i].size(1), 1)
        << "compress_kvs second dim should be 1";
    CHECK_EQ(compress_kvs[i].size(2), head_dim)
        << "compress_kvs third dim should be head_dim";
    CHECK_EQ(compress_lens[i], expected_compress_len)
        << "compress_lens mismatch";

    expect_tensor_stats(
        compress_kvs[i], expected_min, expected_max, expected_sum);

    std::vector<float> expected_first_10(10, 0.15f);
    verify_first_n_elements(compress_kvs[i], expected_first_10);
  }

  // Verify kv_cache was updated
  LOG(INFO) << "Verifying kv_cache:";
  std::vector<float> expected_kv_first_10(10, 0.15f);
  for (int64_t i = 0; i < 3; ++i) {
    auto kv_cache_slice = kv_cache.index({i}).flatten().slice(0, 0, 10);
    LOG(INFO) << "  kv_cache[" << i << "] first 10: " << kv_cache_slice;
    verify_first_n_elements(kv_cache.index({i}), expected_kv_first_10);
  }

  LOG(INFO) << "Compressor prefill forward test (rotate=False) passed";
}

TEST_F(CompressorTest, ForwardKeepsLoadedApeLayout) {
  const int64_t dim = 32;
  const int64_t head_dim = 8;
  const int64_t rope_head_dim = 4;
  const int64_t compress_ratio = 4;
  const int64_t cached_state_num = 1;
  const double norm_eps = 1e-6;
  const bool rotate = false;
  const int64_t max_seq_len = 64;
  const int64_t rope_theta = 10000;
  const int64_t coeff = 2;
  const int64_t internal_dim = coeff * head_dim;

  auto rotary_emb =
      create_rotary_embedding(rope_head_dim, max_seq_len, rope_theta);
  auto compressor = create_compressor(dim,
                                      head_dim,
                                      rope_head_dim,
                                      compress_ratio,
                                      cached_state_num,
                                      norm_eps,
                                      rotate,
                                      rotary_emb);

  torch::Tensor expected_ape =
      torch::arange(compress_ratio * internal_dim, options_fp32_)
          .reshape({compress_ratio, internal_dim});

  std::unordered_map<std::string, torch::Tensor> weight_dict;
  weight_dict["ape"] = expected_ape.clone();
  weight_dict["wkv.weight"] = torch::zeros({internal_dim, dim}, options_fp32_);
  weight_dict["wgate.weight"] =
      torch::zeros({internal_dim, dim}, options_fp32_);
  weight_dict["norm.weight"] = torch::ones({head_dim}, options_fp32_);
  compressor->load_state_dict(StateDict(weight_dict));

  auto freqs_cis = precompute_freqs_cis(rope_head_dim,
                                        max_seq_len,
                                        max_seq_len,
                                        static_cast<double>(rope_theta),
                                        1.0,
                                        32.0,
                                        1.0,
                                        options_.device(),
                                        torch::kComplexFloat);
  auto x = torch::zeros({compress_ratio, dim}, options_);
  auto options_int32 = torch::TensorOptions()
                           .dtype(torch::kInt)
                           .device(Device::type_torch(), 0)
                           .requires_grad(false);
  auto positions = torch::arange(compress_ratio, options_int32);
  auto block_tables = torch::arange(8, options_int32).unsqueeze(0);
  std::vector<int32_t> q_cu_seq_lens_vec = {
      0, static_cast<int32_t>(compress_ratio)};
  auto q_cu_seq_lens = torch::tensor(q_cu_seq_lens_vec, options_int32);
  auto seq_lens = torch::full({1}, compress_ratio, options_int32);
  std::vector<int64_t> batch_to_kv_state = {0};
  auto kv_cache = torch::zeros({8, 1, 1, head_dim}, options_);

  auto [compress_kvs, compress_lens] = compressor->forward(x,
                                                           positions,
                                                           block_tables,
                                                           q_cu_seq_lens,
                                                           seq_lens,
                                                           batch_to_kv_state,
                                                           kv_cache,
                                                           /*window_offset=*/0,
                                                           freqs_cis);
  synchronize_device();

  const auto buffers = compressor->named_buffers(/*recurse=*/false);
  ASSERT_TRUE(buffers.contains("ape"));
  EXPECT_TRUE(torch::allclose(buffers["ape"].cpu(), expected_ape.cpu()))
      << "APE must keep the layout loaded from the model weights";
  ASSERT_EQ(compress_kvs.size(), 1);
  EXPECT_EQ(compress_lens[0], 1);
}

}  // namespace layer
}  // namespace xllm

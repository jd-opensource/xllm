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
#include <numeric>

#include "framework/state_dict/state_dict.h"
#include "layers/mlu/deepseek_v4/indexer_v2.h"
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
    const auto [low, high] = find_correction_range(
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

void expect_int_stats(const torch::Tensor& tensor,
                      int64_t expected_sum,
                      int64_t expected_min,
                      int64_t expected_max) {
  torch::Tensor flat = tensor.flatten().to(torch::kInt64);
  if (tensor.device() != torch::kCPU) {
    flat = flat.cpu();
  }

  const int64_t actual_sum = torch::sum(flat).item<int64_t>();
  const int64_t actual_min = torch::min(flat).item<int64_t>();
  const int64_t actual_max = torch::max(flat).item<int64_t>();

  EXPECT_EQ(actual_sum, expected_sum);
  EXPECT_EQ(actual_min, expected_min);
  EXPECT_EQ(actual_max, expected_max);
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

  const double actual_min = torch::min(flat).item<double>();
  const double actual_max = torch::max(flat).item<double>();
  const double actual_sum = torch::sum(flat).item<double>();
  const double rtol = 1e-2;
  const double atol = 1e-5;

  auto within_tol = [rtol, atol](double actual, double expected) {
    const double tol = atol + rtol * std::abs(expected);
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

class IndexerV2Test : public ::testing::Test {
 protected:
  void SetUp() override {
    torch::Device torch_device(Device::type_torch(), 0);
    Device device(torch_device);
    device.set_seed();

    options_ = torch::TensorOptions()
                   .dtype(torch::kBFloat16)
                   .device(torch_device)
                   .requires_grad(false);
    options_fp32_ = options_.dtype(torch::kFloat32);
    int_options_ = options_.dtype(torch::kInt32);
    process_group_ = std::make_unique<xllm::ProcessGroup>(0, 1, torch_device);
    parallel_args_ = ParallelArgs(0, 1, process_group_.get());
    parallel_args_.tp_group_ = process_group_.get();
    parallel_args_.single_rank_group_ = process_group_.get();
    parallel_args_.sp_group_ = process_group_.get();
  }

  std::shared_ptr<RotaryEmbeddingBase> create_rotary_embedding(
      int64_t head_size,
      int64_t rotary_dim,
      int64_t max_position_embeddings,
      int64_t rope_theta) {
    return std::make_shared<DeepseekScalingRotaryEmbeddingImpl>(
        head_size,
        rotary_dim,
        max_position_embeddings,
        max_position_embeddings,
        rope_theta,
        /*interleaved=*/true,
        /*scaling_factor=*/1.0f,
        /*extrapolation_factor=*/1.0f,
        /*attn_factor=*/1.0f,
        /*beta_fast=*/32.0f,
        /*beta_slow=*/1.0f,
        /*mscale=*/1.0f,
        /*mscale_all_dim=*/1.0f,
        options_);
  }

  IndexerV2 create_indexer_v2(int64_t dim,
                              int64_t index_n_heads,
                              int64_t index_head_dim,
                              int64_t rope_head_dim,
                              int64_t index_topk,
                              int64_t q_lora_rank,
                              int64_t window_size,
                              int64_t compress_ratio,
                              int64_t cached_state_num,
                              double norm_eps,
                              std::shared_ptr<RotaryEmbeddingBase> rotary_emb) {
    return IndexerV2(IndexerV2Impl(dim,
                                   index_n_heads,
                                   index_head_dim,
                                   rope_head_dim,
                                   index_topk,
                                   q_lora_rank,
                                   window_size,
                                   compress_ratio,
                                   cached_state_num,
                                   norm_eps,
                                   rotary_emb,
                                   parallel_args_,
                                   options_));
  }

  std::unordered_map<std::string, torch::Tensor> create_indexer_v2_weights(
      int64_t dim,
      int64_t index_n_heads,
      int64_t index_head_dim,
      int64_t q_lora_rank,
      int64_t compress_ratio) {
    std::unordered_map<std::string, torch::Tensor> weight_dict;
    const int64_t coeff = (compress_ratio == 4) ? 2 : 1;
    const int64_t kv_dim = index_head_dim * coeff;

    weight_dict["wq_b.weight"] = torch::full(
        {index_n_heads * index_head_dim, q_lora_rank}, 0.1f, options_);
    weight_dict["weights_proj.weight"] =
        torch::full({index_n_heads, dim}, 0.2f, options_);
    weight_dict["compressor.ape"] =
        torch::full({compress_ratio, kv_dim}, 0.1f, options_fp32_);
    weight_dict["compressor.wkv.weight"] =
        torch::full({kv_dim, dim}, 0.15f, options_fp32_);
    weight_dict["compressor.wgate.weight"] =
        torch::full({kv_dim, dim}, 0.2f, options_fp32_);
    weight_dict["compressor.norm.weight"] =
        torch::full({index_head_dim}, 0.1f, options_);

    return weight_dict;
  }

  AttentionMetadata create_attention_metadata(int64_t batch_size,
                                              int64_t seq_len,
                                              int64_t num_blocks) {
    AttentionMetadata metadata;

    std::vector<int64_t> cu_seq_lens_vec(batch_size + 1);
    for (int64_t i = 0; i <= batch_size; ++i) {
      cu_seq_lens_vec[i] = i * seq_len;
    }
    metadata.q_cu_seq_lens = torch::tensor(cu_seq_lens_vec, int_options_);
    metadata.q_seq_lens = torch::full({batch_size}, seq_len, int_options_);
    metadata.kv_seq_lens = torch::full({batch_size}, seq_len, int_options_);
    metadata.block_table = torch::arange(num_blocks, int_options_)
                               .unsqueeze(0)
                               .repeat({batch_size, 1});
    metadata.max_query_len = seq_len;
    metadata.max_seq_len = seq_len;
    metadata.is_prefill = true;
    metadata.is_chunked_prefill = false;
    metadata.compute_dtype = "bfloat16";
    return metadata;
  }

  void synchronize_device() {
    xllm::Device device(options_.device());
    device.synchronize_default_stream();
  }

  torch::TensorOptions options_;
  torch::TensorOptions options_fp32_;
  torch::TensorOptions int_options_;
  std::unique_ptr<xllm::ProcessGroup> process_group_;
  ParallelArgs parallel_args_{0, 1, nullptr};
};

TEST_F(IndexerV2Test, ForwardPrefillTest) {
  const int64_t dim = 4096;
  const int64_t index_n_heads = 8;
  const int64_t index_head_dim = 512;
  const int64_t rope_head_dim = 128;
  const int64_t index_topk = 16;
  const int64_t q_lora_rank = 512;
  const int64_t window_size = 4096;
  const int64_t compress_ratio = 4;
  const int64_t cached_state_num = 4;
  const double norm_eps = 1e-6;
  const int64_t max_position_embeddings = 8192;
  const int64_t rope_theta = 10000;

  const int64_t batch_size = 2;
  const int64_t seq_len = 16;
  const int64_t num_tokens = batch_size * seq_len;
  const int64_t num_blocks = 100;

  std::shared_ptr<RotaryEmbeddingBase> rotary_emb = create_rotary_embedding(
      index_head_dim, rope_head_dim, max_position_embeddings, rope_theta);
  IndexerV2 indexer_v2 = create_indexer_v2(dim,
                                           index_n_heads,
                                           index_head_dim,
                                           rope_head_dim,
                                           index_topk,
                                           q_lora_rank,
                                           window_size,
                                           compress_ratio,
                                           cached_state_num,
                                           norm_eps,
                                           rotary_emb);

  std::unordered_map<std::string, torch::Tensor> weight_dict =
      create_indexer_v2_weights(
          dim, index_n_heads, index_head_dim, q_lora_rank, compress_ratio);
  indexer_v2->load_state_dict(StateDict(weight_dict));

  torch::Tensor x = torch::full({num_tokens, dim}, 0.5f, options_);
  torch::Tensor qr = torch::full({num_tokens, q_lora_rank}, 0.3f, options_);
  torch::Tensor positions =
      torch::arange(seq_len, int_options_).repeat({batch_size});
  torch::Tensor offsets =
      torch::tensor({0L, seq_len / compress_ratio}, int_options_);
  AttentionMetadata attn_metadata =
      create_attention_metadata(batch_size, seq_len, num_blocks);
  std::vector<int64_t> batch_to_kv_state = {0, 1};
  torch::Tensor kv_cache =
      torch::zeros({num_blocks, 1, 1, index_head_dim}, options_);
  torch::Tensor freqs_cis =
      precompute_freqs_cis(rope_head_dim,
                           max_position_embeddings,
                           max_position_embeddings,
                           static_cast<double>(rope_theta),
                           1.0,
                           32.0,
                           1.0,
                           options_.device(),
                           torch::kComplexFloat);

  std::vector<torch::Tensor> topk_idxs_list =
      indexer_v2->forward(x,
                          qr,
                          positions,
                          offsets,
                          attn_metadata,
                          batch_to_kv_state,
                          kv_cache,
                          freqs_cis);

  synchronize_device();

  ASSERT_EQ(topk_idxs_list.size(), static_cast<size_t>(batch_size));

  const int64_t expected_k = std::min(index_topk, seq_len / compress_ratio);
  torch::Tensor& topk_idxs_0 = topk_idxs_list[0];
  EXPECT_EQ(topk_idxs_0.size(0), seq_len);
  EXPECT_EQ(topk_idxs_0.size(1), expected_k);
  expect_int_stats(topk_idxs_0,
                   /*expected_sum=*/-14,
                   /*expected_min=*/-1,
                   /*expected_max=*/3);

  torch::Tensor& topk_idxs_1 = topk_idxs_list[1];
  EXPECT_EQ(topk_idxs_1.size(0), seq_len);
  EXPECT_EQ(topk_idxs_1.size(1), expected_k);
  expect_int_stats(topk_idxs_1,
                   /*expected_sum=*/98,
                   /*expected_min=*/-1,
                   /*expected_max=*/7);

  expect_tensor_stats(kv_cache.index({0}).flatten(),
                      /*expected_min=*/0.0,
                      /*expected_max=*/2.265625,
                      /*expected_sum=*/2.265625);
  expect_tensor_stats(kv_cache.index({1}).flatten(),
                      /*expected_min=*/-0.13671875,
                      /*expected_max=*/2.125,
                      /*expected_sum=*/2.265625);
  expect_tensor_stats(kv_cache.index({2}).flatten(),
                      /*expected_min=*/-0.1611328125,
                      /*expected_max=*/2.09375,
                      /*expected_sum=*/2.25);
}

TEST_F(IndexerV2Test, OutputShapeTest) {
  const int64_t dim = 4096;
  const int64_t index_n_heads = 8;
  const int64_t index_head_dim = 512;
  const int64_t rope_head_dim = 128;
  const int64_t index_topk = 16;
  const int64_t q_lora_rank = 512;
  const int64_t window_size = 4096;
  const int64_t compress_ratio = 4;
  const int64_t cached_state_num = 4;
  const double norm_eps = 1e-6;
  const int64_t max_position_embeddings = 8192;
  const int64_t rope_theta = 10000;

  std::vector<std::pair<int64_t, int64_t>> test_configs = {
      {1, 8}, {2, 16}, {4, 32}};

  std::shared_ptr<RotaryEmbeddingBase> rotary_emb = create_rotary_embedding(
      index_head_dim, rope_head_dim, max_position_embeddings, rope_theta);
  IndexerV2 indexer_v2 = create_indexer_v2(dim,
                                           index_n_heads,
                                           index_head_dim,
                                           rope_head_dim,
                                           index_topk,
                                           q_lora_rank,
                                           window_size,
                                           compress_ratio,
                                           cached_state_num,
                                           norm_eps,
                                           rotary_emb);

  std::unordered_map<std::string, torch::Tensor> weight_dict =
      create_indexer_v2_weights(
          dim, index_n_heads, index_head_dim, q_lora_rank, compress_ratio);
  indexer_v2->load_state_dict(StateDict(weight_dict));

  torch::Tensor freqs_cis =
      precompute_freqs_cis(rope_head_dim,
                           max_position_embeddings,
                           max_position_embeddings,
                           static_cast<double>(rope_theta),
                           1.0,
                           32.0,
                           1.0,
                           options_.device(),
                           torch::kComplexFloat);

  for (const auto& [batch_size, seq_len] : test_configs) {
    const int64_t num_tokens = batch_size * seq_len;
    const int64_t num_blocks = 100;

    torch::Tensor x = torch::full({num_tokens, dim}, 0.5f, options_);
    torch::Tensor qr = torch::full({num_tokens, q_lora_rank}, 0.3f, options_);
    torch::Tensor positions =
        torch::arange(seq_len, int_options_).repeat({batch_size});

    std::vector<int64_t> offsets_vec(batch_size);
    for (int64_t i = 0; i < batch_size; ++i) {
      offsets_vec[i] = i * seq_len / compress_ratio;
    }
    torch::Tensor offsets = torch::tensor(offsets_vec, int_options_);

    AttentionMetadata attn_metadata =
        create_attention_metadata(batch_size, seq_len, num_blocks);

    std::vector<int64_t> batch_to_kv_state(batch_size);
    std::iota(batch_to_kv_state.begin(),
              batch_to_kv_state.end(),
              static_cast<int64_t>(0));

    torch::Tensor kv_cache =
        torch::zeros({num_blocks, 1, 1, index_head_dim}, options_);

    std::vector<torch::Tensor> topk_idxs_list =
        indexer_v2->forward(x,
                            qr,
                            positions,
                            offsets,
                            attn_metadata,
                            batch_to_kv_state,
                            kv_cache,
                            freqs_cis);

    synchronize_device();

    ASSERT_EQ(topk_idxs_list.size(), static_cast<size_t>(batch_size));

    const int64_t expected_k = std::min(index_topk, seq_len / compress_ratio);
    for (int64_t i = 0; i < batch_size; ++i) {
      EXPECT_EQ(topk_idxs_list[i].dim(), 2);
      EXPECT_EQ(topk_idxs_list[i].size(0), seq_len);
      EXPECT_EQ(topk_idxs_list[i].size(1), expected_k);
    }
  }
}

}  // namespace layer
}  // namespace xllm

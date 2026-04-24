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

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <memory>

#include "layers/common/rotary_embedding.h"
#include "layers/common/rotary_embedding_util.h"

namespace xllm {
namespace layer {
namespace {

TEST(RotaryEmbeddingTest, InverseCacheFlipsSinOnly) {
  const int64_t rotary_dim = 8;
  const int64_t max_position_embeddings = 16;
  const torch::TensorOptions options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
  torch::Tensor inv_freq =
      rotary::compute_inv_freq(rotary_dim, /*rope_theta=*/10000.0f, options);

  torch::Tensor forward_cache =
      rotary::compute_cos_sin_cache(rotary_dim,
                                    max_position_embeddings,
                                    /*interleaved=*/true,
                                    /*scaling_factor=*/1.0f,
                                    /*attn_factor=*/1.0f,
                                    /*mscale=*/1.0f,
                                    /*mscale_all_dim=*/1.0f,
                                    inv_freq,
                                    options,
                                    /*inverse=*/false);
  torch::Tensor inverse_cache =
      rotary::compute_cos_sin_cache(rotary_dim,
                                    max_position_embeddings,
                                    /*interleaved=*/true,
                                    /*scaling_factor=*/1.0f,
                                    /*attn_factor=*/1.0f,
                                    /*mscale=*/1.0f,
                                    /*mscale_all_dim=*/1.0f,
                                    inv_freq,
                                    options,
                                    /*inverse=*/true);

  std::vector<torch::Tensor> forward_chunks =
      forward_cache.chunk(/*chunks=*/2, /*dim=*/-1);
  std::vector<torch::Tensor> inverse_chunks =
      inverse_cache.chunk(/*chunks=*/2, /*dim=*/-1);
  EXPECT_TRUE(torch::allclose(forward_chunks[0], inverse_chunks[0]));
  EXPECT_TRUE(torch::allclose(forward_chunks[1], -inverse_chunks[1]));
}

TEST(RotaryEmbeddingTest, BasicRopeInverseFlipsSinOnly) {
  const int64_t rotary_dim = 8;
  const int64_t max_position_embeddings = 16;
  const int64_t rope_theta = 10000;
  const torch::TensorOptions options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);

  RotaryEmbeddingImpl forward_rope(rotary_dim,
                                   max_position_embeddings,
                                   rope_theta,
                                   /*interleaved=*/true,
                                   options,
                                   /*inverse=*/false);
  RotaryEmbeddingImpl inverse_rope(rotary_dim,
                                   max_position_embeddings,
                                   rope_theta,
                                   /*interleaved=*/true,
                                   options,
                                   /*inverse=*/true);

  EXPECT_TRUE(torch::allclose(forward_rope.get_cos_cache(),
                              inverse_rope.get_cos_cache()));
  EXPECT_TRUE(torch::allclose(forward_rope.get_sin_cache(),
                              -inverse_rope.get_sin_cache()));
}

TEST(RotaryEmbeddingTest, FactoryCreatesDefaultRope) {
  ModelArgs args;
  args.rope_scaling_rope_type() = "default";
  args.rope_theta() = 10000.0f;
  const torch::TensorOptions options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);

  std::shared_ptr<RotaryEmbeddingBase> rope =
      create_mla_rotary_embedding(args,
                                  /*rotary_dim=*/8,
                                  /*max_position_embeddings=*/16,
                                  /*interleaved=*/true,
                                  options);

  EXPECT_TRUE(std::dynamic_pointer_cast<RotaryEmbeddingImpl>(rope) != nullptr);
  EXPECT_TRUE(std::dynamic_pointer_cast<DeepseekScalingRotaryEmbeddingImpl>(
                  rope) == nullptr);
}

TEST(RotaryEmbeddingTest, FactoryCreatesDeepseekYarnRope) {
  ModelArgs args;
  args.rope_scaling_rope_type() = "deepseek_yarn";
  args.rope_theta() = 10000.0f;
  args.rope_scaling_original_max_position_embeddings() = 16;
  args.rope_scaling_factor() = 1.0f;
  args.rope_extrapolation_factor() = 1.0f;
  args.rope_scaling_attn_factor() = 1.0f;
  args.rope_scaling_beta_fast() = 32;
  args.rope_scaling_beta_slow() = 1;
  args.rope_scaling_mscale() = 1.0f;
  args.rope_scaling_mscale_all_dim() = 1.0f;
  const torch::TensorOptions options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);

  std::shared_ptr<RotaryEmbeddingBase> rope =
      create_mla_rotary_embedding(args,
                                  /*rotary_dim=*/8,
                                  /*max_position_embeddings=*/16,
                                  /*interleaved=*/true,
                                  options);

  EXPECT_TRUE(std::dynamic_pointer_cast<DeepseekScalingRotaryEmbeddingImpl>(
                  rope) != nullptr);
}

TEST(RotaryEmbeddingTest, FactoryDefaultInverseFlipsSinOnly) {
  ModelArgs args;
  args.rope_scaling_rope_type() = "default";
  args.rope_theta() = 10000.0f;
  const torch::TensorOptions options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);

  std::shared_ptr<RotaryEmbeddingBase> forward_rope =
      create_mla_rotary_embedding(args,
                                  /*rotary_dim=*/8,
                                  /*max_position_embeddings=*/16,
                                  /*interleaved=*/true,
                                  options,
                                  /*inverse=*/false);
  std::shared_ptr<RotaryEmbeddingBase> inverse_rope =
      create_mla_rotary_embedding(args,
                                  /*rotary_dim=*/8,
                                  /*max_position_embeddings=*/16,
                                  /*interleaved=*/true,
                                  options,
                                  /*inverse=*/true);

  EXPECT_TRUE(torch::allclose(forward_rope->get_cos_cache(),
                              inverse_rope->get_cos_cache()));
  EXPECT_TRUE(torch::allclose(forward_rope->get_sin_cache(),
                              -inverse_rope->get_sin_cache()));
}

}  // namespace
}  // namespace layer
}  // namespace xllm

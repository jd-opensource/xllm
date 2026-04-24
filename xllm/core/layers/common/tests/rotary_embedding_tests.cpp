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

}  // namespace
}  // namespace layer
}  // namespace xllm

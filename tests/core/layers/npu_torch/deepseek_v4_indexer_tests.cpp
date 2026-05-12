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

#include "framework/model/model_input_params.h"
#include "framework/quant_args.h"
#include "layers/common/dsa_metadata_builder.h"
#include "layers/npu_torch/deepseek_v4_indexer.h"

namespace xllm {
namespace layer {

class DeepseekV4IndexerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    options_ = torch::TensorOptions()
                   .dtype(torch::kFloat32)
                   .device(torch::kCPU)
                   .requires_grad(false);
  }

  torch::TensorOptions options_;
};

TEST_F(DeepseekV4IndexerTest, ConstructorAndMetadataWorks) {
  const int64_t dim = 128;
  const int64_t index_n_heads = 8;
  const int64_t index_head_dim = 16;
  const int64_t rope_head_dim = 8;
  const int64_t index_topk = 32;
  const int64_t q_lora_rank = 32;
  const int64_t compress_ratio = 4;

  QuantArgs quant_args;
  auto indexer = DeepseekV4Indexer(DeepseekV4IndexerImpl(dim,
                                                         index_n_heads,
                                                         index_head_dim,
                                                         rope_head_dim,
                                                         index_topk,
                                                         q_lora_rank,
                                                         compress_ratio,
                                                         /*norm_eps=*/1e-6,
                                                         quant_args,
                                                         options_));

  EXPECT_EQ(indexer->dim(), dim);
  EXPECT_EQ(indexer->n_heads(), index_n_heads);
  EXPECT_EQ(indexer->head_dim(), index_head_dim);
  EXPECT_EQ(indexer->rope_head_dim(), rope_head_dim);
  EXPECT_EQ(indexer->index_topk(), index_topk);
  EXPECT_EQ(indexer->q_lora_rank(), q_lora_rank);
  EXPECT_EQ(indexer->compress_ratio(), compress_ratio);

  EXPECT_TRUE(indexer->wq_b());
  EXPECT_TRUE(indexer->weights_proj());
}

TEST_F(DeepseekV4IndexerTest, DsaTokenSlotsTrackCurrentDecodeStep) {
  ModelInputParams params;
  params.batch_forward_type = BatchForwardType::DECODE;
  params.num_sequences = 2;
  params.kv_seq_lens_vec = {5, 8};
  params.q_seq_lens_vec = {1, 1};
  params.new_cache_slots = torch::tensor({10, 20}, torch::kInt32);
  params.multi_block_tables = {
      torch::tensor({{0}, {1}}, torch::kInt32),
      torch::tensor({{0}, {1}}, torch::kInt32),
  };

  const auto positions = torch::tensor({4, 7}, torch::kInt64);
  const std::vector<DSAGroupInfo> group_infos = {
      {DSACacheType::SLIDING_WINDOW, 1, 128},
      {DSACacheType::TOKEN, 4, 128},
  };
  const std::vector<std::vector<DSACacheInfo>> caches_info = {{
      {1, DSACacheType::TOKEN, 4, 128},
      {0, DSACacheType::SLIDING_WINDOW, 1, 128},
  }};

  auto metadata = DSAMetadataBuilder::build(
      params, positions, torch::Tensor(), caches_info, group_infos);

  ASSERT_TRUE(metadata.dsa_metadata != nullptr);
  const auto& dsa = *metadata.dsa_metadata;
  ASSERT_EQ(dsa.slot_mappings.size(), 1);
  ASSERT_EQ(dsa.slot_mappings[0].size(), 2);

  const auto token_slots = dsa.slot_mappings[0][0];
  const auto expected_slots = torch::tensor({129, 0}, torch::kInt32);
  EXPECT_TRUE(torch::equal(token_slots, expected_slots))
      << "token slots should include only current-step committed compressed "
         "slots plus decode padding";
}

}  // namespace layer
}  // namespace xllm

/* Copyright 2025 The xLLM Authors. All Rights Reserved.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include "batch.h"

#include <absl/time/clock.h>
#include <gtest/gtest.h>

#include <cstdint>

#include "framework/block/block.h"
#include "framework/block/block_manager_impl.h"
#include "framework/model/model_args.h"
#include "framework/request/stopping_checker.h"
#include "framework/sampling/sampling_params.h"

namespace xllm {

template <typename T>
bool equal(const torch::Tensor& t, const std::vector<T>& d) {
  auto flatten_t = t.flatten();
  if (flatten_t.size(0) != d.size()) {
    return false;
  }
  for (int i = 0; i < d.size(); i++) {
    if (flatten_t[i].item<T>() != d[i]) {
      return false;
    }
  }
  return true;
}

TEST(BatchTest, Basic) {
  const uint32_t n_blocks = 20;
  const uint32_t block_size = 4;
  BlockManager::Options options;
  options.num_blocks(n_blocks).block_size(block_size);
  BlockManagerImpl manager(options);

  RequestSamplingParam sampling_param;
  sampling_param.frequency_penalty = 0.1;
  StoppingChecker stopping_checker;
  stopping_checker.set_max_generated_tokens(20);
  const size_t capacity = 100;
  SequenceParams seq_params;
  seq_params.seq_capacity = capacity;
  seq_params.stopping_checker = &stopping_checker;
  seq_params.sampling_param = &sampling_param;
  seq_params.skip_special_tokens = true;
  seq_params.echo = false;
  seq_params.logprobs = false;
  seq_params.enable_schedule_overlap = false;

  torch::Tensor input_embedding;
  MMData mm_data;
  IncrementalDecoder fake_decoder1("", 1, false, false);
  // prepare sequences
  // sequence in prefill phase
  Sequence seq1(/*index=*/0,
                /*token_ids=*/{1, 3, 5, 7, 5, 4, 3, 2, 1},
                input_embedding,
                mm_data,
                std::move(fake_decoder1),
                seq_params);

  seq1.add_kv_blocks(manager.allocate(3));  // [1, 2, 3]

  IncrementalDecoder fake_decoder2("", 2, false, false);
  // seq in decode phase
  Sequence seq2(/*index=*/0,
                /*token_ids=*/{2, 4, 6, 8, 6, 4, 2},
                input_embedding,
                mm_data,
                std::move(fake_decoder2),
                seq_params);
  seq2.add_kv_blocks(manager.allocate(4));  // [4, 5, 6, 7]
  seq2.kv_state().incr_kv_cache_tokens_num(/*size=*/7);
  seq2.append_token(100);

  IncrementalDecoder fake_decoder3("", 3, false, false);
  // seq in decode phase
  Sequence seq3(
      /*index=*/0,
      /*token_ids=*/{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19},
      input_embedding,
      mm_data,
      std::move(fake_decoder3),
      seq_params);
  seq3.add_kv_blocks(manager.allocate(5));  // [8, 9, 10, 11, 12]
  seq3.kv_state().incr_kv_cache_tokens_num(/*size=*/15);
  seq3.append_token(200);

  IncrementalDecoder fake_decoder4("", 4, false, false);
  Sequence seq4(/*index=*/0,
                /*token_ids=*/{1, 3, 5, 7, 5, 4, 3, 2, 1},
                input_embedding,
                mm_data,
                std::move(fake_decoder4),
                seq_params);
  seq4.kv_state().add_kv_blocks(manager.allocate(3));  // [13, 14, 15]
  seq4.kv_state().incr_kv_cache_tokens_num(/*size=*/4);

  // define outputs
  Batch batch({&seq1, &seq2, &seq3});
  // allowed chunk size 4
  batch.add(&seq4, 4);
  ForwardInput forward_input = batch.prepare_forward_input(
      /*num_decoding_tokens=*/1, /*min_decoding_bach_size=*/0, ModelArgs());

  // check num tokens in kv cache
  EXPECT_EQ(seq1.kv_state().kv_cache_tokens_num(), 9);
  EXPECT_EQ(seq2.kv_state().kv_cache_tokens_num(), 8);
  EXPECT_EQ(seq3.kv_state().kv_cache_tokens_num(), 16);
  EXPECT_EQ(seq4.kv_state().kv_cache_tokens_num(), 8);

  // clang-format off
  // check the flatten token ids
  const std::vector<int32_t> expcted_tokens = {
      /*seq1*/ 1, 3, 5, 7, 5, 4, 3, 2, 1, 
      /*seq2*/ 100, 
      /*seq3*/ 200,
      /*seq4*/ 5, 4, 3, 2};
  EXPECT_TRUE(equal(forward_input.token_ids, expcted_tokens));

  // check the flatten positions
  const std::vector<int32_t> expected_pos = {
    /*seq1*/ 0, 1, 2, 3, 4, 5, 6, 7, 8,
    /*seq2*/ 7, 
    /*seq3*/ 15,
    /*seq4*/ 4, 5, 6, 7};
  EXPECT_TRUE(equal(forward_input.positions, expected_pos));

  // check the input parameters
  const ModelInputParams& input_params = forward_input.input_params;
  EXPECT_FALSE(input_params.empty_kv_cache);
  EXPECT_EQ(input_params.num_sequences, 4);
  EXPECT_EQ(input_params.q_max_seq_len, 9);
  EXPECT_EQ(input_params.kv_max_seq_len, 16);

#if defined(USE_NPU)
  const std::vector<int32_t> q_seq_lens = {9, 1, 1, 4};
#else
  const std::vector<int32_t> q_seq_lens = {0, 9, 10, 11, 15};
#endif
  EXPECT_TRUE(equal(input_params.q_seq_lens, q_seq_lens));

//  seq4's kv_seq_len = q_len + num_cached_tokens (q_len<=max_allowed_tokens)
#if defined(USE_NPU)
  const std::vector<int32_t> kv_seq_lens = {9, 8, 16, 8};
#else
  const std::vector<int32_t> kv_seq_lens = {0, 9, 17, 33, 41};
#endif
  EXPECT_TRUE(equal(input_params.kv_seq_lens, kv_seq_lens));

  const std::vector<int32_t> new_cache_slots = {
    /*seq1*/ 4, 5, 6, 7, 8, 9, 10, 11, 12, 
    /*seq2*/ 23, 
    /*seq3*/ 47,
    /*seq4*/ 56,57,58,59
    };
  EXPECT_TRUE(equal(input_params.new_cache_slots, new_cache_slots));

  const std::vector<int32_t> block_tables = {
    /*seq1*/ 1, 2, 3,  0,  0,
    /*seq2*/ 4, 5, 6,  7,  0,
    /*seq3*/ 8, 9, 10, 11, 12,
    /*seq4*/ 13, 14, 15, 0, 0};
  EXPECT_TRUE(equal(input_params.block_tables, block_tables));

  // const std::vector<int32_t> last_token_idxes = {8, 9, 10};
  // EXPECT_TRUE(equal(input_params.last_token_idxes, last_token_idxes));

  const auto& sampling_params = forward_input.sampling_params;
  const std::vector<int64_t> unique_ids = {
    /*seq1*/   2,  4,  7,  5,  3,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    /*seq2*/ 100,  8,  6,  4,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 
    /*seq3*/ 200,  19, 17, 1,  2,  15, 3,  4,  5,  6,  7,  8,  9, 10, 11, 13
    };
  // seq4 has no sampling parameters
  EXPECT_TRUE(equal(sampling_params.unique_token_ids, unique_ids));

  const std::vector<int32_t> unique_counts = {
    /*seq1*/  1,  1,  1,  2,  2,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  
    /*seq2*/  1,  1,  2,  2,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  
    /*seq3*/  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1
  };
  // seq4 has no sampling parameters
  EXPECT_TRUE(equal(sampling_params.unique_token_counts, unique_counts));

  const std::vector<int32_t> token_ids_lens = {6, 5, 16};
  EXPECT_TRUE(equal(sampling_params.unique_token_ids_lens, token_ids_lens));

  // The model's output hidden_states shape is [num_tokens, hidden_size].
  // selected_token_idxes is used to select the required corresponding token's hidden_state from it, perform logits calculation, and obtain the probability that the token belongs to each class.
  // The logits_processor's output logits shape is [num_selected_tokens, vocab_size]. sample_idxes is used to select the required token's corresponding logits from it.
  // Generally, num_selected_tokens equals num_sample_tokens.
  const std::vector<int32_t> expected_selected_token_idxes = {8,9,10};
  // seq4 has no sampling parameters
  EXPECT_EQ(sampling_params.selected_token_idxes.size(0), batch.size()-1);
  EXPECT_TRUE(equal(sampling_params.selected_token_idxes, expected_selected_token_idxes));
  const std::vector<int32_t> expected_sample_idxes = {0,1,2};
  EXPECT_TRUE(equal(sampling_params.sample_idxes, expected_sample_idxes));

  // clang-format on
}

}  // namespace xllm

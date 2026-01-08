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

#include "rejection_sampler.h"

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <torch/types.h>

#include <cstdint>

#include "platform/device.h"
#include "sampler.h"

namespace xllm {

namespace {
// Helper function to get test device: MLU if available, otherwise CPU
torch::Device GetTestDevice() {
  std::string backend = Device::type_str();
  if (backend == "mlu" && Device::device_count() > 0) {
    return torch::Device(Device::type_torch(), 0);
  }
  return torch::Device(torch::kCPU);
}

// Helper function to get test tensor options with automatic device selection
torch::TensorOptions GetTestOptions(torch::ScalarType dtype = torch::kFloat32) {
  return torch::dtype(dtype).device(GetTestDevice());
}
}  // namespace

TEST(RejectionSamplerTest, Basic) {
  // test with hand-crafted example
  const auto options = GetTestOptions(torch::kFloat32);
  const auto device = GetTestDevice();

  // set random seed
  torch::manual_seed(100);

  const auto draft_token_ids =
      torch::tensor({{1, 2, 3}}, options.dtype(torch::kInt64));

  // shape: [1, 3, 5]
  auto draft_probs = torch::tensor({{0.2104, 0.2163, 0.1912, 0.1937, 0.1884},
                                    {0.2100, 0.1803, 0.2398, 0.2088, 0.1610},
                                    {0.1838, 0.2079, 0.2270, 0.2451, 0.1362}},
                                   options)
                         .reshape({1, 3, 5});
  // shape: [1, 3, 5]
  auto target_probs = torch::tensor({{0.1299, 0.2462, 0.1821, 0.1354, 0.3064},
                                     {0.1159, 0.2839, 0.1603, 0.2451, 0.1949},
                                     {0.0002, 0.0433, 0.6629, 0.1469, 0.1467}},
                                    options)
                          .reshape({1, 3, 5});

  // selected_target_probs:  [0.2462  0.1603  0.1469]
  // selected_draft_probs:   [0.2163  0.2398  0.2451]
  // acceptance_probs:       [1.1382  0.6685  0.5993]
  // uniform_rand:           [0.4785  0.6589  0.9399]
  // accepted:               [  1        1       0  ]
  auto uniform_rand = torch::tensor({{0.4785, 0.6589, 0.9399}}, options);
  auto bonus_token_ids = torch::tensor({{5}}, options.dtype(torch::kInt64));

  auto [output, masked_output] =
      RejectionSampler::random_sample(draft_token_ids,
                                      draft_probs,
                                      target_probs,
                                      uniform_rand,
                                      bonus_token_ids,
                                      true);
  auto desired_output =
      torch::tensor({{1, 2, 2, 5}}, options.dtype(torch::kInt64));
  EXPECT_TRUE(torch::allclose(output, desired_output));

  auto desired_masked_output =
      torch::tensor({{1, 2, 2, -1}}, options.dtype(torch::kInt64));
  EXPECT_TRUE(torch::allclose(masked_output, desired_masked_output));
}

TEST(RejectionSamplerTest, Mask) {
  // test accepted mask
  const auto options = GetTestOptions(torch::kBool);

  // clang-format off
  auto accepted = torch::tensor({
        {0, 1, 0, 1},
        {1, 0, 1, 1},
        {1, 1, 0, 1},
        {1, 1, 1, 1}},
        options);
  auto desired_mask = torch::tensor({
        {1, 0, 0, 0, 0},
        {1, 1, 0, 0, 0},
        {1, 1, 1, 0, 0},
        {1, 1, 1, 1, 1}},
        options);
  // clang-format on
  auto mask = RejectionSampler::build_accepted_mask(accepted);
  EXPECT_TRUE(torch::allclose(mask, desired_mask));
}

TEST(RejectionSamplerTest, Greedy) {
  const auto options = GetTestOptions(torch::kFloat32);
  const auto device = GetTestDevice();

  int64_t batch_size = 2;
  int64_t n_speculative_tokens = 3;
  int64_t vocab_size = 4;
  int64_t n_bonus_tokens = 1;

  const auto draft_token_ids =
      torch::randint(0,
                     vocab_size,
                     {batch_size, n_speculative_tokens},
                     torch::dtype(torch::kInt64).device(device));
  auto target_probs =
      torch::randn({batch_size, n_speculative_tokens, vocab_size}, options)
          .softmax(/*dim=*/-1, /*dtype=*/torch::kFloat32);
  const auto bonus_token_ids =
      torch::randint(0,
                     vocab_size,
                     {batch_size, n_bonus_tokens},
                     torch::dtype(torch::kInt64).device(device));

  auto [output, masked_output] =
      RejectionSampler::greedy_sample(draft_token_ids,
                                      target_probs,
                                      bonus_token_ids,
                                      /*mask_out_rejected_tokens=*/false);
  EXPECT_FALSE(masked_output.defined());

  const auto desired_output = target_probs.argmax(/*dim=*/-1);

  // check target tokens
  EXPECT_TRUE(torch::allclose(
      output.slice(/*dim=*/-1, /*start=*/0, /*end=*/n_speculative_tokens),
      desired_output));
  // check bonus tokens
  EXPECT_TRUE(torch::allclose(output.slice(/*dim=*/-1,
                                           /*start=*/n_speculative_tokens),
                              bonus_token_ids));
}

TEST(RejectionSamplerTest, LogProbs) {
  const auto options = GetTestOptions(torch::kFloat32);
  const auto device = GetTestDevice();
  const auto do_sample = torch::tensor({false, true, false, true}, device);
  const int64_t max_top_logprobs = 2;
  RejectionSampler sampler(do_sample,
                           do_sample.all().item<bool>(),
                           !do_sample.any().item<bool>(),
                           /*logprobs=*/true,
                           max_top_logprobs);

  int64_t batch_size = 4;
  int64_t n_speculative_tokens = 4;
  int64_t vocab_size = 8;

  const auto draft_token_ids =
      torch::randint(0,
                     vocab_size,
                     {batch_size, n_speculative_tokens},
                     torch::dtype(torch::kInt64).device(device));
  auto draft_probs =
      torch::randn({batch_size, n_speculative_tokens, vocab_size}, options)
          .softmax(/*dim=*/-1);

  auto target_logits =
      torch::randn({batch_size, n_speculative_tokens + 1, vocab_size}, options);
  const auto bonus_token_ids =
      torch::randint(0,
                     vocab_size,
                     {batch_size, 1},
                     torch::dtype(torch::kInt64).device(device));

  auto output = sampler.forward(draft_token_ids,
                                draft_probs,
                                target_logits,
                                bonus_token_ids,
                                /*mask_out_rejected_tokens=*/false);

  const auto logprobs =
      torch::log_softmax(target_logits, /*dim=*/-1, /*dtype=*/torch::kFloat32);
  const auto selected_tokens = output.next_tokens;

  const auto selected_logprobs =
      logprobs.gather(/*dim=*/-1, selected_tokens.unsqueeze(/*dim=*/-1))
          .squeeze(/*dim=*/-1);
  EXPECT_TRUE(torch::equal(output.logprobs, selected_logprobs));

  auto [top_k_values, top_k_indices] = logprobs.topk(
      max_top_logprobs, /*dim=*/-1, /*largest=*/true, /*sorted=*/true);
  EXPECT_TRUE(torch::equal(output.top_logprobs, top_k_values));
  EXPECT_TRUE(torch::equal(output.top_tokens, top_k_indices));
}

TEST(RejectionSamplerTest, Random) {
  const auto options = GetTestOptions(torch::kFloat32);

  // set random seed
  torch::manual_seed(100);

  int64_t vocab_size = 50;
  int64_t num_samples = 500000;

  auto target_prob = torch::randn({vocab_size}, options).softmax(/*dim=*/-1);
  auto target_probs =
      target_prob.reshape({1, 1, -1}).repeat({num_samples, 1, 1});

  auto draft_probs =
      torch::randn({num_samples, 1, vocab_size}, options).softmax(/*dim=*/-1);
  auto draft_token_ids = Sampler::random_sample(draft_probs);

  // not used
  auto bonus_token_ids =
      torch::ones({num_samples, 1}, options.dtype(torch::kInt64));

  auto uniform_rand = torch::rand(draft_token_ids.sizes(), options);
  auto [output, masked_output] =
      RejectionSampler::random_sample(draft_token_ids,
                                      draft_probs,
                                      target_probs,
                                      uniform_rand,
                                      bonus_token_ids,
                                      false);
  EXPECT_FALSE(masked_output.defined());

  // remove bonus token
  auto token_ids = output
                       .slice(/*dim=*/-1,
                              /*start=*/0,
                              /*end=*/-1)
                       .flatten();

  // calculate the probability of each sampled token
  auto bincount = token_ids.bincount(/*weights=*/torch::nullopt,
                                     /*minlength=*/vocab_size);
  auto sample_prob = bincount.to(torch::kFloat) / num_samples;

  EXPECT_TRUE(torch::allclose(target_prob,
                              sample_prob,
                              /*rtol=*/1e-2,
                              /*atol=*/1e-3));
}

TEST(RejectionSamplerTest, RandomFused) {
  // Skip test if not running on MLU backend
  std::string backend = Device::type_str();
  if (backend != "mlu") {
    GTEST_SKIP() << "Skipping RandomFused test: fused kernel only available on "
                    "MLU backend.";
  }
  if (Device::device_count() == 0) {
    GTEST_SKIP() << "Skipping RandomFused test: no MLU devices available";
  }

  // Prepare random test data
  torch::ScalarType dtype(torch::kFloat32);
  torch::Device device(Device::type_torch(), 0);
  const auto options = torch::dtype(dtype).device(device);
  torch::manual_seed(100);

  int64_t n_spec = 3;
  int64_t vocab_size = 50;
  int64_t num_samples = 1000;

  auto target_prob_base = torch::randn({vocab_size}, options).softmax(-1);
  auto target_probs =
      target_prob_base.reshape({1, 1, -1}).repeat({num_samples, n_spec, 1});
  auto draft_probs =
      torch::randn({num_samples, n_spec, vocab_size}, options).softmax(-1);

  // Sample draft tokens and bonus tokens
  auto draft_token_ids = Sampler::random_sample(draft_probs);
  auto bonus_token_ids = torch::randint(
      0, vocab_size, {num_samples, 1}, options.dtype(torch::kInt64));

  // Shared random tensor, used for acceptance check
  auto uniform_rand = torch::rand(draft_token_ids.sizes(), options);

  // Fused kernel output
  auto [fused_output_unmasked, fused_output_masked] =
      RejectionSampler::random_sample_fused(draft_token_ids,
                                            draft_probs,
                                            target_probs,
                                            uniform_rand,
                                            bonus_token_ids,
                                            /*mask_out_rejected_tokens=*/true);

  // Reference random_sample output
  auto [ref_output_unmasked, ref_output_masked] =
      RejectionSampler::random_sample(draft_token_ids,
                                      draft_probs,
                                      target_probs,
                                      uniform_rand,
                                      bonus_token_ids,
                                      /*mask_out_rejected_tokens=*/true);

  // Check output shape
  EXPECT_EQ(fused_output_masked.size(0), num_samples);
  EXPECT_EQ(fused_output_masked.size(1), n_spec + 1);

  // Mask output should match exactly (same tokens should be accepted/rejected)
  auto fused_mask = (fused_output_masked != -1);
  auto ref_mask = (ref_output_masked != -1);
  EXPECT_TRUE(torch::equal(fused_mask, ref_mask))
      << "Mismatch in acceptance decision between Fused and Ref "
         "implementation!";

  // If a draft token is accepted, it must exactly match the input and the
  // reference output
  for (int j = 0; j < n_spec; ++j) {
    auto current_col = fused_output_masked.slice(1, j, j + 1);
    auto next_col = fused_output_masked.slice(1, j + 1, j + 2);
    auto is_accepted_draft = (current_col != -1) & (next_col != -1);

    if (is_accepted_draft.any().item<bool>()) {
      auto fused_drafts = current_col.masked_select(is_accepted_draft);
      auto input_drafts =
          draft_token_ids.slice(1, j, j + 1).masked_select(is_accepted_draft);

      EXPECT_TRUE(torch::equal(fused_drafts, input_drafts))
          << "Fused kernel altered an accepted draft token at index " << j;

      auto ref_drafts =
          ref_output_masked.slice(1, j, j + 1).masked_select(is_accepted_draft);
      EXPECT_TRUE(torch::equal(fused_drafts, ref_drafts))
          << "Mismatch with Reference on accepted draft token at index " << j;
    }
  }

  // Bonus token column should match input when all previous tokens accepted
  auto last_col = fused_output_masked.slice(1, n_spec, n_spec + 1);
  auto fully_accepted_mask = (last_col != -1);

  if (fully_accepted_mask.any().item<bool>()) {
    auto valid_bonus_out = last_col.masked_select(fully_accepted_mask);
    auto valid_bonus_in = bonus_token_ids.masked_select(fully_accepted_mask);
    EXPECT_TRUE(torch::equal(valid_bonus_out, valid_bonus_in))
        << "Bonus token mismatch for fully accepted sequences";
  } else {
    LOG(INFO)
        << "No fully accepted sequences in this batch, skipping bonus check.";
  }

  // After the first -1 in each row, all remaining values must be -1
  auto cpu_output = fused_output_masked.to(torch::kCPU);
  auto output_a = cpu_output.accessor<int64_t, 2>();

  for (int i = 0; i < num_samples; ++i) {
    bool rejected = false;
    for (int j = 0; j < n_spec + 1; ++j) {
      if (output_a[i][j] == -1) {
        rejected = true;
      } else {
        if (rejected) {
          ADD_FAILURE() << "Found valid token after -1 at row " << i << " col "
                        << j;
        }
      }
    }
  }
}

}  // namespace xllm

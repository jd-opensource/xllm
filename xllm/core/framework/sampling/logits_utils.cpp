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

#include "logits_utils.h"

#include <torch/torch.h>

#include <memory>

namespace xllm {

void apply_frequency_presence_penalties(
    torch::Tensor& logits,
    const torch::Tensor& unique_token_ids,
    const torch::Tensor& unique_token_counts,
    const torch::Tensor& frequency_penalties,
    const torch::Tensor& presence_penalties) {
  auto score = logits.gather(/*dim=*/1, /*index=*/unique_token_ids);
  score.sub_(unique_token_counts * frequency_penalties.unsqueeze(1));
  score.sub_((unique_token_counts > 0) * presence_penalties.unsqueeze(1));

  logits.scatter_(/*dim=*/1, /*index=*/unique_token_ids, /*core=*/score);
}

void apply_repetition_penalties(torch::Tensor& logits,
                                const torch::Tensor& unique_token_ids,
                                const torch::Tensor& penalties) {
  auto unsqueezed_penalties = penalties.unsqueeze(1);
  auto score = logits.gather(/*dim=*/1, /*index=*/unique_token_ids);
  logits.scatter_(/*dim=*/1,
                  /*index=*/unique_token_ids,
                  /*core=*/
                  torch::where(score < 0,
                               score * unsqueezed_penalties,
                               score / unsqueezed_penalties));
}

void apply_temperatures(torch::Tensor& logits,
                        const torch::Tensor& temperatures) {
  auto unsqueezed_temperatures = temperatures.unsqueeze(1);
  unsqueezed_temperatures =
      torch::where(unsqueezed_temperatures == 0,
                   torch::tensor(1.0).to(unsqueezed_temperatures.device()),
                   unsqueezed_temperatures);

  logits.div_(unsqueezed_temperatures);
}

void apply_top_k_top_p(torch::Tensor& logits,
                       const torch::Tensor& top_k,
                       const torch::Tensor& top_p) {
  if (top_k.defined() && top_p.defined()) {
#if defined(USE_NPU)
    auto max_value = std::numeric_limits<int64_t>::max();

    auto processed_top_k =
        torch::where(
            top_k <= 0, torch::tensor(max_value).to(top_k.device()), top_k)
            .to(torch::kInt32);
    xllm_ops::top_k_top_p(logits, processed_top_k, top_p);
#endif
  } else {
    auto [sorted_logits, logits_idx] =
        logits.sort(/*dim=*/-1, /*descending=*/true);

    float filter_value = -std::numeric_limits<float>::infinity();

    if (top_k.defined()) {
      auto processed_top_k = top_k.unsqueeze(1);
      auto max_value = std::numeric_limits<int64_t>::max();

      processed_top_k =
          torch::where(processed_top_k <= 0,
                       torch::tensor(max_value).to(processed_top_k.device()),
                       processed_top_k);

      auto vocab_size = logits.size(-1);
      auto top_k_mask = torch::arange(vocab_size, sorted_logits.device())
                            .expand_as(sorted_logits);
      top_k_mask = top_k_mask >= processed_top_k;
      sorted_logits.masked_fill_(top_k_mask, filter_value);
    }

    if (top_p.defined()) {
      auto processed_top_p = top_p.unsqueeze(1);

      auto probs = sorted_logits.softmax(/*dim=*/-1).to(torch::kFloat32);
      auto probs_sum = probs.cumsum(/*dim=*/-1);
      auto mask = (probs_sum - probs) > processed_top_p;

      sorted_logits.masked_fill_(mask, filter_value);
    }
    logits =
        torch::empty_like(sorted_logits)
            .scatter_(/*dim=*/-1, /*index=*/logits_idx, /*core=*/sorted_logits);
  }
}

}  // namespace xllm

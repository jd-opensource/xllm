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

#include "sequences_group.h"

namespace xllm {

SequencesGroup::SequencesGroup(const std::string& prompt,
                               const std::vector<int32_t>& prompt_tokens,
                               const torch::Tensor& input_embedding,
                               const MMData& mm_data,
                               const SequenceParams& sequence_params)
    : prompt_(prompt),
      prompt_tokens_(prompt_tokens),
      input_embedding_(input_embedding),
      mm_data_(mm_data),
      sequence_params_(std::move(sequence_params)) {
  add();
}

void SequencesGroup::add() {
  const size_t index = sequences_.size();
  IncrementalDecoder decoder(prompt_,
                             prompt_tokens_.size(),
                             sequence_params_.echo,
                             sequence_params_.skip_special_tokens);
  sequences_.emplace_back(std::make_unique<Sequence>(index,
                                                     prompt_tokens_,
                                                     input_embedding_,
                                                     mm_data_,
                                                     std::move(decoder),
                                                     sequence_params_));
}

bool SequencesGroup::finished() const {
  if (sequences_.size() < sequence_params_.best_of) {
    return false;
  }

  return std::all_of(sequences_.begin(),
                     sequences_.end(),
                     [](const auto& sequence) { return sequence->finished(); });
}

bool SequencesGroup::expand_sequences(bool share_prefix) {
  size_t current_seq_count = sequences_.size();
  size_t best_of = sequence_params_.best_of;
  if (current_seq_count == best_of) {
    return false;
  } else if (current_seq_count > best_of) {
    LOG(FATAL) << "Request has more than " << best_of << " sequences.";
    return false;
  }

  CHECK(!sequences_.empty()) << "Request has no sequence.";
  const auto& seq = sequences_[0];
  // prefill is not finished, can not expand
  // FIXME later share_prefix
  if (!share_prefix ||
      seq->kv_state().kv_cache_tokens_num() < seq->num_prompt_tokens()) {
    while (sequences_.size() < best_of) {
      add();
    }
    return true;
  }

  return false;
}

void SequencesGroup::generate_outputs(std::vector<SequenceOutput>& outputs,
                                      const Tokenizer& tokenizer) {
  if (sequence_params_.streaming) {
    for (auto& seq : sequences_) {
      outputs.push_back(std::move(seq->generate_output()));
    }
    return;
  }

  auto n = sequence_params_.n;
  auto num = std::min(sequences_.size(), n);
  outputs.reserve(num);
  if (sequences_.size() > n) {
    std::vector<std::pair<float, size_t> > logprobs_vec;
    logprobs_vec.reserve(sequences_.size());
    for (size_t i = 0; i < sequences_.size(); ++i) {
      logprobs_vec.emplace_back(sequences_[i]->get_average_logprob(), i);
    }
    std::sort(logprobs_vec.begin(),
              logprobs_vec.end(),
              [](const auto& l, const auto& r) { return l.first > r.first; });
    for (size_t i = 0; i < n; ++i) {
      const auto [logprob, index] = logprobs_vec[i];
      auto seq_output = sequences_[index]->generate_output(tokenizer);
      seq_output.index = i;
      outputs.push_back(std::move(seq_output));
    }
  } else {
    for (auto& seq : sequences_) {
      outputs.push_back(seq->generate_output(tokenizer));
    }
  }
}

}  // namespace xllm

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

#include "mm_data_visitor.h"

#include <absl/strings/match.h>

#include <numeric>

namespace xllm {

namespace {
std::pair<int32_t, int32_t> compute_emb_range(int32_t start_pos,
                                              int32_t end_pos,
                                              const torch::Tensor& mask) {
  if (!mask.defined() || mask.numel() == 0) {
    return {start_pos, end_pos};
  }
  auto mask_cpu = mask.to(torch::kCPU);
  auto upto_start_pos =
      mask_cpu.slice(/*dim*/ 0, /*start*/ 0, /*end*/ start_pos)
          .sum()
          .item<int32_t>();
  auto upto_end_pos = mask_cpu.slice(/*dim*/ 0, /*start*/ 0, /*end*/ end_pos)
                          .sum()
                          .item<int32_t>();
  return {upto_start_pos, upto_end_pos};
}
}  // namespace

bool CollectItemTensorVisitor::visit(MMDataItem& item) {
  for (const auto& pair : item.data()) {
    const auto& key = pair.first;

    if (!black_keys_.empty() && black_keys_.count(key)) {
      continue;
    }

    if (!white_keys_.empty() && !white_keys_.count(key)) {
      continue;
    }

    auto& tar = datas_[pair.first];
    if (std::holds_alternative<torch::Tensor>(pair.second)) {
      tar.emplace_back(std::get<torch::Tensor>(pair.second));
    } else if (std::holds_alternative<std::vector<torch::Tensor>>(
                   pair.second)) {
      const auto& lst = std::get<std::vector<torch::Tensor>>(pair.second);
      tar.insert(tar.end(), lst.begin(), lst.end());
    }
  }
  return true;
}

bool CollectItemTensorVisitor::visit(const MMKey& key, MMValue& value) {
  if (!black_keys_.empty() && black_keys_.count(key)) {
    return true;
  }

  if (!white_keys_.empty() && !white_keys_.count(key)) {
    return true;
  }

  auto& tar = datas_[key];
  if (std::holds_alternative<torch::Tensor>(value)) {
    tar.push_back(std::get<torch::Tensor>(value));
  } else if (std::holds_alternative<std::vector<torch::Tensor>>(value)) {
    const auto& lst = std::get<std::vector<torch::Tensor>>(value);
    tar.insert(tar.end(), lst.begin(), lst.end());
  }

  return true;
}

bool CollectMMDataTensorVisitor::visit(MMData& data) {
  type_ |= data.type();
  data.foreach (item_visitor_);
  return true;
}

bool EncoderInputGatherVisitor::visit(MMDataItem& item) {
  if (!item.state().is_scheduling()) return true;
  if (item.is_embedded()) return true;

  for (const auto& [key, value] : item.data()) {
    if (absl::StartsWith(key, filter_prefix_)) continue;
    auto& tar = datas_[key];
    if (std::holds_alternative<torch::Tensor>(value)) {
      tar.push_back(std::get<torch::Tensor>(value));
    } else if (std::holds_alternative<std::vector<torch::Tensor>>(value)) {
      const auto& vec = std::get<std::vector<torch::Tensor>>(value);
      tar.insert(tar.end(), vec.begin(), vec.end());
    }
  }
  return true;
}

bool EncoderInputGatherVisitor::finish(MMBatchData& mm_data) {
  MMDict dict;
  for (const auto& pair : datas_) {
    torch::Tensor tar;
    if (safe_concat(pair.second, tar)) {
      dict[pair.first] = tar;
    } else {
      dict[pair.first] = std::move(pair.second);
    }
  }
  mm_data.replace(dict);
  return true;
}

bool EncoderOutputScatterVisitor::visit(MMDataItem& item) {
  if (!item.state().is_scheduling()) return true;
  if (item.is_embedded()) return true;

  std::string prefix;
  int32_t* idx = nullptr;

  if (item.type() == MMType::IMAGE) {
    prefix = "image|";
    idx = &image_idx;
  } else if (item.type() == MMType::VIDEO) {
    prefix = "video|";
    idx = &video_idx;
  } else if (item.type() == MMType::AUDIO) {
    prefix = "audio|";
    idx = &audio_idx;
  } else {
    LOG(FATAL) << " mm data item type invalid, type is " << item.type();
    return true;
  }

  for (const auto& [key, value] : data_) {
    const auto& vec = std::get<std::vector<torch::Tensor>>(value);
    if (absl::StartsWith(key, prefix)) {
      std::string name = key.substr(prefix.length());
      item.add(name, vec[*idx]);
    }
  }
  ++(*idx);
  return true;
}

bool EncoderOutputScatterVisitor::finish() const {
  for (const auto& [key, value] : data_) {
    std::string name = key.substr(0, key.find("|"));
    uint32_t idx = 0;
    if (name == "image") {
      idx = image_idx;
    } else if (name == "video") {
      idx = video_idx;
    } else if (name == "audio") {
      idx = audio_idx;
    } else {
      LOG(FATAL) << "invalid modality key: " << key;
    }
    if (idx != std::get<std::vector<torch::Tensor>>(value).size()) {
      return false;
    }
  }
  return true;
}

EncoderEmbeddingGatherVisitor::EncoderEmbeddingGatherVisitor(
    const torch::Device& device,
    const std::vector<int32_t>& seq_lens,
    const std::vector<int32_t>& scheduled_seq_lens)
    : device_(device),
      per_seq_context_lens_(seq_lens),
      per_seq_scheduled_lens_(scheduled_seq_lens) {
#if defined(USE_NPU) || defined(USE_MUSA)
  std::vector<int32_t> scheduled_prefix_sums(per_seq_scheduled_lens_.size(), 0);
  std::partial_sum(per_seq_scheduled_lens_.begin(),
                   per_seq_scheduled_lens_.end(),
                   scheduled_prefix_sums.begin());
  per_seq_scheduled_offsets_.resize(per_seq_scheduled_lens_.size(), 0);
  if (!scheduled_prefix_sums.empty()) {
    std::copy(scheduled_prefix_sums.begin(),
              scheduled_prefix_sums.end() - 1,
              per_seq_scheduled_offsets_.begin() + 1);
    total_scheduled_tokens_ = scheduled_prefix_sums.back();
  }
#else
  if (!per_seq_scheduled_lens_.empty()) {
    per_seq_scheduled_offsets_.assign(per_seq_scheduled_lens_.begin(),
                                      per_seq_scheduled_lens_.end() - 1);
    total_scheduled_tokens_ = per_seq_scheduled_lens_.back();
  }
#endif
  image_mask_ = torch::zeros({total_scheduled_tokens_},
                             torch::dtype(torch::kBool).device(device_));
  video_mask_ = torch::zeros({total_scheduled_tokens_},
                             torch::dtype(torch::kBool).device(device_));
  audio_mask_ = torch::zeros({total_scheduled_tokens_},
                             torch::dtype(torch::kBool).device(device_));
}

bool EncoderEmbeddingGatherVisitor::visit(MMDataItem& item, int32_t seq_index) {
  const auto& state = item.state();
  if (!state.is_scheduling()) return true;

  auto token_pos = item.state().token_pos();
  int32_t start_pos = state.schedule_data().start_pos;
  int32_t end_pos = state.schedule_data().end_pos;
  // start_pos / end_pos select the scheduled subrange inside this multimodal
  // item's token span in the prompt. Not every token in that subrange is a real
  // multimodal token, so mm_token_mask is used to filter out non-multimodal
  // positions when slicing multimodal embeddings.
  auto [emb_start, emb_end] =
      compute_emb_range(start_pos, end_pos, state.mm_token_mask());
  int32_t schedule_tokens_num = [&]() {
#if defined(USE_NPU) || defined(USE_MUSA)
    return per_seq_scheduled_lens_[seq_index];
#else
    return per_seq_scheduled_lens_[seq_index + 1] -
           per_seq_scheduled_lens_[seq_index];
#endif
  }();

  int32_t context_tokens_num = [&]() {
#if defined(USE_NPU) || defined(USE_MUSA)
    return per_seq_context_lens_[seq_index];
#else
    return per_seq_context_lens_[seq_index + 1] -
           per_seq_context_lens_[seq_index];
#endif
  }();

  int32_t computed_tokens_num = context_tokens_num - schedule_tokens_num;
  int32_t req_start_idx_ = per_seq_scheduled_offsets_[seq_index];
  int32_t req_start_pos =
      req_start_idx_ + token_pos.offset - computed_tokens_num + start_pos;
  int32_t req_end_pos =
      req_start_idx_ + token_pos.offset - computed_tokens_num + end_pos;

  if (item.type() == MMType::IMAGE) {
    image_mask_.slice(/*dim*/ 0,
                      /*start*/ req_start_pos,
                      /*end*/ req_end_pos) =
        state.mm_token_mask().slice(
            /*dim*/ 0, /*start*/ start_pos, /*end*/ end_pos);
  } else if (item.type() == MMType::VIDEO) {
    video_mask_.slice(/*dim*/ 0,
                      /*start*/ req_start_pos,
                      /*end*/ req_end_pos) =
        state.mm_token_mask().slice(
            /*dim*/ 0, /*start*/ start_pos, /*end*/ end_pos);
  } else if (item.type() == MMType::AUDIO) {
    audio_mask_.slice(/*dim*/ 0,
                      /*start*/ req_start_pos,
                      /*end*/ req_end_pos) =
        state.mm_token_mask().slice(
            /*dim*/ 0, /*start*/ start_pos, /*end*/ end_pos);
  }
  for (auto& [key, value] : item.mutable_data()) {
    auto& emb = std::get<torch::Tensor>(value);
    if (absl::StartsWith(key, gather_prefix_)) {
      emb = safe_to(emb, device_, true);
      datas_[key].push_back(
          emb.slice(/*dim*/ 0, /*start*/ emb_start, /*end*/ emb_end));
    }
  }
  return true;
}

bool EncoderEmbeddingGatherVisitor::finish(MMBatchData& mm_data) {
  MMDict data;
  torch::Tensor tar;
  for (auto& [key, value] : datas_) {
    if (safe_concat(value, tar)) {
      data[key] = tar;
    } else {
      LOG(ERROR) << "safe concat failed.";
      return false;
    }
  }
  // mask for merge multimodal embeddings into text.
  data["image|mask"] = image_mask_;
  data["video|mask"] = video_mask_;
  data["audio|mask"] = audio_mask_;
  mm_data.replace(data);
  return true;
}

bool UpdateMMItemScheduleStateVisitor::visit(MMDataItem& item) {
  auto& schedule_data = item.mutable_state().mutable_schedule_data();
  auto& token_pos = item.state().token_pos();
  int32_t mm_end_idx = token_pos.offset + token_pos.length - 1;

  int32_t schedule_token_num = q_seq_len_;
  if (mm_end_idx < computed_token_num_) {
    return true;
  }
  if (token_pos.offset >= computed_token_num_ + schedule_token_num) {
    return true;
  }
  schedule_data.start_pos = std::max(computed_token_num_ - token_pos.offset, 0);
  schedule_data.end_pos =
      std::min(computed_token_num_ - token_pos.offset + schedule_token_num,
               token_pos.length);
  mm_data_items_.push_back(item);
  return true;
}

}  // namespace xllm

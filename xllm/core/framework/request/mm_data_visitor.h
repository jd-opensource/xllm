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

#pragma once

#include <torch/torch.h>

#include <unordered_set>

#include "mm_batch_data.h"
#include "mm_data.h"

namespace xllm {

class CollectItemTensorVisitor : public MMData::IItemVisitor {
 public:
  CollectItemTensorVisitor(
      std::unordered_map<MMKey, std::vector<torch::Tensor>>& datas,
      const std::unordered_set<MMKey>& black_keys = {},
      const std::unordered_set<MMKey>& white_keys = {})
      : datas_(datas), black_keys_(black_keys), white_keys_(white_keys) {};

  CollectItemTensorVisitor(const std::unordered_set<MMKey>& black_keys = {},
                           const std::unordered_set<MMKey>& white_keys = {})
      : datas_(stub_), black_keys_(black_keys), white_keys_(white_keys) {};

  bool visit(MMDataItem& item) override;
  bool visit(const MMKey& key, MMValue& value) override;

 public:
  std::unordered_map<MMKey, std::vector<torch::Tensor>> stub_;
  std::unordered_map<MMKey, std::vector<torch::Tensor>>& datas_;

  std::unordered_set<MMKey> black_keys_;
  std::unordered_set<MMKey> white_keys_;
};

class CollectMMDataTensorVisitor : public MMData::IVisitor {
 public:
  CollectMMDataTensorVisitor(const std::unordered_set<MMKey>& black_keys = {},
                             const std::unordered_set<MMKey>& white_keys = {})
      : item_visitor_(datas_, black_keys, white_keys) {};

  bool visit(MMData& data) override;

 public:
  uint32_t type_ = MMType::NONE;
  std::unordered_map<MMKey, std::vector<torch::Tensor>> datas_;

  CollectItemTensorVisitor item_visitor_;
};

class EncoderInputGatherVisitor : public MMDataItem::IVisitor {
 public:
  EncoderInputGatherVisitor() = default;

  bool visit(MMDataItem& item) override;
  bool finish(MMBatchData& mm_data);

 public:
  std::unordered_map<MMKey, std::vector<torch::Tensor>> datas_;
  std::string filter_prefix_ = "embedding";
};

class EncoderOutputScatterVisitor : public MMDataItem::IVisitor {
 public:
  EncoderOutputScatterVisitor(const MMDict& data) : data_(data) {}

  bool visit(MMDataItem& data) override;
  bool finish() const;

 public:
  const MMDict& data_;
  int32_t image_idx = 0;
  int32_t video_idx = 0;
  int32_t audio_idx = 0;
};

class EncoderEmbeddingGatherVisitor : public MMDataItem::IVisitor {
 public:
  EncoderEmbeddingGatherVisitor(const torch::Device& device,
                                const std::vector<int32_t>& seq_lens,
                                const std::vector<int32_t>& q_seq_lens)
      : device_(device), seq_lens_(seq_lens), q_seq_lens_(q_seq_lens) {
    req_start_idx_vec_.reserve(seq_lens.size());
    // cusum(q_seq_lens)
    int32_t cumsum = 0;
    for (const auto& q_seq_len : q_seq_lens) {
      req_start_idx_vec_.push_back(cumsum);
      cumsum += q_seq_len;
    }
    // mask
    image_mask_ =
        torch::zeros({cumsum}, torch::dtype(torch::kBool).device(device_));
    video_mask_ =
        torch::zeros({cumsum}, torch::dtype(torch::kBool).device(device_));
    audio_mask_ =
        torch::zeros({cumsum}, torch::dtype(torch::kBool).device(device_));
  }
  bool visit(MMDataItem& data, int32_t seq_index) override;
  bool finish(MMBatchData& mm_data);

 public:
  torch::Device device_;
  std::string gather_prefix_ = "embedding";
  const std::vector<int32_t>& seq_lens_;
  const std::vector<int32_t>& q_seq_lens_;
  std::vector<int32_t> req_start_idx_vec_;
  torch::Tensor image_mask_;
  torch::Tensor video_mask_;
  torch::Tensor audio_mask_;
  std::unordered_map<MMKey, std::vector<torch::Tensor>> datas_;
};

class UpdateMMItemScheduleStateVisitor : public MMDataItem::IVisitor {
 public:
  UpdateMMItemScheduleStateVisitor(int32_t computed_token_num = 0,
                                   int32_t q_seq_len = 0)
      : computed_token_num_(computed_token_num), q_seq_len_(q_seq_len) {}

  bool visit(MMDataItem& item) override;

 public:
  std::vector<MMDataItem> mm_data_items_;
  int32_t computed_token_num_ = 0;
  int32_t q_seq_len_ = 0;
};

}  // namespace xllm

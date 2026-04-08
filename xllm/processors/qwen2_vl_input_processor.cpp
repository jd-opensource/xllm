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

#include "processors/qwen2_vl_input_processor.h"

#include <torch/torch.h>

#include <algorithm>
#include <cassert>

namespace xllm {

Qwen2_5_VLInputProcessor::Qwen2_5_VLInputProcessor(const ModelArgs& args) {
  merge_size_ = args.mm_image_merge_size();
  vision_start_token_id_ = args.vision_start_token_id();
  vision_end_token_id_ = args.vision_end_token_id();
  image_token_id_ = args.image_token_id();
  video_token_id_ = args.video_token_id();
}

void Qwen2_5_VLInputProcessor::process(std::string& prompt,
                                       const MMData& mm_data) {
  torch::Tensor image_grid_thw;
  if (auto res = mm_data.get<torch::Tensor>("image_grid_thw"))
    image_grid_thw = res.value();

  torch::Tensor video_grid_thw;
  if (auto res = mm_data.get<torch::Tensor>("video_grid_thw"))
    video_grid_thw = res.value();

  if (!image_grid_thw.defined() && !video_grid_thw.defined()) return;

  auto merge_length = merge_size_ * merge_size_;
  int32_t total_image_token = 0;
  if (image_grid_thw.defined()) {
    auto count = image_grid_thw.sizes()[0];
    for (int32_t idx = 0; idx < count; ++idx)
      total_image_token +=
          image_grid_thw[idx].prod().item<int32_t>() / merge_length;
  }

  int32_t total_video_token = 0;
  if (video_grid_thw.defined()) {
    auto count = video_grid_thw.sizes()[0];
    for (int32_t idx = 0; idx < count; ++idx)
      total_video_token +=
          video_grid_thw[idx].prod().item<int32_t>() / merge_length;
  }

  size_t total_token_len = total_image_token * image_token_.size() +
                           total_video_token * video_token_.size();
  std::string data;
  data.reserve(prompt.size() + total_token_len);

  int32_t image_index = 0;
  int32_t video_index = 0;

  const torch::Tensor* grid_thw = nullptr;
  const std::string* token = nullptr;
  int32_t* index = nullptr;

  size_t begin = 0;
  auto pair = find_vision_token(prompt, begin);

  while (pair.second != std::string::npos) {
    data.append(prompt, begin, pair.second - begin);

    if (pair.first == TokenType::IMAGE) {
      grid_thw = &image_grid_thw;
      token = &image_token_;
      index = &image_index;
    } else if (pair.first == TokenType::VIDEO) {
      grid_thw = &video_grid_thw;
      token = &video_token_;
      index = &video_index;
    } else {
      assert(false);
    }

    auto token_num =
        (*grid_thw)[(*index)].prod().item<int32_t>() / merge_length;
    while (token_num--) data.append(*token);

    ++(*index);
    begin = pair.second + token->size();
    pair = find_vision_token(prompt, begin);
  }

  if (begin < prompt.size()) data.append(prompt, begin, std::string::npos);

  prompt = std::move(data);
}

void Qwen2_5_VLInputProcessor::find_mm_spans(const std::vector<int>& prompt,
                                             MMData& mm_data) {
  auto start = prompt.begin();
  uint32_t global_mm_index = 0;
  uint32_t offset = 0;
  uint32_t length = 0;
  auto& mm_items = mm_data.items<MMItemVec>();
  while (true) {
    auto vision_start_it =
        std::find(start, prompt.end(), vision_start_token_id_);
    auto vision_end_it = std::find(start, prompt.end(), vision_end_token_id_);
    if (vision_start_it == prompt.end()) {
      break;
    }
    offset = std::distance(prompt.begin(), vision_start_it);
    length = std::distance(vision_start_it + 1, vision_end_it);

    auto& item = mm_items[global_mm_index];
    if (*(vision_start_it + 1) == image_token_id_) {
      item.mutable_state().mutable_token_pos() = {offset + 1, length};
    } else if (*(vision_start_it + 1) == video_token_id_) {
      item.mutable_state().mutable_token_pos() = {offset + 1, length};
    }
    global_mm_index++;
    start = std::next(vision_end_it);
  }
}

std::pair<Qwen2_5_VLInputProcessor::TokenType, size_t>
Qwen2_5_VLInputProcessor::find_vision_token(const std::string& prompt,
                                            size_t begin) {
  auto img_pos = prompt.find(image_token_, begin);
  auto vid_pos = prompt.find(video_token_, begin);

  if (img_pos == std::string::npos && vid_pos == std::string::npos)
    return {TokenType::INVALID, std::string::npos};
  else if (vid_pos == std::string::npos)
    return {TokenType::IMAGE, img_pos};
  else if (img_pos == std::string::npos)
    return {TokenType::VIDEO, vid_pos};
  else
    return img_pos < vid_pos ? std::make_pair(TokenType::IMAGE, img_pos)
                             : std::make_pair(TokenType::VIDEO, vid_pos);
}

}  // namespace xllm

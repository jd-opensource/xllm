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

#include "mposition.h"

#include "framework/model/model_args.h"
#include "framework/request/sequence.h"
namespace xllm {

torch::Tensor MPositionHelper::get_positions() {
  // if (seq_.is_prefill_stage()) {
  if (seq_.kv_state().kv_cache_tokens_num() < seq_.num_prompt_tokens()) {
    auto& mm_data = seq_.get_mm_data();

    torch::Tensor image_grid_thw;
    if (auto res = mm_data.get<torch::Tensor>("image_grid_thw"))
      image_grid_thw = res.value();

    torch::Tensor video_grid_thw;
    if (auto res = mm_data.get<torch::Tensor>("video_grid_thw"))
      video_grid_thw = res.value();

    torch::Tensor second_per_grid_ts;
    if (auto res = mm_data.get<torch::Tensor>("second_per_grid_ts"))
      second_per_grid_ts = res.value();
    auto res =
        get_positions_p(image_grid_thw, video_grid_thw, second_per_grid_ts);
    seq_.set_mrope_position_delta(std::get<1>(res));

    return std::get<0>(res);
  } else {
    return get_positions_d();
  }
}

std::tuple<torch::Tensor, int> MPositionHelper::get_positions_p(
    torch::Tensor image_grid_thw,
    torch::Tensor video_grid_thw,
    torch::Tensor second_per_grid_ts) {
  auto image_token_id = args_.image_token_id();
  auto video_token_id = args_.video_token_id();
  auto vision_start_token_id = args_.vision_start_token_id();
  auto spatial_merge_size = args_.mm_spatial_merge_size();
  auto tokens_per_second = args_.mm_tokens_per_second();

  auto input_tokens = seq_.tokens();
  auto input_tokens_tensor = torch::tensor(std::vector<int>(input_tokens));
  auto vision_start_indices =
      torch::argwhere(input_tokens_tensor == vision_start_token_id).squeeze(1);

  auto vision_tokens = input_tokens_tensor.index({vision_start_indices + 1});
  int image_nums = torch::sum(vision_tokens == image_token_id).item<int>();
  int video_nums = torch::sum(vision_tokens == video_token_id).item<int>();

  std::vector<torch::Tensor> llm_pos_ids_list;
  int st = 0;
  int remain_images = image_nums, remain_videos = video_nums;
  int image_index = 0, video_index = 0;

  for (int i = 0; i < image_nums + video_nums; ++i) {
    float video_second_per_grid_t = 1.0f;
    int ed_image = input_tokens.size() + 1;
    int ed_video = input_tokens.size() + 1;

    if (remain_images > 0) {
      auto it = std::find(
          input_tokens.begin() + st, input_tokens.end(), image_token_id);
      if (it != input_tokens.end()) {
        ed_image = std::distance(input_tokens.begin(), it);
      }
    }

    if (remain_videos > 0) {
      auto it = std::find(
          input_tokens.begin() + st, input_tokens.end(), video_token_id);
      if (it != input_tokens.end()) {
        ed_video = std::distance(input_tokens.begin(), it);
      }
    }

    int t = 0, h = 0, w = 0;
    int ed = 0;
    if (ed_image < ed_video) {
      t = image_grid_thw[image_index][0].item<int>();
      h = image_grid_thw[image_index][1].item<int>();
      w = image_grid_thw[image_index][2].item<int>();
      image_index++;
      remain_images--;
      ed = ed_image;
    } else {
      t = video_grid_thw[video_index][0].item<int>();
      h = video_grid_thw[video_index][1].item<int>();
      w = video_grid_thw[video_index][2].item<int>();

      video_second_per_grid_t = second_per_grid_ts[video_index].item<float>();

      video_index++;
      remain_videos--;
      ed = ed_video;
    }

    int llm_grid_t = t;
    int llm_grid_h = h / spatial_merge_size;
    int llm_grid_w = w / spatial_merge_size;
    int text_len = ed - st;

    int st_idx = 0;
    if (!llm_pos_ids_list.empty()) {
      st_idx = llm_pos_ids_list.back().max().item<int>() + 1;
    }

    if (text_len > 0) {
      auto text_pos =
          torch::arange(text_len, torch::kInt32).view({1, -1}).expand({3, -1}) +
          st_idx;
      llm_pos_ids_list.push_back(text_pos);
    }

    auto t_index = (torch::arange(llm_grid_t, torch::kInt32)
                        .view({-1, 1})
                        .expand({-1, llm_grid_h * llm_grid_w}) *
                    video_second_per_grid_t * tokens_per_second)
                       .to(torch::kInt32)
                       .flatten();

    auto h_index = torch::arange(llm_grid_h, torch::kInt32)
                       .view({1, -1, 1})
                       .expand({llm_grid_t, -1, llm_grid_w})
                       .flatten();

    auto w_index = torch::arange(llm_grid_w, torch::kInt32)
                       .view({1, 1, -1})
                       .expand({llm_grid_t, llm_grid_h, -1})
                       .flatten();

    auto visual_pos =
        torch::stack({t_index, h_index, w_index}) + text_len + st_idx;
    llm_pos_ids_list.push_back(visual_pos);

    st = ed + llm_grid_t * llm_grid_h * llm_grid_w;
  }

  if (st < static_cast<int>(input_tokens.size())) {
    int st_idx = 0;
    if (!llm_pos_ids_list.empty()) {
      st_idx = llm_pos_ids_list.back().max().item<int>() + 1;
    }

    int text_len = input_tokens.size() - st;
    auto text_pos =
        torch::arange(text_len, torch::kInt32).view({1, -1}).expand({3, -1}) +
        st_idx;
    llm_pos_ids_list.push_back(text_pos);
  }

  auto llm_positions = torch::cat(llm_pos_ids_list, 1).reshape({3, -1});
  int mrope_position_delta =
      (llm_positions.max().item<int>() + 1 - input_tokens.size());
  return std::make_tuple(llm_positions, mrope_position_delta);
}

torch::Tensor MPositionHelper::get_positions_d() {
  auto mrope_position_delta = seq_.get_mrope_position_delta();
  auto num_tokens = seq_.num_tokens();
  return torch::arange(int(mrope_position_delta + num_tokens - 1),
                       int(mrope_position_delta + num_tokens),
                       torch::kInt32)
      .expand({3, -1});
}

}  // namespace xllm
